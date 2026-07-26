"""Tests for design doc axis 2 (answer quality) / axis 3 (end-to-end) metrics."""

from unittest.mock import AsyncMock

import pytest

from evaluation import (
    JUDGE_QUALITY_PASS_THRESHOLD,
    build_llm_judge_prompt,
    compute_answer_quality_accuracy,
    compute_end_to_end_accuracy,
    compute_latency_breakdown,
    extract_answer_letter,
    judge_response_quality,
)
from expert_backend import OllamaClient


def test_extract_answer_letter_matches_explicit_japanese_phrasing() -> None:
    """ "正解はB" style phrasing is recognized."""
    assert extract_answer_letter("この質問の正解はBです．") == "B"


def test_extract_answer_letter_matches_markdown_bold() -> None:
    """A bolded letter (**C**) is recognized when no explicit phrasing is present."""
    assert extract_answer_letter("検討の結果，**C**が適切と考えられます．") == "C"


def test_extract_answer_letter_matches_parenthesized_letter() -> None:
    """A parenthesized letter (A) is recognized."""
    assert extract_answer_letter("選択肢のうち(A)が最も妥当です．") == "A"


def test_extract_answer_letter_matches_leading_bare_letter() -> None:
    """A bare leading letter (as the least explicit fallback) is recognized."""
    assert extract_answer_letter("D. これが正しい選択肢です．") == "D"


def test_extract_answer_letter_returns_none_when_no_letter_present() -> None:
    """A response with no identifiable letter returns None, not a guess."""
    assert extract_answer_letter("この質問については専門医への相談をお勧めします．") is None


def test_compute_answer_quality_accuracy_only_grades_jmmlu_rows() -> None:
    """Hand-authored (no jmmlu_answer) rows are excluded from the denominator."""
    dataset = [
        {"id": "medical-001", "jmmlu_answer": "B"},
        {"id": "compound-001"},  # no jmmlu_answer: hand-authored, ungradable this way
    ]
    results = [
        {"id": "medical-001", "answer_text": "正解はBです．"},
        {"id": "compound-001", "answer_text": "anything"},
    ]
    assert compute_answer_quality_accuracy(results, dataset) == 1.0


def test_compute_answer_quality_accuracy_counts_mismatches_as_incorrect() -> None:
    """An extracted letter that doesn't match jmmlu_answer counts against accuracy."""
    dataset = [
        {"id": "medical-001", "jmmlu_answer": "B"},
        {"id": "medical-002", "jmmlu_answer": "A"},
    ]
    results = [
        {"id": "medical-001", "answer_text": "正解はBです．"},
        {"id": "medical-002", "answer_text": "正解はCです．"},
    ]
    assert compute_answer_quality_accuracy(results, dataset) == 0.5


def test_compute_answer_quality_accuracy_returns_zero_when_no_row_gradable() -> None:
    """No JMMLU-sourced rows at all returns 0.0 rather than dividing by zero."""
    dataset = [{"id": "compound-001"}]
    results = [{"id": "compound-001", "answer_text": "anything"}]
    assert compute_answer_quality_accuracy(results, dataset) == 0.0


def test_build_llm_judge_prompt_includes_query_and_response() -> None:
    """The judge prompt embeds both the original query and the expert's response."""
    prompt = build_llm_judge_prompt("頭痛が続いています", "内科の受診をお勧めします")
    assert "頭痛が続いています" in prompt
    assert "内科の受診をお勧めします" in prompt


async def test_judge_response_quality_extracts_score_from_clean_json() -> None:
    """A well-formed judge response yields the parsed integer score."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"score": 4}'

    score = await judge_response_quality(
        ollama_client, "judge-model", "query", "response", timeout_s=2.0
    )
    assert score == 4


async def test_judge_response_quality_returns_none_on_out_of_range_score() -> None:
    """A score outside 1-5 is treated as ungraded, not clamped."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"score": 9}'

    score = await judge_response_quality(
        ollama_client, "judge-model", "query", "response", timeout_s=2.0
    )
    assert score is None


async def test_judge_response_quality_returns_none_on_invalid_json() -> None:
    """An unparseable judge response returns None rather than raising."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "I cannot evaluate this."

    score = await judge_response_quality(
        ollama_client, "judge-model", "query", "response", timeout_s=2.0
    )
    assert score is None


def test_compute_end_to_end_accuracy_requires_both_routing_and_quality() -> None:
    """A row must be correctly routed AND pass quality grading to count."""
    routing_results = [
        {"id": "q1", "selected_domain": "medical", "expected_domains": ["medical"]},
        {"id": "q2", "selected_domain": "legal", "expected_domains": ["medical"]},  # misrouted
        {
            "id": "q3",
            "selected_domain": "medical",
            "expected_domains": ["medical"],
        },  # routed but fails quality
    ]
    quality_pass_by_id = {"q1": True, "q2": True, "q3": False}
    assert compute_end_to_end_accuracy(routing_results, quality_pass_by_id) == pytest.approx(1 / 3)


def test_compute_end_to_end_accuracy_treats_ungraded_rows_as_failing() -> None:
    """A row missing from quality_pass_by_id (not yet graded) counts as not passing."""
    routing_results = [{"id": "q1", "selected_domain": "medical", "expected_domains": ["medical"]}]
    assert compute_end_to_end_accuracy(routing_results, {}) == 0.0


def test_compute_end_to_end_accuracy_empty_results_is_zero() -> None:
    """An empty result set returns 0.0 rather than dividing by zero."""
    assert compute_end_to_end_accuracy([], {}) == 0.0


def test_judge_quality_pass_threshold_is_within_the_1_to_5_rubric() -> None:
    """Sanity check that the pass threshold constant is a valid rubric score."""
    assert 1 <= JUDGE_QUALITY_PASS_THRESHOLD <= 5


def test_compute_latency_breakdown_splits_duration_from_gen_time() -> None:
    """Mean duration and mean dispatch gen time are averaged separately, other_ms is the residual."""
    results = [
        {"duration_ms": 1000, "dispatch_gen_time_ms": 800},
        {"duration_ms": 2000, "dispatch_gen_time_ms": 1600},
    ]
    breakdown = compute_latency_breakdown(results)
    assert breakdown["mean_duration_ms"] == pytest.approx(1500.0)
    assert breakdown["mean_dispatch_gen_time_ms"] == pytest.approx(1200.0)
    assert breakdown["mean_other_ms"] == pytest.approx(300.0)
    assert breakdown["rows_with_dispatch_timing"] == 2


def test_compute_latency_breakdown_excludes_rows_without_dispatch_timing() -> None:
    """A fallback/failed row (dispatch_gen_time_ms is None) is excluded, not treated as 0."""
    results = [
        {"duration_ms": 1000, "dispatch_gen_time_ms": 800},
        {"duration_ms": 5000, "dispatch_gen_time_ms": None},
    ]
    breakdown = compute_latency_breakdown(results)
    assert breakdown["rows_with_dispatch_timing"] == 1
    assert breakdown["mean_duration_ms"] == pytest.approx(1000.0)


def test_compute_latency_breakdown_empty_results_is_zero() -> None:
    """No dispatched rows at all returns all-zero fields rather than dividing by zero."""
    breakdown = compute_latency_breakdown([])
    assert breakdown["rows_with_dispatch_timing"] == 0
    assert breakdown["mean_duration_ms"] == 0.0

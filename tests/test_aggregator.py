"""Tests for the probe aggregation and top-k selection logic."""

from unittest.mock import AsyncMock

import pytest

from aggregator import (
    select_best_dispatch_response,
    select_best_dispatch_response_llm_judge,
    select_best_dispatch_response_majority_vote,
    select_dispatch_targets,
    validate_aggregation_method,
)
from expert_backend import OllamaClient
from protocol import DispatchResponse, ProbeResponse


def _probe_response(node_id: str, confidence: float) -> ProbeResponse:
    """Create a ProbeResponse for testing."""
    return ProbeResponse(
        request_id="uuid-1", node_id=node_id, confidence=confidence, estimated_latency_ms=100
    )


def _dispatch_response(
    node_id: str, confidence: float, answer_text: str | None = None
) -> DispatchResponse:
    """Create a DispatchResponse for testing."""
    return DispatchResponse(
        request_id="uuid-1",
        node_id=node_id,
        answer_text=answer_text if answer_text is not None else f"answer from {node_id}",
        confidence=confidence,
        gen_time_ms=100,
    )


def test_select_dispatch_targets_returns_empty_when_none_eligible() -> None:
    """Return an empty list when all confidences are below the threshold."""
    responses = [_probe_response("A", 0.1), _probe_response("B", 0.2)]
    assert select_dispatch_targets(responses, confidence_threshold=0.5) == []


def test_select_dispatch_targets_returns_highest_confidence_first() -> None:
    """Sort results in descending order by confidence."""
    responses = [_probe_response("A", 0.6), _probe_response("B", 0.9)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=2)
    assert [t.node_id for t in targets] == ["B", "A"]


def test_select_dispatch_targets_respects_top_k() -> None:
    """Never return more than top_k results."""
    responses = [_probe_response("A", 0.9), _probe_response("B", 0.8), _probe_response("C", 0.7)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=1)
    assert [t.node_id for t in targets] == ["A"]


def test_select_dispatch_targets_tiebreaks_by_input_order() -> None:
    """Equal confidence preserves the original input order (peers.yaml order)."""
    responses = [_probe_response("first", 0.7), _probe_response("second", 0.7)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=1)
    assert [t.node_id for t in targets] == ["first"]


def test_select_best_dispatch_response_returns_none_for_empty_list() -> None:
    """Return None when every /dispatch call failed."""
    assert select_best_dispatch_response([]) is None


def test_select_best_dispatch_response_picks_highest_confidence() -> None:
    """Among multiple top-k dispatch results, keep the highest-confidence answer."""
    responses = [_dispatch_response("A", 0.6), _dispatch_response("B", 0.9)]
    assert select_best_dispatch_response(responses).node_id == "B"


def test_select_best_dispatch_response_single_result_passthrough() -> None:
    """With a single response (top_k=1, the Phase 0 default), return it unchanged."""
    response = _dispatch_response("A", 0.8)
    assert select_best_dispatch_response([response]) is response


def test_validate_aggregation_method_raises_on_unknown_value() -> None:
    """A config typo (e.g. 'majority') fails at startup rather than silently degrading."""
    with pytest.raises(ValueError, match="unknown aggregation_method"):
        validate_aggregation_method("majority")


def test_select_best_dispatch_response_majority_vote_returns_none_for_empty_list() -> None:
    """Return None when every /dispatch call failed."""
    assert select_best_dispatch_response_majority_vote([]) is None


def test_select_best_dispatch_response_majority_vote_picks_agreed_letter() -> None:
    """Two candidates agreeing on B win even though a lone A has higher confidence."""
    responses = [
        _dispatch_response("A", 0.95, "正解はAです．"),
        _dispatch_response("B", 0.6, "正解はBです．"),
        _dispatch_response("C", 0.7, "正解はBです．"),
    ]
    result = select_best_dispatch_response_majority_vote(responses)
    assert result.node_id == "C"  # the higher-confidence of the two B-agreeing candidates


def test_select_best_dispatch_response_majority_vote_falls_back_when_no_majority() -> None:
    """With no letter shared by 2+ candidates, fall back to max_confidence."""
    responses = [
        _dispatch_response("A", 0.6, "正解はAです．"),
        _dispatch_response("B", 0.9, "正解はBです．"),
    ]
    result = select_best_dispatch_response_majority_vote(responses)
    assert result.node_id == "B"


def test_select_best_dispatch_response_majority_vote_falls_back_on_free_text_answers() -> None:
    """Hand-authored consultation rows have no extractable letter; fall back to max_confidence."""
    responses = [
        _dispatch_response("A", 0.6, "ご相談内容については専門的な対応が必要です．"),
        _dispatch_response("B", 0.9, "まず医療機関を受診することをお勧めします．"),
    ]
    result = select_best_dispatch_response_majority_vote(responses)
    assert result.node_id == "B"


async def test_select_best_dispatch_response_llm_judge_returns_none_for_empty_list() -> None:
    """Return None when every /dispatch call failed."""
    ollama_client = AsyncMock(spec=OllamaClient)
    result = await select_best_dispatch_response_llm_judge([], "q", ollama_client, "judge", 30.0)
    assert result is None


async def test_select_best_dispatch_response_llm_judge_single_result_skips_judge_call() -> None:
    """With only one candidate, return it without spending an extra LLM call on judging."""
    response = _dispatch_response("A", 0.8)
    ollama_client = AsyncMock(spec=OllamaClient)

    result = await select_best_dispatch_response_llm_judge(
        [response], "q", ollama_client, "judge", 30.0
    )
    assert result is response
    ollama_client.generate.assert_not_awaited()


async def test_select_best_dispatch_response_llm_judge_picks_chosen_candidate() -> None:
    """The judge's chosen 1-based index selects the corresponding candidate."""
    responses = [_dispatch_response("A", 0.9), _dispatch_response("B", 0.5)]
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"best": 2}'

    result = await select_best_dispatch_response_llm_judge(
        responses, "q", ollama_client, "judge", 30.0
    )
    assert result.node_id == "B"


async def test_select_best_dispatch_response_llm_judge_falls_back_on_unparseable_response() -> None:
    """When the judge's own response doesn't parse, fall back to max_confidence."""
    responses = [_dispatch_response("A", 0.9), _dispatch_response("B", 0.5)]
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "そのようには回答できません．"

    result = await select_best_dispatch_response_llm_judge(
        responses, "q", ollama_client, "judge", 30.0
    )
    assert result.node_id == "A"


async def test_select_best_dispatch_response_llm_judge_falls_back_on_out_of_range_index() -> None:
    """When the judge picks an index outside the candidate range, fall back to max_confidence."""
    responses = [_dispatch_response("A", 0.9), _dispatch_response("B", 0.5)]
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"best": 5}'

    result = await select_best_dispatch_response_llm_judge(
        responses, "q", ollama_client, "judge", 30.0
    )
    assert result.node_id == "A"

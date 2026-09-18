"""Tests for scripts/generate_multidomain_training_examples.py's filtering,
pair scheduling, and A7 leak-audit logic.
"""

from unittest.mock import AsyncMock

import pytest

from expert_backend import OllamaClient
from scripts.generate_multidomain_training_examples import (
    _char_ngram_jaccard,
    _generate_one,
    _passes_filters,
    _rows_for_pair,
    audit_leak,
    generate_all_rows,
)


def test_passes_filters_accepts_a_well_formed_consultation_sentence() -> None:
    """A plausible 1-2 sentence natural-language query with no four-choice markers passes."""
    query = "労働災害で怪我をした場合の治療費請求と、会社への損害賠償請求の両方について教えてください。"
    assert _passes_filters(query, already_generated=set())


@pytest.mark.parametrize(
    "query",
    [
        "短い",  # F1: below _MIN_QUERY_LENGTH
        "あ" * 250,  # F1: above _MAX_QUERY_LENGTH
    ],
)
def test_passes_filters_rejects_out_of_range_length(query: str) -> None:
    """F1: length must be within [_MIN_QUERY_LENGTH, _MAX_QUERY_LENGTH]."""
    assert not _passes_filters(query, already_generated=set())


def test_passes_filters_rejects_four_choice_markers() -> None:
    """F2: a generated row reproducing JMMLU's four-choice style must be rejected."""
    query = "契約書の解釈について教えてください。A. 有効 B. 無効 C. 一部無効 D. 保留"
    assert not _passes_filters(query, already_generated=set())


def test_passes_filters_rejects_exact_duplicates() -> None:
    """F3: a query identical to one already accepted for this generation run is rejected."""
    query = "労働災害で怪我をした場合の治療費請求と、会社への損害賠償請求の両方について教えてください。"
    assert not _passes_filters(query, already_generated={query})


def test_passes_filters_rejects_multi_line_output() -> None:
    """F4: a multi-line response (e.g. the model wrote several candidate questions) is rejected."""
    query = "1文目の相談内容です。\n2文目の別の相談内容です。"
    assert not _passes_filters(query, already_generated=set())


def test_rows_for_pair_gives_legal_pairs_more_rows() -> None:
    """legal-involving pairs get per_pair_legal rows; all other pairs get per_pair rows
    (journal.md Iter60 plan: legal is the smallest training domain)."""
    assert _rows_for_pair("legal", "medical", per_pair=3, per_pair_legal=5) == 5
    assert _rows_for_pair("medical", "legal", per_pair=3, per_pair_legal=5) == 5
    assert _rows_for_pair("medical", "education", per_pair=3, per_pair_legal=5) == 3


async def test_generate_one_retries_up_to_max_attempts_then_gives_up() -> None:
    """If every attempt fails the filters, _generate_one returns None rather than
    looping forever or returning an invalid row."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "A. B. C. D. これは四択形式です"  # always rejected by F2

    result = await _generate_one(
        ollama_client, "some-model", "legal", "medical", already_generated=set(), max_attempts=3
    )

    assert result is None
    assert ollama_client.generate.call_count == 3


async def test_generate_one_returns_the_first_query_that_passes_filters() -> None:
    """_generate_one stops retrying as soon as a candidate passes all filters."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = [
        "A. B. C. D. だめな候補",
        "労働災害で怪我をした場合の治療費請求と、会社への損害賠償請求の両方について教えてください。",
    ]

    result = await _generate_one(
        ollama_client, "some-model", "legal", "medical", already_generated=set(), max_attempts=3
    )

    assert result == "労働災害で怪我をした場合の治療費請求と、会社への損害賠償請求の両方について教えてください。"
    assert ollama_client.generate.call_count == 2


async def test_generate_all_rows_produces_synth_ids_and_domain_list_pairs() -> None:
    """generate_all_rows() writes {id, query, domain: [d1, d2]} rows with the expected id scheme."""
    ollama_client = AsyncMock(spec=OllamaClient)
    # Two distinct candidates: F3 (no exact duplicates) would otherwise reject
    # the second slot's candidate if it were identical to the first slot's.
    ollama_client.generate.side_effect = [
        "労働災害で怪我をした場合の治療費請求と、会社への損害賠償請求の両方について教えてください。",
        "医療過誤があった場合の損害賠償請求の進め方について教えてください。",
    ]

    rows = await generate_all_rows(
        ollama_client, "some-model", domains=["legal", "medical"], per_pair=2, per_pair_legal=2
    )

    assert len(rows) == 2
    assert rows[0]["id"] == "synth-legal-medical-001"
    assert rows[1]["id"] == "synth-legal-medical-002"
    assert all(row["domain"] == ["legal", "medical"] for row in rows)


def test_char_ngram_jaccard_is_1_for_identical_strings_and_0_for_disjoint_strings() -> None:
    """Sanity-check the similarity metric's boundary behavior before trusting the A7 threshold."""
    assert _char_ngram_jaccard("こんにちは世界", "こんにちは世界") == pytest.approx(1.0)
    assert _char_ngram_jaccard("abc", "xyz") == pytest.approx(0.0)


def test_audit_leak_flags_a_near_duplicate_of_a_compound_question() -> None:
    """A7: a generated row that is a near-verbatim copy of a _COMPOUND_QUESTIONS entry
    must score a high max_jaccard, so the pre-registered >=0.9 gate would catch it."""
    compound_questions = [
        ("仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．", ["medical", "legal"]),
    ]
    generated_rows = [
        {"id": "synth-legal-medical-001", "query": "仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．"},
    ]

    report = audit_leak(generated_rows, compound_questions)

    assert report["max_jaccard"] == pytest.approx(1.0)


def test_audit_leak_reports_low_similarity_for_unrelated_text() -> None:
    """A generated row unrelated to any compound question should score well below the
    0.9 leak threshold."""
    compound_questions = [
        ("仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．", ["medical", "legal"]),
    ]
    generated_rows = [
        {"id": "synth-mathematics-computer_science-001", "query": "アルゴリズムの計算量を数式で厳密に見積もる方法を知りたいです。"},
    ]

    report = audit_leak(generated_rows, compound_questions)

    assert report["max_jaccard"] < 0.9

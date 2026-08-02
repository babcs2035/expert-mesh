"""Aggregate probe responses and select the top-k dispatch targets by confidence."""

import json
import re

from evaluation import extract_answer_letter
from expert_backend import OllamaClient
from protocol import DispatchResponse, ProbeResponse

# research_frontier item 5 (2026-07-30): design doc 2.5 names three ways to
# pick a final answer among multiple /dispatch candidates (top_k > 1).
# max_confidence was Phase 0's only implementation; majority_vote and
# llm_judge fill in the two the docstring on select_best_dispatch_response
# used to call "a Phase 2+ upgrade". All three are no-ops when top_k == 1
# (a single candidate is always "the best").
AGGREGATION_METHOD_MAX_CONFIDENCE = "max_confidence"
AGGREGATION_METHOD_MAJORITY_VOTE = "majority_vote"
AGGREGATION_METHOD_LLM_JUDGE = "llm_judge"
VALID_AGGREGATION_METHODS = frozenset(
    {AGGREGATION_METHOD_MAX_CONFIDENCE, AGGREGATION_METHOD_MAJORITY_VOTE, AGGREGATION_METHOD_LLM_JUDGE}
)

_JUDGE_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_JUDGE_MAX_TOKENS = 20
_JUDGE_TEMPERATURE = 0.0


def select_dispatch_targets(
    probe_responses: list[ProbeResponse],
    confidence_threshold: float,
    top_k: int = 1,
    dispatch_candidate_threshold: float | None = None,
) -> list[ProbeResponse]:
    """Filter nodes above the confidence threshold and return the top-k.

    Rank 1 (highest confidence) uses ``confidence_threshold``.
    Rank 2 and below use ``dispatch_candidate_threshold`` (defaults to
    ``confidence_threshold`` for backward compatibility).

    Uses stable sort so that nodes with equal confidence preserve their
    input order (matching peers.yaml declaration order). Returns an empty
    list when no node qualifies.
    """
    if dispatch_candidate_threshold is None:
        dispatch_candidate_threshold = confidence_threshold

    # Sort all responses by confidence descending (stable: preserves input
    # order for ties, i.e. peers.yaml declaration order).
    sorted_responses = sorted(probe_responses, key=lambda r: r.confidence, reverse=True)

    if not sorted_responses:
        return []

    rank_1 = sorted_responses[0]
    rest = sorted_responses[1:]

    # Rank 1: use the primary confidence threshold.
    if rank_1.confidence < confidence_threshold:
        return []

    # Rank 2+: use the separate dispatch candidate threshold.
    qualified_rest = [
        r for r in rest if r.confidence >= dispatch_candidate_threshold
    ]

    candidates = [rank_1] + qualified_rest
    return candidates[:top_k]


def validate_aggregation_method(aggregation_method: str) -> None:
    """Raise ValueError if aggregation_method is not one of VALID_AGGREGATION_METHODS.

    Matches http_server.py's validate_node_config_values philosophy: fail at
    startup on a config typo instead of silently falling back to max_confidence.
    """
    if aggregation_method not in VALID_AGGREGATION_METHODS:
        raise ValueError(f"unknown aggregation_method: {aggregation_method!r}")


def select_best_dispatch_response(
    dispatch_responses: list[DispatchResponse],
) -> DispatchResponse | None:
    """Pick the highest-confidence answer among multiple /dispatch results.

    Each DispatchResponse carries the same self-reported confidence computed
    during /probe, so this is a zero-extra-cost selection that requires no
    further LLM calls. Used directly when aggregation_method=max_confidence,
    and as the fallback for majority_vote/llm_judge when they can't reach a
    decision (see select_best_dispatch_response_majority_vote and
    select_best_dispatch_response_llm_judge). Returns None when every
    dispatch failed (empty input).
    """
    if not dispatch_responses:
        return None
    return max(dispatch_responses, key=lambda r: r.confidence)


def select_best_dispatch_response_majority_vote(
    dispatch_responses: list[DispatchResponse],
) -> DispatchResponse | None:
    """Pick the answer agreed on by the most candidates, breaking ties by confidence.

    Only meaningful for JMMLU-style four-choice questions, where
    extract_answer_letter can reduce each candidate's free-text answer to a
    single A/B/C/D pick. Falls back to select_best_dispatch_response (the
    max_confidence policy) when fewer than 2 candidates share a letter — the
    hand-authored consultation rows (no canonical single answer) always take
    this path, as does any question where every candidate disagrees.
    """
    if not dispatch_responses:
        return None
    letters_by_response = {
        id(r): extract_answer_letter(r.answer_text) for r in dispatch_responses
    }
    vote_counts: dict[str, int] = {}
    for letter in letters_by_response.values():
        if letter is not None:
            vote_counts[letter] = vote_counts.get(letter, 0) + 1
    if not vote_counts or max(vote_counts.values()) < 2:
        return select_best_dispatch_response(dispatch_responses)
    winning_letter = max(vote_counts, key=lambda letter: vote_counts[letter])
    winning_group = [
        r for r in dispatch_responses if letters_by_response[id(r)] == winning_letter
    ]
    return select_best_dispatch_response(winning_group)


def _build_judge_selection_prompt(query: str, dispatch_responses: list[DispatchResponse]) -> str:
    """Prompt asking the judge to pick the best of N candidate answers by index."""
    candidates = "\n\n".join(
        f"回答{i + 1}: {r.answer_text}" for i, r in enumerate(dispatch_responses)
    )
    return (
        "次の質問に対する複数の専門家の回答を読み，最も正確で質の高い回答を1つ選んでください．\n\n"
        f"質問: {query}\n\n{candidates}\n\n"
        '回答は{"best": <1から始まる回答番号の整数>}という形式の1行のJSONのみとし，'
        "他のキーや説明文は含めないでください．"
    )


def _parse_judge_selection(raw_response: str, candidate_count: int) -> int | None:
    """Extract the chosen 1-based index; None on parse failure or out-of-range value."""
    match = _JUDGE_JSON_PATTERN.search(raw_response)
    if match is None:
        return None
    try:
        best = int(json.loads(match.group())["best"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return best if 1 <= best <= candidate_count else None


async def select_best_dispatch_response_llm_judge(
    dispatch_responses: list[DispatchResponse],
    query: str,
    ollama_client: OllamaClient,
    judge_model: str,
    timeout_s: float,
) -> DispatchResponse | None:
    """Ask an LLM judge to pick the best of multiple /dispatch candidates.

    Falls back to select_best_dispatch_response (max_confidence) when the
    judge's own response doesn't parse to a valid 1-based index, or when
    there is only one candidate to begin with (no judgment call to make).
    """
    if not dispatch_responses:
        return None
    if len(dispatch_responses) == 1:
        return dispatch_responses[0]
    raw_response = await ollama_client.generate(
        judge_model,
        _build_judge_selection_prompt(query, dispatch_responses),
        timeout_s=timeout_s,
        max_tokens=_JUDGE_MAX_TOKENS,
        temperature=_JUDGE_TEMPERATURE,
    )
    chosen_index = _parse_judge_selection(raw_response, len(dispatch_responses))
    if chosen_index is None:
        return select_best_dispatch_response(dispatch_responses)
    return dispatch_responses[chosen_index - 1]

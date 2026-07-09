"""Aggregate probe responses and select the top-k dispatch targets by confidence."""

from protocol import DispatchResponse, ProbeResponse


def select_dispatch_targets(
    probe_responses: list[ProbeResponse],
    confidence_threshold: float,
    top_k: int = 1,
) -> list[ProbeResponse]:
    """Filter nodes above the confidence threshold and return the top-k.

    Uses stable sort so that nodes with equal confidence preserve their
    input order (matching peers.yaml declaration order). Returns an empty
    list when no node qualifies.
    """
    eligible = [r for r in probe_responses if r.confidence >= confidence_threshold]
    return sorted(eligible, key=lambda r: r.confidence, reverse=True)[:top_k]


def select_best_dispatch_response(
    dispatch_responses: list[DispatchResponse],
) -> DispatchResponse | None:
    """Pick the final answer among multiple /dispatch results (design doc 2.5).

    Design doc 2.5 allows either "simple majority vote" or "LLM-as-judge"
    when top_k > 1. Phase 0 implements the simplest of the two: each
    DispatchResponse carries the same self-reported confidence computed
    during /probe, so picking the highest-confidence responder is a
    zero-extra-cost selection that requires no further LLM calls.
    LLM-as-judge is left as a Phase 2+ upgrade to this function.
    Returns None when every dispatch failed (empty input).
    """
    if not dispatch_responses:
        return None
    return max(dispatch_responses, key=lambda r: r.confidence)

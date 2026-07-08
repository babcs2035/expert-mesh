"""Aggregate probe responses and select the top-k dispatch targets by confidence."""

from protocol import ProbeResponse


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

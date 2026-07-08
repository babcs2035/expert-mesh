"""Tests for the probe aggregation and top-k selection logic."""

from aggregator import select_dispatch_targets
from protocol import ProbeResponse


def _probe_response(node_id: str, confidence: float) -> ProbeResponse:
    """Create a ProbeResponse for testing."""
    return ProbeResponse(
        request_id="uuid-1", node_id=node_id, confidence=confidence, estimated_latency_ms=100
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

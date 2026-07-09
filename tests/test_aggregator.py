"""Tests for the probe aggregation and top-k selection logic."""

from aggregator import select_best_dispatch_response, select_dispatch_targets
from protocol import DispatchResponse, ProbeResponse


def _probe_response(node_id: str, confidence: float) -> ProbeResponse:
    """Create a ProbeResponse for testing."""
    return ProbeResponse(
        request_id="uuid-1", node_id=node_id, confidence=confidence, estimated_latency_ms=100
    )


def _dispatch_response(node_id: str, confidence: float) -> DispatchResponse:
    """Create a DispatchResponse for testing."""
    return DispatchResponse(
        request_id="uuid-1",
        node_id=node_id,
        answer_text=f"answer from {node_id}",
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

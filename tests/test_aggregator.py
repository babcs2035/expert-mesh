"""aggregator.pyのprobe結果集約ロジックを検証する．"""

from aggregator import select_dispatch_targets
from protocol import ProbeResponse


def _probe_response(node_id: str, confidence: float) -> ProbeResponse:
    """テスト用のProbeResponseを組み立てる．"""
    return ProbeResponse(
        request_id="uuid-1", node_id=node_id, confidence=confidence, estimated_latency_ms=100
    )


def test_select_dispatch_targets_returns_empty_when_none_eligible() -> None:
    """全ノードのconfidenceが閾値未満なら空リスト（担当者不在）を返す．"""
    responses = [_probe_response("A", 0.1), _probe_response("B", 0.2)]
    assert select_dispatch_targets(responses, confidence_threshold=0.5) == []


def test_select_dispatch_targets_returns_highest_confidence_first() -> None:
    """confidence降順で並べる．"""
    responses = [_probe_response("A", 0.6), _probe_response("B", 0.9)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=2)
    assert [t.node_id for t in targets] == ["B", "A"]


def test_select_dispatch_targets_respects_top_k() -> None:
    """top_kを超える件数は選定しない．"""
    responses = [_probe_response("A", 0.9), _probe_response("B", 0.8), _probe_response("C", 0.7)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=1)
    assert [t.node_id for t in targets] == ["A"]


def test_select_dispatch_targets_tiebreaks_by_input_order() -> None:
    """confidence同点の場合，probe_responsesに渡された順序（peers.yaml記載順）で先勝ちする．"""
    responses = [_probe_response("first", 0.7), _probe_response("second", 0.7)]
    targets = select_dispatch_targets(responses, confidence_threshold=0.5, top_k=1)
    assert [t.node_id for t in targets] == ["first"]

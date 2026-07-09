"""Tests for the requester-side helpers in node.py (peer building, top-k dispatch, fallback)."""

from unittest.mock import AsyncMock, patch

from expert_backend import OllamaClient
from http_client import PeerClient
from node import (
    FALLBACK_PROMPT_TEMPLATE,
    _build_peers,
    _dispatch_to_targets,
    _fallback_answer,
    run_ask_flow,
)
from protocol import DispatchRequest, DispatchResponse, ProbeResponse


def _probe_response(node_id: str, confidence: float) -> ProbeResponse:
    """Create a ProbeResponse for testing."""
    return ProbeResponse(
        request_id="uuid-1", node_id=node_id, confidence=confidence, estimated_latency_ms=100
    )


def test_build_peers_overrides_self_host_to_localhost() -> None:
    """The self node's host is replaced with localhost to avoid hairpin NAT issues."""
    config = {
        "nodes": {
            "node-a": {"host": "192.168.1.10", "port": 8080, "domain": "general"},
            "node-b": {"host": "192.168.1.11", "port": 8080, "domain": "medical"},
        }
    }
    peers = _build_peers(config, self_node_id="node-a")
    self_peer = next(p for p in peers if p["node_id"] == "node-a")
    other_peer = next(p for p in peers if p["node_id"] == "node-b")
    assert self_peer["host"] == "localhost"
    assert other_peer["host"] == "192.168.1.11"


async def test_dispatch_to_targets_single_target_passthrough() -> None:
    """With one probe target (top_k=1), the single dispatch response is returned."""
    peers = [{"node_id": "node-b", "host": "localhost", "port": 8080, "domain": "medical"}]
    targets = [_probe_response("node-b", 0.9)]
    peer_client = AsyncMock(spec=PeerClient)
    expected = DispatchResponse(
        request_id="r1", node_id="node-b", answer_text="answer", confidence=0.9, gen_time_ms=100
    )
    peer_client.dispatch.return_value = expected

    result = await _dispatch_to_targets(
        peer_client, peers, targets, DispatchRequest(request_id="r1", full_query="q"), 30.0
    )
    assert result is expected


async def test_dispatch_to_targets_picks_highest_confidence_among_top_k() -> None:
    """With top_k>1, all targets are dispatched and the best answer is kept."""
    peers = [
        {"node_id": "node-a", "host": "localhost", "port": 8080, "domain": "general"},
        {"node_id": "node-b", "host": "localhost", "port": 8081, "domain": "medical"},
    ]
    targets = [_probe_response("node-a", 0.6), _probe_response("node-b", 0.9)]
    peer_client = AsyncMock(spec=PeerClient)
    response_a = DispatchResponse(
        request_id="r1", node_id="node-a", answer_text="weak", confidence=0.6, gen_time_ms=100
    )
    response_b = DispatchResponse(
        request_id="r1", node_id="node-b", answer_text="strong", confidence=0.9, gen_time_ms=100
    )

    async def _dispatch_side_effect(peer: dict, request: DispatchRequest, timeout_s: float):
        return response_a if peer["node_id"] == "node-a" else response_b

    peer_client.dispatch.side_effect = _dispatch_side_effect

    result = await _dispatch_to_targets(
        peer_client, peers, targets, DispatchRequest(request_id="r1", full_query="q"), 30.0
    )
    assert result.node_id == "node-b"
    assert peer_client.dispatch.await_count == 2


async def test_dispatch_to_targets_returns_none_when_all_fail() -> None:
    """If every dispatched peer fails, the aggregate result is None."""
    peers = [{"node_id": "node-b", "host": "localhost", "port": 8080, "domain": "medical"}]
    targets = [_probe_response("node-b", 0.9)]
    peer_client = AsyncMock(spec=PeerClient)
    peer_client.dispatch.return_value = None

    result = await _dispatch_to_targets(
        peer_client, peers, targets, DispatchRequest(request_id="r1", full_query="q"), 30.0
    )
    assert result is None


async def test_fallback_answer_uses_light_model_and_includes_query() -> None:
    """The fallback path generates via the requester's own light model."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "hedged answer"

    answer = await _fallback_answer(ollama_client, "qwen3.5:9b", "頭痛が続いています")

    assert answer == "hedged answer"
    call_args = ollama_client.generate.call_args
    assert call_args.args[0] == "qwen3.5:9b"
    assert "頭痛が続いています" in call_args.args[1]
    assert FALLBACK_PROMPT_TEMPLATE.split("{query}")[0] in call_args.args[1]


def _config() -> dict:
    """Minimal two-node config for run_ask_flow tests."""
    return {
        "embedding_model": "nomic-embed-text",
        "confidence_threshold": 0.5,
        "nodes": {
            "requester": {
                "host": "192.168.1.10",
                "port": 8080,
                "domain": "general",
                "light_model": "qwen3.5:9b",
                "expert_model": "qwen3.5:9b",
            },
            "expert": {
                "host": "192.168.1.11",
                "port": 8080,
                "domain": "medical",
                "light_model": "qwen3.5:9b",
                "expert_model": "qwen3.5:9b",
            },
        },
    }


async def test_run_ask_flow_returns_dispatch_response_when_expert_found() -> None:
    """A qualifying probe response leads to a populated dispatch_response, no fallback."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.embed.return_value = [0.1, 0.2]
    probe_response = ProbeResponse(
        request_id="ignored", node_id="expert", confidence=0.9, estimated_latency_ms=10
    )
    dispatch_response = DispatchResponse(
        request_id="ignored",
        node_id="expert",
        answer_text="specialist answer",
        confidence=0.9,
        gen_time_ms=50,
    )

    with (
        patch.object(PeerClient, "probe_all", AsyncMock(return_value=[probe_response])),
        patch.object(PeerClient, "dispatch", AsyncMock(return_value=dispatch_response)),
    ):
        result = await run_ask_flow(_config(), "requester", "頭痛がします", ollama_client)

    assert result.fallback_answer is None
    assert result.dispatch_response is dispatch_response


async def test_run_ask_flow_falls_back_when_no_target_qualifies() -> None:
    """When no probe response clears the confidence threshold, the fallback path runs."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.embed.return_value = [0.1, 0.2]
    ollama_client.generate.return_value = "general hedge"
    low_confidence = ProbeResponse(
        request_id="ignored", node_id="expert", confidence=0.1, estimated_latency_ms=10
    )

    with patch.object(PeerClient, "probe_all", AsyncMock(return_value=[low_confidence])):
        result = await run_ask_flow(_config(), "requester", "おすすめの映画は", ollama_client)

    assert result.dispatch_response is None
    assert result.fallback_answer == "general hedge"

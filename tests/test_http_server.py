"""Tests for the FastAPI endpoint contracts in http_server.py."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from expert_backend import OllamaClient
from http_server import (
    CONFIDENCE_ELICITATION_TOP_K_WITH_PROBS,
    CONFIDENCE_SIGNAL_P_TRUE,
    CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY,
    ROUTING_METHOD_EMBEDDING,
    ROUTING_METHOD_SUPERVISED_CLASSIFIER,
    NodeState,
    create_app,
)


def _build_client(
    ollama_client: OllamaClient,
    light_model: str = "qwen3.5:2b",
    expert_model: str = "qwen3.5:9b",
    **state_kwargs: object,
) -> TestClient:
    """Create a TestClient wired to a NodeState with the given OllamaClient.

    Does not enter the lifespan context (no `with`), matching how uvicorn's
    own TestClient usage in these tests predates the warmup/advertise/embed
    startup steps added later; those steps are covered separately by tests
    that do use the `with` form.
    """
    state = NodeState(
        node_id="node-b",
        domain="medical",
        light_model=light_model,
        expert_model=expert_model,
        confidence_threshold=0.5,
        probe_timeout_s=2.0,
        dispatch_timeout_s=30.0,
        ollama_client=ollama_client,
        **state_kwargs,
    )
    return TestClient(create_app(state))


def test_advertise_returns_ok() -> None:
    """The /advertise endpoint returns 200 with {"status": "ok"}."""
    client = _build_client(AsyncMock(spec=OllamaClient))
    response = client.post(
        "/advertise",
        json={
            "node_id": "node-c",
            "domain": "legal",
            "domain_embedding": [0.1, 0.2],
            "load": 0.3,
            "timestamp": 1730000000,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_probe_returns_confidence_from_light_model() -> None:
    """The /probe endpoint returns the confidence extracted from the model's response."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"confidence": 0.87}'
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == 0.87
    assert body["node_id"] == "node-b"


def test_probe_returns_504_on_timeout() -> None:
    """Return 504 when the lightweight model call times out."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = httpx.TimeoutException("timed out")
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 504
    assert response.json() == {"error": "timeout"}


def test_probe_returns_503_when_model_not_ready() -> None:
    """Return 503 when the connection to ollama fails entirely."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = httpx.ConnectError("connection refused")
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 503
    assert response.json() == {"error": "model not ready"}


def test_probe_rejects_invalid_request_body() -> None:
    """Return 400 when required fields are missing from the request body."""
    client = _build_client(AsyncMock(spec=OllamaClient))
    response = client.post("/probe", json={"request_id": "uuid-1"})
    assert response.status_code == 400
    assert response.json() == {"error": "invalid request"}


def test_dispatch_reuses_probe_confidence() -> None:
    """The /dispatch endpoint reuses the confidence calculated during /probe."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = ['{"confidence": 0.9}', "You had a headache for 3 days."]
    client = _build_client(ollama_client)

    probe_response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert probe_response.status_code == 200

    dispatch_response = client.post(
        "/dispatch", json={"request_id": "uuid-1", "full_query": "You had a headache for 3 days."}
    )
    assert dispatch_response.status_code == 200
    body = dispatch_response.json()
    assert body["confidence"] == 0.9
    assert body["answer_text"] == "You had a headache for 3 days."


def test_dispatch_includes_node_domain_in_prompt() -> None:
    """The /dispatch endpoint passes the node's domain into the generation prompt."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "Answer"
    client = _build_client(ollama_client)

    client.post("/dispatch", json={"request_id": "uuid-1", "full_query": "headache and fever"})

    prompt_arg = ollama_client.generate.call_args.args[1]
    assert "medical" in prompt_arg
    assert "headache and fever" in prompt_arg


def test_dispatch_confidence_defaults_to_zero_without_prior_probe() -> None:
    """Return confidence 0.0 when there is no matching prior /probe call."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "Answer"
    client = _build_client(ollama_client)

    response = client.post("/dispatch", json={"request_id": "unknown", "full_query": "question"})
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0


def test_probe_uses_embedding_similarity_when_routing_method_is_embedding() -> None:
    """Method A (embedding) skips the LLM call and scores via cosine similarity."""
    ollama_client = AsyncMock(spec=OllamaClient)
    client = _build_client(ollama_client, routing_method=ROUTING_METHOD_EMBEDDING)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [1.0, 0.0],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    # domain_embedding defaults to [] (not yet warmed up outside lifespan),
    # so no LLM call should occur regardless of the resulting score; the
    # score itself is covered by test_router.py's cosine_similarity tests.
    assert ollama_client.generate.await_count == 0


def test_probe_uses_top_k_elicitation_when_configured() -> None:
    """confidence_elicitation=top_k_with_probs routes /probe through estimate_confidence_top_k."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = (
        '{"candidates": [{"label": "該当する", "probability": 0.8}, '
        '{"label": "該当しない", "probability": 0.2}]}'
    )
    client = _build_client(
        ollama_client, confidence_elicitation=CONFIDENCE_ELICITATION_TOP_K_WITH_PROBS
    )

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    assert response.json()["confidence"] == pytest.approx(0.8)
    prompt_arg = ollama_client.generate.call_args.args[1]
    assert "該当する" in prompt_arg


def test_probe_uses_semantic_entropy_signal_when_configured() -> None:
    """confidence_signal_method=self_consistency_semantic populates confidence_semantic_entropy."""
    ollama_client = AsyncMock(spec=OllamaClient)
    verdict_response = '{"fits": true, "reason": "医療分野の症状である"}'
    entailment_response = '{"same_claim": true}'
    ollama_client.generate.side_effect = [verdict_response] * 5 + [entailment_response] * 4
    client = _build_client(
        ollama_client,
        confidence_signal_method=CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY,
        semantic_sample_count=5,
    )

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == pytest.approx(1.0)
    assert body["confidence_semantic_entropy"] == pytest.approx(0.0)


def test_probe_uses_p_true_signal_when_configured() -> None:
    """confidence_signal_method=p_true populates confidence_p_true via the two-stage flow."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = [
        "この質問は医療分野に該当します．",
        {
            "content": "A",
            "token_logprobs": [
                {"token": "A", "logprob": -0.1, "top_logprobs": {"A": -0.1, "B": -2.0}}
            ],
        },
    ]
    client = _build_client(ollama_client, confidence_signal_method=CONFIDENCE_SIGNAL_P_TRUE)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence_p_true"] == pytest.approx(body["confidence"])
    assert body["confidence_semantic_entropy"] is None


def test_probe_uses_supervised_classifier_without_any_llm_call() -> None:
    """routing_method=supervised_classifier scores via the injected classifier, no LLM call."""
    toy_classifier = LogisticRegression(max_iter=1000)
    toy_classifier.fit(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1]],
        ["medical", "medical", "legal", "legal"],
    )
    ollama_client = AsyncMock(spec=OllamaClient)
    client = _build_client(
        ollama_client,
        routing_method=ROUTING_METHOD_SUPERVISED_CLASSIFIER,
        domain_classifier=toy_classifier,
    )

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [1.0, 0.0],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    assert response.json()["confidence"] > 0.5
    assert ollama_client.generate.await_count == 0


def test_advertise_heartbeat_reaches_configured_peer(monkeypatch) -> None:
    """On startup, the node sends /advertise to every configured peer."""
    import http_server

    monkeypatch.setattr(http_server, "ADVERTISE_INTERVAL_S", 0.05)

    sent_requests: list[object] = []

    class _StubPeerClient:
        """Records every AdvertiseRequest instead of performing real HTTP calls."""

        def __init__(self, peers: list[dict]) -> None:
            self.peers = peers

        async def advertise_all(self, request: object, timeout_s: float) -> None:
            sent_requests.append(request)

    monkeypatch.setattr(http_server, "PeerClient", _StubPeerClient)

    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"confidence": 0.5}'
    ollama_client.get_running_models.return_value = []
    peers = [{"node_id": "peer-a", "host": "peer-a.invalid", "port": 8080, "domain": "legal"}]
    client = _build_client(ollama_client, peers=peers)

    with client:
        import time

        time.sleep(0.2)

    assert len(sent_requests) >= 1
    assert sent_requests[0].node_id == "node-b"
    assert sent_requests[0].domain == "medical"


def _parse_log_events(stdout: str, event: str) -> list[dict]:
    """Extract log_event JSON payloads for a given event name from captured stdout."""
    records = []
    for line in stdout.splitlines():
        _, _, payload = line.partition("] ")
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if record.get("event") == event:
            records.append(record)
    return records


def test_lifespan_logs_gpu_status_when_model_uses_vram(capsys) -> None:
    """After warmup, a model with size_vram > 0 is logged as using_gpu=True."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "ok"
    ollama_client.get_running_models.return_value = [
        {"name": "qwen3.5:9b", "size_vram": 5_000_000_000},
    ]
    client = _build_client(ollama_client, light_model="qwen3.5:9b", expert_model="qwen3.5:9b")

    with client:
        pass

    records = _parse_log_events(capsys.readouterr().out, "gpu_status")
    assert len(records) == 1
    assert records[0]["model"] == "qwen3.5:9b"
    assert records[0]["using_gpu"] is True


def test_lifespan_logs_cpu_only_status_when_model_has_no_vram(capsys) -> None:
    """A model with size_vram == 0 is logged as using_gpu=False (CPU-only)."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "ok"
    ollama_client.get_running_models.return_value = [
        {"name": "qwen3.5:9b", "size_vram": 0},
    ]
    client = _build_client(ollama_client, light_model="qwen3.5:9b", expert_model="qwen3.5:9b")

    with client:
        pass

    records = _parse_log_events(capsys.readouterr().out, "gpu_status")
    assert len(records) == 1
    assert records[0]["using_gpu"] is False


def test_lifespan_logs_gpu_status_for_both_light_and_expert_models(capsys) -> None:
    """Warmup logs GPU status separately for the light and expert models when they differ."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "ok"
    ollama_client.get_running_models.return_value = [
        {"name": "qwen3.5:2b", "size_vram": 0},
        {"name": "qwen3.5:9b", "size_vram": 5_000_000_000},
    ]
    client = _build_client(ollama_client, light_model="qwen3.5:2b", expert_model="qwen3.5:9b")

    with client:
        pass

    records = {r["model"]: r for r in _parse_log_events(capsys.readouterr().out, "gpu_status")}
    assert records["qwen3.5:2b"]["using_gpu"] is False
    assert records["qwen3.5:9b"]["using_gpu"] is True


def test_lifespan_continues_when_gpu_status_check_fails(capsys) -> None:
    """A /api/ps failure is logged but does not prevent the node from serving requests."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "ok"
    ollama_client.get_running_models.side_effect = httpx.ConnectError("connection refused")
    client = _build_client(ollama_client, light_model="qwen3.5:9b", expert_model="qwen3.5:9b")

    with client:
        response = client.post(
            "/advertise",
            json={
                "node_id": "peer-a",
                "domain": "legal",
                "domain_embedding": [],
                "load": 0.0,
                "timestamp": 0,
            },
        )
        assert response.status_code == 200

    records = _parse_log_events(capsys.readouterr().out, "gpu_status_check_failed")
    assert len(records) == 1


def test_node_state_rejects_embedding_postprocess_without_whitening_path() -> None:
    """embedding_postprocess=whiten with no embedding_whitening_path fails at construction, not silently."""
    with pytest.raises(ValueError, match="embedding_whitening_path"):
        _build_client(AsyncMock(spec=OllamaClient), embedding_postprocess="whiten")


def test_lifespan_rejects_supervised_classifier_without_model_path() -> None:
    """routing_method=supervised_classifier with no classifier_model_path fails at lifespan startup."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "ok"
    ollama_client.get_running_models.return_value = []
    client = _build_client(ollama_client, routing_method=ROUTING_METHOD_SUPERVISED_CLASSIFIER)

    with pytest.raises(ValueError, match="classifier_model_path"), client:
        pass

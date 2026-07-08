"""Tests for the FastAPI endpoint contracts in http_server.py."""

from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from expert_backend import OllamaClient
from http_server import NodeState, create_app


def _build_client(ollama_client: OllamaClient) -> TestClient:
    """Create a TestClient wired to a NodeState with the given OllamaClient."""
    state = NodeState(
        node_id="node-b",
        domain="medical",
        light_model="qwen3.5:2b",
        expert_model="qwen3.5:9b",
        confidence_threshold=0.5,
        probe_timeout_s=2.0,
        dispatch_timeout_s=30.0,
        ollama_client=ollama_client,
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

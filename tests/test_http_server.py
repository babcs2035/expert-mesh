"""http_server.pyのFastAPIエンドポイント契約を検証する．ollama呼び出しはモックする．"""

from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from expert_backend import OllamaClient
from http_server import NodeState, create_app


def _build_client(ollama_client: OllamaClient) -> TestClient:
    """モック済みOllamaClientを束縛したTestClientを構築する．"""
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
    """/advertiseは受理応答{"status": "ok"}を返す．"""
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
    """/probeは軽量モデルの応答から抽出したconfidenceを返す．"""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = '{"confidence": 0.87}'
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "頭痛と発熱",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == 0.87
    assert body["node_id"] == "node-b"


def test_probe_returns_504_on_timeout() -> None:
    """軽量モデルがタイムアウトした場合504+{"error": "timeout"}を返す．"""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = httpx.TimeoutException("timed out")
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "頭痛",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 504
    assert response.json() == {"error": "timeout"}


def test_probe_returns_503_when_model_not_ready() -> None:
    """ollamaへの接続自体に失敗した場合503+{"error": "model not ready"}を返す．"""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = httpx.ConnectError("connection refused")
    client = _build_client(ollama_client)

    response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "頭痛",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert response.status_code == 503
    assert response.json() == {"error": "model not ready"}


def test_probe_rejects_invalid_request_body() -> None:
    """必須フィールド欠如は400+{"error": "invalid request"}を返す．"""
    client = _build_client(AsyncMock(spec=OllamaClient))
    response = client.post("/probe", json={"request_id": "uuid-1"})
    assert response.status_code == 400
    assert response.json() == {"error": "invalid request"}


def test_dispatch_reuses_probe_confidence() -> None:
    """/dispatchは同一request_idの/probeで算出したconfidenceを再利用する．"""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = ['{"confidence": 0.9}', "3日前から頭痛が続いています．"]
    client = _build_client(ollama_client)

    probe_response = client.post(
        "/probe",
        json={
            "request_id": "uuid-1",
            "query_summary": "頭痛",
            "query_embedding": [0.1],
            "from": "node-a",
        },
    )
    assert probe_response.status_code == 200

    dispatch_response = client.post(
        "/dispatch", json={"request_id": "uuid-1", "full_query": "3日前から頭痛が続いています．"}
    )
    assert dispatch_response.status_code == 200
    body = dispatch_response.json()
    assert body["confidence"] == 0.9
    assert body["answer_text"] == "3日前から頭痛が続いています．"


def test_dispatch_confidence_defaults_to_zero_without_prior_probe() -> None:
    """対応する/probeがない場合，confidenceは0.0とする．"""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "回答"
    client = _build_client(ollama_client)

    response = client.post("/dispatch", json={"request_id": "unknown", "full_query": "質問"})
    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0

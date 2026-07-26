"""Tests for OllamaClient's HTTP request/response handling (expert_backend.py)."""

import json

import httpx

from expert_backend import OllamaClient


def _client_with_transport(monkeypatch, handler) -> OllamaClient:
    """Build an OllamaClient whose internal httpx calls are served by a MockTransport.

    OllamaClient constructs its own httpx.AsyncClient(timeout=...)
    internally without exposing a transport hook, so httpx.AsyncClient
    itself is monkeypatched to a subclass that always injects the given
    MockTransport. This exercises the real request-building/response-
    parsing code without a live network call.
    """
    original_async_client = httpx.AsyncClient

    class _PatchedAsyncClient(original_async_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _PatchedAsyncClient)
    return OllamaClient(host="http://mock-host:11434")


async def test_generate_without_top_logprobs_omits_the_field_from_payload(monkeypatch) -> None:
    """When top_logprobs is not passed, the request body has no top_logprobs key."""
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": "answer"},
                "logprobs": [{"token": "A", "logprob": -0.1}],
            },
        )

    client = _client_with_transport(monkeypatch, handler)
    await client.generate("model", "prompt", logprobs=1)

    assert "top_logprobs" not in captured_request["body"]


async def test_generate_with_top_logprobs_includes_it_in_payload(monkeypatch) -> None:
    """Passing top_logprobs adds it to the request body alongside logprobs:true."""
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"content": "answer"},
                "logprobs": [{"token": "A", "logprob": -0.1, "top_logprobs": []}],
            },
        )

    client = _client_with_transport(monkeypatch, handler)
    await client.generate("model", "prompt", logprobs=1, top_logprobs=5)

    assert captured_request["body"]["logprobs"] is True
    assert captured_request["body"]["top_logprobs"] == 5


async def test_generate_parses_top_logprobs_alternatives_per_position(monkeypatch) -> None:
    """Each token position's top_logprobs alternatives are collected into a token->logprob dict."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": "A"},
                "logprobs": [
                    {
                        "token": "A",
                        "logprob": -0.05,
                        "top_logprobs": [
                            {"token": "A", "logprob": -0.05},
                            {"token": "B", "logprob": -3.0},
                        ],
                    }
                ],
            },
        )

    client = _client_with_transport(monkeypatch, handler)
    result = await client.generate("model", "prompt", logprobs=1, top_logprobs=5)

    assert result["token_logprobs"][0]["top_logprobs"] == {"A": -0.05, "B": -3.0}

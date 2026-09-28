"""Tests for OllamaClient's HTTP request/response handling (expert_backend.py)."""

import json

import httpx
import pytest
from unittest.mock import AsyncMock

from expert_backend import DEFAULT_PROMPT_TEMPLATE, OllamaClient, embed_query_views


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


async def test_embed_uses_default_prompt_template_when_none_given(monkeypatch) -> None:
    """Iter99: omitting prompt_template keeps the pre-Iter99 Qwen3-Embedding instruct format."""
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    client = _client_with_transport(monkeypatch, handler)
    await client.embed("model", "text", instruction="task")

    assert captured_request["body"]["prompt"] == "Instruct: task\nQuery: text"


async def test_embed_applies_a_model_specific_prompt_template(monkeypatch) -> None:
    """Iter99 (embedding_model_replacement): a custom prompt_template (e.g. ruri-v3-310m's
    'トピック: {text}') is used verbatim instead of the Qwen3-Embedding instruct format.
    """
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    client = _client_with_transport(monkeypatch, handler)
    await client.embed("model", "text", instruction="task", prompt_template="トピック: {text}")

    assert captured_request["body"]["prompt"] == "トピック: text"


async def test_embed_without_instruction_ignores_prompt_template(monkeypatch) -> None:
    """No instruction means no prefix at all, regardless of prompt_template (plain view)."""
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": [0.1, 0.2]})

    client = _client_with_transport(monkeypatch, handler)
    await client.embed("model", "text", instruction=None, prompt_template="トピック: {text}")

    assert captured_request["body"]["prompt"] == "text"


async def test_embed_query_views_concat_forwards_prompt_template_to_instructed_view_only() -> None:
    """Iter99: prompt_template is passed through to the instructed-view embed() call; the
    plain view is unaffected since it never receives an instruction.
    """
    client = AsyncMock(spec=OllamaClient)
    client.embed.side_effect = [[0.1, 0.2], [0.9, 0.8]]  # plain, then instructed

    result = await embed_query_views(
        client, "model", "query", instruction="task", concat_views=True, prompt_template="トピック: {text}"
    )

    assert result == [0.1, 0.2, 0.9, 0.8]
    plain_call, instructed_call = client.embed.call_args_list
    assert "prompt_template" not in plain_call.kwargs
    assert instructed_call.kwargs["prompt_template"] == "トピック: {text}"


async def test_embed_query_views_without_concat_forwards_default_prompt_template() -> None:
    """concat_views=False forwards the default prompt_template unchanged (single embed() call)."""
    client = AsyncMock(spec=OllamaClient)
    client.embed.return_value = [1.0, 2.0]

    await embed_query_views(client, "model", "query", instruction="task")

    call = client.embed.await_args
    assert call.kwargs["prompt_template"] == DEFAULT_PROMPT_TEMPLATE


async def test_embed_query_views_without_concat_delegates_to_a_single_embed_call() -> None:
    """concat_views=False (the default) is exactly one embed() call, unchanged from pre-Iter82."""
    client = AsyncMock(spec=OllamaClient)
    client.embed.return_value = [1.0, 2.0]

    result = await embed_query_views(client, "model", "query", instruction="task")

    assert result == [1.0, 2.0]
    client.embed.assert_awaited_once()
    call = client.embed.await_args
    assert call.args == ("model", "query")
    assert call.kwargs["instruction"] == "task"


async def test_embed_query_views_concat_orders_plain_before_instructed() -> None:
    """Iter82 (embedding_view_concatenation): concatenation order is fixed [plain, instructed],
    regardless of call order, matching the order G1's CV (np.hstack([P0, P1])) used and the
    order scripts/screen_embedding_models.py's `concat` pseudo-candidate uses.
    """
    client = AsyncMock(spec=OllamaClient)
    client.embed.side_effect = [[0.1, 0.2], [0.9, 0.8]]  # plain, then instructed

    result = await embed_query_views(client, "model", "query", instruction="task", concat_views=True)

    assert result == [0.1, 0.2, 0.9, 0.8]
    plain_call, instructed_call = client.embed.call_args_list
    assert plain_call.kwargs["instruction"] is None
    assert instructed_call.kwargs["instruction"] == "task"


async def test_embed_query_views_concat_without_instruction_raises() -> None:
    """concat_views=True with instruction=None indicates a config mistake (concatenating a
    view with itself), so this must raise rather than silently produce a degenerate feature.
    """
    client = AsyncMock(spec=OllamaClient)

    with pytest.raises(ValueError):
        await embed_query_views(client, "model", "query", instruction=None, concat_views=True)

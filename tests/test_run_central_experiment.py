"""Tests for the central-router experiment script's routing/fallback logic."""

from unittest.mock import AsyncMock

import numpy as np

from node import FALLBACK_PROMPT_TEMPLATE
from scripts.run_central_experiment import _run_one


class _StubClassifier:
    """A minimal stand-in for the real LogisticRegression classifier."""

    def __init__(self, classes: list[str], probabilities: list[float]) -> None:
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, _query_embeddings: list[list[float]]):
        return np.array([self._probabilities])


def _domain_nodes() -> dict[str, dict]:
    return {
        "general": {"host": "general-host", "model": "expert-mesh-general-lora"},
        "medical": {"host": "medical-host", "model": "expert-mesh-medical-lora"},
    }


async def test_run_one_dispatches_to_argmax_domain_above_threshold() -> None:
    """When the argmax domain's probability clears confidence_threshold, dispatch normally."""
    classifier = _StubClassifier(["general", "medical"], [0.2, 0.8])
    embed_client = AsyncMock()
    embed_client.embed.return_value = [0.0] * 768
    generator = AsyncMock()
    generator.generate.return_value = "正解はAです．"

    record = await _run_one(
        classifier,
        _domain_nodes(),
        "nomic-embed-text",
        {"id": "q1", "query": "some question", "expected_domains": ["medical"]},
        embed_client,
        generator,
        confidence_threshold=0.5,
        fallback_node_host="general-host",
        fallback_model="qwen3.5:4b-q4_K_M",
    )

    assert record["selected_domain"] == "medical"
    assert record["used_fallback"] is False
    assert record["confidence"] == 0.8
    generator.generate.assert_awaited_once()
    assert generator.generate.await_args.kwargs["node_host"] == "medical-host"


async def test_run_one_falls_back_below_confidence_threshold() -> None:
    """When no domain clears confidence_threshold, fall back like node.py's run_ask_flow.

    Regression test for backlog B46: Iter26's first run always dispatched via
    a bare argmax with no threshold, which made every McNemar-discordant row
    against the distributed mesh a fallback-policy difference rather than an
    architecture effect.
    """
    classifier = _StubClassifier(["general", "medical"], [0.3, 0.4])
    embed_client = AsyncMock()
    embed_client.embed.return_value = [0.0] * 768
    generator = AsyncMock()
    generator.generate.return_value = "一般的な回答です．"

    record = await _run_one(
        classifier,
        _domain_nodes(),
        "nomic-embed-text",
        {"id": "q2", "query": "ambiguous question", "expected_domains": ["medical"]},
        embed_client,
        generator,
        confidence_threshold=0.5,
        fallback_node_host="general-host",
        fallback_model="qwen3.5:4b-q4_K_M",
    )

    assert record["selected_domain"] == "general"
    assert record["used_fallback"] is True
    assert record["confidence"] is None
    generator.generate.assert_awaited_once()
    call_kwargs = generator.generate.await_args.kwargs
    assert call_kwargs["node_host"] == "general-host"
    assert call_kwargs["model"] == "qwen3.5:4b-q4_K_M"
    assert call_kwargs["prompt"] == FALLBACK_PROMPT_TEMPLATE.format(query="ambiguous question")

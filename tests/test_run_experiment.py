"""Tests for run_experiment.py's per-row recording logic."""

import io
import json
from unittest.mock import AsyncMock, patch

from run_experiment import run_experiment
from http_client import PeerClient
from protocol import DispatchResponse, ProbeResponse


def _config() -> dict:
    """Minimal two-node config matching the shape read from config.yaml."""
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


async def test_run_experiment_records_dispatch_outcome(monkeypatch) -> None:
    """A dispatched answer is recorded with the responding node's domain."""
    monkeypatch.setattr(
        "run_experiment.OllamaClient",
        lambda: AsyncMock(spec=None, embed=AsyncMock(return_value=[0.1])),
    )

    def _fake_read_dataset(path: str) -> list[dict]:
        return [{"id": "medical-001", "query": "頭痛が続きます", "expected_domains": ["medical"]}]

    monkeypatch.setattr("run_experiment._read_dataset", _fake_read_dataset)

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
        output = io.StringIO()
        count = await run_experiment(_config(), "requester", "unused", output)

    assert count == 1
    record = json.loads(output.getvalue().strip())
    assert record["selected_domain"] == "medical"
    assert record["selected_node_id"] == "expert"
    assert record["used_fallback"] is False
    assert record["dispatch_failed"] is False
    assert record["answer_text"] == "specialist answer"


async def test_run_experiment_records_fallback_outcome(monkeypatch) -> None:
    """When no expert qualifies, the record reflects the requester's own fallback."""
    monkeypatch.setattr(
        "run_experiment.OllamaClient",
        lambda: AsyncMock(embed=AsyncMock(return_value=[0.1]), generate=AsyncMock(return_value="hedge")),
    )

    def _fake_read_dataset(path: str) -> list[dict]:
        return [{"id": "general-001", "query": "おすすめの映画は", "expected_domains": ["general"]}]

    monkeypatch.setattr("run_experiment._read_dataset", _fake_read_dataset)

    low_confidence = ProbeResponse(
        request_id="ignored", node_id="expert", confidence=0.1, estimated_latency_ms=10
    )
    with patch.object(PeerClient, "probe_all", AsyncMock(return_value=[low_confidence])):
        output = io.StringIO()
        count = await run_experiment(_config(), "requester", "unused", output)

    assert count == 1
    record = json.loads(output.getvalue().strip())
    assert record["used_fallback"] is True
    assert record["dispatch_failed"] is False
    assert record["selected_domain"] == "general"
    assert record["selected_node_id"] == "requester"
    assert record["confidence"] is None


async def test_run_experiment_records_dispatch_failure_outcome(monkeypatch) -> None:
    """When probing finds a qualifying expert but every /dispatch call to it fails,
    the record must reflect a system-level failure, not a fabricated domain match."""
    monkeypatch.setattr(
        "run_experiment.OllamaClient",
        lambda: AsyncMock(embed=AsyncMock(return_value=[0.1])),
    )

    def _fake_read_dataset(path: str) -> list[dict]:
        return [{"id": "medical-001", "query": "頭痛が続きます", "expected_domains": ["medical"]}]

    monkeypatch.setattr("run_experiment._read_dataset", _fake_read_dataset)

    qualifying_probe = ProbeResponse(
        request_id="ignored", node_id="expert", confidence=0.9, estimated_latency_ms=10
    )
    with (
        patch.object(PeerClient, "probe_all", AsyncMock(return_value=[qualifying_probe])),
        patch.object(PeerClient, "dispatch", AsyncMock(return_value=None)),
    ):
        output = io.StringIO()
        count = await run_experiment(_config(), "requester", "unused", output)

    assert count == 1
    record = json.loads(output.getvalue().strip())
    assert record["dispatch_failed"] is True
    assert record["used_fallback"] is False
    assert record["selected_domain"] is None
    assert record["selected_node_id"] is None
    assert record["answer_text"] is None

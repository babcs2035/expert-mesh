"""Tests for run_experiment.py's per-row recording logic."""

import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import run_experiment
from run_experiment import run_experiment as run_experiment_rows
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
        count = await run_experiment_rows(_config(), "requester", "unused", output)

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
        count = await run_experiment_rows(_config(), "requester", "unused", output)

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
        count = await run_experiment_rows(_config(), "requester", "unused", output)

    assert count == 1
    record = json.loads(output.getvalue().strip())
    assert record["dispatch_failed"] is True
    assert record["used_fallback"] is False
    assert record["selected_domain"] is None
    assert record["selected_node_id"] is None
    assert record["answer_text"] is None


def test_main_touches_done_marker_only_after_output_is_written(monkeypatch, tmp_path) -> None:
    """main() writes `<output>.done` after the output file, not before or instead of it.

    `mise run start` launches this script via `docker compose exec -d` (see
    mise.toml's start task) so it never observes this process's own exit
    status; it polls for this marker instead. If the marker existed before
    the output file were fully written, that polling loop could copy a
    truncated result file.
    """
    monkeypatch.setattr(run_experiment, "load_yaml", lambda path: _config())
    monkeypatch.setattr(
        run_experiment,
        "OllamaClient",
        lambda: AsyncMock(
            embed=AsyncMock(return_value=[0.1]), generate=AsyncMock(return_value="hedge")
        ),
    )
    monkeypatch.setattr(
        run_experiment,
        "_read_dataset",
        lambda path: [{"id": "general-001", "query": "test", "expected_domains": ["general"]}],
    )
    output_path = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--node-id",
            "requester",
            "--dataset",
            "unused",
            "--output",
            str(output_path),
        ],
    )

    low_confidence = ProbeResponse(
        request_id="ignored", node_id="expert", confidence=0.1, estimated_latency_ms=10
    )
    with patch.object(PeerClient, "probe_all", AsyncMock(return_value=[low_confidence])):
        run_experiment.main()

    assert output_path.exists()
    assert json.loads(output_path.read_text().strip())["selected_domain"] == "general"
    assert Path(f"{output_path}.done").exists()


def test_record_experiment_provenance_copies_config_and_records_git_head(
    monkeypatch, tmp_path
) -> None:
    """The results directory gets a config.yaml snapshot and the build's git commit.

    Without this, comparing an old results.jsonl against the repo's current
    config.yaml cannot tell which commit's code actually produced it — the
    exact ambiguity that made the Iter22 deploy-drift incident (docs/d0002
    §6-C/§6-D) take a manual journal.md cross-reference to diagnose.
    """
    monkeypatch.setenv("GIT_HEAD", "30e3627020c986dfd24a3b0a4c0cdd26d1136b85")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("routing_method: supervised_classifier\n", encoding="utf-8")
    output_path = tmp_path / "run" / "results.jsonl"

    run_experiment._record_experiment_provenance(str(output_path), str(config_path))

    output_dir = output_path.parent
    assert (output_dir / "config.yaml").read_text(encoding="utf-8") == config_path.read_text(
        encoding="utf-8"
    )
    assert (
        output_dir / "git_head.txt"
    ).read_text().strip() == "30e3627020c986dfd24a3b0a4c0cdd26d1136b85"


def test_record_experiment_provenance_defaults_to_unknown_without_git_head_env(
    monkeypatch, tmp_path
) -> None:
    """GIT_HEAD is only set inside the Docker image (Dockerfile's ARG/ENV); outside
    it, the marker is explicit rather than silently missing or empty."""
    monkeypatch.delenv("GIT_HEAD", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("routing_method: self_report\n", encoding="utf-8")
    output_path = tmp_path / "run" / "results.jsonl"

    run_experiment._record_experiment_provenance(str(output_path), str(config_path))

    assert (output_path.parent / "git_head.txt").read_text().strip() == "unknown"

"""Tests for parsing and summarizing structured log lines in tools/show_logs.py."""

from tools.show_logs import parse_log_lines, summarize


def test_parse_log_lines_extracts_structured_records() -> None:
    """A docker-compose-prefixed [LEVEL] {json} line yields its parsed payload."""
    lines = [
        'expert-mesh-1  | [INFO] {"level": "INFO", "node_id": "n", "event": "probe_done"}\n',
    ]
    records = parse_log_lines(lines)
    assert records == [{"level": "INFO", "node_id": "n", "event": "probe_done"}]


def test_parse_log_lines_skips_uvicorn_access_log_lines() -> None:
    """Plain uvicorn access-log lines carry no JSON payload and are skipped."""
    lines = [
        'expert-mesh-1  | INFO:     Started server process [1]\n',
        'expert-mesh-1  | INFO:     192.168.1.1:1234 - "POST /probe HTTP/1.1" 200 OK\n',
    ]
    assert parse_log_lines(lines) == []


def test_parse_log_lines_skips_malformed_json() -> None:
    """A line that looks structured but has invalid JSON is skipped, not raised."""
    lines = ['expert-mesh-1  | [INFO] {not valid json}\n']
    assert parse_log_lines(lines) == []


def test_summarize_counts_events_and_averages_latency() -> None:
    """summarize groups by event type and averages local_inference_ms when present."""
    records = [
        {"event": "probe_done", "local_inference_ms": 100},
        {"event": "probe_done", "local_inference_ms": 300},
        {"event": "probe_timeout"},
    ]
    summary = summarize(records)
    assert summary["probe_done"]["count"] == 2
    assert summary["probe_done"]["mean_local_inference_ms"] == 200
    assert summary["probe_done"]["max_local_inference_ms"] == 300
    assert summary["probe_timeout"] == {"count": 1}

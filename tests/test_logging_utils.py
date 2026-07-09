"""Tests for the structured request-level logging in logging_utils.py."""

import json

from logging_utils import LOG_LEVEL_ERROR, LOG_LEVEL_INFO, log_event


def test_log_event_writes_json_line_prefixed_by_level(capsys) -> None:
    """Output follows the `[LEVEL] <json>` convention with the given fields."""
    log_event("node-a", LOG_LEVEL_INFO, "probe_done", request_id="r1", confidence=0.9)

    captured = capsys.readouterr()
    assert captured.out.startswith("[INFO] ")
    payload = json.loads(captured.out.removeprefix("[INFO] ").strip())
    assert payload["node_id"] == "node-a"
    assert payload["event"] == "probe_done"
    assert payload["request_id"] == "r1"
    assert payload["confidence"] == 0.9
    assert "unix_time_s" in payload


def test_log_event_supports_error_level(capsys) -> None:
    """The ERROR level is preserved verbatim in both the prefix and the JSON body."""
    log_event("node-a", LOG_LEVEL_ERROR, "probe_timeout", request_id="r1")

    captured = capsys.readouterr()
    assert captured.out.startswith("[ERROR] ")
    payload = json.loads(captured.out.removeprefix("[ERROR] ").strip())
    assert payload["level"] == "ERROR"

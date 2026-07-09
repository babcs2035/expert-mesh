"""Structured request-level logging for latency breakdown analysis.

Emits one JSON line per event to stdout, following distributed-llm's
`[LEVEL] message` convention (design doc 4.4) while adding machine-parsable
fields. Timestamps and phase durations are recorded separately so that
`mise run analyze` log collection can later separate network round-trip
time from local LLM inference time without re-instrumenting the caller.
"""

import json
import sys
import time

LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_ERROR = "ERROR"


def log_event(node_id: str, level: str, event: str, **fields: object) -> None:
    """Write a single structured log line to stdout.

    event identifies the log point (e.g. "probe_received", "dispatch_done");
    fields carries event-specific data such as request_id, duration_ms, or
    an error message. unix_time_s uses time.time() rather than
    time.monotonic() because, unlike the duration fields computed by
    callers, this value must be comparable across node processes when
    logs from multiple hosts are merged during analysis.
    """
    record = {
        "level": level,
        "node_id": node_id,
        "event": event,
        "unix_time_s": time.time(),
        **fields,
    }
    print(f"[{level}] {json.dumps(record, ensure_ascii=False)}", file=sys.stdout, flush=True)

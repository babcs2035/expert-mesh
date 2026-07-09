"""Parse and summarize the structured log events written by logging_utils.py.

`mise run analyze` collects raw `docker compose logs` output into
results/<datetime>/logs/<node_id>/expert-mesh.log; this tool reads those
files afterward and extracts the `[LEVEL] {json}` lines emitted by
logging_utils.log_event, skipping uvicorn's own access-log lines that
carry no such payload. It prints either a per-event listing or, with
--summary, aggregate counts and latency statistics per event type —
useful for spotting how much of the total latency (design doc 4.4) is
local inference versus everything else.

Usage:
    uv run python tools/show_logs.py --node-id wafl503
    uv run python tools/show_logs.py --node-id wafl503 --event probe_done
    uv run python tools/show_logs.py --node-id wafl503 --summary
"""

import argparse
import json
import re
import statistics
from pathlib import Path

DEFAULT_LOG_DIR = "logs"
# Matches docker compose's `<container>  | [LEVEL] <json>` line format; the
# container-name prefix varies by node, so it is captured but not used.
_LOG_LINE_PATTERN = re.compile(r"^\S+\s*\|\s*\[(\w+)\]\s*(\{.*\})\s*$")


def parse_log_lines(lines: list[str]) -> list[dict]:
    """Extract structured log_event records from raw docker compose log lines.

    Lines that don't match the `[LEVEL] {json}` pattern (uvicorn's own
    access-log output, startup/shutdown banners) are silently skipped.
    """
    records = []
    for line in lines:
        match = _LOG_LINE_PATTERN.match(line)
        if match is None:
            continue
        try:
            records.append(json.loads(match.group(2)))
        except json.JSONDecodeError:
            continue
    return records


def _read_log_file(log_dir: str, node_id: str) -> list[str]:
    """Read the collected log file for a single node."""
    path = Path(log_dir) / node_id / "expert-mesh.log"
    with path.open(encoding="utf-8") as f:
        return f.readlines()


def summarize(records: list[dict]) -> dict:
    """Aggregate event counts and local_inference_ms statistics per event type.

    Only events carrying local_inference_ms (probe_done, dispatch_done) get
    latency stats; heartbeat/error events just get a count.
    """
    by_event: dict[str, list[dict]] = {}
    for record in records:
        by_event.setdefault(record["event"], []).append(record)

    summary = {}
    for event, event_records in sorted(by_event.items()):
        entry = {"count": len(event_records)}
        durations = [r["local_inference_ms"] for r in event_records if "local_inference_ms" in r]
        if durations:
            entry["mean_local_inference_ms"] = statistics.mean(durations)
            entry["max_local_inference_ms"] = max(durations)
        summary[event] = entry
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Parse and summarize structured log events collected by mise run analyze"
    )
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--event", default=None, help="Only show records of this event type")
    parser.add_argument(
        "--summary", action="store_true", help="Print aggregate counts and latency stats"
    )
    args = parser.parse_args()

    lines = _read_log_file(args.log_dir, args.node_id)
    records = parse_log_lines(lines)
    if args.event is not None:
        records = [r for r in records if r["event"] == args.event]

    if args.summary:
        for event, stats in summarize(records).items():
            print(f"{event}: {stats}")
    else:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()

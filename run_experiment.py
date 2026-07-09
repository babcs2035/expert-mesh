"""Run every question in a dataset through the requester flow and record the outcome.

Reuses node.run_ask_flow (the same probe -> select -> dispatch/fallback logic
exercised by `node.py ask`) so that benchmark results reflect actual runtime
behavior rather than a re-implementation. Requires a live mesh (config.yaml's
nodes reachable, ollama warmed up) since it makes real /probe and /dispatch
HTTP calls; there is no mocked mode, matching design doc 4.4's requirement
that experiments run against the deployed nodes.

Usage:
    uv run python run_experiment.py --node-id wafl500 \\
        --dataset data/dataset.jsonl --output results.jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TextIO

# This file is at the project root alongside node.py and expert_backend.py.
# No sys.path manipulation needed — pytest's pythonpath = ["."] handles it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expert_backend import OllamaClient  # noqa: E402 (see sys.path setup above)
from node import load_yaml, run_ask_flow  # noqa: E402


def _read_dataset(path: str) -> list[dict]:
    """Load dataset rows written by build_dataset.py."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _run_one(config: dict, node_id: str, row: dict, ollama_client: OllamaClient) -> dict:
    """Run a single dataset row through the requester flow and record the outcome.

    Records wall-clock duration (network + local inference combined) so that
    metrics.py can report latency alongside routing correctness; the finer
    network-vs-inference split lives in each node's own structured logs
    (logging_utils.log_event), not in this client-side measurement.
    """
    start = time.monotonic()
    result = await run_ask_flow(config, node_id, row["query"], ollama_client)
    duration_ms = int((time.monotonic() - start) * 1000)

    # AskResult has three distinct outcomes (node.py's _ask prints a
    # different message for each): a successful dispatch, a fallback
    # answer when no peer qualified, or neither when qualifying peers were
    # found but every /dispatch call to them failed (e.g. timed out).
    # selected_domain/selected_node_id must stay None in the third case —
    # setting them to the requester's own domain there would misrepresent
    # a failed dispatch as an answered "general" question in metrics.py.
    if result.dispatch_response is not None:
        selected_domain = config["nodes"][result.dispatch_response.node_id]["domain"]
        answer_text = result.dispatch_response.answer_text
        selected_node_id = result.dispatch_response.node_id
        confidence = result.dispatch_response.confidence
    elif result.fallback_answer is not None:
        selected_domain = config["nodes"][node_id]["domain"]
        answer_text = result.fallback_answer
        selected_node_id = node_id
        confidence = None
    else:
        selected_domain = None
        answer_text = None
        selected_node_id = None
        confidence = None

    return {
        "id": row["id"],
        "query": row["query"],
        "expected_domains": row["expected_domains"],
        "selected_node_id": selected_node_id,
        "selected_domain": selected_domain,
        "used_fallback": result.fallback_answer is not None,
        "dispatch_failed": result.dispatch_response is None and result.fallback_answer is None,
        "confidence": confidence,
        "answer_text": answer_text,
        "duration_ms": duration_ms,
    }


async def run_experiment(config: dict, node_id: str, dataset_path: str, output: TextIO) -> int:
    """Run every dataset row sequentially and write one JSON result line per row.

    Sequential (not concurrent) execution mirrors how a single requester
    node would actually be used, and avoids overlapping /probe calls from
    contending for the same node's CPU-bound local inference (design doc
    2.1: CPU-only laptops, no GPU).
    """
    rows = _read_dataset(dataset_path)
    ollama_client = OllamaClient()
    for row in rows:
        record = await _run_one(config, node_id, row, ollama_client)
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        print(f"[run_experiment] {record['id']}: -> {record['selected_domain']}", file=sys.stderr)
    return len(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark dataset through the requester flow"
    )
    parser.add_argument("--node-id", required=True, help="Requester node id from config.yaml")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True, help="JSONL dataset from build_dataset.py")
    parser.add_argument("--output", default=None, help="Output JSONL path; defaults to stdout")
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.output is None:
        count = asyncio.run(run_experiment(config, args.node_id, args.dataset, sys.stdout))
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            count = asyncio.run(run_experiment(config, args.node_id, args.dataset, f))
        # Written only after the output file is closed (all rows flushed to
        # disk), so `mise run start`'s polling loop never observes the
        # marker before the results it is about to copy are complete. The
        # marker (not the process exit code) is what that loop waits on,
        # since it launches this script via `docker compose exec -d` and so
        # never sees this process's own exit status.
        Path(f"{args.output}.done").touch()
    print(f"[run_experiment] completed {count} questions", file=sys.stderr)


if __name__ == "__main__":
    main()

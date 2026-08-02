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
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TextIO

# This file is at the project root alongside node.py and expert_backend.py.
# No sys.path manipulation needed — pytest's pythonpath = ["."] handles it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregator import select_dispatch_targets  # noqa: E402
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
        dispatch_gen_time_ms = result.dispatch_response.gen_time_ms
    elif result.fallback_answer is not None:
        selected_domain = config["nodes"][node_id]["domain"]
        answer_text = result.fallback_answer
        selected_node_id = node_id
        confidence = None
        dispatch_gen_time_ms = None
    else:
        selected_domain = None
        answer_text = None
        selected_node_id = None
        confidence = None
        dispatch_gen_time_ms = None

    # Recompute the dispatch target set from the probe_responses already
    # fetched above (no extra /probe or /dispatch calls) using the same
    # aggregator function and config values run_ask_flow used internally,
    # so this reproduces the actual candidate/dispatch sets for set-valued
    # coverage metrics (metrics.py's compute_compound_coverage_metrics)
    # without changing routing/aggregation behavior itself. Empty when no
    # node cleared confidence_threshold (the fallback case).
    dispatch_targets = select_dispatch_targets(
        result.probe_responses,
        confidence_threshold=config.get("confidence_threshold", 0.5),
        top_k=config.get("dispatch_top_k", 1),
        dispatch_candidate_threshold=config.get("dispatch_candidate_threshold"),
    )
    dispatched_domains = [config["nodes"][t.node_id]["domain"] for t in dispatch_targets]
    probe_candidates = [
        {
            "node_id": r.node_id,
            "domain": config["nodes"][r.node_id]["domain"],
            "confidence": r.confidence,
            "confidence_logprobs_mean": r.confidence_logprobs_mean,
        }
        for r in result.probe_responses
    ]

    # Extract STP logprobs signal from the selected node's probe response.
    stp_logprobs: float | None = None
    if result.probe_responses and selected_node_id is not None:
        for pr in result.probe_responses:
            if pr.node_id == selected_node_id:
                stp_logprobs = pr.confidence_logprobs_mean
                break

    return {
        "id": row["id"],
        "request_id": result.request_id,
        "query": row["query"],
        "expected_domains": row["expected_domains"],
        "selected_node_id": selected_node_id,
        "selected_domain": selected_domain,
        "used_fallback": result.fallback_answer is not None,
        "dispatch_failed": result.dispatch_response is None and result.fallback_answer is None,
        "confidence": confidence,
        "confidence_logprobs_mean": stp_logprobs,
        "answer_text": answer_text,
        "duration_ms": duration_ms,
        # dispatch_gen_time_ms is the expert node's own local generation time
        # (DispatchResponse.gen_time_ms); duration_ms - dispatch_gen_time_ms
        # approximates network + probe-round overhead (design doc 4.4's
        # "latency breakdown"). None when there was no successful dispatch
        # (fallback or dispatch_failed), since there is no expert generation
        # time to report in that case.
        "dispatch_gen_time_ms": dispatch_gen_time_ms,
        "dispatched_domains": dispatched_domains,
        "probe_candidates": probe_candidates,
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


def _record_experiment_provenance(output_path: str, config_path: str) -> None:
    """Copy the active config.yaml and the git commit into the results directory.

    Without this, a stale-deploy incident where the running container's
    code doesn't match what journal.md assumes (Iter22, docs/d0002
    §6-C/§6-D) can only be reconstructed after the fact by cross-referencing
    journal.md's commit hashes against dates. GIT_HEAD is baked into the
    Docker image at build time (Dockerfile's ARG/ENV, set from mise.toml's
    setup task) since .git itself is not copied into the image, so it
    reflects the commit the running container was actually built from —
    not whatever HEAD happens to be at read time.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.yaml")
    git_head = os.environ.get("GIT_HEAD", "unknown")
    (output_dir / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")


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
        _record_experiment_provenance(args.output, args.config)
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

"""Pre-experiment smoke check: catch deploy/config drift before a costly run starts.

Guards against the two known failure modes from journal.md (Iter12, Iter22): a
code change that was committed but never rebuilt into the Docker image, and a
change that was applied to the working tree but never committed (so `mise run
deploy`, which distributes from git HEAD, never shipped it). Both wasted
~2 hours of experiment time because the drift was only noticed after the run
completed. Intended to run after `mise run deploy` and before `mise run start`
(see mise.toml's deploy task, which invokes all three checks below).

Three independent checks, run via --check (each exits 1 on failure):

  git-status  Host-side. Warns (does not fail the pipeline) if the working
              tree has uncommitted changes, since the Docker image that
              backs every node's `app` container is built from git HEAD.
  hashes      Host-side. Compares the md5 of http_server.py/router.py/
              config.yaml on disk against the copy inside each node's
              running container, over SSH.
  probe       Container-side only (Ollama is bound to 127.0.0.1:11434 on
              each remote host, unreachable from this workstation). Sends
              one /probe to a peer and checks that the field implied by the
              active confidence_signal_method/routing_method is populated.

Usage:
    uv run python tools/smoke_check.py --check git-status
    uv run python tools/smoke_check.py --check hashes --remote-dir ~/workspace/ktakahashi/expert-mesh
    # inside the app container (NODE_ID env var set by docker-compose):
    uv run python tools/smoke_check.py --check probe
"""

import argparse
import asyncio
import hashlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import yaml

# This file is at tools/, one level below the project root that holds
# expert_backend.py; add the root so the import below resolves both under
# pytest (pythonpath=["."] already covers this) and via `python
# tools/smoke_check.py` (where only tools/ itself would otherwise be on
# sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_backend import OllamaClient  # noqa: E402

SMOKE_QUERY = "スモークテスト用のダミー質問です。"
SMOKE_PROBE_TIMEOUT_S = 130.0
DEPLOYED_FILES = ["http_server.py", "router.py", "config.yaml"]
SELF_HOST_OVERRIDE = "localhost"  # avoids hairpin-NAT issues probing one's own LAN IP


def check_git_status() -> bool:
    """Warn (do not fail) if the working tree has uncommitted changes.

    Application code ships baked into the Docker image built from git HEAD
    (mise.toml's setup task), so an uncommitted code change silently fails
    to deploy even though `mise run deploy` itself succeeds.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    if result.stdout.strip():
        print("[smoke_check] WARNING: working tree has uncommitted changes:")
        print(result.stdout)
        print(
            "[smoke_check] WARNING: the Docker image is built from git HEAD; "
            "uncommitted code changes will NOT be reflected on the deployed nodes."
        )
    else:
        print("[smoke_check] OK: working tree is clean")
    return True


def _local_md5(path: str) -> str:
    """Return the md5 hex digest of a local file."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _remote_md5(host: str, remote_dir: str, filename: str) -> str | None:
    """Return the md5 hex digest of a file inside the node's running container.

    Returns None if the file could not be read (container down, path
    mismatch, etc.) so the caller can report a clear failure instead of
    crashing on a subprocess error.
    """
    result = subprocess.run(
        ["ssh", host, f"cd {remote_dir} && docker compose exec -T app md5sum /app/{filename}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def check_deployed_hashes(remote_dir: str, hosts: list[str]) -> bool:
    """Compare local file hashes against each node's deployed container copy.

    Returns True only if every host's every tracked file matches the local
    copy exactly.
    """
    local_hashes = {f: _local_md5(f) for f in DEPLOYED_FILES}
    all_ok = True
    for host in hosts:
        for filename, local_hash in local_hashes.items():
            remote_hash = _remote_md5(host, remote_dir, filename)
            if remote_hash is None:
                print(f"[smoke_check] FAIL: {host}: could not read {filename} from container")
                all_ok = False
            elif remote_hash != local_hash:
                print(
                    f"[smoke_check] FAIL: {host}: {filename} hash mismatch "
                    f"(local={local_hash[:12]}, deployed={remote_hash[:12]})"
                )
                all_ok = False
            else:
                print(f"[smoke_check] OK: {host}: {filename} matches deployed container")
    return all_ok


# confidence_signal_method -> response field expected to be populated when
# that method is active. Keys match http_server.py's CONFIDENCE_SIGNAL_*
# constant values exactly (e.g. CONFIDENCE_SIGNAL_SEMANTIC_ENTROPY's actual
# string value is "self_consistency_semantic", not "semantic_entropy" —
# only the Python constant is named that). routing_method=supervised_classifier
# is checked separately below since it affects estimated_latency_ms's order
# of magnitude rather than a dedicated response field (see http_server.py's
# _estimate_probe_confidence: the classifier path makes no LLM call at all).
_SIGNAL_FIELD_EXPECTATIONS = {
    "stp": "confidence_logprobs_mean",
    "self_consistency_semantic": "confidence_semantic_entropy",
    "p_true": "confidence_p_true",
}


async def run_probe_smoke_test(config: dict) -> bool:
    """Send one /probe to a peer and verify the active method's code path actually ran.

    Must run inside a node's app container: Ollama is bound to
    127.0.0.1:11434 on the remote host (docker-compose.yml), so embedding
    computation is only reachable from inside that host's containers.

    This is the check that would have caught both Iter12 (STP fields staying
    null because the old image was still running) and Iter22 (semantic
    entropy fields staying null because the bug-fix commit never got
    deployed): in both cases routing/answer metrics looked normal, only the
    per-method diagnostic field silently stayed empty.
    """
    self_node_id = os.environ.get("NODE_ID")
    peers = {
        node_id: ({**node_cfg, "host": SELF_HOST_OVERRIDE} if node_id == self_node_id else node_cfg)
        for node_id, node_cfg in config["nodes"].items()
    }
    # Probe a peer other than self when possible so this doesn't just test
    # the requester's own /probe handler in isolation.
    target_node_id = next((nid for nid in peers if nid != self_node_id), next(iter(peers)))
    target_peer = peers[target_node_id]

    embedding_model = config["embedding_model"]
    routing_method = config.get("routing_method", "self_report")
    signal_method = config.get("confidence_signal_method", "self_report")

    ollama_client = OllamaClient()
    query_embedding = await ollama_client.embed(embedding_model, SMOKE_QUERY)

    request_body = {
        "request_id": f"smoke-{uuid.uuid4().hex[:8]}",
        "query_summary": SMOKE_QUERY,
        "query_embedding": query_embedding,
        "from": self_node_id or "smoke_check",
    }
    url = f"http://{target_peer['host']}:{target_peer['port']}/probe"
    async with httpx.AsyncClient(timeout=SMOKE_PROBE_TIMEOUT_S) as client:
        response = await client.post(url, json=request_body)
        response.raise_for_status()
        probe_result = response.json()

    print(f"[smoke_check] probe response from {target_node_id}: {probe_result}")

    if routing_method == "supervised_classifier":
        latency_ms = probe_result["estimated_latency_ms"]
        if latency_ms > 1000:
            print(
                f"[smoke_check] FAIL: routing_method=supervised_classifier expects a "
                f"no-LLM-call probe (few ms), but estimated_latency_ms={latency_ms}. "
                f"This usually means the classifier branch in http_server.py was not reached "
                f"(config drift, or a confidence_signal_method that short-circuits before it — "
                f"see docs/d0002_research_cycle_findings_2026-07.md §6-B/6-D)."
            )
            return False
        print(f"[smoke_check] OK: supervised_classifier probe latency={latency_ms}ms (no LLM call)")
        return True

    expected_field = _SIGNAL_FIELD_EXPECTATIONS.get(signal_method)
    if expected_field is None:
        print(f"[smoke_check] OK: confidence_signal_method={signal_method} has no dedicated field to check")
        return True

    if probe_result.get(expected_field) is None:
        print(
            f"[smoke_check] FAIL: confidence_signal_method={signal_method} expects "
            f"'{expected_field}' to be non-null, but it is null. The configured method's "
            f"code path was not reached (stale deploy or branch-order bug)."
        )
        return False
    print(f"[smoke_check] OK: {expected_field}={probe_result[expected_field]}")
    return True


def main() -> None:
    """CLI entry point: exit 0 if the requested check passed, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Pre-experiment smoke check")
    parser.add_argument("--check", required=True, choices=["git-status", "hashes", "probe"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--remote-dir", help="Required for --check hashes")
    args = parser.parse_args()

    if args.check == "git-status":
        ok = check_git_status()
    elif args.check == "hashes":
        if not args.remote_dir:
            parser.error("--remote-dir is required for --check hashes")
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ok = check_deployed_hashes(args.remote_dir, list(config["nodes"].keys()))
    else:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ok = asyncio.run(run_probe_smoke_test(config))

    if not ok:
        print(f"[smoke_check] FAILED ({args.check}): fix the above before starting the experiment.")
        sys.exit(1)
    print(f"[smoke_check] {args.check} passed")


if __name__ == "__main__":
    main()

"""Run a benchmark dataset through a central supervised classifier + answer generation flow.

Unlike run_experiment.py which exercises the full distributed mesh
(probe all peers -> aggregate -> dispatch), this script implements the
centralized routing architecture described in design doc 4.2(b):

  1. Load the same multi-class classifier (models/domain_classifier.joblib)
     used by every node in the distributed mesh.
  2. For each question:
     a. Generate a query embedding via SSH + curl to an expert node's Ollama.
     b. Classify the embedding to select the single best domain (argmax
        over all domain probabilities).
     c. Generate an answer on the expert node for that domain via SSH + curl.
     d. Write one JSON line per question to results.jsonl.

Both embedding and answer generation run on expert nodes via SSH + curl
(Ollama is bound to 127.0.0.1 inside Docker on each node).  The central
host only runs the classifier and orchestrates the flow.

SSH credentials and domain-to-node mapping are read from
config.yaml's `central_router` section (keys: `ssh_user`, `embed_node_host`,
`domain_nodes`).

This produces results.jsonl in the same schema as run_experiment.py so that
metrics.py can analyze both distributed and central results identically.

Usage:
    uv run python scripts/run_central_experiment.py \\
        --config config.yaml \\
        --dataset data/dataset.jsonl \\
        --output results/central/results.jsonl
"""

import argparse
import asyncio
import json
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import TextIO

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier import load_domain_classifier  # noqa: E402
from http_server import DISPATCH_MAX_TOKENS, build_dispatch_prompt  # noqa: E402
from node import FALLBACK_MAX_TOKENS, FALLBACK_PROMPT_TEMPLATE  # noqa: E402


# ---------------------------------------------------------------------------
# SSH clients for remote Ollama access (embedding + answer generation)
# ---------------------------------------------------------------------------
# Ollama runs inside Docker on each expert node, bound to 127.0.0.1:11434.
# Both embedding and answer generation use `ssh <user>@<node> curl ...`
# to invoke the Ollama HTTP API remotely.
#
# The target user and node mappings are read from config.yaml's
# `central_router` section (keys: `ssh_user`, `domain_nodes`).
# Embeddings are fetched from the first expert node (wafl500/general)
# since nomic-embed-text is available on all nodes.


class SshEmbeddingClient:
    """Fetch embeddings via SSH + curl to an expert node's Ollama API."""

    def __init__(self, ssh_user: str, embed_node_host: str) -> None:
        self._ssh_user = ssh_user
        self._embed_node_host = embed_node_host

    async def embed(self, model: str, text: str) -> list[float]:
        """Return a 768-D embedding vector for the given text.

        Uses /api/embeddings with a `prompt` field (not the newer /api/embed
        batch endpoint with `input`), matching expert_backend.OllamaClient.embed()
        exactly. Iter24 used /api/embed here, a different Ollama endpoint than
        the distributed mesh's OllamaClient.embed() — a likely cause of the
        routing mismatch found in Iter24 (backlog B44), independent of the
        answer-generation prompt bug.
        """
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        # Use stdin piping to avoid SSH quoting issues entirely.
        cmd = (
            f"ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "
            f"{shlex.quote(self._ssh_user)}@{shlex.quote(self._embed_node_host)} "
            f"curl -s --max-time 60 "
            f"http://127.0.0.1:11434/api/embeddings -d @-"
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload), timeout=60
            )
            if proc.returncode != 0:
                print(
                    f"[SshEmbeddingClient] embed failed (rc={proc.returncode}): "
                    f"{stderr.decode(errors='replace')[:200]}",
                    file=sys.stderr,
                )
                return [0.0] * 768
            resp = json.loads(stdout.decode("utf-8"))
            emb = resp.get("embedding", [])
            return [float(v) for v in emb]
        except asyncio.TimeoutError:
            proc.kill()
            print("[SshEmbeddingClient] embed timed out", file=sys.stderr)
            return [0.0] * 768
        except json.JSONDecodeError:
            print("[SshEmbeddingClient] invalid JSON from embed node", file=sys.stderr)
            return [0.0] * 768
        except Exception as exc:
            print(f"[SshEmbeddingClient] embed error: {exc}", file=sys.stderr)
            return [0.0] * 768


class HttpOllamaGenerator:
    """Generate answers on expert nodes via SSH + curl to Ollama HTTP API."""

    def __init__(self, ssh_user: str) -> None:
        self._ssh_user = ssh_user

    async def generate(
        self,
        node_host: str,
        model: str,
        prompt: str,
        *,
        max_tokens: int,
        timeout_s: int = 120,
    ) -> str | None:
        """Call Ollama on the given node and return the answer text.

        Uses /api/chat with a messages list (not /api/generate with a bare
        prompt), matching expert_backend.OllamaClient.generate() exactly —
        Iter24 used /api/generate here, which is a different Ollama endpoint
        than the distributed mesh's answer-generation path (backlog B44).
        """
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")
        cmd = (
            f"ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "
            f"{shlex.quote(self._ssh_user)}@{shlex.quote(node_host)} "
            f"curl -s --max-time {timeout_s} "
            f"http://127.0.0.1:11434/api/chat -d @-"
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=payload), timeout=timeout_s
            )
            if proc.returncode != 0:
                print(
                    f"[HttpOllamaGenerator] curl/ssh failed (rc={proc.returncode}): "
                    f"{stderr.decode(errors='replace')[:200]}",
                    file=sys.stderr,
                )
                return None
            resp = json.loads(stdout.decode("utf-8"))
            return (resp.get("message", {}).get("content") or "").strip() or None
        except asyncio.TimeoutError:
            proc.kill()
            print(f"[HttpOllamaGenerator] generate timed out (model={model})", file=sys.stderr)
            return None
        except json.JSONDecodeError:
            print(
                f"[HttpOllamaGenerator] invalid JSON response from {node_host}",
                file=sys.stderr,
            )
            return None
        except Exception as exc:
            print(f"[HttpOllamaGenerator] generate error: {exc}", file=sys.stderr)
            return None


def _read_dataset(path: str) -> list[dict]:
    """Load dataset rows written by build_dataset.py."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]



async def _run_one(
    classifier,
    domain_nodes: dict[str, dict],
    embedding_model: str,
    row: dict,
    embed_client: SshEmbeddingClient,
    generator: HttpOllamaGenerator,
    confidence_threshold: float,
    fallback_node_host: str,
    fallback_model: str,
) -> dict:
    """Run a single dataset row through central routing and answer generation.

    Central flow (no probe/dispatch):
      1. Generate embedding for the query (SSH + curl to expert node Ollama).
      2. Classify to pick the best domain (argmax over all probabilities).
      3a. If the argmax domain's own probability clears confidence_threshold,
          generate an answer on that expert node for that domain.
      3b. Otherwise, fall back to fallback_model on fallback_node_host —
          mirroring node.py's run_ask_flow, which routes to the requester's
          own light_model rather than dispatching when no candidate clears
          the threshold (design doc 2.5). Iter26 originally always dispatched
          via a bare argmax with no threshold, which produced a routing
          accuracy difference vs the distributed mesh that was NOT an
          architecture effect: all 77 McNemar-discordant rows in that run
          were exactly the distributed mesh's fallback rows (backlog B46).
          Applying the same threshold+fallback policy here isolates the
          single lever X2 is meant to test (architecture only).

    The result record uses the same schema as run_experiment.py so that
    metrics.py can process both outputs identically.
    """
    start = time.monotonic()

    # Step 1: Generate query embedding (SSH + curl to expert node Ollama).
    query_embedding = await embed_client.embed(
        model=embedding_model, text=row["query"]
    )

    # Step 2: Classify — predict_proba returns one probability per domain.
    probabilities = classifier.predict_proba([query_embedding])[0]
    classes = list(classifier.classes_)

    # argmax: select the domain with the highest predicted probability.
    max_idx = int(probabilities.argmax())
    argmax_domain = classes[max_idx]
    argmax_confidence = float(probabilities[max_idx])

    if argmax_confidence >= confidence_threshold:
        # Step 3a: Generate answer on the expert node for the selected domain.
        # Uses the same build_dispatch_prompt() the distributed mesh's /dispatch
        # endpoint uses (http_server.py), not the raw query — Iter24 passed
        # row["query"] directly here, which was the proximate cause of the
        # 9-character truncated answers that made Iter24's answer_quality_accuracy
        # an artifact (backlog B44).
        selected_domain = argmax_domain
        confidence = argmax_confidence
        used_fallback = False
        node_info = domain_nodes.get(selected_domain)
        if node_info is None:
            answer_text = None
        else:
            answer_text = await generator.generate(
                node_host=node_info["host"],
                model=node_info["model"],
                prompt=build_dispatch_prompt(selected_domain, row["query"]),
                max_tokens=DISPATCH_MAX_TOKENS,
                timeout_s=120,
            )
    else:
        # Step 3b: fall back, matching node.py's _fallback_answer exactly
        # (same prompt template, same light_model, no per-node confidence).
        used_fallback = True
        confidence = None
        answer_text = await generator.generate(
            node_host=fallback_node_host,
            model=fallback_model,
            prompt=FALLBACK_PROMPT_TEMPLATE.format(query=row["query"]),
            max_tokens=FALLBACK_MAX_TOKENS,
            timeout_s=600,
        )
        # run_experiment.py's fallback case sets selected_domain to the
        # requester's own domain (here: the domain owning fallback_node_host).
        selected_domain = next(
            domain for domain, info in domain_nodes.items() if info["host"] == fallback_node_host
        )

    duration_ms = int((time.monotonic() - start) * 1000)

    # Build the result record with the same schema as run_experiment.py.
    #
    # Fields that reflect the distributed mesh architecture (selected_node_id,
    # probe_candidates, dispatched_domains) are set to their central-equivalent
    # values: None or the selected domain itself, since there is no actual
    # node selection or dispatch in the central flow.
    #
    # confidence_logprobs_mean is null because the supervised classifier
    # does not produce token-level confidence signals.
    return {
        "id": row["id"],
        "request_id": str(uuid.uuid4()),
        "query": row["query"],
        "expected_domains": row["expected_domains"],
        "selected_node_id": None,
        "selected_domain": selected_domain,
        "used_fallback": used_fallback,
        "dispatch_failed": False,
        "confidence": confidence,
        "confidence_logprobs_mean": None,
        "answer_text": answer_text,
        "duration_ms": duration_ms,
        # dispatch_gen_time_ms is the expert node's own local generation time;
        # None here because there is no dispatch step in the central flow.
        "dispatch_gen_time_ms": None,
        "dispatched_domains": [selected_domain],
        "probe_candidates": [],
    }


async def run_experiment(
    classifier,
    domain_nodes: dict[str, dict],
    embedding_model: str,
    dataset_path: str,
    output: TextIO,
    ssh_user: str,
    embed_node_host: str,
    confidence_threshold: float,
    fallback_node_host: str,
    fallback_model: str,
) -> int:
    """Run every dataset row sequentially and write one JSON result line per row.

    Sequential execution mirrors how a single requester node would actually
    be used, and avoids overlapping /probe calls from contending for the
    same node's CPU-bound local inference (design doc 2.1).

    Args:
        ssh_user: SSH user for remote answer generation.
        embed_node_host: Host for embedding (one of the expert nodes).
        confidence_threshold: Below this, fall back instead of dispatching
          to the argmax domain (matches config.yaml's confidence_threshold).
        fallback_node_host: Host used for fallback generation (matches
          run_experiment.py's default requester node: the first configured
          peer, i.e. domain_nodes["general"]'s host).
        fallback_model: light_model used for fallback generation.
    """
    rows = _read_dataset(dataset_path)
    embed_client = SshEmbeddingClient(ssh_user, embed_node_host)
    generator = HttpOllamaGenerator(ssh_user)
    for row in rows:
        record = await _run_one(
            classifier,
            domain_nodes,
            embedding_model,
            row,
            embed_client,
            generator,
            confidence_threshold,
            fallback_node_host,
            fallback_model,
        )
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        print(
            f"[run_central_experiment] {record['id']}: -> {record['selected_domain']}",
            file=sys.stderr,
        )
        # Delay between questions to avoid overwhelming SSH daemons.
        await asyncio.sleep(2.0)
    return len(rows)


def _record_experiment_provenance(output_path: str, config_path: str) -> None:
    """Copy the active config.yaml and the git commit into the results directory.

    See run_experiment.py for the full docstring rationale — identical
    provenance concerns apply to the central experiment.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil  # noqa: F811

    shutil.copy(config_path, output_dir / "config.yaml")
    git_head = "unknown"
    try:
        import subprocess  # noqa: F401

        git_head = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    (output_dir / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark dataset through the central supervised classifier"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="JSONL dataset from build_dataset.py",
    )
    parser.add_argument(
        "--classifier",
        default="models/domain_classifier.joblib",
        help="Path to the trained classifier (joblib)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path; defaults to stdout",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    classifier = load_domain_classifier(args.classifier)
    # domain_nodes maps domain -> {host, model} for answer generation.
    domain_nodes = config.get("central_router", {}).get("domain_nodes", {})
    if not domain_nodes:
        print(
            "[run_central_experiment] error: central_router.domain_nodes is "
            "required in config.yaml",
            file=sys.stderr,
        )
        sys.exit(1)
    embedding_model = config.get("embedding_model", "nomic-embed-text")

    # SSH user and embed node host (read from config).
    cr = config.get("central_router", {})
    ssh_user = cr.get("ssh_user")
    embed_node_host = cr.get("embed_node_host")
    if ssh_user is None or embed_node_host is None:
        print(
            "[run_central_experiment] error: central_router.ssh_user and "
            "central_router.embed_node_host are required in config.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fallback (design doc 2.5): mirrors run_experiment.py's default requester
    # (the first configured peer, domain_nodes["general"]'s host) so both
    # architectures apply the identical confidence_threshold + fallback
    # policy — see _run_one's docstring / backlog B46 for why this matters.
    confidence_threshold = config.get("confidence_threshold", 0.5)
    general_node = domain_nodes.get("general")
    if general_node is None:
        print(
            "[run_central_experiment] error: central_router.domain_nodes must "
            "include a 'general' entry for fallback generation",
            file=sys.stderr,
        )
        sys.exit(1)
    fallback_node_host = general_node["host"]
    fallback_model = next(
        (
            node_config["light_model"]
            for node_config in config["nodes"].values()
            if node_config["host"] == fallback_node_host
        ),
        None,
    )
    if fallback_model is None:
        print(
            f"[run_central_experiment] error: no config.yaml nodes entry with "
            f"host={fallback_node_host!r} to read light_model from",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.output is None:
        count = asyncio.run(
            run_experiment(
                classifier,
                domain_nodes,
                embedding_model,
                args.dataset,
                sys.stdout,
                ssh_user=ssh_user,
                embed_node_host=embed_node_host,
                confidence_threshold=confidence_threshold,
                fallback_node_host=fallback_node_host,
                fallback_model=fallback_model,
            )
        )
    else:
        _record_experiment_provenance(args.output, args.config)
        with open(args.output, "w", encoding="utf-8") as f:
            count = asyncio.run(
                run_experiment(
                    classifier,
                    domain_nodes,
                    embedding_model,
                    args.dataset,
                    f,
                    ssh_user=ssh_user,
                    confidence_threshold=confidence_threshold,
                    fallback_node_host=fallback_node_host,
                    fallback_model=fallback_model,
                    embed_node_host=embed_node_host,
                )
            )
        Path(f"{args.output}.done").touch()
    print(
        f"[run_central_experiment] completed {count} questions", file=sys.stderr
    )


if __name__ == "__main__":
    main()

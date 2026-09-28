"""Iter95 helper (run on wafl-ctrl5 only, absolute condition (B)): embed every row of a
classifier-training JSONL once and cache the result as .npy + sidecar metadata, so the
Iter95 pre-screening (scripts/screen_classifier_training_volume_expansion.py) and the
final `scripts/train_domain_classifier.py` both compute embeddings from the exact same
feature pipeline (embed_query_views with instruction + concat_views, matching node.py's
runtime feature spec) without re-embedding for every screening fold/seed.

Not a production artifact: it exists only to avoid ~9,000 redundant `/api/embeddings`
calls across the screening's 3 seeds x 5 folds x 2 arms. `train_domain_classifier.py`
remains the sole source of truth for the final `models/domain_classifier.joblib`.

Usage (on wafl-ctrl5, pointing at its own local Ollama):
    uv run python scripts/embed_classifier_train_pool.py \\
        --train-data data/classifier_train_iter95_fullpool.jsonl \\
        --embedding-model qwen3-embedding:4b \\
        --ollama-host 127.0.0.1 \\
        --embedding-instruction "..." \\
        --embedding-view-concat \\
        --output data/embcache_iter95_fullpool.npy
"""

import argparse
import asyncio
import json
import sys

import numpy as np

from expert_backend import OllamaClient, embed_query_views


async def _embed_all(
    rows: list[dict], embedding_model: str, ollama_host: str, ollama_port: int,
    instruction: str | None, concat_views: bool,
) -> list[list[float]]:
    """Sequentially embed every row's query (matches train_domain_classifier.py's
    build_training_features, which is itself sequential to mirror run_experiment.py)."""
    client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings = []
    for i, row in enumerate(rows):
        embeddings.append(
            await embed_query_views(
                client, embedding_model, row["query"],
                instruction=instruction, concat_views=concat_views,
            )
        )
        if (i + 1) % 200 == 0:
            print(f"[embed_classifier_train_pool] {i + 1}/{len(rows)}", file=sys.stderr)
    return embeddings


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cache embeddings for a classifier training JSONL")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--ollama-host", required=True)
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--embedding-instruction", default=None)
    parser.add_argument("--embedding-view-concat", action="store_true")
    parser.add_argument("--output", required=True, help="Output .npy path")
    args = parser.parse_args()

    with open(args.train_data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    embeddings = asyncio.run(
        _embed_all(
            rows, args.embedding_model, args.ollama_host, args.ollama_port,
            args.embedding_instruction, args.embedding_view_concat,
        )
    )
    np.save(args.output, np.array(embeddings, dtype=np.float64))

    meta_path = args.output + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"ids": [r["id"] for r in rows], "domains": [r["domain"] for r in rows]},
            f, ensure_ascii=False,
        )
    print(
        f"[embed_classifier_train_pool] wrote {args.output} "
        f"(shape={np.array(embeddings).shape}) and {meta_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

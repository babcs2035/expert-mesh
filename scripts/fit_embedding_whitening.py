"""E7: fit a mean-centering/whitening transform for embedding-based routing (method A).

Offline-only: fetches background embeddings from a live ollama node for
every query in a dataset, fits mu (and, in whiten mode, a whitening matrix
via BERT-whitening/Su et al. 2021 arXiv:2103.15316), and writes the result
as plain JSON so router.py's serving path never needs numpy. Addresses the
embedding anisotropy that collapsed Iter2's cosine similarities into a
narrow band (Ethayarajh 2019) — this is a diagnostic fix, not a claim that
whitening alone resolves survey's broader "similarity-based routing is
unsupervised and therefore fragile" critique.

Usage (module mode; requires the "research" extra: uv sync --extra research):
    uv run python -m scripts.fit_embedding_whitening \\
        --dataset data/dataset.jsonl --embedding-model nomic-embed-text \\
        --ollama-host 192.168.15.100 \\
        --output config/embedding_whitening.json
    uv run python -m scripts.fit_embedding_whitening ... --mode mean_center
"""

import argparse
import asyncio
import json
import os
import sys

from expert_backend import OllamaClient

_MODE_MEAN_CENTER = "mean_center"
_MODE_WHITEN = "whiten"


def _fit_whitening_from_vectors(
    vectors: list[list[float]], mode: str
) -> tuple[list[float], list[list[float]] | None]:
    """Pure numpy computation, isolated from the ollama-fetching glue in main().

    mu = mean(vectors). In whiten mode, W = U @ diag(1/sqrt(S)) from the
    SVD of the covariance matrix (Su et al. 2021's BERT-whitening).
    Returns (mu, W) as plain nested Python lists (no numpy array leaks into
    the returned artifact, so the caller can json.dump it directly).
    """
    import numpy as np

    matrix = np.array(vectors)
    mu = matrix.mean(axis=0)
    if mode == _MODE_MEAN_CENTER:
        return mu.tolist(), None

    centered = matrix - mu
    covariance = (centered.T @ centered) / centered.shape[0]
    u, s, _vt = np.linalg.svd(covariance)
    whitening_matrix = u @ np.diag(1.0 / np.sqrt(s))
    return mu.tolist(), whitening_matrix.tolist()


async def _collect_background_embeddings(
    ollama_client: OllamaClient, embedding_model: str, texts: list[str]
) -> list[list[float]]:
    """Embed every background text sequentially (mirrors run_experiment.py's sequential design)."""
    vectors = []
    for text in texts:
        vectors.append(await ollama_client.embed(embedding_model, text))
    return vectors


def _load_background_queries(dataset_path: str) -> list[str]:
    """Read every row's `query` field from a build_dataset.py-produced JSONL file."""
    with open(dataset_path, encoding="utf-8") as f:
        return [json.loads(line)["query"] for line in f if line.strip()]


async def _fit_and_write(
    dataset_path: str,
    embedding_model: str,
    ollama_host: str,
    ollama_port: int,
    mode: str,
    output_path: str,
) -> None:
    texts = _load_background_queries(dataset_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    vectors = await _collect_background_embeddings(ollama_client, embedding_model, texts)
    mean_vector, whitening_matrix = _fit_whitening_from_vectors(vectors, mode)
    artifact = {
        "mean_vector": mean_vector,
        "whitening_matrix": whitening_matrix,
        "dim": len(mean_vector),
        "n_background_samples": len(vectors),
        "embedding_model": embedding_model,
        "mode": mode,
    }
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=2)
    print(
        f"[fit_embedding_whitening] wrote {output_path} "
        f"(mode={mode}, dim={len(mean_vector)}, n={len(vectors)})",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fit an embedding_postprocess artifact (mean_center or whiten) for router.py"
    )
    parser.add_argument(
        "--dataset", default="data/dataset.jsonl", help="Background corpus (query column used)"
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument(
        "--ollama-port", type=int, default=11434, help="ollama's own port, not the mesh HTTP port"
    )
    parser.add_argument("--mode", choices=[_MODE_MEAN_CENTER, _MODE_WHITEN], default=_MODE_WHITEN)
    parser.add_argument("--output", default="config/embedding_whitening.json")
    args = parser.parse_args()

    asyncio.run(
        _fit_and_write(
            args.dataset,
            args.embedding_model,
            args.ollama_host,
            args.ollama_port,
            args.mode,
            args.output,
        )
    )


if __name__ == "__main__":
    main()

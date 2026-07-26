"""E6: offline training of the supervised domain classifier (routing_method=supervised_classifier).

Trains a multi-class LogisticRegression from question-embedding features
and domain labels ONLY. Deliberately never reads results/*/results.jsonl:
Iter10's label leakage came from training on probe/dispatch-derived
features (self_confidence, margin, is_top1, ...) evaluated on the same
questions used for online testing. Every function here takes
{"query": ..., "domain": ...} rows (e.g. from
build_dataset.py's build_classifier_training_rows), so there is no
parameter through which probe/dispatch results could leak in.

Usage (module mode; requires a live ollama node reachable for embeddings):
    uv run python -m scripts.train_domain_classifier \\
        --train-data data/classifier_train.jsonl \\
        --embedding-model nomic-embed-text \\
        --ollama-host 192.168.15.100 \\
        --output models/domain_classifier.joblib
"""

import argparse
import asyncio
import json
import os
import sys

import joblib
from sklearn.linear_model import LogisticRegression

from expert_backend import OllamaClient

# scikit-learn >=1.5 always fits a single softmax (multinomial-equivalent)
# over all classes for multi-class LogisticRegression with the default
# solver (the old multi_class="ovr"/"multinomial" switch was removed),
# which is what classifier.estimate_confidence_classifier relies on:
# predict_proba's per-class probabilities sum to 1 across domains, making
# cross-node confidence values directly comparable.
_MAX_ITER = 1000


def _load_training_rows(train_data_path: str) -> list[dict]:
    """Read {"id", "query", "domain"} rows from a JSONL file."""
    with open(train_data_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def build_training_features(
    ollama_client: OllamaClient, embedding_model: str, rows: list[dict]
) -> tuple[list[list[float]], list[str]]:
    """Embed every row's query text; return (embeddings, domain labels) in matching order.

    Sequential (not concurrent) to mirror run_experiment.py's and
    fit_embedding_whitening.py's sequential embedding calls.
    """
    embeddings = []
    labels = []
    for row in rows:
        embeddings.append(await ollama_client.embed(embedding_model, row["query"]))
        labels.append(row["domain"])
    return embeddings, labels


def train_classifier(embeddings: list[list[float]], labels: list[str]) -> LogisticRegression:
    """Fit a multi-class LogisticRegression from embedding features to domain labels.

    class_weight="balanced" re-weights each class inversely to its
    frequency: legal's training pool is structurally about half the size
    of every other domain's at the default domain_target_size (JMMLU has
    no professional_law task; see build_dataset.py's
    build_classifier_training_rows docstring), and without this the
    classifier would be trained to under-predict legal simply because it
    saw fewer examples, not because the signal is weaker.
    """
    model = LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced")
    model.fit(embeddings, labels)
    return model


async def _train_and_save(
    train_data_path: str, embedding_model: str, ollama_host: str, ollama_port: int, output_path: str
) -> None:
    rows = _load_training_rows(train_data_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings, labels = await build_training_features(ollama_client, embedding_model, rows)
    model = train_classifier(embeddings, labels)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, output_path)
    print(
        f"[train_domain_classifier] wrote {output_path} "
        f"(n_samples={len(rows)}, classes={sorted(set(labels))})",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train the E6 supervised domain classifier from {query, domain} rows"
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="JSONL of {id, query, domain} rows (e.g. build_dataset.py's --classifier-train-output)",
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--output", default="models/domain_classifier.joblib")
    args = parser.parse_args()

    asyncio.run(
        _train_and_save(
            args.train_data, args.embedding_model, args.ollama_host, args.ollama_port, args.output
        )
    )


if __name__ == "__main__":
    main()

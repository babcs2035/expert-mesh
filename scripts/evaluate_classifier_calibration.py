"""Iter29 (classifier_calibration=platt): generate calibrated-classifier predictions offline.

Recomputes (selected_domain, confidence) for every row of the 1600-question
evaluation dataset using a newly trained CalibratedClassifierCV artifact
(scripts/train_domain_classifier.py's output), WITHOUT running any of
run_experiment.py's probe/dispatch/LLM-generation flow: under the frozen
Iter28 config (confidence_threshold=0.0, dispatch_top_k=1,
aggregation_method=max_confidence), every node's probe confidence is that
node's own domain's classifier probability, and select_dispatch_targets
(aggregator.py) simply picks the single highest one. Since every node
loads the same shared classifier (classifier.py's design), this reduces to
one predict_proba call per row -- no 10-node round trip needed.

Only calls out to a live ollama node for embeddings (query_embedding is not
persisted in results.jsonl, so it must be recomputed); no LLM generation,
probe, or dispatch traffic is produced. The uncalibrated ("before") side of
the Iter29 comparison is NOT recomputed here -- it reuses Iter28's already-
measured results/20260731_162722/results.jsonl as-is (journal Iter29 plan,
evaluation steps 2-3).

This script only emits the calibrated-side JSONL; the before/after ECE,
McNemar, per-domain CI, and flip-rate comparisons themselves (journal
Iter29 plan, evaluation steps 4-7) are computed in the experiment phase
using metrics.py's existing compute_ece / compute_mcnemar_test /
compute_precision_recall_per_domain / compute_wilson_confidence_interval,
not here.

Usage (module mode; requires a live ollama node reachable for embeddings):
    uv run python -m scripts.evaluate_classifier_calibration \\
        --dataset data/dataset.jsonl \\
        --classifier models/domain_classifier_platt.joblib \\
        --embedding-model nomic-embed-text \\
        --ollama-host 192.168.15.100 \\
        --output results/iter29_calibrated_predictions.jsonl
"""

import argparse
import asyncio
import json
import sys
from typing import TextIO

from sklearn.calibration import CalibratedClassifierCV

from classifier import load_domain_classifier
from expert_backend import OllamaClient


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def predict_calibrated_rows(
    ollama_client: OllamaClient,
    embedding_model: str,
    classifier: CalibratedClassifierCV,
    dataset: list[dict],
) -> list[dict]:
    """Recompute (selected_domain, confidence) for every dataset row via the calibrated classifier.

    Sequential (not concurrent) embedding calls, matching
    train_domain_classifier.py's build_training_features and
    fit_embedding_whitening.py's existing pattern for single-node
    offline embedding jobs.
    """
    classes = list(classifier.classes_)
    rows = []
    for row in dataset:
        query_embedding = await ollama_client.embed(embedding_model, row["query"])
        probabilities = classifier.predict_proba([query_embedding])[0]
        best_index = max(range(len(classes)), key=lambda i: probabilities[i])
        rows.append(
            {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": classes[best_index],
                "confidence": float(probabilities[best_index]),
            }
        )
    return rows


async def _run(
    dataset_path: str,
    classifier_path: str,
    embedding_model: str,
    ollama_host: str,
    ollama_port: int,
    output: TextIO,
) -> None:
    dataset = _read_jsonl(dataset_path)
    classifier = load_domain_classifier(classifier_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    rows = await predict_calibrated_rows(ollama_client, embedding_model, classifier, dataset)
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[evaluate_classifier_calibration] wrote {len(rows)} rows (classifier={classifier_path})",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute a calibrated classifier's (selected_domain, confidence) over the "
            "evaluation dataset, for offline before/after comparison against a results.jsonl "
            "from the uncalibrated production classifier (Iter29)"
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="JSONL of {id, query, expected_domains} rows"
    )
    parser.add_argument(
        "--classifier", required=True, help="Path to the CalibratedClassifierCV joblib artifact"
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the calibrated-side JSONL to (default: stdout)",
    )
    args = parser.parse_args()

    if args.output is None:
        asyncio.run(
            _run(
                args.dataset,
                args.classifier,
                args.embedding_model,
                args.ollama_host,
                args.ollama_port,
                sys.stdout,
            )
        )
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            asyncio.run(
                _run(
                    args.dataset,
                    args.classifier,
                    args.embedding_model,
                    args.ollama_host,
                    args.ollama_port,
                    f,
                )
            )


if __name__ == "__main__":
    main()

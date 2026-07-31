"""E6: offline training of the supervised domain classifier (routing_method=supervised_classifier).

Trains a multi-class LogisticRegression from question-embedding features
and domain labels ONLY, then wraps it in CalibratedClassifierCV (Iter29,
classifier_calibration=platt) for better-calibrated predict_proba output.
Deliberately never reads results/*/results.jsonl: Iter10's label leakage
came from training on probe/dispatch-derived features (self_confidence,
margin, is_top1, ...) evaluated on the same questions used for online
testing. Every function here takes {"query": ..., "domain": ...} rows
(e.g. from build_dataset.py's build_classifier_training_rows), so there
is no parameter through which probe/dispatch results could leak in.

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from expert_backend import OllamaClient

# scikit-learn >=1.5 always fits a single softmax (multinomial-equivalent)
# over all classes for multi-class LogisticRegression with the default
# solver (the old multi_class="ovr"/"multinomial" switch was removed).
# This base estimator's own predict_proba already sums to 1 across
# domains, but Iter25/Iter27 measured it as poorly calibrated (ECE=0.204,
# mean_confidence > accuracy in every domain). CalibratedClassifierCV
# (below) recalibrates it; classifier.estimate_confidence_classifier
# still relies on the wrapped model's predict_proba summing to 1 across
# domains, which sklearn's internal one-vs-rest renormalization preserves
# (see scikit-learn.org/stable/modules/calibration.html #1.16.3.3).
_MAX_ITER = 1000

# Iter29 (classifier_calibration=platt): method="sigmoid" (Platt scaling)
# was chosen over "isotonic" because this training set is far below the
# scale sklearn's own docs call safe for isotonic ("not recommended when
# the number of calibration samples is too low (<<1000) since it then
# tends to overfit" -- scikit-learn.org/stable/modules/calibration.html).
# With the default cv=5 StratifiedKFold and ensemble=True, each fold's
# held-out calibration slice is only ~30 rows/domain (~15 for legal,
# the smallest domain at 77 rows total), well under that threshold.
# isotonic is deferred to a future iteration, only if sigmoid fails to
# meet the ECE<=0.150 success condition (journal Iter29 plan).
_CALIBRATION_METHOD = "sigmoid"
_CALIBRATION_CV = 5


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


def train_classifier(
    embeddings: list[list[float]], labels: list[str], cv: int = _CALIBRATION_CV
) -> CalibratedClassifierCV:
    """Fit a calibrated multi-class classifier from embedding features to domain labels.

    Base estimator is LogisticRegression; class_weight="balanced"
    re-weights each class inversely to its frequency: legal's training
    pool is structurally about half the size of every other domain's at
    the default domain_target_size (JMMLU has no professional_law task;
    see build_dataset.py's build_classifier_training_rows docstring), and
    without this the classifier would be trained to under-predict legal
    simply because it saw fewer examples, not because the signal is
    weaker.

    The base estimator is then wrapped in CalibratedClassifierCV
    (method="sigmoid"=Platt, ensemble=True; see _CALIBRATION_METHOD's
    comment above for why sigmoid over isotonic at this data scale).
    `cv` defaults to the production value (5-fold StratifiedKFold) but is
    exposed as a parameter so tests can exercise the same code path with
    a smaller fold count on toy data, rather than branching test code
    away from what production actually runs.
    """
    base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced")
    calibrated_model = CalibratedClassifierCV(
        base_estimator, method=_CALIBRATION_METHOD, cv=cv, ensemble=True
    )
    calibrated_model.fit(embeddings, labels)
    return calibrated_model


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

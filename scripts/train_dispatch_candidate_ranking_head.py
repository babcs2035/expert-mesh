"""Iter59 (dispatch_candidate_ranking=multilabel_binary_relevance_head): offline
training of a One-vs-Rest (binary relevance) sigmoid head that re-ranks
dispatch candidates 2nd place onward.

This head is a SEPARATE artifact from models/domain_classifier.joblib and
never replaces it. The production classifier (scripts/train_domain_classifier.py,
CalibratedClassifierCV(LogisticRegression)) remains the sole source of rank_1
(argmax) for every node's dispatch decision; this script only produces a
second, independent model whose per-domain sigmoid scores are used (by
scripts/evaluate_dispatch_candidate_ranking.py) to re-order the remaining
9 domains for rank_2 selection. Saving to a different joblib path
(models/dispatch_candidate_ranking_head.joblib by default) is intentional
and must not be changed to overwrite models/domain_classifier.joblib.

Training data loading and embedding generation are reused verbatim from
scripts/train_domain_classifier.py's `_load_training_rows()` and
`build_training_features()` (same data/classifier_train.jsonl rows, same
nomic-embed-text embedding_model, same sequential embed-call pattern) so
this script cannot diverge from the production classifier's input
features -- only the estimator differs.

`_extract_sample_weights()` from train_domain_classifier.py is deliberately
NOT reused here: OneVsRestClassifier fits one independent binary problem
per domain, and passing `class_weight="balanced"` to each binary
LogisticRegression already computes a balanced weighting from that binary
problem's own positive/negative counts. Combining that with the
multiclass-domain-count-derived sample_weight from
_extract_sample_weights() would risk reproducing the Iter32
sample_weight * class_weight_ multiplicative-shift bug in a new form; the
single-lever principle for this iteration is "add an OvR ranking head",
not "resolve how domain-imbalance weighting composes across two
independent weighting mechanisms".

legal is the smallest training domain (77 rows vs 150 for every other
domain; JMMLU has no professional_law task, see build_dataset.py). Per
Zhang & Zhou 2017 ("Binary Relevance for Multi-Label Learning: An
Overview"), binary relevance's per-label binary problems each risk their
own class imbalance; this script reports 5-fold stratified cross-validated
per-domain ROC-AUC and average precision to stderr specifically so a
degenerate legal head (e.g. near-random ROC-AUC) is visible before the
evaluation phase runs, rather than only surfacing much later as an
unexplained compound_domain_set_recall regression.

Usage (module mode; requires a live ollama node reachable for embeddings,
e.g. via the project's standard SSH local port forward,
`ssh -fNT -L 11435:localhost:11434 wafl500`):
    uv run python -m scripts.train_dispatch_candidate_ranking_head \\
        --train-data data/classifier_train.jsonl \\
        --embedding-model nomic-embed-text \\
        --ollama-host 127.0.0.1 --ollama-port 11435 \\
        --output models/dispatch_candidate_ranking_head.joblib
"""

import argparse
import asyncio
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from expert_backend import OllamaClient
from scripts.train_domain_classifier import _load_training_rows, build_training_features

# Matches train_domain_classifier.py's _MAX_ITER: same base estimator family,
# same convergence budget, so the only intentional difference between the
# two artifacts is single-label multiclass (softmax, calibrated) vs
# one-vs-rest binary relevance (independent sigmoids, uncalibrated).
_MAX_ITER = 1000

# 5-fold cross-validation for the diagnostic ROC-AUC/average-precision
# report only (see module docstring); the final saved model is refit on
# all rows, matching train_domain_classifier.py's cv=5 default for its
# own (different-purpose) CalibratedClassifierCV.
_DIAGNOSTIC_CV = 5


def train_ranking_head(
    embeddings: list[list[float]], labels: list[str]
) -> OneVsRestClassifier:
    """Fit a One-vs-Rest (binary relevance) sigmoid head from embeddings to domain labels.

    Each of the n_domains binary sub-problems ("is this domain, or not")
    is fit independently via LogisticRegression(class_weight="balanced"),
    so no domain's binary classifier's weighting depends on any other
    domain's row count -- unlike a single joint softmax, adding or removing
    rows for one domain cannot shift another domain's decision boundary.
    This model is intentionally left uncalibrated (no CalibratedClassifierCV
    wrapper): its scores are only ever used to rank rank_2 candidates
    against each other, never compared across a probability threshold or
    reported as a calibrated probability (journal Iter59 plan, R1).
    """
    model = OneVsRestClassifier(LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced"))
    model.fit(embeddings, labels)
    return model


def _print_per_domain_cv_diagnostics(
    embeddings: list[list[float]], labels: list[str], cv: int = _DIAGNOSTIC_CV
) -> None:
    """Print 5-fold stratified cross-validated per-domain ROC-AUC / average precision to stderr.

    Uses cross_val_predict(method="decision_function") on a freshly
    constructed (unfitted) OneVsRestClassifier so these diagnostics never
    touch the final model returned by train_ranking_head() (out-of-fold
    scores only, avoiding the in-sample overconfidence that would hide a
    degenerate minority-class head -- see module docstring on legal).
    """
    classes = sorted(set(labels))
    embeddings_array = np.asarray(embeddings)
    diagnostic_model = OneVsRestClassifier(
        LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced")
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_scores = cross_val_predict(
        diagnostic_model, embeddings_array, labels, cv=skf, method="decision_function"
    )
    labels_binarized = label_binarize(labels, classes=classes)
    for i, domain in enumerate(classes):
        roc_auc = roc_auc_score(labels_binarized[:, i], oof_scores[:, i])
        average_precision = average_precision_score(labels_binarized[:, i], oof_scores[:, i])
        print(
            f"[train_dispatch_candidate_ranking_head] domain={domain} "
            f"n={int(labels_binarized[:, i].sum())} "
            f"cv_roc_auc={roc_auc:.4f} cv_average_precision={average_precision:.4f}",
            file=sys.stderr,
        )


async def _train_and_save(
    train_data_path: str, embedding_model: str, ollama_host: str, ollama_port: int,
    output_path: str,
) -> None:
    """Load training rows, embed them, run CV diagnostics, fit, and save the OvR head."""
    rows = _load_training_rows(train_data_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings, labels = await build_training_features(ollama_client, embedding_model, rows)

    _print_per_domain_cv_diagnostics(embeddings, labels)

    model = train_ranking_head(embeddings, labels)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, output_path)
    print(
        f"[train_dispatch_candidate_ranking_head] wrote {output_path} "
        f"(n_samples={len(rows)}, classes={sorted(set(labels))})",
        file=sys.stderr,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the Iter59 One-vs-Rest dispatch-candidate-ranking head from "
            "{query, domain} rows (rank_1 remains the unchanged production classifier)"
        )
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="JSONL of {id, query, domain} rows (e.g. data/classifier_train.jsonl)",
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--output", default="models/dispatch_candidate_ranking_head.joblib")
    args = parser.parse_args()

    asyncio.run(
        _train_and_save(
            args.train_data, args.embedding_model, args.ollama_host, args.ollama_port,
            args.output,
        )
    )


if __name__ == "__main__":
    main()

"""Iter60 (multilabel_training_signal=synthetic_two_domain_training_examples): offline
training of a One-vs-Rest (binary relevance) sigmoid head from a TRUE multi-label
target, built via MultiLabelBinarizer from classifier_train.jsonl's 1427 single-domain
rows plus scripts/generate_multidomain_training_examples.py's synthetic two-domain rows.

This is a SEPARATE artifact from both models/domain_classifier.joblib (the
production rank_1 classifier, untouched) and
models/dispatch_candidate_ranking_head.joblib (Iter59's OvR head, trained from
single-label rows only and kept unchanged as the control group -- see
scripts/train_dispatch_candidate_ranking_head.py, which this iteration does
not modify). scripts/evaluate_dispatch_candidate_ranking.py's rank_1 always
comes from a fixed baseline results.jsonl; this head only ever supplies rank_2
scores for the remaining 9 domains.

MultiLabelBinarizer is required here (unlike Iter59): passing a list of
label-lists directly to OneVsRestClassifier.fit() raises "Sequence of
sequences are no longer supported" on this project's sklearn 1.9.0 (journal
Iter60 investigation, Q1). A structural consequence of going through
MultiLabelBinarizer is that the fitted OneVsRestClassifier's own `.classes_`
becomes a plain integer column-index array (0..9), NOT the domain-name
strings Iter59's classes_ was -- so the domain-name mapping (`mlb.classes_`)
must be saved alongside the model. This is why this script saves a dict
payload ({"model": ..., "classes": ...}) instead of Iter59's bare
`joblib.dump(model, ...)`; scripts/evaluate_dispatch_candidate_ranking.py's
_load_head() understands both formats.

Usage (module mode; requires a live ollama node reachable for embeddings,
e.g. via `ssh -fNT -L 11435:localhost:11434 wafl500`):
    uv run python -m scripts.train_multilabel_dispatch_head \\
        --train-data data/classifier_train.jsonl \\
        --multilabel-train-data data/classifier_train_multidomain.jsonl \\
        --embedding-model nomic-embed-text \\
        --ollama-host 127.0.0.1 --ollama-port 11435 \\
        --output models/dispatch_multilabel_head.joblib
"""

import argparse
import asyncio
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from expert_backend import OllamaClient
from scripts.train_domain_classifier import _load_training_rows, build_training_features

# Matches train_domain_classifier.py's and train_dispatch_candidate_ranking_head.py's
# _MAX_ITER: same base estimator family and convergence budget across all three
# artifacts, so estimator settings are never a confound between them.
_MAX_ITER = 1000

# 5-fold cross-validation for the diagnostic per-domain ROC-AUC/average-precision
# report only; the final saved model is refit on all rows (matches Iter59's
# train_dispatch_candidate_ranking_head.py convention). KFold (not
# StratifiedKFold) is used because Y is a 2D multi-label indicator matrix,
# which StratifiedKFold does not support (journal Iter60 plan).
_DIAGNOSTIC_CV = 5
_DIAGNOSTIC_CV_RANDOM_STATE = 42

# A0 (journal.md Iter60 plan, "no-op対策"): the number of MLB rows carrying
# >=2 positive labels must both equal the synthetic row count exactly (no
# single-label row should ever end up multi-label, and no synthetic row
# should ever collapse to single-label) and clear this floor, matching the
# generation script's own _MINIMUM_ACCEPTABLE_ROW_COUNT.
_A0_MINIMUM_MULTILABEL_ROW_COUNT = 120
_EXPECTED_DOMAIN_COUNT = 10


def _normalize_labels(labels: list[str | list[str]]) -> list[list[str]]:
    """Normalize build_training_features()'s heterogeneous `labels` output to list[list[str]].

    build_training_features() (train_domain_classifier.py) does
    `labels.append(row["domain"])` verbatim for every row, so single-label
    rows (domain: str) and this iteration's synthetic two-domain rows
    (domain: list[str]) come back mixed in one list; MultiLabelBinarizer
    requires every element to be an iterable of labels.
    """
    return [[label] if isinstance(label, str) else list(label) for label in labels]


def build_multilabel_targets(
    labels: list[str | list[str]],
) -> tuple[np.ndarray, MultiLabelBinarizer]:
    """Fit a MultiLabelBinarizer over the normalized labels and return (Y, mlb).

    `mlb.classes_` (the domain-name array corresponding to Y's columns) must
    be saved by the caller: once Y is a 0/1 indicator matrix,
    OneVsRestClassifier.classes_ degrades to plain integer column indices
    (journal Iter60 investigation Q1), so mlb.classes_ is the only surviving
    record of which column is which domain.
    """
    normalized = _normalize_labels(labels)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(normalized)
    return Y, mlb


def _assert_a0_true_multilabel_signal(Y: np.ndarray, mlb: MultiLabelBinarizer, n_synthetic_rows: int) -> int:
    """A0: the multi-label teacher signal actually reached the target matrix.

    Guards against the exact failure mode this iteration's single lever is
    supposed to fix (Iter59's teacher signal was 100% single-label): if this
    assertion fails, the synthetic rows were dropped, mis-joined, or
    collapsed to single-label somewhere upstream, and Y is indistinguishable
    from Iter59's target -- i.e. this iteration would be a silent no-op.
    """
    multilabel_row_count = int((Y.sum(axis=1) >= 2).sum())
    if multilabel_row_count != n_synthetic_rows:
        raise AssertionError(
            f"A0 failed: {multilabel_row_count} rows in Y have >=2 positive labels, "
            f"expected exactly {n_synthetic_rows} (the synthetic row count) -- "
            "a single-label row became multi-label, or a synthetic row collapsed "
            "to single-label"
        )
    if multilabel_row_count < _A0_MINIMUM_MULTILABEL_ROW_COUNT:
        raise AssertionError(
            f"A0 failed: only {multilabel_row_count} multi-label rows, below the "
            f"pre-registered floor of {_A0_MINIMUM_MULTILABEL_ROW_COUNT}"
        )
    if len(mlb.classes_) != _EXPECTED_DOMAIN_COUNT:
        raise AssertionError(
            f"A0 failed: mlb.classes_ has {len(mlb.classes_)} domains, "
            f"expected {_EXPECTED_DOMAIN_COUNT}"
        )
    if not all(isinstance(domain, str) for domain in mlb.classes_):
        raise AssertionError(f"A0 failed: mlb.classes_ contains non-string entries: {mlb.classes_}")
    return multilabel_row_count


def _covered_domain_pairs(labels: list[str | list[str]]) -> set[frozenset[str]]:
    """Set of distinct 2-domain pairs actually present among the (normalized) synthetic rows."""
    pairs: set[frozenset[str]] = set()
    for label in _normalize_labels(labels):
        if len(label) >= 2:
            pairs.add(frozenset(label))
    return pairs


def train_multilabel_ranking_head(
    embeddings: list[list[float]], Y: np.ndarray
) -> OneVsRestClassifier:
    """Fit a One-vs-Rest sigmoid head from embeddings to a true multi-label indicator matrix Y.

    Each of the 10 domains' binary sub-problem is fit independently via
    LogisticRegression(class_weight="balanced"), exactly as in Iter59's
    train_ranking_head() -- only the target (Y, a true multi-label matrix,
    vs Iter59's single-label array) differs, per the single-lever principle.
    """
    model = OneVsRestClassifier(LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced"))
    model.fit(embeddings, Y)
    return model


def _print_per_domain_cv_diagnostics(
    embeddings: list[list[float]], Y: np.ndarray, classes: np.ndarray, cv: int = _DIAGNOSTIC_CV
) -> None:
    """Print 5-fold cross-validated per-domain ROC-AUC / average precision to stderr.

    Uses cross_val_predict(method="decision_function") on a freshly
    constructed (unfitted) OneVsRestClassifier so these diagnostics never
    touch the final model returned by train_multilabel_ranking_head()
    (out-of-fold scores only, matching Iter59's train_dispatch_candidate_
    ranking_head.py). KFold (not StratifiedKFold, which does not accept a
    2D multi-label y) is used, matching this script's module docstring.
    """
    embeddings_array = np.asarray(embeddings)
    diagnostic_model = OneVsRestClassifier(
        LogisticRegression(max_iter=_MAX_ITER, class_weight="balanced")
    )
    kf = KFold(n_splits=cv, shuffle=True, random_state=_DIAGNOSTIC_CV_RANDOM_STATE)
    oof_scores = cross_val_predict(
        diagnostic_model, embeddings_array, Y, cv=kf, method="decision_function"
    )
    for i, domain in enumerate(classes):
        roc_auc = roc_auc_score(Y[:, i], oof_scores[:, i])
        average_precision = average_precision_score(Y[:, i], oof_scores[:, i])
        print(
            f"[train_multilabel_dispatch_head] domain={domain} "
            f"n_positive={int(Y[:, i].sum())} "
            f"cv_roc_auc={roc_auc:.4f} cv_average_precision={average_precision:.4f}",
            file=sys.stderr,
        )


async def _train_and_save(
    train_data_path: str,
    multilabel_train_data_path: str,
    embedding_model: str,
    ollama_host: str,
    ollama_port: int,
    output_path: str,
) -> None:
    """Load both training files, embed all rows together, fit the MLB-based OvR head, and save it."""
    single_label_rows = _load_training_rows(train_data_path)
    synthetic_rows = _load_training_rows(multilabel_train_data_path)
    rows = single_label_rows + synthetic_rows

    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings, labels = await build_training_features(ollama_client, embedding_model, rows)

    Y, mlb = build_multilabel_targets(labels)
    multilabel_row_count = _assert_a0_true_multilabel_signal(Y, mlb, len(synthetic_rows))
    covered_pairs = _covered_domain_pairs(labels)
    print(
        f"[train_multilabel_dispatch_head] A0 PASS: {multilabel_row_count} multi-label rows "
        f"covering {len(covered_pairs)} distinct domain pairs "
        f"(domains={list(mlb.classes_)})",
        file=sys.stderr,
    )

    _print_per_domain_cv_diagnostics(embeddings, Y, mlb.classes_)

    model = train_multilabel_ranking_head(embeddings, Y)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    joblib.dump({"model": model, "classes": list(mlb.classes_)}, output_path)
    print(
        f"[train_multilabel_dispatch_head] wrote {output_path} "
        f"(n_single_label_rows={len(single_label_rows)}, n_synthetic_rows={len(synthetic_rows)}, "
        f"classes={list(mlb.classes_)})",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the Iter60 MultiLabelBinarizer-based One-vs-Rest dispatch-candidate-"
            "ranking head from {query, domain} single-label rows plus synthetic "
            "two-domain rows (rank_1 remains the unchanged production classifier)"
        )
    )
    parser.add_argument(
        "--train-data",
        required=True,
        help="JSONL of {id, query, domain: str} single-label rows (data/classifier_train.jsonl)",
    )
    parser.add_argument(
        "--multilabel-train-data",
        required=True,
        help="JSONL of {id, query, domain: list[str]} synthetic two-domain rows "
        "(data/classifier_train_multidomain.jsonl)",
    )
    parser.add_argument(
        "--embedding-model", required=True, help="Must match config.yaml's embedding_model"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--output", default="models/dispatch_multilabel_head.joblib")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    asyncio.run(
        _train_and_save(
            args.train_data,
            args.multilabel_train_data,
            args.embedding_model,
            args.ollama_host,
            args.ollama_port,
            args.output,
        )
    )


if __name__ == "__main__":
    main()

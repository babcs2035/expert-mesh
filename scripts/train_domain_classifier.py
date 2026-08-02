"""E6: offline training of the supervised domain classifier (routing_method=supervised_classifier).

Trains a multi-class LogisticRegression from question-embedding features
and domain labels ONLY, then wraps it in CalibratedClassifierCV (Iter31,
classifier_calibration=temperature) for better-calibrated predict_proba output.
Deliberately never reads results/*/results.jsonl: Iter10's label leakage
came from training on probe/dispatch-derived features (self_confidence,
margin, is_top1, ...) evaluated on the same questions used for online
testing. Every function here takes {"query": ..., "domain": ...} rows
(e.g. from build_dataset.py's build_classifier_training_rows), so there
is no parameter through which probe/dispatch results could leak in.
Rows may additionally carry a "sample_weight" field (Iter32,
classifier_training_data_composition=education_proxy_task_revision),
but this field is currently unused: `_extract_sample_weights()` computes
domain-balanced weights automatically from domain counts, reproducing
sklearn's class_weight='balanced' effective weighting without the Iter32
multiplicative bug (sample_weight *= class_weight_).

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

# Iter31 (classifier_calibration=temperature): method="temperature" is used
# here because Iter30's method="isotonic" cleared the ECE success threshold
# (0.19336->0.12142) but was judged partial -- after Benjamini-Hochberg
# correction (q=0.05) across 20 per-domain precision/recall metrics,
# medical_recall regressed significantly (0.4831->0.3820, p=0.000144;
# journal Iter30 discussion, backlog B51's automatic next choice).
# Iter31's investigation confirmed the structural reason this class-
# specific distortion is plausible for isotonic/sigmoid but not for
# temperature: both isotonic and Platt scaling fit one calibrator per
# class under scikit-learn's one-vs-rest (OvR) decomposition, so each
# class's curve can be skewed independently by its own held-out fold
# data (e.g. legal/medical's sparser samples). Temperature scaling instead
# fits a single scalar T that rescales the whole logit vector before a
# shared softmax, so there is no per-class calibrator to skew -- this
# structurally rules out the OvR-specific class distortion suspected of
# causing medical_recall's regression. cv=5 and ensemble=True are kept
# unchanged from Iter29/Iter30 (not re-tuned here) so that this run is a
# controlled comparison isolating the calibration method as the only
# variable (journal Iter31 plan, single-lever principle).
_CALIBRATION_METHOD = "temperature"
_CALIBRATION_CV = 5


def _load_training_rows(train_data_path: str) -> list[dict]:
    """Read {"id", "query", "domain"} rows from a JSONL file."""
    with open(train_data_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _extract_sample_weights(rows: list[dict]) -> list[float]:
    """Per-row training weight: domain-balanced weights matching sklearn's class_weight='balanced'.

    With class_weight=None in LogisticRegression, we compute sample_weight here
    to reproduce the exact same effective weighting that class_weight='balanced'
    provided (n_samples / (n_classes * n_domain_samples)). This avoids the
    Iter32 bug where sample_weight *= class_weight_ caused unintended multiplicative shifts.
    """
    from collections import Counter
    domain_counts = Counter(row["domain"] for row in rows)
    n_samples = len(rows)
    n_classes = len(domain_counts)
    weights = []
    for row in rows:
        d = row["domain"]
        weights.append(n_samples / (n_classes * domain_counts[d]))
    return weights


async def build_training_features(
    ollama_client: OllamaClient, embedding_model: str, rows: list[dict],
    fine_tuned_embed_model: str | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Embed every row's query text; return (embeddings, domain labels) in matching order.

    If fine_tuned_embed_model is provided, uses a local SentenceTransformer
    instead of the Ollama client for embedding generation.
    Sequential (not concurrent) to mirror run_experiment.py's and
    fit_embedding_whitening.py's sequential embedding calls.
    """
    from sentence_transformers import SentenceTransformer

    if fine_tuned_embed_model is not None:
        # Use local fine-tuned model (supports PEFT LoRA adapter)
        print(f"[train_domain_classifier] using fine-tuned embed model: {fine_tuned_embed_model}",
              file=sys.stderr)
        local_model = SentenceTransformer(
            fine_tuned_embed_model, trust_remote_code=True, device="cpu"
        )
        # Load and activate the LoRA adapter (PEFT default adapter name).
        # Dense projection head models include the Dense module internally and
        # have no adapter files -- this is silently skipped for those models.
        try:
            local_model.load_adapter(fine_tuned_embed_model, "default")
            local_model.set_adapter("default")
        except ValueError:
            # No adapter files (e.g., Dense projection head model).
            pass
        embeddings = []
        labels = []
        for row in rows:
            emb = local_model.encode(row["query"], normalize_embeddings=True,
                                     show_progress_bar=False)
            embeddings.append(emb.tolist())
            labels.append(row["domain"])
        return embeddings, labels
    else:
        embeddings = []
        labels = []
        for row in rows:
            embeddings.append(await ollama_client.embed(embedding_model, row["query"]))
            labels.append(row["domain"])
        return embeddings, labels


def train_classifier(
    embeddings: list[list[float]],
    labels: list[str],
    cv: int = _CALIBRATION_CV,
    sample_weight: list[float] | None = None,
) -> CalibratedClassifierCV:
    """Fit a calibrated multi-class classifier from embedding features to domain labels.

    Base estimator is LogisticRegression with class_weight=None.
    Domain balancing is achieved via `sample_weight` passed from
    `_extract_sample_weights()`, which computes per-row weights as
    n_samples / (n_classes * n_domain_samples) to reproduce the exact
    effective weighting of sklearn's class_weight='balanced' without the
    Iter32 bug (sample_weight *= class_weight_ multiplicative shift).
    Legal's training pool is structurally about half the size of every
    other domain's at the default domain_target_size (JMMLU has no
    professional_law task; see build_dataset.py's
    build_classifier_training_rows docstring), so its per-row sample_weight
    is ~2x that of 150-row domains, giving equal total effective weight
    per domain.

    The base estimator is then wrapped in CalibratedClassifierCV
    (method="temperature", ensemble=True; see _CALIBRATION_METHOD's comment
    above for why this method was chosen). Unlike isotonic/Platt (Iter29/
    Iter30), temperature scaling fits a single scalar T applied to the
    whole logit vector rather than one calibrator per class, so isotonic's
    per-class failure modes (tied probabilities across domains, exact
    0.0/1.0 outputs from held-out scores clipping to an extreme observed
    value, or an all-zero uniform-fallback row) do not apply to this
    implementation by construction -- callers evaluating this model still
    check for them (journal Iter31 plan, evaluation step 7) but expect
    zero occurrences rather than treating the check as a live risk.
    `cv` defaults to the production value (5-fold StratifiedKFold) but is
    exposed as a parameter so tests can exercise the same code path with
    a smaller fold count on toy data, rather than branching test code
    away from what production actually runs.

    `sample_weight` is forwarded to CalibratedClassifierCV.fit(), which
    passes it through to the base estimator's fit() on each fold. With
    class_weight=None, the sample_weight values are used directly as
    effective weights without any multiplicative adjustment. Defaults to
    None (all rows weighted equally).
    """
    base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)
    calibrated_model = CalibratedClassifierCV(
        base_estimator, method=_CALIBRATION_METHOD, cv=cv, ensemble=True
    )
    calibrated_model.fit(embeddings, labels, sample_weight=sample_weight)

    # education_boundary_tuning: shift education class intercept upward to move
    # the decision boundary towards the education side. The coefficient vector
    # (discrimination direction) remains unchanged -- only the parallel position
    # of the boundary shifts. This is a single-lever change: argmax flip should
    # be ~3-5% (<15% threshold). The shift is applied to the base estimator
    # inside each CalibratedClassifierCV fold.
    intercept_delta = 0.7  # education_boundary_tuning (Iter45: +0.5->+0.7)
    classes = calibrated_model.classes_
    edu_idx = list(classes).index("education")
    for cal in calibrated_model.calibrated_classifiers_:
        cal.estimator.intercept_[edu_idx] += intercept_delta

    return calibrated_model


async def _train_and_save(
    train_data_path: str, embedding_model: str, ollama_host: str, ollama_port: int,
    output_path: str, fine_tuned_embed_model: str | None = None,
) -> None:
    rows = _load_training_rows(train_data_path)
    sample_weight = _extract_sample_weights(rows)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    embeddings, labels = await build_training_features(
        ollama_client, embedding_model, rows, fine_tuned_embed_model=fine_tuned_embed_model
    )
    model = train_classifier(embeddings, labels, sample_weight=sample_weight)
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
    parser.add_argument(
        "--fine-tuned-embed-model",
        default=None,
        help="Path to a fine-tuned SentenceTransformer model (optional). "
             "If provided, uses this local model for embeddings instead of Ollama.",
    )
    parser.add_argument("--output", default="models/domain_classifier.joblib")
    args = parser.parse_args()

    asyncio.run(
        _train_and_save(
            args.train_data, args.embedding_model, args.ollama_host, args.ollama_port,
            args.output, fine_tuned_embed_model=args.fine_tuned_embed_model,
        )
    )


if __name__ == "__main__":
    main()

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

Iter30 (classifier_calibration=isotonic): each row's dict also carries a
`probabilities` field ({domain: float} for every domain the classifier
was trained on), not just the selected domain's confidence. isotonic's
non-monotonic-across-folds fit can produce exact 0.0/1.0 probabilities,
tied top candidates, or (rarely) an all-zero row that predict_proba
replaces with a uniform distribution -- none of these are visible from
the previously-sufficient (selected_domain, confidence) pair alone, so
the experiment phase's isotonic-specific checklist (journal Iter30 plan,
evaluation step 7) needs the full probability vector.

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

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from classifier import load_domain_classifier
from expert_backend import OllamaClient


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _compute_prediction_set(
    probabilities: np.ndarray,
    cp_data: dict,
    confidence_level: float = 0.90,
) -> tuple[list[int], int]:
    """Compute a conformal prediction set using cumulative APS method.

    Calibration: for each (sample, class) pair, compute nonconformity score
    S(x, j) = 1 - cumulative_prob_up_to_class_j (classes sorted by prob desc).
    q_hat = (1-alpha) quantile of ALL calibration scores.
    Prediction set: include classes in decreasing probability order
    while score (1 - cumsum) <= q_hat.

    This ensures the top class is always included (smallest score),
    and lower classes are added while the score stays below q_hat.

    Returns (list of class indices in prediction set, set size).
    """
    alpha = 1.0 - confidence_level
    all_scores = cp_data["all_scores"]  # shape=(n_cal, n_classes)

    # q_hat = (1-alpha) quantile of ALL nonconformity scores.
    # Using all scores (not just true-class) ensures proper coverage.
    flat_scores = all_scores.flatten()
    target = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(flat_scores)))
    q_hat = float(np.quantile(flat_scores, target, method="higher"))

    # Include classes in decreasing probability order while score <= q_hat
    # Score for class j = 1 - cumsum_prob_up_to_j (top class has smallest score)
    sorted_indices = np.argsort(-probabilities)  # descending
    pred_set: list[int] = []
    cumsum = 0.0
    for idx in sorted_indices:
        cumsum += probabilities[idx]
        score = 1.0 - cumsum
        if score <= q_hat:
            pred_set.append(int(idx))
        else:
            break

    # Fallback: if no class meets threshold, include top class
    if len(pred_set) == 0:
        pred_set = [int(np.argmax(probabilities))]

    return pred_set, len(pred_set)


async def predict_calibrated_rows(
    ollama_client: OllamaClient,
    embedding_model: str,
    classifier: CalibratedClassifierCV,
    dataset: list[dict],
    fine_tuned_embed_model: str | None = None,
    education_logit_bias: float = 0.0,
    education_threshold: float = 0.0,
    conformal_prediction: bool = False,
    calibration_dataset_path: str | None = None,
    confidence_level: float = 0.90,
) -> list[dict]:
    """Recompute (selected_domain, confidence) for every dataset row via the calibrated classifier.

    If fine_tuned_embed_model is provided, uses a local SentenceTransformer
    instead of the Ollama client for embedding generation.
    Sequential (not concurrent) embedding calls, matching
    train_domain_classifier.py's build_training_features and
    fit_embedding_whitening.py's existing pattern for single-node
    offline embedding jobs.
    """
    from sentence_transformers import SentenceTransformer

    classes = list(classifier.classes_)
    rows = []

    # Pre-compute conformal prediction calibration data (APS method).
    # Uses out-of-fold predictions from the CalibratedClassifierCV's internal
    # 5-fold CV to avoid the data-overlap problem (in-sample predictions are
    # overconfident, producing empty prediction sets).
    #
    # Non-conformity score for class j (sorted by prob descending):
    #   S(x, j) = 1 - cumulative_prob_up_to_class_j
    # The top class gets the HIGHEST score (least conforming), so we use the
    # alpha-quantile of TRUE-CLASS scores as q_hat. This ensures the top class
    # is included in the prediction set when its score <= q_hat.
    cp_data: dict | None = None
    if conformal_prediction:
        cal_dataset = _read_jsonl(calibration_dataset_path)  # type: ignore[arg-type]
        n_cal = len(cal_dataset)
        n_classes = len(classes)

        # Reconstruct 5-fold CV split using the same class labels
        from sklearn.model_selection import StratifiedKFold

        labels = [
            classes.index(r["domain"]) if r["domain"] in classes else 0
            for r in cal_dataset
        ]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Compute out-of-fold embeddings and predictions
        cal_embeddings = []
        if fine_tuned_embed_model is not None:
            local_model = SentenceTransformer(
                fine_tuned_embed_model, trust_remote_code=True, device="cpu"
            )
            try:
                local_model.load_adapter(fine_tuned_embed_model, "default")
                local_model.set_adapter("default")
            except ValueError:
                pass
            for cal_row in cal_dataset:
                cal_embeddings.append(
                    local_model.encode(cal_row["query"], normalize_embeddings=True,
                                       show_progress_bar=False)
                )
        else:
            for cal_row in cal_dataset:
                cal_embeddings.append(
                    await ollama_client.embed(embedding_model, cal_row["query"])
                )
        cal_embeddings = np.array(cal_embeddings)

        # Get out-of-fold predictions using the base estimator (LogisticRegression)
        base_estimator = classifier.estimator
        oof_probs = np.zeros((n_cal, n_classes))
        for train_idx, test_idx in skf.split(np.zeros(n_cal), labels):
            fold_clf = type(base_estimator)(max_iter=1000)
            fold_clf.fit(cal_embeddings[train_idx],
                         [labels[j] for j in train_idx])
            oof_probs[test_idx] = fold_clf.predict_proba(cal_embeddings[test_idx])

        # Compute nonconformity scores for ALL (sample, class) pairs.
        # S[i, j] = 1 - cumulative_prob_up_to_class_j (classes sorted by prob desc).
        # The top class gets the SMALLEST score (most conforming), so it's always
        # included in the prediction set. Lower classes are added while score <= q_hat.
        all_scores = np.zeros((n_cal, n_classes))
        for i in range(n_cal):
            probs = oof_probs[i]
            sorted_idx = np.argsort(-probs)  # descending
            cumsum = 0.0
            for rank, idx in enumerate(sorted_idx):
                cumsum += probs[idx]
                all_scores[i, idx] = 1.0 - cumsum

        cp_data = {"all_scores": all_scores}

    if fine_tuned_embed_model is not None:
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
        for row in dataset:
            query_embedding = local_model.encode(row["query"], normalize_embeddings=True,
                                                 show_progress_bar=False)
            probabilities = classifier.predict_proba([query_embedding])[0]
            # Apply post-hoc logit bias to education class
            if education_logit_bias != 0.0:
                edu_idx = classes.index("education") if "education" in classes else -1
                if edu_idx >= 0:
                    logits = np.log(probabilities + 1e-10)
                    logits[edu_idx] += education_logit_bias
                    logits_max = np.max(logits)
                    exp_logits = np.exp(logits - logits_max)
                    probabilities = exp_logits / np.sum(exp_logits)
            # Apply per-class threshold to education class (lowers decision boundary)
            if education_threshold > 0.0:
                edu_idx = classes.index("education") if "education" in classes else -1
                if edu_idx >= 0:
                    probabilities[edu_idx] += education_threshold
            # Compute conformal prediction set (uses original predict_proba probabilities)
            if conformal_prediction and cp_data is not None:
                pred_set, set_size = _compute_prediction_set(
                    probabilities, cp_data, confidence_level
                )
            best_index = max(range(len(classes)), key=lambda i: probabilities[i])
            row_dict = {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": classes[best_index],
                "confidence": float(probabilities[best_index]),
                "probabilities": {domain: float(p) for domain, p in zip(classes, probabilities)},
            }
            if conformal_prediction:
                row_dict["prediction_set"] = [classes[i] for i in pred_set]
                row_dict["set_size"] = set_size
            rows.append(row_dict)
    else:
        for row in dataset:
            query_embedding = await ollama_client.embed(embedding_model, row["query"])
            probabilities = classifier.predict_proba([query_embedding])[0]
            # Apply post-hoc logit bias to education class
            if education_logit_bias != 0.0:
                edu_idx = classes.index("education") if "education" in classes else -1
                if edu_idx >= 0:
                    logits = np.log(probabilities + 1e-10)
                    logits[edu_idx] += education_logit_bias
                    logits_max = np.max(logits)
                    exp_logits = np.exp(logits - logits_max)
                    probabilities = exp_logits / np.sum(exp_logits)
            # Apply per-class threshold to education class (lowers decision boundary)
            if education_threshold > 0.0:
                edu_idx = classes.index("education") if "education" in classes else -1
                if edu_idx >= 0:
                    probabilities[edu_idx] += education_threshold
            # Compute conformal prediction set (uses original predict_proba probabilities)
            if conformal_prediction and cp_data is not None:
                pred_set, set_size = _compute_prediction_set(
                    probabilities, cp_data, confidence_level
                )
            best_index = max(range(len(classes)), key=lambda i: probabilities[i])
            row_dict = {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": classes[best_index],
                "confidence": float(probabilities[best_index]),
                "probabilities": {domain: float(p) for domain, p in zip(classes, probabilities)},
            }
            if conformal_prediction:
                row_dict["prediction_set"] = [classes[i] for i in pred_set]
                row_dict["set_size"] = set_size
            rows.append(row_dict)
    return rows


async def _run(
    dataset_path: str,
    classifier_path: str,
    embedding_model: str,
    ollama_host: str,
    ollama_port: int,
    output: TextIO,
    fine_tuned_embed_model: str | None = None,
    education_logit_bias: float = 0.0,
    education_threshold: float = 0.0,
    conformal_prediction: bool = False,
    calibration_dataset_path: str | None = None,
    confidence_level: float = 0.90,
) -> None:
    dataset = _read_jsonl(dataset_path)
    classifier = load_domain_classifier(classifier_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    rows = await predict_calibrated_rows(
        ollama_client, embedding_model, classifier, dataset,
        fine_tuned_embed_model=fine_tuned_embed_model,
        education_logit_bias=education_logit_bias,
        education_threshold=education_threshold,
        conformal_prediction=conformal_prediction,
        calibration_dataset_path=calibration_dataset_path,
        confidence_level=confidence_level,
    )
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
        "--fine-tuned-embed-model",
        default=None,
        help="Path to a fine-tuned SentenceTransformer model (optional). "
             "If provided, uses this local model for embeddings instead of Ollama.",
    )
    parser.add_argument(
        "--education-logit-bias",
        type=float,
        default=0.0,
        help="Post-hoc logit bias for education class (applied after predict_proba)",
    )
    parser.add_argument(
        "--education-threshold",
        type=float,
        default=0.0,
        help="Per-class threshold addition for education class (added to probability before argmax; lowers decision boundary)",
    )
    parser.add_argument(
        "--conformal-prediction",
        action="store_true",
        default=False,
        help="Enable conformal prediction (APS method) to compute prediction sets",
    )
    parser.add_argument(
        "--calibration-dataset",
        default=None,
        help="Path to calibration dataset JSONL for non-conformity score computation (default: same as --dataset)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.90,
        help="Confidence level for conformal prediction coverage guarantee (default: 0.90)",
    )
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
                fine_tuned_embed_model=args.fine_tuned_embed_model,
                education_logit_bias=args.education_logit_bias,
                education_threshold=args.education_threshold,
                conformal_prediction=args.conformal_prediction,
                calibration_dataset_path=args.calibration_dataset,
                confidence_level=args.confidence_level,
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
                    fine_tuned_embed_model=args.fine_tuned_embed_model,
                    education_logit_bias=args.education_logit_bias,
                    education_threshold=args.education_threshold,
                    conformal_prediction=args.conformal_prediction,
                    calibration_dataset_path=args.calibration_dataset,
                    confidence_level=args.confidence_level,
                )
            )


if __name__ == "__main__":
    main()

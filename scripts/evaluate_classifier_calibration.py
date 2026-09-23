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
    qhat_source: str = "all",
    set_construction: str = "broken",
    qhat_quantile_direction: str = "upper",
    randomization_u: float | None = None,
) -> tuple[list[int], int]:
    """Compute a conformal prediction set using cumulative APS method.

    Calibration: for each (sample, class) pair, compute nonconformity score
    S(x, j) = 1 - cumulative_prob_up_to_class_j (classes sorted by prob desc).
    q_hat is the (1-alpha) quantile of a calibration score population, whose
    choice is controlled by qhat_source:
    - "all": ALL (sample, class) nonconformity scores (n_cal * n_classes
      values). This is the original Iter29-56 implementation. It does NOT
      match the standard APS calibration procedure (Romano et al., 2020,
      "Classification with Valid and Adaptive Coverage sets"), which defines
      q_hat over TRUE-CLASS scores only; using all scores was found (Iter69)
      to under-cover its nominal confidence level.
    - "true_class": only the TRUE-CLASS nonconformity score for each
      calibration sample (n_cal values), per the standard APS procedure.

    Prediction set construction (classes visited in decreasing probability
    order, cumsum = running cumulative probability), controlled by
    set_construction:
    - "broken": the pre-Iter70 implementation. Appends a class to the
      prediction set only AFTER checking `1 - cumsum <= q_hat`, i.e. the
      break decision is made before the append. Since `1 - cumsum` is
      monotonically decreasing as cumsum grows, this collapses to a single
      binary gate on the top class alone (`1 - p_max <= q_hat`): either
      every class passes and the loop never breaks (set_size = n_classes),
      or the very first class fails and the loop breaks immediately with an
      empty set that falls through to the top-class fallback below
      (set_size = 1). Intermediate set sizes (2..n_classes-1) cannot occur.
      Kept as the default to preserve prior runs' byte-for-byte output.
    - "corrected_aps": the standard APS construction (Romano et al., 2020).
      Each class is appended to the prediction set BEFORE the cumsum-based
      break check, so the set grows one class at a time until the
      cumulative probability first reaches the (1 - q_hat) coverage mass.
      The top class (rank 1) is therefore always included, and intermediate
      set sizes occur whenever the desired coverage mass falls strictly
      between two classes' cumulative probabilities.
    - "randomized_aps": the RANDOMIZED APS construction (Romano, Sesia &
      Candès, 2020, arXiv:2006.02544). Requires `randomization_u` (a single
      uniform draw shared by the calibration score for this row and this
      prediction-set computation; see predict_calibrated_rows()'s
      `randomization_seed`). A class is appended to the prediction set while
      `1 - cumsum + randomization_u * probabilities[idx] >= q_hat` holds; the
      first class for which this fails is NOT appended and the loop breaks
      (unlike "corrected_aps", which always appends the crossing class).
      This removes the +4.0pt over-coverage the non-randomized "corrected_aps"
      construction exhibited (Iter72 investigation), by letting the
      threshold-crossing class be included only with probability
      proportional to how much of its probability mass is needed to reach
      the coverage mass, instead of being included wholesale.

    qhat_quantile_direction controls which side of the nonconformity score
    population q_hat is drawn from, per the finite-sample-corrected quantile
    definitions in Angelopoulos & Bates (2021, arXiv:2107.07511) and
    Barber et al. (2021, "Predictive Inference with the Jackknife+", Annals
    of Statistics), which for a score population v of size n define:
        q-hat+_{n,alpha}{v} = ceil((n+1)(1-alpha)) / n   (upper)
        q-hat-_{n,alpha}{v} = floor((n+1)*alpha) / n = -q-hat+_{n,alpha}{-v}  (lower)
    - "upper" (default): the (1-alpha) quantile, i.e. q-hat+. This repo's
      score S(x, j) = 1 - cumulative_prob_up_to_j is the COMPLEMENT of the
      standard APS score (cumulative_prob_up_to_j itself), so taking the
      upper quantile of this complement corresponds to the LOWER quantile of
      the standard score -- the wrong side (Iter71 investigation Q1/Q2),
      producing under-coverage. Kept as the default to preserve prior runs'
      byte-for-byte output.
    - "alpha_lower": the alpha quantile, i.e. q-hat-. Because Y = 1 - X maps
      X's (1-alpha) quantile to Y's alpha quantile, this is the correct side
      to draw q_hat from when the score is the complement (1 - cumsum), as
      implemented in this repo.

    randomization_u is the single uniform draw required by
    set_construction="randomized_aps" (ignored otherwise). It must be the
    SAME value used to compute this row's calibration true-class score
    (predict_calibrated_rows()'s randomization_seed-derived u), so that the
    calibration and evaluation halves apply the identical randomized rule
    (Romano et al., 2020) rather than the mismatched u=0/u=1 combination the
    non-randomized "corrected_aps" construction implicitly uses (Iter73
    investigation Q2).

    Returns (list of class indices in prediction set, set size).
    """
    if qhat_source not in ("all", "true_class"):
        raise ValueError(f"qhat_source must be 'all' or 'true_class', got {qhat_source!r}")
    if set_construction not in ("broken", "corrected_aps", "randomized_aps"):
        raise ValueError(
            f"set_construction must be 'broken', 'corrected_aps', or 'randomized_aps', "
            f"got {set_construction!r}"
        )
    if set_construction == "randomized_aps" and randomization_u is None:
        raise ValueError(
            "set_construction='randomized_aps' requires randomization_u (a uniform "
            "draw shared with the calibration score); refusing to silently fall back "
            "to a non-randomized rule."
        )
    if qhat_quantile_direction not in ("upper", "alpha_lower"):
        raise ValueError(
            f"qhat_quantile_direction must be 'upper' or 'alpha_lower', "
            f"got {qhat_quantile_direction!r}"
        )

    alpha = 1.0 - confidence_level
    if qhat_source == "true_class":
        flat_scores = cp_data["true_class_scores"]  # shape=(n_cal,)
    else:
        all_scores = cp_data["all_scores"]  # shape=(n_cal, n_classes)
        flat_scores = all_scores.flatten()

    # q_hat = finite-sample-corrected quantile of the selected nonconformity
    # score population; direction controlled by qhat_quantile_direction (see
    # docstring above for the upper/alpha_lower definitions and rationale).
    if qhat_quantile_direction == "alpha_lower":
        target = alpha * (1.0 + 1.0 / len(flat_scores))
        q_hat = float(np.quantile(flat_scores, target, method="lower"))
    else:  # "upper"
        target = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(flat_scores)))
        q_hat = float(np.quantile(flat_scores, target, method="higher"))

    # Score for class j = 1 - cumsum_prob_up_to_j (monotonically decreasing).
    sorted_indices = np.argsort(-probabilities)  # descending
    pred_set: list[int] = []
    cumsum = 0.0
    if set_construction == "corrected_aps":
        for idx in sorted_indices:
            cumsum += probabilities[idx]
            pred_set.append(int(idx))  # append BEFORE the break check (Iter70 fix)
            score = 1.0 - cumsum
            if score <= q_hat:
                break
    elif set_construction == "randomized_aps":
        # Randomized APS (Romano et al., 2020): class idx is appended only
        # while `1 - cumsum + u*probabilities[idx] >= q_hat` holds (cumsum
        # updated INCLUSIVE of idx before the check, same update order as
        # "corrected_aps"); the first class for which this fails is NOT
        # appended and the loop breaks there. At randomization_u=1.0 this is
        # algebraically identical to "corrected_aps" for any input (both
        # reduce to "1 - cumsum_before_idx >= q_hat"); at randomization_u=0.0
        # it matches the non-randomized true-class calibration score
        # (1 - cumsum, no u term), so u interpolates between the two.
        for idx in sorted_indices:
            cumsum += probabilities[idx]
            score = 1.0 - cumsum + randomization_u * probabilities[idx]
            if score >= q_hat:
                pred_set.append(int(idx))
            else:
                break
    else:  # "broken": pre-Iter70 behavior, kept verbatim for reproducibility
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
    qhat_source: str = "all",
    set_construction: str = "broken",
    qhat_quantile_direction: str = "upper",
    calibration_source: str = "oof_train",
    holdout_seed: int = 42,
    randomization_seed: int = 42,
) -> list[dict]:
    """Recompute (selected_domain, confidence) for every dataset row via the calibrated classifier.

    If fine_tuned_embed_model is provided, uses a local SentenceTransformer
    instead of the Ollama client for embedding generation.
    Sequential (not concurrent) embedding calls, matching
    train_domain_classifier.py's build_training_features and
    fit_embedding_whitening.py's existing pattern for single-node
    offline embedding jobs.

    calibration_source controls where the conformal calibration nonconformity
    scores come from (Iter72 investigation):
    - "oof_train" (default): calibration scores are computed from
      `calibration_dataset_path` (classifier_train.jsonl) via a freshly
      refit 5-fold OOF LogisticRegression -- the pre-Iter72 implementation,
      kept verbatim below for byte-for-byte reproducibility.
    - "eval_holdout": calibration scores instead come from a stratified 50/50
      split of `dataset` itself, scored by the SAME already-fitted
      `classifier.predict_proba` used for the evaluation half. This removes
      the oof_train path's exchangeability violation (calibration scores from
      a different, unfitted model than the evaluation scores; Iter72
      investigation Q1) at the cost of halving the evaluation population that
      contributes to the final coverage/mean_set_size aggregates.

    randomization_seed (Iter73 investigation) seeds the per-dataset-row
    uniform draws `u_all = np.random.default_rng(randomization_seed).random(
    len(dataset))` used by set_construction="randomized_aps". The SAME u_all
    (indexed by dataset row position) is used for both the calibration
    true-class score (only for calibration_source="eval_holdout", the sole
    supported combination -- see the ValueError below) and the evaluation
    row's prediction-set construction, per Romano et al. (2020)'s requirement
    that calibration and test share the same u. Ignored for any other
    set_construction value.
    """
    from sentence_transformers import SentenceTransformer

    classes = list(classifier.classes_)
    rows = []

    if calibration_source not in ("oof_train", "eval_holdout"):
        raise ValueError(
            f"calibration_source must be 'oof_train' or 'eval_holdout', "
            f"got {calibration_source!r}"
        )
    if set_construction == "randomized_aps" and calibration_source != "eval_holdout":
        raise ValueError(
            "set_construction='randomized_aps' requires calibration_source="
            "'eval_holdout': the calibration and evaluation halves must share "
            "dataset row indices so the same per-row uniform draw u can be applied "
            "on both sides (Iter73 plan); 'oof_train' calibrates over a separate "
            "dataset (classifier_train.jsonl) with no such shared indexing."
        )

    # Pre-compute conformal prediction calibration data (APS method).
    #
    # Non-conformity score for class j (sorted by prob descending):
    #   S(x, j) = 1 - cumulative_prob_up_to_class_j
    # The top class gets the HIGHEST score (least conforming), so we use the
    # alpha-quantile of TRUE-CLASS scores as q_hat. This ensures the top class
    # is included in the prediction set when its score <= q_hat.
    cp_data: dict | None = None
    holdout_split: dict[int, str] | None = None  # dataset row index -> "cal" | "eval"
    precomputed_eval_holdout: dict | None = None  # set below only for calibration_source="eval_holdout"
    u_all: np.ndarray | None = None  # set below only for set_construction="randomized_aps"
    if conformal_prediction and calibration_source == "oof_train":
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

        # True-class-only nonconformity scores (n_cal values), used by the
        # standard APS calibration procedure (qhat_source="true_class").
        # Iter69: the pre-existing "all" population (n_cal * n_classes scores)
        # was found to under-cover its nominal confidence level.
        true_class_scores = np.array([all_scores[i, labels[i]] for i in range(n_cal)])

        cp_data = {"all_scores": all_scores, "true_class_scores": true_class_scores}
    elif conformal_prediction:  # calibration_source == "eval_holdout"
        if qhat_source == "all":
            raise ValueError(
                "calibration_source='eval_holdout' only supports qhat_source="
                "'true_class' (the standard APS calibration population); "
                "'all' is rejected rather than silently ignored (Iter72 plan step 3)."
            )
        from sklearn.model_selection import StratifiedShuffleSplit

        n_eval = len(dataset)
        n_classes = len(classes)
        eval_labels = [
            classes.index(row["expected_domains"][0])
            if row["expected_domains"][0] in classes
            else 0
            for row in dataset
        ]

        # Score ALL 1,600 rows with the SAME already-fitted classifier used
        # for evaluation, so calibration and evaluation scores come from one
        # fixed model that never saw the calibration half during fitting
        # (Iter72 investigation Q2). No calibration-set-only embeddings need
        # to be computed here; the evaluation loop below recomputes each
        # row's embedding once, and we reuse those below to avoid a second
        # embedding pass.
        all_embeddings = []
        if fine_tuned_embed_model is not None:
            local_model = SentenceTransformer(
                fine_tuned_embed_model, trust_remote_code=True, device="cpu"
            )
            try:
                local_model.load_adapter(fine_tuned_embed_model, "default")
                local_model.set_adapter("default")
            except ValueError:
                pass
            for row in dataset:
                all_embeddings.append(
                    local_model.encode(row["query"], normalize_embeddings=True,
                                       show_progress_bar=False)
                )
        else:
            for row in dataset:
                all_embeddings.append(
                    await ollama_client.embed(embedding_model, row["query"])
                )
        all_embeddings = np.array(all_embeddings)
        all_probs = classifier.predict_proba(all_embeddings)

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=holdout_seed
        )
        # First returned index array is the calibration half, second is the
        # evaluation half (Iter72 plan: reversing this changes the seed=42
        # prediction from coverage=0.9400 to the cross-fit value 0.9487).
        cal_idx, eval_idx = next(splitter.split(np.zeros(n_eval), eval_labels))
        holdout_split = {}
        for i in cal_idx:
            holdout_split[int(i)] = "cal"
        for i in eval_idx:
            holdout_split[int(i)] = "eval"

        # For set_construction="randomized_aps" (Iter73), draw one uniform
        # value per dataset row (dataset order, not cal/eval order) so the
        # SAME u is used for a row's calibration true-class score below and
        # its evaluation prediction-set construction further down (Romano et
        # al., 2020's requirement that calibration and test share u).
        if set_construction == "randomized_aps":
            u_all = np.random.default_rng(randomization_seed).random(n_eval)

        true_class_scores = np.zeros(len(cal_idx))
        for pos, i in enumerate(cal_idx):
            probs = all_probs[i]
            sorted_idx = np.argsort(-probs)  # descending
            cumsum = 0.0
            for idx in sorted_idx:
                cumsum += probs[idx]
                if idx == eval_labels[i]:
                    if set_construction == "randomized_aps":
                        true_class_scores[pos] = 1.0 - cumsum + u_all[i] * probs[idx]
                    else:
                        true_class_scores[pos] = 1.0 - cumsum
                    break

        n_cal = len(cal_idx)
        cp_data = {
            # qhat_source="all" is rejected above, so all_scores is never read;
            # kept as an empty-shaped placeholder purely so any future code
            # path that accesses cp_data["all_scores"] fails loudly instead of
            # KeyError-ing silently.
            "all_scores": np.zeros((0, n_classes)),
            "true_class_scores": true_class_scores,
        }

        # Cache the already-computed embeddings/probabilities so the main
        # per-row loop below (fine_tuned or ollama branch) can reuse them
        # instead of recomputing an embedding for each of the same 1,600
        # queries a second time.
        precomputed_eval_holdout = {
            "embeddings": all_embeddings,
            "probabilities": all_probs,
        }

    if conformal_prediction:
        # Print which q_hat population is actually used, and its resulting
        # value, so a mis-wired qhat_source/calibration_source is visible
        # instead of silently falling back to the default (Iter69: past
        # no-op failures were only caught by an independent post-hoc
        # recomputation).
        alpha = 1.0 - confidence_level
        if qhat_source == "true_class":
            _diag_scores = cp_data["true_class_scores"]
        else:
            _diag_scores = cp_data["all_scores"].flatten()
        if qhat_quantile_direction == "alpha_lower":
            _diag_target = alpha * (1.0 + 1.0 / len(_diag_scores))
            _diag_q_hat = float(np.quantile(_diag_scores, _diag_target, method="lower"))
        else:  # "upper"
            _diag_target = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / len(_diag_scores)))
            _diag_q_hat = float(np.quantile(_diag_scores, _diag_target, method="higher"))
        print(
            f"[evaluate_classifier_calibration] calibration_source={calibration_source} "
            f"qhat_source={qhat_source} "
            f"q_hat={_diag_q_hat:.4f} n_cal={len(cp_data['true_class_scores'])} "
            f"set_construction={set_construction} "
            f"qhat_quantile_direction={qhat_quantile_direction} "
            f"randomization_seed={randomization_seed}",
            file=sys.stderr,
        )

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
        for row_idx, row in enumerate(dataset):
            if precomputed_eval_holdout is not None:
                probabilities = precomputed_eval_holdout["probabilities"][row_idx].copy()
            else:
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
                    probabilities, cp_data, confidence_level,
                    qhat_source=qhat_source, set_construction=set_construction,
                    qhat_quantile_direction=qhat_quantile_direction,
                    randomization_u=(u_all[row_idx] if u_all is not None else None),
                )
            best_index = max(range(len(classes)), key=lambda i: probabilities[i])
            row_dict = {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": classes[best_index],
                "confidence": float(probabilities[best_index]),
                "probabilities": {domain: float(p) for domain, p in zip(classes, probabilities)},
            }
            if holdout_split is not None:
                row_dict["split"] = holdout_split[row_idx]
            if conformal_prediction:
                row_dict["prediction_set"] = [classes[i] for i in pred_set]
                row_dict["set_size"] = set_size
            rows.append(row_dict)
    else:
        for row_idx, row in enumerate(dataset):
            if precomputed_eval_holdout is not None:
                probabilities = precomputed_eval_holdout["probabilities"][row_idx].copy()
            else:
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
                    probabilities, cp_data, confidence_level,
                    qhat_source=qhat_source, set_construction=set_construction,
                    qhat_quantile_direction=qhat_quantile_direction,
                    randomization_u=(u_all[row_idx] if u_all is not None else None),
                )
            best_index = max(range(len(classes)), key=lambda i: probabilities[i])
            row_dict = {
                "id": row["id"],
                "expected_domains": row["expected_domains"],
                "selected_domain": classes[best_index],
                "confidence": float(probabilities[best_index]),
                "probabilities": {domain: float(p) for domain, p in zip(classes, probabilities)},
            }
            if holdout_split is not None:
                row_dict["split"] = holdout_split[row_idx]
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
    qhat_source: str = "all",
    set_construction: str = "broken",
    qhat_quantile_direction: str = "upper",
    calibration_source: str = "oof_train",
    holdout_seed: int = 42,
    randomization_seed: int = 42,
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
        qhat_source=qhat_source,
        set_construction=set_construction,
        qhat_quantile_direction=qhat_quantile_direction,
        calibration_source=calibration_source,
        holdout_seed=holdout_seed,
        randomization_seed=randomization_seed,
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
        "--qhat-source",
        choices=["all", "true_class"],
        default="all",
        help="Nonconformity score population used to compute q_hat: 'all' (n_cal*n_classes "
             "scores, the original implementation) or 'true_class' (n_cal scores, the "
             "standard APS calibration procedure). Default 'all' preserves prior behavior.",
    )
    parser.add_argument(
        "--set-construction",
        choices=["broken", "corrected_aps", "randomized_aps"],
        default="broken",
        help="Prediction set construction rule: 'broken' (pre-Iter70 implementation, "
             "append happens after the break check and collapses to a binary "
             "1-p_max<=q_hat gate), 'corrected_aps' (standard non-randomized APS: "
             "append happens before the break check, so classes are greedily added "
             "until cumulative probability reaches the coverage mass), or "
             "'randomized_aps' (Romano et al., 2020: same as corrected_aps but mixes "
             "in a per-row uniform draw u so the crossing class is included only "
             "fractionally, removing corrected_aps's structural over-coverage; "
             "requires --calibration-source eval_holdout, see --randomization-seed). "
             "Default 'broken' preserves prior behavior.",
    )
    parser.add_argument(
        "--qhat-quantile-direction",
        choices=["upper", "alpha_lower"],
        default="upper",
        help="Which side of the nonconformity score population q_hat is drawn from: "
             "'upper' (the (1-alpha) quantile, the original implementation) or "
             "'alpha_lower' (the alpha quantile, correct for this repo's complement "
             "score S=1-cumsum per Iter71 investigation). Default 'upper' preserves "
             "prior behavior.",
    )
    parser.add_argument(
        "--calibration-source",
        choices=["oof_train", "eval_holdout"],
        default="oof_train",
        help="Where conformal calibration nonconformity scores come from: "
             "'oof_train' (the original implementation: a fresh 5-fold OOF refit of "
             "the base LogisticRegression over data/classifier_train.jsonl) or "
             "'eval_holdout' (a stratified 50/50 split of --dataset itself, scored by "
             "the SAME already-fitted classifier used for evaluation, restoring "
             "calibration/evaluation score exchangeability per Iter72 investigation). "
             "Default 'oof_train' preserves prior behavior.",
    )
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=42,
        help="random_state for the --calibration-source=eval_holdout stratified 50/50 "
             "split (ignored for 'oof_train').",
    )
    parser.add_argument(
        "--randomization-seed",
        type=int,
        default=42,
        help="Seed for the per-dataset-row uniform draws used by "
             "--set-construction randomized_aps (ignored otherwise).",
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
                qhat_source=args.qhat_source,
                set_construction=args.set_construction,
                qhat_quantile_direction=args.qhat_quantile_direction,
                calibration_source=args.calibration_source,
                holdout_seed=args.holdout_seed,
                randomization_seed=args.randomization_seed,
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
                    qhat_source=args.qhat_source,
                    set_construction=args.set_construction,
                    qhat_quantile_direction=args.qhat_quantile_direction,
                    calibration_source=args.calibration_source,
                    holdout_seed=args.holdout_seed,
                    randomization_seed=args.randomization_seed,
                )
            )


if __name__ == "__main__":
    main()

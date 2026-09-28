"""Iter97/Iter98 pre-registered screening (run on wafl-ctrl5, absolute condition (B)):
fixed stratified-5-fold CV comparison of the current single-softmax
LogisticRegression base estimator ("arm A") against two decomposed
base estimators: one-vs-one pairwise ("arm B", 45 pairwise binary
LogisticRegressions via sklearn.multiclass.OneVsOneClassifier, Iter97,
already run and refuted -- not re-run here, its recorded values are only
carried as constants for side-by-side reporting) and error-correcting
output codes ("arm C", ecoc_head.EcocLogLossClassifier, Iter98), gating
whether the full experiment (real routing run on wafl500-509) is worth
180 minutes of node time (config.yml levers
`classifier_multiclass_decomposition` note, journal.md "Iteration 97" /
"Iteration 98").

Design (pre-registered in journal.md "Iteration 98"):
  - All three arms would train on the *same* existing 2,275-row training
    set (data/classifier_train_iter94_dedup.jsonl, embedded once in
    data/embcache_iter96_scaled.npy's first 2,275 rows -- reused unchanged
    from Iter96, zero new embedding calls). Arm C is the only arm actually
    (re-)trained by this run; arm B's numbers below are Iter97's recorded
    measurements on the identical rows/folds, reproduced here only as
    literal constants (journal.md "Iteration 97" result table), not
    recomputed, per B163(f)(5)'s "re-run arm A and add arm C, don't
    re-run arm B" instruction.
  - A fixed 5-fold StratifiedKFold split on domain labels, 3 seeds
    (96/196/296), matched/paired between arms A and C (same train/test
    row indices in every (seed, fold)), reusing
    scripts/screen_composition_preserving_volume_expansion.py's structure.
  - Arm A calls scripts/train_domain_classifier.py's train_classifier()
    unchanged (single softmax LogisticRegression base estimator).
  - Arm C calls _train_ecoc_classifier() below: CalibratedClassifierCV
    wrapping ecoc_head.EcocLogLossClassifier (dense random code, L=34,
    random_state=98, log-loss/maximum-likelihood decoding -- all
    pre-registered in journal.md "Iteration 98", not swept). Unlike arm
    B's OneVsOneClassifier, EcocLogLossClassifier declares `sample_weight`
    as an explicit fit() parameter, so sklearn's has_fit_parameter check
    finds it without sklearn.config_context(enable_metadata_routing=True)
    -- but every fit() call is still wrapped in the same warning guard as
    arm B, since a silent sample_weight drop would invalidate the
    experiment either way (journal.md "Iteration 98" (4), B164(a)).
  - Statistics (McNemar for overall accuracy and per-domain recall,
    Benjamini-Hochberg for the per-domain multiple-comparison correction)
    are computed by calling metrics.py's existing implementations on
    out-of-fold predictions reshaped into {id, selected_domain,
    expected_domains} rows -- not reimplemented here. A conservative
    variant of the seed-merged McNemar test (dividing each discordant
    count by the number of seeds before re-running metrics.py's own
    continuity-corrected chi-square/p-value arithmetic, since naively
    merging 3 seeds' folds triple-counts each of the 2,275 underlying
    rows -- B163(b), journal.md "Iteration 97" learning 4) is reported
    alongside the naive seed-merged value for every gate-relevant
    statistic; only the naive (seed-merged) value is used for the actual
    gate decision, per journal.md "Iteration 98"'s screening design.
  - Diagnostics (not used for the gate, but always reported, per
    B163(f)(4) / journal.md "Iteration 98" (4)): 10x10 confusion
    matrices for arms A and C, arm C's code matrix (shape, minimum/mean
    pairwise row Hamming distance), each of arm C's 34 columns' own
    binary CV accuracy (a direct test of whether the column's induced
    metaclass split is learnable at n=2,275/p=5,120, addressing P5),
    ECE/Brier for both arms, the top-2 predict_proba gap distribution for
    both arms, and arm C's *uncalibrated* decision_function's top-1 minus
    top-2 score distribution (the continuous analogue of arm B's discrete
    vote-count gap, journal.md "Iteration 98" P3).

Usage (on wafl-ctrl5):
    uv run python scripts/screen_classifier_multiclass_decomposition.py \\
        --train-embcache data/embcache_iter96_scaled.npy \\
        --n-existing 2275 \\
        --output /tmp/iter98_screening.json
"""

import argparse
import json
import sys
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from ecoc_head import EcocLogLossClassifier, build_dense_random_code
from scripts.train_domain_classifier import (
    _CALIBRATION_CV,
    _CALIBRATION_METHOD,
    _MAX_ITER,
    _extract_sample_weights,
    train_classifier,
)
import metrics

_SEEDS = [96, 196, 296]
_N_FOLDS = 5
_DISPATCH_GAP_THRESHOLD = 0.36  # config.yaml's current value; read-only diagnostic, never changed here.
_ECOC_RANDOM_STATE = 98  # journal.md "Iteration 98": fixed to the iteration number, not swept.
_DOMAIN_CV_FOLDS = 5  # for the per-column binary CV accuracy diagnostic (P5), separate from the main loop.
_DOMAIN_CV_SEED = 98

# Iter97's recorded measurements (journal.md "Iteration 97" result table / "Iteration 97
# executed" section), reproduced here as literal constants for side-by-side reporting
# only -- arm B is *not* re-run by this script (B163(f)(5)).
_ITER97_ARM_B_RECORDED = {
    "cv_accuracy_b_ovo": 0.794579,
    "cv_delta_pt_b_minus_a": -0.806,
    "overall_mcnemar_seed_merged": {"discordant_a_only": 138, "discordant_b_only": 83, "chi2": 13.19, "p_value": 0.00028},
    "overall_mcnemar_conservative": {"discordant_a_only": 46, "discordant_b_only": 28, "chi2": 4.56, "p_value": 0.033},
    "n_bh_significant_regressions": 2,
    "per_domain_recall_delta_pt": {"history_culture": -2.86, "social_science": -4.53, "medical": -1.87, "education": 2.24, "legal": 0.0},
    "gap_lt_threshold_fraction_a": 0.201,
    "gap_lt_threshold_fraction_b": 0.083,
}


class SampleWeightDroppedError(RuntimeError):
    """Raised when sklearn emits any UserWarning during a fit() call, signalling that
    sample_weight was silently dropped for a base estimator (the pre-registered
    "1 warning = experiment invalid" rule, journal.md "Iteration 97", reused unchanged
    for arm C in "Iteration 98")."""


def _fit_with_warning_guard(model: CalibratedClassifierCV, embeddings, labels, sample_weight) -> None:
    """Call model.fit(), promoting any warning raised during the call to a hard error.

    Uses warnings.catch_warnings(record=True) rather than simplefilter("error")
    directly, so the offending warning's full message is captured and reported
    before raising, instead of only sklearn's default terse traceback.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(embeddings, labels, sample_weight=sample_weight)
    if caught:
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        raise SampleWeightDroppedError(
            "fit() raised warning(s), sample_weight may have been silently dropped: "
            + " | ".join(messages)
        )


def _train_ecoc_classifier(
    embeddings: list[list[float]], labels: list[str], sample_weight: list[float]
) -> CalibratedClassifierCV:
    """Arm C: train_classifier() with only the base estimator swapped to
    ecoc_head.EcocLogLossClassifier (dense random code, L=34,
    random_state=98, log-loss/maximum-likelihood decoding). Calibration
    method/cv, max_iter, and sample_weight are otherwise identical to
    train_classifier()'s arm A.

    EcocLogLossClassifier declares sample_weight as an explicit fit()
    parameter (unlike OneVsOneClassifier), so sklearn's has_fit_parameter
    check finds it without sklearn.config_context(enable_metadata_routing=
    True) -- journal.md "Iteration 98" (4). _fit_with_warning_guard() is
    still applied, so any unexpected warning (silent drop or otherwise)
    aborts the run rather than silently producing an invalid comparison.
    """
    base_estimator = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)
    ecoc_estimator = EcocLogLossClassifier(estimator=base_estimator, random_state=_ECOC_RANDOM_STATE)
    calibrated_model = CalibratedClassifierCV(
        ecoc_estimator, method=_CALIBRATION_METHOD, cv=_CALIBRATION_CV, ensemble=True
    )
    _fit_with_warning_guard(calibrated_model, embeddings, labels, sample_weight)
    return calibrated_model


def _rows_for_metrics(ids: list[str], true_domains: list[str], pred_domains: list[str]) -> list[dict]:
    """Build {id, selected_domain, expected_domains} rows matching metrics.py's schema."""
    return [
        {"id": i, "selected_domain": pred, "expected_domains": [true]}
        for i, true, pred in zip(ids, true_domains, pred_domains)
    ]


def _rows_for_ece_brier(
    ids: list[str], true_domains: list[str], pred_domains: list[str], confidences: list[float]
) -> list[dict]:
    """Build {id, selected_domain, expected_domains, confidence} rows for metrics.py's
    compute_ece/compute_brier_score, which additionally require a `confidence` field
    (the row's max predict_proba value, matching classifier.py's usage)."""
    return [
        {"id": i, "selected_domain": pred, "expected_domains": [true], "confidence": conf}
        for i, true, pred, conf in zip(ids, true_domains, pred_domains, confidences)
    ]


def _top2_gap_fraction_below_threshold(proba: np.ndarray, threshold: float) -> float:
    """Fraction of rows whose top-1 minus top-2 predict_proba is below `threshold`
    (config.yaml's dispatch_gap_threshold), a read-only diagnostic of how many rows
    would newly qualify for multi-domain dispatch under this arm's probability shape."""
    sorted_proba = np.sort(proba, axis=1)
    gap = sorted_proba[:, -1] - sorted_proba[:, -2]
    return float(np.mean(gap < threshold))


def _confusion_matrix(domains: list[str], true_domains: list[str], pred_domains: list[str]) -> dict:
    """10x10 confusion matrix (rows=true, cols=predicted) as a nested dict, for the
    always-on diagnostic output required by B163(f)(4)."""
    matrix = {true_d: {pred_d: 0 for pred_d in domains} for true_d in domains}
    for true_d, pred_d in zip(true_domains, pred_domains):
        matrix[true_d][pred_d] += 1
    return matrix


def _conservative_mcnemar(discordant_a_only: int, discordant_b_only: int, n_seeds: int) -> dict:
    """Re-run metrics.py's own continuity-corrected McNemar chi-square/p-value
    arithmetic (metrics._mcnemar_from_correctness) on discordant counts divided by
    the number of seeds, instead of the naive seed-merged counts.

    Naively merging `n_seeds` seeds' 5-fold out-of-fold predictions counts each of
    the underlying 2,275 rows `n_seeds` times, which is anti-conservative (B163(b),
    journal.md "Iteration 97" learning 4). This does not re-derive the chi-square/
    p-value formula: it builds a synthetic same-shape correctness map (one True/False
    pair of ids per discordant count, rounded to the nearest integer since dividing by
    n_seeds need not be exact) and calls metrics.py's existing private helper on it,
    so the arithmetic itself is metrics.py's.
    """
    adjusted_a_only = round(discordant_a_only / n_seeds)
    adjusted_b_only = round(discordant_b_only / n_seeds)
    correct_a: dict[str, bool] = {}
    correct_b: dict[str, bool] = {}
    row_index = 0
    for _ in range(adjusted_a_only):
        correct_a[f"syn{row_index}"], correct_b[f"syn{row_index}"] = True, False
        row_index += 1
    for _ in range(adjusted_b_only):
        correct_a[f"syn{row_index}"], correct_b[f"syn{row_index}"] = False, True
        row_index += 1
    result = metrics._mcnemar_from_correctness(correct_a, correct_b)
    result["adjusted_discordant_a_only"] = adjusted_a_only
    result["adjusted_discordant_b_only"] = adjusted_b_only
    return result


def _ecoc_column_diagnostics(
    embeddings: np.ndarray, domains: np.ndarray
) -> dict:
    """P5's direct test: each of the 34 code-matrix columns' own binary CV accuracy on
    the full n=2,275 training set, plus the code matrix's shape and pairwise row
    Hamming distances. A single 5-fold split (not the main 3-seed loop) is used since
    this diagnoses whether the metaclass split itself generalizes at n=2,275/
    p=5,120, independent of which main-loop fold a row happened to land in.
    """
    rng = np.random.default_rng(_ECOC_RANDOM_STATE)
    classes = np.unique(domains)
    code_book = build_dense_random_code(len(classes), rng)
    class_to_index = {c: i for i, c in enumerate(classes)}
    y_index = np.asarray([class_to_index[d] for d in domains])

    pairwise_distance = (code_book[:, None, :] != code_book[None, :, :]).sum(axis=2).astype(float)
    np.fill_diagonal(pairwise_distance, np.inf)
    min_row_hamming_distance = float(pairwise_distance.min())
    pairwise_distance_finite = pairwise_distance[np.isfinite(pairwise_distance)]
    mean_row_hamming_distance = float(pairwise_distance_finite.mean())

    skf = StratifiedKFold(n_splits=_DOMAIN_CV_FOLDS, shuffle=True, random_state=_DOMAIN_CV_SEED)
    n_columns = code_book.shape[1]
    column_correct = np.zeros(n_columns)
    column_total = np.zeros(n_columns)
    for train_idx, test_idx in skf.split(embeddings, domains):
        train_emb, test_emb = embeddings[train_idx], embeddings[test_idx]
        train_y_index, test_y_index = y_index[train_idx], y_index[test_idx]
        for column in range(n_columns):
            train_bit = code_book[train_y_index, column]
            test_bit = code_book[test_y_index, column]
            column_model = LogisticRegression(max_iter=_MAX_ITER, class_weight=None)
            column_model.fit(train_emb, train_bit)
            column_correct[column] += (column_model.predict(test_emb) == test_bit).sum()
            column_total[column] += len(test_idx)
    column_cv_accuracy = (column_correct / column_total).tolist()

    return {
        "code_book_shape": list(code_book.shape),
        "min_row_hamming_distance": min_row_hamming_distance,
        "mean_row_hamming_distance": mean_row_hamming_distance,
        "column_cv_accuracy": column_cv_accuracy,
        "column_cv_accuracy_median": float(np.median(column_cv_accuracy)),
        "column_cv_accuracy_min": float(np.min(column_cv_accuracy)),
    }


def run_screening(train_embcache_path: str, n_existing: int) -> dict:
    """Run the 3-seed x 5-fold screening (arms A and C); return a JSON-serializable summary dict."""
    embeddings = np.load(train_embcache_path)
    with open(train_embcache_path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["ids"][:n_existing]
    domains = meta["domains"][:n_existing]
    assert len(ids) == n_existing, "embcache metadata must have at least n_existing rows"
    existing_emb_arr = embeddings[:n_existing]
    existing_ids_arr = np.array(ids)
    existing_domains_arr = np.array(domains)
    all_domains = sorted(set(domains))

    oof_a: list[dict] = []
    oof_c: list[dict] = []
    proba_row_sums_a: list[float] = []
    proba_row_sums_c: list[float] = []
    gap_fractions_a: list[float] = []
    gap_fractions_c: list[float] = []
    ece_rows_a: list[dict] = []
    ece_rows_c: list[dict] = []
    raw_decode_top1_minus_top2: list[float] = []

    for seed in _SEEDS:
        skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(existing_emb_arr, existing_domains_arr):
            test_ids = [f"{existing_ids_arr[i]}_s{seed}" for i in test_idx]
            test_true = list(existing_domains_arr[test_idx])
            test_emb = existing_emb_arr[test_idx].tolist()

            train_rows = [{"domain": existing_domains_arr[i]} for i in train_idx]
            weight = _extract_sample_weights(train_rows)
            train_emb = existing_emb_arr[train_idx].tolist()
            train_labels = list(existing_domains_arr[train_idx])

            # Arm A: unmodified train_classifier() (single softmax LogisticRegression).
            model_a = train_classifier(train_emb, train_labels, sample_weight=weight)
            proba_a = model_a.predict_proba(test_emb)
            pred_a_test = model_a.classes_[np.argmax(proba_a, axis=1)]
            oof_a.extend(_rows_for_metrics(test_ids, test_true, list(pred_a_test)))
            proba_row_sums_a.extend(proba_a.sum(axis=1).tolist())
            gap_fractions_a.append(_top2_gap_fraction_below_threshold(proba_a, _DISPATCH_GAP_THRESHOLD))
            ece_rows_a.extend(
                _rows_for_ece_brier(test_ids, test_true, list(pred_a_test), proba_a.max(axis=1).tolist())
            )

            # Arm C: EcocLogLossClassifier base estimator (dense random code, L=34, log-loss decoding).
            model_c = _train_ecoc_classifier(train_emb, train_labels, weight)
            proba_c = model_c.predict_proba(test_emb)
            pred_c_test = model_c.classes_[np.argmax(proba_c, axis=1)]
            oof_c.extend(_rows_for_metrics(test_ids, test_true, list(pred_c_test)))
            proba_row_sums_c.extend(proba_c.sum(axis=1).tolist())
            gap_fractions_c.append(_top2_gap_fraction_below_threshold(proba_c, _DISPATCH_GAP_THRESHOLD))
            ece_rows_c.extend(
                _rows_for_ece_brier(test_ids, test_true, list(pred_c_test), proba_c.max(axis=1).tolist())
            )

            # Uncalibrated decode-score gap (P3's continuous-score diagnostic): a
            # separate, un-calibrated EcocLogLossClassifier fit on the same fold so
            # the raw log-loss decoding scores (before temperature scaling) can be
            # inspected directly.
            raw_ecoc = EcocLogLossClassifier(
                estimator=LogisticRegression(max_iter=_MAX_ITER, class_weight=None),
                random_state=_ECOC_RANDOM_STATE,
            )
            raw_ecoc.fit(train_emb, train_labels, sample_weight=weight)
            raw_scores = raw_ecoc.decision_function(test_emb)
            sorted_scores = np.sort(raw_scores, axis=1)
            raw_decode_top1_minus_top2.extend((sorted_scores[:, -1] - sorted_scores[:, -2]).tolist())

            print(
                f"[screen] seed={seed} fold done (train={len(train_idx)}, test={len(test_idx)})",
                file=sys.stderr,
            )

    cv_acc_a = metrics.compute_top1_accuracy(oof_a)
    cv_acc_c = metrics.compute_top1_accuracy(oof_c)
    overall_mcnemar = metrics.compute_mcnemar_test(oof_a, oof_c)
    overall_mcnemar_conservative = _conservative_mcnemar(
        overall_mcnemar["discordant_a_only"], overall_mcnemar["discordant_b_only"], len(_SEEDS)
    )

    per_domain = {}
    p_values = []
    for domain in all_domains:
        recall_test = metrics.compute_domain_recall_mcnemar_test(oof_a, oof_c, domain)
        p_values.append(recall_test["p_value"])
        per_domain[domain] = recall_test
    bh_significant = metrics.apply_benjamini_hochberg(p_values, q=0.05)
    for domain, sig in zip(all_domains, bh_significant):
        per_domain[domain]["bh_significant"] = sig
        conservative = _conservative_mcnemar(
            per_domain[domain]["discordant_a_only"], per_domain[domain]["discordant_b_only"], len(_SEEDS)
        )
        per_domain[domain]["conservative_p_value"] = conservative["p_value"]
        denom = sum(1 for r in oof_a if domain in r["expected_domains"])
        recall_a_value = sum(
            1 for r in oof_a if domain in r["expected_domains"] and r["selected_domain"] == domain
        ) / denom if denom else float("nan")
        recall_c_value = sum(
            1 for r in oof_c if domain in r["expected_domains"] and r["selected_domain"] == domain
        ) / denom if denom else float("nan")
        per_domain[domain]["recall_a"] = recall_a_value
        per_domain[domain]["recall_c"] = recall_c_value
        per_domain[domain]["recall_delta_pt"] = (recall_c_value - recall_a_value) * 100.0
    # BH is re-applied to the conservative p-values too, per B163(b)'s instruction
    # that footgate 2's pass/fail can flip between the naive and conservative view.
    conservative_p_values = [per_domain[d]["conservative_p_value"] for d in all_domains]
    conservative_bh_significant = metrics.apply_benjamini_hochberg(conservative_p_values, q=0.05)
    for domain, sig in zip(all_domains, conservative_bh_significant):
        per_domain[domain]["conservative_bh_significant"] = sig

    cv_delta_pt = (cv_acc_c - cv_acc_a) * 100.0
    n_bh_significant_regressions = sum(
        1 for d in all_domains if per_domain[d]["bh_significant"] and per_domain[d]["recall_delta_pt"] < 0
    )
    n_conservative_bh_significant_regressions = sum(
        1 for d in all_domains if per_domain[d]["conservative_bh_significant"] and per_domain[d]["recall_delta_pt"] < 0
    )

    # P1 (primary prediction, = gate 1): CV top1 delta (C-A) >= +1.0pt.
    p1_pass = cv_delta_pt >= 1.0

    # P2 (mechanism discriminator): sum of Iter97's two BH-significant regressed
    # domains' recall delta (social_science, history_culture) for arm C vs A.
    p2_domains = ["social_science", "history_culture"]
    p2_recall_delta_sum_pt = sum(per_domain[d]["recall_delta_pt"] for d in p2_domains)
    p2_pass = p2_recall_delta_sum_pt >= -1.0

    # P3: predict_proba row-sum sanity (both arms) + gap<threshold fraction within +-5pt.
    proba_sums_a_arr = np.array(proba_row_sums_a)
    proba_sums_c_arr = np.array(proba_row_sums_c)
    p3_proba_sums_ok_a = bool(np.allclose(proba_sums_a_arr, 1.0, atol=1e-6))
    p3_proba_sums_ok_c = bool(np.allclose(proba_sums_c_arr, 1.0, atol=1e-6))
    mean_gap_fraction_a = float(np.mean(gap_fractions_a))
    mean_gap_fraction_c = float(np.mean(gap_fractions_c))
    p3_gap_fraction_delta_pt = (mean_gap_fraction_c - mean_gap_fraction_a) * 100.0
    p3_pass = p3_proba_sums_ok_a and p3_proba_sums_ok_c and abs(p3_gap_fraction_delta_pt) <= 5.0

    # P4: arm C's ECE must not exceed arm A's ECE.
    ece_a = metrics.compute_ece(ece_rows_a)
    ece_c = metrics.compute_ece(ece_rows_c)
    brier_a = metrics.compute_brier_score(ece_rows_a)
    brier_c = metrics.compute_brier_score(ece_rows_c)
    p4_pass = ece_c["ece"] <= ece_a["ece"]

    # P5: 34 columns' binary CV accuracy median >= 0.70.
    column_diagnostics = _ecoc_column_diagnostics(existing_emb_arr, existing_domains_arr)
    p5_pass = column_diagnostics["column_cv_accuracy_median"] >= 0.70

    confusion_a = _confusion_matrix(all_domains, [r["expected_domains"][0] for r in oof_a], [r["selected_domain"] for r in oof_a])
    confusion_c = _confusion_matrix(all_domains, [r["expected_domains"][0] for r in oof_c], [r["selected_domain"] for r in oof_c])

    raw_decode_gap_arr = np.array(raw_decode_top1_minus_top2)

    return {
        "n_existing": n_existing,
        "seeds": _SEEDS,
        "n_folds": _N_FOLDS,
        "cv_accuracy_a_softmax": cv_acc_a,
        "cv_accuracy_c_ecoc": cv_acc_c,
        "cv_delta_pt_c_minus_a": cv_delta_pt,
        "overall_mcnemar_seed_merged": overall_mcnemar,
        "overall_mcnemar_conservative": overall_mcnemar_conservative,
        "per_domain": per_domain,
        "n_bh_significant_regressions_seed_merged": n_bh_significant_regressions,
        "n_bh_significant_regressions_conservative": n_conservative_bh_significant_regressions,
        "p1_pass_cv_delta_ge_1pt": p1_pass,
        "p2_domains": p2_domains,
        "p2_recall_delta_sum_pt": p2_recall_delta_sum_pt,
        "p2_pass_ge_minus_1pt": p2_pass,
        "p3_proba_sums_ok_a": p3_proba_sums_ok_a,
        "p3_proba_sums_ok_c": p3_proba_sums_ok_c,
        "p3_mean_gap_lt_threshold_fraction_a": mean_gap_fraction_a,
        "p3_mean_gap_lt_threshold_fraction_c": mean_gap_fraction_c,
        "p3_gap_fraction_delta_pt": p3_gap_fraction_delta_pt,
        "p3_pass": p3_pass,
        "p4_ece_a": ece_a,
        "p4_ece_c": ece_c,
        "p4_brier_a": brier_a,
        "p4_brier_c": brier_c,
        "p4_pass_ece_not_worse": p4_pass,
        "p5_column_diagnostics": column_diagnostics,
        "p5_pass_median_column_accuracy_ge_0_70": p5_pass,
        "confusion_matrix_a": confusion_a,
        "confusion_matrix_c": confusion_c,
        "raw_decode_top1_minus_top2_summary": {
            "min": float(raw_decode_gap_arr.min()),
            "max": float(raw_decode_gap_arr.max()),
            "mean": float(raw_decode_gap_arr.mean()),
            "median": float(np.median(raw_decode_gap_arr)),
            "p10": float(np.percentile(raw_decode_gap_arr, 10)),
            "n_exact_duplicate_values": int(len(raw_decode_gap_arr) - len(np.unique(raw_decode_gap_arr))),
            "n_rows": int(len(raw_decode_gap_arr)),
        },
        "iter97_arm_b_recorded_for_reference": _ITER97_ARM_B_RECORDED,
        "screening_gate_1_cv_delta_ge_1pt": p1_pass,
        "screening_gate_2_zero_bh_regressions_seed_merged": n_bh_significant_regressions == 0,
        "screening_gate_pass": p1_pass and n_bh_significant_regressions == 0,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iter98 pre-screening: fixed-fold CV, softmax vs error-correcting-output-code base estimator"
    )
    parser.add_argument("--train-embcache", required=True)
    parser.add_argument("--n-existing", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        summary = run_screening(args.train_embcache, args.n_existing)
    except SampleWeightDroppedError as exc:
        error_summary = {"experiment_invalid": True, "reason": str(exc)}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(error_summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(error_summary, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

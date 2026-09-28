"""Iter97 pre-registered screening (run on wafl-ctrl5, absolute condition (B)):
fixed stratified-5-fold CV comparison of the current single-softmax
LogisticRegression base estimator ("arm A") against a one-vs-one pairwise
base estimator ("arm B", 45 pairwise binary LogisticRegressions via
sklearn.multiclass.OneVsOneClassifier), gating whether the full experiment
(real routing run on wafl500-509) is worth 180 minutes of node time
(config.yml levers `classifier_multiclass_decomposition` note, journal.md
"Iteration 97").

Design (pre-registered in journal.md "Iteration 97"):
  - Both arms train on the *same* existing 2,275-row training set
    (data/classifier_train_iter94_dedup.jsonl, embedded once in
    data/embcache_iter96_scaled.npy's first 2,275 rows -- reused unchanged
    from Iter96, zero new embedding calls). No rows are added or removed;
    the only difference between arms is the base estimator passed to
    CalibratedClassifierCV.
  - A fixed 5-fold StratifiedKFold split on domain labels, 3 seeds
    (96/196/296), matched/paired between arms (same train/test row indices
    in every (seed, fold)), reusing scripts/screen_composition_preserving_
    volume_expansion.py's structure.
  - Arm A calls scripts/train_domain_classifier.py's train_classifier()
    unchanged (single softmax LogisticRegression base estimator).
  - Arm B calls _train_ovo_classifier() below, which is train_classifier()
    with only the base estimator swapped to
    OneVsOneClassifier(LogisticRegression(...)). Per the journal's
    pre-registered pitfall, sample_weight routing to the 45 pairwise base
    estimators requires sklearn.config_context(enable_metadata_routing=True)
    plus LogisticRegression.set_fit_request(sample_weight=True); every
    model.fit() call is wrapped so that any UserWarning (e.g. sklearn
    silently dropping sample_weight for a base estimator that does not
    request it) aborts the whole screening run immediately, per the
    pre-registered "1 warning = experiment invalid" rule.
  - Statistics (McNemar for overall accuracy and per-domain recall,
    Benjamini-Hochberg for the per-domain multiple-comparison correction)
    are computed by calling metrics.py's existing implementations on
    out-of-fold predictions reshaped into {id, selected_domain,
    expected_domains} rows -- not reimplemented here.
  - Diagnostics (not used for the gate, but always reported): per-domain
    recall deltas, predict_proba row-sum sanity (must be 1.0 for both
    arms, classifier.py:estimate_confidence_classifier()'s precondition),
    and the top-2 probability gap distribution (fraction of rows with
    gap < 0.36, the current dispatch_gap_threshold) for both arms.

Usage (on wafl-ctrl5):
    uv run python scripts/screen_classifier_multiclass_decomposition.py \\
        --train-embcache data/embcache_iter96_scaled.npy \\
        --n-existing 2275 \\
        --output /tmp/iter97_screening.json
"""

import argparse
import json
import sys
import warnings

import numpy as np
from sklearn import config_context
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier

sys.path.insert(0, ".")
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


class SampleWeightDroppedError(RuntimeError):
    """Raised when sklearn emits any UserWarning during a fit() call under metadata
    routing, signalling that sample_weight was silently dropped for a base estimator
    (the pre-registered "1 warning = experiment invalid" rule, journal.md Iteration 97)."""


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


def _train_ovo_classifier(
    embeddings: list[list[float]], labels: list[str], sample_weight: list[float]
) -> CalibratedClassifierCV:
    """Arm B: train_classifier() with only the base estimator swapped to
    OneVsOneClassifier(LogisticRegression(...)) -- 45 pairwise binary LRs
    instead of 1 shared softmax. Calibration method/cv, max_iter, and
    sample_weight are otherwise identical to train_classifier()'s arm A.

    Metadata routing (sklearn.config_context(enable_metadata_routing=True)
    + LogisticRegression.set_fit_request(sample_weight=True)) is required
    so sample_weight reaches all 45 pairwise base estimators instead of
    being silently dropped (journal.md Iteration 97's pre-registered
    pitfall). _fit_with_warning_guard() aborts if sklearn emits any
    warning during fit(), which is how a silent drop would otherwise
    manifest.
    """
    with config_context(enable_metadata_routing=True):
        base_estimator = LogisticRegression(
            max_iter=_MAX_ITER, class_weight=None
        ).set_fit_request(sample_weight=True)
        ovo_estimator = OneVsOneClassifier(base_estimator)
        calibrated_model = CalibratedClassifierCV(
            ovo_estimator, method=_CALIBRATION_METHOD, cv=_CALIBRATION_CV, ensemble=True
        )
        _fit_with_warning_guard(calibrated_model, embeddings, labels, sample_weight)
    return calibrated_model


def _rows_for_metrics(ids: list[str], true_domains: list[str], pred_domains: list[str]) -> list[dict]:
    """Build {id, selected_domain, expected_domains} rows matching metrics.py's schema."""
    return [
        {"id": i, "selected_domain": pred, "expected_domains": [true]}
        for i, true, pred in zip(ids, true_domains, pred_domains)
    ]


def _top2_gap_fraction_below_threshold(proba: np.ndarray, threshold: float) -> float:
    """Fraction of rows whose top-1 minus top-2 predict_proba is below `threshold`
    (config.yaml's dispatch_gap_threshold), a read-only diagnostic of how many rows
    would newly qualify for multi-domain dispatch under this arm's probability shape."""
    sorted_proba = np.sort(proba, axis=1)
    gap = sorted_proba[:, -1] - sorted_proba[:, -2]
    return float(np.mean(gap < threshold))


def run_screening(train_embcache_path: str, n_existing: int) -> dict:
    """Run the 3-seed x 5-fold screening; return a JSON-serializable summary dict."""
    embeddings = np.load(train_embcache_path)
    with open(train_embcache_path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["ids"][:n_existing]
    domains = meta["domains"][:n_existing]
    assert len(ids) == n_existing, "embcache metadata must have at least n_existing rows"
    existing_emb_arr = embeddings[:n_existing]
    existing_ids_arr = np.array(ids)
    existing_domains_arr = np.array(domains)

    oof_a: list[dict] = []
    oof_b: list[dict] = []
    proba_row_sums_a: list[float] = []
    proba_row_sums_b: list[float] = []
    gap_fractions_a: list[float] = []
    gap_fractions_b: list[float] = []

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

            # Arm B: OneVsOneClassifier base estimator, metadata-routed sample_weight.
            model_b = _train_ovo_classifier(train_emb, train_labels, weight)
            proba_b = model_b.predict_proba(test_emb)
            pred_b_test = model_b.classes_[np.argmax(proba_b, axis=1)]
            oof_b.extend(_rows_for_metrics(test_ids, test_true, list(pred_b_test)))
            proba_row_sums_b.extend(proba_b.sum(axis=1).tolist())
            gap_fractions_b.append(_top2_gap_fraction_below_threshold(proba_b, _DISPATCH_GAP_THRESHOLD))

            print(
                f"[screen] seed={seed} fold done (train={len(train_idx)}, test={len(test_idx)})",
                file=sys.stderr,
            )

    cv_acc_a = metrics.compute_top1_accuracy(oof_a)
    cv_acc_b = metrics.compute_top1_accuracy(oof_b)
    overall_mcnemar = metrics.compute_mcnemar_test(oof_a, oof_b)

    all_domains = sorted(set(domains))
    per_domain = {}
    p_values = []
    for domain in all_domains:
        recall_test = metrics.compute_domain_recall_mcnemar_test(oof_a, oof_b, domain)
        p_values.append(recall_test["p_value"])
        per_domain[domain] = recall_test
    bh_significant = metrics.apply_benjamini_hochberg(p_values, q=0.05)
    for domain, sig in zip(all_domains, bh_significant):
        per_domain[domain]["bh_significant"] = sig
        denom = sum(1 for r in oof_a if domain in r["expected_domains"])
        recall_a_value = sum(
            1 for r in oof_a if domain in r["expected_domains"] and r["selected_domain"] == domain
        ) / denom if denom else float("nan")
        recall_b_value = sum(
            1 for r in oof_b if domain in r["expected_domains"] and r["selected_domain"] == domain
        ) / denom if denom else float("nan")
        per_domain[domain]["recall_a"] = recall_a_value
        per_domain[domain]["recall_b"] = recall_b_value
        per_domain[domain]["recall_delta_pt"] = (recall_b_value - recall_a_value) * 100.0

    cv_delta_pt = (cv_acc_b - cv_acc_a) * 100.0
    n_bh_significant_regressions = sum(
        1 for d in all_domains if per_domain[d]["bh_significant"] and per_domain[d]["recall_delta_pt"] < 0
    )

    # P1: sum of the two weak domains' (education, medical) recall deltas.
    p1_weak_domains = ["education", "medical"]
    p1_weak_domain_delta_sum_pt = sum(per_domain[d]["recall_delta_pt"] for d in p1_weak_domains)

    # P3: legal (77-row minority domain) must not regress by more than 1.0pt.
    p3_legal_recall_delta_pt = per_domain["legal"]["recall_delta_pt"]

    # P4: predict_proba row-sum sanity + gap<threshold fraction comparability.
    proba_sums_a_arr = np.array(proba_row_sums_a)
    proba_sums_b_arr = np.array(proba_row_sums_b)
    p4_proba_sums_ok_a = bool(np.allclose(proba_sums_a_arr, 1.0, atol=1e-6))
    p4_proba_sums_ok_b = bool(np.allclose(proba_sums_b_arr, 1.0, atol=1e-6))
    mean_gap_fraction_a = float(np.mean(gap_fractions_a))
    mean_gap_fraction_b = float(np.mean(gap_fractions_b))
    p4_gap_fraction_delta_pt = (mean_gap_fraction_b - mean_gap_fraction_a) * 100.0
    p4_gap_fraction_within_5pt = abs(p4_gap_fraction_delta_pt) <= 5.0

    return {
        "n_existing": n_existing,
        "seeds": _SEEDS,
        "n_folds": _N_FOLDS,
        "cv_accuracy_a_softmax": cv_acc_a,
        "cv_accuracy_b_ovo": cv_acc_b,
        "cv_delta_pt": cv_delta_pt,
        "overall_mcnemar": overall_mcnemar,
        "per_domain": per_domain,
        "n_bh_significant_regressions": n_bh_significant_regressions,
        "p1_weak_domain_delta_sum_pt": p1_weak_domain_delta_sum_pt,
        "p1_pass_ge_3pt": p1_weak_domain_delta_sum_pt >= 3.0,
        "p2_pass_cv_delta_ge_1pt": cv_delta_pt >= 1.0,
        "p3_legal_recall_delta_pt": p3_legal_recall_delta_pt,
        "p3_pass_legal_not_regressed": p3_legal_recall_delta_pt >= -1.0,
        "p4_proba_sums_ok_a": p4_proba_sums_ok_a,
        "p4_proba_sums_ok_b": p4_proba_sums_ok_b,
        "p4_mean_gap_lt_threshold_fraction_a": mean_gap_fraction_a,
        "p4_mean_gap_lt_threshold_fraction_b": mean_gap_fraction_b,
        "p4_gap_fraction_delta_pt": p4_gap_fraction_delta_pt,
        "p4_pass_gap_within_5pt": p4_gap_fraction_within_5pt,
        "screening_gate_1_cv_delta_ge_1pt": cv_delta_pt >= 1.0,
        "screening_gate_2_zero_bh_regressions": n_bh_significant_regressions == 0,
        "screening_gate_pass": cv_delta_pt >= 1.0 and n_bh_significant_regressions == 0,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Iter97 pre-screening: fixed-fold CV, softmax vs one-vs-one base estimator"
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

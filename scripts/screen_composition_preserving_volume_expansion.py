"""Iter96 pre-registered screening (run on wafl-ctrl5, absolute condition (B)):
fixed stratified-5-fold CV comparison of the current classifier training set
("no-add" arm A, 2,275 rows) against the current set plus the per-task
proportional (m=2) addition ("add" arm B, 3,271 rows), gating whether the
full experiment (real routing run on wafl500-509) is worth 180 minutes of
node time (config.yml levers note, journal.md "Iteration 96").

Design (pre-registered in journal.md "Iteration 96" / config.yml's
composition_preserving_volume_expansion note):
  - (i) Primary: identical to Iter95's screen_classifier_training_volume_
    expansion.py -- a fixed 5-fold split (StratifiedKFold on domain labels)
    over the *existing* 2,275-row training set only, for 3 seeds. The held-
    out test fold is identical between arms in every (seed, fold)
    combination, so the comparison is a matched/paired one. Arm A fits on
    the fold's training rows only; Arm B fits on the fold's training rows
    plus every one of the 996 per-task-proportional addition rows (disjoint
    from the existing 2,275 by construction, so adding them to every
    fold's training side never leaks a held-out row).
  - (ii) Secondary: the 426-row pool hold-out (never in either arm's
    training set, disjoint from data/dataset.jsonl and from both arms'
    training rows) is scored by the *same* per-fold models used for (i)
    and appended to that fold's test predictions, giving a
    "test-fold + hold-out" accuracy per arm. Because the hold-out only
    exists for 5 pool-rich domains, this is reported as a sign-consistency
    check against (i), not as an unbiased estimate (journal.md's stated
    limitation).
  - Both arms reuse scripts/train_domain_classifier.py's train_classifier()
    (LogisticRegression + CalibratedClassifierCV(temperature, cv=5)) and
    _extract_sample_weights() unchanged, so this is the "production
    pipeline, calibration included" screening the plan calls for.
  - Statistics (McNemar for overall accuracy and per-domain recall,
    Benjamini-Hochberg for the per-domain multiple-comparison correction)
    are computed by calling metrics.py's existing implementations on
    out-of-fold predictions reshaped into {id, selected_domain,
    expected_domains} rows -- not reimplemented here.

Usage (on wafl-ctrl5):
    uv run python scripts/screen_composition_preserving_volume_expansion.py \\
        --train-embcache data/embcache_iter96_scaled.npy \\
        --n-existing 2275 \\
        --holdout-embcache data/embcache_iter96_holdout.npy \\
        --output /tmp/iter96_screening.json
"""

import argparse
import json
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from scripts.train_domain_classifier import _extract_sample_weights, train_classifier
import metrics

_SEEDS = [96, 196, 296]
_N_FOLDS = 5


def _rows_for_metrics(ids: list[str], true_domains: list[str], pred_domains: list[str]) -> list[dict]:
    """Build {id, selected_domain, expected_domains} rows matching metrics.py's schema."""
    return [
        {"id": i, "selected_domain": pred, "expected_domains": [true]}
        for i, true, pred in zip(ids, true_domains, pred_domains)
    ]


def run_screening(train_embcache_path: str, n_existing: int, holdout_embcache_path: str) -> dict:
    """Run the 3-seed x 5-fold screening (i) plus the hold-out sign-check (ii);
    return a JSON-serializable summary dict."""
    embeddings = np.load(train_embcache_path)
    with open(train_embcache_path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["ids"]
    domains = meta["domains"]
    assert len(ids) == embeddings.shape[0], "embcache row count must match its metadata"
    assert len(ids) >= n_existing, "embcache must include the existing rows plus the addition rows"

    existing_ids_arr = np.array(ids[:n_existing])
    existing_domains_arr = np.array(domains[:n_existing])
    existing_emb_arr = embeddings[:n_existing]

    add_embeddings = embeddings[n_existing:]
    add_domains = domains[n_existing:]

    holdout_embeddings = np.load(holdout_embcache_path)
    with open(holdout_embcache_path + ".meta.json", encoding="utf-8") as f:
        holdout_meta = json.load(f)
    holdout_ids = holdout_meta["ids"]
    holdout_domains = holdout_meta["domains"]
    assert len(holdout_ids) == holdout_embeddings.shape[0]

    oof_a: list[dict] = []
    oof_b: list[dict] = []
    oof_a_plus_holdout: list[dict] = []
    oof_b_plus_holdout: list[dict] = []

    for seed in _SEEDS:
        skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(existing_emb_arr, existing_domains_arr):
            test_ids = [f"{existing_ids_arr[i]}_s{seed}" for i in test_idx]
            test_true = list(existing_domains_arr[test_idx])
            test_emb = existing_emb_arr[test_idx].tolist()

            # Arm A: fold's training rows only.
            train_rows_a = [{"domain": existing_domains_arr[i]} for i in train_idx]
            weight_a = _extract_sample_weights(train_rows_a)
            model_a = train_classifier(
                existing_emb_arr[train_idx].tolist(), list(existing_domains_arr[train_idx]),
                sample_weight=weight_a,
            )
            pred_a_test = model_a.predict(test_emb)
            oof_a.extend(_rows_for_metrics(test_ids, test_true, list(pred_a_test)))

            # Arm B: fold's training rows plus every per-task-proportional addition row.
            train_emb_b = np.concatenate([existing_emb_arr[train_idx], add_embeddings], axis=0)
            train_domains_b = list(existing_domains_arr[train_idx]) + list(add_domains)
            train_rows_b = [{"domain": d} for d in train_domains_b]
            weight_b = _extract_sample_weights(train_rows_b)
            model_b = train_classifier(train_emb_b.tolist(), train_domains_b, sample_weight=weight_b)
            pred_b_test = model_b.predict(test_emb)
            oof_b.extend(_rows_for_metrics(test_ids, test_true, list(pred_b_test)))

            # (ii) Score the pool hold-out with this fold's models and append
            # to this fold's test predictions (per-fold "test + hold-out" set).
            holdout_ids_fold = [f"{hid}_s{seed}_f{test_idx[0]}" for hid in holdout_ids]
            pred_a_holdout = model_a.predict(holdout_embeddings.tolist())
            pred_b_holdout = model_b.predict(holdout_embeddings.tolist())

            oof_a_plus_holdout.extend(_rows_for_metrics(test_ids, test_true, list(pred_a_test)))
            oof_a_plus_holdout.extend(_rows_for_metrics(holdout_ids_fold, holdout_domains, list(pred_a_holdout)))
            oof_b_plus_holdout.extend(_rows_for_metrics(test_ids, test_true, list(pred_b_test)))
            oof_b_plus_holdout.extend(_rows_for_metrics(holdout_ids_fold, holdout_domains, list(pred_b_holdout)))

            print(
                f"[screen] seed={seed} fold done "
                f"(train_a={len(train_idx)}, train_b={len(train_idx) + len(add_embeddings)}, "
                f"test={len(test_idx)}, holdout={len(holdout_ids)})",
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

    pool_domains = ["medical", "natural_science", "history_culture", "business_economics", "mathematics"]
    depleted_domains = ["legal", "computer_science", "social_science", "general", "education"]
    added_delta_sum = sum(per_domain[d]["recall_delta_pt"] for d in pool_domains)
    zero_add_delta_sum = sum(per_domain[d]["recall_delta_pt"] for d in depleted_domains)

    cv_delta_pt = (cv_acc_b - cv_acc_a) * 100.0
    n_bh_significant_regressions = sum(
        1 for d in all_domains if per_domain[d]["bh_significant"] and per_domain[d]["recall_delta_pt"] < 0
    )

    # (ii) secondary evaluation: test-fold + hold-out accuracy, sign-checked against (i).
    holdout_acc_a = metrics.compute_top1_accuracy(oof_a_plus_holdout)
    holdout_acc_b = metrics.compute_top1_accuracy(oof_b_plus_holdout)
    holdout_delta_pt = (holdout_acc_b - holdout_acc_a) * 100.0

    return {
        "n_existing": n_existing,
        "n_addition": len(add_domains),
        "n_holdout": len(holdout_ids),
        "seeds": _SEEDS,
        "n_folds": _N_FOLDS,
        "cv_accuracy_a_no_add": cv_acc_a,
        "cv_accuracy_b_add": cv_acc_b,
        "cv_delta_pt": cv_delta_pt,
        "overall_mcnemar": overall_mcnemar,
        "per_domain": per_domain,
        "c5_pool_domains_recall_delta_sum_pt": added_delta_sum,
        "c5_depleted_domains_recall_delta_sum_pt": zero_add_delta_sum,
        "p2_pool_domains_exceed_depleted": added_delta_sum > zero_add_delta_sum,
        "n_bh_significant_regressions": n_bh_significant_regressions,
        "screening_gate_1_cv_delta_ge_1pt": cv_delta_pt >= 1.0,
        "screening_gate_2_zero_bh_regressions": n_bh_significant_regressions == 0,
        "screening_gate_pass": cv_delta_pt >= 1.0 and n_bh_significant_regressions == 0,
        "secondary_holdout_accuracy_a": holdout_acc_a,
        "secondary_holdout_accuracy_b": holdout_acc_b,
        "secondary_holdout_delta_pt": holdout_delta_pt,
        "p4_sign_consistent_with_primary": (holdout_delta_pt >= 0) == (cv_delta_pt >= 0),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Iter96 pre-screening: fixed-fold CV + hold-out sign-check")
    parser.add_argument("--train-embcache", required=True)
    parser.add_argument("--n-existing", type=int, required=True)
    parser.add_argument("--holdout-embcache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = run_screening(args.train_embcache, args.n_existing, args.holdout_embcache)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

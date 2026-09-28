"""Iter95 pre-registered screening (run on wafl-ctrl5, absolute condition (B)):
fixed stratified-5-fold CV comparison of the current classifier training set
("no-add" arm) against the current set plus every disjoint JMMLU pool row
("add" arm), gating whether the full experiment (real routing run on
wafl500-509) is worth 180 minutes of node time (config.yml levers note,
backlog B158). This does NOT replace the full experiment (absolute
condition (A)); it only decides whether to run it.

Design (pre-registered in journal.md "Iteration 95" / config.yml's
classifier_training_volume_expansion note):
  - The 5-fold split is fixed over the *existing* 2,275-row training set
    only (StratifiedKFold on domain labels), for 3 seeds. The held-out test
    fold is identical between arms in every (seed, fold) combination, so the
    comparison is a matched/paired one.
  - Arm A ("no-add"): fit on the fold's training rows only.
  - Arm B ("add"): fit on the fold's training rows *plus every new pool
    row* (the pool rows are disjoint from the existing 2,275 by
    construction — scripts/expand_classifier_train_pool.py — so adding all
    of them to every fold's training side does not leak any held-out row).
  - Both arms reuse scripts/train_domain_classifier.py's train_classifier()
    (LogisticRegression + CalibratedClassifierCV(temperature, cv=5)) and
    _extract_sample_weights() unchanged, so this is the "production
    pipeline, calibration included" screening the plan calls for, not a
    simplified stand-in.
  - Statistics (McNemar for overall accuracy and per-domain recall,
    Benjamini-Hochberg for the per-domain multiple-comparison correction)
    are computed by calling metrics.py's existing implementations on
    out-of-fold predictions reshaped into {id, selected_domain,
    expected_domains} rows — not reimplemented here (see repo-wide rule
    against re-deriving statistics already implemented elsewhere).

Usage (on wafl-ctrl5):
    uv run python scripts/screen_classifier_training_volume_expansion.py \\
        --embcache data/embcache_iter95_fullpool.npy \\
        --n-existing 2275 \\
        --output /tmp/iter95_screening.json
"""

import argparse
import json
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, ".")
from scripts.train_domain_classifier import _extract_sample_weights, train_classifier
import metrics

_SEEDS = [95, 195, 295]
_N_FOLDS = 5


def _rows_for_metrics(ids: list[str], true_domains: list[str], pred_domains: list[str]) -> list[dict]:
    """Build {id, selected_domain, expected_domains} rows matching metrics.py's schema."""
    return [
        {"id": i, "selected_domain": pred, "expected_domains": [true]}
        for i, true, pred in zip(ids, true_domains, pred_domains)
    ]


def run_screening(embcache_path: str, n_existing: int) -> dict:
    """Run the 3-seed x 5-fold screening; return a JSON-serializable summary dict."""
    embeddings = np.load(embcache_path)
    with open(embcache_path + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ids = meta["ids"]
    domains = meta["domains"]
    assert len(ids) == embeddings.shape[0], "embcache row count must match its metadata"
    assert len(ids) >= n_existing, "embcache must include the existing rows plus the pool rows"

    existing_rows = [
        {"id": ids[i], "domain": domains[i], "embedding": embeddings[i]}
        for i in range(n_existing)
    ]
    pool_embeddings = embeddings[n_existing:]
    pool_domains = domains[n_existing:]

    oof_a: list[dict] = []
    oof_b: list[dict] = []
    for seed in _SEEDS:
        existing_domains_arr = np.array([r["domain"] for r in existing_rows])
        existing_ids_arr = np.array([r["id"] for r in existing_rows])
        existing_emb_arr = np.array([r["embedding"] for r in existing_rows])

        skf = StratifiedKFold(n_splits=_N_FOLDS, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(existing_emb_arr, existing_domains_arr):
            test_ids = [f"{existing_ids_arr[i]}_s{seed}" for i in test_idx]
            test_true = list(existing_domains_arr[test_idx])
            test_emb = existing_emb_arr[test_idx].tolist()

            # Arm A: fold's training rows only.
            train_rows_a = [
                {"domain": existing_domains_arr[i]} for i in train_idx
            ]
            weight_a = _extract_sample_weights(train_rows_a)
            model_a = train_classifier(
                existing_emb_arr[train_idx].tolist(), list(existing_domains_arr[train_idx]),
                sample_weight=weight_a,
            )
            pred_a = model_a.predict(test_emb)
            oof_a.extend(_rows_for_metrics(test_ids, test_true, list(pred_a)))

            # Arm B: fold's training rows plus every disjoint pool row.
            train_emb_b = np.concatenate([existing_emb_arr[train_idx], pool_embeddings], axis=0)
            train_domains_b = list(existing_domains_arr[train_idx]) + list(pool_domains)
            train_rows_b = [{"domain": d} for d in train_domains_b]
            weight_b = _extract_sample_weights(train_rows_b)
            model_b = train_classifier(
                train_emb_b.tolist(), train_domains_b, sample_weight=weight_b,
            )
            pred_b = model_b.predict(test_emb)
            oof_b.extend(_rows_for_metrics(test_ids, test_true, list(pred_b)))

            print(
                f"[screen] seed={seed} fold done "
                f"(train_a={len(train_idx)}, train_b={len(train_idx) + len(pool_embeddings)}, "
                f"test={len(test_idx)})",
                file=sys.stderr,
            )

    cv_acc_a = metrics.compute_top1_accuracy(oof_a)
    cv_acc_b = metrics.compute_top1_accuracy(oof_b)
    overall_mcnemar = metrics.compute_mcnemar_test(oof_a, oof_b)

    all_domains = sorted(set(domains))
    per_domain = {}
    p_values = []
    for domain in all_domains:
        recall_a = metrics.compute_domain_recall_mcnemar_test(oof_a, oof_b, domain)
        p_values.append(recall_a["p_value"])
        per_domain[domain] = recall_a
    bh_significant = metrics.apply_benjamini_hochberg(p_values, q=0.05)
    for domain, sig in zip(all_domains, bh_significant):
        per_domain[domain]["bh_significant"] = sig
        # Recall_a / recall_b for readability alongside the McNemar output.
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

    added_domains = ["medical", "natural_science", "history_culture", "business_economics"]
    zero_add_domains = ["legal", "computer_science", "social_science", "general", "education"]
    added_delta_sum = sum(per_domain[d]["recall_delta_pt"] for d in added_domains)
    zero_add_delta_sum = sum(per_domain[d]["recall_delta_pt"] for d in zero_add_domains)

    return {
        "n_existing": n_existing,
        "n_pool": len(pool_domains),
        "seeds": _SEEDS,
        "n_folds": _N_FOLDS,
        "cv_accuracy_a_no_add": cv_acc_a,
        "cv_accuracy_b_add": cv_acc_b,
        "cv_delta_pt": (cv_acc_b - cv_acc_a) * 100.0,
        "overall_mcnemar": overall_mcnemar,
        "per_domain": per_domain,
        "c5_added_domains_recall_delta_sum_pt": added_delta_sum,
        "c5_zero_add_domains_recall_delta_sum_pt": zero_add_delta_sum,
        "c5_hypothesis_holds": added_delta_sum > zero_add_delta_sum,
        "screening_gate_1pt_pass": (cv_acc_b - cv_acc_a) * 100.0 >= 1.0,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Iter95 pre-screening: fixed-fold CV, add vs no-add")
    parser.add_argument("--embcache", required=True)
    parser.add_argument("--n-existing", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = run_screening(args.embcache, args.n_existing)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

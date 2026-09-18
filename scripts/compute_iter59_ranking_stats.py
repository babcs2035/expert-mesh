"""Iter59 (dispatch_candidate_ranking=multilabel_binary_relevance_head): formal
statistical tests over the offline predictions already produced by
scripts/evaluate_dispatch_candidate_ranking.py (results/iter59_ovr_ranking_predictions.jsonl).

This script performs no new statistical machinery of its own: every p-value and
metric is computed by calling metrics.py's existing functions
(compute_compound_coverage_metrics, _mcnemar_from_correctness,
compute_top1_accuracy), matching Iter58's experiment-phase pattern (journal.md
Iter58 "実験" section). The only new code here is the (row_id, expected_domain)
pairing used as input to _mcnemar_from_correctness and the exact-binomial
cross-check via scipy.stats.binomtest -- both are direct re-implementations of
the same pattern already used in Iter58, not new statistical logic.

Judgment / interpretation of these numbers (accept/reject, partial, etc.) is
explicitly out of scope for this script and this phase -- see journal.md
Iter59 plan's pre-registered S1-S4/N1-N4 criteria; the next phase (analyst/
reflector) applies them.

Usage:
    uv run python -m scripts.compute_iter59_ranking_stats \\
        --baseline results/20260918_202613/results.jsonl \\
        --new results/iter59_ovr_ranking_predictions.jsonl \\
        --output results/iter59_stats.json
"""

import argparse
import json
import sys

from scipy.stats import binomtest

from metrics import (
    _mcnemar_from_correctness,
    compute_compound_coverage_metrics,
    compute_top1_accuracy,
)

# Pre-registered thresholds (journal.md Iter59 "計画" section, S1-S4/N1-N4).
_BASELINE_COMPOUND_DOMAIN_SET_RECALL = 0.345  # 69/200, results/20260918_202613/results.jsonl
_S2_EFFECT_SIZE_FLOOR_PT = 0.04
_ALPHA = 0.05
_BASELINE_LEGAL_SELF_COVERAGE = 8  # of 30 legal-involving compound domain pairs
_EXPECTED_ROW_COUNT = 1600
_EXPECTED_COMPOUND_ROW_COUNT = 100
_EXPECTED_COMPOUND_PAIR_COUNT = 200
_IMPLEMENTATION_PHASE_RANK2_FLIP_RATE = 0.356875
_IMPLEMENTATION_PHASE_COMPOUND_DOMAIN_SET_RECALL = 0.35


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _index_by_id(rows: list[dict]) -> dict[str, dict]:
    """Build an id -> row lookup, matching baseline and new rows for paired comparisons."""
    return {row["id"]: row for row in rows}


def _domain_pair_coverage_maps(
    baseline_rows: list[dict], new_rows: list[dict]
) -> tuple[dict[str, bool], dict[str, bool], list[tuple[str, str]]]:
    """Build the (row_id, expected_domain) -> covered bool maps used by S1's McNemar test.

    Only compound rows (len(expected_domains) > 1) contribute pairs, matching
    compute_compound_coverage_metrics's own population (metrics.py L146-149).
    Returns the two coverage maps plus the ordered list of (row_id, domain)
    pairs so N4 can re-use the same pairing for its breakdown.
    """
    baseline_correct: dict[str, bool] = {}
    new_correct: dict[str, bool] = {}
    pair_keys: list[tuple[str, str]] = []
    for base_row in baseline_rows:
        if len(base_row["expected_domains"]) <= 1:
            continue
        new_row = new_rows[base_row["id"]]
        base_dispatched = set(base_row["dispatched_domains"])
        new_dispatched = set(new_row["dispatched_domains"])
        for domain in base_row["expected_domains"]:
            pair_id = f"{base_row['id']}::{domain}"
            baseline_correct[pair_id] = domain in base_dispatched
            new_correct[pair_id] = domain in new_dispatched
            pair_keys.append((base_row["id"], domain))
    return baseline_correct, new_correct, pair_keys


def _exact_mcnemar_binomtest(discordant_a_only: int, discordant_b_only: int) -> dict:
    """Exact (non-continuity-corrected) McNemar test via scipy.stats.binomtest.

    Cross-check for the continuity-corrected chi2 version already produced by
    metrics._mcnemar_from_correctness, per the Iter59 plan's "実施方法" step 3
    ("併せて metrics._mcnemar_from_correctness() の連続性補正版も参考値として
    併記"). Under the null hypothesis that discordant pairs are equally likely
    to favor either side, the count of the smaller discordant category is
    Binomial(n=discordant_pairs, p=0.5).
    """
    discordant_pairs = discordant_a_only + discordant_b_only
    if discordant_pairs == 0:
        return {"discordant_pairs": 0, "p_value": 1.0}
    k = min(discordant_a_only, discordant_b_only)
    result = binomtest(k, discordant_pairs, 0.5, alternative="two-sided")
    return {"discordant_pairs": discordant_pairs, "p_value": float(result.pvalue)}


def _compute_s1(baseline_rows: list[dict], new_rows: dict[str, dict]) -> dict:
    """S1: domain-unit paired comparison of compound_domain_set_recall (n=200 pairs)."""
    baseline_correct, new_correct, pair_keys = _domain_pair_coverage_maps(baseline_rows, new_rows)
    if len(pair_keys) != _EXPECTED_COMPOUND_PAIR_COUNT:
        raise AssertionError(
            f"S1: expected {_EXPECTED_COMPOUND_PAIR_COUNT} compound domain pairs, "
            f"found {len(pair_keys)}"
        )
    mcnemar = _mcnemar_from_correctness(baseline_correct, new_correct)
    exact = _exact_mcnemar_binomtest(mcnemar["discordant_a_only"], mcnemar["discordant_b_only"])
    return {
        "n_pairs": len(pair_keys),
        "improved_pairs": mcnemar["discordant_b_only"],  # baseline uncovered -> new covered
        "regressed_pairs": mcnemar["discordant_a_only"],  # baseline covered -> new uncovered
        "discordant_pairs": mcnemar["discordant_pairs"],
        "chi2_statistic_continuity_corrected": mcnemar["chi2_statistic"],
        "p_value_continuity_corrected": mcnemar["p_value"],
        "p_value_exact_binomtest": exact["p_value"],
        "pass": bool(exact["p_value"] < _ALPHA),
    }, pair_keys, baseline_correct, new_correct


def _compute_s2(baseline_recall: float, new_recall: float) -> dict:
    """S2: point-estimate improvement must be >= +0.04 (69/200 -> >=77/200)."""
    delta = new_recall - baseline_recall
    return {
        "baseline_recall": baseline_recall,
        "new_recall": new_recall,
        "delta_pt": delta,
        "floor_pt": _S2_EFFECT_SIZE_FLOOR_PT,
        "pass": bool(delta >= _S2_EFFECT_SIZE_FLOOR_PT),
    }


def _compute_s3(new_rows_list: list[dict]) -> dict:
    """S3: cost neutrality -- every row keeps exactly 2 distinct dispatched domains."""
    lengths = [len(r["dispatched_domains"]) for r in new_rows_list]
    duplicate_count = sum(
        1 for r in new_rows_list if r["dispatched_domains"][0] == r["dispatched_domains"][1]
    )
    mean_dispatch = sum(lengths) / len(lengths)
    all_length_2 = all(length == 2 for length in lengths)
    return {
        "n_rows": len(new_rows_list),
        "length_distribution": {str(length): lengths.count(length) for length in set(lengths)},
        "duplicate_rank1_rank2_count": duplicate_count,
        "mean_dispatch": mean_dispatch,
        "pass": bool(all_length_2 and duplicate_count == 0 and mean_dispatch == 2.0),
    }


def _compute_s4(baseline_rows: list[dict], new_rows: dict[str, dict]) -> dict:
    """S4: rank2_flip_rate, recomputed independently from the evaluate script's own count."""
    flips = 0
    for base_row in baseline_rows:
        new_row = new_rows[base_row["id"]]
        if base_row["dispatched_domains"][1] != new_row["dispatched_domains"][1]:
            flips += 1
    rate = flips / len(baseline_rows)
    matches_implementation_phase = abs(rate - _IMPLEMENTATION_PHASE_RANK2_FLIP_RATE) < 1e-9
    return {
        "n_rows": len(baseline_rows),
        "flips": flips,
        "rank2_flip_rate": rate,
        "matches_implementation_phase_value": matches_implementation_phase,
        "implementation_phase_value": _IMPLEMENTATION_PHASE_RANK2_FLIP_RATE,
        "pass": bool(rate > 0.0),
    }


def _compute_n1(baseline_rows: list[dict], new_rows: dict[str, dict]) -> dict:
    """N1: rank_1 (dispatched_domains[0]) must be identical to baseline on every row."""
    mismatches = [
        base_row["id"]
        for base_row in baseline_rows
        if base_row["dispatched_domains"][0] != new_rows[base_row["id"]]["dispatched_domains"][0]
    ]
    return {
        "n_rows": len(baseline_rows),
        "mismatch_count": len(mismatches),
        "mismatch_ids": mismatches[:10],
        "pass": bool(len(mismatches) == 0),
    }


def _compute_n2(baseline_rows: list[dict], new_rows_list: list[dict]) -> dict:
    """N2: top1_accuracy recomputed on new_rows must exactly match baseline.

    Caveat (must be reported, not silently assumed): evaluate_dispatch_candidate_ranking.py's
    build_new_rows() copies `selected_domain` verbatim from the baseline row
    rather than re-deriving it from the new dispatched_domains -- so this
    equality holds by construction of the offline scoring script, not because
    the aggregation logic (aggregator.select_best_dispatch_response) was
    re-run against the new candidate set. In the live system,
    select_best_dispatch_response DOES depend on which nodes were actually
    dispatched (it picks the highest-confidence answer among the dispatched
    nodes' live responses), so this offline check only confirms "the field
    was preserved", not "the live aggregator would be unchanged if rank_2
    changed" -- that would require an online re-run, which this iteration's
    design (journal.md Iter59 plan) explicitly does not perform.
    """
    baseline_top1 = compute_top1_accuracy(baseline_rows)
    new_top1 = compute_top1_accuracy(new_rows_list)
    return {
        "baseline_top1_accuracy": baseline_top1,
        "new_top1_accuracy": new_top1,
        "exact_match": bool(baseline_top1 == new_top1),
        "caveat": (
            "new_rows' selected_domain is copied verbatim from the baseline row by "
            "build_new_rows(), not re-derived from the new dispatched_domains via "
            "aggregator.select_best_dispatch_response. In the live system, "
            "select_best_dispatch_response depends on which nodes were actually "
            "dispatched (it selects among the dispatched nodes' live LLM responses), "
            "so this check confirms field preservation under the offline scoring "
            "script, not that the live aggregator's output is invariant to a "
            "rank_2 change -- that would require an online re-run not performed "
            "in this iteration."
        ),
        "pass": bool(baseline_top1 == new_top1),
    }


def _compute_n3(
    baseline_rows: list[dict], new_rows: dict[str, dict], baseline_correct: dict[str, bool], new_correct: dict[str, bool]
) -> dict:
    """N3: legal-involving compound pairs' legal self-coverage must not drop below baseline (8/30)."""
    legal_pair_ids = [
        f"{r['id']}::legal"
        for r in baseline_rows
        if len(r["expected_domains"]) > 1 and "legal" in r["expected_domains"]
    ]
    baseline_legal_covered = sum(1 for pid in legal_pair_ids if baseline_correct[pid])
    new_legal_covered = sum(1 for pid in legal_pair_ids if new_correct[pid])
    return {
        "n_legal_involving_pairs": len(legal_pair_ids),
        "baseline_legal_self_coverage": baseline_legal_covered,
        "new_legal_self_coverage": new_legal_covered,
        "expected_baseline_value": _BASELINE_LEGAL_SELF_COVERAGE,
        "baseline_matches_journal_record": bool(
            baseline_legal_covered == _BASELINE_LEGAL_SELF_COVERAGE
        ),
        "pass": bool(new_legal_covered >= baseline_legal_covered),
    }


def _row_category(expected_domains: list[str]) -> str:
    """Classify a compound row as legal-involving / medical-involving / other.

    legal takes priority over medical (matches journal.md Iter59 investigation's
    counting convention: "legalが絡む行が30件...medicalの28件" are reported as
    separate, non-additive counts of the same 100 rows). Every observed
    compound row in this dataset has exactly 2 expected_domains (verified
    separately: Counter({2: 100})), so legal+medical co-occurring in one row
    does not arise in this dataset, but the priority rule is kept explicit in
    case that assumption ever changes.
    """
    if "legal" in expected_domains:
        return "legal_involving"
    if "medical" in expected_domains:
        return "medical_involving"
    return "other"


def _compute_n4(
    baseline_rows: list[dict],
    pair_keys: list[tuple[str, str]],
    baseline_correct: dict[str, bool],
    new_correct: dict[str, bool],
) -> dict:
    """N4: break down improved/regressed compound domain pairs by row category."""
    row_by_id = _index_by_id(baseline_rows)
    breakdown = {
        "legal_involving": {"n_pairs": 0, "improved": 0, "regressed": 0, "unchanged": 0},
        "medical_involving": {"n_pairs": 0, "improved": 0, "regressed": 0, "unchanged": 0},
        "other": {"n_pairs": 0, "improved": 0, "regressed": 0, "unchanged": 0},
    }
    for row_id, domain in pair_keys:
        pair_id = f"{row_id}::{domain}"
        category = _row_category(row_by_id[row_id]["expected_domains"])
        breakdown[category]["n_pairs"] += 1
        was_covered = baseline_correct[pair_id]
        is_covered = new_correct[pair_id]
        if not was_covered and is_covered:
            breakdown[category]["improved"] += 1
        elif was_covered and not is_covered:
            breakdown[category]["regressed"] += 1
        else:
            breakdown[category]["unchanged"] += 1
    return breakdown


def compute_all_stats(baseline_rows: list[dict], new_rows_list: list[dict]) -> dict:
    """Run S1-S4/N1-N4 and return one machine-readable dict."""
    if len(baseline_rows) != _EXPECTED_ROW_COUNT or len(new_rows_list) != _EXPECTED_ROW_COUNT:
        raise AssertionError(
            f"expected {_EXPECTED_ROW_COUNT} rows in both files, "
            f"got baseline={len(baseline_rows)}, new={len(new_rows_list)}"
        )
    baseline_ids = {r["id"] for r in baseline_rows}
    new_ids = {r["id"] for r in new_rows_list}
    if baseline_ids != new_ids:
        raise ValueError("baseline and new results must cover the same set of ids")

    new_rows = _index_by_id(new_rows_list)

    baseline_compound_metrics = compute_compound_coverage_metrics(baseline_rows)
    new_compound_metrics = compute_compound_coverage_metrics(new_rows_list)
    if baseline_compound_metrics["compound_rows_evaluated"] != _EXPECTED_COMPOUND_ROW_COUNT:
        raise AssertionError(
            "baseline compound row count changed from the expected "
            f"{_EXPECTED_COMPOUND_ROW_COUNT}: {baseline_compound_metrics['compound_rows_evaluated']}"
        )
    if baseline_compound_metrics["compound_domain_set_recall"] != _BASELINE_COMPOUND_DOMAIN_SET_RECALL:
        raise AssertionError(
            "baseline compound_domain_set_recall does not match the pre-registered "
            f"0.345: got {baseline_compound_metrics['compound_domain_set_recall']}"
        )

    s1_result, pair_keys, baseline_correct, new_correct = _compute_s1(baseline_rows, new_rows)
    s2_result = _compute_s2(
        baseline_compound_metrics["compound_domain_set_recall"],
        new_compound_metrics["compound_domain_set_recall"],
    )
    s3_result = _compute_s3(new_rows_list)
    s4_result = _compute_s4(baseline_rows, new_rows)
    n1_result = _compute_n1(baseline_rows, new_rows)
    n2_result = _compute_n2(baseline_rows, new_rows_list)
    n3_result = _compute_n3(baseline_rows, new_rows, baseline_correct, new_correct)
    n4_result = _compute_n4(baseline_rows, pair_keys, baseline_correct, new_correct)

    implementation_phase_recall_matches = bool(
        new_compound_metrics["compound_domain_set_recall"]
        == _IMPLEMENTATION_PHASE_COMPOUND_DOMAIN_SET_RECALL
    )

    return {
        "baseline_compound_domain_set_recall": baseline_compound_metrics["compound_domain_set_recall"],
        "new_compound_domain_set_recall": new_compound_metrics["compound_domain_set_recall"],
        "matches_implementation_phase_diagnostic_recall": implementation_phase_recall_matches,
        "S1_primary_criterion": s1_result,
        "S2_effect_size_floor": s2_result,
        "S3_cost_neutrality": s3_result,
        "S4_flip_rate_evidence_of_firing": s4_result,
        "N1_rank1_invariance": n1_result,
        "N2_top1_accuracy_invariance": n2_result,
        "N3_legal_non_regression": n3_result,
        "N4_improvement_breakdown_by_domain_category": n4_result,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Iter59's pre-registered S1-S4/N1-N4 statistical checks"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Fixed dispatch_top_k=2 results.jsonl (results/20260918_202613/results.jsonl)",
    )
    parser.add_argument(
        "--new",
        required=True,
        help="Output of evaluate_dispatch_candidate_ranking.py "
        "(results/iter59_ovr_ranking_predictions.jsonl)",
    )
    parser.add_argument("--output", required=True, help="Path to write the stats JSON to")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    baseline_rows = _read_jsonl(args.baseline)
    new_rows_list = _read_jsonl(args.new)
    stats = compute_all_stats(baseline_rows, new_rows_list)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Compute routing-accuracy metrics from run_experiment.py's output (design doc 4.1, axis 1).

Only axis 1 (routing accuracy: Top-1 accuracy, precision/recall, misrouting
rate) is implemented here. Axis 2 (answer quality, e.g. LLM-as-judge) and
axis 3 (end-to-end accuracy combining both) require either human raters or
domain QA benchmarks with graded answers, which are out of scope for this
placeholder dataset (see build_dataset.py's docstring) and are left as a
Phase 2+ follow-up once real evaluation data exists.

Usage:
    uv run python metrics.py --results results.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import TextIO


def _read_results(path: str) -> list[dict]:
    """Load result rows written by run_experiment.py."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_top1_accuracy(results: list[dict]) -> float:
    """Fraction of rows where the selected domain is among the expected domains.

    "Top-1" here means the single node actually selected by the mesh
    (design doc 4.1's Top-1 metric), not a ranked list — Phase 0's
    aggregator only ever surfaces one final answer per query (or a
    fallback), so there is no Top-k ranking to evaluate beyond this.
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["selected_domain"] in r["expected_domains"])
    return correct / len(results)


def compute_misrouting_rate(results: list[dict]) -> float:
    """Fraction of rows where the selected domain is NOT among the expected domains.

    Complementary to Top-1 accuracy, but reported separately (design doc
    4.1) because a low accuracy can arise either from misrouting or from
    unanswered (fallback-without-match) queries, which callers may want to
    distinguish when the two rates don't sum in an obviously wrong way.
    """
    return 1.0 - compute_top1_accuracy(results)


def compute_precision_recall_per_domain(results: list[dict]) -> dict[str, dict[str, float]]:
    """Per-domain precision and recall, treating routing as a multi-label classifier.

    A domain's precision is P(query actually needs this domain | mesh
    selected it); recall is P(mesh selected this domain | query needs it).
    Compound-domain rows (multiple expected_domains) count as a correct
    selection for recall on every domain they list, matching how
    select_dispatch_targets treats "any qualifying expert" as acceptable
    (design doc 2.5).
    """
    all_domains = {domain for r in results for domain in r["expected_domains"]} | {
        r["selected_domain"] for r in results if r["selected_domain"] is not None
    }
    per_domain: dict[str, dict[str, float]] = {}
    for domain in sorted(all_domains):
        true_positive = sum(
            1
            for r in results
            if r["selected_domain"] == domain and domain in r["expected_domains"]
        )
        selected_as_domain = sum(1 for r in results if r["selected_domain"] == domain)
        should_be_domain = sum(1 for r in results if domain in r["expected_domains"])
        precision = true_positive / selected_as_domain if selected_as_domain > 0 else 0.0
        recall = true_positive / should_be_domain if should_be_domain > 0 else 0.0
        per_domain[domain] = {"precision": precision, "recall": recall}
    return per_domain


def compute_fallback_rate(results: list[dict]) -> float:
    """Fraction of rows answered by the requester's own fallback model.

    Not one of design doc 4.1's named metrics, but directly relevant to
    axis 3 (system-level usability): a high fallback rate means the
    confidence_threshold or probe prompts are too conservative even when
    routing logic itself is sound.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["used_fallback"]) / len(results)


def compute_dispatch_failure_rate(results: list[dict]) -> float:
    """Fraction of rows where a qualifying expert was found but every /dispatch call failed.

    Distinct from misrouting: this is a system-level failure (e.g. the
    selected node timed out or its ollama connection dropped), not a
    routing decision that pointed at the wrong domain. Kept separate so a
    high misrouting_rate isn't mistaken for a network/timeout problem.
    """
    if not results:
        return 0.0
    return sum(1 for r in results if r["dispatch_failed"]) / len(results)


def compute_mean_duration_ms(results: list[dict]) -> float:
    """Average end-to-end wall-clock duration in milliseconds."""
    if not results:
        return 0.0
    return sum(r["duration_ms"] for r in results) / len(results)


def compute_compound_coverage_metrics(results: list[dict]) -> dict:
    """Set-valued coverage of compound-domain rows by the actual dispatch candidate set.

    Motivation (journal.md Iter1, backlog.md B2/B3): with the current
    aggregator (aggregator.select_best_dispatch_response picks a single
    highest-confidence answer), a compound-domain row's `selected_domain`
    can only ever match one of its `expected_domains`, so `dispatch_top_k`
    has no effect on top1_accuracy/misrouting_rate for those rows. This
    function instead asks "did the dispatch candidate set (before final
    selection) cover the expected domain set?", which is the quantity
    `dispatch_top_k` can actually move.

    Requires run_experiment.py's `dispatched_domains` field (added
    alongside this function; see run_experiment.py's `_run_one`). Rows from
    older results.jsonl files that predate that field lack the key
    entirely, so `r.get("dispatched_domains")` is used (not `r[...]`) and
    such rows are skipped — this keeps the function backward compatible
    with results produced before this metric existed, rather than raising.

    Only compound rows (more than one expected domain) are considered:
    single-domain rows are covered by top1_accuracy already and diluting
    the average with them would blur the "did dispatch reach both experts"
    signal this metric exists to isolate.
    """
    compound_rows = [
        r
        for r in results
        if len(r["expected_domains"]) > 1 and r.get("dispatched_domains") is not None
    ]
    if not compound_rows:
        return {
            "compound_rows_evaluated": 0,
            "compound_covered_domain_count": 0,
            "compound_expected_domain_total": 0,
            "compound_domain_set_recall": 0.0,
            "compound_domain_coverage_ratio_mean": 0.0,
            "compound_domain_jaccard_mean": 0.0,
            "compound_mean_dispatched_count": 0.0,
            "compound_coverage_available": False,
        }

    covered_domain_count = 0
    expected_domain_total = 0
    coverage_ratio_sum = 0.0
    jaccard_sum = 0.0
    dispatched_count_sum = 0
    for r in compound_rows:
        expected = set(r["expected_domains"])
        dispatched = set(r["dispatched_domains"])
        intersection_size = len(dispatched & expected)
        union_size = len(dispatched | expected)

        covered_domain_count += intersection_size
        expected_domain_total += len(expected)
        coverage_ratio_sum += intersection_size / len(expected)
        jaccard_sum += intersection_size / union_size if union_size > 0 else 0.0
        dispatched_count_sum += len(dispatched)

    row_count = len(compound_rows)
    return {
        "compound_rows_evaluated": row_count,
        "compound_covered_domain_count": covered_domain_count,
        "compound_expected_domain_total": expected_domain_total,
        "compound_domain_set_recall": covered_domain_count / expected_domain_total,
        "compound_domain_coverage_ratio_mean": coverage_ratio_sum / row_count,
        "compound_domain_jaccard_mean": jaccard_sum / row_count,
        "compound_mean_dispatched_count": dispatched_count_sum / row_count,
        "compound_coverage_available": True,
    }


def compute_all_metrics(results: list[dict]) -> dict:
    """Bundle every axis-1 metric plus supporting counts into one summary dict."""
    by_compound = defaultdict(list)
    for r in results:
        by_compound[len(r["expected_domains"]) > 1].append(r)

    return {
        "total_questions": len(results),
        "top1_accuracy": compute_top1_accuracy(results),
        "misrouting_rate": compute_misrouting_rate(results),
        "fallback_rate": compute_fallback_rate(results),
        "dispatch_failure_rate": compute_dispatch_failure_rate(results),
        "mean_duration_ms": compute_mean_duration_ms(results),
        "precision_recall_per_domain": compute_precision_recall_per_domain(results),
        "single_domain_question_count": len(by_compound[False]),
        "single_domain_top1_accuracy": compute_top1_accuracy(by_compound[False]),
        "compound_domain_question_count": len(by_compound[True]),
        "compound_domain_top1_accuracy": compute_top1_accuracy(by_compound[True]),
        "compound_coverage": compute_compound_coverage_metrics(results),
    }


def print_summary(metrics: dict, output: TextIO) -> None:
    """Print a human-readable summary of the computed metrics."""
    print(f"総質問数: {metrics['total_questions']}", file=output)
    print(f"Top-1正解率: {metrics['top1_accuracy']:.3f}", file=output)
    print(f"誤ルーティング率: {metrics['misrouting_rate']:.3f}", file=output)
    print(f"フォールバック率: {metrics['fallback_rate']:.3f}", file=output)
    print(f"dispatch失敗率（システム的失敗）: {metrics['dispatch_failure_rate']:.3f}", file=output)
    print(f"平均応答時間: {metrics['mean_duration_ms']:.0f}ms", file=output)
    print(
        f"単一ドメイン質問のTop-1正解率: {metrics['single_domain_top1_accuracy']:.3f}"
        f"（{metrics['single_domain_question_count']}問）",
        file=output,
    )
    print(
        f"複合ドメイン質問のTop-1正解率: {metrics['compound_domain_top1_accuracy']:.3f}"
        f"（{metrics['compound_domain_question_count']}問）",
        file=output,
    )
    print("ドメイン別 適合率・再現率:", file=output)
    for domain, scores in metrics["precision_recall_per_domain"].items():
        print(
            f"  {domain}: precision={scores['precision']:.3f}, recall={scores['recall']:.3f}",
            file=output,
        )
    compound_coverage = metrics.get("compound_coverage", {})
    if compound_coverage.get("compound_coverage_available"):
        print("複合ドメイン行の dispatch 被覆率（dispatch_top_k の効果測定用）:", file=output)
        print(
            f"  対象行数: {compound_coverage['compound_rows_evaluated']}, "
            f"set recall(micro): {compound_coverage['compound_domain_set_recall']:.3f} "
            f"({compound_coverage['compound_covered_domain_count']}/"
            f"{compound_coverage['compound_expected_domain_total']})",
            file=output,
        )
        print(
            f"  被覆率(macro平均): {compound_coverage['compound_domain_coverage_ratio_mean']:.3f}, "
            f"Jaccard(macro平均): {compound_coverage['compound_domain_jaccard_mean']:.3f}, "
            f"平均dispatch数: {compound_coverage['compound_mean_dispatched_count']:.2f}",
            file=output,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute routing-accuracy metrics from run_experiment.py output"
    )
    parser.add_argument("--results", required=True, help="JSONL results from run_experiment.py")
    parser.add_argument(
        "--json", action="store_true", help="Print the raw metrics dict as JSON instead of text"
    )
    args = parser.parse_args()

    results = _read_results(args.results)
    metrics = compute_all_metrics(results)
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print_summary(metrics, sys.stdout)


if __name__ == "__main__":
    main()

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

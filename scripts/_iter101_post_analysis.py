"""Iter101 execute-phase report script: reuses metrics.py's existing statistical
functions (McNemar, Fisher exact, BH correction) against the Iter101 main run,
plus numpy.percentile (standard order-statistic definition, no custom formula)
for the duration quantiles required by the plan's C6 condition.

Not part of the permanent pipeline: a one-off helper invoked once by the
execute-phase agent to assemble the machine-readable comparison the
analysis/conclusion phase (rc-evaluator) needs, matching the "reuse existing
implementations, do not re-derive statistics" rule (see the research-cycle
skill's execute-phase instructions).

Usage:
    uv run python scripts/_iter101_post_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# This file is at scripts/, one level below the project root that holds
# metrics.py; add the root so the import below resolves when invoked as
# `uv run python scripts/_iter101_post_analysis.py` (mirrors
# tools/smoke_check.py's same sys.path setup).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (  # noqa: E402
    apply_benjamini_hochberg,
    compute_domain_precision_fisher_test,
    compute_domain_recall_mcnemar_test,
    compute_mcnemar_test,
)

DOMAINS = [
    "business_economics",
    "computer_science",
    "education",
    "general",
    "history_culture",
    "legal",
    "mathematics",
    "medical",
    "natural_science",
    "social_science",
]

NEW_RESULTS_PATH = "results/20260929_192157/results.jsonl"
BASELINE_RESULTS_PATH = "results/20260928_160921/results.jsonl"  # pre-Iter100, top1=0.833067
ITER100_RESULTS_PATH = "results/20260929_081612/results.jsonl"  # same representation, top1=0.839467


def _read(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _duration_quantiles(results: list[dict]) -> dict:
    single = [r["duration_ms"] for r in results if len(r["expected_domains"]) == 1]
    compound = [r["duration_ms"] for r in results if len(r["expected_domains"]) > 1]
    return {
        "single_n": len(single),
        "single_median_ms": float(np.percentile(single, 50)),
        "single_p95_ms": float(np.percentile(single, 95)),
        "compound_n": len(compound),
        "compound_p25_ms": float(np.percentile(compound, 25)),
        "compound_median_ms": float(np.percentile(compound, 50)),
    }


def main() -> None:
    """Compute the McNemar/BH/quantile comparisons the Iter101 plan's C1/C6 conditions require."""
    new_results = _read(NEW_RESULTS_PATH)
    baseline_results = _read(BASELINE_RESULTS_PATH)
    iter100_results = _read(ITER100_RESULTS_PATH)

    report: dict = {}

    report["duration_quantiles_new"] = _duration_quantiles(new_results)

    report["mcnemar_vs_baseline_20260928_160921"] = compute_mcnemar_test(
        new_results, baseline_results
    )
    report["mcnemar_vs_iter100_20260929_081612"] = compute_mcnemar_test(
        new_results, iter100_results
    )

    # C1: per-domain precision/recall regression check (20 tests), BH-corrected at q=0.05,
    # against the pre-Iter100 baseline (same non-regression reference point metrics.py's
    # summary output compares top1 against).
    per_domain = {}
    p_values = []
    labels = []
    for domain in DOMAINS:
        recall_test = compute_domain_recall_mcnemar_test(new_results, baseline_results, domain)
        per_domain[f"{domain}_recall"] = recall_test
        p_values.append(recall_test["p_value"])
        labels.append(f"{domain}_recall")
        try:
            precision_test = compute_domain_precision_fisher_test(
                new_results, baseline_results, domain
            )
            per_domain[f"{domain}_precision"] = precision_test
            p_values.append(precision_test["p_value"])
            labels.append(f"{domain}_precision")
        except ValueError as exc:
            per_domain[f"{domain}_precision"] = {"error": str(exc)}

    significance = apply_benjamini_hochberg(p_values, q=0.05)
    bh_flags = dict(zip(labels, significance))
    report["per_domain_tests_vs_baseline"] = per_domain
    report["per_domain_bh_significant_vs_baseline"] = bh_flags
    report["per_domain_bh_significant_count"] = sum(bh_flags.values())

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

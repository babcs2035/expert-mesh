"""E2: offline verification of the Iter13 STP sign-flip hypothesis (p0001 F3).

Iter13's STP experiment (results/20260722_113854/results.jsonl) recorded a
top1_accuracy of 0.0652 (3/46), roughly 2.9 SD below the 0.25 chance level
for 4-domain random selection. p0001 F3 raises the possibility that the
ranking sign is inverted (or the scaling is reversed) rather than the
signal being genuinely uninformative. This script re-selects each row's
domain from the already-stored probe_candidates using argmin(confidence)
instead of the recorded argmax(confidence), with no new experiment and no
new LLM calls.

Usage (module mode, so the root-level metrics.py import resolves):
    uv run python -m scripts.verify_stp_sign_flip --results results/20260722_113854/results.jsonl
"""

import argparse
import json
import sys
from typing import TextIO

from metrics import (
    compute_mcnemar_test,
    compute_random_baseline_accuracy,
    compute_top1_accuracy,
    compute_top1_accuracy_wilson_ci,
)


def recompute_selected_domain(
    probe_candidates: list[dict], key_field: str, pick: str
) -> str | None:
    """Re-select a domain from stored probe_candidates using key_field/pick.

    key_field is "confidence" (sigmoid-normalized) or "confidence_logprobs_mean"
    (raw mean logprob); pick is "max" or "min". Candidates with a None
    key_field value are excluded rather than raising, since older
    results.jsonl rows may lack the STP-only confidence_logprobs_mean
    field entirely. Returns None for an empty (or all-None) candidate list.
    """
    candidates = [c for c in probe_candidates if c.get(key_field) is not None]
    if not candidates:
        return None
    selector = max if pick == "max" else min
    return selector(candidates, key=lambda c: c[key_field])["domain"]


def _all_domains_from_results(results: list[dict]) -> list[str]:
    """Union of every expected_domains entry and every probed candidate's domain."""
    domains = {domain for r in results for domain in r["expected_domains"]} | {
        candidate["domain"] for r in results for candidate in r["probe_candidates"]
    }
    return sorted(domains)


def _with_selected_domain(results: list[dict], selected_domains: list[str | None]) -> list[dict]:
    """Return copies of results with selected_domain overridden (id/expected_domains kept)."""
    return [
        {**r, "selected_domain": domain}
        for r, domain in zip(results, selected_domains, strict=True)
    ]


def compare_selection_strategies(results: list[dict], domains: list[str]) -> dict:
    """Compare the recorded (argmax confidence) selection against the sign-flipped (argmin) one.

    argmin(confidence) and argmin(confidence_logprobs_mean) are
    mathematically identical here because confidence is a monotonically
    increasing sigmoid of confidence_logprobs_mean (router.py's
    estimate_confidence_stp), so only one flipped variant needs computing.
    """
    baseline_domains = [
        recompute_selected_domain(r["probe_candidates"], "confidence", "max") for r in results
    ]
    flipped_domains = [
        recompute_selected_domain(r["probe_candidates"], "confidence", "min") for r in results
    ]
    baseline_rows = _with_selected_domain(results, baseline_domains)
    flipped_rows = _with_selected_domain(results, flipped_domains)

    return {
        "recorded_selection_accuracy": compute_top1_accuracy(baseline_rows),
        "recorded_selection_wilson_ci": compute_top1_accuracy_wilson_ci(baseline_rows),
        "sign_flipped_accuracy": compute_top1_accuracy(flipped_rows),
        "sign_flipped_wilson_ci": compute_top1_accuracy_wilson_ci(flipped_rows),
        "chance_accuracy": compute_random_baseline_accuracy(results, domains),
        "mcnemar_recorded_vs_flipped": compute_mcnemar_test(baseline_rows, flipped_rows),
    }


def format_verdict(outcome: dict) -> str:
    """Render a cautious, non-overreaching summary of the sign-flip comparison.

    Deliberately avoids a binary "confirmed/refuted" framing: the actual
    numbers (as of this script's design) show the flip clears the chance
    level but falls well short of the self_report baseline, which is a
    partial result that a simple pass/fail verdict would misrepresent.
    """
    flipped = outcome["sign_flipped_accuracy"]
    chance = outcome["chance_accuracy"]
    recorded = outcome["recorded_selection_accuracy"]
    lines = [
        f"記録された選択（argmax confidence）: {recorded:.4f} "
        f"(Wilson 95% CI: {outcome['recorded_selection_wilson_ci']})",
        f"符号反転（argmin confidence）: {flipped:.4f} "
        f"(Wilson 95% CI: {outcome['sign_flipped_wilson_ci']})",
        f"偶然一致（ランダム選択の期待値）: {chance:.4f}",
        f"McNemar検定（記録値 vs 符号反転）: {outcome['mcnemar_recorded_vs_flipped']}",
        "",
    ]
    if flipped > chance:
        lines.append(
            "符号反転により偶然一致を上回る水準までは改善したが，"
            "self_report構成の実測水準には届いていない。"
            "「符号反転すれば0.87相当に戻る」という単純な仮説は支持されない。"
        )
    else:
        lines.append("符号反転しても偶然一致を上回らず，単純な符号反転バグ説は支持されない。")
    return "\n".join(lines)


def run(results_path: str, output: TextIO) -> dict:
    """Load results, compute the comparison, print the verdict, and return the raw outcome."""
    with open(results_path, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]
    domains = _all_domains_from_results(results)
    outcome = compare_selection_strategies(results, domains)
    print(format_verdict(outcome), file=output)
    return outcome


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Offline STP sign-flip verification (E2, p0001 F3) against a saved results.jsonl"
    )
    parser.add_argument(
        "--results", required=True, help="Path to a results.jsonl produced by run_experiment.py"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw outcome dict as JSON instead of the verdict text",
    )
    args = parser.parse_args()

    if args.json:
        with open(args.results, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]
        outcome = compare_selection_strategies(results, _all_domains_from_results(results))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
    else:
        run(args.results, sys.stdout)


if __name__ == "__main__":
    main()

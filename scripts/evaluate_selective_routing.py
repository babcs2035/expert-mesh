"""Iter76 (routing_abstention_signal=conformal_set_size): offline selective-routing
evaluation comparing candidate "棄権（abstain）" signals against the existing
runtime confidence baseline, without running any of run_experiment.py's
probe/dispatch/LLM-generation flow.

This reuses an already-recorded predictions jsonl (each row carrying
expected_domains / selected_domain / probabilities / split / set_size, e.g.
Iter75's conformal output) as a fixed, read-only input: no embeddings are
recomputed and no live ollama node is contacted. The only lever this script
reads is --abstention-signal, which selects one entry from the abstention
score function table _SIGNALS (journal Iter76 plan, "レバーを読む行"); an
unknown signal name raises ValueError rather than silently falling back to
the baseline (Iter69's lesson: a silent fallback would look like a passing
run while never having exercised the new code path).

Evaluation follows Geifman & El-Yaniv (NeurIPS 2017)'s selective-risk /
coverage framework for AURC, plus Traub, Bungert, Lueth, Baumgartner,
Maier-Hein, Maier-Hein & Jaeger, "Overcoming Common Flaws in the Evaluation
of Selective Classification Systems" (NeurIPS 2024, arXiv:2407.01032)'s
AUGRC (area under the *generalized* risk-coverage curve), which the paper
recommends as the primary metric because AURC is dominated by the handful
of samples retained at very low coverage. Both are discretized as the mean,
over coverage levels 1/n .. n/n, of the selective risk (AURC) or generalized
risk (AUGRC) at that level, with samples ranked by abstention score
descending (ties broken by a stable mergesort, so tied rows keep their
original jsonl order rather than a data-dependent order that would make the
curve non-reproducible across runs).

Confidence intervals for the per-coverage error rate reuse
metrics.compute_wilson_confidence_interval (this repo's existing Wilson
interval implementation; see that function's docstring for why it is
preferred over the naive normal approximation at this project's sample
sizes) rather than re-deriving a binomial interval formula here. The
paired bootstrap for Delta(AUGRC)/Delta(AURC) and the size<=k threshold
discordant-pair test (via scipy.stats.binomtest) have no existing
implementation elsewhere in this repository; both are standard textbook
constructions (percentile-method paired bootstrap; exact two-sided
binomial test against p=0.5 for a McNemar-style discordant-pair
comparison) rather than anything repository-specific.

Usage (fully offline; no ollama, no GPU, no config.yaml/http_server changes):
    uv run python scripts/evaluate_selective_routing.py \\
        --predictions results/20260923_161147/Iter75_variantA_edu005.jsonl \\
        --split eval \\
        --abstention-signal max_probability \\
        --abstention-signal conformal_set_size \\
        --abstention-signal margin \\
        --abstention-signal negative_entropy \\
        --bootstrap 10000 \\
        --bootstrap-seed 42 \\
        --output results/<ts>/Iter76_selective_routing.json
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest

from metrics import compute_wilson_confidence_interval

# Abstention rates (== 1 - coverage) reported in the risk-coverage artifact
# table (journal Iter76 plan, success-condition row "risk-coverage 表").
_REPORTED_ABSTENTION_RATES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

# size<=k thresholds swept for the discordant-pair comparison against
# max_probability (journal Iter76 plan step "到達コードパス"). Iter75's
# adopted conformal configuration produces set sizes in [1, 8] for this
# input file; thresholds at or above the maximum size would select all
# rows (coverage == 1.0, no abstention), so they are skipped at runtime.
_SIZE_THRESHOLDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

_BOOTSTRAP_CI_LOWER_PERCENTILE = 2.5
_BOOTSTRAP_CI_UPPER_PERCENTILE = 97.5


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _score_max_probability(row: dict) -> float:
    """Runtime confidence baseline: the classifier's own top-class probability."""
    return max(row["probabilities"].values())


def _score_margin(row: dict) -> float:
    """Top-1 minus top-2 class probability (larger margin == more trustworthy)."""
    probs = sorted(row["probabilities"].values(), reverse=True)
    top2 = probs[1] if len(probs) > 1 else 0.0
    return probs[0] - top2


def _score_negative_entropy(row: dict) -> float:
    """Negative Shannon entropy of the probability vector (peaked == high score).

    H(p) = -sum(p * log(p)) is the (non-negative) entropy; the score is -H(p),
    computed directly as sum(p * log(p)) to avoid a redundant negation. Zero
    probabilities are dropped from the sum (0 * log(0) is defined as 0 in the
    entropy convention and would otherwise raise on log(0)).
    """
    probs = np.array(list(row["probabilities"].values()), dtype=np.float64)
    probs = probs[probs > 0.0]
    return float(np.sum(probs * np.log(probs)))


def _score_conformal_set_size(row: dict) -> float:
    """Iter76's candidate signal: negative conformal prediction-set size.

    Smaller sets indicate a more confident/trustworthy prediction, so the
    score (higher == more trustworthy, matching the other three signals'
    convention) is the negated size.
    """
    return -float(row["set_size"])


_SIGNALS: dict[str, Callable[[dict], float]] = {
    "max_probability": _score_max_probability,
    "margin": _score_margin,
    "negative_entropy": _score_negative_entropy,
    "conformal_set_size": _score_conformal_set_size,
}


def _get_signal_scorer(name: str) -> Callable[[dict], float]:
    """Look up an abstention score function by name (the sole lever-reading line).

    Raises ValueError for an unknown signal name instead of silently falling
    back to a default (Iter69's lesson: a silent fallback would still "pass"
    while never having exercised the requested code path).
    """
    if name not in _SIGNALS:
        raise ValueError(
            f"unknown abstention signal {name!r}; expected one of {sorted(_SIGNALS)}"
        )
    return _SIGNALS[name]


def compute_is_correct(rows: list[dict]) -> np.ndarray:
    """Return a bool array: argmax(probabilities) in expected_domains, per row."""
    correct = np.empty(len(rows), dtype=bool)
    for i, row in enumerate(rows):
        argmax_domain = max(row["probabilities"], key=row["probabilities"].get)
        correct[i] = argmax_domain in row["expected_domains"]
    return correct


def compute_scores(rows: list[dict], signal: str) -> np.ndarray:
    """Compute the abstention score array for `signal` over `rows`."""
    scorer = _get_signal_scorer(signal)
    return np.array([scorer(row) for row in rows], dtype=np.float64)


def _cumulative_errors_sorted_by_score(is_correct: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Cumulative error count after retaining the top-k highest-scored rows, k=1..n.

    Rows are ranked by score descending (most trustworthy first); ties are
    broken by a stable mergesort so the curve is reproducible regardless of
    input ordering ambiguity.
    """
    order = np.argsort(-score, kind="mergesort")
    errors_sorted = (~is_correct[order]).astype(np.float64)
    return np.cumsum(errors_sorted)


def compute_aurc_augrc(is_correct: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return (AURC, AUGRC) for one abstention signal.

    AURC (Geifman & El-Yaniv 2017) is the mean, over coverage levels
    k/n for k=1..n, of the *selective* risk cumulative_errors(k) / k.
    AUGRC (Traub et al. 2024) is the mean, over the same levels, of the
    *generalized* risk cumulative_errors(k) / n (i.e. risk weighted by
    coverage rather than conditioned on it, which is what makes AUGRC
    insensitive to the low-coverage instability that dominates AURC).
    """
    n = len(is_correct)
    cum_errors = _cumulative_errors_sorted_by_score(is_correct, score)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    aurc = float(np.mean(cum_errors / ranks))
    augrc = float(np.mean(cum_errors / n))
    return aurc, augrc


def compute_risk_coverage_table(
    is_correct: np.ndarray, score: np.ndarray, abstention_rates: tuple[float, ...]
) -> list[dict]:
    """Selective error rate (with Wilson 95% CI) at each requested abstention rate."""
    n = len(is_correct)
    cum_errors = _cumulative_errors_sorted_by_score(is_correct, score)
    table = []
    for rate in abstention_rates:
        coverage = 1.0 - rate
        k = round(coverage * n)
        if k <= 0:
            continue
        errors_at_k = int(cum_errors[k - 1])
        error_rate = errors_at_k / k
        ci_lo, ci_hi = compute_wilson_confidence_interval(errors_at_k, k)
        table.append(
            {
                "abstention_rate": rate,
                "coverage": k / n,
                "n": k,
                "errors": errors_at_k,
                "error_rate": error_rate,
                "error_rate_ci95": [ci_lo, ci_hi],
            }
        )
    return table


def bootstrap_delta_aurc_augrc(
    is_correct: np.ndarray,
    score_baseline: np.ndarray,
    score_candidate: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired percentile-method bootstrap for Delta(AURC) and Delta(AUGRC).

    Delta = candidate - baseline (positive == candidate is worse, matching
    the sign convention used throughout journal Iter76's investigation).
    Both signals are re-evaluated on the same resampled row indices in each
    bootstrap replicate (paired resampling), not independently resampled.
    """
    n = len(is_correct)
    aurc_base_obs, augrc_base_obs = compute_aurc_augrc(is_correct, score_baseline)
    aurc_cand_obs, augrc_cand_obs = compute_aurc_augrc(is_correct, score_candidate)
    delta_aurc_obs = aurc_cand_obs - aurc_base_obs
    delta_augrc_obs = augrc_cand_obs - augrc_base_obs

    rng = np.random.default_rng(seed)
    deltas_aurc = np.empty(n_boot, dtype=np.float64)
    deltas_augrc = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aurc_base, augrc_base = compute_aurc_augrc(is_correct[idx], score_baseline[idx])
        aurc_cand, augrc_cand = compute_aurc_augrc(is_correct[idx], score_candidate[idx])
        deltas_aurc[b] = aurc_cand - aurc_base
        deltas_augrc[b] = augrc_cand - augrc_base

    return {
        "delta_aurc": delta_aurc_obs,
        "delta_aurc_ci95": [
            float(np.percentile(deltas_aurc, _BOOTSTRAP_CI_LOWER_PERCENTILE)),
            float(np.percentile(deltas_aurc, _BOOTSTRAP_CI_UPPER_PERCENTILE)),
        ],
        "delta_augrc": delta_augrc_obs,
        "delta_augrc_ci95": [
            float(np.percentile(deltas_augrc, _BOOTSTRAP_CI_LOWER_PERCENTILE)),
            float(np.percentile(deltas_augrc, _BOOTSTRAP_CI_UPPER_PERCENTILE)),
        ],
        "p_improve_augrc": float(np.mean(deltas_augrc < 0.0)),
        "n_boot": n_boot,
        "seed": seed,
    }


def compute_size_threshold_comparison(
    is_correct: np.ndarray,
    set_size: np.ndarray,
    max_probability_score: np.ndarray,
    thresholds: tuple[int, ...],
) -> list[dict]:
    """Discordant-pair comparison between size<=k retention and a max_probability
    retention set of the same cardinality, for each threshold k.

    For each k, "size-retained" is the fixed set {row : set_size <= k}; the
    "max_probability-retained" set is the top-n_k rows by max_probability
    score (n_k == |size-retained|, so the two sets are directly comparable
    at equal coverage). Since both sets have equal cardinality, their
    symmetric-difference halves (only-size, only-max_probability) are also
    equal in size; scipy.stats.binomtest checks whether the total errors
    among those discordant rows split evenly (p=0.5) between the two halves
    (a McNemar-style test on which signal's discordant retentions are wrong
    more often).
    """
    n = len(is_correct)
    order_by_maxprob = np.argsort(-max_probability_score, kind="mergesort")
    results = []
    for k in thresholds:
        size_selected = set_size <= k
        n_k = int(size_selected.sum())
        if n_k == 0 or n_k >= n:
            continue
        maxprob_selected = np.zeros(n, dtype=bool)
        maxprob_selected[order_by_maxprob[:n_k]] = True

        errors_size = int((~is_correct[size_selected]).sum())
        errors_maxprob = int((~is_correct[maxprob_selected]).sum())

        only_size = size_selected & ~maxprob_selected
        only_maxprob = maxprob_selected & ~size_selected
        errors_only_size = int((~is_correct[only_size]).sum())
        errors_only_maxprob = int((~is_correct[only_maxprob]).sum())
        n_discordant_each = int(only_size.sum())

        total_discordant_errors = errors_only_size + errors_only_maxprob
        p_value = (
            binomtest(errors_only_size, total_discordant_errors, 0.5).pvalue
            if total_discordant_errors > 0
            else 1.0
        )

        results.append(
            {
                "size_threshold": k,
                "coverage": n_k / n,
                "n": n_k,
                "error_rate_size": errors_size / n_k,
                "error_rate_max_probability": errors_maxprob / n_k,
                "only_size_errors": errors_only_size,
                "only_max_probability_errors": errors_only_maxprob,
                "n_discordant_each_side": n_discordant_each,
                "binomtest_p_value": float(p_value),
            }
        )
    return results


def compute_diagnostics(all_rows: list[dict], split_rows: list[dict], split: str) -> dict:
    """Non-regression diagnostics (journal Iter76 plan, success-condition item
    "非退行条件"): rank_1 invariance (this script never touches routing
    selection) and the input's set_size distribution, both independent of
    which --abstention-signal was requested.
    """

    def _top1_and_argmax_match(rows: list[dict]) -> tuple[float, float]:
        n = len(rows)
        top1_hits = 0
        argmax_matches = 0
        for row in rows:
            argmax_domain = max(row["probabilities"], key=row["probabilities"].get)
            if argmax_domain in row["expected_domains"]:
                top1_hits += 1
            if argmax_domain == row["selected_domain"]:
                argmax_matches += 1
        return top1_hits / n, argmax_matches / n

    top1_split, argmax_match_split = _top1_and_argmax_match(split_rows)
    top1_all, argmax_match_all = _top1_and_argmax_match(all_rows)
    set_size_distribution = dict(sorted(Counter(row["set_size"] for row in split_rows).items()))

    return {
        "split": split,
        "n_split": len(split_rows),
        "n_all": len(all_rows),
        "top1_accuracy_split": top1_split,
        "argmax_matches_selected_domain_rate_split": argmax_match_split,
        "top1_accuracy_all": top1_all,
        "argmax_matches_selected_domain_rate_all": argmax_match_all,
        "set_size_distribution_split": set_size_distribution,
    }


def build_report(
    all_rows: list[dict],
    split: str,
    signals: list[str],
    bootstrap_n: int,
    bootstrap_seed: int,
) -> dict:
    """Assemble the full offline selective-routing evaluation report."""
    split_rows = [row for row in all_rows if row["split"] == split]
    is_correct = compute_is_correct(split_rows)
    set_size = np.array([row["set_size"] for row in split_rows], dtype=np.int64)

    signal_reports: dict[str, dict] = {}
    scores_by_signal: dict[str, np.ndarray] = {}
    for signal in signals:
        score = compute_scores(split_rows, signal)
        scores_by_signal[signal] = score
        aurc, augrc = compute_aurc_augrc(is_correct, score)
        signal_reports[signal] = {
            "aurc": aurc,
            "augrc": augrc,
            "risk_coverage_table": compute_risk_coverage_table(
                is_correct, score, _REPORTED_ABSTENTION_RATES
            ),
        }

    bootstrap_reports: dict[str, dict] = {}
    if "max_probability" in scores_by_signal:
        baseline_score = scores_by_signal["max_probability"]
        for signal, score in scores_by_signal.items():
            if signal == "max_probability":
                continue
            bootstrap_reports[signal] = bootstrap_delta_aurc_augrc(
                is_correct, baseline_score, score, bootstrap_n, bootstrap_seed
            )

    size_threshold_comparison = []
    if "max_probability" in scores_by_signal:
        size_threshold_comparison = compute_size_threshold_comparison(
            is_correct, set_size, scores_by_signal["max_probability"], _SIZE_THRESHOLDS
        )

    return {
        "diagnostics": compute_diagnostics(all_rows, split_rows, split),
        "signals": signal_reports,
        "bootstrap_vs_max_probability": bootstrap_reports,
        "size_threshold_comparison_vs_max_probability": size_threshold_comparison,
    }


def _print_firing_evidence(report: dict, split: str, signals: list[str]) -> None:
    """Emit發火証拠 (firing evidence) to stderr per journal Iter76 plan step (last bullet)."""
    diag = report["diagnostics"]
    print(
        f"abstention_signal={signals} split={split} n={diag['n_split']} "
        f"set_size_distribution={diag['set_size_distribution_split']}",
        file=sys.stderr,
    )
    for signal, signal_report in report["signals"].items():
        print(
            f"  signal={signal} aurc={signal_report['aurc']:.6f} "
            f"augrc={signal_report['augrc']:.6f}",
            file=sys.stderr,
        )


def main() -> None:
    """CLI entry point: offline selective-routing evaluation (see module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Input predictions jsonl path")
    parser.add_argument("--split", default="eval", help="split value to evaluate (default: eval)")
    parser.add_argument(
        "--abstention-signal",
        dest="abstention_signals",
        action="append",
        required=True,
        choices=sorted(_SIGNALS),
        help="Abstention score signal to evaluate; may be repeated",
    )
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap replicate count")
    parser.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap RNG seed")
    parser.add_argument("--output", required=True, help="Output json path")
    args = parser.parse_args()

    all_rows = _read_jsonl(args.predictions)
    report = build_report(
        all_rows, args.split, args.abstention_signals, args.bootstrap, args.bootstrap_seed
    )
    _print_firing_evidence(report, args.split, args.abstention_signals)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

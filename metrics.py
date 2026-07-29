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
import math
import sys
from collections import Counter, defaultdict
from typing import TextIO

# z-score for a two-sided 95% confidence interval (Wilson score interval).
_Z_95 = 1.959963984540054


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


def _observed_domains(results: list[dict]) -> list[str]:
    """Union of every expected_domains entry and every non-null selected_domain, sorted.

    Used as the default domain set for metrics (precision/recall,
    baselines, Cohen's kappa) that need "every domain seen in this
    experiment" without requiring config.yaml to be passed alongside the
    results file.
    """
    domains = {domain for r in results for domain in r["expected_domains"]} | {
        r["selected_domain"] for r in results if r["selected_domain"] is not None
    }
    return sorted(domains)


def compute_precision_recall_per_domain(results: list[dict]) -> dict[str, dict[str, float]]:
    """Per-domain precision and recall, treating routing as a multi-label classifier.

    A domain's precision is P(query actually needs this domain | mesh
    selected it); recall is P(mesh selected this domain | query needs it).
    Compound-domain rows (multiple expected_domains) count as a correct
    selection for recall on every domain they list, matching how
    select_dispatch_targets treats "any qualifying expert" as acceptable
    (design doc 2.5).
    """
    per_domain: dict[str, dict[str, float]] = {}
    for domain in _observed_domains(results):
        true_positive = sum(
            1 for r in results if r["selected_domain"] == domain and domain in r["expected_domains"]
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


def compute_wilson_confidence_interval(
    successes: int, total: int, z: float = _Z_95
) -> tuple[float, float]:
    """Return the (lower, upper) Wilson score interval for a binomial proportion.

    Preferred over the naive normal-approximation interval (p +/- z*SE)
    because the latter can extend past [0, 1] and is a poor approximation
    at small n (this project's per-domain n is as low as ~10-40 rows).
    """
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half_width = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denominator
    return max(center - half_width, 0.0), min(center + half_width, 1.0)


def compute_top1_accuracy_wilson_ci(results: list[dict], z: float = _Z_95) -> tuple[float, float]:
    """Wilson confidence interval for the top1_accuracy proportion."""
    successes = sum(1 for r in results if r["selected_domain"] in r["expected_domains"])
    return compute_wilson_confidence_interval(successes, len(results), z)


def _standard_normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf (avoids adding scipy for one function)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_mcnemar_test(results_a: list[dict], results_b: list[dict]) -> dict[str, float]:
    """Continuity-corrected McNemar test comparing two paired top1-correctness sets.

    Rows are paired by `id` (the same evaluation question run under two
    different configurations), so this answers "is configuration B's
    accuracy change from A distinguishable from noise on these exact
    questions" rather than comparing two independent samples. Raises
    ValueError if the two result sets don't cover the same question ids,
    since a McNemar test over mismatched pairs would silently compare the
    wrong rows against each other.
    """
    ids_a = {r["id"] for r in results_a}
    ids_b = {r["id"] for r in results_b}
    if ids_a != ids_b:
        raise ValueError("results_a and results_b must cover the same set of ids")

    correct_a = {r["id"]: r["selected_domain"] in r["expected_domains"] for r in results_a}
    correct_b = {r["id"]: r["selected_domain"] in r["expected_domains"] for r in results_b}

    discordant_a_only = sum(1 for row_id in ids_a if correct_a[row_id] and not correct_b[row_id])
    discordant_b_only = sum(1 for row_id in ids_a if correct_b[row_id] and not correct_a[row_id])
    discordant_pairs = discordant_a_only + discordant_b_only
    if discordant_pairs == 0:
        chi2_statistic = 0.0
    else:
        chi2_statistic = (abs(discordant_a_only - discordant_b_only) - 1) ** 2 / discordant_pairs
    # chi-square(1) is the distribution of a squared standard normal, so its
    # upper-tail probability is the two-sided normal tail: 2*(1-Phi(sqrt(x))).
    p_value = 2.0 * (1.0 - _standard_normal_cdf(math.sqrt(max(chi2_statistic, 0.0))))
    return {
        "discordant_a_only": discordant_a_only,
        "discordant_b_only": discordant_b_only,
        "discordant_pairs": discordant_pairs,
        "chi2_statistic": chi2_statistic,
        "p_value": p_value,
    }


def compute_random_baseline_accuracy(results: list[dict], domains: list[str]) -> float:
    """Closed-form expected accuracy of selecting a domain uniformly at random.

    A compound row with k acceptable domains out of len(domains) candidates
    has hit probability k/len(domains); no RNG/simulation is needed since
    this is the exact expectation, not an estimate.
    """
    if not results or not domains:
        return 0.0
    return sum(len(r["expected_domains"]) / len(domains) for r in results) / len(results)


def compute_best_single_domain_baseline(
    results: list[dict], domains: list[str]
) -> dict[str, float]:
    """Accuracy if every row were routed to a single fixed domain, for each candidate domain.

    Includes the literature-specified "always route to general" baseline
    as one entry among all configured domains, rather than hardcoding
    "general" as a special case that would break if a mesh has no such
    domain (e.g. a specialized-only deployment).
    """
    if not results:
        return {domain: 0.0 for domain in domains}
    return {
        domain: sum(1 for r in results if domain in r["expected_domains"]) / len(results)
        for domain in domains
    }


def compute_oracle_accuracy(results: list[dict], available_domains: list[str]) -> float:
    """Fraction of rows answerable at all by some domain the mesh actually has configured.

    This is the upper bound imposed by mesh coverage, not by routing
    quality: it stays below 1.0 only when a row's expected_domains don't
    intersect available_domains at all (e.g. a dataset row tagged with a
    domain no node in config.yaml currently serves).
    """
    if not results:
        return 0.0
    available = set(available_domains)
    return sum(1 for r in results if available.intersection(r["expected_domains"])) / len(results)


def compute_cohens_kappa(results: list[dict], domains: list[str]) -> float:
    """Chance-corrected agreement between selected_domain and expected_domains.

    Computed over single-domain rows only (compound rows have no single
    "actual" class to test agreement against, so including them would
    require an arbitrary tie-breaking rule). Needed because raw accuracy is
    not comparable across mesh configurations with a different domain
    count: a 10-domain mesh's chance accuracy (~0.10) is much lower than a
    4-domain mesh's (~0.25), so the same raw accuracy reflects very
    different amounts of real signal.

    Raises ValueError when there are single-domain rows to score but
    domains is empty: chance_agreement would silently sum to 0 over an
    empty domain list, making the return value degenerate to plain
    observed accuracy — exactly the "raw accuracy across different domain
    counts" comparison this function exists to prevent (the caller almost
    certainly passed an incomplete domain list by mistake, not an
    intentional zero chance level). An empty results/single_domain_results
    is a different, legitimate case (no data at all) and still returns 0.0,
    matching every other metric's empty-input convention.
    """
    single_domain_results = [r for r in results if len(r["expected_domains"]) == 1]
    total = len(single_domain_results)
    if total == 0:
        return 0.0
    if not domains:
        raise ValueError("domains must be non-empty for a chance-corrected kappa computation")
    observed_agreement = (
        sum(1 for r in single_domain_results if r["selected_domain"] == r["expected_domains"][0])
        / total
    )
    actual_counts = Counter(r["expected_domains"][0] for r in single_domain_results)
    predicted_counts = Counter(
        r["selected_domain"] for r in single_domain_results if r["selected_domain"] is not None
    )
    chance_agreement = sum(
        (actual_counts.get(domain, 0) / total) * (predicted_counts.get(domain, 0) / total)
        for domain in domains
    )
    if chance_agreement >= 1.0:
        return 1.0 if observed_agreement >= 1.0 else 0.0
    return (observed_agreement - chance_agreement) / (1.0 - chance_agreement)


def compute_ece(results: list[dict], n_bins: int = 10) -> dict:
    """Expected Calibration Error over rows with a non-null `confidence`.

    Rows with confidence=None (fallback or dispatch_failed, see
    run_experiment.py's `_run_one`) are excluded rather than treated as
    confidence=0, since a missing confidence is a different phenomenon
    (no dispatch happened at all) from a low but present one. `n_rows` is
    returned alongside `ece` because that exclusion changes what population
    the number describes (docs/d0003 F3): comparing ECE across runs with a
    very different fallback_rate without also comparing n_rows can be
    misleading.

    Uses equal-width bins in [0, 1] (design doc's calibration axis), with
    the final bin's upper edge inclusive so a confidence of exactly 1.0
    doesn't fall outside every bin.
    """
    rows = [r for r in results if r["confidence"] is not None]
    if not rows:
        return {"ece": 0.0, "n_rows": 0}
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(rows)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            bin_rows = [r for r in rows if lo <= r["confidence"] <= hi]
        else:
            bin_rows = [r for r in rows if lo <= r["confidence"] < hi]
        if not bin_rows:
            continue
        bin_confidence = sum(r["confidence"] for r in bin_rows) / len(bin_rows)
        bin_accuracy = sum(
            1 for r in bin_rows if r["selected_domain"] in r["expected_domains"]
        ) / len(bin_rows)
        ece += (len(bin_rows) / total) * abs(bin_confidence - bin_accuracy)
    return {"ece": ece, "n_rows": total}


def compute_brier_score(results: list[dict]) -> dict:
    """Mean squared error between `confidence` and routing correctness (0/1).

    Unlike ECE (which only measures calibration within bins), Brier score
    also penalizes poor discrimination, so the two are reported together
    rather than as substitutes for each other. Same non-null-confidence
    population and `n_rows` caveat as compute_ece.
    """
    rows = [r for r in results if r["confidence"] is not None]
    if not rows:
        return {"brier_score": 0.0, "n_rows": 0}
    total_squared_error = sum(
        (r["confidence"] - (1.0 if r["selected_domain"] in r["expected_domains"] else 0.0)) ** 2
        for r in rows
    )
    return {"brier_score": total_squared_error / len(rows), "n_rows": len(rows)}


def _rank_with_average_ties(values: list[float]) -> list[float]:
    """Return 1-indexed ranks for `values`, averaging ranks within tied groups.

    Needed for a scipy-free Mann-Whitney U (AUROC) computation, matching
    this module's existing math.erf-based approach for the normal CDF.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def compute_auroc(results: list[dict]) -> dict:
    """AUROC of `confidence` as a discriminator between correct and incorrect routing.

    Computed via the Mann-Whitney U statistic (rank-sum), which is
    equivalent to the AUROC and needs no scipy/numpy dependency, matching
    this module's existing style (compute_mcnemar_test's math.erf-based
    p-value). Calibration (ECE) and discrimination (AUROC) answer different
    questions — a classifier can be well-calibrated on average while still
    ranking correct and incorrect predictions no better than chance, and
    vice versa — so both are reported (docs/d0003 F3).

    Returns auroc=None when every non-null-confidence row is correct or
    every one is incorrect: AUROC (probability a random correct row
    outranks a random incorrect one) is undefined with only one class
    present, not 0.0 or 1.0.
    """
    rows = [r for r in results if r["confidence"] is not None]
    if not rows:
        return {"auroc": None, "n_rows": 0}
    labels = [1 if r["selected_domain"] in r["expected_domains"] else 0 for r in rows]
    scores = [r["confidence"] for r in rows]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return {"auroc": None, "n_rows": len(rows)}
    ranks = _rank_with_average_ties(scores)
    positive_rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    u_statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return {"auroc": u_statistic / (n_positive * n_negative), "n_rows": len(rows)}


def compute_tie_rate(results: list[dict]) -> dict:
    """Fraction of probe rounds where the top-2 candidate confidences are exactly equal.

    Motivated by the self_report signal's bimodal saturation (journal.md
    Iter15/16): when most nodes report 0.9-0.95 or 0.1-0.2, ties at the top
    are common and the aggregator's tie-break (declaration order) silently
    decides routing instead of the confidence signal itself. Requires
    run_experiment.py's `probe_candidates` field; rows predating that field
    are skipped for backward compatibility (same convention as
    compute_compound_coverage_metrics).
    """
    rows = [r for r in results if r.get("probe_candidates")]
    if not rows:
        return {"tie_rate": 0.0, "n_rows": 0}
    tie_count = 0
    for r in rows:
        confidences = sorted((c["confidence"] for c in r["probe_candidates"]), reverse=True)
        if len(confidences) >= 2 and confidences[0] == confidences[1]:
            tie_count += 1
    return {"tie_rate": tie_count / len(rows), "n_rows": len(rows)}


def compute_confidence_dispersion(results: list[dict]) -> dict:
    """Mean per-probe standard deviation and sum of candidate confidences across nodes.

    The sum matters because self_report confidences from independent nodes
    don't sum to 1 (each node scores itself in isolation), while
    top_k_with_probs's candidates are constructed to sum to 1 by design
    (Tian et al. 2023) — reporting the mean sum lets a caller see this
    structural difference directly rather than inferring it from the
    elicitation method name. Same backward-compatibility skip as
    compute_tie_rate.
    """
    rows = [r for r in results if r.get("probe_candidates")]
    if not rows:
        return {"mean_confidence_std": 0.0, "mean_confidence_sum": 0.0, "n_rows": 0}
    stds = []
    sums = []
    for r in rows:
        confidences = [c["confidence"] for c in r["probe_candidates"]]
        n = len(confidences)
        mean_confidence = sum(confidences) / n
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / n
        stds.append(math.sqrt(variance))
        sums.append(sum(confidences))
    return {
        "mean_confidence_std": sum(stds) / len(stds),
        "mean_confidence_sum": sum(sums) / len(sums),
        "n_rows": len(rows),
    }


def compute_all_metrics(results: list[dict]) -> dict:
    """Bundle every axis-1 metric plus supporting counts into one summary dict."""
    by_compound = defaultdict(list)
    for r in results:
        by_compound[len(r["expected_domains"]) > 1].append(r)
    domains = _observed_domains(results)

    return {
        "total_questions": len(results),
        "top1_accuracy": compute_top1_accuracy(results),
        "top1_accuracy_wilson_ci": compute_top1_accuracy_wilson_ci(results),
        "cohens_kappa": compute_cohens_kappa(results, domains),
        "random_baseline_accuracy": compute_random_baseline_accuracy(results, domains),
        "best_single_domain_baseline": compute_best_single_domain_baseline(results, domains),
        "oracle_accuracy": compute_oracle_accuracy(results, domains),
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
        "ece": compute_ece(results),
        "brier_score": compute_brier_score(results),
        "auroc": compute_auroc(results),
        "tie_rate": compute_tie_rate(results),
        "confidence_dispersion": compute_confidence_dispersion(results),
    }


def print_summary(metrics: dict, output: TextIO) -> None:
    """Print a human-readable summary of the computed metrics."""
    print(f"総質問数: {metrics['total_questions']}", file=output)
    ci_lower, ci_upper = metrics["top1_accuracy_wilson_ci"]
    print(
        f"Top-1正解率: {metrics['top1_accuracy']:.3f}"
        f"（Wilson 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]）",
        file=output,
    )
    print(
        f"Cohen's kappa（偶然一致補正後，単一ドメイン行のみ）: {metrics['cohens_kappa']:.3f}",
        file=output,
    )
    print(f"Randomベースライン正解率: {metrics['random_baseline_accuracy']:.3f}", file=output)
    best_single = metrics["best_single_domain_baseline"]
    if best_single:
        best_domain, best_accuracy = max(best_single.items(), key=lambda item: item[1])
        print(
            f"BestSingleベースライン（最良の単一ドメイン固定）: {best_domain}={best_accuracy:.3f}",
            file=output,
        )
    print(
        f"Oracle正解率（メッシュが構成上到達可能なドメインの上限）: {metrics['oracle_accuracy']:.3f}",
        file=output,
    )
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
    ece = metrics["ece"]
    print(f"ECE（{ece['n_rows']}行，confidence非nullのみ対象）: {ece['ece']:.4f}", file=output)
    brier = metrics["brier_score"]
    print(f"Brier score（{brier['n_rows']}行）: {brier['brier_score']:.4f}", file=output)
    auroc = metrics["auroc"]
    auroc_str = f"{auroc['auroc']:.4f}" if auroc["auroc"] is not None else "undefined（単一クラスのみ）"
    print(f"AUROC（{auroc['n_rows']}行）: {auroc_str}", file=output)
    tie_rate = metrics["tie_rate"]
    print(
        f"同点タイ率（probe上位2件が完全一致，{tie_rate['n_rows']}行）: {tie_rate['tie_rate']:.4f}",
        file=output,
    )
    dispersion = metrics["confidence_dispersion"]
    print(
        f"ノード間confidence分散（{dispersion['n_rows']}行）: "
        f"mean_std={dispersion['mean_confidence_std']:.4f}, "
        f"mean_sum={dispersion['mean_confidence_sum']:.4f}",
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

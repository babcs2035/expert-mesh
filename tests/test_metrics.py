"""Tests for the routing-accuracy metrics computed from run_experiment.py output."""

import itertools

import pytest

from metrics import (
    compute_all_metrics,
    compute_auroc,
    compute_best_single_domain_baseline,
    compute_brier_score,
    compute_cohens_kappa,
    compute_compound_coverage_metrics,
    compute_confidence_dispersion,
    compute_dispatch_failure_rate,
    compute_ece,
    compute_fallback_rate,
    compute_mcnemar_test,
    compute_mean_duration_ms,
    compute_misrouting_rate,
    compute_oracle_accuracy,
    compute_precision_recall_per_domain,
    compute_random_baseline_accuracy,
    compute_tie_rate,
    compute_top1_accuracy,
    compute_top1_accuracy_wilson_ci,
    compute_wilson_confidence_interval,
)


_row_id_counter = itertools.count()


def _result(
    selected_domain: str | None,
    expected_domains: list[str],
    used_fallback: bool = False,
    dispatch_failed: bool = False,
    duration_ms: int = 100,
    dispatched_domains: list[str] | None = None,
    row_id: str | None = None,
    confidence: float | None = None,
    probe_candidates: list[dict] | None = None,
) -> dict:
    """Build a minimal result row matching run_experiment.py's output shape.

    `dispatched_domains` defaults to None so existing callers keep
    producing rows that match the pre-Iter1 results.jsonl schema (no such
    key at all in practice, but `.get(...)` treats a present-but-None value
    the same way) — this is what compute_compound_coverage_metrics's
    backward-compatibility skip is meant to exercise. `row_id` defaults to
    a fresh id per call (via a module-level counter, not id(object()) —
    CPython can reuse a just-freed temporary object's address, making
    id(object()) collide across calls) so tests that don't care about
    pairing (McNemar) still produce unique ids. `confidence` defaults to
    None (matches a fallback/dispatch_failed row) and `probe_candidates`
    defaults to None (pre-Iter1 schema, same backward-compatibility
    convention as `dispatched_domains`) so existing callers that don't care
    about the confidence-calibration metrics (compute_ece etc.) are
    unaffected.
    """
    return {
        "id": row_id if row_id is not None else f"row-{next(_row_id_counter)}",
        "selected_domain": selected_domain,
        "expected_domains": expected_domains,
        "used_fallback": used_fallback,
        "dispatch_failed": dispatch_failed,
        "duration_ms": duration_ms,
        "dispatched_domains": dispatched_domains,
        "confidence": confidence,
        "probe_candidates": probe_candidates,
    }


def test_compute_top1_accuracy_all_correct() -> None:
    """Every row's selected domain matches its expected domain -> accuracy 1.0."""
    results = [_result("medical", ["medical"]), _result("legal", ["legal"])]
    assert compute_top1_accuracy(results) == 1.0


def test_compute_top1_accuracy_counts_compound_domain_as_correct() -> None:
    """A compound-domain row counts as correct if the selection matches any expected domain."""
    results = [_result("legal", ["medical", "legal"])]
    assert compute_top1_accuracy(results) == 1.0


def test_compute_top1_accuracy_mixed() -> None:
    """Half correct, half wrong -> accuracy 0.5."""
    results = [_result("medical", ["medical"]), _result("legal", ["medical"])]
    assert compute_top1_accuracy(results) == 0.5


def test_compute_top1_accuracy_empty_results_is_zero() -> None:
    """An empty result set returns 0.0 rather than dividing by zero."""
    assert compute_top1_accuracy([]) == 0.0


def test_compute_misrouting_rate_is_complement_of_accuracy() -> None:
    """Misrouting rate is exactly 1 - top1_accuracy."""
    results = [_result("medical", ["medical"]), _result("legal", ["medical"])]
    assert compute_misrouting_rate(results) == 0.5


def test_compute_fallback_rate() -> None:
    """Fallback rate counts only rows where used_fallback is True."""
    results = [
        _result("general", ["general"], used_fallback=True),
        _result("medical", ["medical"], used_fallback=False),
    ]
    assert compute_fallback_rate(results) == 0.5


def test_compute_dispatch_failure_rate() -> None:
    """Dispatch failure rate counts only rows where dispatch_failed is True."""
    results = [
        _result(None, ["medical"], dispatch_failed=True),
        _result("legal", ["legal"], dispatch_failed=False),
    ]
    assert compute_dispatch_failure_rate(results) == 0.5


def test_compute_top1_accuracy_treats_dispatch_failure_as_incorrect() -> None:
    """A dispatch-failed row (selected_domain=None) never counts as a correct match."""
    results = [_result(None, ["medical"], dispatch_failed=True)]
    assert compute_top1_accuracy(results) == 0.0


def test_compute_mean_duration_ms() -> None:
    """Mean duration averages the duration_ms field across all rows."""
    results = [
        _result("medical", ["medical"], duration_ms=100),
        _result("legal", ["legal"], duration_ms=300),
    ]
    assert compute_mean_duration_ms(results) == 200.0


def test_compute_precision_recall_per_domain() -> None:
    """Precision and recall are computed independently for each domain seen."""
    results = [
        _result("medical", ["medical"]),  # true positive for medical
        _result("medical", ["legal"]),  # false positive for medical, false negative for legal
        _result("legal", ["legal"]),  # true positive for legal
    ]
    scores = compute_precision_recall_per_domain(results)
    # medical: selected twice, correct once -> precision 0.5; should-be-medical once, hit once -> recall 1.0
    assert scores["medical"]["precision"] == 0.5
    assert scores["medical"]["recall"] == 1.0
    # legal: selected once, correct once -> precision 1.0; should-be-legal twice, hit once -> recall 0.5
    assert scores["legal"]["precision"] == 1.0
    assert scores["legal"]["recall"] == 0.5


def test_compute_all_metrics_splits_single_and_compound_domain_accuracy() -> None:
    """compute_all_metrics reports single- and compound-domain accuracy separately."""
    results = [
        _result("medical", ["medical"]),  # single-domain, correct
        _result("general", ["medical", "legal"]),  # compound-domain, incorrect
    ]
    metrics = compute_all_metrics(results)
    assert metrics["total_questions"] == 2
    assert metrics["single_domain_question_count"] == 1
    assert metrics["single_domain_top1_accuracy"] == 1.0
    assert metrics["compound_domain_question_count"] == 1
    assert metrics["compound_domain_top1_accuracy"] == 0.0


def test_compute_compound_coverage_metrics_mixed_full_and_partial_coverage() -> None:
    """Coverage/set-recall/Jaccard are averaged across compound rows only.

    Row 1: dispatch covered both expected domains (full coverage, Jaccard 1.0).
    Row 2: dispatch covered only one of the two expected domains (half coverage).
    A single-domain row is included to confirm it is excluded from the aggregate.
    """
    results = [
        _result("medical", ["medical", "legal"], dispatched_domains=["medical", "legal"]),
        _result("legal", ["medical", "legal"], dispatched_domains=["legal"]),
        _result("medical", ["medical"], dispatched_domains=["medical"]),  # single-domain, excluded
    ]
    coverage = compute_compound_coverage_metrics(results)
    assert coverage["compound_coverage_available"] is True
    assert coverage["compound_rows_evaluated"] == 2
    assert coverage["compound_covered_domain_count"] == 3  # 2 (row1) + 1 (row2)
    assert coverage["compound_expected_domain_total"] == 4  # 2 + 2
    assert coverage["compound_domain_set_recall"] == 0.75  # 3/4
    assert coverage["compound_domain_coverage_ratio_mean"] == 0.75  # mean(1.0, 0.5)
    assert coverage["compound_domain_jaccard_mean"] == 0.75  # mean(1.0, 0.5)
    assert coverage["compound_mean_dispatched_count"] == 1.5  # mean(2, 1)


def test_compute_compound_coverage_metrics_skips_rows_missing_the_new_field() -> None:
    """Rows from results.jsonl files written before this metric existed are skipped, not errored.

    run_experiment.py only started emitting `dispatched_domains` in Iter1;
    older results rows have no such key at all. `_result(...)`'s default
    (present-but-None) covers the same `.get(...) is None` skip path, and a
    raw dict here additionally exercises true key-absence, matching the
    actual shape of a pre-Iter1 results.jsonl line.
    """
    legacy_row_missing_key = {
        "selected_domain": "medical",
        "expected_domains": ["medical", "legal"],
    }
    results = [
        legacy_row_missing_key,
        _result(
            "medical", ["medical", "legal"]
        ),  # present but None (dispatched_domains defaults to None)
    ]
    coverage = compute_compound_coverage_metrics(results)
    assert coverage["compound_coverage_available"] is False
    assert coverage["compound_rows_evaluated"] == 0


def test_compute_compound_coverage_metrics_empty_results_is_unavailable() -> None:
    """An empty result set reports compound_coverage_available=False, not a ZeroDivisionError."""
    coverage = compute_compound_coverage_metrics([])
    assert coverage["compound_coverage_available"] is False
    assert coverage["compound_rows_evaluated"] == 0
    assert coverage["compound_domain_set_recall"] == 0.0
    assert coverage["compound_mean_dispatched_count"] == 0.0


def test_compute_all_metrics_includes_compound_coverage_key() -> None:
    """compute_all_metrics exposes compound_coverage without altering pre-existing keys."""
    results = [
        _result("medical", ["medical", "legal"], dispatched_domains=["medical", "legal"]),
    ]
    metrics = compute_all_metrics(results)
    assert metrics["compound_coverage"]["compound_coverage_available"] is True
    assert metrics["compound_coverage"]["compound_rows_evaluated"] == 1
    # Pre-existing keys/values are untouched by this addition.
    assert metrics["compound_domain_question_count"] == 1
    assert metrics["compound_domain_top1_accuracy"] == 1.0


def test_compute_wilson_confidence_interval_matches_d0001_reference() -> None:
    """Matches the d0001 literature survey's reference value for 40/46: [74.3%, 93.9%]."""
    lower, upper = compute_wilson_confidence_interval(40, 46)
    assert lower == pytest.approx(0.7433, abs=1e-3)
    assert upper == pytest.approx(0.9388, abs=1e-3)


def test_compute_wilson_confidence_interval_bounds_within_unit_interval() -> None:
    """The interval never extends outside [0, 1], even at the extremes (0/n or n/n)."""
    lower, upper = compute_wilson_confidence_interval(0, 10)
    assert 0.0 <= lower <= upper <= 1.0
    lower, upper = compute_wilson_confidence_interval(10, 10)
    assert 0.0 <= lower <= upper <= 1.0


def test_compute_wilson_confidence_interval_empty_total_is_zero() -> None:
    """total=0 returns (0.0, 0.0) rather than dividing by zero."""
    assert compute_wilson_confidence_interval(0, 0) == (0.0, 0.0)


def test_compute_top1_accuracy_wilson_ci_wraps_top1_accuracy() -> None:
    """The CI is computed over the same successes/total as compute_top1_accuracy."""
    results = [_result("medical", ["medical"]), _result("legal", ["medical"])]
    lower, upper = compute_top1_accuracy_wilson_ci(results)
    assert lower < 0.5 < upper


def test_compute_mcnemar_test_matches_known_chi_square_critical_values() -> None:
    """Discordant pair counts chosen to reproduce the standard chi2(1) 0.05 critical value."""
    # (|b - c| - 1)^2 / (b + c) = (|29 - 15| - 1)^2 / 44 = 3.8409 (continuity-corrected).
    results_a = [_result("medical", ["medical"], row_id=f"q{i}") for i in range(29)] + [
        _result("legal", ["medical"], row_id=f"q{i}") for i in range(29, 44)
    ]
    results_b = [_result("legal", ["medical"], row_id=f"q{i}") for i in range(29)] + [
        _result("medical", ["medical"], row_id=f"q{i}") for i in range(29, 44)
    ]
    outcome = compute_mcnemar_test(results_a, results_b)
    assert outcome["discordant_a_only"] == 29
    assert outcome["discordant_b_only"] == 15
    assert outcome["chi2_statistic"] == pytest.approx(3.841, abs=0.01)
    assert outcome["p_value"] == pytest.approx(0.05, abs=0.005)


def test_compute_mcnemar_test_no_discordant_pairs_gives_p_value_one() -> None:
    """Identical selections on every row means no evidence of a difference (p=1.0)."""
    results_a = [_result("medical", ["medical"], row_id="q1")]
    results_b = [_result("medical", ["medical"], row_id="q1")]
    outcome = compute_mcnemar_test(results_a, results_b)
    assert outcome["discordant_pairs"] == 0
    assert outcome["chi2_statistic"] == 0.0
    assert outcome["p_value"] == 1.0


def test_compute_mcnemar_test_raises_on_mismatched_ids() -> None:
    """A McNemar test over rows that don't share the same id set is refused, not silently wrong."""
    results_a = [_result("medical", ["medical"], row_id="q1")]
    results_b = [_result("medical", ["medical"], row_id="q2")]
    with pytest.raises(ValueError):
        compute_mcnemar_test(results_a, results_b)


def test_compute_random_baseline_accuracy_uniform_single_domain_rows() -> None:
    """With 4 domains and single-domain rows, the random baseline is 1/4 per row."""
    results = [_result("medical", ["medical"]), _result("legal", ["legal"])]
    assert compute_random_baseline_accuracy(
        results, ["medical", "legal", "general", "education"]
    ) == pytest.approx(0.25)


def test_compute_random_baseline_accuracy_compound_rows_have_higher_expected_value() -> None:
    """A compound row with 2 acceptable domains out of 4 has hit probability 0.5."""
    results = [_result("medical", ["medical", "legal"])]
    assert compute_random_baseline_accuracy(
        results, ["medical", "legal", "general", "education"]
    ) == pytest.approx(0.5)


def test_compute_best_single_domain_baseline_per_domain_breakdown() -> None:
    """Each candidate domain's baseline is the fraction of rows it alone would answer correctly."""
    results = [
        _result("medical", ["medical"]),
        _result("medical", ["medical"]),
        _result("legal", ["legal"]),
    ]
    baseline = compute_best_single_domain_baseline(results, ["medical", "legal"])
    assert baseline["medical"] == pytest.approx(2 / 3)
    assert baseline["legal"] == pytest.approx(1 / 3)


def test_compute_oracle_accuracy_below_one_when_domain_unavailable() -> None:
    """Oracle accuracy drops below 1.0 when a row's domain isn't in available_domains."""
    results = [_result("medical", ["medical"]), _result(None, ["finance"])]
    assert compute_oracle_accuracy(results, ["medical", "legal"]) == pytest.approx(0.5)


def test_compute_cohens_kappa_perfect_agreement_is_one() -> None:
    """Perfect routing on single-domain rows gives kappa 1.0 regardless of class balance."""
    results = [
        _result("medical", ["medical"]),
        _result("medical", ["medical"]),
        _result("legal", ["legal"]),
    ]
    assert compute_cohens_kappa(results, ["medical", "legal"]) == pytest.approx(1.0)


def test_compute_cohens_kappa_chance_level_routing_is_near_zero() -> None:
    """Routing uncorrelated with the true label (balanced confusion) gives kappa near 0."""
    results = [
        _result("medical", ["medical"]),
        _result("legal", ["medical"]),
        _result("medical", ["legal"]),
        _result("legal", ["legal"]),
    ]
    assert compute_cohens_kappa(results, ["medical", "legal"]) == pytest.approx(0.0, abs=1e-9)


def test_compute_cohens_kappa_ignores_compound_rows() -> None:
    """A compound row (no single ground-truth class) does not affect the kappa computation."""
    single_domain_results = [_result("medical", ["medical"]), _result("legal", ["legal"])]
    with_compound = single_domain_results + [_result("medical", ["medical", "legal"])]
    assert compute_cohens_kappa(
        single_domain_results, ["medical", "legal"]
    ) == compute_cohens_kappa(with_compound, ["medical", "legal"])


def test_compute_cohens_kappa_raises_on_empty_domains_with_scorable_rows() -> None:
    """An empty domains list with real single-domain rows to score is a caller bug, not chance=0."""
    results = [_result("medical", ["medical"])]
    with pytest.raises(ValueError):
        compute_cohens_kappa(results, [])


def test_compute_cohens_kappa_empty_results_returns_zero_even_with_empty_domains() -> None:
    """No data at all is the legitimate empty case and still returns 0.0, not an error."""
    assert compute_cohens_kappa([], []) == 0.0


def test_compute_cohens_kappa_single_domain_full_agreement_does_not_divide_by_zero() -> None:
    """A single-domain dataset with perfect agreement drives chance_agreement to exactly 1.0;
    without the >= 1.0 guard, (observed - chance) / (1.0 - chance) would raise
    ZeroDivisionError instead of returning the well-defined kappa=1.0."""
    results = [_result("medical", ["medical"]), _result("medical", ["medical"])]
    assert compute_cohens_kappa(results, ["medical"]) == 1.0


def test_compute_all_metrics_includes_new_statistical_fields() -> None:
    """compute_all_metrics exposes Wilson CI, kappa, and the three baselines."""
    results = [_result("medical", ["medical"]), _result("legal", ["medical"])]
    metrics = compute_all_metrics(results)
    assert "top1_accuracy_wilson_ci" in metrics
    assert "cohens_kappa" in metrics
    assert "random_baseline_accuracy" in metrics
    assert "best_single_domain_baseline" in metrics
    assert "oracle_accuracy" in metrics


def test_compute_ece_perfectly_calibrated_is_zero() -> None:
    """Confidence exactly equal to the bin's observed accuracy yields ECE 0.0."""
    results = [
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("medical", ["medical"], confidence=0.9),
        _result("legal", ["medical"], confidence=0.9),  # 9/10 correct at confidence 0.9
    ]
    ece = compute_ece(results)
    assert ece["ece"] == pytest.approx(0.0, abs=1e-9)
    assert ece["n_rows"] == 10


def test_compute_ece_miscalibrated_matches_hand_computed_gap() -> None:
    """All rows in one bin, all wrong: ECE collapses to |mean_confidence - 0|."""
    results = [
        _result("legal", ["medical"], confidence=0.9),
        _result("legal", ["medical"], confidence=0.9),
    ]
    ece = compute_ece(results)
    assert ece["ece"] == pytest.approx(0.9, abs=1e-9)


def test_compute_ece_excludes_null_confidence_rows() -> None:
    """Fallback/dispatch-failed rows (confidence=None) are excluded, not treated as 0."""
    results = [
        _result("medical", ["medical"], confidence=1.0),
        _result(None, ["medical"], used_fallback=True, confidence=None),
    ]
    ece = compute_ece(results)
    assert ece["n_rows"] == 1
    assert ece["ece"] == pytest.approx(0.0, abs=1e-9)


def test_compute_ece_empty_results_is_zero() -> None:
    """No rows at all returns 0.0/0 rather than dividing by zero."""
    assert compute_ece([]) == {"ece": 0.0, "n_rows": 0}


def test_compute_brier_score_perfect_predictions_is_zero() -> None:
    """Confidence 1.0 on a correct row and 0.0 on an incorrect one both score 0 squared error."""
    results = [
        _result("medical", ["medical"], confidence=1.0),
        _result("legal", ["medical"], confidence=0.0),
    ]
    assert compute_brier_score(results)["brier_score"] == pytest.approx(0.0, abs=1e-9)


def test_compute_brier_score_worst_case_is_one() -> None:
    """Full confidence on a wrong answer is the maximum possible squared error."""
    results = [_result("legal", ["medical"], confidence=1.0)]
    assert compute_brier_score(results)["brier_score"] == pytest.approx(1.0, abs=1e-9)


def test_compute_auroc_perfect_discrimination_is_one() -> None:
    """Correct rows all outrank incorrect rows in confidence -> AUROC 1.0."""
    results = [
        _result("medical", ["medical"], confidence=0.9),
        _result("legal", ["legal"], confidence=0.8),
        _result("legal", ["medical"], confidence=0.2),
        _result("medical", ["legal"], confidence=0.1),
    ]
    assert compute_auroc(results)["auroc"] == pytest.approx(1.0, abs=1e-9)


def test_compute_auroc_undefined_with_single_class() -> None:
    """AUROC needs both a correct and an incorrect row to be defined; all-correct returns None."""
    results = [
        _result("medical", ["medical"], confidence=0.9),
        _result("legal", ["legal"], confidence=0.5),
    ]
    result = compute_auroc(results)
    assert result["auroc"] is None
    assert result["n_rows"] == 2


def test_compute_tie_rate_detects_exact_top_two_tie() -> None:
    """Two candidates reporting the identical top confidence counts as a tie."""
    results = [
        _result(
            "medical",
            ["medical"],
            probe_candidates=[{"confidence": 0.5}, {"confidence": 0.5}, {"confidence": 0.1}],
        )
    ]
    tie = compute_tie_rate(results)
    assert tie["tie_rate"] == pytest.approx(1.0, abs=1e-9)
    assert tie["n_rows"] == 1


def test_compute_tie_rate_no_tie_when_top_two_differ() -> None:
    """Distinct top-two confidences are not a tie."""
    results = [
        _result(
            "medical",
            ["medical"],
            probe_candidates=[{"confidence": 0.9}, {"confidence": 0.5}],
        )
    ]
    assert compute_tie_rate(results)["tie_rate"] == pytest.approx(0.0, abs=1e-9)


def test_compute_tie_rate_skips_rows_without_probe_candidates() -> None:
    """Rows predating the probe_candidates field (None) are excluded, matching
    compute_compound_coverage_metrics's backward-compatibility convention."""
    results = [_result("medical", ["medical"], probe_candidates=None)]
    assert compute_tie_rate(results) == {"tie_rate": 0.0, "n_rows": 0}


def test_compute_confidence_dispersion_matches_hand_computed_std_and_sum() -> None:
    """A single probe round with confidences [0.2, 0.8] has population std 0.3 and sum 1.0."""
    results = [
        _result(
            "medical",
            ["medical"],
            probe_candidates=[{"confidence": 0.2}, {"confidence": 0.8}],
        )
    ]
    dispersion = compute_confidence_dispersion(results)
    assert dispersion["mean_confidence_std"] == pytest.approx(0.3, abs=1e-9)
    assert dispersion["mean_confidence_sum"] == pytest.approx(1.0, abs=1e-9)
    assert dispersion["n_rows"] == 1


def test_compute_all_metrics_includes_calibration_fields() -> None:
    """compute_all_metrics exposes ECE, Brier score, AUROC, tie rate, and dispersion."""
    results = [_result("medical", ["medical"], confidence=0.9, probe_candidates=[{"confidence": 0.9}])]
    metrics = compute_all_metrics(results)
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert "auroc" in metrics
    assert "tie_rate" in metrics
    assert "confidence_dispersion" in metrics

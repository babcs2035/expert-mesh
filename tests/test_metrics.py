"""Tests for the routing-accuracy metrics computed from run_experiment.py output."""

from metrics import (
    compute_all_metrics,
    compute_compound_coverage_metrics,
    compute_dispatch_failure_rate,
    compute_fallback_rate,
    compute_mean_duration_ms,
    compute_misrouting_rate,
    compute_precision_recall_per_domain,
    compute_top1_accuracy,
)


def _result(
    selected_domain: str | None,
    expected_domains: list[str],
    used_fallback: bool = False,
    dispatch_failed: bool = False,
    duration_ms: int = 100,
    dispatched_domains: list[str] | None = None,
) -> dict:
    """Build a minimal result row matching run_experiment.py's output shape.

    `dispatched_domains` defaults to None so existing callers keep
    producing rows that match the pre-Iter1 results.jsonl schema (no such
    key at all in practice, but `.get(...)` treats a present-but-None value
    the same way) — this is what compute_compound_coverage_metrics's
    backward-compatibility skip is meant to exercise.
    """
    return {
        "selected_domain": selected_domain,
        "expected_domains": expected_domains,
        "used_fallback": used_fallback,
        "dispatch_failed": dispatch_failed,
        "duration_ms": duration_ms,
        "dispatched_domains": dispatched_domains,
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
    results = [_result("medical", ["medical"], duration_ms=100), _result("legal", ["legal"], duration_ms=300)]
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
        _result(
            "medical", ["medical", "legal"], dispatched_domains=["medical", "legal"]
        ),
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
        _result("medical", ["medical", "legal"]),  # present but None (dispatched_domains defaults to None)
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

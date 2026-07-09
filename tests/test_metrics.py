"""Tests for the routing-accuracy metrics computed from run_experiment.py output."""

from benchmark.metrics import (
    compute_all_metrics,
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
) -> dict:
    """Build a minimal result row matching run_experiment.py's output shape."""
    return {
        "selected_domain": selected_domain,
        "expected_domains": expected_domains,
        "used_fallback": used_fallback,
        "dispatch_failed": dispatch_failed,
        "duration_ms": duration_ms,
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

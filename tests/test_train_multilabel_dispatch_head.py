"""Tests for Iter60's MultiLabelBinarizer-based OvR head training
(scripts/train_multilabel_dispatch_head.py).
"""

import numpy as np
import pytest

from scripts.train_multilabel_dispatch_head import (
    _A0_MINIMUM_MULTILABEL_ROW_COUNT,
    _assert_a0_true_multilabel_signal,
    _covered_domain_pairs,
    _normalize_labels,
    build_multilabel_targets,
    train_multilabel_ranking_head,
)


def test_normalize_labels_wraps_bare_strings_and_passes_lists_through() -> None:
    """build_training_features() returns a heterogeneous labels list (str for single-label
    rows, list[str] for this iteration's synthetic two-domain rows); normalization must
    make every element a list before MultiLabelBinarizer sees it."""
    labels = ["medical", ["legal", "medical"], "legal"]

    normalized = _normalize_labels(labels)

    assert normalized == [["medical"], ["legal", "medical"], ["legal"]]


def test_build_multilabel_targets_produces_a_true_multilabel_indicator_matrix() -> None:
    """MultiLabelBinarizer's classes_ (not the eventual OneVsRestClassifier's) is the
    surviving domain-name-per-column mapping; this test locks in that ordering
    contract, since scripts/evaluate_dispatch_candidate_ranking.py's _load_head()
    depends on it exactly matching Y's column order."""
    labels = ["medical", "legal", ["legal", "medical"]]

    Y, mlb = build_multilabel_targets(labels)

    assert list(mlb.classes_) == ["legal", "medical"]
    np.testing.assert_array_equal(Y, np.array([[0, 1], [1, 0], [1, 1]]))


_ALL_TEN_DOMAINS = [
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


def test_a0_assertion_passes_and_returns_the_multilabel_row_count_at_the_floor() -> None:
    """A0 succeeds (no raise) once the >=2-label row count exactly matches the caller's
    synthetic row count, clears the pre-registered 120-row floor, and all 10 production
    domains are present -- and returns that count."""
    n_multilabel = _A0_MINIMUM_MULTILABEL_ROW_COUNT
    labels = list(_ALL_TEN_DOMAINS) + [["legal", "medical"]] * n_multilabel
    Y, mlb = build_multilabel_targets(labels)

    result = _assert_a0_true_multilabel_signal(Y, mlb, n_synthetic_rows=n_multilabel)

    assert result == n_multilabel


def test_a0_assertion_fails_when_multilabel_row_count_disagrees_with_synthetic_count() -> None:
    """A0 must catch a mis-join: if the actual multi-label row count in Y does not equal
    the caller-supplied synthetic row count, something upstream dropped, duplicated, or
    mislabeled rows -- this is exactly the silent-no-op failure mode this lever exists
    to prevent (Iter59's teacher signal was 100% single-label)."""
    n_multilabel = _A0_MINIMUM_MULTILABEL_ROW_COUNT
    labels = ["medical"] * 10 + ["legal"] * 10 + [["legal", "medical"]] * n_multilabel
    Y, mlb = build_multilabel_targets(labels)

    with pytest.raises(AssertionError, match="A0 failed"):
        _assert_a0_true_multilabel_signal(Y, mlb, n_synthetic_rows=n_multilabel - 1)


def test_a0_assertion_fails_below_minimum_row_count_floor() -> None:
    """A0's pre-registered floor (120 rows) must reject small synthetic sets even when
    the row count matches the caller's count exactly, so a too-small generation run
    cannot silently pass."""
    labels = ["medical"] * 3 + ["legal"] * 3 + [["legal", "medical"]] * 2
    Y, mlb = build_multilabel_targets(labels)

    with pytest.raises(AssertionError, match="below the pre-registered floor"):
        _assert_a0_true_multilabel_signal(Y, mlb, n_synthetic_rows=2)


def test_a0_assertion_fails_on_wrong_domain_count() -> None:
    """A0 must catch a domain-count regression (e.g. a bug that drops a domain entirely)."""
    n_multilabel = _A0_MINIMUM_MULTILABEL_ROW_COUNT
    labels = ["medical"] * 10 + [["legal", "medical"]] * n_multilabel
    Y, mlb = build_multilabel_targets(labels)

    with pytest.raises(AssertionError, match="2 domains, expected 10"):
        _assert_a0_true_multilabel_signal(Y, mlb, n_synthetic_rows=n_multilabel)


def test_covered_domain_pairs_counts_distinct_two_domain_combinations() -> None:
    """_covered_domain_pairs() reports how many of the 45 possible pairs actually
    appear among the synthetic rows, for the training script's stderr diagnostic."""
    labels = [["legal", "medical"], ["medical", "legal"], ["legal", "education"], "medical"]

    pairs = _covered_domain_pairs(labels)

    # ["legal","medical"] and ["medical","legal"] are the same pair (order-independent).
    assert pairs == {frozenset({"legal", "medical"}), frozenset({"legal", "education"})}


def test_train_multilabel_ranking_head_fits_a_model_that_predicts_seen_multilabel_rows() -> None:
    """A model trained on separable multi-label data recovers both positive labels
    for a row it was trained on (sanity check that Y, not a 1D label array, is
    actually reaching OneVsRestClassifier.fit()).

    Iter62 (multilabel_rank2_score_calibration=per_domain_holdout_calibration)
    wraps the base LogisticRegression in CalibratedClassifierCV(cv=_CALIBRATION_CV),
    which raises ValueError unless every binary sub-problem's y has at least
    _CALIBRATION_CV examples of each class (sklearn 1.9.0 requirement) -- hence
    6 rows per cluster below (Iter59/60/61's fixture used 2 rows per cluster,
    which is no longer sufficient)."""
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.05],
        [1.0, 0.1],
        [1.0, -0.05],
        [1.0, -0.1],
        [1.0, 0.02],
        [0.0, 1.0],
        [0.0, 1.05],
        [0.0, 1.1],
        [0.0, 0.95],
        [0.0, 0.9],
        [0.0, 1.02],
        [1.0, 1.0],
        [1.0, 1.05],
        [1.0, 0.95],
        [1.0, 1.1],
        [1.0, 0.9],
        [1.0, 1.02],
    ]
    labels = (
        ["medical"] * 6
        + ["legal"] * 6
        + [["legal", "medical"]] * 6
    )
    Y, mlb = build_multilabel_targets(labels)

    model = train_multilabel_ranking_head(embeddings, Y)
    predicted = model.predict([[1.0, 1.0]])[0]

    predicted_domains = {domain for domain, flag in zip(mlb.classes_, predicted) if flag}
    assert predicted_domains == {"legal", "medical"}


def test_multilabel_binarizer_rejects_nothing_but_direct_list_of_lists_would_fail() -> None:
    """Regression guard for the sklearn 1.9.0 pitfall documented in journal.md Iter60
    investigation Q1: OneVsRestClassifier.fit() with a raw list-of-lists target raises,
    which is exactly why build_multilabel_targets()'s MultiLabelBinarizer step exists."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    with pytest.raises(ValueError, match="legacy multi-label"):
        OneVsRestClassifier(LogisticRegression()).fit(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [["a"], ["b"], ["a", "b"]]
        )

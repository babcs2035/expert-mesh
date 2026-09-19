"""Tests for scripts/evaluate_classifier_calibration.py's qhat_source switch (Iter69).

Regression guard against the Iter56 q_hat bug: `_compute_prediction_set()` originally
computed q_hat from ALL (sample, class) nonconformity scores instead of the standard
APS true-class-only population, causing under-coverage. This test verifies that the
new `qhat_source` argument actually reaches the quantile computation (past reflector
notes recorded 6 prior no-op failures where a config change never reached the running
code), by asserting q_hat and the resulting prediction set size differ between
qhat_source="all" (default, unchanged behavior) and qhat_source="true_class".
"""

import numpy as np
import pytest

from scripts.evaluate_classifier_calibration import _compute_prediction_set

# 10-class toy calibration data (matching the production classifier's 10 domain
# classes), deliberately constructed with all non-true-class columns kept small
# (0.03-0.18) and the true-class column (index 1) kept large (0.58-0.95). This
# mirrors the real data's shape (Iter69 journal: true-class 90th percentile
# 0.5956 >> all-scores 90th percentile 0.3865), so the two qhat_source modes'
# resulting q_hat -- and therefore the prediction set they produce -- must diverge.
_N_CLASSES = 10
_TRUE_CLASS_COLUMN = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.62, 0.68, 0.72, 0.90, 0.58, 0.95]
_N_CAL = len(_TRUE_CLASS_COLUMN)
_ALL_SCORES = np.array(
    [
        [0.05, t, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15, 0.05, 0.15]
        for t in _TRUE_CLASS_COLUMN
    ]
)
# True class index per calibration sample (arbitrary but fixed labeling used to
# derive true_class_scores the same way predict_calibrated_rows() does).
_LABELS = [1] * _N_CAL
_TRUE_CLASS_SCORES = np.array([_ALL_SCORES[i, _LABELS[i]] for i in range(_N_CAL)])

_CP_DATA = {"all_scores": _ALL_SCORES, "true_class_scores": _TRUE_CLASS_SCORES}

# Query-time probabilities for a single evaluation row over the 10 classes,
# chosen so the top class's nonconformity score (1 - max_prob = 0.70) falls
# strictly between q_hat("all") (~0.58) and q_hat("true_class") (~0.95): it is
# excluded under "all" (triggering the top-class fallback, set size 1) but
# included under "true_class" (where every subsequent, smaller score also
# clears the higher q_hat, so all 10 classes end up in the set).
_QUERY_PROBABILITIES = np.array([0.30, 0.11, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.04, 0.04])


def test_qhat_source_all_and_true_class_yield_different_q_hat() -> None:
    """The two qhat_source populations must produce numerically distinct q_hat values.

    This is the direct regression check for the Iter56 bug: q_hat computed from
    ALL (sample, class) scores differs from q_hat computed from TRUE-CLASS-only
    scores whenever the true class is not uniformly distributed across the score
    matrix (the realistic case), so a correct implementation must show a gap here.
    """
    _, size_all = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90, qhat_source="all"
    )
    _, size_true_class = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90, qhat_source="true_class"
    )
    assert size_all != size_true_class, (
        "qhat_source='all' and 'true_class' produced the same prediction set size; "
        "the qhat_source branch is not actually being reached (Iter69 no-op guard)."
    )
    # Exact expected sizes for this toy dataset (see module-level comment on
    # _QUERY_PROBABILITIES for the derivation): "all" falls back to the top
    # class only, "true_class" admits every class.
    assert size_all == 1
    assert size_true_class == _N_CLASSES


def test_qhat_source_true_class_uses_only_true_class_score_population() -> None:
    """qhat_source='true_class' must ignore cp_data['all_scores'] entirely.

    Verified by monkeypatching all_scores to an empty/garbage placeholder that
    would raise if flattened and quantiled; true_class mode must not touch it.
    """
    cp_data_with_poisoned_all_scores = {
        "all_scores": np.array([]),  # would raise on np.quantile if used
        "true_class_scores": _TRUE_CLASS_SCORES,
    }
    pred_set, set_size = _compute_prediction_set(
        _QUERY_PROBABILITIES,
        cp_data_with_poisoned_all_scores,
        confidence_level=0.90,
        qhat_source="true_class",
    )
    assert set_size == len(pred_set)
    assert set_size >= 1


def test_qhat_source_default_is_all_for_backward_compatibility() -> None:
    """Omitting qhat_source must reproduce the pre-Iter69 'all' behavior exactly."""
    pred_set_default, size_default = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90
    )
    pred_set_explicit_all, size_explicit_all = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90, qhat_source="all"
    )
    assert pred_set_default == pred_set_explicit_all
    assert size_default == size_explicit_all


def test_qhat_source_rejects_unknown_value() -> None:
    """An unsupported qhat_source string must raise, not silently fall back to 'all'."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90, qhat_source="bogus"
        )

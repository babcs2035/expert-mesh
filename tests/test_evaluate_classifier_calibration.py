"""Tests for scripts/evaluate_classifier_calibration.py's qhat_source (Iter69),
set_construction (Iter70), and qhat_quantile_direction (Iter71) switches.

Regression guard against the Iter56 q_hat bug: `_compute_prediction_set()` originally
computed q_hat from ALL (sample, class) nonconformity scores instead of the standard
APS true-class-only population, causing under-coverage. This test verifies that the
new `qhat_source` argument actually reaches the quantile computation (past reflector
notes recorded 6 prior no-op failures where a config change never reached the running
code), by asserting q_hat and the resulting prediction set size differ between
qhat_source="all" (default, unchanged behavior) and qhat_source="true_class".

Regression guard against the Iter56/69 set-construction bug (diagnosed Iter70): the
pre-Iter70 ("broken") construction appended a class to the prediction set only AFTER
the cumsum-based break check, which collapses to a binary `1 - p_max <= q_hat` gate on
the top class alone and can never produce an intermediate set size. The Iter70 tests
below verify that `set_construction="corrected_aps"` (append BEFORE the break check)
actually reaches a different, intermediate-sized prediction set on the same input.

Regression guard against the Iter56-70 q_hat quantile-direction bug (diagnosed
Iter71): this repo's nonconformity score is the COMPLEMENT of the standard APS score
(`score = 1 - cumsum`, not `cumsum` itself), so q_hat must be drawn from the score
population's alpha quantile (`qhat_quantile_direction="alpha_lower"`), not the
(1-alpha) quantile the pre-Iter71 implementation used
(`qhat_quantile_direction="upper"`, kept as default for backward compatibility). The
Iter71 tests below verify the new argument actually reaches the quantile computation
and that its finite-sample correction matches the standard `floor((n+1)*alpha)/n`
rank statistic (Angelopoulos & Bates 2021; Barber et al. 2021).
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


# --- Iter70: set_construction ("broken" vs "corrected_aps") ---------------------

# A 5-class toy input, chosen so q_hat (fixed at exactly 0.15 by making every
# true-class calibration score equal to 0.15, which forces any quantile of that
# population to be 0.15 regardless of the finite-sample correction target) falls
# strictly between the cumulative score after class 1 (rank-2, score=0.20) and
# after class 2 (rank-3, score=0.10). Under "broken", the top class's own score
# (0.5) already exceeds q_hat, so the loop breaks on the very first iteration
# before ever appending -- collapsing to the empty-set fallback (size 1). Under
# "corrected_aps", classes are appended before the break check, so the set grows
# to rank-3 (size 3) -- the intermediate size that "broken" can never produce.
_ITER70_PROBABILITIES = np.array([0.5, 0.3, 0.1, 0.06, 0.04])
_ITER70_CP_DATA = {
    "all_scores": np.array([[0.15] * 5]),  # unused by qhat_source="true_class"
    "true_class_scores": np.full(20, 0.15),
}


def test_set_construction_broken_and_corrected_aps_yield_different_set_sizes() -> None:
    """The two set_construction modes must diverge on an input with an intermediate crossing.

    Direct regression check for the Iter56/69 set-construction bug (diagnosed
    Iter70): "broken" collapses to a binary 1-p_max<=q_hat gate and can only
    return size 1 or n_classes, never an intermediate size.
    """
    pred_set_broken, size_broken = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="broken",
    )
    pred_set_corrected, size_corrected = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
    )
    assert pred_set_broken == [0]
    assert size_broken == 1
    assert pred_set_corrected == [0, 1, 2]
    assert size_corrected == 3
    assert size_broken != size_corrected


def test_set_construction_corrected_aps_always_includes_top_class() -> None:
    """corrected_aps must append the argmax class on its first loop iteration.

    Standard APS guarantees the top-ranked class is always in the prediction
    set (it is appended before any break check can fire).
    """
    pred_set, _ = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
    )
    top_class = int(np.argmax(_ITER70_PROBABILITIES))
    assert top_class in pred_set
    assert pred_set[0] == top_class


def test_set_construction_corrected_aps_stops_at_first_cumulative_crossing() -> None:
    """corrected_aps must stop at the first class whose cumulative probability
    crosses the q_hat threshold, not one class earlier or later.

    With q_hat=0.15, the set must include rank-3 (cumulative score 0.10 <= 0.15,
    crossing achieved) but the rank-2-only prefix (cumulative score 0.20 > 0.15,
    not yet crossed) must not be sufficient on its own.
    """
    pred_set, size = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
    )
    assert size == 3  # one class past the rank-2 prefix, which alone would not cross q_hat
    cumsum_up_to_rank2 = float(np.sum(_ITER70_PROBABILITIES[:2]))
    assert 1.0 - cumsum_up_to_rank2 > 0.15  # rank-2 prefix alone does not cross q_hat
    cumsum_up_to_rank3 = float(np.sum(_ITER70_PROBABILITIES[:3]))
    assert 1.0 - cumsum_up_to_rank3 <= 0.15  # rank-3 prefix is where the loop breaks


def test_set_construction_default_is_broken_for_backward_compatibility() -> None:
    """Omitting set_construction must reproduce the pre-Iter70 'broken' behavior exactly."""
    pred_set_default, size_default = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90, qhat_source="true_class"
    )
    pred_set_explicit_broken, size_explicit_broken = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="broken",
    )
    assert pred_set_default == pred_set_explicit_broken
    assert size_default == size_explicit_broken


def test_set_construction_rejects_unknown_value() -> None:
    """An unsupported set_construction string must raise, not silently fall back to 'broken'."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
            set_construction="bogus",
        )


# --- Iter71: qhat_quantile_direction ("upper" vs "alpha_lower") -----------------

# Reuse the 10-class true-class score population from the qhat_source tests above
# (n_cal=12, true_class_scores=_TRUE_CLASS_COLUMN). For confidence_level=0.90
# (alpha=0.10): q_hat("upper") = quantile(scores, min(1, 0.9*13/12), method="higher")
# = 0.95 (the max score); q_hat("alpha_lower") = quantile(scores, 0.1*13/12,
# method="lower") = 0.60. Under "corrected_aps" construction and
# _QUERY_PROBABILITIES (top-class cumsum=0.30, score=0.70), the top class alone
# satisfies score<=q_hat("upper")=0.95 (set size 1), but score=0.70 > q_hat
# ("alpha_lower")=0.60 requires one more class (cumsum=0.41, score=0.59<=0.60,
# set size 2) -- the two directions must diverge, and the correct ("alpha_lower")
# direction must yield the LARGER set (it needs more cumulative probability mass,
# 1 - 0.60 = 0.40, vs 1 - 0.95 = 0.05 for "upper").
def test_qhat_quantile_direction_upper_and_alpha_lower_yield_different_q_hat() -> None:
    """The two qhat_quantile_direction modes must produce numerically distinct q_hat,
    with alpha_lower's resulting prediction set strictly larger than upper's.

    Direct regression check for the Iter56-70 quantile-direction bug (diagnosed
    Iter71): "upper" draws q_hat from the (1-alpha) quantile of the complement
    score, which is the wrong side and under-covers; "alpha_lower" draws it from
    the alpha quantile, the correct side for a complement score.
    """
    pred_set_upper, size_upper = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
        qhat_quantile_direction="upper",
    )
    pred_set_alpha_lower, size_alpha_lower = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
        qhat_quantile_direction="alpha_lower",
    )
    assert size_upper != size_alpha_lower, (
        "qhat_quantile_direction='upper' and 'alpha_lower' produced the same "
        "prediction set size; the qhat_quantile_direction branch is not actually "
        "being reached (Iter71 no-op guard)."
    )
    assert size_upper == 1
    assert pred_set_upper == [0]
    assert size_alpha_lower == 2
    assert pred_set_alpha_lower == [0, 1]
    assert size_alpha_lower > size_upper, (
        "alpha_lower must draw a smaller q_hat than upper for this complement-score "
        "population, requiring MORE cumulative probability mass and thus a LARGER "
        "prediction set."
    )


def test_qhat_quantile_direction_alpha_lower_matches_finite_sample_rank_statistic() -> None:
    """alpha_lower's q_hat must equal the standard floor((n+1)*alpha)/n rank
    statistic (Angelopoulos & Bates 2021; Barber et al. 2021), not an ad hoc
    approximation.

    Uses a synthetic true-class score population of n=19 evenly spaced values
    (0.05, 0.10, ..., 0.95) where the finite-sample-corrected rank
    floor((19+1)*0.10) = 2 picks out the 2nd-smallest score, 0.10, exactly
    (verified independently via np.quantile in this test's setup, matching the
    journal Iter71 plan's Q3 derivation). Query-time probabilities are crafted
    with a wide margin (score=0.15 just before the expected crossing, score=0.05
    just after) so the assertion is robust to floating-point rounding rather than
    depending on an exact equality at the crossing point.
    """
    n = 19
    alpha = 0.10  # confidence_level=0.90
    evenly_spaced_scores = np.array([i / 20 for i in range(1, n + 1)])
    expected_q_hat = 0.10  # floor((n+1)*alpha)/n-th order statistic, see docstring
    assert float(np.quantile(evenly_spaced_scores, alpha * (1 + 1 / n), method="lower")) == expected_q_hat

    cp_data = {
        "all_scores": np.zeros((1, 6)),  # unused by qhat_source="true_class"
        "true_class_scores": evenly_spaced_scores,
    }
    # cumsum: 0.30, 0.50, 0.70, 0.85, 0.95, 1.00 -> score: 0.70, 0.50, 0.30, 0.15, 0.05, 0.00
    probabilities = np.array([0.30, 0.20, 0.20, 0.15, 0.10, 0.05])
    pred_set, size = _compute_prediction_set(
        probabilities, cp_data, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
        qhat_quantile_direction="alpha_lower",
    )
    # Rank-4 prefix (cumsum=0.85, score=0.15) must NOT yet satisfy score<=q_hat=0.10;
    # rank-5 prefix (cumsum=0.95, score=0.05) must be where the loop breaks.
    assert size == 5
    assert pred_set == [0, 1, 2, 3, 4]


def test_qhat_quantile_direction_default_is_upper_for_backward_compatibility() -> None:
    """Omitting qhat_quantile_direction must reproduce the pre-Iter71 'upper' behavior exactly."""
    pred_set_default, size_default = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
    )
    pred_set_explicit_upper, size_explicit_upper = _compute_prediction_set(
        _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
        qhat_quantile_direction="upper",
    )
    assert pred_set_default == pred_set_explicit_upper
    assert size_default == size_explicit_upper


def test_qhat_quantile_direction_rejects_unknown_value() -> None:
    """An unsupported qhat_quantile_direction string must raise, not silently fall back to 'upper'."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _QUERY_PROBABILITIES, _CP_DATA, confidence_level=0.90,
            qhat_quantile_direction="bogus",
        )

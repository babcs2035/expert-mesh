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

Regression guard against the Iter56-71 exchangeability bug (diagnosed Iter72): the
calibration nonconformity scores came from a freshly-refit OOF LogisticRegression
while evaluation scores came from the already-fitted CalibratedClassifierCV -- two
different models, violating the exchangeability assumption split conformal requires.
The Iter72 tests below verify the new `calibration_source` argument actually reaches
`predict_calibrated_rows()` and produces a stratified 50/50 calibration/evaluation
split scored by the SAME classifier used for evaluation.

Regression guard against the Iter72 residual over-coverage bug (diagnosed Iter73):
the non-randomized "corrected_aps" construction implicitly mixes a calibration-side
score computed as if u=0 with an evaluation-side rule equivalent to u=1, always
admitting the crossing class wholesale. The Iter73 tests below verify the new
`set_construction="randomized_aps"` argument (a) reduces to "corrected_aps" at
randomization_u=1.0, (b) is monotonically non-increasing in randomization_u,
(c)/(d) rejects a missing u and an incompatible calibration_source rather than
silently falling back, and (e) is reproducible for a fixed randomization_seed.

Iter74 adds `set_construction="raps_penalty"` (RAPS size regularization,
Angelopoulos et al. 2021, arXiv:2009.14193): the same randomized-APS inclusion
rule plus a rank-based penalty `raps_lambda * max(0, rank - raps_k_reg)`. The
Iter74 tests below verify (a) it degenerates to "randomized_aps" exactly at
raps_lambda=0.0, (b) increasing raps_lambda can only shrink or hold the set size
(monotonicity), (c) it rejects a missing raps_lambda/raps_k_reg/randomization_u
rather than silently falling back to the un-penalized rule, and (d) it rejects an
incompatible calibration_source (the same "oof_train" restriction as
randomized_aps, since raps_penalty is built on top of it).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from scripts.evaluate_classifier_calibration import _compute_prediction_set, predict_calibrated_rows

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


# --- Iter72: calibration_source ("oof_train" vs "eval_holdout") -----------------

# A 20-row synthetic dataset, perfectly balanced 10/10 across two classes, so a
# stratified 50/50 split must land exactly 5/5 per class (10/10 overall) with no
# rounding ambiguity. classifier.predict_proba is mocked to a fixed 2-column
# array (ignores the actual embedding values), matching the fake OllamaClient's
# fixed-length embedding stub below.
_ITER72_N_ROWS = 20
_ITER72_DATASET = [
    {
        "id": i,
        "query": f"query {i}",
        "expected_domains": ["a"] if i % 2 == 0 else ["b"],
    }
    for i in range(_ITER72_N_ROWS)
]


def _make_iter72_classifier() -> MagicMock:
    """A classifier stub whose predict_proba always returns a fixed 2-class distribution.

    The exact probabilities don't matter for the split/reproducibility tests
    below (they only inspect cp_data population size and the `split` field),
    so a constant matrix keeps the stub simple.
    """
    classifier = MagicMock()
    classifier.classes_ = ["a", "b"]

    def _predict_proba(embeddings: np.ndarray) -> np.ndarray:
        return np.tile([0.9, 0.1], (len(embeddings), 1))

    classifier.predict_proba.side_effect = _predict_proba
    return classifier


def _make_iter72_ollama_client() -> MagicMock:
    """A fake OllamaClient whose embed() returns a fixed-length placeholder vector."""
    client = MagicMock()
    client.embed = AsyncMock(return_value=[1.0, 0.0])
    return client


def test_calibration_source_eval_holdout_splits_50_50_stratified() -> None:
    """eval_holdout must produce a stratified 50/50 (10/10) calibration/evaluation split.

    Regression guard for the Iter72 lever: the calibration population size
    (n_cal) must be exactly half of the dataset, matching the plan's
    StratifiedShuffleSplit(test_size=0.5) configuration.
    """
    rows = asyncio.run(
        predict_calibrated_rows(
            _make_iter72_ollama_client(), "fake-embed-model",
            _make_iter72_classifier(), _ITER72_DATASET,
            conformal_prediction=True, confidence_level=0.90,
            qhat_source="true_class", set_construction="corrected_aps",
            calibration_source="eval_holdout", holdout_seed=42,
        )
    )
    splits = [row["split"] for row in rows]
    assert splits.count("cal") == _ITER72_N_ROWS // 2
    assert splits.count("eval") == _ITER72_N_ROWS // 2
    cal_domains = [
        row["expected_domains"][0] for row in rows if row["split"] == "cal"
    ]
    # Stratification: each class must be split evenly (5/5) into the cal half.
    assert cal_domains.count("a") == 5
    assert cal_domains.count("b") == 5


def test_calibration_source_eval_holdout_is_reproducible_for_same_seed() -> None:
    """The same holdout_seed must produce an identical cal/eval assignment across runs."""
    kwargs = dict(
        conformal_prediction=True, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
        calibration_source="eval_holdout", holdout_seed=42,
    )
    rows_a = asyncio.run(
        predict_calibrated_rows(
            _make_iter72_ollama_client(), "fake-embed-model",
            _make_iter72_classifier(), _ITER72_DATASET, **kwargs,
        )
    )
    rows_b = asyncio.run(
        predict_calibrated_rows(
            _make_iter72_ollama_client(), "fake-embed-model",
            _make_iter72_classifier(), _ITER72_DATASET, **kwargs,
        )
    )
    assert [r["split"] for r in rows_a] == [r["split"] for r in rows_b]


def test_calibration_source_rejects_unknown_value() -> None:
    """An unsupported calibration_source string must raise, not silently fall back to 'oof_train'."""
    with pytest.raises(ValueError):
        asyncio.run(
            predict_calibrated_rows(
                _make_iter72_ollama_client(), "fake-embed-model",
                _make_iter72_classifier(), _ITER72_DATASET,
                conformal_prediction=True, calibration_source="bogus",
            )
        )


def test_calibration_source_default_is_oof_train_for_backward_compatibility() -> None:
    """Omitting calibration_source must take the pre-Iter72 oof_train code path.

    The oof_train path requires calibration_dataset_path to point at a real
    JSONL file (data/classifier_train.jsonl in production); passing None (the
    default) must surface as a TypeError from the file-open call inside
    _read_jsonl(), not as a silent eval_holdout fallback that would ignore the
    missing calibration dataset entirely.
    """
    with pytest.raises(TypeError):
        asyncio.run(
            predict_calibrated_rows(
                _make_iter72_ollama_client(), "fake-embed-model",
                _make_iter72_classifier(), _ITER72_DATASET,
                conformal_prediction=True, calibration_dataset_path=None,
            )
        )


def test_calibration_source_eval_holdout_rejects_qhat_source_all() -> None:
    """eval_holdout only supports qhat_source='true_class'; 'all' must raise, not silently ignore."""
    with pytest.raises(ValueError):
        asyncio.run(
            predict_calibrated_rows(
                _make_iter72_ollama_client(), "fake-embed-model",
                _make_iter72_classifier(), _ITER72_DATASET,
                conformal_prediction=True, calibration_source="eval_holdout",
                qhat_source="all",
            )
        )


# --- Iter73: set_construction="randomized_aps" (Romano et al., 2020 U-term) -----

# Reuse the Iter70 5-class toy input/cp_data (q_hat fixed at 0.15 by construction).


def test_randomized_aps_at_u_1_matches_corrected_aps() -> None:
    """randomization_u=1.0 must reproduce 'corrected_aps' exactly for any input.

    Both constructions reduce algebraically to "1 - cumsum_before_idx >= q_hat"
    at u=1 (see scripts/evaluate_classifier_calibration.py's _compute_prediction_set
    docstring); this is the u=1 equivalence the Iter73 plan registered as the
    primary correctness check for the new branch.
    """
    pred_set_corrected, size_corrected = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="corrected_aps",
    )
    pred_set_randomized, size_randomized = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="randomized_aps",
        randomization_u=1.0,
    )
    assert pred_set_randomized == pred_set_corrected
    assert size_randomized == size_corrected


def test_randomized_aps_at_u_0_is_smaller_or_equal_to_u_1() -> None:
    """Set size must be monotonically non-increasing in randomization_u.

    Lowering u drops the `+ u*probabilities[idx]` slack term from the crossing
    class's score, making the >= q_hat append condition strictly harder to
    satisfy, so the resulting prediction set can only shrink or stay the same.
    """
    _, size_u0 = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="randomized_aps",
        randomization_u=0.0,
    )
    _, size_u1 = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="randomized_aps",
        randomization_u=1.0,
    )
    assert size_u0 <= size_u1


def test_randomized_aps_requires_randomization_u() -> None:
    """Omitting randomization_u must raise, not silently fall back to a fixed u."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
            qhat_source="true_class", set_construction="randomized_aps",
        )


def test_randomized_aps_rejects_oof_train_calibration_source() -> None:
    """randomized_aps requires calibration_source='eval_holdout' (shared row-indexed u).

    'oof_train' calibrates over a separate dataset (classifier_train.jsonl) with
    no shared row indexing against the evaluation dataset, so the same u cannot
    be applied on both sides; this must raise rather than silently using a
    mismatched u.
    """
    with pytest.raises(ValueError):
        asyncio.run(
            predict_calibrated_rows(
                _make_iter72_ollama_client(), "fake-embed-model",
                _make_iter72_classifier(), _ITER72_DATASET,
                conformal_prediction=True, calibration_source="oof_train",
                calibration_dataset_path=None,
                qhat_source="true_class", set_construction="randomized_aps",
            )
        )


def test_randomized_aps_is_reproducible_for_same_randomization_seed() -> None:
    """The same randomization_seed must produce identical prediction sets across runs."""
    kwargs = dict(
        conformal_prediction=True, confidence_level=0.90,
        qhat_source="true_class", set_construction="randomized_aps",
        calibration_source="eval_holdout", holdout_seed=42, randomization_seed=7,
    )
    rows_a = asyncio.run(
        predict_calibrated_rows(
            _make_iter72_ollama_client(), "fake-embed-model",
            _make_iter72_classifier(), _ITER72_DATASET, **kwargs,
        )
    )
    rows_b = asyncio.run(
        predict_calibrated_rows(
            _make_iter72_ollama_client(), "fake-embed-model",
            _make_iter72_classifier(), _ITER72_DATASET, **kwargs,
        )
    )
    assert [r["prediction_set"] for r in rows_a] == [r["prediction_set"] for r in rows_b]
    assert [r["set_size"] for r in rows_a] == [r["set_size"] for r in rows_b]


def test_set_construction_default_still_broken_with_randomized_aps_added() -> None:
    """Adding the 'randomized_aps' choice must not disturb the 'broken' default."""
    pred_set_default, size_default = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90, qhat_source="true_class"
    )
    assert pred_set_default == [0]
    assert size_default == 1


# --- Iter74: set_construction="raps_penalty" (RAPS size regularization) ---------

# Reuse the Iter70 5-class toy input/cp_data (q_hat fixed at 0.15 by construction).


def test_raps_penalty_at_lambda_0_matches_randomized_aps() -> None:
    """raps_lambda=0.0 must reproduce 'randomized_aps' exactly for any raps_k_reg.

    The penalty term `raps_lambda * max(0, rank - raps_k_reg)` vanishes at
    raps_lambda=0.0 regardless of raps_k_reg, so this is the plan's step 5(a)
    degeneracy check for the new branch.
    """
    pred_set_randomized, size_randomized = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="randomized_aps",
        randomization_u=0.5,
    )
    pred_set_raps, size_raps = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="raps_penalty",
        randomization_u=0.5, raps_lambda=0.0, raps_k_reg=2,
    )
    assert pred_set_raps == pred_set_randomized
    assert size_raps == size_randomized


def test_raps_penalty_set_size_is_monotonically_non_increasing_in_lambda() -> None:
    """Set size must be monotonically non-increasing as raps_lambda grows.

    Raising raps_lambda only ever subtracts more from a class's score (the
    penalty is non-negative and non-decreasing in rank), making the >= q_hat
    append condition strictly harder to satisfy at every rank beyond
    raps_k_reg, so the resulting prediction set can only shrink or stay the
    same (Iter74 investigation Q2's monotonicity argument).
    """
    _, size_lambda_0 = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="raps_penalty",
        randomization_u=1.0, raps_lambda=0.0, raps_k_reg=2,
    )
    _, size_lambda_small = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="raps_penalty",
        randomization_u=1.0, raps_lambda=0.05, raps_k_reg=2,
    )
    _, size_lambda_large = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
        qhat_source="true_class", set_construction="raps_penalty",
        randomization_u=1.0, raps_lambda=1.0, raps_k_reg=2,
    )
    assert size_lambda_large <= size_lambda_small <= size_lambda_0


def test_raps_penalty_requires_raps_lambda_and_raps_k_reg() -> None:
    """Omitting raps_lambda or raps_k_reg must raise, not silently use un-penalized randomized_aps."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
            qhat_source="true_class", set_construction="raps_penalty",
            randomization_u=1.0, raps_lambda=None, raps_k_reg=2,
        )
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
            qhat_source="true_class", set_construction="raps_penalty",
            randomization_u=1.0, raps_lambda=0.02, raps_k_reg=None,
        )


def test_raps_penalty_requires_randomization_u() -> None:
    """Omitting randomization_u must raise, since raps_penalty is built on randomized_aps."""
    with pytest.raises(ValueError):
        _compute_prediction_set(
            _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90,
            qhat_source="true_class", set_construction="raps_penalty",
            raps_lambda=0.02, raps_k_reg=2,
        )


def test_raps_penalty_rejects_oof_train_calibration_source() -> None:
    """raps_penalty requires calibration_source='eval_holdout' (shared row-indexed u),
    the same restriction randomized_aps enforces since raps_penalty reuses its u_all draw.
    """
    with pytest.raises(ValueError):
        asyncio.run(
            predict_calibrated_rows(
                _make_iter72_ollama_client(), "fake-embed-model",
                _make_iter72_classifier(), _ITER72_DATASET,
                conformal_prediction=True, calibration_source="oof_train",
                calibration_dataset_path=None,
                qhat_source="true_class", set_construction="raps_penalty",
                raps_lambda=0.02, raps_k_reg=2,
            )
        )


def test_set_construction_default_still_broken_with_raps_penalty_added() -> None:
    """Adding the 'raps_penalty' choice must not disturb the 'broken' default."""
    pred_set_default, size_default = _compute_prediction_set(
        _ITER70_PROBABILITIES, _ITER70_CP_DATA, confidence_level=0.90, qhat_source="true_class"
    )
    assert pred_set_default == [0]
    assert size_default == 1


# --- Iter75: education_threshold must reach calibration_source="eval_holdout" ---

# 8-row synthetic dataset over 4 classes (a, education, b, c), 2 rows per class,
# so StratifiedShuffleSplit(test_size=0.5) lands exactly 1 row per class in each
# half. classifier.predict_proba always returns the SAME fixed 4-column vector
# regardless of the embedding, with "education" deliberately placed 2nd by rank
# (0.30, behind "b" at 0.32) at threshold=0.0 so that adding the +0.05 threshold
# (-> 0.35) flips the rank order (education becomes rank-1). This rank flip is
# what makes q_hat (computed from the true-class score population) sensitive to
# education_threshold under calibration_source="eval_holdout": prior to the
# Iter75 fix, this correction never reached the calibration side, so q_hat was
# identical regardless of education_threshold (the bug this test guards against).
_ITER75_CLASSES = ["a", "education", "b", "c"]
_ITER75_RAW_PROBS = [0.10, 0.30, 0.32, 0.28]  # a, education, b, c
_ITER75_N_ROWS = 8
_ITER75_DATASET = [
    {
        "id": i,
        "query": f"query {i}",
        "expected_domains": [_ITER75_CLASSES[i % 4]],
    }
    for i in range(_ITER75_N_ROWS)
]


def _make_iter75_classifier() -> MagicMock:
    """A classifier stub whose predict_proba always returns the same fixed 4-class distribution.

    The constant output isolates the education_threshold's effect on the
    calibration/evaluation scores from any embedding-dependent variation.
    """
    classifier = MagicMock()
    classifier.classes_ = _ITER75_CLASSES

    def _predict_proba(embeddings: np.ndarray) -> np.ndarray:
        return np.tile(_ITER75_RAW_PROBS, (len(embeddings), 1))

    classifier.predict_proba.side_effect = _predict_proba
    return classifier


def _make_iter75_ollama_client() -> MagicMock:
    """A fake OllamaClient whose embed() returns a fixed-length placeholder vector."""
    client = MagicMock()
    client.embed = AsyncMock(return_value=[1.0, 0.0])
    return client


def _run_iter75(education_threshold: float | None) -> list[dict]:
    """Run predict_calibrated_rows() under calibration_source='eval_holdout' with
    the given education_threshold (None omits the keyword entirely, exercising
    the default)."""
    kwargs = dict(
        conformal_prediction=True, confidence_level=0.90,
        qhat_source="true_class", calibration_source="eval_holdout", holdout_seed=42,
    )
    if education_threshold is not None:
        kwargs["education_threshold"] = education_threshold
    return asyncio.run(
        predict_calibrated_rows(
            _make_iter75_ollama_client(), "fake-embed-model",
            _make_iter75_classifier(), _ITER75_DATASET, **kwargs,
        )
    )


def test_education_threshold_0_0_is_unchanged_from_omitting_the_argument() -> None:
    """education_threshold=0.0 (explicit) must reproduce omitting the argument exactly.

    Backward-compatibility guard for the Iter75 change: rows produced with the
    keyword explicitly set to its default value must be byte-for-byte identical
    to the pre-Iter75 default code path.
    """
    rows_default = _run_iter75(education_threshold=None)
    rows_explicit_zero = _run_iter75(education_threshold=0.0)
    assert rows_default == rows_explicit_zero


def test_education_threshold_applies_exactly_once_under_eval_holdout() -> None:
    """Under calibration_source='eval_holdout', probabilities['education'] in the
    output must be exactly raw_predict_proba['education'] + education_threshold
    -- not +2x the threshold.

    Direct regression guard for the Iter75 double-counting failure mode: prior
    to the fix, education_threshold was applied once at the all_probs stage
    (new) and unconditionally a second time in the per-row evaluation loop
    (pre-existing), which would have produced +0.10 instead of +0.05 here.
    """
    rows = _run_iter75(education_threshold=0.05)
    raw_education_prob = _ITER75_RAW_PROBS[_ITER75_CLASSES.index("education")]
    for row in rows:
        assert row["probabilities"]["education"] == pytest.approx(raw_education_prob + 0.05)
        assert row["probabilities"]["education"] != pytest.approx(raw_education_prob + 0.10)


def test_education_threshold_propagates_to_calibration_side_q_hat(capsys: pytest.CaptureFixture) -> None:
    """education_threshold must change q_hat under calibration_source='eval_holdout'.

    Direct regression guard for the Iter75 investigation Q1 finding: before the
    fix, calibration true-class scores were computed from the uncorrected
    predict_proba output (L476), so q_hat printed to stderr was identical
    regardless of education_threshold. The synthetic probability vector here
    is constructed so the +0.05 threshold flips "education" past "b" in rank
    order (see module comment above), which changes the true-class
    nonconformity score for both the education-labeled AND the b-labeled
    calibration row, and therefore q_hat.
    """
    capsys.readouterr()  # drain any prior output
    _run_iter75(education_threshold=0.0)
    q_hat_at_0_00 = _parse_q_hat_from_stderr(capsys.readouterr().err)

    _run_iter75(education_threshold=0.05)
    q_hat_at_0_05 = _parse_q_hat_from_stderr(capsys.readouterr().err)

    assert q_hat_at_0_00 != q_hat_at_0_05, (
        "q_hat did not change with education_threshold; the calibration-side "
        "correction is not actually reaching the true-class score population "
        "(Iter75 no-op guard)."
    )
    assert q_hat_at_0_00 == pytest.approx(0.68)
    assert q_hat_at_0_05 == pytest.approx(0.65)


def _parse_q_hat_from_stderr(stderr_text: str) -> float:
    """Extract the q_hat=<value> diagnostic printed by predict_calibrated_rows()."""
    for line in stderr_text.splitlines():
        if "q_hat=" in line:
            token = next(part for part in line.split() if part.startswith("q_hat="))
            return float(token.removeprefix("q_hat="))
    raise AssertionError(f"no q_hat= diagnostic found in stderr:\n{stderr_text}")

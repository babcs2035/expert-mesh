"""Tests for scripts/evaluate_selective_routing.py (Iter76,
routing_abstention_signal=conformal_set_size).

Covers the journal Iter76 plan's four required cases: (a) AURC/AUGRC match a
hand-computed value on a small toy sample, (b) a constant score degenerates
AURC to the overall error rate (every coverage level selects the same rows
in the same relative order, so selective risk == overall risk at every k),
(c) an unknown --abstention-signal name raises ValueError rather than
silently falling back to a default (Iter69's "no-op lever" lesson), and (d)
a perfectly-separating score (all correct rows ranked above all incorrect
rows) makes AUGRC equal its theoretical lower bound.
"""

import numpy as np
import pytest

from scripts.evaluate_selective_routing import (
    _get_signal_scorer,
    compute_aurc_augrc,
)


def test_aurc_augrc_match_hand_computed_toy_sample() -> None:
    """4 rows, scores strictly decreasing, is_correct = [True, True, False, True].

    Ranking by score descending gives ranks 1..4 in the given row order
    (score is already sorted descending). Cumulative errors after k rows:
    k=1: 0, k=2: 0, k=3: 1, k=4: 1.
    Selective risk (AURC terms): 0/1, 0/2, 1/3, 1/4 -> mean = (0+0+1/3+1/4)/4.
    Generalized risk (AUGRC terms): 0/4, 0/4, 1/4, 1/4 -> mean = (0+0+0.25+0.25)/4.
    """
    is_correct = np.array([True, True, False, True])
    score = np.array([4.0, 3.0, 2.0, 1.0])

    aurc, augrc = compute_aurc_augrc(is_correct, score)

    expected_aurc = (0.0 + 0.0 + 1.0 / 3.0 + 1.0 / 4.0) / 4.0
    expected_augrc = (0.0 + 0.0 + 0.25 + 0.25) / 4.0
    assert aurc == pytest.approx(expected_aurc)
    assert augrc == pytest.approx(expected_augrc)


def test_constant_score_makes_aurc_equal_overall_error_rate() -> None:
    """A single (tied) score for every row makes selective risk at coverage
    k/n equal cum_errors(k)/k, where cum_errors(k) depends on where the
    errors happen to fall in the (arbitrary, stable-mergesort-preserved)
    input order for a mixed correct/incorrect sample -- so this identity
    only holds exactly, independent of order, in the degenerate case where
    every row shares the same correctness label (error rate 0 or 1): then
    cum_errors(k)/k == overall_error_rate for every k from 1 to n, so the
    mean over k (AURC, which averages the SELECTIVE risk cum_errors(k)/k)
    trivially equals it too. AUGRC does not share this identity even in
    this degenerate case, because it averages the COVERAGE-WEIGHTED risk
    cum_errors(k)/n instead -- by construction it is not testable this way,
    so only AURC is asserted here.
    """
    is_correct = np.array([False, False, False, False, False])
    score = np.zeros(len(is_correct))
    overall_error_rate = float((~is_correct).sum()) / len(is_correct)

    aurc, _augrc = compute_aurc_augrc(is_correct, score)

    assert aurc == pytest.approx(overall_error_rate)


def test_unknown_abstention_signal_raises_value_error() -> None:
    """An unrecognized --abstention-signal name must fail loudly, never
    silently fall back to a default signal (Iter69's no-op-lever lesson)."""
    with pytest.raises(ValueError, match="unknown abstention signal"):
        _get_signal_scorer("not_a_real_signal")


def test_perfect_signal_augrc_equals_theoretical_lower_bound() -> None:
    """When every correct row outranks every incorrect row, no errors are
    ever selected until abstention has already excluded all of them, so the
    generalized-risk terms are 0 for k <= n_correct and (k - n_correct)/n
    for k > n_correct. AUGRC's theoretical lower bound for a fixed error
    count e among n rows is therefore sum_{j=1}^{e} j/n / n (only reached
    when the ranking is perfect), computed here directly and compared
    against compute_aurc_augrc's output.
    """
    n_correct = 6
    n_incorrect = 2
    n = n_correct + n_incorrect
    # Perfect ranking: correct rows score highest, incorrect rows lowest,
    # in an arbitrary (non-sorted-by-correctness) input order to confirm the
    # scorer -- not accidental input ordering -- drives the ranking.
    is_correct = np.array([True, True, False, True, True, True, False, True])
    score = np.array([10.0, 9.0, 1.0, 8.0, 7.0, 6.0, 2.0, 5.0])

    _, augrc = compute_aurc_augrc(is_correct, score)

    theoretical_lower_bound = sum(j for j in range(1, n_incorrect + 1)) / n / n
    assert augrc == pytest.approx(theoretical_lower_bound)

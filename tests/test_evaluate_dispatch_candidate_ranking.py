"""Tests for scripts/evaluate_dispatch_candidate_ranking.py's dual-format head loading
(Iter59 bare-estimator heads vs Iter60 MultiLabelBinarizer-based dict-payload heads),
the Iter60 no-op guards (A5/A6/N5), and the Iter63 rank1_source modes (build_new_rows'
baseline vs head_argmax rank_1 selection, A9, and the S4 rank1_change_count report).
"""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from scripts.evaluate_dispatch_candidate_ranking import (
    _RANK1_SOURCE_BASELINE,
    _RANK1_SOURCE_HEAD_ARGMAX,
    _assert_head_scores_are_domain_names,
    _assert_rank1_matches_head_argmax,
    _compute_a5_iter59_disagreement,
    _compute_rank1_change_count,
    _compute_single_domain_argmax_accuracy,
    _head_scores,
    _load_head,
    build_new_rows,
)

_TOY_EMBEDDINGS = [
    [1.0, 0.0, 0.0],
    [1.0, 0.1, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.1, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.1],
]

# 3 domains (not 2): OneVsRestClassifier's decision_function() degenerates to a
# 1D array (one score per sample, not per class) when there are exactly 2
# classes -- a real sklearn quirk unrelated to the Iter59/Iter60 format
# difference under test here, so these fixtures avoid it by using 3 domains.
_TOY_LABELS_SINGLE = ["medical", "medical", "legal", "legal", "education", "education"]
_TOY_LABELS_MULTI = [
    ["medical"],
    ["medical"],
    ["legal"],
    ["legal"],
    ["education"],
    ["legal", "education"],
]


def _fit_iter59_style_head() -> OneVsRestClassifier:
    """A bare OneVsRestClassifier trained on single-label string targets (Iter59 format):
    its own .classes_ is already the domain-name array."""
    model = OneVsRestClassifier(LogisticRegression(class_weight="balanced"))
    model.fit(_TOY_EMBEDDINGS, _TOY_LABELS_SINGLE)
    return model


def _fit_iter60_style_payload() -> dict:
    """A dict payload {"model", "classes"} trained via MultiLabelBinarizer (Iter60 format):
    the fitted estimator's own .classes_ degrades to integer column indices."""
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(_TOY_LABELS_MULTI)
    model = OneVsRestClassifier(LogisticRegression(class_weight="balanced"))
    model.fit(_TOY_EMBEDDINGS, Y)
    return {"model": model, "classes": list(mlb.classes_)}


def test_load_head_reads_iter59_bare_estimator_format(tmp_path) -> None:
    """_load_head() on an Iter59-style bare estimator returns its own classes_ as domain names."""
    head_path = tmp_path / "iter59_head.joblib"
    joblib.dump(_fit_iter59_style_head(), head_path)

    model, classes = _load_head(str(head_path))

    assert set(classes) == {"legal", "medical", "education"}
    assert isinstance(model, OneVsRestClassifier)


def test_load_head_reads_iter60_dict_payload_format(tmp_path) -> None:
    """_load_head() on an Iter60-style dict payload returns the saved "classes" list, not
    the estimator's own (integer-degraded) classes_ attribute."""
    head_path = tmp_path / "iter60_head.joblib"
    joblib.dump(_fit_iter60_style_payload(), head_path)

    model, classes = _load_head(str(head_path))

    assert set(classes) == {"legal", "medical", "education"}
    # The regression this guards against: naively using model.classes_ here
    # would yield [0, 1], not domain-name strings.
    assert all(isinstance(c, str) for c in classes)


def test_head_scores_are_keyed_by_domain_name_for_both_formats(tmp_path) -> None:
    """_head_scores() returns the same domain-name-keyed shape regardless of which
    artifact format _load_head() resolved (this is the bug Iter60 investigation Q1
    identified in the Iter59 zip(head.classes_, ...) implementation)."""
    iter59_path = tmp_path / "iter59_head.joblib"
    iter60_path = tmp_path / "iter60_head.joblib"
    joblib.dump(_fit_iter59_style_head(), iter59_path)
    joblib.dump(_fit_iter60_style_payload(), iter60_path)

    for head_path in (iter59_path, iter60_path):
        model, classes = _load_head(str(head_path))
        scores = _head_scores(model, classes, [1.0, 0.0, 0.0])
        assert set(scores.keys()) == {"legal", "medical", "education"}
        assert all(isinstance(key, str) for key in scores)
        assert all(isinstance(value, float) for value in scores.values())


def test_assert_head_scores_are_domain_names_passes_on_well_formed_rows() -> None:
    """A6: no exception when every row's head_scores keys exactly match the domain set."""
    new_rows = [
        {"id": "r1", "head_scores": {"legal": 0.1, "medical": 0.9}},
        {"id": "r2", "head_scores": {"legal": 0.8, "medical": 0.2}},
    ]

    _assert_head_scores_are_domain_names(new_rows, ["legal", "medical"])


def test_assert_head_scores_are_domain_names_rejects_integer_keys() -> None:
    """A6 must catch the exact regression this iteration's investigation warned about:
    head_scores keyed by integer column index (0, 1, ...) instead of domain-name strings."""
    new_rows = [{"id": "r1", "head_scores": {0: 0.1, 1: 0.9}}]

    with pytest.raises(AssertionError, match="A6"):
        _assert_head_scores_are_domain_names(new_rows, ["legal", "medical"])


def test_assert_head_scores_are_domain_names_rejects_missing_domain() -> None:
    """A6 must also catch a partial key set (e.g. a dropped domain), not just wrong types."""
    new_rows = [{"id": "r1", "head_scores": {"legal": 0.1}}]

    with pytest.raises(AssertionError, match="A6"):
        _assert_head_scores_are_domain_names(new_rows, ["legal", "medical"])


def test_compute_single_domain_argmax_accuracy_counts_only_non_compound_rows() -> None:
    """N5 evaluates argmax(head_scores) against expected_domains[0] on single-domain
    (len(expected_domains)==1) rows only, ignoring compound rows entirely."""
    baseline_rows = [
        {"id": "single-1", "expected_domains": ["legal"]},
        {"id": "single-2", "expected_domains": ["medical"]},
        {"id": "compound-1", "expected_domains": ["legal", "medical"]},
    ]
    new_rows = [
        {"id": "single-1", "head_scores": {"legal": 0.9, "medical": 0.1}},
        {"id": "single-2", "head_scores": {"legal": 0.9, "medical": 0.1}},  # wrong argmax
        {"id": "compound-1", "head_scores": {"legal": 0.5, "medical": 0.5}},
    ]

    result = _compute_single_domain_argmax_accuracy(baseline_rows, new_rows)

    assert result["n_single_domain_rows"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["pass"] is False


def test_compute_a5_iter59_disagreement_counts_rank2_mismatches(tmp_path) -> None:
    """A5 counts rows where this head's rank2_new differs from Iter59's own prediction file."""
    iter59_path = tmp_path / "iter59_predictions.jsonl"
    iter59_path.write_text(
        '{"id": "r1", "rank2_new": "medical"}\n{"id": "r2", "rank2_new": "legal"}\n',
        encoding="utf-8",
    )
    new_rows = [
        {"id": "r1", "rank2_new": "legal"},  # differs from Iter59's "medical"
        {"id": "r2", "rank2_new": "legal"},  # matches Iter59
    ]

    result = _compute_a5_iter59_disagreement(new_rows, str(iter59_path))

    assert result == {"n_rows": 2, "mismatches": 1, "mismatch_rate": 0.5}


def test_build_new_rows_defaults_to_baseline_rank1_and_verbatim_selected_domain() -> None:
    """rank1_source='baseline' (the default): rank_1 and selected_domain are copied
    through from the baseline row unchanged, matching Iter59-62 behavior (A10)."""
    model = _fit_iter59_style_head()
    classes = list(model.classes_)
    baseline_rows = [
        {
            "id": "r1",
            "expected_domains": ["legal"],
            "selected_domain": "legal",
            "dispatched_domains": ["legal", "education"],
        }
    ]
    # A medical-like embedding, deliberately disagreeing with the baseline's
    # rank_1="legal", so this test also proves baseline mode ignores head_scores
    # for rank_1 (unlike head_argmax mode, tested below).
    embeddings_by_id = {"r1": [1.0, 0.0, 0.0]}

    new_rows = build_new_rows(
        baseline_rows, model, classes, embeddings_by_id, rank1_source=_RANK1_SOURCE_BASELINE
    )

    assert new_rows[0]["dispatched_domains"][0] == "legal"
    assert new_rows[0]["selected_domain"] == "legal"


def test_build_new_rows_head_argmax_mode_overrides_rank1_and_selected_domain() -> None:
    """rank1_source='head_argmax': rank_1 becomes argmax(head_scores) instead of the
    baseline's rank_1, and selected_domain is overwritten to match (journal.md Iter63
    plan decision 3 -- without this, compute_top1_accuracy() would not exercise the lever)."""
    model = _fit_iter59_style_head()
    classes = list(model.classes_)
    baseline_rows = [
        {
            "id": "r1",
            "expected_domains": ["legal"],
            "selected_domain": "legal",
            "dispatched_domains": ["legal", "education"],
        }
    ]
    embeddings_by_id = {"r1": [1.0, 0.0, 0.0]}  # medical-like

    new_rows = build_new_rows(
        baseline_rows, model, classes, embeddings_by_id, rank1_source=_RANK1_SOURCE_HEAD_ARGMAX
    )

    expected_argmax = max(new_rows[0]["head_scores"], key=new_rows[0]["head_scores"].get)
    assert new_rows[0]["dispatched_domains"][0] == expected_argmax
    assert new_rows[0]["selected_domain"] == expected_argmax
    # rank_2 must still exclude the (possibly new) rank_1, never duplicate it.
    assert new_rows[0]["dispatched_domains"][1] != expected_argmax


def test_assert_rank1_matches_head_argmax_passes_when_rank1_is_the_argmax() -> None:
    """A9 passes when every row's rank_1 equals argmax(head_scores)."""
    new_rows = [
        {"id": "r1", "dispatched_domains": ["medical", "legal"], "head_scores": {"legal": 0.1, "medical": 0.9}},
    ]

    _assert_rank1_matches_head_argmax(new_rows)


def test_assert_rank1_matches_head_argmax_rejects_mismatch() -> None:
    """A9 must catch the case build_new_rows() regresses to a rank_1 not actually
    equal to argmax(head_scores) while still running in head_argmax mode."""
    new_rows = [
        {"id": "r1", "dispatched_domains": ["legal", "medical"], "head_scores": {"legal": 0.1, "medical": 0.9}},
    ]

    with pytest.raises(AssertionError, match="A9"):
        _assert_rank1_matches_head_argmax(new_rows)


def test_compute_rank1_change_count_counts_rank1_mismatches() -> None:
    """S4: counts rows where the new rank_1 differs from --baseline's rank_1 (non-fatal report)."""
    baseline_rows = [
        {"id": "r1", "dispatched_domains": ["legal", "education"]},
        {"id": "r2", "dispatched_domains": ["medical", "education"]},
    ]
    new_rows = [
        {"id": "r1", "dispatched_domains": ["medical", "legal"]},  # changed
        {"id": "r2", "dispatched_domains": ["medical", "legal"]},  # unchanged
    ]

    assert _compute_rank1_change_count(baseline_rows, new_rows) == 1

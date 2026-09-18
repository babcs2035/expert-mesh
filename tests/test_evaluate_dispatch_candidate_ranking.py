"""Tests for scripts/evaluate_dispatch_candidate_ranking.py's dual-format head loading
(Iter59 bare-estimator heads vs Iter60 MultiLabelBinarizer-based dict-payload heads)
and the Iter60 no-op guards (A5/A6/N5).
"""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from scripts.evaluate_dispatch_candidate_ranking import (
    _assert_head_scores_are_domain_names,
    _compute_a5_iter59_disagreement,
    _compute_single_domain_argmax_accuracy,
    _head_scores,
    _load_head,
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

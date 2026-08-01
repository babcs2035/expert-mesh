"""Tests for E6's offline classifier training (scripts/train_domain_classifier.py)."""

import pytest
from unittest.mock import AsyncMock, patch

from expert_backend import OllamaClient
from scripts.train_domain_classifier import (
    _extract_sample_weights,
    build_training_features,
    train_classifier,
)


async def test_build_training_features_embeds_each_row_in_order() -> None:
    """Embeddings and labels line up positionally with the input rows."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.embed.side_effect = [[1.0, 0.0], [0.0, 1.0]]
    rows = [
        {"id": "medical-train-001", "query": "headache", "domain": "medical"},
        {"id": "legal-train-001", "query": "contract", "domain": "legal"},
    ]

    embeddings, labels = await build_training_features(ollama_client, "nomic-embed-text", rows)

    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]
    assert labels == ["medical", "legal"]
    assert ollama_client.embed.call_args_list[0].args == ("nomic-embed-text", "headache")


def test_train_classifier_fits_a_model_that_predicts_seen_labels() -> None:
    """A classifier trained on separable data predicts the correct class for its own training points.

    Iter29 (classifier_calibration=platt) wraps the base LogisticRegression
    in CalibratedClassifierCV, whose default cv=5 uses a StratifiedKFold
    that requires at least 5 samples per class (n_splits must not exceed
    the smallest class's sample count). The previous 2-samples-per-class
    toy data (cv=2 implicitly) no longer exercises the same code path as
    production, so each class here has 5 slightly-jittered points around
    two well-separated 2D clusters, keeping cv at its production default
    instead of passing a smaller cv that would take a different branch.
    """
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.1],
        [1.0, -0.1],
        [1.1, 0.0],
        [0.9, 0.0],
        [0.0, 1.0],
        [0.0, 1.1],
        [0.0, 0.9],
        [0.1, 1.0],
        [-0.1, 1.0],
    ]
    labels = ["medical"] * 5 + ["legal"] * 5

    model = train_classifier(embeddings, labels)

    assert list(model.predict([[1.0, 0.0]])) == ["medical"]
    assert list(model.predict([[0.0, 1.0]])) == ["legal"]


def test_extract_sample_weights_computes_domain_balanced_weights() -> None:
    """_extract_sample_weights() computes n_samples/(n_classes*n_domain) for each row,
    reproducing sklearn's class_weight='balanced' effective weighting.

    With equal domain counts, all rows get weight=1.0. With an imbalanced domain
    (e.g. legal=77 vs others=150), the minority domain gets ~2x weight per row.
    The row-level "sample_weight" field is ignored (Iter39: manual sample_weight
    replaces class_weight='balanced').
    """
    # Equal counts: 2 domains, 2 rows each => weight = 4/(2*2) = 1.0
    rows_equal = [
        {"id": "e1", "query": "q1", "domain": "education"},
        {"id": "e2", "query": "q2", "domain": "education"},
        {"id": "m1", "query": "q3", "domain": "medical"},
        {"id": "m2", "query": "q4", "domain": "medical"},
    ]
    assert _extract_sample_weights(rows_equal) == [1.0, 1.0, 1.0, 1.0]

    # Imbalanced: 3 education + 1 legal => weight_edu=4/(2*3)=0.6667, weight_legal=4/(2*1)=2.0
    rows_imbalanced = [
        {"id": "e1", "query": "q1", "domain": "education"},
        {"id": "e2", "query": "q2", "domain": "education"},
        {"id": "e3", "query": "q3", "domain": "education"},
        {"id": "l1", "query": "q4", "domain": "legal"},
    ]
    weights = _extract_sample_weights(rows_imbalanced)
    assert weights[3] == pytest.approx(2.0, rel=1e-6)
    assert all(w == pytest.approx(4 / 6, rel=1e-6) for w in weights[:3])


def test_train_classifier_forwards_sample_weight_to_calibrated_fit() -> None:
    """train_classifier() passes its sample_weight argument through to
    CalibratedClassifierCV.fit() unchanged (Iter32,
    classifier_training_data_composition=education_proxy_task_revision), so a
    caller supplying per-task weights actually reaches the fitting code
    rather than being silently dropped (config.yml's recurring
    "lever never reaches the code path" failure pattern)."""
    embeddings = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5
    labels = ["medical"] * 5 + ["legal"] * 5
    weights = [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    with patch(
        "scripts.train_domain_classifier.CalibratedClassifierCV.fit",
        autospec=True,
    ) as mock_fit:
        train_classifier(embeddings, labels, sample_weight=weights)

    assert mock_fit.call_args.kwargs["sample_weight"] == weights


def test_train_classifier_defaults_sample_weight_to_none() -> None:
    """Callers that do not pass sample_weight (pre-Iter32 call sites, if any remain)
    still reach CalibratedClassifierCV.fit() with sample_weight=None, i.e. unweighted,
    identical to pre-Iter32 behavior."""
    embeddings = [[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5
    labels = ["medical"] * 5 + ["legal"] * 5

    with patch(
        "scripts.train_domain_classifier.CalibratedClassifierCV.fit",
        autospec=True,
    ) as mock_fit:
        train_classifier(embeddings, labels)

    assert mock_fit.call_args.kwargs["sample_weight"] is None

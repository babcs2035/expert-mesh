"""Tests for E6's offline classifier training (scripts/train_domain_classifier.py)."""

from unittest.mock import AsyncMock

from expert_backend import OllamaClient
from scripts.train_domain_classifier import build_training_features, train_classifier


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

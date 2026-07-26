"""Tests for E6's supervised domain classifier serving (classifier.py)."""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from classifier import estimate_confidence_classifier, load_domain_classifier


def _toy_classifier() -> LogisticRegression:
    """A real (not mocked) LogisticRegression fit on trivially separable 2D points."""
    embeddings = [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1], [-1.0, 0.0], [-1.0, 0.1]]
    labels = ["medical", "medical", "legal", "legal", "general", "general"]
    model = LogisticRegression(max_iter=1000)
    model.fit(embeddings, labels)
    return model


def test_estimate_confidence_classifier_returns_highest_probability_for_matching_domain() -> None:
    """A query embedding near the "medical" cluster gives medical the highest probability."""
    model = _toy_classifier()
    medical_confidence = estimate_confidence_classifier(model, "medical", [1.0, 0.0])
    legal_confidence = estimate_confidence_classifier(model, "legal", [1.0, 0.0])
    assert medical_confidence > legal_confidence


def test_estimate_confidence_classifier_returns_zero_for_unknown_domain() -> None:
    """A domain never seen during training returns 0.0 rather than raising."""
    model = _toy_classifier()
    assert estimate_confidence_classifier(model, "finance", [1.0, 0.0]) == 0.0


def test_load_domain_classifier_round_trips_through_joblib(tmp_path) -> None:
    """A classifier saved with joblib.dump loads back into an equivalent model."""
    model = _toy_classifier()
    model_path = tmp_path / "domain_classifier.joblib"
    joblib.dump(model, str(model_path))

    loaded = load_domain_classifier(str(model_path))
    assert list(loaded.classes_) == list(model.classes_)
    assert estimate_confidence_classifier(loaded, "medical", [1.0, 0.0]) == pytest.approx(
        estimate_confidence_classifier(model, "medical", [1.0, 0.0])
    )


def test_load_domain_classifier_raises_file_not_found_for_missing_path(tmp_path) -> None:
    """A missing classifier artifact fails at load time, not silently at request time."""
    with pytest.raises(FileNotFoundError):
        load_domain_classifier(str(tmp_path / "does_not_exist.joblib"))

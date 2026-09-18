"""Tests for E6's supervised domain classifier serving (classifier.py)."""

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from classifier import (
    EDUCATION_THRESHOLD,
    estimate_confidence_classifier,
    load_domain_classifier,
)


def _toy_classifier() -> LogisticRegression:
    """A real (not mocked) LogisticRegression fit on trivially separable 2D points."""
    embeddings = [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1], [-1.0, 0.0], [-1.0, 0.1]]
    labels = ["medical", "medical", "legal", "legal", "general", "general"]
    model = LogisticRegression(max_iter=1000)
    model.fit(embeddings, labels)
    return model


def _toy_classifier_with_education() -> LogisticRegression:
    """A real (not mocked) LogisticRegression fit on trivially separable 2D points, including education."""
    embeddings = [
        [1.0, 0.0],
        [1.0, 0.1],
        [0.0, 1.0],
        [0.0, 1.1],
        [-1.0, 0.0],
        [-1.0, 0.1],
        [0.0, -1.0],
        [0.0, -1.1],
    ]
    labels = [
        "medical",
        "medical",
        "legal",
        "legal",
        "general",
        "general",
        "education",
        "education",
    ]
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


def test_estimate_confidence_classifier_adds_threshold_for_education_only() -> None:
    """The education domain reports the raw probability plus EDUCATION_THRESHOLD."""
    model = _toy_classifier_with_education()
    query = [0.0, -1.0]
    raw_probability = float(
        model.predict_proba([query])[0][list(model.classes_).index("education")]
    )
    assert estimate_confidence_classifier(model, "education", query) == pytest.approx(
        raw_probability + EDUCATION_THRESHOLD
    )


def test_estimate_confidence_classifier_leaves_other_domains_unchanged() -> None:
    """A non-education domain is returned as the raw probability, without the threshold."""
    model = _toy_classifier_with_education()
    query = [1.0, 0.0]
    raw_probability = float(
        model.predict_proba([query])[0][list(model.classes_).index("medical")]
    )
    assert estimate_confidence_classifier(model, "medical", query) == pytest.approx(
        raw_probability
    )


def test_estimate_confidence_classifier_returns_zero_for_unknown_domain_with_education(
) -> None:
    """A domain never seen during training returns 0.0 even when education is a class."""
    model = _toy_classifier_with_education()
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

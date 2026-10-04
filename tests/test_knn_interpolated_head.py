"""Tests for the kNN-interpolated classifier wrapper (Iter104/105) and its build-script input check."""

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from knn_interpolated_head import KnnInterpolatedClassifier, l2_normalize_rows
from scripts.build_knn_interpolated_classifier import verify_cache_matches_train_rows

_TRAIN_EMBEDDINGS = np.array(
    [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1], [-1.0, 0.0], [-1.0, 0.1]]
)
_TRAIN_LABELS = ["medical", "medical", "legal", "legal", "general", "general"]
_QUERIES = np.array([[1.0, 0.05], [0.1, 1.0], [-0.9, 0.2], [0.5, 0.5]])


# A temperature below 1 (the Iter105 value), so the sharpening branch of predict_proba runs.
_SHARPENING_TEMPERATURE = 0.9426


def _build_wrapper(
    base: LogisticRegression, k: int, interpolation_lambda: float, temperature: float = 1.0
) -> KnnInterpolatedClassifier:
    """Wrap a fitted toy LogisticRegression with its own training points as the neighbour set."""
    class_to_index = {label: index for index, label in enumerate(base.classes_)}
    return KnnInterpolatedClassifier(
        base_classifier=base,
        train_embeddings_normalized=l2_normalize_rows(_TRAIN_EMBEDDINGS),
        train_label_indices=np.array([class_to_index[label] for label in _TRAIN_LABELS]),
        train_weights=np.ones(len(_TRAIN_LABELS)),
        k=k,
        interpolation_lambda=interpolation_lambda,
        temperature=temperature,
    )


@pytest.fixture
def base_classifier() -> LogisticRegression:
    """A real (not mocked) LogisticRegression fit on trivially separable 2D points."""
    model = LogisticRegression(max_iter=1000)
    model.fit(_TRAIN_EMBEDDINGS, _TRAIN_LABELS)
    return model


def test_predict_proba_rows_sum_to_one(base_classifier: LogisticRegression) -> None:
    """Every interpolated probability row sums to 1."""
    proba = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3).predict_proba(_QUERIES)
    assert proba.shape == (len(_QUERIES), len(base_classifier.classes_))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classes_match_base_classifier(base_classifier: LogisticRegression) -> None:
    """`classes_` is the base classifier's labels in the same order."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3)
    assert list(wrapper.classes_) == list(base_classifier.classes_)


def test_lambda_zero_reproduces_base_probabilities(base_classifier: LogisticRegression) -> None:
    """With interpolation_lambda=0 the output equals the base classifier's predict_proba."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.0)
    np.testing.assert_array_equal(
        wrapper.predict_proba(_QUERIES), base_classifier.predict_proba(_QUERIES)
    )


def test_lambda_one_returns_neighbour_votes_only(base_classifier: LogisticRegression) -> None:
    """With interpolation_lambda=1 and k=2, a query inside the legal cluster gets legal=1, others 0."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=1.0)
    proba = wrapper.predict_proba(np.array([[0.05, 1.0]]))[0]
    classes = list(base_classifier.classes_)
    assert proba[classes.index("legal")] == pytest.approx(1.0)
    assert proba[classes.index("medical")] == pytest.approx(0.0)
    assert proba[classes.index("general")] == pytest.approx(0.0)


def test_neighbour_ties_prefer_smaller_training_index(base_classifier: LogisticRegression) -> None:
    """An all-zero query (all similarities 0) takes the first k training rows as neighbours."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=1.0)
    proba = wrapper.predict_proba(np.zeros((1, 2)))[0]
    assert proba[list(base_classifier.classes_).index("medical")] == pytest.approx(1.0)


def test_joblib_round_trip_keeps_output(base_classifier: LogisticRegression, tmp_path) -> None:
    """Dumping and reloading with joblib leaves predict_proba unchanged."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3)
    path = tmp_path / "knn_interpolated.joblib"
    joblib.dump(wrapper, str(path))
    loaded = joblib.load(str(path))
    assert type(loaded).__name__ == "KnnInterpolatedClassifier"
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), wrapper.predict_proba(_QUERIES))


def test_invalid_k_is_rejected(base_classifier: LogisticRegression) -> None:
    """k larger than the number of training rows raises ValueError at construction."""
    with pytest.raises(ValueError):
        _build_wrapper(base_classifier, k=len(_TRAIN_LABELS) + 1, interpolation_lambda=0.3)


def test_temperature_one_returns_plain_interpolation(base_classifier: LogisticRegression) -> None:
    """With temperature=1 the output is exactly lambda * p_kNN + (1 - lambda) * p_base."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3, temperature=1.0)
    expected = 0.3 * wrapper.knn_label_distribution(_QUERIES) + 0.7 * np.asarray(
        base_classifier.predict_proba(_QUERIES), dtype=np.float64
    )
    np.testing.assert_array_equal(wrapper.predict_proba(_QUERIES), expected)


def test_temperature_scaled_rows_sum_to_one(base_classifier: LogisticRegression) -> None:
    """With temperature != 1 every probability row still sums to 1."""
    wrapper = _build_wrapper(
        base_classifier, k=2, interpolation_lambda=0.3, temperature=_SHARPENING_TEMPERATURE
    )
    np.testing.assert_allclose(wrapper.predict_proba(_QUERIES).sum(axis=1), 1.0)


def test_temperature_keeps_argmax(base_classifier: LogisticRegression) -> None:
    """Temperature scaling (sharpening or flattening) leaves every row's argmax unchanged."""
    reference = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3).predict_proba(
        _QUERIES
    )
    for temperature in (_SHARPENING_TEMPERATURE, 2.0):
        scaled = _build_wrapper(
            base_classifier, k=2, interpolation_lambda=0.3, temperature=temperature
        ).predict_proba(_QUERIES)
        np.testing.assert_array_equal(scaled.argmax(axis=1), reference.argmax(axis=1))


@pytest.mark.parametrize("temperature", [0.0, -0.5])
def test_non_positive_temperature_is_rejected(
    base_classifier: LogisticRegression, temperature: float
) -> None:
    """temperature <= 0 raises ValueError at construction."""
    with pytest.raises(ValueError):
        _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3, temperature=temperature)


def test_pickle_without_temperature_loads_as_one(
    base_classifier: LogisticRegression, tmp_path
) -> None:
    """An Iter104-style pickle lacking `temperature` loads with temperature=1 and the same output."""
    wrapper = _build_wrapper(base_classifier, k=2, interpolation_lambda=0.3)
    expected = wrapper.predict_proba(_QUERIES)
    del wrapper.__dict__["temperature"]
    path = tmp_path / "iter104_style.joblib"
    joblib.dump(wrapper, str(path))
    loaded = joblib.load(str(path))
    assert loaded.temperature == 1.0
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), expected)


def test_joblib_round_trip_keeps_temperature_output(
    base_classifier: LogisticRegression, tmp_path
) -> None:
    """Dumping and reloading a temperature-scaled wrapper keeps temperature and predict_proba."""
    wrapper = _build_wrapper(
        base_classifier, k=2, interpolation_lambda=0.3, temperature=_SHARPENING_TEMPERATURE
    )
    path = tmp_path / "knn_interpolated_temperature.joblib"
    joblib.dump(wrapper, str(path))
    loaded = joblib.load(str(path))
    assert loaded.temperature == _SHARPENING_TEMPERATURE
    np.testing.assert_array_equal(loaded.predict_proba(_QUERIES), wrapper.predict_proba(_QUERIES))


def test_cache_order_mismatch_is_rejected() -> None:
    """The build script refuses a cache whose ids are in a different order than the train data."""
    rows = [{"id": "a", "domain": "legal"}, {"id": "b", "domain": "medical"}]
    meta = {"ids": ["b", "a"], "domains": ["medical", "legal"]}
    with pytest.raises(ValueError):
        verify_cache_matches_train_rows(meta, rows, n_cache_rows=2)


def test_cache_matching_train_data_is_accepted() -> None:
    """The build script accepts a cache whose ids and domains match the train data in order."""
    rows = [{"id": "a", "domain": "legal"}, {"id": "b", "domain": "medical"}]
    meta = {"ids": ["a", "b"], "domains": ["legal", "medical"]}
    verify_cache_matches_train_rows(meta, rows, n_cache_rows=2)

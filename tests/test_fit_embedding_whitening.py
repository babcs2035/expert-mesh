"""Tests for the numpy-based whitening fit (E7). Skipped when numpy isn't installed
(the "research" optional-dependency group; not part of the default `uv sync`)."""

import pytest

np = pytest.importorskip("numpy")

from scripts.fit_embedding_whitening import _fit_whitening_from_vectors  # noqa: E402


def test_fit_whitening_from_vectors_mean_center_mode_returns_no_matrix() -> None:
    """mode="mean_center" returns only mu; no SVD is computed, so whitening_matrix is None."""
    vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    mean_vector, whitening_matrix = _fit_whitening_from_vectors(vectors, "mean_center")
    assert mean_vector == pytest.approx([3.0, 4.0])
    assert whitening_matrix is None


def test_fit_whitening_from_vectors_whiten_mode_recovers_identity_covariance() -> None:
    """Applying the fitted whitening transform to its own background yields ~unit covariance."""
    rng = np.random.default_rng(20260726)
    background = rng.multivariate_normal(mean=[5.0, -2.0], cov=[[9.0, 0.0], [0.0, 4.0]], size=2000)
    mean_vector, whitening_matrix = _fit_whitening_from_vectors(background.tolist(), "whiten")

    mu = np.array(mean_vector)
    w = np.array(whitening_matrix)
    whitened = (background - mu) @ w
    covariance = np.cov(whitened, rowvar=False)
    assert covariance == pytest.approx(np.eye(2), abs=0.1)

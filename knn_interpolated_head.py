"""Iter104 (knn_label_distribution_interpolation): wraps the production domain classifier and
linearly interpolates its probabilities with a k-nearest-neighbour label distribution.

    p = interpolation_lambda * p_kNN,k + (1 - interpolation_lambda) * p_base

The interpolation form follows kNN-LM (Khandelwal et al., ICLR 2020, arXiv:1911.00172) and
KNN-BERT (Li et al., 2021, arXiv:2110.02523), applied to the 10-domain class distribution.
`p_kNN,k` takes the top-k training rows by cosine similarity (inner product of L2-normalized
fused embeddings) and votes their labels with the same domain-balanced weights as
`scripts/train_domain_classifier.py:_extract_sample_weights()`, then normalizes.

Why this module lives at the repository root: `joblib.dump`/`joblib.load` serialize a class by
its module import path, so the build script (`scripts/build_knn_interpolated_classifier.py`)
and the deployed container (`classifier.py:load_domain_classifier`, after `Dockerfile` COPYs
this file) must both import the same `knn_interpolated_head.KnnInterpolatedClassifier`
(same reason as `ecoc_head.py`).

Only `classes_` and `predict_proba(X)` are provided, because those are the only attributes the
serving path (`classifier.py:estimate_confidence_classifier`) and the auxiliary scripts read.
"""

from __future__ import annotations

import numpy as np

# Lower bound on a vector norm before division, so an all-zero query embedding yields a zero
# similarity row (and thus the lowest-index neighbours) instead of NaN.
_MIN_NORM = 1e-12


def l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    """Return `features` with every row scaled to unit L2 norm (cosine similarity = inner product)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, _MIN_NORM)


class KnnInterpolatedClassifier:
    """Base classifier probabilities linearly interpolated with a weighted kNN label distribution.

    `base_classifier` is used as-is (not refit). Neighbour ties in similarity are broken in
    favour of the smaller training-row index (stable sort), so the output is deterministic.
    """

    def __init__(
        self,
        base_classifier: object,
        train_embeddings_normalized: np.ndarray,
        train_label_indices: np.ndarray,
        train_weights: np.ndarray,
        k: int,
        interpolation_lambda: float,
    ) -> None:
        n_train = train_embeddings_normalized.shape[0]
        if train_label_indices.shape != (n_train,) or train_weights.shape != (n_train,):
            raise ValueError(
                "train_label_indices and train_weights must both have shape (n_train,) "
                f"with n_train={n_train}"
            )
        if not 1 <= k <= n_train:
            raise ValueError(f"k must be in [1, {n_train}], got {k}")
        if not 0.0 <= interpolation_lambda <= 1.0:
            raise ValueError(f"interpolation_lambda must be in [0, 1], got {interpolation_lambda}")
        n_classes = len(base_classifier.classes_)
        if train_label_indices.min() < 0 or train_label_indices.max() >= n_classes:
            raise ValueError(
                f"train_label_indices must index base_classifier.classes_ (0..{n_classes - 1})"
            )
        if np.any(train_weights <= 0.0):
            raise ValueError(
                "train_weights must be strictly positive so every kNN vote sums to > 0"
            )
        self.base_classifier = base_classifier
        self.train_embeddings_normalized = np.asarray(train_embeddings_normalized, dtype=np.float64)
        self.train_label_indices = np.asarray(train_label_indices, dtype=np.int64)
        self.train_weights = np.asarray(train_weights, dtype=np.float64)
        self.k = int(k)
        self.interpolation_lambda = float(interpolation_lambda)

    @property
    def classes_(self) -> np.ndarray:
        """The base classifier's class labels, unchanged (same array, same order)."""
        return self.base_classifier.classes_

    def knn_label_distribution(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Return the (n, n_classes) weighted label distribution of each row's top-k training rows."""
        query_normalized = l2_normalize_rows(np.asarray(X, dtype=np.float64))
        similarities = query_normalized @ self.train_embeddings_normalized.T
        # Stable sort on the negated similarity keeps the smaller training index first on ties.
        neighbor_indices = np.argsort(-similarities, axis=1, kind="stable")[:, : self.k]
        distribution = np.zeros((query_normalized.shape[0], len(self.classes_)))
        for row, neighbors in enumerate(neighbor_indices):
            np.add.at(
                distribution[row],
                self.train_label_indices[neighbors],
                self.train_weights[neighbors],
            )
        return distribution / distribution.sum(axis=1, keepdims=True)

    def predict_proba(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Return the interpolated (n, n_classes) probabilities; each row sums to 1."""
        base_proba = np.asarray(self.base_classifier.predict_proba(X), dtype=np.float64)
        if self.interpolation_lambda == 0.0:
            return base_proba
        knn_proba = self.knn_label_distribution(X)
        return (
            self.interpolation_lambda * knn_proba + (1.0 - self.interpolation_lambda) * base_proba
        )

    def predict(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Return the class label with the highest interpolated probability for each row."""
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]

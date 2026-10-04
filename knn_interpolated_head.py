"""Iter104 (knn_label_distribution_interpolation) and Iter105 (knn_interpolation_temperature):
wraps the production domain classifier, linearly interpolates its probabilities with a
k-nearest-neighbour label distribution, and then recalibrates the mixture with a temperature.

    p   = interpolation_lambda * p_kNN,k + (1 - interpolation_lambda) * p_base
    p_T ∝ exp(log(max(p, 1e-12)) / temperature)        (row-normalized; p_T = p when T = 1)

The interpolation form follows kNN-LM (Khandelwal et al., ICLR 2020, arXiv:1911.00172) and
KNN-BERT (Li et al., 2021, arXiv:2110.02523), applied to the 10-domain class distribution.
`p_kNN,k` takes the top-k training rows by cosine similarity (inner product of L2-normalized
fused embeddings) and votes their labels with the same domain-balanced weights as
`scripts/train_domain_classifier.py:_extract_sample_weights()`, then normalizes.

The temperature step is temperature scaling (Guo et al., "On Calibration of Modern Neural
Networks", ICML 2017, arXiv:1706.04599) with log p treated as the logit. It is applied after
the interpolation because the base classifier's own temperature was fit before kNN mixing.
It preserves each row's argmax; T < 1 sharpens and T > 1 flattens the distribution. The Iter105
value is the single-label NLL minimizer on 5-fold out-of-fold mixtures of the training set
(reproducible with `.claude/research/_iter105_replay_temp.py`).

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


# Floor on a probability before log(); a kNN vote of 0 would otherwise give log(0) = -inf.
_PROB_FLOOR = 1e-12


def apply_temperature_to_proba(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Return `proba` rescaled as p_T ∝ p^(1/temperature), row-normalized (argmax preserved)."""
    logits = np.log(np.maximum(proba, _PROB_FLOOR)) / temperature
    # Subtracting the row max before exp() keeps it from overflowing when temperature is small.
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


class KnnInterpolatedClassifier:
    """Base classifier probabilities linearly interpolated with a weighted kNN label distribution.

    `base_classifier` is used as-is (not refit). Neighbour ties in similarity are broken in
    favour of the smaller training-row index (stable sort), so the output is deterministic.
    `temperature` (> 0) rescales the interpolated distribution; 1.0 leaves it unchanged.
    """

    def __init__(
        self,
        base_classifier: object,
        train_embeddings_normalized: np.ndarray,
        train_label_indices: np.ndarray,
        train_weights: np.ndarray,
        k: int,
        interpolation_lambda: float,
        temperature: float = 1.0,
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
        if not temperature > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
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
        self.temperature = float(temperature)

    def __setstate__(self, state: dict) -> None:
        """Restore a pickled instance, defaulting `temperature` to 1.0 for Iter104 artifacts.

        Iter104 artifacts were pickled before `temperature` existed; without the default their
        `predict_proba` would raise AttributeError instead of returning the uncalibrated mixture.
        """
        state.setdefault("temperature", 1.0)
        self.__dict__.update(state)

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
        """Return the interpolated, temperature-scaled (n, n_classes) probabilities; rows sum to 1."""
        base_proba = np.asarray(self.base_classifier.predict_proba(X), dtype=np.float64)
        if self.interpolation_lambda == 0.0:
            mixture = base_proba
        else:
            knn_proba = self.knn_label_distribution(X)
            mixture = (
                self.interpolation_lambda * knn_proba
                + (1.0 - self.interpolation_lambda) * base_proba
            )
        if self.temperature == 1.0:
            # Returned untouched so T=1 stays bit-identical to the Iter104 output.
            return mixture
        return apply_temperature_to_proba(mixture, self.temperature)

    def predict(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Return the class label with the highest interpolated probability for each row."""
        return np.asarray(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]

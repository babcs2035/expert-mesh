"""Iter98 (classifier_multiclass_decomposition=error_correcting_output_codes): a thin
sklearn-compatible wrapper that decomposes a K-class problem into L pairwise-agnostic
binary problems via a dense random error-correcting output code (Allwein, Schapire &
Singer 2000, JMLR 1:113-141) and recombines them with log-loss (maximum-likelihood)
decoding, exposing a continuous `decision_function` / `predict_proba` that
`sklearn.calibration.CalibratedClassifierCV` can wrap.

Why this module exists instead of `sklearn.multiclass.OutputCodeClassifier`: as
measured on sklearn 1.9.0 (journal.md "Iteration 98" (4)), `OutputCodeClassifier`
implements only `predict` -- it has neither `predict_proba` nor `decision_function` --
so `CalibratedClassifierCV(OutputCodeClassifier(...))` cannot be fit at all
(`InvalidParameterError`). `classifier.py:estimate_confidence_classifier()` requires a
`predict_proba` whose row sums to 1 across the 10 domains, which only a custom
decoder can provide here.

Why this module lives at the repository root rather than inside
`scripts/train_domain_classifier.py`: `joblib.dump`/`joblib.load` serialize a class by
its module import path, so a class defined inside a script run as `__main__` cannot be
unpickled on another host. Placing `EcocLogLossClassifier` in its own top-level module
lets both `scripts/train_domain_classifier.py` (at training time) and the deployed
container (at inference time, after `Dockerfile` COPYs this file) import the same
`ecoc_head.EcocLogLossClassifier` path.

Pre-registered design (journal.md "Iteration 98", not swept, see B164):
  - Code family: dense random code, {-1, +1} entries.
  - Code length L = ceil(10 * log2(K)) (Allwein et al.'s dense-random-code default;
    L=34 for K=10).
  - Code selection: among `_N_CODE_CANDIDATES` random candidates, keep the one with
    the largest minimum pairwise row Hamming distance, rejecting any candidate with a
    constant column (a column that does not split the classes at all) or a zero
    minimum row distance (duplicate rows, which would make two classes
    undecodable-distinct).
  - Decoding: log-loss (maximum-likelihood) decoding, which Allwein et al. show
    dominates Hamming decoding: `decision_function(X)[i, d] = sum_j log(sigmoid(
    M[d, j] * f_j(x_i)))`, where `f_j` is column j's binary base estimator's margin
    (`decision_function`). This is a continuous score (unlike vote-counting decoders),
    so it does not reproduce the discrete-score over-confidence found in Iter97's
    one-vs-one arm (journal.md "Iteration 97" P4).
  - `random_state` is fixed per caller (Iter98 uses 98, the iteration number) so the
    same code book is reproduced across every CV fold and every re-fit.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

# Allwein, Schapire & Singer (2000): dense random code length is ceil(10*log2(K)),
# selected as the best of 10,000 random candidates by minimum pairwise row Hamming
# distance (their Section 4's code construction procedure).
_N_CODE_CANDIDATES = 10_000
_CODE_LENGTH_COEFFICIENT = 10.0

# Clip the margin*bit product before logaddexp so a confidently-correct or
# confidently-wrong base estimator (large |margin|) cannot overflow the float64
# exponent; softplus(-z) for |z| > 50 is indistinguishable from 0 or -z at float64
# precision, so clipping here changes no decodable score.
_MARGIN_PRODUCT_CLIP = 50.0


def build_dense_random_code(n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Return a {-1, +1} dense random code matrix of shape (n_classes, code_length).

    `code_length` is `ceil(_CODE_LENGTH_COEFFICIENT * log2(n_classes))`. Among
    `_N_CODE_CANDIDATES` random draws, keeps the candidate with the largest minimum
    pairwise row Hamming distance, discarding any candidate with a constant column
    (a column assigning every class to the same side, i.e. a degenerate binary
    problem) or a zero minimum row distance (two classes given an identical code
    word, which no decoder could ever tell apart).
    """
    code_length = int(np.ceil(_CODE_LENGTH_COEFFICIENT * np.log2(n_classes)))
    best_code: np.ndarray | None = None
    best_min_row_distance = -1.0
    for _ in range(_N_CODE_CANDIDATES):
        candidate = rng.choice((-1.0, 1.0), size=(n_classes, code_length))
        if np.any(np.abs(candidate.sum(axis=0)) == n_classes):
            continue  # a constant column: every class on the same side.
        pairwise_distance = (candidate[:, None, :] != candidate[None, :, :]).sum(axis=2).astype(float)
        np.fill_diagonal(pairwise_distance, np.inf)
        min_row_distance = float(pairwise_distance.min())
        if min_row_distance == 0.0:
            continue  # two classes share an identical code word.
        if min_row_distance > best_min_row_distance:
            best_code, best_min_row_distance = candidate, min_row_distance
    if best_code is None:
        raise RuntimeError(
            f"failed to build a non-degenerate dense random code after "
            f"{_N_CODE_CANDIDATES} candidates (n_classes={n_classes})"
        )
    return best_code


class EcocLogLossClassifier(ClassifierMixin, BaseEstimator):
    """Error-correcting-output-code multi-class classifier with log-loss decoding.

    `estimator` is cloned once per code-matrix column (L clones total) and fit on the
    full training set with that column's {-1, +1} binary label -- unlike pairwise
    (one-vs-one) decomposition, every column sees every training row, so this does not
    reproduce the sample-splitting problem Iter97 measured for
    `OneVsOneClassifier` (journal.md "Iteration 97" (d)).

    `decision_function` returns a continuous per-class log-likelihood score suitable
    for `sklearn.calibration.CalibratedClassifierCV`, which is required because
    `classifier.py:estimate_confidence_classifier()` needs `predict_proba` to sum to 1
    across all `classes_`.
    """

    def __init__(self, estimator: BaseEstimator, random_state: int = 0) -> None:
        self.estimator = estimator
        self.random_state = random_state

    def fit(
        self,
        X: list[list[float]] | np.ndarray,
        y: list[str] | np.ndarray,
        sample_weight: list[float] | np.ndarray | None = None,
    ) -> "EcocLogLossClassifier":
        """Fit one binary base estimator per code-matrix column on the full training set.

        `sample_weight`, if given, is forwarded unchanged to every column's
        `estimator.fit(X, column_bit, sample_weight=sample_weight)` call -- since this
        wrapper (unlike `OneVsOneClassifier`) declares `sample_weight` as an explicit
        `fit()` parameter, sklearn's `has_fit_parameter` check finds it without
        requiring `sklearn.config_context(enable_metadata_routing=True)` (Iter97's
        `OneVsOneClassifier` pitfall does not apply here).
        """
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.code_book_ = build_dense_random_code(len(self.classes_), rng)
        class_to_index = {class_label: index for index, class_label in enumerate(self.classes_)}
        y_index = np.asarray([class_to_index[label] for label in y])
        self.estimators_ = []
        for column in range(self.code_book_.shape[1]):
            column_bit = self.code_book_[y_index, column]
            column_estimator = clone(self.estimator)
            column_estimator.fit(X, column_bit, sample_weight=sample_weight)
            self.estimators_.append(column_estimator)
        return self

    def decision_function(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Log-loss (maximum-likelihood) decoding: continuous per-class score.

        `score[i, d] = sum_j log(sigmoid(code_book_[d, j] * margin_j(x_i)))`, computed
        via `-logaddexp(0, -z)` (the numerically stable form of `log(sigmoid(z))`).
        Larger is a better fit to class `d`'s code word; `predict` takes the argmax
        over `d`.
        """
        check_is_fitted(self)
        margins = np.column_stack([column_estimator.decision_function(X) for column_estimator in self.estimators_])
        margin_times_bit = margins[:, None, :] * self.code_book_[None, :, :]
        margin_times_bit = np.clip(margin_times_bit, -_MARGIN_PRODUCT_CLIP, _MARGIN_PRODUCT_CLIP)
        log_sigmoid = -np.logaddexp(0.0, -margin_times_bit)
        return log_sigmoid.sum(axis=2)

    def predict(self, X: list[list[float]] | np.ndarray) -> np.ndarray:
        """Predict the class whose code word has the highest log-loss decoding score."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

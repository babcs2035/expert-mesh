"""E6: supervised domain classifier serving (routing_method=supervised_classifier).

Each node loads the same multi-class classifier (trained by
scripts/train_domain_classifier.py from question text + domain label only,
never from probe/dispatch results — see that module's docstring for why)
and, at /probe time, reads off only its own domain's predicted probability.
No LLM call is involved: the classifier consumes the query_embedding
already computed by the requester (node.py's run_ask_flow), so this is a
pure function of already-available data.

Since Iter29 (classifier_calibration=platt), the persisted artifact is a
CalibratedClassifierCV wrapping the LogisticRegression base estimator
rather than the bare LogisticRegression itself. Both functions below rely
only on `.classes_` and `.predict_proba()`, which CalibratedClassifierCV
exposes with the same semantics (duck typing; no code change needed here
beyond the type annotation). sklearn's one-vs-rest calibration
renormalizes per-class probabilities internally so they still sum to 1
across domains (scikit-learn.org/stable/modules/calibration.html
#1.16.3.3) -- this module does not need to renormalize itself.

As of Iter31 (journal.md "考察 (Iter31)"), the production artifact at
models/domain_classifier.joblib is the method="temperature" calibration
(adopted: ECE 0.193358->0.071201, top1_accuracy McNemar p=0.000906
improvement, 0/20 per-domain metrics BH-significant regression), not the
method="platt" or method="isotonic" variants tried in Iter29/Iter30
(both partial, deferred). The renormalization note above still applies:
temperature's shared-softmax output also sums to 1 across domains, just
via a single scalar rescale of the logit vector rather than per-class
isotonic/sigmoid calibrators.
"""

import joblib
from sklearn.calibration import CalibratedClassifierCV


def load_domain_classifier(model_path: str) -> CalibratedClassifierCV:
    """Load a joblib-persisted multi-class domain classifier.

    Raises FileNotFoundError (via joblib.load) if the path is missing
    rather than catching it: a routing_method=supervised_classifier node
    with no classifier artifact is a configuration error that should fail
    at startup, not degrade silently into always returning 0.0 confidence.
    """
    return joblib.load(model_path)


def estimate_confidence_classifier(
    classifier: CalibratedClassifierCV, domain: str, query_embedding: list[float]
) -> float:
    """Return this node's own domain's predicted probability from the shared classifier.

    Returns 0.0 (the same safe-default convention as
    estimate_embedding_confidence's zero-vector case) when the classifier
    wasn't trained on this domain at all, keeping such a node out of
    dispatch consideration rather than raising at request time.
    """
    if domain not in classifier.classes_:
        return 0.0
    domain_index = list(classifier.classes_).index(domain)
    probabilities = classifier.predict_proba([query_embedding])[0]
    return float(probabilities[domain_index])

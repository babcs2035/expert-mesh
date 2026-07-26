"""E6: supervised domain classifier serving (routing_method=supervised_classifier).

Each node loads the same multi-class classifier (trained by
scripts/train_domain_classifier.py from question text + domain label only,
never from probe/dispatch results — see that module's docstring for why)
and, at /probe time, reads off only its own domain's predicted probability.
No LLM call is involved: the classifier consumes the query_embedding
already computed by the requester (node.py's run_ask_flow), so this is a
pure function of already-available data.
"""

import joblib
from sklearn.linear_model import LogisticRegression


def load_domain_classifier(model_path: str) -> LogisticRegression:
    """Load a joblib-persisted multi-class domain classifier.

    Raises FileNotFoundError (via joblib.load) if the path is missing
    rather than catching it: a routing_method=supervised_classifier node
    with no classifier artifact is a configuration error that should fail
    at startup, not degrade silently into always returning 0.0 confidence.
    """
    return joblib.load(model_path)


def estimate_confidence_classifier(
    classifier: LogisticRegression, domain: str, query_embedding: list[float]
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

"""Build the kNN-interpolated domain classifier artifact (Iter104; temperature since Iter105).

Wraps an existing classifier artifact in `knn_interpolated_head.KnnInterpolatedClassifier`,
using a precomputed training embedding cache as the neighbour population. No retraining and
no embedding calls are made.

Usage (module mode, from the repository root, CPU only):
    uv run python -m scripts.build_knn_interpolated_classifier \\
        --base-classifier models/domain_classifier_pre_iter104_lr.joblib \\
        --train-cache data/embcache_train_iter100_fused.npy \\
        --train-data data/classifier_train_iter94_dedup.jsonl \\
        --k 2 --interpolation-lambda 0.3 --temperature 0.9426 \\
        --output models/domain_classifier.joblib

`--temperature` (default 1.0 = no rescaling) is applied after the interpolation. The Iter105
value 0.9426 is the out-of-fold single-label NLL minimizer reproduced by
`.claude/research/_iter105_replay_temp.py`.

The cache's `<cache>.meta.json` must list the same `ids` and `domains` as `--train-data`, in the
same order; otherwise the embeddings would be paired with the wrong labels, so the script stops.
"""

import argparse
import json

import joblib
import numpy as np

from knn_interpolated_head import KnnInterpolatedClassifier, l2_normalize_rows
from scripts.train_domain_classifier import _extract_sample_weights


def read_jsonl_rows(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify_cache_matches_train_rows(meta: dict, rows: list[dict], n_cache_rows: int) -> None:
    """Raise ValueError unless the cache rows and `rows` agree on count, ids and domains in order."""
    if n_cache_rows != len(rows):
        raise ValueError(f"cache has {n_cache_rows} rows but train data has {len(rows)} rows")
    if list(meta["ids"]) != [row["id"] for row in rows]:
        raise ValueError("cache meta ids do not match train data ids in order")
    if list(meta["domains"]) != [row["domain"] for row in rows]:
        raise ValueError("cache meta domains do not match train data domains in order")


def main() -> None:
    """Load the base artifact and training cache, wrap them, and dump the new artifact."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-classifier", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--interpolation-lambda", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_classifier = joblib.load(args.base_classifier)
    train_embeddings = np.load(args.train_cache)
    with open(f"{args.train_cache}.meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    rows = read_jsonl_rows(args.train_data)
    verify_cache_matches_train_rows(meta, rows, train_embeddings.shape[0])

    class_to_index = {str(label): index for index, label in enumerate(base_classifier.classes_)}
    unknown_domains = sorted({row["domain"] for row in rows} - set(class_to_index))
    if unknown_domains:
        raise ValueError(f"train data domains not in base classifier classes_: {unknown_domains}")
    train_label_indices = np.array([class_to_index[row["domain"]] for row in rows], dtype=np.int64)
    train_weights = np.array(_extract_sample_weights(rows), dtype=np.float64)

    classifier = KnnInterpolatedClassifier(
        base_classifier=base_classifier,
        train_embeddings_normalized=l2_normalize_rows(train_embeddings.astype(np.float64)),
        train_label_indices=train_label_indices,
        train_weights=train_weights,
        k=args.k,
        interpolation_lambda=args.interpolation_lambda,
        temperature=args.temperature,
    )
    joblib.dump(classifier, args.output)
    print(
        f"wrote {args.output}: n_train={len(rows)} dim={train_embeddings.shape[1]} "
        f"k={args.k} lambda={args.interpolation_lambda} temperature={args.temperature}"
    )


if __name__ == "__main__":
    main()

"""Iter96 (composition_preserving_volume_expansion): assemble the embedding
cache for the 3,271-row scaled training set (and the 426-row hold-out set)
by looking up each row's query text against the Iter95 full-pool cache
(`data/embcache_iter95_fullpool.npy`, 4,507 rows), so that no new
`/api/embeddings` calls are needed on wafl-ctrl5 (absolute condition (B):
this script itself runs on wafl-ctrl5, not on wafl500-509).

Every row in classifier_train_iter96_scaled.jsonl and
classifier_train_iter96_holdout.jsonl is a subset of the Iter95 4,507-row
pool (2,275 existing + up to 2,232 disjoint JMMLU rows), so a query-text
join against `data/classifier_train_iter95_fullpool.jsonl` (whose row
order matches the Iter95 embcache exactly) recovers every embedding. Rows
that fail to match are reported so the caller can embed only those
(target: 0).

Usage (on wafl-ctrl5, inside ~/expert-mesh-iter95/):
    uv run python scripts/build_iter96_embcache_from_iter95_pool.py \\
        --iter95-pool-jsonl data/classifier_train_iter95_fullpool.jsonl \\
        --iter95-embcache data/embcache_iter95_fullpool.npy \\
        --iter96-train-jsonl data/classifier_train_iter96_scaled.jsonl \\
        --iter96-holdout-jsonl data/classifier_train_iter96_holdout.jsonl \\
        --output-train-embcache data/embcache_iter96_scaled.npy \\
        --output-holdout-embcache data/embcache_iter96_holdout.npy
"""

import argparse
import json
import sys

import numpy as np


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_query_index(pool_rows: list[dict]) -> dict[str, int]:
    """Map query text to its row index in the Iter95 pool (first match wins;
    the pool has no duplicate queries by construction)."""
    index: dict[str, int] = {}
    for i, row in enumerate(pool_rows):
        index.setdefault(row["query"], i)
    return index


def _lookup(rows: list[dict], query_index: dict[str, int], embeddings: np.ndarray) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Return (embeddings_matched, ids_matched, domains_matched, unmatched_ids)."""
    matched_vectors = []
    matched_ids = []
    matched_domains = []
    unmatched_ids = []
    for row in rows:
        idx = query_index.get(row["query"])
        if idx is None:
            unmatched_ids.append(row["id"])
            continue
        matched_vectors.append(embeddings[idx])
        matched_ids.append(row["id"])
        matched_domains.append(row["domain"])
    matched = np.stack(matched_vectors, axis=0) if matched_vectors else np.zeros((0, embeddings.shape[1]))
    return matched, matched_ids, matched_domains, unmatched_ids


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Assemble Iter96 embcaches from the Iter95 full-pool cache")
    parser.add_argument("--iter95-pool-jsonl", required=True)
    parser.add_argument("--iter95-embcache", required=True)
    parser.add_argument("--iter96-train-jsonl", required=True)
    parser.add_argument("--iter96-holdout-jsonl", required=True)
    parser.add_argument("--output-train-embcache", required=True)
    parser.add_argument("--output-holdout-embcache", required=True)
    args = parser.parse_args()

    pool_rows = _load_jsonl(args.iter95_pool_jsonl)
    embeddings = np.load(args.iter95_embcache)
    assert embeddings.shape[0] == len(pool_rows), (
        f"embcache row count {embeddings.shape[0]} != pool jsonl row count {len(pool_rows)}"
    )
    query_index = _build_query_index(pool_rows)

    train_rows = _load_jsonl(args.iter96_train_jsonl)
    holdout_rows = _load_jsonl(args.iter96_holdout_jsonl)

    train_emb, train_ids, train_domains, train_unmatched = _lookup(train_rows, query_index, embeddings)
    holdout_emb, holdout_ids, holdout_domains, holdout_unmatched = _lookup(holdout_rows, query_index, embeddings)

    np.save(args.output_train_embcache, train_emb)
    with open(args.output_train_embcache + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": train_ids, "domains": train_domains}, f, ensure_ascii=False)

    np.save(args.output_holdout_embcache, holdout_emb)
    with open(args.output_holdout_embcache + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"ids": holdout_ids, "domains": holdout_domains}, f, ensure_ascii=False)

    print(
        f"[build_iter96_embcache_from_iter95_pool] train: matched={len(train_ids)}/{len(train_rows)} "
        f"unmatched={len(train_unmatched)}",
        file=sys.stderr,
    )
    print(
        f"[build_iter96_embcache_from_iter95_pool] holdout: matched={len(holdout_ids)}/{len(holdout_rows)} "
        f"unmatched={len(holdout_unmatched)}",
        file=sys.stderr,
    )
    if train_unmatched:
        print(f"[build_iter96_embcache_from_iter95_pool] unmatched train ids: {train_unmatched}", file=sys.stderr)
    if holdout_unmatched:
        print(f"[build_iter96_embcache_from_iter95_pool] unmatched holdout ids: {holdout_unmatched}", file=sys.stderr)


if __name__ == "__main__":
    main()

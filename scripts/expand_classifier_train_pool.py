"""Iter95 (classifier_training_volume_expansion=full_eval_disjoint_jmmlu_pool): additive
expansion of the classifier training pool.

Appends every JMMLU row that is disjoint from both the evaluation dataset
(`data/dataset.jsonl`) and the existing classifier training file to a new
output file. The rule is a single, 10-domain-common, parameter-free
statement: "for each domain, add every row of its JMMLU task pool that is
not already used by the evaluation set or the existing training set."
There is no sampling and no seed, because nothing is subsampled: every
disjoint row is added, so the output is deterministic given the same
JMMLU.zip and the same two input files.

This is a *pure addition*: the existing training rows are copied through
unchanged (byte-for-byte, same order, same ids), and new rows are appended
after them with ids `{domain}-pool095-NNN` and `sample_weight: 1.0`
(matching the existing convention documented in
build_dataset.py:build_classifier_training_rows).

Deliberately does not call build_dataset.py's CLI: `--domain-target-size`
there is shared between the evaluation dataset and the classifier training
pool, so regenerating via the CLI would also resample (and silently
change) `data/dataset.jsonl`. This script only reads `data/dataset.jsonl`
(to compute the exclusion set) and never writes it.

Usage:
    uv run python scripts/expand_classifier_train_pool.py \\
        --eval-data data/dataset.jsonl \\
        --existing-train-data data/classifier_train_iter94_dedup.jsonl \\
        --jmmlu-zip /tmp/expert-mesh-cache/JMMLU.zip \\
        --output data/classifier_train_iter95_fullpool.jsonl
"""

import argparse
import io
import json
import sys
import zipfile

from build_dataset import (
    _DOMAIN_TASK_MAP,
    _format_jmmlu_query,
    _load_jmmlu_zip_bytes,
    _parse_jmmlu_task_csv,
)


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_expanded_rows(
    jmmlu_zip_path: str,
    eval_data_path: str,
    existing_train_data_path: str,
) -> tuple[list[dict], list[dict]]:
    """Return (existing_rows_unchanged, new_rows) for the additive expansion.

    new_rows covers every domain's JMMLU pool row that is absent from both
    the evaluation set's non-compound queries and the existing training
    file's queries. Row order within a domain follows JMMLU task order
    (per _DOMAIN_TASK_MAP) and then CSV row order, so the result is fully
    deterministic (no sampling, no seed).
    """
    eval_rows = _load_jsonl(eval_data_path)
    eval_queries = frozenset(row["query"] for row in eval_rows if not row["is_compound"])

    existing_rows = _load_jsonl(existing_train_data_path)
    existing_queries = frozenset(row["query"] for row in existing_rows)

    exclude_queries = eval_queries | existing_queries

    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    new_rows: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for domain in sorted(_DOMAIN_TASK_MAP):
            counter = 0
            for task_name in _DOMAIN_TASK_MAP[domain]:
                for row in _parse_jmmlu_task_csv(zf, task_name):
                    query = _format_jmmlu_query(row)
                    if query in exclude_queries:
                        continue
                    counter += 1
                    new_rows.append(
                        {
                            "id": f"{domain}-pool095-{counter:03d}",
                            "query": query,
                            "domain": domain,
                            "sample_weight": 1.0,
                        }
                    )
    return existing_rows, new_rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Additively expand the classifier training pool with every JMMLU row "
            "disjoint from the evaluation set and the existing training file."
        )
    )
    parser.add_argument("--eval-data", default="data/dataset.jsonl")
    parser.add_argument("--existing-train-data", required=True)
    parser.add_argument("--jmmlu-zip", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    existing_rows, new_rows = build_expanded_rows(
        args.jmmlu_zip, args.eval_data, args.existing_train_data
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for row in existing_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    domain_counts = Counter(row["domain"] for row in new_rows)
    print(
        f"[expand_classifier_train_pool] existing={len(existing_rows)} "
        f"new={len(new_rows)} total={len(existing_rows) + len(new_rows)}",
        file=sys.stderr,
    )
    print(f"[expand_classifier_train_pool] new rows per domain: {dict(sorted(domain_counts.items()))}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

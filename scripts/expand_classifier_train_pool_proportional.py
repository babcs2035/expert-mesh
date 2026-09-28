"""Iter96 (composition_preserving_volume_expansion=per_task_proportional_scale):
per-task proportional expansion of the classifier training pool.

Unlike scripts/expand_classifier_train_pool.py (Iter95, which added the
*entire* disjoint JMMLU pool and thereby shifted each domain's per-task
composition), this script adds, for each JMMLU task, up to
`multiplier - 1` times the task's *current* training row count, capped by
the task's remaining pool. This keeps each domain's per-task composition
close to unchanged (target: total-variation distance <= 0.004 for the
pool-rich domains; see journal.md "Iteration 96" (2)) while still growing
n, so that n's effect on CV accuracy is not confounded with a composition
shift (journal.md "Iteration 96" hypothesis, mechanism M1).

Three additions on top of Iter95's expand_classifier_train_pool.py:
  (a) per-task cap: add min(cur_task * (multiplier - 1), pool_task) rows
      per task, where cur_task is the number of existing training rows
      whose query belongs to that task.
  (b) cross-task duplicate resolution: a JMMLU question that is verbatim
      shared by two tasks (Iter95 produced 5 such duplicates inside
      natural_science) is assigned to exactly one task -- the first one
      encountered while walking _DOMAIN_TASK_MAP in dict order (domains)
      and list order (tasks within a domain) -- so it is counted and
      offered for addition exactly once.
  (c) hold-out carve-out: before capping each task's pool at
      cur_task * (multiplier - 1), a fixed 20%-per-task slice (floor(L / 5)
      rows at CSV-order indices 0, 5, 10, ... within the task's already-
      deduplicated pool of length L; a partial final group of < 5 rows is
      truncated, not rounded up) is set aside as a hold-out that is never
      added to either arm's training set. The remaining (non-hold-out)
      pool is what gets capped in (a).

Deliberately does not call build_dataset.py's CLI: `--domain-target-size`
there is shared between the evaluation dataset and the classifier training
pool, so regenerating via the CLI would also resample (and silently
change) `data/dataset.jsonl`. This script only reads `data/dataset.jsonl`
(to compute the exclusion set) and never writes it.

Usage:
    uv run python scripts/expand_classifier_train_pool_proportional.py \\
        --eval-data data/dataset.jsonl \\
        --existing-train-data data/classifier_train_iter94_dedup.jsonl \\
        --jmmlu-zip /tmp/expert-mesh-cache/JMMLU.zip \\
        --multiplier 2 \\
        --output data/classifier_train_iter96_scaled.jsonl \\
        --holdout-output data/classifier_train_iter96_holdout.jsonl
"""

import argparse
import io
import json
import sys
import zipfile
from collections import Counter

from build_dataset import (
    _DOMAIN_TASK_MAP,
    _format_jmmlu_query,
    _load_jmmlu_zip_bytes,
    _parse_jmmlu_task_csv,
)

_HOLDOUT_MODULUS = 5  # per-task hold-out: keep every 5th row (index % 5 == 0), i.e. ~20%.


def _load_jsonl(path: str) -> list[dict]:
    """Read a JSONL file into a list of dicts (skips blank lines)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _assign_queries_to_tasks(zf: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    """Return {query: (domain, task_name)}, assigning each query to the first
    task that contains it while walking _DOMAIN_TASK_MAP in dict order and
    each task's CSV in row order. Resolves cross-task verbatim duplicates
    (e.g. the 5 natural_science duplicates found in Iter95) deterministically.
    """
    assignment: dict[str, tuple[str, str]] = {}
    for domain, task_names in _DOMAIN_TASK_MAP.items():
        for task_name in task_names:
            for row in _parse_jmmlu_task_csv(zf, task_name):
                query = _format_jmmlu_query(row)
                if query not in assignment:
                    assignment[query] = (domain, task_name)
    return assignment


def build_scaled_rows(
    jmmlu_zip_path: str,
    eval_data_path: str,
    existing_train_data_path: str,
    multiplier: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (existing_rows_unchanged, new_rows, holdout_rows).

    new_rows: per-task capped addition (b) obeying (a) and excluding the
    hold-out carve-out (c). holdout_rows: the carved-out rows, never
    included in new_rows and never usable for training by either arm.
    """
    eval_rows = _load_jsonl(eval_data_path)
    eval_queries = frozenset(row["query"] for row in eval_rows if not row["is_compound"])

    existing_rows = _load_jsonl(existing_train_data_path)
    existing_queries = frozenset(row["query"] for row in existing_rows)

    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        query_to_task = _assign_queries_to_tasks(zf)

        # cur_task: existing training rows' queries, bucketed by their assigned task.
        cur_task_counts: Counter[tuple[str, str]] = Counter()
        for query in existing_queries:
            task_key = query_to_task.get(query)
            if task_key is not None:
                cur_task_counts[task_key] += 1

        new_rows: list[dict] = []
        holdout_rows: list[dict] = []
        new_counter_by_domain: Counter[str] = Counter()
        holdout_counter_by_domain: Counter[str] = Counter()

        for domain, task_names in _DOMAIN_TASK_MAP.items():
            for task_name in task_names:
                task_key = (domain, task_name)
                # Task's disjoint pool, in CSV order, deduplicated to this task only.
                task_pool_queries: list[str] = []
                seen_in_task: set[str] = set()
                for row in _parse_jmmlu_task_csv(zf, task_name):
                    query = _format_jmmlu_query(row)
                    if query_to_task[query] != task_key:
                        continue  # verbatim duplicate assigned to an earlier task
                    if query in eval_queries or query in existing_queries:
                        continue
                    if query in seen_in_task:
                        continue  # verbatim duplicate row within this task's own CSV
                    seen_in_task.add(query)
                    task_pool_queries.append(query)

                # Per-task 20% hold-out, truncated (floor(len/5) rows, at
                # indices 0, 5, 10, ... so a partial final group of < 5 rows
                # is never rounded up into an extra hold-out row).
                holdout_count = len(task_pool_queries) // _HOLDOUT_MODULUS
                holdout_indices = {k * _HOLDOUT_MODULUS for k in range(holdout_count)}
                holdout_queries = [
                    q for i, q in enumerate(task_pool_queries) if i in holdout_indices
                ]
                remaining_pool_queries = [
                    q for i, q in enumerate(task_pool_queries) if i not in holdout_indices
                ]

                cur_task = cur_task_counts.get(task_key, 0)
                cap = cur_task * (multiplier - 1)
                add_queries = remaining_pool_queries[:cap]

                for query in add_queries:
                    new_counter_by_domain[domain] += 1
                    new_rows.append(
                        {
                            "id": f"{domain}-pool096-{new_counter_by_domain[domain]:03d}",
                            "query": query,
                            "domain": domain,
                            "sample_weight": 1.0,
                        }
                    )
                for query in holdout_queries:
                    holdout_counter_by_domain[domain] += 1
                    holdout_rows.append(
                        {
                            "id": f"{domain}-holdout096-{holdout_counter_by_domain[domain]:03d}",
                            "query": query,
                            "domain": domain,
                            "task": task_name,
                        }
                    )

    return existing_rows, new_rows, holdout_rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-task proportional (composition-preserving) expansion of the "
            "classifier training pool, plus a per-task hold-out carve-out."
        )
    )
    parser.add_argument("--eval-data", default="data/dataset.jsonl")
    parser.add_argument("--existing-train-data", required=True)
    parser.add_argument("--jmmlu-zip", required=True)
    parser.add_argument("--multiplier", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-output", required=True)
    args = parser.parse_args()

    existing_rows, new_rows, holdout_rows = build_scaled_rows(
        args.jmmlu_zip, args.eval_data, args.existing_train_data, args.multiplier
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for row in existing_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.holdout_output, "w", encoding="utf-8") as f:
        for row in holdout_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    new_domain_counts = Counter(row["domain"] for row in new_rows)
    holdout_domain_counts = Counter(row["domain"] for row in holdout_rows)
    print(
        f"[expand_classifier_train_pool_proportional] existing={len(existing_rows)} "
        f"new={len(new_rows)} total={len(existing_rows) + len(new_rows)} "
        f"holdout={len(holdout_rows)}",
        file=sys.stderr,
    )
    print(f"[expand_classifier_train_pool_proportional] new rows per domain: "
          f"{dict(sorted(new_domain_counts.items()))}", file=sys.stderr)
    print(f"[expand_classifier_train_pool_proportional] holdout rows per domain: "
          f"{dict(sorted(holdout_domain_counts.items()))}", file=sys.stderr)


if __name__ == "__main__":
    main()

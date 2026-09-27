"""Iter85 (single_domain_eval_set_expansion=jmmlu_unused_rows_power_targeted): sample
additional JMMLU single-domain rows to expand the evaluation dataset from 1,915 to
3,447 rows (1,532 new rows), raising McNemar detection power for future single-lever
comparisons. This script only produces the standalone JSONL of new rows; it does not
touch data/dataset.jsonl itself (build_dataset.py's --single-domain-expansion appends
it at build time, journal.md "Iteration 85" 計画節 参照).

Selection regime (10 domains share one rule; no per-domain parameters):

1. Pool: for each domain, all JMMLU rows (build_dataset.py's `_DOMAIN_TASK_MAP`) minus
   the current evaluation set's 1,915 question texts (data/dataset.jsonl) and the
   classifier training set's 1,427 question texts (data/classifier_train.jsonl).
2. Training reserve: 100 rows per domain are held back from the pool (not eligible
   for selection here) so a future cross_domain_training_data_augmentation retry
   still has an unused pool to draw from.
3. Target count: min(200, max(0, pool_size - 100)) rows per domain.
4. Layer order (clean-first): within the eligible pool, rows whose question text does
   NOT appear in any data/lora_train/*.jsonl file are shuffled first and taken before
   rows that DO appear there (LoRA-contaminated rows only fill remaining quota). Both
   layers are shuffled with the same random.Random(20260927) instance, processing
   domains in the fixed order below, clean layer before dirty layer -- so the run is
   fully reproducible from an unchanged JMMLU.zip.
5. Output ids: "{domain}-exp085-{seq:03d}", seq starting at 1 per domain, in the order
   rows were selected (clean layer rows first, then dirty layer rows if needed).

No difficulty-based selection is used (evaluation rows must not be biased toward
harder/easier items, which would shift the top1 level and break comparability with
the existing 1,915-row subset and past baselines).

Usage:
    uv run python scripts/expand_single_domain_eval.py \\
        --output data/single_domain_expansion_iter85.jsonl \\
        --jmmlu-zip /path/to/JMMLU.zip
"""

import argparse
import io
import json
import random
import sys
import zipfile
from pathlib import Path

# Reuse build_dataset.py's JMMLU parsing helpers instead of re-deriving them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_dataset import (  # noqa: E402
    _DOMAIN_TASK_MAP,
    _format_jmmlu_query,
    _load_jmmlu_zip_bytes,
    _parse_jmmlu_task_csv,
)

_SEED = 20260927
_TRAIN_RESERVE = 100
_TARGET_PER_DOMAIN = 200

# Fixed domain processing order (matches journal.md Iteration 85 計画節's table so the
# rng consumption sequence, and therefore the sampled rows, is reproducible).
_DOMAIN_ORDER = [
    "medical", "education", "business_economics", "natural_science", "mathematics",
    "history_culture", "computer_science", "social_science", "general", "legal",
]
assert set(_DOMAIN_ORDER) == set(_DOMAIN_TASK_MAP)


def _load_query_set(path: str) -> frozenset[str]:
    """Return the set of `query` (or chat user-turn content) strings found in a JSONL file."""
    queries: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "query" in row:
                queries.add(row["query"])
            elif "messages" in row:
                for message in row["messages"]:
                    if message.get("role") == "user":
                        queries.add(message["content"])
    return queries


def _load_lora_train_query_set(lora_train_dir: str) -> frozenset[str]:
    """Union of question texts across all data/lora_train/*.jsonl files."""
    queries: set[str] = set()
    for path in sorted(Path(lora_train_dir).glob("*.jsonl")):
        queries |= _load_query_set(str(path))
    return queries


def _build_domain_pool(
    zf: zipfile.ZipFile,
    domain: str,
    exclude_queries: frozenset[str],
) -> list[tuple[str, str, str]]:
    """All (query, answer, task_name) rows for a domain's tasks, minus exclude_queries.

    Deduplicates on query text across the domain's own constituent tasks (JMMLU has one
    known cross-task duplicate question, in education's sociology/moral_disputes proxy
    tasks) so the returned pool never yields two rows with identical query text.
    """
    pool: list[tuple[str, str, str]] = []
    seen_in_domain: set[str] = set()
    for task_name in _DOMAIN_TASK_MAP[domain]:
        for row in _parse_jmmlu_task_csv(zf, task_name):
            query = _format_jmmlu_query(row)
            if query in exclude_queries or query in seen_in_domain:
                continue
            seen_in_domain.add(query)
            pool.append((query, row["answer"], task_name))
    return pool


def build_expansion_rows(
    zip_bytes: bytes,
    eval_queries: frozenset[str],
    classifier_train_queries: frozenset[str],
    lora_train_queries: frozenset[str],
) -> tuple[list[dict], dict[str, int]]:
    """Return (new rows, {domain: selected_count}) per the selection regime above."""
    exclude_queries = eval_queries | classifier_train_queries
    rng = random.Random(_SEED)
    rows: list[dict] = []
    counts: dict[str, int] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for domain in _DOMAIN_ORDER:
            pool = _build_domain_pool(zf, domain, exclude_queries)
            eligible_count = max(0, len(pool) - _TRAIN_RESERVE)
            target = min(_TARGET_PER_DOMAIN, eligible_count)

            clean_layer = [row for row in pool if row[0] not in lora_train_queries]
            dirty_layer = [row for row in pool if row[0] in lora_train_queries]
            rng.shuffle(clean_layer)
            rng.shuffle(dirty_layer)

            selected = (clean_layer + dirty_layer)[:target]
            counts[domain] = len(selected)
            for seq, (query, answer, task_name) in enumerate(selected, start=1):
                rows.append(
                    {
                        "id": f"{domain}-exp085-{seq:03d}",
                        "query": query,
                        "expected_domains": [domain],
                        "is_compound": False,
                        "jmmlu_task": task_name,
                        "jmmlu_answer": answer,
                    }
                )
    return rows, counts


def _assert_no_duplicates(rows: list[dict]) -> None:
    """Assert the new rows contain no internal duplicate question texts."""
    queries = [row["query"] for row in rows]
    assert len(queries) == len(set(queries)), "duplicate query text within expansion rows"


def _assert_disjoint(rows: list[dict], eval_queries: frozenset[str], classifier_train_queries: frozenset[str]) -> None:
    """Assert the new rows share no question text with the eval set or classifier train set."""
    new_queries = {row["query"] for row in rows}
    assert not (new_queries & eval_queries), "expansion rows overlap with data/dataset.jsonl"
    assert not (new_queries & classifier_train_queries), (
        "expansion rows overlap with data/classifier_train.jsonl"
    )


# journal.md "Iteration 85" 計画節の配分表（投資フェーズで実測した pool サイズを基に
# 決めた目標値）. This repo's cached JMMLU.zip has sha256 3ba7d912... rather than the
# pinned-commit's expected 3637b25e... (a pre-existing, already-flagged-benign content
# discrepancy; see journal_archive.md's Iter?? note), so its per-task row counts differ
# by a handful from what the plan's table assumed. Actual counts are asserted against
# this table with a small tolerance rather than exact equality (main() prints both so
# the analysis phase can re-verify P1/P3 against the achieved, not planned, N).
_PLANNED_COUNTS = {
    "medical": 200,
    "education": 200,
    "business_economics": 200,
    "natural_science": 200,
    "mathematics": 200,
    "history_culture": 200,
    "computer_science": 156,
    "social_science": 148,
    "general": 28,
    "legal": 0,
}
_COUNT_TOLERANCE = 10


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sample Iter85's 1,532 additional single-domain JMMLU rows"
    )
    parser.add_argument("--output", default="data/single_domain_expansion_iter85.jsonl")
    parser.add_argument("--jmmlu-zip", default=None, help="Local path to JMMLU.zip")
    parser.add_argument("--eval-dataset", default="data/dataset.jsonl")
    parser.add_argument("--classifier-train-data", default="data/classifier_train.jsonl")
    parser.add_argument("--lora-train-dir", default="data/lora_train")
    args = parser.parse_args()

    eval_queries = _load_query_set(args.eval_dataset)
    classifier_train_queries = _load_query_set(args.classifier_train_data)
    lora_train_queries = _load_lora_train_query_set(args.lora_train_dir)
    zip_bytes = _load_jmmlu_zip_bytes(args.jmmlu_zip)

    rows, counts = build_expansion_rows(
        zip_bytes, eval_queries, classifier_train_queries, lora_train_queries
    )

    _assert_no_duplicates(rows)
    _assert_disjoint(rows, eval_queries, classifier_train_queries)
    for domain, planned in _PLANNED_COUNTS.items():
        actual = counts[domain]
        assert abs(actual - planned) <= _COUNT_TOLERANCE, (
            f"{domain}: actual count {actual} deviates from planned {planned} by more "
            f"than tolerance {_COUNT_TOLERANCE} (pool smaller than the plan's investigate-"
            f"phase estimate; see this module's docstring on the JMMLU.zip sha mismatch)"
        )
    print(
        f"[expand_single_domain_eval] planned vs actual counts: {_PLANNED_COUNTS} vs {counts}",
        file=sys.stderr,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dirty_count = sum(1 for row in rows if row["query"] in lora_train_queries)
    print(f"[expand_single_domain_eval] wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    print(f"[expand_single_domain_eval] per-domain counts: {counts}", file=sys.stderr)
    print(
        f"[expand_single_domain_eval] LoRA-train-contaminated rows: {dirty_count}/{len(rows)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

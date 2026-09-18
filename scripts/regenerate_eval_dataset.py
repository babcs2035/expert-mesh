"""Regenerate the eval dataset (data/dataset.jsonl) with a new _DOMAIN_TASK_MAP.

This script is used for Iter37 (classifier_training_data_composition=
history_culture_japanese_civics_reassignment_to_education). The original
eval dataset was built with education mapped to sociology/high_school_
psychology/moral_disputus. After reassigning japanese_civics from
history_culture to education, the eval dataset must be regenerated so
that education eval rows come from japanese_civics (150 rows) and
history_culture eval rows come from the remaining 7 tasks (126 rows).

The compound questions (100 rows) are preserved unchanged.

Usage:
    uv run python -m scripts.regenerate_eval_dataset \\
        --jmmlu-zip /path/to/JMMLU.zip \\
        --output data/dataset.jsonl
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter

# New _DOMAIN_TASK_MAP for Iter37 (japanese_civics removed from history_culture)
_DOMAIN_TASK_MAP = {
    "medical": [
        "virology", "nutrition", "human_sexuality", "clinical_knowledge",
        "human_aging", "anatomy", "professional_psychology", "college_medicine",
        "professional_medicine", "medical_genetics",
    ],
    "legal": ["international_law", "jurisprudence"],
    "education": ["japanese_civics"],
    "business_economics": [
        "econometrics", "high_school_microeconomics", "business_ethics",
        "marketing", "high_school_macroeconomics", "management",
        "public_relations", "professional_accounting",
    ],
    "computer_science": [
        "computer_security", "machine_learning", "high_school_computer_science",
        "college_computer_science", "electrical_engineering",
    ],
    "natural_science": [
        "high_school_chemistry", "high_school_physics", "college_physics",
        "conceptual_physics", "college_biology", "high_school_biology",
        "college_chemistry", "astronomy",
    ],
    "mathematics": [
        "college_mathematics", "high_school_statistics", "elementary_mathematics",
        "high_school_mathematics", "abstract_algebra",
    ],
    "history_culture": [
        "japanese_history", "high_school_european_history",
        "prehistory", "japanese_idiom", "japanese_geography",
        "high_school_geography", "world_history",
    ],
    "social_science": [
        "security_studies", "world_religions", "philosophy", "global_facts",
    ],
    "general": ["miscellaneous", "logical_fallacies", "formal_logic"],
}

# Target rows per domain (single-domain, non-compound)
_DOMAIN_TARGET_SIZE = 150

# Compound questions are loaded from an existing eval dataset or generated inline
_COMPOUND_QUESTIONS_PATH = None  # Set to path of existing dataset to extract compound rows


def _load_jmmlu_tasks(zf: zipfile.ZipFile) -> dict[str, list[dict]]:
    """Load all JMMLU tasks from the zip file into memory."""
    tasks = {}
    for task_name in _DOMAIN_TASK_MAP.values():
        for t in task_name:
            try:
                raw_bytes = zf.read(f"JMMLU/test/{t}.csv")
                text = raw_bytes.decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(text))
                tasks[t] = list(reader)
            except KeyError:
                print(f"[regenerate_eval] WARNING: Task '{t}' not found in zip", file=sys.stderr)
    return tasks


def _format_query(row: dict[str, str]) -> str:
    """Format a JMMLU row as a four-choice question prompt."""
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"


def _load_compound_questions(dataset_path: str | None) -> list[dict]:
    """Load compound questions from an existing eval dataset, or return empty list."""
    if dataset_path is None or not os.path.exists(dataset_path):
        print("[regenerate_eval] No existing dataset for compound questions; skipping", file=sys.stderr)
        return []
    compounds = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("is_compound"):
                compounds.append(row)
    return compounds


def _build_eval_rows(
    tasks: dict[str, list[dict]],
    domain_task_map: dict[str, list[str]],
    target_size: int,
    used_queries: set[str],
    seed: int = 42,
) -> list[dict]:
    """Build eval rows for a single domain, excluding already-used queries."""
    rng = random.Random(seed)
    pool: list[dict] = []
    for task_name in domain_task_map:
        if task_name not in tasks:
            continue
        for row in tasks[task_name]:
            query = _format_query(row)
            if query not in used_queries:
                pool.append({
                    "query": query,
                    "domain": task_name,
                    "expected_domain": None,  # filled below
                    "jmmlu_answer": row["answer"],
                })

    rng.shuffle(pool)

    rows = []
    for item in pool[:target_size]:
        rows.append({
            "id": f"{item['expected_domain']}-{len(rows)+1:03d}",
            "expected_domains": [item["expected_domain"]],
            "is_compound": False,
            "jmmlu_task": item["domain"],
            "jmmlu_answer": item["jmmlu_answer"],
            "query": item["query"],
        })
        used_queries.add(item["query"])
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate eval dataset with new _DOMAIN_TASK_MAP"
    )
    parser.add_argument(
        "--jmmlu-zip",
        required=True,
        help="Path to JMMLU.zip file",
    )
    parser.add_argument(
        "--output",
        default="data/dataset.jsonl",
        help="Output path for the regenerated eval dataset",
    )
    parser.add_argument(
        "--input-dataset",
        default=None,
        help="Path to existing dataset (for compound questions and seed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load JMMLU tasks
    print(f"[regenerate_eval] Loading JMMLU from: {args.jmmlu_zip}", file=sys.stderr)
    with zipfile.ZipFile(args.jmmlu_zip) as zf:
        tasks = _load_jmmlu_tasks(zf)

    # Load existing compound questions if available
    compound_rows = _load_compound_questions(args.input_dataset)
    print(f"[regenerate_eval] Loaded {len(compound_rows)} compound questions", file=sys.stderr)

    # Build eval rows domain by domain
    all_rows = []
    used_queries = set()
    domain_counts = Counter()

    for domain in _DOMAIN_TASK_MAP:
        task_names = _DOMAIN_TASK_MAP[domain]
        print(
            f"[regenerate_eval] Building {domain} ({len(task_names)} tasks, "
            f"target={_DOMAIN_TARGET_SIZE})",
            file=sys.stderr,
        )

        rows = _build_eval_rows(
            tasks=tasks,
            domain_task_map=task_names,
            target_size=_DOMAIN_TARGET_SIZE,
            used_queries=used_queries,
            seed=args.seed,
        )
        domain_counts[domain] = len(rows)

        # Update IDs and expected_domains
        for i, row in enumerate(rows, start=1):
            row["id"] = f"{domain}-{i:03d}"
            row["expected_domains"] = [domain]

        all_rows.extend(rows)
        print(
            f"[regenerate_eval]   {domain}: {len(rows)} rows "
            f"(tasks: {task_names})",
            file=sys.stderr,
        )

    # Add compound questions
    all_rows.extend(compound_rows)

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"[regenerate_eval] Wrote {len(all_rows)} rows to {args.output}",
        file=sys.stderr,
    )
    print(
        f"[regenerate_eval] Domain distribution: {dict(domain_counts)}",
        file=sys.stderr,
    )
    print(
        f"[regenerate_eval] Total single-domain: {sum(domain_counts.values())}, "
        f"compound: {len(compound_rows)}",
        file=sys.stderr,
    )

    # Verification
    edu_rows = [r for r in all_rows if not r.get("is_compound") and "education" in r.get("expected_domains", [])]
    hc_rows = [r for r in all_rows if not r.get("is_compound") and "history_culture" in r.get("expected_domains", [])]
    edu_tasks = Counter(r.get("jmmlu_task") for r in edu_rows)
    hc_tasks = Counter(r.get("jmmlu_task") for r in hc_rows)
    print(f"[regenerate_eval] Education eval tasks: {dict(edu_tasks)}", file=sys.stderr)
    print(f"[regenerate_eval] History_culture eval tasks: {dict(hc_tasks)}", file=sys.stderr)


if __name__ == "__main__":
    main()

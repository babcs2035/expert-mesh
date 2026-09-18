"""Regenerate classifier training data for Iter37.

Iter37 (classifier_training_data_composition=history_culture_japanese_civics_reassignment_to_education):
- education maps to japanese_civics only (150 rows)
- history_culture maps to 7 tasks (no japanese_civics), 150 rows
- No Iter35 handmade questions (single-lever: only reassignment, no handmade)
- japanese_civics pool size is exactly 150, same as eval target, so all 150
  questions are used for both eval and classifier training. This is a known
  data limitation of the japanese_civics proxy task.

Usage:
    uv run python scripts/regenerate_classifier_train_iter37.py \\
        --jmmlu-zip /path/to/JMMLU.zip \\
        --output data/classifier_train_iter37_reassigned.jsonl
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


def _load_jmmlu_tasks(zf: zipfile.ZipFile) -> dict[str, list[dict]]:
    """Load all JMMLU tasks from the zip file into memory."""
    tasks = {}
    for task_name in [
        "virology", "nutrition", "human_sexuality", "clinical_knowledge",
        "human_aging", "anatomy", "professional_psychology", "college_medicine",
        "professional_medicine", "medical_genetics",
        "international_law", "jurisprudence",
        "japanese_civics",
        "econometrics", "high_school_microeconomics", "business_ethics",
        "marketing", "high_school_macroeconomics", "management",
        "public_relations", "professional_accounting",
        "computer_security", "machine_learning", "high_school_computer_science",
        "college_computer_science", "electrical_engineering",
        "high_school_chemistry", "high_school_physics", "college_physics",
        "conceptual_physics", "college_biology", "high_school_biology",
        "college_chemistry", "astronomy",
        "college_mathematics", "high_school_statistics", "elementary_mathematics",
        "high_school_mathematics", "abstract_algebra",
        "japanese_history", "high_school_european_history", "prehistory",
        "japanese_idiom", "japanese_geography", "high_school_geography",
        "world_history",
        "security_studies", "world_religions", "philosophy", "global_facts",
        "miscellaneous", "logical_fallacies", "formal_logic",
    ]:
        try:
            raw_bytes = zf.read(f"JMMLU/test/{task_name}.csv")
            text = raw_bytes.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            tasks[task_name] = list(reader)
        except KeyError:
            print(f"[regen] WARNING: Task '{task_name}' not found in zip", file=sys.stderr)
    return tasks


def _format_query(row: dict[str, str]) -> str:
    """Format a JMMLU row as a four-choice question prompt."""
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"


def _build_classifier_rows(
    tasks: dict[str, list[dict]],
    domain_task_map: dict[str, list[str]],
    domain_target_size: int,
    seed: int = 20260727,
) -> list[dict]:
    """Build classifier training rows for all domains.

    Note: For education (japanese_civics), the pool size is exactly 150,
    same as the eval target. So all 150 questions are used for both eval
    and classifier training. This is a known data limitation.
    """
    rng = random.Random(seed)
    all_rows = []

    for domain in sorted(domain_task_map):
        task_names = domain_task_map[domain]
        pool: list[tuple[str, str]] = []  # (query, task_name)

        for task_name in task_names:
            if task_name not in tasks:
                continue
            for row in tasks[task_name]:
                query = _format_query(row)
                pool.append((query, task_name))

        # Sample up to target_size
        sample_size = min(domain_target_size, len(pool))
        if sample_size < len(pool):
            sampled = rng.sample(pool, sample_size)
        else:
            sampled = pool

        for index, (query, task_name) in enumerate(sampled, start=1):
            all_rows.append({
                "id": f"{domain}-train-{index:03d}",
                "query": query,
                "domain": domain,
                "sample_weight": 1.0,
            })

    return all_rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate classifier training data for Iter37"
    )
    parser.add_argument(
        "--jmmlu-zip",
        required=True,
        help="Path to JMMLU.zip file",
    )
    parser.add_argument(
        "--output",
        default="data/classifier_train_iter37_reassigned.jsonl",
        help="Output path for the regenerated classifier training data",
    )
    args = parser.parse_args()

    # Load JMMLU tasks
    print(f"[regen] Loading JMMLU from: {args.jmmlu_zip}", file=sys.stderr)
    with zipfile.ZipFile(args.jmmlu_zip) as zf:
        tasks = _load_jmmlu_tasks(zf)

    # Domain-task map (current Iter37: japanese_civics -> education, removed from history_culture)
    domain_task_map = {
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

    # Build classifier rows
    rows = _build_classifier_rows(
        tasks=tasks,
        domain_task_map=domain_task_map,
        domain_target_size=150,
    )

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    domain_counts = Counter(row["domain"] for row in rows)
    print(f"[regen] Wrote {len(rows)} rows to {args.output}", file=sys.stderr)
    print(f"[regen] Domain distribution: {dict(domain_counts)}", file=sys.stderr)
    print(f"[regen] Education rows: {domain_counts.get('education', 0)}", file=sys.stderr)
    print(f"[regen] History_culture rows: {domain_counts.get('history_culture', 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()

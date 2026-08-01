"""Prepare LoRA training data from JMMLU, excluding the evaluation set.

Extracts instruction-response pairs for specified domains from the JMMLU
dataset, ensuring complete separation from the evaluation dataset questions.
Outputs JSONL in the format expected by train_domain_lora.py:
    {"instruction": "<query>", "input": "", "output": "<answer>"}

Usage:
    uv run python scripts/prepare_lora_training_data.py \\
        --domains medical legal \\
        --output-dir data/lora_train \\
        --eval-dataset data/dataset.jsonl \\
        --jmmlu-zip /path/to/JMMLU.zip
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import zipfile
from typing import TextIO

import httpx

_JMMLU_ZIP_SHA = "3637b25e444ccfdcde4d23a783cbe8e674faa01b"
_JMMLU_ZIP_URL = (
    f"https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/{_JMMLU_ZIP_SHA}/JMMLU.zip"
)
_JMMLU_DOWNLOAD_TIMEOUT_S = 120.0
_JMMLU_CSV_PATH_TEMPLATE = "JMMLU/test/{task_name}.csv"

_DOMAIN_TASK_MAP: dict[str, list[str]] = {
    "medical": [
        "virology", "nutrition", "human_sexuality", "clinical_knowledge",
        "human_aging", "anatomy", "professional_psychology", "college_medicine",
        "professional_medicine", "medical_genetics",
    ],
    "legal": ["international_law", "jurisprudence"],
    "education": ["japanese_civics", "sociology", "high_school_psychology", "moral_disputes"],
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

# Domain-specific system prompts for LoRA training context
_DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    "medical": "あなたは医療分野の専門家です．医学的な質問に正確かつ専門的に回答してください．",
    "legal": "あなたは法律分野の専門家です．法的な質問に正確かつ専門的に回答してください．",
    "education": "あなたは教育分野の専門家です．教育的な質問に正確かつ専門的に回答してください．",
    "business_economics": "あなたはビジネス・経済分野の専門家です．ビジネスや経済に関する質問に正確かつ専門的に回答してください．",
    "computer_science": "あなたはコンピュータサイエンス分野の専門家です．コンピュータサイエンスに関する質問に正確かつ専門的に回答してください．",
    "natural_science": "あなたは自然科学分野の専門家です．自然科学に関する質問に正確かつ専門的に回答してください．",
    "mathematics": "あなたは数学分野の専門家です．数学に関する質問に正確かつ専門的に回答してください．",
    "history_culture": "あなたは歴史・文化分野の専門家です．歴史や文化に関する質問に正確かつ専門的に回答してください．",
    "social_science": "あなたは社会科学分野の専門家です．社会科学に関する質問に正確かつ専門的に回答してください．",
    "general": "あなたは一般的な知識を持つアシスタントです．様々な質問に正確に回答してください．",
}


def _load_jmmlu_zip(jmmlu_zip_path: str | None) -> zipfile.ZipFile:
    """Load JMMLU.zip from local path or download it."""
    if jmmlu_zip_path is not None and os.path.exists(jmmlu_zip_path):
        print(f"[prepare_lora_train] Loading JMMLU from: {jmmlu_zip_path}", file=sys.stderr)
        return zipfile.ZipFile(jmmlu_zip_path)
    print(f"[prepare_lora_train] Downloading JMMLU from HuggingFace...", file=sys.stderr)
    response = httpx.get(_JMMLU_ZIP_URL, timeout=_JMMLU_DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    response.raise_for_status()
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "JMMLU.zip")
    with open(tmp_path, "wb") as f:
        f.write(response.content)
    print(f"[prepare_lora_train] Downloaded to {tmp_path}", file=sys.stderr)
    return zipfile.ZipFile(tmp_path)


def _parse_jmmlu_task_csv(zf: zipfile.ZipFile, task_name: str) -> list[dict[str, str]]:
    """Parse one JMMLU task's CSV into rows."""
    raw_bytes = zf.read(f"JMMLU/test/{task_name}.csv")
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return [{key: value.strip() for key, value in row.items()} for row in reader]


def _format_query(row: dict[str, str]) -> str:
    """Format a JMMLU row as a four-choice question prompt."""
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"


def _load_eval_queries(eval_dataset_path: str) -> set[str]:
    """Load queries from the evaluation dataset to exclude them from training."""
    eval_queries = set()
    with open(eval_dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            query = row.get("query", "")
            if query:
                eval_queries.add(query)
    return eval_queries


def _prepare_domain_data(
    zf: zipfile.ZipFile,
    domain: str,
    eval_queries: set[str],
    seed: int = 42,
    max_samples: int = 300,
) -> list[dict]:
    """Extract training samples for a domain, excluding evaluation queries."""
    task_names = _DOMAIN_TASK_MAP.get(domain, [])
    if not task_names:
        print(f"[prepare_lora_train] WARNING: No tasks mapped for domain '{domain}'", file=sys.stderr)
        return []

    pool: list[dict] = []
    for task_name in task_names:
        try:
            rows = _parse_jmmlu_task_csv(zf, task_name)
            for row in rows:
                query = _format_query(row)
                if query not in eval_queries:
                    pool.append({
                        "query": query,
                        "answer": row["answer"],
                        "task": task_name,
                    })
        except Exception as e:
            print(f"[prepare_lora_train] WARNING: Failed to parse task '{task_name}': {e}", file=sys.stderr)

    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(pool)

    # Cap at max_samples
    if len(pool) > max_samples:
        pool = pool[:max_samples]

    return pool


def _format_instruction_tuning(sample: dict, system_prompt: str) -> dict:
    """Format a sample as instruction-tuning data with chat messages."""
    # Create a response that includes the correct answer with brief reasoning
    answer = sample["answer"]
    response = f"正解は {answer} です．"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["query"]},
            {"role": "assistant", "content": response},
        ]
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training data from JMMLU"
    )
    parser.add_argument(
        "--domains", nargs="+", required=True,
        help="Domains to prepare training data for (e.g., medical legal)"
    )
    parser.add_argument(
        "--output-dir", default="data/lora_train",
        help="Output directory for training data JSONL files"
    )
    parser.add_argument(
        "--eval-dataset", default="data/dataset.jsonl",
        help="Path to evaluation dataset (questions to exclude from training)"
    )
    parser.add_argument(
        "--jmmlu-zip", default=None,
        help="Local path to JMMLU.zip (downloads if not provided)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=300,
        help="Max training samples per domain (default: 300)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Validate domains
    for domain in args.domains:
        if domain not in _DOMAIN_TASK_MAP:
            print(f"[prepare_lora_train] ERROR: Unknown domain '{domain}'", file=sys.stderr)
            print(f"  Available domains: {list(_DOMAIN_TASK_MAP.keys())}", file=sys.stderr)
            sys.exit(1)

    # Load evaluation queries to exclude
    eval_queries = _load_eval_queries(args.eval_dataset)
    print(f"[prepare_lora_train] Loaded {len(eval_queries)} evaluation queries to exclude", file=sys.stderr)

    # Load JMMLU
    with _load_jmmlu_zip(args.jmmlu_zip) as zf:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        for domain in args.domains:
            print(f"\n[prepare_lora_train] Processing domain: {domain}", file=sys.stderr)
            samples = _prepare_domain_data(zf, domain, eval_queries, seed=args.seed, max_samples=args.max_samples)
            print(f"[prepare_lora_train] Found {len(samples)} training samples for {domain}", file=sys.stderr)

            if not samples:
                print(f"[prepare_lora_train] WARNING: No training data for {domain}, skipping", file=sys.stderr)
                continue

            # Format as instruction-tuning data
            system_prompt = _DOMAIN_SYSTEM_PROMPTS.get(domain, "")
            training_data = [_format_instruction_tuning(s, system_prompt) for s in samples]

            # Write output
            output_path = os.path.join(args.output_dir, f"{domain}.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[prepare_lora_train] Wrote {len(training_data)} samples to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

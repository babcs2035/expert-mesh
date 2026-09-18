"""Iter60 (multilabel_training_signal=synthetic_two_domain_training_examples): LLM
generation of natural-language consultation sentences that genuinely require
knowledge of TWO domains at once, to be used as multi-label training rows for
scripts/train_multilabel_dispatch_head.py.

Design rationale (journal.md Iter60 plan, "生成方式の選択"): arXiv 2312.11276
found that concatenating two single-label examples ("Concat") consistently
underperforms LLM generation for multi-label data augmentation because
concatenated text is "neither semantically nor syntactically coherent". This
script instead asks config.yaml's judge_model (via
expert_backend.OllamaClient.generate(), already used in production by
scripts/evaluate_response_quality.py -- no new client code needed) to write
ONE coherent 1-2 sentence Japanese consultation that needs both domains'
knowledge to answer, for every (domain1, domain2) pair.

LEAKAGE GUARD (journal.md Iter60 plan, "必須の制約"): this module never
imports build_dataset. The 10 domain names are derived from
data/classifier_train.jsonl's own `domain` column (see
_load_domain_names()), not from build_dataset._DOMAIN_TASKS, and the
per-domain description strings used in prompts (_DOMAIN_DESCRIPTIONS_JA)
are authored independently of build_dataset._COMPOUND_QUESTIONS (the
100-question evaluation set) -- neither the text nor the "scenario ideas"
of those questions were read while writing this dict. The one place this
script does read _COMPOUND_QUESTIONS is the --audit-leak mode (A7 below),
which is a post-hoc near-duplicate detector, not a selector that
influences what gets generated or kept; the import for it is local to
_audit_leak() so it never executes during normal generation.

Usage (module mode; requires a live ollama node, e.g. via
`ssh -fNT -L 11435:localhost:11434 wafl500`):
    uv run python -m scripts.generate_multidomain_training_examples \\
        --train-data data/classifier_train.jsonl \\
        --model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \\
        --ollama-host 127.0.0.1 --ollama-port 11435 \\
        --per-pair 3 --per-pair-legal 5 \\
        --output data/classifier_train_multidomain.jsonl

    # A7 leak audit (reads the already-written --output file; does not
    # regenerate or filter anything):
    uv run python -m scripts.generate_multidomain_training_examples --audit-leak \\
        --output data/classifier_train_multidomain.jsonl
"""

import argparse
import asyncio
import itertools
import json
import sys

from expert_backend import OllamaClient

# Generation-quality filters (journal.md Iter60 plan, "生成後フィルタ").
_MIN_QUERY_LENGTH = 20
_MAX_QUERY_LENGTH = 200
# JMMLU-style four-choice markers; a generated row containing these would
# reproduce classifier_train.jsonl's four-choice-question style instead of
# the natural-consultation style the evaluation set's compound questions use.
_FOUR_CHOICE_MARKERS = ("A.", "B.", "C.", "D.", "Ａ．", "Ｂ．", "Ｃ．", "Ｄ．")
_MAX_GENERATION_ATTEMPTS_PER_SLOT = 3
_GENERATION_TEMPERATURE = 0.8

# Per-pair generation counts (journal.md Iter60 plan, "生成規模"): 45 pairs x 3
# rows as the base, with the 9 legal-involving pairs raised to 5 rows each
# (legal has only 77 of 1427 rows in classifier_train.jsonl, the fewest of
# any domain, and appears in 30/100 of the evaluation set's compound
# questions -- the most of any domain -- so its binary-relevance problem is
# the most exposed to positive-class scarcity; see journal Q1/Q3).
_DEFAULT_ROWS_PER_PAIR = 3
_DEFAULT_ROWS_PER_LEGAL_PAIR = 5
_MINIMUM_ACCEPTABLE_ROW_COUNT = 120  # journal Iter60 plan: below this, do not proceed

# Leak audit threshold (A7): character 3-gram Jaccard similarity against the
# evaluation set's 100 hand-authored compound questions. This is a
# near-duplicate/plagiarism guard, not a content selector (see module
# docstring) -- generated rows are never filtered by this score outside of
# --audit-leak reporting.
_LEAK_AUDIT_JACCARD_THRESHOLD = 0.9
_LEAK_AUDIT_NGRAM_SIZE = 3

# Domain descriptions used only to build generation prompts. Domain names
# themselves come from data/classifier_train.jsonl (see _load_domain_names);
# these Japanese descriptions were authored from the domain names alone, not
# from reading build_dataset.py's _COMPOUND_QUESTIONS.
_DOMAIN_DESCRIPTIONS_JA: dict[str, str] = {
    "business_economics": "経営学・経済学（マーケティング、会計、金融、経済理論など）",
    "computer_science": "情報科学（アルゴリズム、データベース、ネットワーク、プログラミングなど）",
    "education": "教育学（授業設計、学習理論、学校運営など）",
    "general": "特定の専門分野に限らない一般的な話題や日常の相談ごと",
    "history_culture": "歴史・文化（歴史的出来事、文化人類学、芸術など）",
    "legal": "法律（契約、労働法、民事・刑事手続き、規制など）",
    "mathematics": "数学（代数、解析、統計、幾何など）",
    "medical": "医学・健康（診断、治療、公衆衛生など）",
    "natural_science": "自然科学（物理学、化学、生物学、地学など）",
    "social_science": "社会科学（社会学、心理学、政治学など）",
}


def _load_domain_names(train_data_path: str) -> list[str]:
    """Derive the domain name set from classifier_train.jsonl's own `domain` column.

    Deliberately does not import build_dataset._DOMAIN_TASKS (see module
    docstring's leakage guard): this is the only source of domain names for
    generation.
    """
    domains: set[str] = set()
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                domains.add(json.loads(line)["domain"])
    return sorted(domains)


def _build_prompt(domain1: str, domain2: str) -> str:
    """Prompt asking the judge_model for one natural Japanese consultation needing both domains."""
    description1 = _DOMAIN_DESCRIPTIONS_JA[domain1]
    description2 = _DOMAIN_DESCRIPTIONS_JA[domain2]
    return (
        "あなたは日本語の相談文を作成するアシスタントです。\n"
        f"分野A「{description1}」と分野B「{description2}」の両方の知識がないと"
        "適切に答えられないような、1〜2文の自然な日本語の相談文を1件だけ作成してください。\n"
        "制約:\n"
        "- 選択肢（A. B. C. D. など）や解説・前置きは一切書かないこと。\n"
        "- 相談文の本文のみを1行で出力すること（改行を含めないこと）。\n"
        "- 四択問題の形式にしないこと。実際に困っている人が書くような自然な相談文にすること。\n"
    )


def _passes_filters(query: str, already_generated: set[str]) -> bool:
    """F1 length / F2 no four-choice markers / F3 not an exact duplicate / F4 single line."""
    stripped = query.strip()
    if not (_MIN_QUERY_LENGTH <= len(stripped) <= _MAX_QUERY_LENGTH):
        return False
    if any(marker in stripped for marker in _FOUR_CHOICE_MARKERS):
        return False
    if stripped in already_generated:
        return False
    if "\n" in stripped:
        return False
    return True


async def _generate_one(
    ollama_client: OllamaClient,
    model: str,
    domain1: str,
    domain2: str,
    already_generated: set[str],
    temperature: float = _GENERATION_TEMPERATURE,
    max_attempts: int = _MAX_GENERATION_ATTEMPTS_PER_SLOT,
) -> str | None:
    """Generate one filtered consultation sentence for (domain1, domain2), or None if all attempts fail."""
    prompt = _build_prompt(domain1, domain2)
    for _attempt in range(max_attempts):
        raw = await ollama_client.generate(model, prompt, temperature=temperature)
        assert isinstance(raw, str)  # logprobs not requested, generate() returns str
        candidate = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if _passes_filters(candidate, already_generated):
            return candidate
    return None


def _rows_for_pair(domain1: str, domain2: str, per_pair: int, per_pair_legal: int) -> int:
    """Number of rows to generate for a domain pair (legal-involving pairs get more; see module docstring)."""
    if "legal" in (domain1, domain2):
        return per_pair_legal
    return per_pair


async def generate_all_rows(
    ollama_client: OllamaClient,
    model: str,
    domains: list[str],
    per_pair: int = _DEFAULT_ROWS_PER_PAIR,
    per_pair_legal: int = _DEFAULT_ROWS_PER_LEGAL_PAIR,
) -> list[dict]:
    """Generate synthetic two-domain rows for every domain pair; skipped slots are logged, not padded."""
    rows: list[dict] = []
    already_generated: set[str] = set()
    pairs = list(itertools.combinations(domains, 2))
    for domain1, domain2 in pairs:
        target_count = _rows_for_pair(domain1, domain2, per_pair, per_pair_legal)
        written = 0
        for _slot in range(target_count):
            query = await _generate_one(ollama_client, model, domain1, domain2, already_generated)
            if query is None:
                print(
                    f"[generate_multidomain_training_examples] WARNING: skipped a slot for "
                    f"({domain1}, {domain2}) after {_MAX_GENERATION_ATTEMPTS_PER_SLOT} attempts",
                    file=sys.stderr,
                )
                continue
            already_generated.add(query)
            written += 1
            row_id = f"synth-{domain1}-{domain2}-{written:03d}"
            rows.append({"id": row_id, "query": query, "domain": [domain1, domain2]})
    return rows


def _char_ngram_jaccard(a: str, b: str, n: int = _LEAK_AUDIT_NGRAM_SIZE) -> float:
    """Jaccard similarity of character n-gram sets (whitespace-free, so suited to Japanese text)."""
    ngrams_a = {a[i : i + n] for i in range(max(0, len(a) - n + 1))}
    ngrams_b = {b[i : i + n] for i in range(max(0, len(b) - n + 1))}
    if not ngrams_a and not ngrams_b:
        return 1.0
    union = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(union)


def audit_leak(generated_rows: list[dict], compound_questions: list[tuple[str, list[str]]]) -> dict:
    """A7: max/median near-duplicate similarity of generated rows against the eval set's 100 compound questions.

    This is a detector, not a selector (see module docstring): it never
    removes or edits generated_rows, only reports how similar the closest
    match is.
    """
    compound_texts = [text for text, _domains in compound_questions]
    per_row_max: list[float] = []
    for row in generated_rows:
        similarities = [_char_ngram_jaccard(row["query"], text) for text in compound_texts]
        per_row_max.append(max(similarities) if similarities else 0.0)
    per_row_max_sorted = sorted(per_row_max)
    n = len(per_row_max_sorted)
    median = (
        per_row_max_sorted[n // 2]
        if n % 2 == 1
        else (per_row_max_sorted[n // 2 - 1] + per_row_max_sorted[n // 2]) / 2
        if n > 0
        else 0.0
    )
    return {
        "n_generated_rows": len(generated_rows),
        "n_compound_questions": len(compound_texts),
        "max_jaccard": max(per_row_max) if per_row_max else 0.0,
        "median_max_jaccard": median,
    }


def _audit_leak_from_output_file(output_path: str) -> None:
    """CLI entry for --audit-leak: read the generated rows and report against build_dataset's compound set.

    Local import of build_dataset here (not at module top-level) so that the
    generation code path above never touches build_dataset at all -- only
    this explicit, separately-invoked audit path does (module docstring).
    """
    from build_dataset import _COMPOUND_QUESTIONS  # noqa: PLC0415 (see docstring)

    with open(output_path, encoding="utf-8") as f:
        generated_rows = [json.loads(line) for line in f if line.strip()]
    report = audit_leak(generated_rows, _COMPOUND_QUESTIONS)
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
    if report["max_jaccard"] >= _LEAK_AUDIT_JACCARD_THRESHOLD:
        raise AssertionError(
            f"A7 (leak audit) failed: max_jaccard={report['max_jaccard']:.4f} >= "
            f"{_LEAK_AUDIT_JACCARD_THRESHOLD} against build_dataset._COMPOUND_QUESTIONS"
        )
    print(
        f"[generate_multidomain_training_examples] A7 leak audit PASS "
        f"(max_jaccard={report['max_jaccard']:.4f} < {_LEAK_AUDIT_JACCARD_THRESHOLD})",
        file=sys.stderr,
    )


async def _generate_and_save(
    train_data_path: str,
    model: str,
    ollama_host: str,
    ollama_port: int,
    per_pair: int,
    per_pair_legal: int,
    output_path: str,
) -> None:
    """Load domain names, generate all rows, write them, and enforce the minimum row-count floor."""
    domains = _load_domain_names(train_data_path)
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    rows = await generate_all_rows(
        ollama_client, model, domains, per_pair=per_pair, per_pair_legal=per_pair_legal
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[generate_multidomain_training_examples] wrote {len(rows)} rows to {output_path} "
        f"(domains={domains})",
        file=sys.stderr,
    )
    if len(rows) < _MINIMUM_ACCEPTABLE_ROW_COUNT:
        raise AssertionError(
            f"generated only {len(rows)} rows, below the pre-registered floor of "
            f"{_MINIMUM_ACCEPTABLE_ROW_COUNT} (journal.md Iter60 plan); do not proceed to training"
        )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic two-domain consultation sentences via judge_model "
            "for scripts/train_multilabel_dispatch_head.py"
        )
    )
    parser.add_argument(
        "--train-data",
        default="data/classifier_train.jsonl",
        help="JSONL of {id, query, domain} rows, used only to derive the 10 domain names",
    )
    parser.add_argument("--model", help="Generation model; should match config.yaml's judge_model")
    parser.add_argument("--ollama-host", help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--per-pair", type=int, default=_DEFAULT_ROWS_PER_PAIR)
    parser.add_argument("--per-pair-legal", type=int, default=_DEFAULT_ROWS_PER_LEGAL_PAIR)
    parser.add_argument("--output", required=True, help="Path to write/read the generated JSONL")
    parser.add_argument(
        "--audit-leak",
        action="store_true",
        help="A7 mode: read --output and report near-duplicate similarity against "
        "build_dataset._COMPOUND_QUESTIONS instead of generating",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    if args.audit_leak:
        _audit_leak_from_output_file(args.output)
        return
    if not args.model or not args.ollama_host:
        raise SystemExit("--model and --ollama-host are required unless --audit-leak is set")
    asyncio.run(
        _generate_and_save(
            args.train_data,
            args.model,
            args.ollama_host,
            args.ollama_port,
            args.per_pair,
            args.per_pair_legal,
            args.output,
        )
    )


if __name__ == "__main__":
    main()

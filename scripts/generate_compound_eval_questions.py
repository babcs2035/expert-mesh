"""Iter78 (compound_eval_set_expansion=llm_generated_separate_generator): LLM
generation of natural-language Japanese consultation sentences spanning two
of the 10 mesh domains, to expand build_dataset.py's hand-authored
_COMPOUND_QUESTIONS evaluation tier from 100 to 415 rows (journal.md
Iteration 78 plan).

Why a separate script rather than reusing scripts/generate_multidomain_training_examples.py
(journal.md Iter78 調査 Q2, "循環的妥当性の回避"): that script's generation
model defaults to config.yaml's judge_model
(schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m), the same model used
both (a) as the training-signal generator for rank_2 multi-label data and
(b) as the LLM-as-judge in scripts/evaluate_response_quality.py. Using it
again here would make the *evaluation* set's compound questions come from
the same model that grades responses and that already appears in training
data -- a two-fold circularity. This script therefore (i) is written
independently, without importing generate_multidomain_training_examples,
using its own prompt wording, and (ii) requires --generator-model to differ
from --verifier-model (the verifier defaults to config.yaml's judge_model,
used only to independently check that a generated row actually needs both
intended domains -- never to generate rows itself).

LEAKAGE GUARD: this module never imports build_dataset at module scope.
The one place it reads build_dataset._COMPOUND_QUESTIONS is
_load_existing_compound_texts(), used only for near-duplicate filtering
against the existing 100 hand-authored rows (a post-hoc dedup guard, not a
selector that shapes what gets generated) -- the import there is local to
keep this generation path free of any coupling to build_dataset's contents.

Usage (module mode; must run against wafl-ctrl5 per config.yml's permanent
operating rule that single-GPU auxiliary LLM traffic never touches the
wafl500-509 experiment nodes):
    uv run python -m scripts.generate_compound_eval_questions \\
        --generator-model qwen3.5:9b \\
        --verifier-model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \\
        --ollama-host 127.0.0.1 --ollama-port 11499 \\
        --per-pair 12 --target-per-pair 7 \\
        --output data/compound_questions_generated.jsonl
"""

import argparse
import asyncio
import itertools
import json
import sys
from datetime import datetime, timezone

from expert_backend import OllamaClient

# Generation-quality filters (mirrors the acceptance filters used for the
# existing 100 hand-authored rows: natural consultation prose, not a JMMLU-
# style four-choice question).
_MIN_QUERY_LENGTH = 20
_MAX_QUERY_LENGTH = 200
_FOUR_CHOICE_MARKERS = ("A.", "B.", "C.", "D.", "Ａ．", "Ｂ．", "Ｃ．", "Ｄ．")
_MAX_GENERATION_ATTEMPTS_PER_SLOT = 4
_GENERATION_TEMPERATURE = 0.9

_DEFAULT_ROWS_PER_PAIR = 12
_DEFAULT_TARGET_PER_PAIR = 7
_NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.6
_NGRAM_SIZE = 3

# Domain names that would give away the intended pair verbatim; a row that
# names its own target domains explicitly is testing keyword matching, not
# genuine cross-domain routing ability (journal.md Iter78 plan, filter (d)).
_DOMAIN_LABEL_WORDS_JA: dict[str, tuple[str, ...]] = {
    "business_economics": ("経営学", "経済学"),
    "computer_science": ("情報科学", "コンピュータサイエンス"),
    "education": ("教育学",),
    "general": ("一般分野",),
    "history_culture": ("歴史学", "文化人類学"),
    "legal": ("法律学",),
    "mathematics": ("数学分野",),
    "medical": ("医学分野",),
    "natural_science": ("自然科学",),
    "social_science": ("社会科学",),
}

# Domain descriptions authored independently of
# generate_multidomain_training_examples._DOMAIN_DESCRIPTIONS_JA (different
# wording, same intent: give the generator model enough context per domain
# without reusing that script's exact strings).
_DOMAIN_PROMPT_HINTS_JA: dict[str, str] = {
    "business_economics": "会社経営、マーケティング、会計処理、資金調達、市場動向の分析など",
    "computer_science": "ソフトウェア開発、データベース設計、ネットワーク構築、情報セキュリティなど",
    "education": "学校教育の制度・運営、授業や学習指導、生徒指導など",
    "general": "特定分野に限定されない、暮らしの中の一般的な困りごと",
    "history_culture": "歴史的な出来事や史料、伝統文化、芸術作品の背景など",
    "legal": "契約や労働紛争、民事・刑事手続き、各種法令の適用など",
    "mathematics": "数式処理、統計分析、確率計算、幾何学的な問題解決など",
    "medical": "病気の診断や治療方針、健康管理、公衆衛生上の対応など",
    "natural_science": "物理現象、化学反応、生物学的なしくみ、地学的な事象など",
    "social_science": "社会構造の分析、心理的な要因、政治・制度に関する論点など",
}


def _load_domain_names(train_data_path: str) -> list[str]:
    """Derive the domain name set from classifier_train.jsonl's own `domain` column."""
    domains: set[str] = set()
    with open(train_data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                domains.add(json.loads(line)["domain"])
    return sorted(domains)


def _build_generation_prompt(domain1: str, domain2: str) -> str:
    """Prompt for the generator model; independent wording from generate_multidomain_training_examples._build_prompt."""
    hint1 = _DOMAIN_PROMPT_HINTS_JA[domain1]
    hint2 = _DOMAIN_PROMPT_HINTS_JA[domain2]
    return (
        "あなたは実際に困っている相談者になりきって、窓口に投げかける質問文を作る役目です。\n"
        f"次の二つの領域「{hint1}」と「{hint2}」の両方に踏み込まないと、"
        "的確な回答ができないような具体的な悩みごとを、1〜2文の日本語で書いてください。\n"
        "守ってほしいこと:\n"
        "- 出力は相談文そのものだけとし、見出しや選択肢、解説は付けないこと。\n"
        "- 改行を含まない一続きの文章にすること。\n"
        "- 四択問題のような形式にはしないこと。\n"
        "- 分野名や専門用語のラベルをそのまま書かず、具体的な状況描写にすること。\n"
    )


def _build_verification_prompt(query: str, domains: list[str]) -> str:
    """Prompt asking the (separate) verifier model to name the 2 needed domains, unaided by the intended pair."""
    numbered = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(domains))
    return (
        "次の相談内容に的確に答えるためには、以下の10個の専門分野のうち、"
        "どの2つの知識が必要ですか。\n"
        f"相談内容: {query}\n\n"
        f"専門分野一覧:\n{numbered}\n\n"
        "回答は、必要な2つの分野名だけをカンマ区切りで1行で出力してください"
        "（例: legal,medical）。それ以外の文章は書かないでください。"
    )


def _passes_local_filters(query: str, domain1: str, domain2: str, already_generated: set[str]) -> bool:
    """F1 length / F2 no four-choice markers / F3 not an exact duplicate / F4 single line / F5 no label words."""
    stripped = query.strip()
    if not (_MIN_QUERY_LENGTH <= len(stripped) <= _MAX_QUERY_LENGTH):
        return False
    if any(marker in stripped for marker in _FOUR_CHOICE_MARKERS):
        return False
    if stripped in already_generated:
        return False
    if "\n" in stripped:
        return False
    for domain in (domain1, domain2):
        if any(label in stripped for label in _DOMAIN_LABEL_WORDS_JA[domain]):
            return False
    return True


def _char_ngram_jaccard(a: str, b: str, n: int = _NGRAM_SIZE) -> float:
    """Jaccard similarity of character n-gram sets (whitespace-free, suited to Japanese text)."""
    ngrams_a = {a[i : i + n] for i in range(max(0, len(a) - n + 1))}
    ngrams_b = {b[i : i + n] for i in range(max(0, len(b) - n + 1))}
    if not ngrams_a and not ngrams_b:
        return 1.0
    union = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(union)


def _is_near_duplicate(query: str, reference_texts: list[str]) -> bool:
    """True if query's max 3-gram Jaccard similarity against any reference text meets the near-dup threshold."""
    return any(
        _char_ngram_jaccard(query, reference) >= _NEAR_DUPLICATE_JACCARD_THRESHOLD
        for reference in reference_texts
    )


def _load_existing_compound_texts() -> list[str]:
    """Local import of build_dataset (see module docstring's leakage guard) to fetch the 100 existing compound texts."""
    from build_dataset import _COMPOUND_QUESTIONS  # noqa: PLC0415 (see docstring)

    return [text for text, _domains in _COMPOUND_QUESTIONS]


def _load_reference_texts_from_jsonl_files(paths: list[str]) -> list[str]:
    """Collect `query` fields from a list of JSONL files that may or may not exist."""
    texts: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        texts.append(json.loads(line)["query"])
        except FileNotFoundError:
            continue
    return texts


def _parse_verifier_domains(raw: str, valid_domains: list[str]) -> list[str] | None:
    """Parse the verifier model's comma-separated domain-name answer; None if malformed."""
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    candidates = [token.strip() for token in first_line.split(",")]
    matched = [name for name in candidates if name in valid_domains]
    if len(matched) != 2:
        return None
    return matched


async def _generate_one(
    ollama_client: OllamaClient,
    generator_model: str,
    domain1: str,
    domain2: str,
    already_generated: set[str],
    near_duplicate_reference_texts: list[str],
    max_attempts: int = _MAX_GENERATION_ATTEMPTS_PER_SLOT,
) -> str | None:
    """Generate one filtered, non-near-duplicate consultation sentence for (domain1, domain2), or None."""
    prompt = _build_generation_prompt(domain1, domain2)
    for _attempt in range(max_attempts):
        raw = await ollama_client.generate(generator_model, prompt, temperature=_GENERATION_TEMPERATURE)
        assert isinstance(raw, str)  # logprobs not requested, generate() returns str
        candidate = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if not _passes_local_filters(candidate, domain1, domain2, already_generated):
            continue
        if _is_near_duplicate(candidate, near_duplicate_reference_texts):
            continue
        return candidate
    return None


async def _verify_one(
    ollama_client: OllamaClient,
    verifier_model: str,
    query: str,
    domain1: str,
    domain2: str,
    all_domains: list[str],
) -> bool:
    """Independent verifier (config.yaml's judge_model by default): both intended domains named, order-unaided."""
    prompt = _build_verification_prompt(query, all_domains)
    raw = await ollama_client.generate(verifier_model, prompt, temperature=0.0)
    assert isinstance(raw, str)
    parsed = _parse_verifier_domains(raw, all_domains)
    if parsed is None:
        return False
    return {domain1, domain2} == set(parsed)


async def generate_all_rows(
    ollama_client: OllamaClient,
    generator_model: str,
    verifier_model: str,
    domains: list[str],
    per_pair: int,
    target_per_pair: int,
    near_duplicate_reference_texts: list[str],
) -> tuple[list[dict], dict]:
    """Generate, verify, and trim rows for every domain pair; returns (rows, stats)."""
    rows: list[dict] = []
    already_generated: set[str] = set()
    pairs = list(itertools.combinations(domains, 2))
    generated_at = datetime.now(timezone.utc).isoformat()
    stats = {"pairs": len(pairs), "generated_attempts": 0, "verified_accepted": 0, "shortfall_pairs": []}
    for domain1, domain2 in pairs:
        accepted_for_pair: list[dict] = []
        for _slot in range(per_pair):
            query = await _generate_one(
                ollama_client, generator_model, domain1, domain2, already_generated, near_duplicate_reference_texts
            )
            stats["generated_attempts"] += 1
            if query is None:
                continue
            already_generated.add(query)
            ok = await _verify_one(ollama_client, verifier_model, query, domain1, domain2, domains)
            if not ok:
                continue
            accepted_for_pair.append(
                {
                    "query": query,
                    "expected_domains": sorted([domain1, domain2]),
                    "pair": f"{domain1}+{domain2}",
                    "generator_model": generator_model,
                    "generated_at": generated_at,
                }
            )
            if len(accepted_for_pair) >= target_per_pair:
                break
        stats["verified_accepted"] += len(accepted_for_pair)
        if len(accepted_for_pair) < target_per_pair:
            stats["shortfall_pairs"].append(
                {"pair": f"{domain1}+{domain2}", "accepted": len(accepted_for_pair), "target": target_per_pair}
            )
        rows.extend(accepted_for_pair[:target_per_pair])
    return rows, stats


async def _generate_and_save(
    train_data_path: str,
    generator_model: str,
    verifier_model: str,
    ollama_host: str,
    ollama_port: int,
    per_pair: int,
    target_per_pair: int,
    output_path: str,
    multidomain_glob_paths: list[str],
) -> None:
    """Load domain names, generate+verify+trim all rows, write them out."""
    if generator_model == verifier_model:
        raise SystemExit(
            "--generator-model must differ from --verifier-model (circularity guard, module docstring)"
        )
    domains = _load_domain_names(train_data_path)
    near_duplicate_reference_texts = (
        _load_existing_compound_texts()
        + _load_reference_texts_from_jsonl_files([train_data_path, *multidomain_glob_paths])
    )
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    rows, stats = await generate_all_rows(
        ollama_client,
        generator_model,
        verifier_model,
        domains,
        per_pair=per_pair,
        target_per_pair=target_per_pair,
        near_duplicate_reference_texts=near_duplicate_reference_texts,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[generate_compound_eval_questions] wrote {len(rows)} rows to {output_path}; "
        f"stats={json.dumps(stats, ensure_ascii=False)}",
        file=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate+verify 2-domain Japanese consultation sentences to expand "
            "build_dataset.py's evaluation-set compound tier (Iter78)."
        )
    )
    parser.add_argument(
        "--train-data",
        default="data/classifier_train.jsonl",
        help="JSONL of {id, query, domain} rows, used only to derive the 10 domain names and for dedup",
    )
    parser.add_argument("--generator-model", required=True, help="Generation model; must differ from --verifier-model")
    parser.add_argument(
        "--verifier-model",
        default="schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m",
        help="Independent verifier model (defaults to config.yaml's judge_model)",
    )
    parser.add_argument("--ollama-host", required=True, help="wafl-ctrl5's ollama daemon host/IP (config.yml B)")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--per-pair", type=int, default=_DEFAULT_ROWS_PER_PAIR, help="Generation attempts per pair")
    parser.add_argument(
        "--target-per-pair", type=int, default=_DEFAULT_TARGET_PER_PAIR, help="Accepted rows kept per pair"
    )
    parser.add_argument("--output", required=True, help="Path to write the generated+verified JSONL")
    parser.add_argument(
        "--multidomain-dedup-file",
        action="append",
        default=[],
        help="Additional JSONL file(s) to dedup against (e.g. data/classifier_train_multidomain*.jsonl); repeatable",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    asyncio.run(
        _generate_and_save(
            args.train_data,
            args.generator_model,
            args.verifier_model,
            args.ollama_host,
            args.ollama_port,
            args.per_pair,
            args.target_per_pair,
            args.output,
            args.multidomain_dedup_file,
        )
    )


if __name__ == "__main__":
    main()

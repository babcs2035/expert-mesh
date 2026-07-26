"""Build the evaluation dataset (design doc 4.3) from JMMLU, fixed at 10 domains.

JMMLU (https://huggingface.co/datasets/nlp-waseda/JMMLU, commit
3637b25e444ccfdcde4d23a783cbe8e674faa01b) is a 56-task, 7,536-question
Japanese four-choice benchmark. This module maps its 56 tasks onto the 10
mesh domains below and samples up to `--domain-target-size` questions per
domain, so the same underlying question pool can support both the routing
(axis 1) and, via the `jmmlu_answer` field, answer-quality (axis 2) metrics.

The dataset is fixed at 10 domains (medical, legal, education,
business_economics, computer_science, natural_science, mathematics,
history_culture, social_science, general); there is no 4-domain mode.

Known mapping limitations (see docs/d0001_literature_survey_2026-07.md and
plans/p0001_research_direction_2026-07.md for the underlying research
rationale):

- `legal` has only 2 constituent tasks (international_law, jurisprudence;
  227 questions total) because JMMLU has no `professional_law` task (unlike
  the English MMLU it derives from). This is a hard ceiling: legal cannot
  reach the 150-question target of the other domains without duplication,
  so it is capped at its actual pool size.
- `education` has no directly corresponding JMMLU task; sociology,
  high_school_psychology, and moral_disputes (448 questions) are used as a
  proxy for the mesh's actual education-administration domain. This is a
  deliberate compromise, not a claim that these tasks measure the same
  thing as the hand-authored education questions used for compound rows.

Licensing: the entire JMMLU dataset is CC BY-NC-ND 4.0 (non-commercial,
no-derivatives; research/evaluation use is explicitly permitted). Five
tasks (japanese_history, world_history, japanese_idiom, japanese_civics,
japanese_geography) additionally carry a named-copyright-holder clause
that separately confirms research/evaluation use is allowed. All five fall
under `history_culture` in this mapping. `--exclude-restricted-license-tasks`
is provided so a future non-research redistribution of this dataset can
opt out of them without needing a code change; it is off by default since
this project's use (routing research) already qualifies as permitted use.

Usage:
    uv run python build_dataset.py --output data/dataset.jsonl
    uv run python build_dataset.py --output data/dataset.jsonl --jmmlu-zip /path/to/JMMLU.zip
    uv run python build_dataset.py --output data/dataset.jsonl --exclude-restricted-license-tasks
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

# Pinned to a specific commit so the sampled dataset does not silently
# change if upstream revises or re-translates questions after this was written.
_JMMLU_ZIP_SHA = "3637b25e444ccfdcde4d23a783cbe8e674faa01b"
_JMMLU_ZIP_URL = (
    f"https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/{_JMMLU_ZIP_SHA}/JMMLU.zip"
)
_JMMLU_DOWNLOAD_TIMEOUT_S = 60.0
_JMMLU_CSV_PATH_TEMPLATE = "JMMLU/test/{task_name}.csv"

# Target question count per domain. legal's actual pool (227) is below this;
# _sample_domain_questions caps at the pool size rather than erroring.
_DOMAIN_TARGET_SIZE = 150
# Fixed seed so the same JMMLU.zip always yields the same sampled dataset.
_JMMLU_SAMPLE_SEED = 20260726
# Distinct seed for E6's classifier training pool (build_classifier_training_rows),
# so its sample is independent of (and, via exclude_queries, disjoint from) the
# evaluation dataset's sample — see that function's docstring.
_CLASSIFIER_TRAIN_SAMPLE_SEED = 20260727

# Task -> domain mapping. Every one of JMMLU's 56 tasks appears in exactly
# one domain's list (verified against the task_list in JMMLU.py at the
# pinned commit); see the module docstring for the rationale behind the
# legal/education assignments specifically.
_DOMAIN_TASK_MAP: dict[str, list[str]] = {
    "medical": [
        "virology",
        "nutrition",
        "human_sexuality",
        "clinical_knowledge",
        "human_aging",
        "anatomy",
        "professional_psychology",
        "college_medicine",
        "professional_medicine",
        "medical_genetics",
    ],
    "legal": [
        "international_law",
        "jurisprudence",
    ],
    "education": [
        "sociology",
        "high_school_psychology",
        "moral_disputes",
    ],
    "business_economics": [
        "econometrics",
        "high_school_microeconomics",
        "business_ethics",
        "marketing",
        "high_school_macroeconomics",
        "management",
        "public_relations",
        "professional_accounting",
    ],
    "computer_science": [
        "computer_security",
        "machine_learning",
        "high_school_computer_science",
        "college_computer_science",
        "electrical_engineering",
    ],
    "natural_science": [
        "high_school_chemistry",
        "high_school_physics",
        "college_physics",
        "conceptual_physics",
        "college_biology",
        "high_school_biology",
        "college_chemistry",
        "astronomy",
    ],
    "mathematics": [
        "college_mathematics",
        "high_school_statistics",
        "elementary_mathematics",
        "high_school_mathematics",
        "abstract_algebra",
    ],
    "history_culture": [
        "japanese_history",
        "japanese_civics",
        "high_school_european_history",
        "prehistory",
        "japanese_idiom",
        "japanese_geography",
        "high_school_geography",
        "world_history",
    ],
    "social_science": [
        "security_studies",
        "world_religions",
        "philosophy",
        "global_facts",
    ],
    "general": [
        "miscellaneous",
        "logical_fallacies",
        "formal_logic",
    ],
}

_RESTRICTED_LICENSE_TASKS: frozenset[str] = frozenset(
    {
        "japanese_history",
        "world_history",
        "japanese_idiom",
        "japanese_civics",
        "japanese_geography",
    }
)

# Hand-authored compound-domain questions (design doc 4.3: "questions
# spanning multiple domains"). JMMLU's four-choice questions each belong to
# a single task and cannot express genuine cross-domain ambiguity, so these
# remain hand-authored rather than JMMLU-derived.
_COMPOUND_QUESTIONS: list[tuple[str, list[str]]] = [
    (
        "仕事中に転倒して怪我をしました．治療費と休業補償について知りたいです．",
        ["medical", "legal"],
    ),
    ("交通事故で怪我をして通院していますが，慰謝料の相場が分かりません．", ["medical", "legal"]),
    ("職場のハラスメントでうつ状態になり，休職を検討しています．", ["medical", "legal"]),
    (
        "ペットが近隣トラブルの原因で怪我をさせてしまいました．治療費と責任について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "学校で子供のアレルギー対応について，給食と保健室の両方の配慮が必要です．",
        ["education", "medical"],
    ),
    ("いじめの問題で，学校への対応と法的なアドバイスが必要です．", ["education", "legal"]),
    (
        "交通事故で後遺障害が残り，後遺障害等級認定の手続きと今後の通院方針の両方について相談したいです．",
        ["medical", "legal"],
    ),
    (
        "職場の化学物質にばく露して体調を崩しました．労災認定と治療方針を教えてください．",
        ["medical", "legal"],
    ),
    (
        "医療事故に遭った可能性があります．診療記録の開示請求と今後の治療についてどう進めればよいですか．",
        ["medical", "legal"],
    ),
    (
        "高齢の親が施設で転倒し骨折しました．施設側の責任追及と治療の両方を検討しています．",
        ["medical", "legal"],
    ),
    (
        "スポーツ中の事故で相手にケガを負わせてしまいました．治療費の負担と損害賠償請求への対応を知りたいです．",
        ["medical", "legal"],
    ),
    (
        "感染症にかかった従業員がいる職場で，就業制限の法的根拠と医学的な対応基準を知りたいです．",
        ["medical", "legal"],
    ),
    (
        "美容医療の施術後に合併症が出ました．治療方針の相談と施術業者への責任追及を同時に進めたいです．",
        ["medical", "legal"],
    ),
    (
        "ペットに噛まれてケガをしました．治療費の請求先と飼い主の法的責任について知りたいです．",
        ["medical", "legal"],
    ),
    (
        "学校での部活動中の熱中症で生徒が搬送されました．今後の予防策と応急対応の指導について知りたいです．",
        ["education", "medical"],
    ),
    (
        "発達障害のある生徒への服薬管理について，学校と医療機関の連携方法を教えてください．",
        ["education", "medical"],
    ),
    (
        "給食のアレルギー事故が発生しました．再発防止策と当日の医学的対応の両方を検証したいです．",
        ["education", "medical"],
    ),
    (
        "校内で発生した器物損壊について，生徒への指導と保護者への損害賠償請求の両方を検討しています．",
        ["education", "legal"],
    ),
    (
        "学校事故で生徒がケガをした場合の学校の法的責任と，学校側の説明責任について知りたいです．",
        ["education", "legal"],
    ),
    (
        "私立学校の退学処分に対して，処分の妥当性と法的な異議申立て手続きを知りたいです．",
        ["education", "legal"],
    ),
]


def _load_jmmlu_zip_bytes(jmmlu_zip_path: str | None) -> bytes:
    """Return the JMMLU.zip contents, from a local path if given or by download."""
    if jmmlu_zip_path is not None:
        with open(jmmlu_zip_path, "rb") as f:
            return f.read()
    response = httpx.get(_JMMLU_ZIP_URL, timeout=_JMMLU_DOWNLOAD_TIMEOUT_S, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _parse_jmmlu_task_csv(zf: zipfile.ZipFile, task_name: str) -> list[dict[str, str]]:
    """Parse one JMMLU task's CSV into rows of {question, A, B, C, D, answer}."""
    raw_bytes = zf.read(_JMMLU_CSV_PATH_TEMPLATE.format(task_name=task_name))
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return [{key: value.strip() for key, value in row.items()} for row in reader]


def _format_jmmlu_query(row: dict[str, str]) -> str:
    """Format a JMMLU row as a four-choice question prompt."""
    return f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"


def _sample_domain_questions(
    zf: zipfile.ZipFile,
    task_names: list[str],
    target_size: int,
    seed: int,
    exclude_tasks: frozenset[str],
    exclude_queries: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """Sample up to target_size (query, answer, task_name) tuples for one domain's tasks.

    Pools all constituent tasks' rows together before sampling, so the
    domain's questions are not required to be evenly split across tasks.
    Caps at the pool size rather than raising when a domain's tasks
    together hold fewer than target_size questions (true for legal).
    exclude_queries removes specific questions from the pool before
    sampling (used by build_classifier_training_rows to guarantee its
    output never overlaps the evaluation dataset's questions).
    """
    pool: list[tuple[str, str, str]] = []
    for task_name in task_names:
        if task_name in exclude_tasks:
            continue
        for row in _parse_jmmlu_task_csv(zf, task_name):
            query = _format_jmmlu_query(row)
            if query in exclude_queries:
                continue
            pool.append((query, row["answer"], task_name))
    sample_size = min(target_size, len(pool))
    return random.Random(seed).sample(pool, sample_size)


def _build_jmmlu_backed_groups(
    zf: zipfile.ZipFile,
    domain_target_size: int,
    exclude_restricted: bool,
    domain_task_map: dict[str, list[str]],
    seed: int = _JMMLU_SAMPLE_SEED,
    exclude_queries: frozenset[str] = frozenset(),
) -> dict[str, list[tuple[str, str, str]]]:
    """Sample every domain's questions from its mapped JMMLU tasks."""
    exclude_tasks = _RESTRICTED_LICENSE_TASKS if exclude_restricted else frozenset()
    return {
        domain: _sample_domain_questions(
            zf, task_names, domain_target_size, seed, exclude_tasks, exclude_queries
        )
        for domain, task_names in domain_task_map.items()
    }


def _build_rows(
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]],
) -> list[dict]:
    """Assemble JMMLU-derived single-domain rows and hand-authored compound rows."""
    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        domain_groups = _build_jmmlu_backed_groups(
            zf, domain_target_size, exclude_restricted_license_tasks, domain_task_map
        )

    rows = []
    for domain in sorted(domain_groups):
        for index, (query, answer, task_name) in enumerate(domain_groups[domain], start=1):
            rows.append(
                {
                    "id": f"{domain}-{index:03d}",
                    "query": query,
                    "expected_domains": [domain],
                    "is_compound": False,
                    "jmmlu_task": task_name,
                    "jmmlu_answer": answer,
                }
            )
    for index, (query, expected_domains) in enumerate(_COMPOUND_QUESTIONS, start=1):
        rows.append(
            {
                "id": f"compound-{index:03d}",
                "query": query,
                "expected_domains": expected_domains,
                "is_compound": True,
            }
        )
    return rows


def write_dataset(
    output: TextIO,
    jmmlu_zip_path: str | None = None,
    domain_target_size: int = _DOMAIN_TARGET_SIZE,
    exclude_restricted_license_tasks: bool = False,
    domain_task_map: dict[str, list[str]] | None = None,
) -> int:
    """Write all dataset rows as JSON Lines to the given stream; return the row count.

    domain_task_map overrides the module-level _DOMAIN_TASK_MAP; tests use
    this to point at a fixture zip containing only one task per domain
    instead of all 56 real JMMLU tasks.
    """
    rows = _build_rows(
        jmmlu_zip_path,
        domain_target_size,
        exclude_restricted_license_tasks,
        domain_task_map if domain_task_map is not None else _DOMAIN_TASK_MAP,
    )
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_classifier_training_rows(
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]],
    eval_rows: list[dict],
) -> list[dict]:
    """Build E6 classifier training rows ({id, query, domain}), disjoint from eval_rows' questions.

    Guards against Iter10's label leakage (the training features there were
    derived from probe/dispatch results on the same 46 questions used for
    evaluation): eval_rows' single-domain questions are excluded from the
    sampling pool *before* sampling (not just tagged afterward), so overlap
    with the evaluation set is structurally impossible rather than merely
    avoided by convention. Uses _CLASSIFIER_TRAIN_SAMPLE_SEED (distinct from
    the eval set's seed) so the two samples are independent draws.

    Passing an explicit --jmmlu-zip (a locally cached JMMLU.zip) avoids
    downloading it a second time when both this and the eval dataset are
    generated in the same run.

    Known imbalance: since eval and training draw from the same
    task-limited pool without overlap, a domain whose pool is close to
    2x domain_target_size ends up with a noticeably smaller training set
    than the rest. At the default domain_target_size=150, legal's pool is
    227 (verified against the real JMMLU.zip): after 150 are reserved for
    eval, only 77 remain for training, versus 150 for every other domain.
    scripts/train_domain_classifier.py does not currently compensate for
    this (e.g. via class_weight), so the classifier may underperform on
    legal specifically for reasons unrelated to the signal itself.
    """
    eval_queries = frozenset(row["query"] for row in eval_rows if not row["is_compound"])
    zip_bytes = _load_jmmlu_zip_bytes(jmmlu_zip_path)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        domain_groups = _build_jmmlu_backed_groups(
            zf,
            domain_target_size,
            exclude_restricted_license_tasks,
            domain_task_map,
            seed=_CLASSIFIER_TRAIN_SAMPLE_SEED,
            exclude_queries=eval_queries,
        )

    rows = []
    for domain in sorted(domain_groups):
        for index, (query, _answer, _task_name) in enumerate(domain_groups[domain], start=1):
            rows.append({"id": f"{domain}-train-{index:03d}", "query": query, "domain": domain})
    return rows


def write_classifier_training_data(
    output: TextIO,
    jmmlu_zip_path: str | None,
    domain_target_size: int,
    exclude_restricted_license_tasks: bool,
    domain_task_map: dict[str, list[str]] | None,
    eval_rows: list[dict],
) -> int:
    """Write classifier training rows as JSON Lines; return the row count."""
    rows = build_classifier_training_rows(
        jmmlu_zip_path,
        domain_target_size,
        exclude_restricted_license_tasks,
        domain_task_map if domain_task_map is not None else _DOMAIN_TASK_MAP,
        eval_rows,
    )
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of path if needed (data/ and results/ are gitignored,
    so a clean checkout has neither until something creates them)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build the JMMLU-backed, 10-domain evaluation dataset as JSONL"
    )
    parser.add_argument("--output", default=None, help="Output file path; defaults to stdout")
    parser.add_argument(
        "--jmmlu-zip",
        default=None,
        help="Local path to a JMMLU.zip (skips downloading); mainly for tests/offline use",
    )
    parser.add_argument(
        "--domain-target-size",
        type=int,
        default=_DOMAIN_TARGET_SIZE,
        help="Max questions sampled per domain (capped at the domain's actual pool size)",
    )
    parser.add_argument(
        "--exclude-restricted-license-tasks",
        action="store_true",
        help="Exclude the 5 JMMLU tasks with named-copyright-holder clauses (all in history_culture)",
    )
    parser.add_argument(
        "--classifier-train-output",
        default=None,
        help="If set, also write E6 classifier training rows (disjoint from --output's questions) here",
    )
    args = parser.parse_args()

    eval_rows = _build_rows(
        args.jmmlu_zip,
        args.domain_target_size,
        args.exclude_restricted_license_tasks,
        _DOMAIN_TASK_MAP,
    )
    if args.output is None:
        for row in eval_rows:
            sys.stdout.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        _ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            for row in eval_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_dataset] wrote {len(eval_rows)} rows", file=sys.stderr)

    if args.classifier_train_output is not None:
        _ensure_parent_dir(args.classifier_train_output)
        with open(args.classifier_train_output, "w", encoding="utf-8") as f:
            classifier_count = write_classifier_training_data(
                f,
                args.jmmlu_zip,
                args.domain_target_size,
                args.exclude_restricted_license_tasks,
                _DOMAIN_TASK_MAP,
                eval_rows,
            )
        print(f"[build_dataset] wrote {classifier_count} classifier training rows", file=sys.stderr)


if __name__ == "__main__":
    main()

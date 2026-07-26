"""Design doc 4.1 axis 2 (answer quality) and axis 3 (end-to-end) metrics.

Axis 1 (routing accuracy) is metrics.py's job. This module fills the two
axes the design doc names but never specifies an implementation for
(docs/encounter_expert_mesh_design.md §4.1 gives only the indicator names,
no judge prompt/model/rubric — see each function's docstring for the
concrete design decision made here):

- JMMLU-sourced rows (build_dataset.py rows carrying a `jmmlu_answer`) have
  an objective ground-truth letter, so their quality is a straight
  extract-and-compare (compute_answer_quality_accuracy).
- Hand-authored consultation rows (no ground truth) fall back to
  LLM-as-judge (judge_response_quality), reusing the general node's
  expert_model as judge rather than standing up a dedicated judge model.
"""

import json
import re

from expert_backend import OllamaClient

_ANSWER_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

# Tried in order, most-explicit first: an LLM asked an open-ended question
# built from a 4-choice JMMLU prompt (build_dispatch_prompt does not
# constrain it to a single letter) tends to state its pick in one of these
# forms rather than reply with a bare letter. This is a best-effort
# heuristic, not a guaranteed-correct parse; its failure rate should be
# measured empirically before compute_answer_quality_accuracy's output is
# treated as ground truth.
_ANSWER_LETTER_PATTERNS = [
    re.compile(r"(?:正解|答え)は\s*[\(（]?([ABCD])[\)）]?"),
    re.compile(r"\*\*([ABCD])\*\*"),
    re.compile(r"[\(（]([ABCD])[\)）]"),
    re.compile(r"^\s*([ABCD])[.:)\s]", re.MULTILINE),
]

# 1-5 Likert rubric for LLM-as-judge (no ground truth available). A score
# at or above this is treated as a "quality pass" for compute_end_to_end_accuracy.
JUDGE_QUALITY_PASS_THRESHOLD = 3
_JUDGE_MAX_TOKENS = 50
_JUDGE_TEMPERATURE = 0.0


def extract_answer_letter(response_text: str) -> str | None:
    """Best-effort extraction of an A/B/C/D pick from a free-form expert response."""
    for pattern in _ANSWER_LETTER_PATTERNS:
        match = pattern.search(response_text)
        if match:
            return match.group(1)
    return None


def compute_answer_quality_accuracy(results: list[dict], dataset: list[dict]) -> float:
    """Fraction of JMMLU-sourced rows where the extracted answer letter matches jmmlu_answer.

    Only rows whose dataset entry carries a jmmlu_answer are gradable this
    way (hand-authored consultation rows have no ground truth; use
    judge_response_quality for those instead). Returns 0.0 when no row is
    gradable, matching metrics.py's zero-row conventions.
    """
    dataset_by_id = {row["id"]: row for row in dataset}
    gradable = [
        (result, dataset_by_id[result["id"]])
        for result in results
        if result["id"] in dataset_by_id and "jmmlu_answer" in dataset_by_id[result["id"]]
    ]
    if not gradable:
        return 0.0
    correct = sum(
        1
        for result, dataset_row in gradable
        if extract_answer_letter(result.get("answer_text", "")) == dataset_row["jmmlu_answer"]
    )
    return correct / len(gradable)


def build_llm_judge_prompt(query: str, expert_response: str) -> str:
    """1-5 rubric prompt for a hand-authored (no-ground-truth) consultation question.

    Not specified by the design doc; this rubric (accuracy, relevance,
    actionability) is a new design decision made for this implementation.
    """
    return (
        "次の相談内容と，専門家による回答を読み，回答の質を1〜5の5段階で評価してください．\n"
        "評価基準:\n"
        "5: 正確かつ相談内容との関連性が高く，実用的な助言を含む\n"
        "3: おおむね妥当だが，正確性か実用性のいずれかに不足がある\n"
        "1: 不正確，相談内容と無関係，または実用性がない\n\n"
        f"相談内容: {query}\n\n"
        f"専門家の回答: {expert_response}\n\n"
        '回答は{"score": <1から5の整数>}という形式の1行のJSONのみとし，'
        "他のキーや説明文は含めないでください．"
    )


def _parse_judge_score(raw_response: str) -> int | None:
    """Extract the 1-5 score; None on parse failure or an out-of-range value."""
    match = _ANSWER_JSON_PATTERN.search(raw_response)
    if match is None:
        return None
    try:
        score = int(json.loads(match.group())["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


async def judge_response_quality(
    ollama_client: OllamaClient,
    judge_model: str,
    query: str,
    expert_response: str,
    timeout_s: float,
) -> int | None:
    """LLM-as-judge 1-5 score for a hand-authored consultation question's expert response.

    Reuses the general node's expert_model as judge_model (config.yaml's
    judge_model key) rather than deploying a dedicated judge model, to
    avoid additional VRAM/infra. Returns None when the judge's own
    response doesn't parse to a valid score (treated as ungraded, not as
    a failing score, by callers).
    """
    raw_response = await ollama_client.generate(
        judge_model,
        build_llm_judge_prompt(query, expert_response),
        timeout_s=timeout_s,
        max_tokens=_JUDGE_MAX_TOKENS,
        temperature=_JUDGE_TEMPERATURE,
    )
    return _parse_judge_score(raw_response)


def compute_end_to_end_accuracy(
    routing_results: list[dict], quality_pass_by_id: dict[str, bool]
) -> float:
    """Fraction of rows correct on both axis 1 (routing) and axis 2 (quality) — axis 3.

    quality_pass_by_id maps a row's id to whether its answer passed
    quality grading (JMMLU: extracted letter matches; consultation: judge
    score >= JUDGE_QUALITY_PASS_THRESHOLD). A row absent from
    quality_pass_by_id (not yet graded) counts as failing axis 2 — the
    same "unknown counts as unverified" convention as
    metrics.py's other axis-1 functions apply to missing/None fields.
    """
    if not routing_results:
        return 0.0
    correct = sum(
        1
        for r in routing_results
        if r["selected_domain"] in r["expected_domains"] and quality_pass_by_id.get(r["id"], False)
    )
    return correct / len(routing_results)


def compute_latency_breakdown(results: list[dict]) -> dict:
    """Split mean wall-clock duration into expert generation time vs everything else.

    duration_ms (run_experiment.py, client-side wall clock) covers probing
    every node plus dispatch; dispatch_gen_time_ms (DispatchResponse.gen_time_ms,
    the expert node's own local generation time) is the one component of
    that total actually measured on the server side today. "everything_else"
    is not purely network transfer — it also includes every /probe call's
    own local inference time — so it is reported as a single residual
    bucket rather than mislabeled as "communication time" (design doc
    4.4's finer split would need per-probe timing added to http_client.py,
    which does not exist yet — see this module's docstring / README).
    Only rows with a successful dispatch (dispatch_gen_time_ms is not None)
    are included, since a fallback/failed row has no expert generation
    time to subtract.
    """
    dispatched = [r for r in results if r.get("dispatch_gen_time_ms") is not None]
    if not dispatched:
        return {
            "mean_duration_ms": 0.0,
            "mean_dispatch_gen_time_ms": 0.0,
            "mean_other_ms": 0.0,
            "rows_with_dispatch_timing": 0,
        }
    mean_duration_ms = sum(r["duration_ms"] for r in dispatched) / len(dispatched)
    mean_gen_time_ms = sum(r["dispatch_gen_time_ms"] for r in dispatched) / len(dispatched)
    return {
        "mean_duration_ms": mean_duration_ms,
        "mean_dispatch_gen_time_ms": mean_gen_time_ms,
        "mean_other_ms": mean_duration_ms - mean_gen_time_ms,
        "rows_with_dispatch_timing": len(dispatched),
    }

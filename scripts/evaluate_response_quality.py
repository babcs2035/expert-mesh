"""Post-process a results.jsonl into axis 2 (answer quality) and axis 3 (end-to-end) metrics.

Reads the same results.jsonl produced by run_experiment.py (which already
stores each row's answer_text and duration_ms/dispatch_gen_time_ms) plus
the dataset.jsonl it was run against (for jmmlu_answer ground truth), and
computes:
  - answer_quality_accuracy: JMMLU-sourced rows only (objective ground truth)
  - end_to_end_accuracy: routing correct AND answer quality passed
  - latency_breakdown: dispatch generation time vs residual wall-clock

Hand-authored consultation rows (no jmmlu_answer) are graded via
LLM-as-judge, requiring a live ollama node to reach judge_model.

Usage (module mode):
    uv run python -m scripts.evaluate_response_quality \\
        --results results/20260726_120000/results.jsonl \\
        --dataset data/dataset.jsonl \\
        --judge-model schroneko/llama-3.1-swallow-8b-instruct-v0.1:q4_k_m \\
        --ollama-host 192.168.15.100
"""

import argparse
import asyncio
import json
import sys

from evaluation import (
    JUDGE_QUALITY_PASS_THRESHOLD,
    compute_answer_quality_accuracy,
    compute_end_to_end_accuracy,
    compute_latency_breakdown,
    extract_answer_letter,
    judge_response_quality,
)
from expert_backend import OllamaClient
from node import load_yaml


def _read_jsonl(path: str) -> list[dict]:
    """Load JSON Lines rows from a file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _judge_consultation_rows(
    ollama_client: OllamaClient,
    judge_model: str,
    timeout_s: float,
    results: list[dict],
    dataset_by_id: dict[str, dict],
) -> dict[str, bool]:
    """LLM-judge every hand-authored (no jmmlu_answer) row; return {id: passed}.

    Sequential, matching run_experiment.py's own sequential LLM call
    pattern (avoids contending for the same node's CPU/GPU-bound inference).
    """
    quality_pass_by_id: dict[str, bool] = {}
    for result in results:
        dataset_row = dataset_by_id.get(result["id"])
        if dataset_row is None or "jmmlu_answer" in dataset_row:
            continue  # graded by compute_answer_quality_accuracy instead
        answer_text = result.get("answer_text")
        if not answer_text:
            continue
        score = await judge_response_quality(
            ollama_client, judge_model, result["query"], answer_text, timeout_s
        )
        if score is not None:
            quality_pass_by_id[result["id"]] = score >= JUDGE_QUALITY_PASS_THRESHOLD
    return quality_pass_by_id


async def _run(
    results_path: str,
    dataset_path: str,
    judge_model: str | None,
    ollama_host: str | None,
    ollama_port: int,
    timeout_s: float,
) -> dict:
    results = _read_jsonl(results_path)
    dataset = _read_jsonl(dataset_path)
    dataset_by_id = {row["id"]: row for row in dataset}

    quality_pass_by_id: dict[str, bool] = {}
    for result in results:
        dataset_row = dataset_by_id.get(result["id"])
        if dataset_row is not None and "jmmlu_answer" in dataset_row:
            quality_pass_by_id[result["id"]] = (
                extract_answer_letter(result.get("answer_text", "") or "")
                == dataset_row["jmmlu_answer"]
            )

    if judge_model is not None and ollama_host is not None:
        ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
        quality_pass_by_id.update(
            await _judge_consultation_rows(
                ollama_client, judge_model, timeout_s, results, dataset_by_id
            )
        )

    return {
        "answer_quality_accuracy": compute_answer_quality_accuracy(results, dataset),
        "end_to_end_accuracy": compute_end_to_end_accuracy(results, quality_pass_by_id),
        "latency_breakdown": compute_latency_breakdown(results),
        "graded_row_count": len(quality_pass_by_id),
        "total_row_count": len(results),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute axis 2/3 metrics (answer quality, end-to-end) from a results.jsonl"
    )
    parser.add_argument("--results", required=True, help="results.jsonl from run_experiment.py")
    parser.add_argument("--dataset", required=True, help="dataset.jsonl from build_dataset.py")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Read config.yaml's judge_model as the --judge-model default (ignored if --judge-model is set)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model to use for LLM-as-judge on non-JMMLU rows; defaults to config.yaml's judge_model; "
        "omit both to skip judging (JMMLU rows only)",
    )
    parser.add_argument("--ollama-host", default=None, help="Required if --judge-model is set")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    judge_model = args.judge_model
    if judge_model is None:
        judge_model = load_yaml(args.config).get("judge_model")

    outcome = asyncio.run(
        _run(
            args.results,
            args.dataset,
            judge_model,
            args.ollama_host,
            args.ollama_port,
            args.timeout_s,
        )
    )
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    if judge_model is None:
        print(
            "[evaluate_response_quality] --judge-model が未指定のため，"
            "手作りの相談設問（jmmlu_answerを持たない行）は end_to_end_accuracy 上で未採点扱い（不合格）です．",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

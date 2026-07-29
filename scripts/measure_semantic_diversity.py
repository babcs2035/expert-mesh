"""E4 prerequisite diagnostic: measure semantic diversity before running self_consistency_semantic.

Iter11 sampled at temperature=0.1 with N=3 and concluded multi_sample
"has no effect" — but at that temperature the samples were near-identical,
so there was no diversity for the method to act on in the first place; the
null result reflected a broken experiment, not a failed method (p0001 F2).

This script measures unique-cluster diversity and semantic entropy at
temperature=0.7, N=5 against a live ollama node so that mistake isn't
repeated for E4. It runs estimate_confidence_semantic_entropy() on a
sample of questions from data/dataset.jsonl and reports aggregate
statistics.

Usage:
    uv run python -m scripts.measure_semantic_diversity \\
        --ollama-host 192.168.15.100 --light-model qwen3.5:4b-q4_K_M \\
        --dataset data/dataset.jsonl --sample 20
"""

import argparse
import asyncio
import json
import math
from pathlib import Path

from expert_backend import OllamaClient
from router import estimate_confidence_semantic_entropy


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


async def measure_semantic_diversity(
    ollama_client: OllamaClient,
    light_model: str,
    timeout_s: float,
    dataset_path: str,
    sample_size: int,
) -> dict:
    """Run semantic entropy estimation on a sample of dataset questions.

    Returns a dict with per-question cluster counts and entropy values,
    plus aggregate statistics.
    """
    dataset = [json.loads(line) for line in Path(dataset_path).read_text().splitlines()]
    questions = dataset[:sample_size]

    cluster_counts: list[int] = []
    entropies: list[float] = []

    for q in questions:
        domain = q["expected_domains"][0]
        query = q["query"]
        confidence, entropy = await estimate_confidence_semantic_entropy(
            ollama_client,
            light_model,
            domain,
            query,
            timeout_s=timeout_s,
        )
        # entropy = 0 bits -> 1 cluster (full agreement)
        # entropy = log2(N) bits -> N clusters (all different)
        max_entropy = math.log2(5) if 5 > 1 else 0.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        # Estimate cluster count from normalized entropy:
        # 0 -> 1 cluster, 1 -> 5 clusters (linear interpolation is a rough proxy)
        est_clusters = max(1, round(normalized * 4) + 1)
        # Clamp to valid range [1, 5]
        est_clusters = min(5, max(1, est_clusters))

        cluster_counts.append(est_clusters)
        entropies.append(entropy)

    return {
        "n_questions": len(questions),
        "cluster_counts": cluster_counts,
        "entropies": entropies,
        "mean_cluster_count": _mean(cluster_counts),
        "std_cluster_count": _std(cluster_counts),
        "mean_entropy": _mean(entropies),
        "std_entropy": _std(entropies),
    }


async def _run(
    ollama_host: str,
    ollama_port: int,
    light_model: str,
    timeout_s: float,
    dataset_path: str,
    sample_size: int,
) -> None:
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    result = await measure_semantic_diversity(
        ollama_client, light_model, timeout_s, dataset_path, sample_size
    )

    print(f"[measure_semantic_diversity] n_questions={result['n_questions']}")
    print(f"  mean cluster count : {result['mean_cluster_count']:.2f} +/- {result['std_cluster_count']:.2f}")
    print(f"  mean entropy       : {result['mean_entropy']:.3f} +/- {result['std_entropy']:.3f} bits")

    # Per-question detail
    for i, (cc, ent) in enumerate(zip(result["cluster_counts"], result["entropies"])):
        print(f"  q{i+1:02d}: clusters={cc}, entropy={ent:.3f} bits")

    # Pass/fail judgment
    n_pass_cluster = sum(1 for cc in result["cluster_counts"] if cc >= 2)
    n_pass_entropy = sum(1 for ent in result["entropies"] if ent > 0.5)
    both_pass = sum(
        1 for cc, ent in zip(result["cluster_counts"], result["entropies"])
        if cc >= 2 and ent > 0.5
    )

    print()
    print(f"  cluster_count >= 2 : {n_pass_cluster}/{result['n_questions']}")
    print(f"  entropy > 0.5 bits : {n_pass_entropy}/{result['n_questions']}")
    print(f"  both conditions     : {both_pass}/{result['n_questions']}")

    if both_pass < result["n_questions"]:
        print()
        print("警告: 全問で多様性条件（cluster>=2 かつ entropy>0.5 bits）を満たしません．")
        print("      temperature を上げるか N を増やす検討が必要です．")
        print("      (Iter11 の失敗パターン再現に注意)")
    else:
        print()
        print("OK: 全問で多様性条件を満たしました．E4 着手の前提条件を満たします．")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure semantic diversity (cluster count + entropy) before E4"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--light-model", required=True, help="Model name for verdict sampling")
    parser.add_argument(
        "--dataset",
        default="data/dataset.jsonl",
        help="Path to the JMMLU dataset JSONL file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of questions to sample (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-question timeout in seconds (default: 300, allows ~180 LLM calls)",
    )
    args = parser.parse_args()

    asyncio.run(
        _run(
            args.ollama_host,
            args.ollama_port,
            args.light_model,
            args.timeout,
            args.dataset,
            args.sample,
        )
    )


if __name__ == "__main__":
    main()

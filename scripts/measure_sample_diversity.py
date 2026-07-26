"""E4 prerequisite diagnostic: measure sampling diversity before running self_consistency_semantic.

Iter11 sampled at temperature=0.1 with N=3 and concluded multi_sample
"has no effect" — but at that temperature the samples were near-identical,
so there was no diversity for the method to act on in the first place; the
null result reflected a broken experiment, not a failed method (p0001 F2).
This script measures unique-response diversity at a given temperature/N
against a live ollama node so that mistake isn't repeated for E4. It is a
manual diagnostic (requires a live node), not a unit test.

Usage:
    uv run python -m scripts.measure_sample_diversity \\
        --ollama-host 192.168.15.103 --light-model isotnek/qwen3.5:9B-Unsloth-UD-Q4_K_XL \\
        --domain medical --query "3日前から頭痛と38度の発熱が続いています．" \\
        --n-samples 5 --temperature 0.7
"""

import argparse
import asyncio
import sys

from expert_backend import OllamaClient
from router import build_domain_verdict_prompt, parse_domain_verdict


async def measure_diversity(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query: str,
    n_samples: int,
    temperature: float,
    timeout_s: float,
) -> dict:
    """Sample n_samples domain verdicts and report response/reason diversity."""
    raw_responses = []
    for _ in range(n_samples):
        raw = await ollama_client.generate(
            light_model,
            build_domain_verdict_prompt(domain, query),
            timeout_s=timeout_s,
            max_tokens=150,
            temperature=temperature,
        )
        raw_responses.append(raw)

    parsed = [v for v in (parse_domain_verdict(r) for r in raw_responses) if v is not None]
    unique_reasons = {reason for _, reason in parsed}
    unique_fits = {fits for fits, _ in parsed}
    return {
        "n_samples": n_samples,
        "n_parsed": len(parsed),
        "n_unique_raw_responses": len(set(raw_responses)),
        "n_unique_reasons": len(unique_reasons),
        "fits_values_seen": sorted(unique_fits),
    }


async def _run(
    ollama_host: str,
    ollama_port: int,
    light_model: str,
    domain: str,
    query: str,
    n_samples: int,
    temperature: float,
    timeout_s: float,
) -> None:
    ollama_client = OllamaClient(host=f"http://{ollama_host}:{ollama_port}")
    result = await measure_diversity(
        ollama_client, light_model, domain, query, n_samples, temperature, timeout_s
    )
    print(f"[measure_sample_diversity] {result}", file=sys.stderr)
    if result["n_unique_reasons"] <= 1:
        print(
            f"警告: 全{n_samples}サンプルの理由文が実質1種類のみ．"
            f"T={temperature}, N={n_samples} では多様性が出ていない可能性が高い"
            "（Iter11の失敗パターンの再現に注意．本実験の前にtemperatureを上げるか"
            "N を増やすことを検討する）．",
            file=sys.stderr,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure sampling diversity before running E4 (self_consistency_semantic)"
    )
    parser.add_argument("--ollama-host", required=True, help="A live node's ollama daemon host/IP")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--light-model", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    asyncio.run(
        _run(
            args.ollama_host,
            args.ollama_port,
            args.light_model,
            args.domain,
            args.query,
            args.n_samples,
            args.temperature,
            args.timeout_s,
        )
    )


if __name__ == "__main__":
    main()

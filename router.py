"""Calculate domain-matching confidence via a lightweight LLM (method B) or embeddings (method A)."""

import json
import math
import re

from expert_backend import OllamaClient

_CONFIDENCE_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
PARSE_FAILURE_CONFIDENCE = 0.0
# Cap on output tokens for confidence scoring. Lightweight models may
# ignore instructions and generate endless text; this prevents runaway
# generation (ollama defaults to unlimited when num_predict is absent).
CONFIDENCE_MAX_TOKENS = 100
# Default temperature produces too much variance for reliable scoring.
# A low value ensures deterministic outputs across repeated calls.
CONFIDENCE_TEMPERATURE = 0.1
# Special domain name used as a catch-all fallback (see design doc 2.5).
# The regular prompt causes the model to assign high confidence to
# everything; this domain uses an inverted prompt instead.
GENERAL_DOMAIN = "general"


def _build_general_confidence_prompt(query_summary: str) -> str:
    """Construct the inverted-confidence prompt for the general fallback node."""
    return (
        "あなたは特定の専門分野を持たない汎用ノードです．\n"
        "次の質問が，医療・法律などの専門知識を必要とする特定分野の質問かどうかを，"
        "0.0〜1.0の数値（confidence）で評価してください．\n"
        "confidenceは「この質問を専門知識なしで一般的に回答できる度合い」を表します．\n\n"
        "評価基準:\n"
        "- 医療・法律等の専門知識を要する具体的な相談: 0.0〜0.3\n"
        "- 専門知識を要しない日常的な質問（雑談，レシピ，一般常識等）: 0.7〜1.0\n"
        "- 判断に迷う: 0.4〜0.6\n\n"
        '例1：質問「歯の痛みが続いています」は医療の専門知識を要するため{"confidence": 0.1}．\n'
        '例2：質問「おすすめの映画を教えてください」は専門知識を要しないため{"confidence": 0.9}．\n\n'
        f"質問: {query_summary}\n\n"
        '回答は{"confidence": <数値>}という1行のJSONのみとし，'
        "reasoning等の他のキーや説明文は一切含めないでください．"
    )


def build_confidence_prompt(domain: str, query_summary: str) -> str:
    """Construct the confidence-scoring prompt for a domain expert node.

    The prompt instructs the model to score based on whether the question's
    subject matter directly belongs to the domain, not whether it is merely
    tangentially related. This avoids false positives (e.g., a legal node
    claiming high confidence for a medical question).

    The "general" domain is delegated to _build_general_confidence_prompt.
    """
    if domain == GENERAL_DOMAIN:
        return _build_general_confidence_prompt(query_summary)
    return (
        f'あなたは「{domain}」分野の専門家ノードです．\n'
        f"次の質問の【主題】が「{domain}」分野の専門知識を必要とするかどうかを，"
        f"0.0〜1.0の数値（confidence）で評価してください．\n"
        f"confidenceは「質問がこの分野に該当する度合い」であり，あなたの判定の自信度ではありません．\n"
        f"質問の一部に{domain}と間接的に関連しうる語句が含まれていても，"
        f"主題が他分野であれば低い値にしてください．\n\n"
        f"評価基準:\n"
        f"- 主題が明確に{domain}分野に属する: 0.7〜1.0\n"
        f"- 主題が{domain}分野と無関係，または他分野がより適切: 0.0〜0.3\n"
        f"- 判断に迷う: 0.4〜0.6\n"
        f"- {domain}関連の語句が含まれていても，主題が他分野であれば{domain} confidence は低くする（例: 読書・勉強・習い事は general 分野）．\n\n"
        f'例1：質問「歯の痛みが続いています」はmedical分野に該当するため，'
        f'domainがmedicalなら{{"confidence": 0.9}}，domainがeducationなら{{"confidence": 0.1}}，domainがgeneralなら{{"confidence": 0.1}}，domainがlegalなら{{"confidence": 0.1}}．\n'
        f'例2：質問「賃貸契約を解除したい」はlegal分野に該当するため，'
        f'domainがlegalなら{{"confidence": 0.9}}，domainがmedicalなら{{"confidence": 0.1}}，domainがeducationなら{{"confidence": 0.1}}，domainがgeneralなら{{"confidence": 0.1}}．\n'
        f'例3：質問「学習指導要領における探究的学習の位置付けは」はeducation分野に該当するため，'
        f'domainがeducationなら{{"confidence": 0.9}}，domainがmedicalなら{{"confidence": 0.1}}，domainがlegalなら{{"confidence": 0.1}}，domainがgeneralなら{{"confidence": 0.1}}．\n'
        f'例4：質問「読書感想文の書き方」はgeneral分野に該当するため，'
        f'domainがgeneralなら{{"confidence": 0.9}}，domainがeducationなら{{"confidence": 0.1}}，domainがmedicalなら{{"confidence": 0.1}}，domainがlegalなら{{"confidence": 0.1}}，educationノードは{{"confidence": 0.1}}とする（general分野でありeducation分野ではない）．\n\n'
        f"質問: {query_summary}\n\n"
        '回答は{"confidence": <数値>}という1行のJSONのみとし，'
        "reasoning等の他のキーや説明文は一切含めないでください．"
    )


def parse_confidence(raw_response: str) -> float:
    """Extract the confidence value from the model's JSON output.

    Returns 0.0 on any parse failure to avoid false-positive dispatches.
    """
    match = _CONFIDENCE_JSON_PATTERN.search(raw_response)
    if match is None:
        return PARSE_FAILURE_CONFIDENCE
    try:
        parsed = json.loads(match.group())
        confidence = float(parsed["confidence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return PARSE_FAILURE_CONFIDENCE
    return min(max(confidence, 0.0), 1.0)


async def estimate_confidence(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
) -> float:
    """Send a confidence-scoring request to the lightweight model and return the result.

    This is routing method B (self-reported score) from design doc 2.4.
    """
    prompt = build_confidence_prompt(domain, query_summary)
    raw_response = await ollama_client.generate(
        light_model,
        prompt,
        timeout_s=timeout_s,
        max_tokens=CONFIDENCE_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
    )
    return parse_confidence(raw_response)


async def estimate_confidence_multi_sample(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
    n_samples: int = 3,
) -> tuple[float, float]:
    """Call estimate_confidence N times and return (mean_confidence, variance).

    Running the probe LLM multiple times on the same query averages out
    run-to-run noise (e.g. temperature=0.1 induced jitter of +/-0.05),
    producing a more stable confidence signal for routing decisions.
    """
    confidences = []
    for _ in range(n_samples):
        c = await estimate_confidence(ollama_client, light_model, domain, query_summary, timeout_s)
        confidences.append(c)
    mean_c = sum(confidences) / len(confidences)
    var_c = sum((x - mean_c) ** 2 for x in confidences) / len(confidences)
    return mean_c, var_c


async def estimate_confidence_stp(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
) -> tuple[float, float | None]:
    """Estimate confidence via Surrogate Token Probability (STP).

    Calls the LLM with logprobs enabled and uses the mean of all output
    token logprob values as a calibration signal. Unlike verbalized
    self-report confidence, this reflects the model's internal probability
    distribution over its vocabulary at each generation step.

    Returns (confidence_from_logprobs, raw_mean_logprob) where:
      - confidence_from_logprobs: normalized to [0, 1] for routing compatibility
      - raw_mean_logprob: the unnormalized mean logprob (or None if unavailable)
    """
    result = await ollama_client.generate(
        light_model,
        build_confidence_prompt(domain, query_summary),
        timeout_s=timeout_s,
        max_tokens=CONFIDENCE_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
        logprobs=1,  # Request 1 top-logprob per token
    )
    if isinstance(result, str):
        # Fallback to self-report if logprobs unavailable (e.g. old ollama)
        return parse_confidence(result), None

    token_logprobs = result.get("token_logprobs")
    if not token_logprobs:
        return parse_confidence(result["content"]), None

    mean_logprob = sum(entry["logprob"] for entry in token_logprobs) / len(token_logprobs)
    # Normalize: typical logprob range is [-10, 0]. Map to [0, 1] via sigmoid.
    normalized = 1.0 / (1.0 + math.exp(-mean_logprob - 2.0))
    return normalized, mean_logprob


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two vectors.

    Returns 0.0 for a zero-length or dimension-mismatched vector instead of
    raising, since this can legitimately happen at runtime: domain_embedding
    starts as [] until the /probe-serving node finishes its lifespan warmup
    (or has no embedding_model configured at all), and confidence 0.0 is
    the safe default that keeps such a node out of dispatch consideration.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def estimate_embedding_confidence(
    query_embedding: list[float], domain_embedding: list[float]
) -> float:
    """Score domain match via cosine similarity, rescaled from [-1, 1] to [0, 1].

    This is routing method A (embedding-based semantic routing) from design
    doc 2.4. Unlike method B it requires no LLM call, trading routing
    accuracy for near-zero probe latency; the two methods are compared
    directly since both plug into the same /probe response shape.
    """
    similarity = cosine_similarity(query_embedding, domain_embedding)
    return min(max((similarity + 1.0) / 2.0, 0.0), 1.0)

"""Calculate domain-matching confidence via a lightweight LLM (method B) or embeddings (method A)."""

import json
import math
import re
from collections.abc import Awaitable, Callable

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

# Fallback domain list for build_confidence_prompt when the caller has no
# peer list available (e.g. a NodeState built without `peers` in tests).
# Kept as the original 4-domain mesh so existing single-domain-argument
# call sites keep producing the same few-shot content as before 10-domain
# support was added.
_DEFAULT_ALL_DOMAINS: list[str] = ["medical", "legal", "education", "general"]

# One canonical example question per mesh domain, used to build a
# contrastive few-shot example (see _build_few_shot_examples). Domains not
# listed here (e.g. a future custom deployment) fall back to a generic
# placeholder question rather than raising, since the few-shot section is
# illustrative and does not need to be exhaustive to be useful.
_DOMAIN_EXAMPLE_QUERIES: dict[str, str] = {
    "medical": "歯の痛みが続いています",
    "legal": "賃貸契約を解除したい",
    "education": "学習指導要領における探究的学習の位置付けは",
    "general": "読書感想文の書き方",
    "business_economics": "自社の新商品のマーケティング戦略を立てたい",
    "computer_science": "自社サーバーのセキュリティ対策を強化したい",
    "natural_science": "物質の状態変化のメカニズムについて知りたい",
    "mathematics": "微分方程式の解き方を教えてほしい",
    "history_culture": "江戸時代の政治制度について知りたい",
    "social_science": "世界の宗教分布について知りたい",
}


def _build_few_shot_examples(all_domains: list[str]) -> str:
    """Build one contrastive few-shot example per domain in all_domains.

    Each example scores a domain's own canonical question at 0.9 for that
    domain and 0.1 for every other domain in the mesh. Generating one
    example per domain (rather than hand-authoring every example for a
    fixed domain count) is what lets this scale from 4 to 10+ domains
    without a prompt rewrite each time a domain is added.
    """
    lines = []
    for index, domain in enumerate(all_domains, start=1):
        query = _DOMAIN_EXAMPLE_QUERIES.get(domain, f"{domain}に関する質問")
        scores = "，".join(
            f'domainが{d}なら{{"confidence": {0.9 if d == domain else 0.1}}}' for d in all_domains
        )
        lines.append(f"例{index}：質問「{query}」は{domain}分野に該当するため，{scores}．")
    return "\n".join(lines)


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


def build_confidence_prompt(
    domain: str, query_summary: str, all_domains: list[str] | None = None
) -> str:
    """Construct the confidence-scoring prompt for a domain expert node.

    The prompt instructs the model to score based on whether the question's
    subject matter directly belongs to the domain, not whether it is merely
    tangentially related. This avoids false positives (e.g., a legal node
    claiming high confidence for a medical question).

    all_domains lists every domain in the mesh, used to build one
    contrastive few-shot example per domain; defaults to the original
    4-domain mesh when omitted (see _DEFAULT_ALL_DOMAINS) so this scales to
    any domain count without a prompt rewrite.

    The "general" domain is delegated to _build_general_confidence_prompt.
    """
    if domain == GENERAL_DOMAIN:
        return _build_general_confidence_prompt(query_summary)
    few_shot_examples = _build_few_shot_examples(all_domains or _DEFAULT_ALL_DOMAINS)
    return (
        f"あなたは「{domain}」分野の専門家ノードです．\n"
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
        f"{few_shot_examples}\n\n"
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
    all_domains: list[str] | None = None,
) -> float:
    """Send a confidence-scoring request to the lightweight model and return the result.

    This is routing method B (self-reported score) from design doc 2.4.
    """
    prompt = build_confidence_prompt(domain, query_summary, all_domains)
    raw_response = await ollama_client.generate(
        light_model,
        prompt,
        timeout_s=timeout_s,
        max_tokens=CONFIDENCE_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
    )
    return parse_confidence(raw_response)


# E3 (confidence_elicitation=top_k_with_probs): Tian et al. EMNLP 2023
# (arXiv:2305.14975) show that asking for K candidate labels with
# probabilities constrained to sum to 1 breaks the 0/1 saturation seen with
# a single numeric_scalar self-report (ECE 0.131 -> 0.047 for K=2). K=2 is
# used here since the underlying judgment is binary (fits this domain or
# not), not an open-ended answer set.
TOP_K_CANDIDATES = 2
_PROB_SUM_TOLERANCE = 0.02
_FITS_LABEL = "該当する"
_DOES_NOT_FIT_LABEL = "該当しない"


def _build_general_top_k_confidence_prompt(query_summary: str) -> str:
    """Inverted top_k_with_probs prompt for the general fallback node (see _build_general_confidence_prompt)."""
    return (
        "次の質問が専門知識を必要とせず一般的に回答できるかどうかについて，"
        f'"{_FITS_LABEL}"（専門知識なしで一般的に回答できる）と'
        f'"{_DOES_NOT_FIT_LABEL}"（医療・法律等の専門知識を要する）の'
        "2つの可能性それぞれの確率を見積もってください．\n"
        "確率の合計は1.0になるようにしてください．\n\n"
        f"質問: {query_summary}\n\n"
        "回答は次の形式の1行のJSONのみとし，他のキーや説明文は含めないでください:\n"
        f'{{"candidates": [{{"label": "{_FITS_LABEL}", "probability": <0.0-1.0の数値>}}, '
        f'{{"label": "{_DOES_NOT_FIT_LABEL}", "probability": <0.0-1.0の数値>}}]}}'
    )


def build_top_k_confidence_prompt(domain: str, query_summary: str) -> str:
    """Construct the top_k_with_probs confidence-scoring prompt (E3).

    Asks for exactly TOP_K_CANDIDATES=2 labeled candidates ("該当する" /
    "該当しない") with probabilities, rather than a single numeric_scalar,
    per Tian et al. 2023's top-K elicitation method. The "general" domain
    is delegated to _build_general_top_k_confidence_prompt, mirroring
    build_confidence_prompt's structure.
    """
    if domain == GENERAL_DOMAIN:
        return _build_general_top_k_confidence_prompt(query_summary)
    return (
        f"次の質問の【主題】が「{domain}」分野の専門知識を必要とするかどうかについて，"
        f'"{_FITS_LABEL}"と"{_DOES_NOT_FIT_LABEL}"の2つの可能性それぞれの確率を見積もってください．\n'
        "確率の合計は1.0になるようにしてください．\n\n"
        f"質問: {query_summary}\n\n"
        "回答は次の形式の1行のJSONのみとし，他のキーや説明文は含めないでください:\n"
        f'{{"candidates": [{{"label": "{_FITS_LABEL}", "probability": <0.0-1.0の数値>}}, '
        f'{{"label": "{_DOES_NOT_FIT_LABEL}", "probability": <0.0-1.0の数値>}}]}}'
    )


def parse_top_k_confidence(raw_response: str) -> float:
    """Extract the "該当する" probability from a top_k_with_probs response.

    Renormalizes when the candidate probabilities don't sum to ~1.0
    (small models often drift slightly), rather than trusting the model's
    arithmetic outright. Returns PARSE_FAILURE_CONFIDENCE on any parse
    failure, matching parse_confidence's fail-safe convention.
    """
    match = _CONFIDENCE_JSON_PATTERN.search(raw_response)
    if match is None:
        return PARSE_FAILURE_CONFIDENCE
    try:
        candidates = json.loads(match.group())["candidates"]
        total = sum(float(c["probability"]) for c in candidates)
        fits_probability = next(
            float(c["probability"]) for c in candidates if c["label"] == _FITS_LABEL
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, StopIteration):
        return PARSE_FAILURE_CONFIDENCE
    if total <= 0.0:
        return PARSE_FAILURE_CONFIDENCE
    if abs(total - 1.0) > _PROB_SUM_TOLERANCE:
        fits_probability = fits_probability / total
    return min(max(fits_probability, 0.0), 1.0)


async def estimate_confidence_top_k(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
) -> float:
    """Send a top_k_with_probs confidence-scoring request to the lightweight model."""
    prompt = build_top_k_confidence_prompt(domain, query_summary)
    raw_response = await ollama_client.generate(
        light_model,
        prompt,
        timeout_s=timeout_s,
        max_tokens=CONFIDENCE_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
    )
    return parse_top_k_confidence(raw_response)


async def estimate_confidence_multi_sample(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
    n_samples: int = 3,
    all_domains: list[str] | None = None,
) -> tuple[float, float]:
    """Call estimate_confidence N times and return (mean_confidence, variance).

    Running the probe LLM multiple times on the same query averages out
    run-to-run noise (e.g. temperature=0.1 induced jitter of +/-0.05),
    producing a more stable confidence signal for routing decisions.
    """
    confidences = []
    for _ in range(n_samples):
        c = await estimate_confidence(
            ollama_client, light_model, domain, query_summary, timeout_s, all_domains
        )
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
    all_domains: list[str] | None = None,
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
        build_confidence_prompt(domain, query_summary, all_domains),
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


# E4 (confidence_signal_method=self_consistency_semantic): Discrete Semantic
# Entropy (Farquhar et al., Nature 630:625-630, 2024; formalized by Cecere
# et al., TrustNLP 2025 arXiv:2502.18389). Samples N domain-fit verdicts at
# T=0.7-1.0 (Wang et al. 2022 / Xiong et al. ICLR2024: "T=0.7 to gather a
# more diverse answer set"), clusters their stated reasons by entailment,
# and uses the resulting cluster-size entropy to discount the fits
# fraction. This is Iter11's multi_sample idea done at the temperature the
# literature actually specifies for uncertainty estimation, plus semantic
# clustering instead of a plain numeric average (Iter11 could not tell
# "one answer sampled three times" apart from "three different answers").
SEMANTIC_SAMPLE_COUNT = 5
SEMANTIC_SAMPLE_TEMPERATURE = 0.7
SEMANTIC_VERDICT_MAX_TOKENS = 150
ENTAILMENT_MAX_TOKENS = 10
# Deterministic entailment judgments (same question asked of the same
# model should get a consistent same/different-claim answer).
ENTAILMENT_TEMPERATURE = 0.0


def build_domain_verdict_prompt(domain: str, query_summary: str) -> str:
    """Ask for a domain-fit judgment plus a one-sentence justification.

    Unlike build_confidence_prompt's single numeric score, this captures
    *why* the model reached its verdict, which _cluster_reasons_by_entailment
    then uses to tell "the same judgment restated" apart from "genuinely
    different reasoning" across independent samples.
    """
    return (
        f"次の質問が「{domain}」分野の専門知識を必要とするかどうかを判定してください．\n"
        "判定理由を一文で簡潔に述べてください．\n\n"
        f"質問: {query_summary}\n\n"
        '回答は{"fits": true または false, "reason": "一文の理由"}という形式の1行のJSONのみとし，'
        "他のキーや説明文は含めないでください．"
    )


def parse_domain_verdict(raw_response: str) -> tuple[bool, str] | None:
    """Extract (fits, reason) from a domain-verdict response; None on any parse failure."""
    match = _CONFIDENCE_JSON_PATTERN.search(raw_response)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group())
        fits = bool(parsed["fits"])
        reason = str(parsed["reason"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return fits, reason


async def _sample_domain_verdicts(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
    n_samples: int,
    temperature: float,
) -> list[tuple[bool, str]]:
    """Sample n_samples independent domain verdicts; unparseable samples are dropped.

    Dropping (rather than treating a parse failure as its own cluster or
    defaulting to fits=False) keeps compute_discrete_semantic_entropy's
    accounting simple: every element of the returned list is a genuine
    model judgment, not a parse-error artifact.
    """
    verdicts = []
    for _ in range(n_samples):
        raw_response = await ollama_client.generate(
            light_model,
            build_domain_verdict_prompt(domain, query_summary),
            timeout_s=timeout_s,
            max_tokens=SEMANTIC_VERDICT_MAX_TOKENS,
            temperature=temperature,
        )
        verdict = parse_domain_verdict(raw_response)
        if verdict is not None:
            verdicts.append(verdict)
    return verdicts


def _build_entailment_prompt(statement_a: str, statement_b: str) -> str:
    """Ask whether two short reason strings assert the same underlying claim."""
    return (
        "次の2つの文が実質的に同じ主張をしているかどうかを判定してください．\n"
        f"文A: {statement_a}\n"
        f"文B: {statement_b}\n\n"
        '回答は{"same_claim": true または false}という形式の1行のJSONのみとし，'
        "他のキーや説明文は含めないでください．"
    )


def _parse_entailment(raw_response: str) -> bool:
    """Extract the same_claim boolean; a parse failure is treated as "not the same claim"

    (conservative: prefers to split into more, smaller clusters over
    silently merging two genuinely different reasons)."""
    match = _CONFIDENCE_JSON_PATTERN.search(raw_response)
    if match is None:
        return False
    try:
        return bool(json.loads(match.group())["same_claim"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


async def _entails(
    ollama_client: OllamaClient,
    light_model: str,
    statement_a: str,
    statement_b: str,
    timeout_s: float,
) -> bool:
    """Ask the same LLM whether two reason strings assert the same underlying claim.

    Stands in for a dedicated NLI model, which this Ollama-only setup does
    not have available (d0001 §2.2: "NLI モデルが必要だが同一LLMへの
    entailment プロンプトで代用可").
    """
    raw_response = await ollama_client.generate(
        light_model,
        _build_entailment_prompt(statement_a, statement_b),
        timeout_s=timeout_s,
        max_tokens=ENTAILMENT_MAX_TOKENS,
        temperature=ENTAILMENT_TEMPERATURE,
    )
    return _parse_entailment(raw_response)


async def _cluster_reasons_by_entailment(
    reasons: list[str], entailment_lookup: Callable[[str, str], Awaitable[bool]]
) -> list[list[int]]:
    """Greedy single-linkage clustering of reasons by pairwise entailment.

    Each reason joins the first existing cluster whose representative
    (its first member) it's judged to share a claim with; otherwise it
    starts a new cluster. entailment_lookup is `async (str, str) -> bool`
    so tests can substitute a cheap stub instead of calling an LLM.
    At most O(n_reasons x n_clusters) calls, which is <= O(n^2) but
    typically far less since clusters tend to merge quickly.
    """
    clusters: list[list[int]] = []
    for index, reason in enumerate(reasons):
        joined = False
        for cluster in clusters:
            representative = reasons[cluster[0]]
            if await entailment_lookup(reason, representative):
                cluster.append(index)
                joined = True
                break
        if not joined:
            clusters.append([index])
    return clusters


def compute_discrete_semantic_entropy(cluster_sizes: list[int], total: int) -> float:
    """Shannon entropy (bits) of the cluster-size distribution.

    0.0 when every sample fell into one cluster (full agreement, no
    uncertainty); maximal (log2(total)) when every sample formed its own
    singleton cluster (total disagreement). total=0 returns 0.0 (nothing to
    measure) rather than raising.
    """
    if total == 0:
        return 0.0
    return -sum((size / total) * math.log2(size / total) for size in cluster_sizes if size > 0)


async def estimate_confidence_semantic_entropy(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
    n_samples: int = SEMANTIC_SAMPLE_COUNT,
    temperature: float = SEMANTIC_SAMPLE_TEMPERATURE,
) -> tuple[float, float]:
    """Estimate confidence via Discrete Semantic Entropy (E4).

    Returns (confidence, raw_entropy) where confidence is the fraction of
    samples that judged the query to fit this domain, discounted by the
    normalized entropy of their clustered reasons (entropy / log2(n) so it
    is comparable across a different number of successfully-parsed
    samples). raw_entropy is the undiscounted Shannon entropy in bits, kept
    for offline analysis of same-vs-different-answer diversity (guarding
    against a repeat of Iter11's failure to distinguish the two).
    """
    verdicts = await _sample_domain_verdicts(
        ollama_client, light_model, domain, query_summary, timeout_s, n_samples, temperature
    )
    if not verdicts:
        return PARSE_FAILURE_CONFIDENCE, 0.0

    fits_fraction = sum(1 for fits, _ in verdicts if fits) / len(verdicts)
    reasons = [reason for _, reason in verdicts]

    async def entailment_lookup(a: str, b: str) -> bool:
        return await _entails(ollama_client, light_model, a, b, timeout_s)

    clusters = await _cluster_reasons_by_entailment(reasons, entailment_lookup)
    cluster_sizes = [len(cluster) for cluster in clusters]
    entropy = compute_discrete_semantic_entropy(cluster_sizes, len(reasons))

    max_entropy = math.log2(len(reasons)) if len(reasons) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    confidence = fits_fraction * (1.0 - normalized_entropy)
    return confidence, entropy


# E5 (confidence_signal_method=p_true): Kadavath et al. 2022 (arXiv:2207.05221)
# self-evaluation. Two stages: (1) a free-form domain-fit verdict (the
# "proposed answer"), (2) asking the model to judge that verdict's truth
# via a constrained "(A) True (B) False" answer, reading P("A") from
# top_logprobs rather than trusting whichever letter was actually sampled
# (the model might generate "B" while still assigning "A" a non-trivial
# probability). Kadavath et al.'s strong results are from a 52B base model
# with 20-shot prompting; a direct counterexample (Tian et al. 2023 Table 1)
# reports "Is True" probability calibrating *worse* than verbalized
# confidence for gpt-3.5-turbo, so this is not assumed to reproduce as-is
# on a small RLHF'd model in zero-shot (see docs/d0001_literature_survey §2.3).
P_TRUE_VERDICT_MAX_TOKENS = 150
P_TRUE_CHECK_MAX_TOKENS = 5
P_TRUE_TOP_LOGPROBS = 5
P_TRUE_TRUE_TOKEN = "A"


def build_p_true_verdict_prompt(domain: str, query_summary: str) -> str:
    """Stage 1: a free-form domain-fit judgment (Kadavath et al.'s "proposed answer")."""
    return (
        f"次の質問が「{domain}」分野の専門知識を必要とするかどうかを，"
        "理由とともに簡潔に判定してください．\n\n"
        f"質問: {query_summary}"
    )


def build_p_true_check_prompt(domain: str, query_summary: str, proposed_verdict: str) -> str:
    """Stage 2: Kadavath et al. 2022's P(True) self-evaluation prompt."""
    return (
        f"質問: {query_summary}\n"
        f"この質問が「{domain}」分野に該当するかについての提案された判定: {proposed_verdict}\n\n"
        "この提案された判定は正しいですか？\n"
        "(A) True\n"
        "(B) False\n\n"
        '"A"または"B"の1文字のみで回答してください．'
    )


def extract_p_true(token_logprobs: list[dict] | None) -> float:
    """Extract P(True) from the first token position's top_logprobs alternatives.

    Returns PARSE_FAILURE_CONFIDENCE when token_logprobs is empty/None, or
    when P_TRUE_TRUE_TOKEN doesn't appear among the reported alternatives
    (possible with a small top_logprobs count if the model is very
    confident in "B" instead). A genuine logprob is always <= 0, so
    exp(logprob) is mathematically bounded to (0, 1]; the clamp only
    guards against a malformed/out-of-range value from the ollama
    response reaching ProbeResponse's confidence field (Field(ge=0, le=1)),
    which would otherwise surface as an unhandled validation error instead
    of a graceful PARSE_FAILURE_CONFIDENCE-style degradation.
    """
    if not token_logprobs:
        return PARSE_FAILURE_CONFIDENCE
    alternatives = token_logprobs[0].get("top_logprobs", {})
    if P_TRUE_TRUE_TOKEN not in alternatives:
        return PARSE_FAILURE_CONFIDENCE
    return min(max(math.exp(alternatives[P_TRUE_TRUE_TOKEN]), 0.0), 1.0)


async def estimate_confidence_p_true(
    ollama_client: OllamaClient,
    light_model: str,
    domain: str,
    query_summary: str,
    timeout_s: float,
) -> tuple[float, float | None]:
    """Estimate confidence via P(True) self-evaluation (E5).

    Falls back to the numeric_scalar self_report signal when top_logprobs
    is unavailable (pre-v0.12.11 ollama silently ignoring the request),
    matching estimate_confidence_stp's fallback pattern. Returns
    (confidence, raw_p_true) where raw_p_true is None when the fallback
    path was used (so callers/logs can tell "p_true was actually measured"
    apart from "p_true silently degraded to self_report").
    """
    proposed_verdict = await ollama_client.generate(
        light_model,
        build_p_true_verdict_prompt(domain, query_summary),
        timeout_s=timeout_s,
        max_tokens=P_TRUE_VERDICT_MAX_TOKENS,
        temperature=CONFIDENCE_TEMPERATURE,
    )
    result = await ollama_client.generate(
        light_model,
        build_p_true_check_prompt(domain, query_summary, proposed_verdict),
        timeout_s=timeout_s,
        max_tokens=P_TRUE_CHECK_MAX_TOKENS,
        temperature=0.0,
        logprobs=1,
        top_logprobs=P_TRUE_TOP_LOGPROBS,
    )
    token_logprobs = None if isinstance(result, str) else result.get("token_logprobs")
    if not token_logprobs:
        fallback_confidence = await estimate_confidence(
            ollama_client, light_model, domain, query_summary, timeout_s
        )
        return fallback_confidence, None
    p_true = extract_p_true(token_logprobs)
    return p_true, p_true


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


# embedding_postprocess identifiers (E7). "none" preserves current
# behavior; "mean_center" and "whiten" address the anisotropy that
# collapsed Iter2's cosine similarities into a narrow [0.667, 0.737] band
# (Ethayarajh 2019; Su et al. 2021 arXiv:2103.15316).
EMBEDDING_POSTPROCESS_NONE = "none"
EMBEDDING_POSTPROCESS_MEAN_CENTER = "mean_center"
EMBEDDING_POSTPROCESS_WHITEN = "whiten"


def load_embedding_postprocess_params(path: str) -> tuple[list[float], list[list[float]] | None]:
    """Load the (mean_vector, whitening_matrix) artifact produced by fit_embedding_whitening.py.

    whitening_matrix is None for a mean_center-only artifact (no SVD was
    computed), so mean_center mode can be fit without ever importing numpy.
    """
    with open(path, encoding="utf-8") as f:
        artifact = json.load(f)
    return artifact["mean_vector"], artifact.get("whitening_matrix")


def apply_mean_centering(vector: list[float], mean_vector: list[float]) -> list[float]:
    """Subtract the background mean from a vector."""
    return [x - m for x, m in zip(vector, mean_vector, strict=True)]


def apply_whitening(
    vector: list[float], mean_vector: list[float], whitening_matrix: list[list[float]]
) -> list[float]:
    """Mean-center then apply a precomputed whitening transform: centered @ whitening_matrix.

    Plain Python matrix-vector product (no numpy) since this runs once per
    /probe call on the serving path; O(dim^2) is negligible next to LLM
    generation latency, and keeping numpy out of router.py means node
    processes never need it installed.
    """
    centered = apply_mean_centering(vector, mean_vector)
    dim = len(whitening_matrix[0])
    return [
        sum(centered[i] * whitening_matrix[i][j] for i in range(len(centered))) for j in range(dim)
    ]


def apply_embedding_postprocess(
    query_embedding: list[float],
    domain_embedding: list[float],
    method: str,
    mean_vector: list[float] | None,
    whitening_matrix: list[list[float]] | None,
) -> tuple[list[float], list[float]]:
    """Apply the configured postprocess to both vectors before cosine scoring.

    Returns the inputs unchanged when method is "none" or the artifact
    hasn't been loaded (mean_vector is None), which keeps the current
    behavior as the default. Raises ValueError for an unrecognized method
    rather than silently falling back, matching the fail-fast validation
    used for routing_method/confidence_signal_method (see node.py's
    build_node_state).
    """
    if method == EMBEDDING_POSTPROCESS_NONE or mean_vector is None:
        return query_embedding, domain_embedding
    if method == EMBEDDING_POSTPROCESS_MEAN_CENTER:
        return (
            apply_mean_centering(query_embedding, mean_vector),
            apply_mean_centering(domain_embedding, mean_vector),
        )
    if method == EMBEDDING_POSTPROCESS_WHITEN:
        if whitening_matrix is None:
            raise ValueError("embedding_postprocess=whiten requires a whitening_matrix")
        return (
            apply_whitening(query_embedding, mean_vector, whitening_matrix),
            apply_whitening(domain_embedding, mean_vector, whitening_matrix),
        )
    raise ValueError(f"unknown embedding_postprocess: {method!r}")

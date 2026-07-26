"""Tests for the pure-function parts of confidence scoring in router.py."""

import json
import math
from unittest.mock import AsyncMock

import pytest

from expert_backend import OllamaClient
from router import (
    EMBEDDING_POSTPROCESS_MEAN_CENTER,
    EMBEDDING_POSTPROCESS_NONE,
    EMBEDDING_POSTPROCESS_WHITEN,
    PARSE_FAILURE_CONFIDENCE,
    _cluster_reasons_by_entailment,
    apply_embedding_postprocess,
    apply_mean_centering,
    apply_whitening,
    build_confidence_prompt,
    build_domain_verdict_prompt,
    build_top_k_confidence_prompt,
    compute_discrete_semantic_entropy,
    cosine_similarity,
    estimate_confidence_semantic_entropy,
    estimate_embedding_confidence,
    load_embedding_postprocess_params,
    estimate_confidence_p_true,
    extract_p_true,
    parse_confidence,
    parse_domain_verdict,
    parse_top_k_confidence,
)


def test_build_confidence_prompt_includes_domain_and_summary() -> None:
    """The prompt contains both the domain name and the query summary."""
    prompt = build_confidence_prompt("medical", "question about headache and fever")
    assert "medical" in prompt
    assert "question about headache and fever" in prompt


def test_build_confidence_prompt_uses_reversed_criteria_for_general() -> None:
    """The general domain gets a dedicated prompt with inverted scoring criteria."""
    prompt = build_confidence_prompt("general", "question about headache and fever")
    assert "question about headache and fever" in prompt
    assert "専門知識を要しない" in prompt


def test_build_confidence_prompt_generates_one_few_shot_example_per_domain() -> None:
    """Passing a 10-domain all_domains list produces 10 few-shot examples, not a fixed 4."""
    ten_domains = [
        "medical",
        "legal",
        "education",
        "general",
        "business_economics",
        "computer_science",
        "natural_science",
        "mathematics",
        "history_culture",
        "social_science",
    ]
    prompt = build_confidence_prompt("computer_science", "question", all_domains=ten_domains)
    for domain in ten_domains:
        assert domain in prompt


def test_build_confidence_prompt_defaults_to_four_domain_mesh_without_all_domains() -> None:
    """Omitting all_domains keeps the original 4-domain few-shot content (back-compat)."""
    prompt = build_confidence_prompt("medical", "question")
    for domain in ["medical", "legal", "education", "general"]:
        assert domain in prompt


def test_parse_confidence_extracts_value_from_clean_json() -> None:
    """Parse confidence from a well-formed JSON string."""
    assert parse_confidence('{"confidence": 0.87}') == 0.87


def test_parse_confidence_extracts_value_from_json_with_surrounding_text() -> None:
    """Handle LLM output that includes text before or after the JSON object."""
    raw = 'The confidence is as follows.\n{"confidence": 0.42}\nHope this helps.'
    assert parse_confidence(raw) == 0.42


def test_parse_confidence_falls_back_on_invalid_json() -> None:
    """Return 0.0 when the response is not valid JSON."""
    assert parse_confidence("I don't know") == PARSE_FAILURE_CONFIDENCE


def test_parse_confidence_falls_back_on_missing_key() -> None:
    """Return 0.0 when the JSON object lacks the confidence key."""
    assert parse_confidence('{"score": 0.9}') == PARSE_FAILURE_CONFIDENCE


def test_parse_confidence_clamps_value_above_one() -> None:
    """Cap confidence at 1.0 even if the model outputs a higher value."""
    assert parse_confidence('{"confidence": 1.5}') == 1.0


def test_parse_confidence_clamps_negative_value() -> None:
    """Cap confidence at 0.0 even if the model outputs a negative value."""
    assert parse_confidence('{"confidence": -0.3}') == 0.0


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    """Identical vectors have maximum similarity."""
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    """Orthogonal vectors have zero similarity."""
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_of_opposite_vectors_is_negative_one() -> None:
    """Exactly opposite vectors have minimum similarity."""
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_cosine_similarity_returns_zero_for_zero_vector() -> None:
    """A zero-magnitude vector returns 0.0 instead of raising a division error."""
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_returns_zero_for_empty_vectors() -> None:
    """An empty domain_embedding (before lifespan warmup) returns 0.0, not an error."""
    assert cosine_similarity([], []) == 0.0


def test_cosine_similarity_returns_zero_for_mismatched_dimensions() -> None:
    """Vectors of different length return 0.0 instead of raising."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_estimate_embedding_confidence_rescales_similarity_to_unit_range() -> None:
    """Cosine similarity in [-1, 1] is rescaled to a confidence in [0, 1]."""
    assert estimate_embedding_confidence([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert estimate_embedding_confidence([1.0, 0.0], [0.0, 1.0]) == 0.5
    assert estimate_embedding_confidence([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_apply_mean_centering_subtracts_mean() -> None:
    """Mean centering subtracts the background mean elementwise."""
    assert apply_mean_centering([5.0, 3.0], [2.0, 1.0]) == [3.0, 2.0]


def test_apply_whitening_decorrelates_a_diagonal_toy_covariance() -> None:
    """Whitening a background with covariance diag(4, 1) yields unit variance on each axis.

    Uses a hand-computable whitening matrix (no numpy/SVD needed here:
    for a diagonal covariance diag(v1, v2), W = diag(1/sqrt(v1), 1/sqrt(v2))
    is exactly the whitening transform), so this test stays independent of
    scripts/fit_embedding_whitening.py's numpy-based SVD implementation.
    """
    mean_vector = [0.0, 0.0]
    whitening_matrix = [[0.5, 0.0], [0.0, 1.0]]  # 1/sqrt(4)=0.5, 1/sqrt(1)=1.0
    assert apply_whitening([2.0, 1.0], mean_vector, whitening_matrix) == [1.0, 1.0]
    assert apply_whitening([-2.0, -1.0], mean_vector, whitening_matrix) == [-1.0, -1.0]


def test_apply_embedding_postprocess_returns_originals_when_method_none() -> None:
    """method="none" is a no-op regardless of whether params were loaded."""
    query, domain = apply_embedding_postprocess(
        [1.0, 2.0],
        [3.0, 4.0],
        EMBEDDING_POSTPROCESS_NONE,
        mean_vector=[0.1, 0.1],
        whitening_matrix=None,
    )
    assert query == [1.0, 2.0]
    assert domain == [3.0, 4.0]


def test_apply_embedding_postprocess_returns_originals_when_params_not_loaded() -> None:
    """A None mean_vector (artifact not loaded) is treated as method=none, not an error."""
    query, domain = apply_embedding_postprocess(
        [1.0, 2.0],
        [3.0, 4.0],
        EMBEDDING_POSTPROCESS_WHITEN,
        mean_vector=None,
        whitening_matrix=None,
    )
    assert query == [1.0, 2.0]
    assert domain == [3.0, 4.0]


def test_apply_embedding_postprocess_mean_centers_both_vectors() -> None:
    """method="mean_center" applies the same centering to both query and domain vectors."""
    query, domain = apply_embedding_postprocess(
        [5.0, 3.0],
        [4.0, 2.0],
        EMBEDDING_POSTPROCESS_MEAN_CENTER,
        mean_vector=[1.0, 1.0],
        whitening_matrix=None,
    )
    assert query == [4.0, 2.0]
    assert domain == [3.0, 1.0]


def test_apply_embedding_postprocess_raises_on_unknown_method() -> None:
    """An unrecognized method fails fast rather than silently falling back to none."""
    with pytest.raises(ValueError):
        apply_embedding_postprocess([1.0], [1.0], "bogus", mean_vector=[0.0], whitening_matrix=None)


def test_apply_embedding_postprocess_raises_on_whiten_without_matrix() -> None:
    """method="whiten" without a loaded whitening_matrix fails fast."""
    with pytest.raises(ValueError):
        apply_embedding_postprocess(
            [1.0], [1.0], EMBEDDING_POSTPROCESS_WHITEN, mean_vector=[0.0], whitening_matrix=None
        )


def test_load_embedding_postprocess_params_reads_mean_center_artifact(tmp_path) -> None:
    """A mean_center-only artifact (no whitening_matrix key) loads with whitening_matrix=None."""
    artifact_path = tmp_path / "embedding_whitening.json"
    artifact_path.write_text(json.dumps({"mean_vector": [0.1, 0.2]}), encoding="utf-8")

    mean_vector, whitening_matrix = load_embedding_postprocess_params(str(artifact_path))
    assert mean_vector == [0.1, 0.2]
    assert whitening_matrix is None


def test_load_embedding_postprocess_params_reads_whiten_artifact(tmp_path) -> None:
    """A whiten artifact's whitening_matrix round-trips through JSON unchanged."""
    artifact_path = tmp_path / "embedding_whitening.json"
    artifact_path.write_text(
        json.dumps({"mean_vector": [0.0, 0.0], "whitening_matrix": [[1.0, 0.0], [0.0, 1.0]]}),
        encoding="utf-8",
    )

    mean_vector, whitening_matrix = load_embedding_postprocess_params(str(artifact_path))
    assert mean_vector == [0.0, 0.0]
    assert whitening_matrix == [[1.0, 0.0], [0.0, 1.0]]


def test_build_top_k_confidence_prompt_includes_domain_and_summary() -> None:
    """The top_k prompt names both candidate labels and the query summary."""
    prompt = build_top_k_confidence_prompt("medical", "headache and fever")
    assert "medical" in prompt
    assert "headache and fever" in prompt
    assert "該当する" in prompt
    assert "該当しない" in prompt


def test_build_top_k_confidence_prompt_uses_inverted_wording_for_general() -> None:
    """The general domain gets the inverted (専門知識不要 vs 専門知識が必要) framing."""
    prompt = build_top_k_confidence_prompt("general", "headache and fever")
    assert "専門知識" in prompt


def test_parse_top_k_confidence_extracts_fits_probability_from_clean_json() -> None:
    """A well-formed, already-normalized candidates array is parsed directly."""
    raw = '{"candidates": [{"label": "該当する", "probability": 0.7}, {"label": "該当しない", "probability": 0.3}]}'
    assert parse_top_k_confidence(raw) == pytest.approx(0.7)


def test_parse_top_k_confidence_renormalizes_when_sum_deviates_from_one() -> None:
    """Probabilities summing to 1.2 are renormalized rather than trusted as-is."""
    raw = '{"candidates": [{"label": "該当する", "probability": 0.6}, {"label": "該当しない", "probability": 0.6}]}'
    assert parse_top_k_confidence(raw) == pytest.approx(0.5)


def test_parse_top_k_confidence_falls_back_on_invalid_json() -> None:
    """Return PARSE_FAILURE_CONFIDENCE when the response is not valid JSON."""
    assert parse_top_k_confidence("I don't know") == PARSE_FAILURE_CONFIDENCE


def test_parse_top_k_confidence_falls_back_when_candidates_key_missing() -> None:
    """Return PARSE_FAILURE_CONFIDENCE when the JSON lacks a candidates array."""
    assert parse_top_k_confidence('{"confidence": 0.9}') == PARSE_FAILURE_CONFIDENCE


def test_parse_top_k_confidence_falls_back_when_fits_label_missing() -> None:
    """Return PARSE_FAILURE_CONFIDENCE when no candidate carries the "該当する" label."""
    raw = '{"candidates": [{"label": "unknown", "probability": 0.7}, {"label": "該当しない", "probability": 0.3}]}'
    assert parse_top_k_confidence(raw) == PARSE_FAILURE_CONFIDENCE


def test_build_domain_verdict_prompt_includes_domain_and_summary() -> None:
    """The verdict prompt names the domain and includes the query summary."""
    prompt = build_domain_verdict_prompt("medical", "headache and fever")
    assert "medical" in prompt
    assert "headache and fever" in prompt


def test_parse_domain_verdict_extracts_fits_and_reason() -> None:
    """A well-formed verdict response yields (fits, reason)."""
    raw = '{"fits": true, "reason": "頭痛と発熱は医療分野の症状である"}'
    assert parse_domain_verdict(raw) == (True, "頭痛と発熱は医療分野の症状である")


def test_parse_domain_verdict_falls_back_on_invalid_json() -> None:
    """Return None (not a default verdict) on unparseable output."""
    assert parse_domain_verdict("I don't know") is None


def test_parse_domain_verdict_falls_back_on_missing_keys() -> None:
    """Return None when fits or reason is absent."""
    assert parse_domain_verdict('{"fits": true}') is None


def test_compute_discrete_semantic_entropy_all_same_cluster_is_zero() -> None:
    """Full agreement (one cluster containing every sample) has zero entropy."""
    assert compute_discrete_semantic_entropy([5], 5) == 0.0


def test_compute_discrete_semantic_entropy_all_singleton_clusters_is_maximal() -> None:
    """Total disagreement (every sample its own cluster) has entropy log2(n)."""
    assert compute_discrete_semantic_entropy([1, 1, 1, 1], 4) == pytest.approx(math.log2(4))


def test_compute_discrete_semantic_entropy_empty_is_zero() -> None:
    """total=0 returns 0.0 rather than dividing by zero."""
    assert compute_discrete_semantic_entropy([], 0) == 0.0


async def test_cluster_reasons_by_entailment_groups_matching_reasons() -> None:
    """Reasons judged equivalent by entailment_lookup join the same cluster."""
    reasons = ["頭痛は医療分野である", "医療の話題である", "全く関係ない話題"]

    async def stub_lookup(a: str, b: str) -> bool:
        # First two reasons are "the same claim"; the third is not.
        return {a, b} == {reasons[0], reasons[1]}

    clusters = await _cluster_reasons_by_entailment(reasons, stub_lookup)
    assert sorted(clusters, key=len) == [[2], [0, 1]]


async def test_cluster_reasons_by_entailment_keeps_distinct_reasons_separate() -> None:
    """When entailment_lookup always returns False, every reason is its own cluster."""

    async def never_entails(a: str, b: str) -> bool:
        return False

    clusters = await _cluster_reasons_by_entailment(["a", "b", "c"], never_entails)
    assert clusters == [[0], [1], [2]]


async def test_estimate_confidence_semantic_entropy_full_agreement_gives_full_confidence() -> None:
    """5 identical fits=True verdicts with matching reasons -> confidence 1.0, entropy 0.0."""
    ollama_client = AsyncMock(spec=OllamaClient)
    verdict_response = '{"fits": true, "reason": "医療分野の症状である"}'
    entailment_response = '{"same_claim": true}'
    # 5 verdict-sampling calls, followed by up to 4 entailment calls.
    ollama_client.generate.side_effect = [verdict_response] * 5 + [entailment_response] * 4

    confidence, entropy = await estimate_confidence_semantic_entropy(
        ollama_client, "light-model", "medical", "headache", timeout_s=2.0, n_samples=5
    )
    assert confidence == pytest.approx(1.0)
    assert entropy == pytest.approx(0.0)


async def test_estimate_confidence_semantic_entropy_returns_zero_when_all_samples_unparseable() -> (
    None
):
    """If every verdict sample fails to parse, confidence falls back to PARSE_FAILURE_CONFIDENCE."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.return_value = "not json"

    confidence, entropy = await estimate_confidence_semantic_entropy(
        ollama_client, "light-model", "medical", "headache", timeout_s=2.0, n_samples=3
    )
    assert confidence == PARSE_FAILURE_CONFIDENCE
    assert entropy == 0.0


def test_extract_p_true_reads_true_token_probability_from_alternatives() -> None:
    """P(True) is exp(logprob) of the "A" alternative at the first token position."""
    token_logprobs = [{"token": "B", "logprob": -3.0, "top_logprobs": {"A": -0.1, "B": -3.0}}]
    assert extract_p_true(token_logprobs) == pytest.approx(math.exp(-0.1))


def test_extract_p_true_returns_failure_when_true_token_absent() -> None:
    """If "A" never appears among reported alternatives, return PARSE_FAILURE_CONFIDENCE."""
    token_logprobs = [{"token": "B", "logprob": -0.1, "top_logprobs": {"B": -0.1, "C": -2.0}}]
    assert extract_p_true(token_logprobs) == PARSE_FAILURE_CONFIDENCE


def test_extract_p_true_returns_failure_on_empty_token_logprobs() -> None:
    """An empty or None token_logprobs list returns PARSE_FAILURE_CONFIDENCE, not an IndexError."""
    assert extract_p_true([]) == PARSE_FAILURE_CONFIDENCE
    assert extract_p_true(None) == PARSE_FAILURE_CONFIDENCE


def test_extract_p_true_clamps_a_malformed_positive_logprob_to_one() -> None:
    """A genuine logprob is always <= 0, but a malformed/out-of-range value from a
    misbehaving ollama response must still be clamped to the valid [0, 1] confidence
    range rather than producing exp(logprob) > 1.0, which would fail ProbeResponse's
    Field(ge=0.0, le=1.0) validation."""
    token_logprobs = [{"token": "A", "logprob": 0.5, "top_logprobs": {"A": 0.5}}]
    assert extract_p_true(token_logprobs) == 1.0


async def test_estimate_confidence_p_true_two_stage_flow() -> None:
    """Stage 1 generates a free-form verdict; stage 2 extracts P(True) from top_logprobs."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = [
        "この質問は医療分野に該当します．",  # stage 1: free-form verdict (plain str, no logprobs requested)
        {
            "content": "A",
            "token_logprobs": [
                {"token": "A", "logprob": -0.05, "top_logprobs": {"A": -0.05, "B": -3.0}}
            ],
        },
    ]

    confidence, raw_p_true = await estimate_confidence_p_true(
        ollama_client, "light-model", "medical", "headache", timeout_s=2.0
    )
    assert confidence == pytest.approx(math.exp(-0.05))
    assert raw_p_true == confidence
    assert ollama_client.generate.await_count == 2


async def test_estimate_confidence_p_true_falls_back_when_top_logprobs_unavailable() -> None:
    """A pre-v0.12.11 ollama that ignores top_logprobs falls back to numeric_scalar self_report."""
    ollama_client = AsyncMock(spec=OllamaClient)
    ollama_client.generate.side_effect = [
        "この質問は医療分野に該当します．",  # stage 1
        {"content": "A", "token_logprobs": None},  # stage 2: no logprobs support
        '{"confidence": 0.6}',  # fallback estimate_confidence call
    ]

    confidence, raw_p_true = await estimate_confidence_p_true(
        ollama_client, "light-model", "medical", "headache", timeout_s=2.0
    )
    assert confidence == 0.6
    assert raw_p_true is None
    assert ollama_client.generate.await_count == 3

"""Tests for the STP sign-flip offline verification (E2, p0001 F3)."""

from scripts.verify_stp_sign_flip import recompute_selected_domain


def test_recompute_selected_domain_argmax_matches_recorded_selection() -> None:
    """argmax(confidence) picks the highest-confidence candidate's domain."""
    candidates = [
        {"domain": "medical", "confidence": 0.5},
        {"domain": "legal", "confidence": 0.9},
        {"domain": "general", "confidence": 0.3},
    ]
    assert recompute_selected_domain(candidates, "confidence", "max") == "legal"


def test_recompute_selected_domain_argmin_flips_choice() -> None:
    """argmin(confidence) picks the lowest-confidence candidate's domain."""
    candidates = [
        {"domain": "medical", "confidence": 0.5},
        {"domain": "legal", "confidence": 0.9},
        {"domain": "general", "confidence": 0.3},
    ]
    assert recompute_selected_domain(candidates, "confidence", "min") == "general"


def test_recompute_selected_domain_handles_empty_candidates() -> None:
    """An empty candidate list returns None instead of raising."""
    assert recompute_selected_domain([], "confidence", "max") is None


def test_recompute_selected_domain_skips_candidates_with_none_key_field() -> None:
    """A candidate missing confidence_logprobs_mean (None) is excluded, not treated as 0."""
    candidates = [
        {"domain": "medical", "confidence_logprobs_mean": None},
        {"domain": "legal", "confidence_logprobs_mean": -0.5},
    ]
    assert recompute_selected_domain(candidates, "confidence_logprobs_mean", "max") == "legal"


def test_recompute_selected_domain_all_none_key_field_returns_none() -> None:
    """If every candidate's key_field is None, there is nothing to select from."""
    candidates = [{"domain": "medical", "confidence_logprobs_mean": None}]
    assert recompute_selected_domain(candidates, "confidence_logprobs_mean", "max") is None

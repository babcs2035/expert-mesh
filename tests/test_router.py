"""Tests for the pure-function parts of confidence scoring in router.py."""

from router import PARSE_FAILURE_CONFIDENCE, build_confidence_prompt, parse_confidence


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

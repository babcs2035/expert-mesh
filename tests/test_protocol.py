"""Tests for Pydantic schema validation, aliases, and boundary values."""

import pytest
from pydantic import ValidationError

from protocol import AdvertiseRequest, ErrorResponse, ProbeRequest, ProbeResponse


def test_probe_request_accepts_from_alias() -> None:
    """Accept 'from' as the JSON key (mapped to Python attribute from_)."""
    request = ProbeRequest.model_validate(
        {
            "request_id": "uuid-1",
            "query_summary": "headache and fever",
            "query_embedding": [0.1, 0.2],
            "from": "laptop-A",
        }
    )
    assert request.from_ == "laptop-A"


def test_probe_request_accepts_field_name_via_populate_by_name() -> None:
    """Also accept the Python attribute name 'from_' when constructing directly."""
    request = ProbeRequest(
        request_id="uuid-1", query_summary="headache", query_embedding=[0.1], from_="laptop-A"
    )
    assert request.from_ == "laptop-A"


def test_probe_response_rejects_confidence_above_one() -> None:
    """Reject confidence values exceeding 1.0."""
    with pytest.raises(ValidationError):
        ProbeResponse(
            request_id="uuid-1", node_id="laptop-B", confidence=1.5, estimated_latency_ms=100
        )


def test_probe_response_rejects_negative_confidence() -> None:
    """Reject negative confidence values."""
    with pytest.raises(ValidationError):
        ProbeResponse(
            request_id="uuid-1", node_id="laptop-B", confidence=-0.1, estimated_latency_ms=100
        )


def test_advertise_request_rejects_load_out_of_range() -> None:
    """Reject load values outside the 0.0-1.0 range."""
    with pytest.raises(ValidationError):
        AdvertiseRequest(
            node_id="laptop-B", domain="medical", domain_embedding=[0.1], load=1.2, timestamp=0
        )


def test_error_response_serializes_to_error_key() -> None:
    """Serialize to the standard {"error": "..."} format."""
    response = ErrorResponse(error="model not ready")
    assert response.model_dump() == {"error": "model not ready"}

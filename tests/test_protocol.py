"""protocol.pyのpydanticスキーマの境界値・エイリアス動作を検証する．"""

import pytest
from pydantic import ValidationError

from protocol import AdvertiseRequest, ErrorResponse, ProbeRequest, ProbeResponse


def test_probe_request_accepts_from_alias() -> None:
    """/probeのリクエストボディでは予約語"from"というキー名をエイリアスとして受け付ける．"""
    request = ProbeRequest.model_validate(
        {
            "request_id": "uuid-1",
            "query_summary": "頭痛と発熱",
            "query_embedding": [0.1, 0.2],
            "from": "laptop-A",
        }
    )
    assert request.from_ == "laptop-A"


def test_probe_request_accepts_field_name_via_populate_by_name() -> None:
    """populate_by_name=Trueにより，Python側の属性名from_でも構築できる．"""
    request = ProbeRequest(
        request_id="uuid-1", query_summary="頭痛", query_embedding=[0.1], from_="laptop-A"
    )
    assert request.from_ == "laptop-A"


def test_probe_response_rejects_confidence_above_one() -> None:
    """confidenceは0.0〜1.0の範囲外を許容しない．"""
    with pytest.raises(ValidationError):
        ProbeResponse(
            request_id="uuid-1", node_id="laptop-B", confidence=1.5, estimated_latency_ms=100
        )


def test_probe_response_rejects_negative_confidence() -> None:
    """confidenceの下限0.0を下回る値は拒否される．"""
    with pytest.raises(ValidationError):
        ProbeResponse(
            request_id="uuid-1", node_id="laptop-B", confidence=-0.1, estimated_latency_ms=100
        )


def test_advertise_request_rejects_load_out_of_range() -> None:
    """loadは0.0〜1.0の範囲でなければならない．"""
    with pytest.raises(ValidationError):
        AdvertiseRequest(
            node_id="laptop-B", domain="medical", domain_embedding=[0.1], load=1.2, timestamp=0
        )


def test_error_response_serializes_to_error_key() -> None:
    """ErrorResponseは{"error": "..."}形式にシリアライズされる．"""
    response = ErrorResponse(error="model not ready")
    assert response.model_dump() == {"error": "model not ready"}

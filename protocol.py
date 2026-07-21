"""Pydantic models for inter-node communication over HTTP."""

from pydantic import BaseModel, Field


class AdvertiseRequest(BaseModel):
    """Heartbeat payload: a node announces its identity, domain, and current load."""

    node_id: str
    domain: str
    domain_embedding: list[float]
    load: float = Field(ge=0.0, le=1.0)
    timestamp: int = Field(description="UNIX epoch seconds")


class AdvertiseResponse(BaseModel):
    """Acknowledgment for a successful /advertise call."""

    status: str = "ok"


class ProbeRequest(BaseModel):
    """Question sent to all peers to ask whether they can handle it."""

    request_id: str
    query_summary: str
    query_embedding: list[float]
    from_: str = Field(alias="from", description="Source node ID")

    model_config = {"populate_by_name": True}


class ProbeResponse(BaseModel):
    """Peer's answer to a probe: self-reported confidence and estimated latency."""

    request_id: str
    node_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_latency_ms: int
    confidence_logprobs_mean: float | None = None  # STP mean logprob signal (null when not used)


class DispatchRequest(BaseModel):
    """Request to the selected peer to generate the full answer."""

    request_id: str
    full_query: str


class DispatchResponse(BaseModel):
    """Result from the selected peer after answer generation."""

    request_id: str
    node_id: str
    answer_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    gen_time_ms: int


class ErrorResponse(BaseModel):
    """Standardized error payload used for 400, 503, and 504 responses."""

    error: str

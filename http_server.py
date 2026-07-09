"""FastAPI server exposing the inter-node protocol endpoints."""

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from expert_backend import OllamaClient
from http_client import PeerClient
from logging_utils import LOG_LEVEL_ERROR, LOG_LEVEL_INFO, log_event
from protocol import (
    AdvertiseRequest,
    AdvertiseResponse,
    DispatchRequest,
    DispatchResponse,
    ErrorResponse,
    ProbeRequest,
    ProbeResponse,
)
from router import estimate_confidence, estimate_embedding_confidence

# Maximum tokens for full answer generation. Without an explicit cap,
# ollama generates indefinitely. 512 tokens is sufficient for typical
# answers and fits within the configured dispatch timeout.
DISPATCH_MAX_TOKENS = 512

# Startup warmup settings. Pre-loading the model prevents its latency
# from contaminating probe and dispatch timing measurements.
WARMUP_MAX_TOKENS = 1
WARMUP_TIMEOUT_S = 120.0
WARMUP_RETRY_INTERVAL_S = 2.0
WARMUP_MAX_RETRIES = 30

# Heartbeat cadence for the /advertise background task. The design doc
# (2.3) only requires "low frequency" since Phase 0 does not yet act on
# the advertised load value; 30s keeps known_peers reasonably fresh
# without adding meaningful network load on the LAN.
ADVERTISE_INTERVAL_S = 30.0
# Timeout for a single /advertise call. Unlike /probe and /dispatch this
# never touches an LLM, so a short timeout is sufficient.
ADVERTISE_TIMEOUT_S = 5.0
# Placeholder self-reported load. Phase 0 does not implement any resource
# monitoring, and no dispatch-selection logic currently reads this value;
# a constant keeps the AdvertiseRequest schema satisfied without inventing
# an unused metric. Real load reporting is left to a later phase.
PLACEHOLDER_LOAD = 0.0
# Routing method identifiers (design doc 2.4: method A = embedding-based,
# method B = self-reported score). Kept as plain strings rather than an
# enum since config.yaml stores them as YAML strings and NodeState only
# ever compares them once per /probe call.
ROUTING_METHOD_SELF_REPORT = "self_report"
ROUTING_METHOD_EMBEDDING = "embedding"


def build_dispatch_prompt(domain: str, full_query: str) -> str:
    """Build the answer-generation prompt with the node's domain context."""
    return f"あなたは「{domain}」分野の専門家です．次の質問に，あなたの専門知識を活かして具体的に回答してください．\n質問: {full_query}"


async def warmup_model(ollama_client: OllamaClient, model: str) -> None:
    """Load a model into ollama memory on startup."""
    for attempt in range(WARMUP_MAX_RETRIES):
        try:
            await ollama_client.generate(
                model, "こんにちは", timeout_s=WARMUP_TIMEOUT_S, max_tokens=WARMUP_MAX_TOKENS
            )
            return
        except httpx.HTTPError:
            if attempt == WARMUP_MAX_RETRIES - 1:
                raise
            await asyncio.sleep(WARMUP_RETRY_INTERVAL_S)


class NodeState:
    """Mutable runtime state and configuration for a single mesh node."""

    def __init__(
        self,
        node_id: str,
        domain: str,
        light_model: str,
        expert_model: str,
        confidence_threshold: float,
        probe_timeout_s: float,
        dispatch_timeout_s: float,
        ollama_client: OllamaClient,
        peers: list[dict] | None = None,
        embedding_model: str | None = None,
        routing_method: str = ROUTING_METHOD_SELF_REPORT,
    ) -> None:
        self.node_id = node_id
        self.domain = domain
        self.light_model = light_model
        self.expert_model = expert_model
        self.confidence_threshold = confidence_threshold
        self.probe_timeout_s = probe_timeout_s
        self.dispatch_timeout_s = dispatch_timeout_s
        self.ollama_client = ollama_client
        # peers/embedding_model are optional because existing tests build a
        # NodeState with no peer list at all (advertise heartbeat and
        # embedding routing are both no-ops without them).
        self.peers = peers or []
        self.embedding_model = embedding_model
        self.routing_method = routing_method
        self.domain_embedding: list[float] = []
        self.known_peers: dict[str, AdvertiseRequest] = {}
        self._probe_confidence_cache: dict[str, float] = {}

    def cache_probe_confidence(self, request_id: str, confidence: float) -> None:
        """Store the confidence score from probe for later reuse in dispatch."""
        self._probe_confidence_cache[request_id] = confidence

    def pop_probe_confidence(self, request_id: str) -> float:
        """Retrieve and remove the cached probe confidence; returns 0.0 if absent."""
        return self._probe_confidence_cache.pop(request_id, 0.0)


async def _advertise_loop(state: NodeState, peer_client: PeerClient) -> None:
    """Periodically broadcast this node's own AdvertiseRequest to all peers.

    Runs until cancelled by the lifespan shutdown. A fresh timestamp is
    attached on every iteration so recipients can detect a stale peer by
    comparing it against wall-clock time (not yet consumed by any endpoint
    in Phase 0, but required by the AdvertiseRequest schema).
    """
    while True:
        request = AdvertiseRequest(
            node_id=state.node_id,
            domain=state.domain,
            domain_embedding=state.domain_embedding,
            load=PLACEHOLDER_LOAD,
            timestamp=int(time.time()),
        )
        await peer_client.advertise_all(request, timeout_s=ADVERTISE_TIMEOUT_S)
        await asyncio.sleep(ADVERTISE_INTERVAL_S)


def create_app(state: NodeState) -> FastAPI:
    """Create and wire up the FastAPI application with the given node state."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Warm up models, then start the advertise heartbeat, before serving."""
        await warmup_model(state.ollama_client, state.light_model)
        if state.expert_model != state.light_model:
            await warmup_model(state.ollama_client, state.expert_model)
        if state.embedding_model is not None:
            state.domain_embedding = await state.ollama_client.embed(
                state.embedding_model, state.domain
            )

        advertise_task = None
        if state.peers:
            peer_client = PeerClient(state.peers)
            advertise_task = asyncio.create_task(_advertise_loop(state, peer_client))
        try:
            yield
        finally:
            if advertise_task is not None:
                advertise_task.cancel()

    app = FastAPI(title=f"encounter-expert-mesh:{state.node_id}", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return a standardized 400 error for malformed requests."""
        return JSONResponse(status_code=400, content=ErrorResponse(error="invalid request").model_dump())

    @app.post("/advertise", response_model=AdvertiseResponse)
    async def advertise(body: AdvertiseRequest) -> AdvertiseResponse:
        """Record or update a peer's advertised state."""
        state.known_peers[body.node_id] = body
        log_event(state.node_id, LOG_LEVEL_INFO, "advertise_received", from_node=body.node_id)
        return AdvertiseResponse()

    @app.post("/probe", response_model=None)
    async def probe(body: ProbeRequest) -> ProbeResponse | JSONResponse:
        """Score whether this node can handle the query and return the confidence."""
        received_at = time.time()
        start = time.monotonic()
        try:
            if state.routing_method == ROUTING_METHOD_EMBEDDING:
                confidence = estimate_embedding_confidence(
                    body.query_embedding, state.domain_embedding
                )
            else:
                confidence = await estimate_confidence(
                    state.ollama_client,
                    state.light_model,
                    state.domain,
                    body.query_summary,
                    timeout_s=state.probe_timeout_s,
                )
        except httpx.TimeoutException:
            log_event(
                state.node_id, LOG_LEVEL_ERROR, "probe_timeout", request_id=body.request_id
            )
            return JSONResponse(status_code=504, content=ErrorResponse(error="timeout").model_dump())
        except httpx.HTTPError as exc:
            log_event(
                state.node_id,
                LOG_LEVEL_ERROR,
                "probe_model_not_ready",
                request_id=body.request_id,
                error=str(exc),
            )
            return JSONResponse(
                status_code=503, content=ErrorResponse(error="model not ready").model_dump()
            )
        estimated_latency_ms = int((time.monotonic() - start) * 1000)
        state.cache_probe_confidence(body.request_id, confidence)
        log_event(
            state.node_id,
            LOG_LEVEL_INFO,
            "probe_done",
            request_id=body.request_id,
            from_node=body.from_,
            routing_method=state.routing_method,
            confidence=confidence,
            received_at_unix_time_s=received_at,
            local_inference_ms=estimated_latency_ms,
        )
        return ProbeResponse(
            request_id=body.request_id,
            node_id=state.node_id,
            confidence=confidence,
            estimated_latency_ms=estimated_latency_ms,
        )

    @app.post("/dispatch", response_model=None)
    async def dispatch(body: DispatchRequest) -> DispatchResponse | JSONResponse:
        """Generate the full answer and return it with timing and confidence."""
        received_at = time.time()
        start = time.monotonic()
        try:
            answer_text = await state.ollama_client.generate(
                state.expert_model,
                build_dispatch_prompt(state.domain, body.full_query),
                timeout_s=state.dispatch_timeout_s,
                max_tokens=DISPATCH_MAX_TOKENS,
            )
        except httpx.TimeoutException:
            log_event(
                state.node_id, LOG_LEVEL_ERROR, "dispatch_timeout", request_id=body.request_id
            )
            return JSONResponse(status_code=504, content=ErrorResponse(error="timeout").model_dump())
        except httpx.HTTPError as exc:
            log_event(
                state.node_id,
                LOG_LEVEL_ERROR,
                "dispatch_model_not_ready",
                request_id=body.request_id,
                error=str(exc),
            )
            return JSONResponse(
                status_code=503, content=ErrorResponse(error="model not ready").model_dump()
            )
        gen_time_ms = int((time.monotonic() - start) * 1000)
        log_event(
            state.node_id,
            LOG_LEVEL_INFO,
            "dispatch_done",
            request_id=body.request_id,
            received_at_unix_time_s=received_at,
            local_inference_ms=gen_time_ms,
        )
        return DispatchResponse(
            request_id=body.request_id,
            node_id=state.node_id,
            answer_text=answer_text,
            confidence=state.pop_probe_confidence(body.request_id),
            gen_time_ms=gen_time_ms,
        )

    return app

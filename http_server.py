"""FastAPI server exposing the inter-node protocol endpoints."""

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from expert_backend import OllamaClient
from protocol import (
    AdvertiseRequest,
    AdvertiseResponse,
    DispatchRequest,
    DispatchResponse,
    ErrorResponse,
    ProbeRequest,
    ProbeResponse,
)
from router import estimate_confidence

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
    ) -> None:
        self.node_id = node_id
        self.domain = domain
        self.light_model = light_model
        self.expert_model = expert_model
        self.confidence_threshold = confidence_threshold
        self.probe_timeout_s = probe_timeout_s
        self.dispatch_timeout_s = dispatch_timeout_s
        self.ollama_client = ollama_client
        self.known_peers: dict[str, AdvertiseRequest] = {}
        self._probe_confidence_cache: dict[str, float] = {}

    def cache_probe_confidence(self, request_id: str, confidence: float) -> None:
        """Store the confidence score from probe for later reuse in dispatch."""
        self._probe_confidence_cache[request_id] = confidence

    def pop_probe_confidence(self, request_id: str) -> float:
        """Retrieve and remove the cached probe confidence; returns 0.0 if absent."""
        return self._probe_confidence_cache.pop(request_id, 0.0)


def create_app(state: NodeState) -> FastAPI:
    """Create and wire up the FastAPI application with the given node state."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Warm up models before the server starts accepting requests."""
        await warmup_model(state.ollama_client, state.light_model)
        if state.expert_model != state.light_model:
            await warmup_model(state.ollama_client, state.expert_model)
        yield

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
        return AdvertiseResponse()

    @app.post("/probe", response_model=None)
    async def probe(body: ProbeRequest) -> ProbeResponse | JSONResponse:
        """Score whether this node can handle the query and return the confidence."""
        start = time.monotonic()
        try:
            confidence = await estimate_confidence(
                state.ollama_client,
                state.light_model,
                state.domain,
                body.query_summary,
                timeout_s=state.probe_timeout_s,
            )
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content=ErrorResponse(error="timeout").model_dump())
        except httpx.HTTPError:
            return JSONResponse(
                status_code=503, content=ErrorResponse(error="model not ready").model_dump()
            )
        estimated_latency_ms = int((time.monotonic() - start) * 1000)
        state.cache_probe_confidence(body.request_id, confidence)
        return ProbeResponse(
            request_id=body.request_id,
            node_id=state.node_id,
            confidence=confidence,
            estimated_latency_ms=estimated_latency_ms,
        )

    @app.post("/dispatch", response_model=None)
    async def dispatch(body: DispatchRequest) -> DispatchResponse | JSONResponse:
        """Generate the full answer and return it with timing and confidence."""
        start = time.monotonic()
        try:
            answer_text = await state.ollama_client.generate(
                state.expert_model,
                build_dispatch_prompt(state.domain, body.full_query),
                timeout_s=state.dispatch_timeout_s,
                max_tokens=DISPATCH_MAX_TOKENS,
            )
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content=ErrorResponse(error="timeout").model_dump())
        except httpx.HTTPError:
            return JSONResponse(
                status_code=503, content=ErrorResponse(error="model not ready").model_dump()
            )
        gen_time_ms = int((time.monotonic() - start) * 1000)
        return DispatchResponse(
            request_id=body.request_id,
            node_id=state.node_id,
            answer_text=answer_text,
            confidence=state.pop_probe_confidence(body.request_id),
            gen_time_ms=gen_time_ms,
        )

    return app

"""FastAPIによるノード間通信サーバー：/advertise, /probe, /dispatch を提供する．"""

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

# 専門家モデルの本回答生成トークン数の上限．num_predict未指定だとollamaは無制限に
# 生成し続けるため，dispatch_timeout_s内に収まる現実的な上限を明示的に課す．
# qwen3.5:9bの実測生成速度（約3 tok/s，CPU推論）を踏まえ，
# config.yamlのdispatch_timeout_s内に収まる値としている．
DISPATCH_MAX_TOKENS = 512

# 起動時ウォームアップ関連の定数．モデルロード時間がprobe/dispatchの計測値に
# 混入しないよう，起動時に一度モデルをollamaのメモリへ読み込ませておく．
WARMUP_MAX_TOKENS = 1
WARMUP_TIMEOUT_S = 120.0
WARMUP_RETRY_INTERVAL_S = 2.0
WARMUP_MAX_RETRIES = 30


async def warmup_model(ollama_client: OllamaClient, model: str) -> None:
    """モデルをollamaのメモリに読み込ませる．

    コンテナ起動直後はollama自体のAPIがまだ受け付けられないことがあるため，
    接続エラー時は一定回数リトライする．
    """
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
    """1ノード分の設定（config.yaml由来）と実行時状態を保持する．"""

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
        """probe時に算出したconfidenceを，対応するdispatch応答で再利用するため保持する．"""
        self._probe_confidence_cache[request_id] = confidence

    def pop_probe_confidence(self, request_id: str) -> float:
        """dispatch時にprobe時のconfidenceを取り出す．対応するprobeがなければ0.0とする．"""
        return self._probe_confidence_cache.pop(request_id, 0.0)


def create_app(state: NodeState) -> FastAPI:
    """NodeStateを束縛したFastAPIアプリケーションを構築する．"""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """起動時にlight_model/expert_modelをウォームアップする．"""
        await warmup_model(state.ollama_client, state.light_model)
        if state.expert_model != state.light_model:
            await warmup_model(state.ollama_client, state.expert_model)
        yield

    app = FastAPI(title=f"encounter-expert-mesh:{state.node_id}", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """JSON形式不正等のリクエストバリデーションエラーを400+共通エラー形式に変換する．"""
        return JSONResponse(status_code=400, content=ErrorResponse(error="invalid request").model_dump())

    @app.post("/advertise", response_model=AdvertiseResponse)
    async def advertise(body: AdvertiseRequest) -> AdvertiseResponse:
        """ピアからのハートビートを受け取り，既知ピア情報を更新する．"""
        state.known_peers[body.node_id] = body
        return AdvertiseResponse()

    @app.post("/probe", response_model=None)
    async def probe(body: ProbeRequest) -> ProbeResponse | JSONResponse:
        """軽量モデルで担当可否confidenceを算出して返す．"""
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
        """専門家モデルで本文の回答を生成して返す．"""
        start = time.monotonic()
        try:
            answer_text = await state.ollama_client.generate(
                state.expert_model,
                body.full_query,
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

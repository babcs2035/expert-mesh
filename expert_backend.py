"""Async client for the ollama inference and embedding APIs."""

import asyncio
import os

import httpx

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RETRIES = 3
RETRY_DELAY_S = 15.0


class OllamaClient:
    """Communicate with an ollama server for generation and embeddings.

    Uses the provided host when given, otherwise reads OLLAMA_HOST from
    the environment (configured by docker-compose for same-host access).
    Retries failed requests on transient connection errors.
    """

    def __init__(self, host: str | None = None) -> None:
        self._host = host or os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

    async def generate(
        self,
        model: str,
        prompt: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a non-streaming response using the chat endpoint.

        Uses /api/chat instead of /api/generate because thinking models
        (qwen3.5, etc.) ignore the think: false flag on the generate
        endpoint — a known ollama issue (#14793). This can exhaust the
        token budget on internal reasoning and leave the final answer
        empty.

        max_tokens maps to ollama's num_predict option. Without it,
        generation is unlimited and models may loop on short-output
        prompts. Always set a reasonable cap.

        temperature controls output randomness. The default (~0.8) is
        too high for deterministic tasks like confidence scoring.

        Retries up to DEFAULT_RETRIES times on transient connection errors.
        """
        options: dict = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
        }
        if options:
            payload["options"] = options
        for attempt in range(DEFAULT_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(f"{self._host}/api/chat", json=payload)
                    response.raise_for_status()
                    return response.json()["message"]["content"]
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt < DEFAULT_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_S)
                else:
                    raise

    async def get_running_models(self, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[dict]:
        """Return ollama's /api/ps list of currently loaded models.

        Each entry includes size_vram (bytes resident on GPU); size_vram of
        0 means the model is running CPU-only. Used to verify GPU
        utilization after warmup (see http_server.py's log_gpu_status).
        Not retried like generate/embed since it is a best-effort
        diagnostic call, not one on the request-serving path.
        """
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(f"{self._host}/api/ps")
            response.raise_for_status()
            return response.json().get("models", [])

    async def embed(self, model: str, text: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[float]:
        """Return the embedding vector for a text string.

        Retries up to DEFAULT_RETRIES times on transient connection errors.
        """
        for attempt in range(DEFAULT_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(
                        f"{self._host}/api/embeddings",
                        json={"model": model, "prompt": text},
                    )
                    response.raise_for_status()
                    return response.json()["embedding"]
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.NetworkError, httpx.RemoteProtocolError):
                if attempt < DEFAULT_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_S)
                else:
                    raise

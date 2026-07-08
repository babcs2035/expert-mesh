"""Async client for the ollama inference and embedding APIs."""

import os

import httpx

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 30.0


class OllamaClient:
    """Communicate with an ollama server for generation and embeddings.

    Uses the provided host when given, otherwise reads OLLAMA_HOST from
    the environment (configured by docker-compose for same-host access).
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
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(f"{self._host}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]

    async def embed(self, model: str, text: str, timeout_s: float = DEFAULT_TIMEOUT_S) -> list[float]:
        """Return the embedding vector for a text string."""
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{self._host}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

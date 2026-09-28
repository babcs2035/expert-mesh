"""Async client for the ollama inference and embedding APIs."""

import asyncio
import os

import httpx

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RETRIES = 3
RETRY_DELAY_S = 15.0
# Iter99 (embedding_model_replacement): default embedding_prompt_template value,
# byte-for-byte identical to the pre-Iter99 hardcoded Qwen3-Embedding instruct
# format, so callers that omit config["embedding_prompt_template"] keep the
# exact pre-Iter99 prompt. See config.yaml's embedding_prompt_template comment.
DEFAULT_PROMPT_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"


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
        logprobs: int | None = None,
        top_logprobs: int | None = None,
    ) -> str | dict:
        """Generate text with optional token-level logprobs.

        When logprobs is set (> 0), uses /api/chat endpoint with logprobs:true
        for token probability extraction (this endpoint returns logprobs for the
        models used in this project, whereas /api/generate does not). Otherwise
        uses /api/chat for thinking-model compatibility.

        top_logprobs additionally requests N alternative-token candidates per
        position (Ollama v0.12.11+; see docs.ollama.com/api/chat), needed by
        router.py's extract_p_true to read P("A") even when "A" wasn't the
        actually-generated top-1 token. Ignored when logprobs is not set.

        Returns a string (content only) when logprobs is not requested, or a
        dict with 'content' (str) and optionally 'token_logprobs' (list[dict]
        with 'token', 'logprob', and — only when top_logprobs was requested —
        'top_logprobs': dict[str, float] mapping alternative tokens to their
        logprob) when logprobs is set.

        Retries up to DEFAULT_RETRIES times on transient connection errors.
        """
        options: dict = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature

        if logprobs and logprobs > 0:
            # Use /api/chat for logprobs support — this endpoint returns
            # token-level logprobs for the models used in this project,
            # whereas /api/generate does not (even with logprobs:true).
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "logprobs": True,
            }
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
            if options:
                payload["options"] = options
        else:
            # Use /api/chat for thinking-model compatibility
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
            }
            if options:
                payload["options"] = options

        for attempt in range(DEFAULT_RETRIES):
            try:
                endpoint = "/api/chat"
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(f"{self._host}{endpoint}", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    if logprobs and logprobs > 0:
                        # /api/chat returns content under message.content
                        # and logprobs under the 'logprobs' key (list of {token, logprob, bytes})
                        content = data.get("message", {}).get("content", "")
                        raw_logprobs = data.get("logprobs")  # list[dict] with token/logprob/bytes
                        token_logprobs: list[dict] | None = None
                        if raw_logprobs:
                            token_logprobs = [
                                {"token": entry["token"], "logprob": entry["logprob"]}
                                for entry in raw_logprobs
                            ]
                            if top_logprobs is not None:
                                for position, raw_entry in zip(
                                    token_logprobs, raw_logprobs, strict=True
                                ):
                                    position["top_logprobs"] = {
                                        alt["token"]: alt["logprob"]
                                        for alt in raw_entry.get("top_logprobs", [])
                                    }
                        return {"content": content, "token_logprobs": token_logprobs}
                    return data["message"]["content"]
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ):
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

    async def embed(
        self,
        model: str,
        text: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        instruction: str | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> list[float]:
        """Return the embedding vector for a text string.

        Retries up to DEFAULT_RETRIES times on transient connection errors.

        If `instruction` is given, the prompt is built from `prompt_template`
        (formatted with `instruction` and `text`) before being sent to Ollama,
        which does not add such prefixes on its own (see journal.md Iteration
        81 Q3). `prompt_template` defaults to the pre-Iter99 Qwen3-Embedding
        instruct format (`Instruct: {instruction}\\nQuery: {text}`); Iter99
        (embedding_model_replacement) made this a parameter so a model-specific
        prefix scheme (e.g. ruri-v3-310m's "トピック: {text}", which ignores the
        {instruction} placeholder) can be supplied via
        config["embedding_prompt_template"] without touching this hardcoded
        default. `instruction` defaults to None so existing callers that do
        not pass this argument keep the current unprefixed behavior.
        """
        prompt = prompt_template.format(instruction=instruction, text=text) if instruction else text
        for attempt in range(DEFAULT_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    response = await client.post(
                        f"{self._host}/api/embeddings",
                        json={"model": model, "prompt": prompt},
                    )
                    response.raise_for_status()
                    return response.json()["embedding"]
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ):
                if attempt < DEFAULT_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_S)
                else:
                    raise


async def embed_query_views(
    client: "OllamaClient",
    model: str,
    text: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    instruction: str | None = None,
    concat_views: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    fusion_models: list[dict] | None = None,
) -> list[float]:
    """Return the query embedding, optionally as a concatenation of two views.

    Iter82 (embedding_view_concatenation): when `concat_views` is False, this
    is equivalent to a single `client.embed(model, text, instruction=instruction)`
    call, preserving the pre-Iter82 behavior exactly. When `concat_views` is
    True, this embeds `text` twice — once without any instruction (the
    "prefix なし" view) and once with `instruction` (the "prefix あり" view,
    Iter81's selected P1 wording) — and concatenates the two vectors into a
    single vector, ORDER FIXED as [plain, instructed]. `prompt_template`
    (Iter99) governs how the instructed view's prompt is built; see
    `OllamaClient.embed`'s docstring. It has no effect on the plain view,
    which is always sent as raw text.

    Iter100 (embedding_space_fusion): `fusion_models`, if given, is a list of
    `{"model": ..., "prompt_template": ...}` dicts (config.yaml's
    `embedding_fusion_models`). Each entry's own [plain, instructed] view pair
    is embedded the same way as the primary model above (same shared
    `instruction`, but the entry's own `prompt_template` for its instructed
    view) and appended, in list order, after the primary model's two views —
    giving the fixed block order [primary plain, primary instructed,
    fusion[0] plain, fusion[0] instructed, fusion[1] plain, fusion[1]
    instructed, ...]. `fusion_models` being None or an empty list (the
    pre-Iter100 default) returns exactly the pre-Iter100 vector, unchanged.

    The order is intentionally not a parameter: both the training path
    (scripts/train_domain_classifier.py) and the runtime path (node.py) call
    this same function, so hard-coding the order here is what guarantees they
    stay consistent (see journal.md "Iteration 82" Q3 / backlog B129 (1) —
    an Iter36-style train/eval mismatch would otherwise pass silently).

    Raises ValueError if `concat_views` is True but `instruction` is None,
    since that combination indicates a config mistake (concatenating a view
    with itself) rather than a valid feature-view spec. Also raises
    ValueError if `fusion_models` is given while `concat_views` is False,
    since fusion appends [plain, instructed] block pairs and has no matching
    structure to attach to a single-view primary embedding.
    """
    if not concat_views:
        if fusion_models:
            raise ValueError(
                "embed_query_views: fusion_models requires concat_views=True "
                "(fusion appends [plain, instructed] block pairs onto the "
                "primary model's own two views)"
            )
        return await client.embed(
            model, text, timeout_s=timeout_s, instruction=instruction, prompt_template=prompt_template
        )
    if instruction is None:
        raise ValueError(
            "embed_query_views: concat_views=True requires a non-None instruction "
            "(embedding_view_concat and embedding_instruction must both be set)"
        )
    plain_view = await client.embed(model, text, timeout_s=timeout_s, instruction=None)
    instructed_view = await client.embed(
        model, text, timeout_s=timeout_s, instruction=instruction, prompt_template=prompt_template
    )
    views = [*plain_view, *instructed_view]
    for fusion_spec in fusion_models or []:
        fusion_model = fusion_spec["model"]
        fusion_prompt_template = fusion_spec["prompt_template"]
        fusion_plain_view = await client.embed(fusion_model, text, timeout_s=timeout_s, instruction=None)
        fusion_instructed_view = await client.embed(
            fusion_model, text, timeout_s=timeout_s, instruction=instruction,
            prompt_template=fusion_prompt_template,
        )
        views.extend(fusion_plain_view)
        views.extend(fusion_instructed_view)
    return views

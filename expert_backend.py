"""ollamaのHTTP API（/api/chat, /api/embeddings）を呼び出す非同期クライアント．"""

import os

import httpx

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 30.0


class OllamaClient:
    """ollamaサーバーへの推論・埋め込みリクエストを行うクライアント．

    接続先はコンストラクタ引数を優先し，未指定の場合は環境変数OLLAMA_HOST
    （docker-composeが同一ホスト内のollamaサービスを指すために設定する）を用いる．
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
        """指定モデルでプロンプトに対する非ストリーミング応答を返す．

        /api/chat（/api/generateではない）をthink: falseで呼び出す．qwen3.5等の
        thinkingモデルは/api/generateではthink: falseが無視されるバグがあり
        （ollama/ollama#14793），内部思考過程の出力にnum_predictを使い切って
        最終応答が空になることがあるため，/api/chatを用いる．

        max_tokensはollamaのoptions.num_predictに渡す生成トークン数の上限．
        ollamaは指定しない場合num_predict=-1（無制限）で動作し，モデルが指示に
        従わずJSON等の短い出力を求めるプロンプトでも延々と生成を続けることが
        あるため，用途に応じた上限を必ず指定する．
        temperatureはollamaのデフォルト(0.8前後)だと同一プロンプトでも出力が
        大きく揺れるため，confidence算出のような決定性を要する用途で下げて使う．
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
        """指定モデルでテキストの埋め込みベクトルを返す．"""
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{self._host}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

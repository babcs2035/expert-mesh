"""他ノードへのHTTP POST（/probe, /dispatch）を行う非同期クライアント．"""

import asyncio

import httpx

from protocol import DispatchRequest, DispatchResponse, ProbeRequest, ProbeResponse


class PeerClient:
    """peers.yamlに記載された各ピアへHTTP POSTリクエストを送るクライアント．"""

    def __init__(self, peers: list[dict]) -> None:
        """peersはpeers.yamlのnodesリスト（node_id, host, port, domainを持つ辞書）を想定する．"""
        self._peers = peers

    async def probe_all(self, request: ProbeRequest, timeout_s: float) -> list[ProbeResponse]:
        """全ピアへ並行して/probeを送り，制限時間内に応答したものだけを返す．

        peers.yamlに記載された順序でタスクを組み立ててasyncio.gatherに渡すため，
        戻り値の順序は完了順ではなく記載順になる．これはaggregator.pyの
        confidence同点タイブレーク（記載順で先勝ち）の前提となる．
        """
        results = await asyncio.gather(
            *(self._probe_one(peer, request, timeout_s) for peer in self._peers)
        )
        return [r for r in results if r is not None]

    async def _probe_one(
        self, peer: dict, request: ProbeRequest, timeout_s: float
    ) -> ProbeResponse | None:
        """単一ピアへの/probe送信．タイムアウト・接続エラー時はNone（不在扱い）を返す．"""
        url = f"http://{peer['host']}:{peer['port']}/probe"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=request.model_dump(by_alias=True))
                response.raise_for_status()
                return ProbeResponse.model_validate(response.json())
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

    async def dispatch(
        self, peer: dict, request: DispatchRequest, timeout_s: float
    ) -> DispatchResponse | None:
        """選定した1ピアへ/dispatchを送り，回答生成結果を取得する．失敗時はNoneを返す．"""
        url = f"http://{peer['host']}:{peer['port']}/dispatch"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=request.model_dump())
                response.raise_for_status()
                return DispatchResponse.model_validate(response.json())
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

"""Async HTTP client for peer-to-peer /probe and /dispatch calls."""

import asyncio

import httpx

from protocol import DispatchRequest, DispatchResponse, ProbeRequest, ProbeResponse


class PeerClient:
    """Send requests to all peers declared in peers.yaml."""

    def __init__(self, peers: list[dict]) -> None:
        """peers: list of dicts containing node_id, host, port, domain."""
        self._peers = peers

    async def probe_all(self, request: ProbeRequest, timeout_s: float) -> list[ProbeResponse]:
        """Send /probe to every peer concurrently and collect responses within the deadline.

        The tasks are created in peers.yaml order and passed to asyncio.gather,
        so the result list preserves declaration order rather than completion
        order. This ordering is required for deterministic tie-breaking.
        """
        results = await asyncio.gather(
            *(self._probe_one(peer, request, timeout_s) for peer in self._peers)
        )
        return [r for r in results if r is not None]

    async def _probe_one(
        self, peer: dict, request: ProbeRequest, timeout_s: float
    ) -> ProbeResponse | None:
        """Send /probe to a single peer; return None on any failure."""
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
        """Send /dispatch to a single peer and return the answer, or None on failure."""
        url = f"http://{peer['host']}:{peer['port']}/dispatch"
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, json=request.model_dump())
                response.raise_for_status()
                return DispatchResponse.model_validate(response.json())
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

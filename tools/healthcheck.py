"""Send dummy /advertise requests to all configured nodes and report reachability."""

import argparse
import asyncio
import os
import sys

import httpx
import yaml

HEALTHCHECK_TIMEOUT_S = 3.0
SELF_HOST_OVERRIDE = "localhost"


async def check_one(peer: dict) -> tuple[str, bool, str]:
    """Send a dummy /advertise to a single node and return (node_id, success, message)."""
    url = f"http://{peer['host']}:{peer['port']}/advertise"
    dummy_request = {
        "node_id": "healthcheck",
        "domain": "healthcheck",
        "domain_embedding": [],
        "load": 0.0,
        "timestamp": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=HEALTHCHECK_TIMEOUT_S) as client:
            response = await client.post(url, json=dummy_request)
            response.raise_for_status()
            return peer["node_id"], True, "ok"
    except httpx.HTTPError as exc:
        return peer["node_id"], False, str(exc)


def _resolve_peers(config: dict) -> list[dict]:
    """Convert config nodes to peer dicts, overriding host to localhost for the self node."""
    self_node_id = os.environ.get("NODE_ID")
    peers = []
    for node_id, node_config in config["nodes"].items():
        peer = {"node_id": node_id, **node_config}
        if node_id == self_node_id:
            peer = {**peer, "host": SELF_HOST_OVERRIDE}
        peers.append(peer)
    return peers


async def run_healthcheck(config_path: str) -> bool:
    """Check all nodes and print results. Return True only if every node is reachable."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    peers = _resolve_peers(config)
    results = await asyncio.gather(*(check_one(peer) for peer in peers))
    all_ok = True
    for node_id, ok, message in results:
        status = "OK" if ok else "NG"
        print(f"[{status}] {node_id}: {message}")
        all_ok = all_ok and ok
    return all_ok


def main() -> None:
    """CLI entry point: exit 0 if all nodes are healthy, 1 otherwise."""
    parser = argparse.ArgumentParser(description="Health-check all nodes via /advertise")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    all_ok = asyncio.run(run_healthcheck(args.config))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

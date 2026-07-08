"""config.yaml記載の全ノードへ/advertiseを送り，生存確認を行うツール（distributed-llmの移植方針を踏襲）．"""

import argparse
import asyncio
import os
import sys

import httpx
import yaml

HEALTHCHECK_TIMEOUT_S = 3.0
SELF_HOST_OVERRIDE = "localhost"


async def check_one(peer: dict) -> tuple[str, bool, str]:
    """1ノードへ/advertiseのダミーリクエストを送り，(node_id, 成功可否, メッセージ)を返す．

    domain_embedding/timestampは疎通確認のみが目的のダミー値であり，
    実際のノード状態としては扱われない．
    """
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
    """config.yamlのnodesをpeerのリストに変換する．

    自ノード（環境変数NODE_IDと一致するノード）は，外部IP経由でのhairpin NAT
    （コンテナから自ホストの公開ポートへの折り返し接続）が環境によって機能しない
    ことがあるため，hostをlocalhostに置き換える．
    """
    self_node_id = os.environ.get("NODE_ID")
    peers = []
    for node_id, node_config in config["nodes"].items():
        peer = {"node_id": node_id, **node_config}
        if node_id == self_node_id:
            peer = {**peer, "host": SELF_HOST_OVERRIDE}
        peers.append(peer)
    return peers


async def run_healthcheck(config_path: str) -> bool:
    """全ノードの生存確認を実行し，結果を表示する．全ノード成功ならTrueを返す．"""
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
    """CLIエントリポイント．全ノードが生存していない場合は非ゼロで終了する．"""
    parser = argparse.ArgumentParser(description="全ノードの/advertise疎通確認を行う")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    all_ok = asyncio.run(run_healthcheck(args.config))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

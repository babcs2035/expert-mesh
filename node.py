"""各ノードのエントリポイント．expert（serve）・requester（ask）の役割をCLIサブコマンドで切り替える．

Phase 0では，advertiseの定期送信（ハートビート）は設計書上「低頻度でよい・任意」とされて
いるため実装せず，受信側（http_server.pyの/advertiseハンドラ）のみを用意している．
"""

import argparse
import asyncio
import uuid

import uvicorn
import yaml

from aggregator import select_dispatch_targets
from expert_backend import OllamaClient
from http_client import PeerClient
from http_server import NodeState, create_app
from protocol import DispatchRequest, ProbeRequest

QUERY_SUMMARY_MAX_LENGTH = 200  # 軽量モデルへのprobeプロンプトを長大化させないための上限


def load_yaml(path: str) -> dict:
    """YAML設定ファイルを読み込む．"""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_node_state(config: dict, node_id: str) -> NodeState:
    """config.yamlから指定node_id分のNodeStateを構築する．"""
    node_config = config["nodes"][node_id]
    return NodeState(
        node_id=node_id,
        domain=node_config["domain"],
        light_model=node_config["light_model"],
        expert_model=node_config["expert_model"],
        confidence_threshold=config.get("confidence_threshold", 0.5),
        probe_timeout_s=config.get("probe_timeout_s", 2.0),
        dispatch_timeout_s=config.get("dispatch_timeout_s", 30.0),
        ollama_client=OllamaClient(),
    )


def run_serve(args: argparse.Namespace) -> None:
    """FastAPIサーバーを起動し，本ノードをexpertとして待ち受けさせる．"""
    config = load_yaml(args.config)
    state = build_node_state(config, args.node_id)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


async def _ask(args: argparse.Namespace) -> None:
    """requesterとして質問を受け付け，probe→dispatchを実行して回答を表示する．"""
    config = load_yaml(args.config)
    peers = [
        {"node_id": node_id, **node_config}
        for node_id, node_config in config["nodes"].items()
        if node_id != args.node_id
    ]
    ollama_client = OllamaClient()

    request_id = str(uuid.uuid4())
    query_summary = args.query[:QUERY_SUMMARY_MAX_LENGTH]
    query_embedding = await ollama_client.embed(config["embedding_model"], args.query)

    probe_request = ProbeRequest(
        request_id=request_id,
        query_summary=query_summary,
        query_embedding=query_embedding,
        from_=args.node_id,
    )
    peer_client = PeerClient(peers)
    probe_responses = await peer_client.probe_all(
        probe_request, timeout_s=config.get("probe_timeout_s", 2.0)
    )
    targets = select_dispatch_targets(
        probe_responses, confidence_threshold=config.get("confidence_threshold", 0.5)
    )
    if not targets:
        print("担当できる専門家が見つかりませんでした．")
        return

    chosen = targets[0]
    chosen_peer = next(p for p in peers if p["node_id"] == chosen.node_id)
    dispatch_request = DispatchRequest(request_id=request_id, full_query=args.query)
    dispatch_response = await peer_client.dispatch(
        chosen_peer, dispatch_request, timeout_s=config.get("dispatch_timeout_s", 30.0)
    )
    if dispatch_response is None:
        print(f"{chosen.node_id} への/dispatchが失敗しました．")
        return
    print(
        f"[{dispatch_response.node_id}] "
        f"(confidence={dispatch_response.confidence:.2f}, {dispatch_response.gen_time_ms}ms)\n"
        f"{dispatch_response.answer_text}"
    )


def run_ask(args: argparse.Namespace) -> None:
    """_askの同期エントリポイント．"""
    asyncio.run(_ask(args))


def build_arg_parser() -> argparse.ArgumentParser:
    """serve/askサブコマンドを持つCLIパーサーを構築する．"""
    parser = argparse.ArgumentParser(description="出会い型専門家メッシュ ノードプロセス")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="本ノードをHTTPサーバーとして起動する")
    serve_parser.add_argument("--node-id", required=True)
    serve_parser.add_argument("--config", default="config.yaml")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(func=run_serve)

    ask_parser = subparsers.add_parser("ask", help="requesterとして質問を投げる")
    ask_parser.add_argument("--node-id", required=True)
    ask_parser.add_argument("--config", default="config.yaml")
    ask_parser.add_argument("query")
    ask_parser.set_defaults(func=run_ask)

    return parser


def main() -> None:
    """CLIエントリポイント．"""
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

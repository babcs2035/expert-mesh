"""CLI entry point: switch between serve (expert) and ask (requester) modes."""

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

# Truncate the query summary sent to the lightweight probe model.
QUERY_SUMMARY_MAX_LENGTH = 200
# When this node also owns the target domain, reach it via localhost.
# External IP + hairpin NAT does not work reliably in all environments.
SELF_HOST_OVERRIDE = "localhost"


def load_yaml(path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_node_state(config: dict, node_id: str) -> NodeState:
    """Construct a NodeState for the specified node from the config."""
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
    """Start the FastAPI server so this node acts as an expert."""
    config = load_yaml(args.config)
    state = build_node_state(config, args.node_id)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


async def _ask(args: argparse.Namespace) -> None:
    """Run the requester flow: embed the query, probe peers, dispatch to the best one."""
    config = load_yaml(args.config)
    peers = []
    for node_id, node_config in config["nodes"].items():
        peer = {"node_id": node_id, **node_config}
        if node_id == args.node_id:
            peer = {**peer, "host": SELF_HOST_OVERRIDE}
        peers.append(peer)
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
        print("No available expert found.")
        return

    chosen = targets[0]
    chosen_peer = next(p for p in peers if p["node_id"] == chosen.node_id)
    dispatch_request = DispatchRequest(request_id=request_id, full_query=args.query)
    dispatch_response = await peer_client.dispatch(
        chosen_peer, dispatch_request, timeout_s=config.get("dispatch_timeout_s", 30.0)
    )
    if dispatch_response is None:
        print(f"Dispatch to {chosen.node_id} failed.")
        return
    print(
        f"[{dispatch_response.node_id}] "
        f"(confidence={dispatch_response.confidence:.2f}, {dispatch_response.gen_time_ms}ms)\n"
        f"{dispatch_response.answer_text}"
    )


def run_ask(args: argparse.Namespace) -> None:
    """Synchronous wrapper around the async _ask coroutine."""
    asyncio.run(_ask(args))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with serve and ask subcommands."""
    parser = argparse.ArgumentParser(description="Encounter expert-mesh node process")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start this node as an HTTP server")
    serve_parser.add_argument("--node-id", required=True)
    serve_parser.add_argument("--config", default="config.yaml")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.set_defaults(func=run_serve)

    ask_parser = subparsers.add_parser("ask", help="Send a question as a requester")
    ask_parser.add_argument("--node-id", required=True)
    ask_parser.add_argument("--config", default="config.yaml")
    ask_parser.add_argument("query")
    ask_parser.set_defaults(func=run_ask)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

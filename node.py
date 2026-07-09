"""CLI entry point: switch between serve (expert) and ask (requester) modes."""

import argparse
import asyncio
import dataclasses
import uuid

import uvicorn
import yaml

from aggregator import select_best_dispatch_response, select_dispatch_targets
from expert_backend import OllamaClient
from http_client import PeerClient
from http_server import ROUTING_METHOD_SELF_REPORT, NodeState, create_app
from protocol import DispatchRequest, DispatchResponse, ProbeRequest, ProbeResponse

# Truncate the query summary sent to the lightweight probe model.
QUERY_SUMMARY_MAX_LENGTH = 200
# When this node also owns the target domain, reach it via localhost.
# External IP + hairpin NAT does not work reliably in all environments.
SELF_HOST_OVERRIDE = "localhost"
# Fallback prompt used when no peer reports sufficient confidence (design
# doc 2.5: "the requester's own general model, or an honest 'I don't know'").
FALLBACK_MAX_TOKENS = 512
FALLBACK_PROMPT_TEMPLATE = (
    "あなたは特定の専門分野を持たない汎用アシスタントです．"
    "専門知識を要する内容であれば，断定を避けつつ一般的に分かる範囲で回答し，"
    "専門家への相談を推奨してください．\n質問: {query}"
)


def load_yaml(path: str) -> dict:
    """Load and parse a YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_peers(config: dict, self_node_id: str) -> list[dict]:
    """Build the peer list from config, routing the self entry to localhost."""
    peers = []
    for node_id, node_config in config["nodes"].items():
        peer = {"node_id": node_id, **node_config}
        if node_id == self_node_id:
            peer = {**peer, "host": SELF_HOST_OVERRIDE}
        peers.append(peer)
    return peers


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
        peers=_build_peers(config, node_id),
        embedding_model=config.get("embedding_model"),
        routing_method=config.get("routing_method", ROUTING_METHOD_SELF_REPORT),
    )


def run_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server so this node acts as an expert."""
    config = load_yaml(args.config)
    state = build_node_state(config, args.node_id)
    app = create_app(state)
    uvicorn.run(app, host=args.host, port=args.port)


async def _dispatch_to_targets(
    peer_client: PeerClient,
    peers: list[dict],
    targets: list[ProbeResponse],
    dispatch_request: DispatchRequest,
    dispatch_timeout_s: float,
) -> DispatchResponse | None:
    """Send /dispatch to every selected target concurrently and pick the best answer.

    With top_k == 1 (the Phase 0 default) this degrades to a single call;
    with top_k > 1 (design doc 2.5: "multiple candidates step forward") all
    candidates are dispatched in parallel and select_best_dispatch_response
    picks the highest-confidence responder.
    """
    target_peers = [next(p for p in peers if p["node_id"] == t.node_id) for t in targets]
    dispatch_responses = await asyncio.gather(
        *(
            peer_client.dispatch(peer, dispatch_request, timeout_s=dispatch_timeout_s)
            for peer in target_peers
        )
    )
    return select_best_dispatch_response([r for r in dispatch_responses if r is not None])


async def _fallback_answer(ollama_client: OllamaClient, light_model: str, query: str) -> str:
    """Generate a hedged, non-expert answer when no peer reports sufficient confidence.

    Uses the same timeout as dispatch (400s) because full generation with the
    light model can take several minutes on CPU-only inference.
    """
    return await ollama_client.generate(
        light_model,
        FALLBACK_PROMPT_TEMPLATE.format(query=query),
        timeout_s=600.0,
        max_tokens=FALLBACK_MAX_TOKENS,
    )


@dataclasses.dataclass
class AskResult:
    """Structured outcome of a single ask flow run, for both CLI display and benchmarking.

    Exactly one of dispatch_response/fallback_answer is set unless the run
    failed entirely (both None), which happens when probing found qualifying
    targets but every /dispatch call to them failed.
    """

    request_id: str
    probe_responses: list[ProbeResponse]
    dispatch_response: DispatchResponse | None
    fallback_answer: str | None


async def run_ask_flow(
    config: dict, self_node_id: str, query: str, ollama_client: OllamaClient | None = None
) -> AskResult:
    """Run the requester flow and return a structured result (no printing).

    Shared by node.py's `ask` CLI subcommand and run_experiment.py
    so that the probe/dispatch/fallback logic is exercised identically in
    both interactive use and automated evaluation.
    """
    peers = _build_peers(config, self_node_id)
    ollama_client = ollama_client or OllamaClient()
    self_node_config = config["nodes"][self_node_id]

    request_id = str(uuid.uuid4())
    query_summary = query[:QUERY_SUMMARY_MAX_LENGTH]
    query_embedding = await ollama_client.embed(config["embedding_model"], query)

    probe_request = ProbeRequest(
        request_id=request_id,
        query_summary=query_summary,
        query_embedding=query_embedding,
        from_=self_node_id,
    )
    peer_client = PeerClient(peers)
    probe_responses = await peer_client.probe_all(
        probe_request, timeout_s=config.get("probe_timeout_s", 2.0)
    )
    targets = select_dispatch_targets(
        probe_responses,
        confidence_threshold=config.get("confidence_threshold", 0.5),
        top_k=config.get("dispatch_top_k", 1),
    )
    if not targets:
        # Design doc 2.5: fall back to the requester's own general model
        # rather than a bare "no expert found" when nobody claims the query.
        answer = await _fallback_answer(ollama_client, self_node_config["light_model"], query)
        return AskResult(
            request_id=request_id,
            probe_responses=probe_responses,
            dispatch_response=None,
            fallback_answer=answer,
        )

    dispatch_request = DispatchRequest(request_id=request_id, full_query=query)
    dispatch_response = await _dispatch_to_targets(
        peer_client, peers, targets, dispatch_request, config.get("dispatch_timeout_s", 30.0)
    )
    return AskResult(
        request_id=request_id,
        probe_responses=probe_responses,
        dispatch_response=dispatch_response,
        fallback_answer=None,
    )


async def _ask(args: argparse.Namespace) -> None:
    """CLI wrapper: run the requester flow and print a human-readable result."""
    config = load_yaml(args.config)
    result = await run_ask_flow(config, args.node_id, args.query)

    if result.fallback_answer is not None:
        print(
            f"[fallback:{args.node_id}] No expert met the confidence threshold.\n"
            f"{result.fallback_answer}"
        )
        return
    if result.dispatch_response is None:
        eligible = [r.node_id for r in result.probe_responses]
        print(f"Dispatch failed (probed candidates: {eligible}).")
        return
    print(
        f"[{result.dispatch_response.node_id}] "
        f"(confidence={result.dispatch_response.confidence:.2f}, "
        f"{result.dispatch_response.gen_time_ms}ms)\n"
        f"{result.dispatch_response.answer_text}"
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

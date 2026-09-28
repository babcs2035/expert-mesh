"""Print the model names used by a specific node (light, expert, embedding)."""

import argparse

import yaml


def get_models(config_path: str, node_id: str) -> list[str]:
    """Return [light_model, expert_model, embedding_model, *fusion_models] for the given node.

    Iter100 (embedding_space_fusion): embedding_fusion_models entries (e.g.
    ruri-v3-310m) must also be pulled on every node, or the entry node fails
    with a 404 from Ollama the first time embed_query_views() requests a
    fusion model that was never `ollama pull`-ed (mise.toml's deploy task
    iterates this function's return value). Missing/empty
    embedding_fusion_models (pre-Iter100 config) returns the same 3-model
    list as before.
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    node_config = config["nodes"][node_id]
    models = [node_config["light_model"], node_config["expert_model"], config["embedding_model"]]
    models.extend(fusion["model"] for fusion in config.get("embedding_fusion_models") or [])
    return models


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Print model names for a node")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("node_id")
    args = parser.parse_args()
    print(" ".join(get_models(args.config, args.node_id)))


if __name__ == "__main__":
    main()

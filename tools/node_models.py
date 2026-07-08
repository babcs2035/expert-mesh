"""Print the model names used by a specific node (light, expert, embedding)."""

import argparse

import yaml


def get_models(config_path: str, node_id: str) -> list[str]:
    """Return [light_model, expert_model, embedding_model] for the given node."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    node_config = config["nodes"][node_id]
    return [node_config["light_model"], node_config["expert_model"], config["embedding_model"]]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Print model names for a node")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("node_id")
    args = parser.parse_args()
    print(" ".join(get_models(args.config, args.node_id)))


if __name__ == "__main__":
    main()

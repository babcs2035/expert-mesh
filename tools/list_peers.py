"""Print all node IDs from config.yaml, one per line."""

import argparse

import yaml


def load_node_ids(config_path: str) -> list[str]:
    """Return the list of node IDs from the configuration."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return list(config["nodes"].keys())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="List all node IDs from config.yaml")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    for node_id in load_node_ids(args.config):
        print(node_id)


if __name__ == "__main__":
    main()

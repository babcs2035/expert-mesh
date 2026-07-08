"""Print the remote deployment directory from config.yaml."""

import argparse

import yaml

DEFAULT_REMOTE_DIR = "~/expert-mesh"


def get_remote_dir(config_path: str) -> str:
    """Return the configured remote_dir, falling back to the default."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("remote_dir", DEFAULT_REMOTE_DIR)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Print the remote deployment directory")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(get_remote_dir(args.config))


if __name__ == "__main__":
    main()

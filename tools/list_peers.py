"""config.yamlに記載された全ノードのnode_id一覧を1行ずつ標準出力へ書き出す．

mise-tasksのシェルスクリプトから `for h in $(uv run python tools/list_peers.py); do ...`
のようにホストループを組み立てるために用いる．
"""

import argparse

import yaml


def load_node_ids(config_path: str) -> list[str]:
    """config.yamlから全ノードのnode_idをリストで返す．"""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return list(config["nodes"].keys())


def main() -> None:
    """CLIエントリポイント．"""
    parser = argparse.ArgumentParser(description="config.yamlのnode_id一覧を出力する")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    for node_id in load_node_ids(args.config):
        print(node_id)


if __name__ == "__main__":
    main()

"""config.yamlから指定node_idが使用するモデル名（軽量・専門家・embedding）を空白区切りで出力する．

mise-tasks/deployが `docker compose exec ollama ollama pull` に渡すモデル名を得るために使う．
"""

import argparse

import yaml


def get_models(config_path: str, node_id: str) -> list[str]:
    """指定ノードが使用するモデル名一覧（軽量モデル・専門家モデル・embeddingモデル）を返す．"""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    node_config = config["nodes"][node_id]
    return [node_config["light_model"], node_config["expert_model"], config["embedding_model"]]


def main() -> None:
    """CLIエントリポイント．"""
    parser = argparse.ArgumentParser(description="指定ノードが使用するモデル名一覧を出力する")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("node_id")
    args = parser.parse_args()
    print(" ".join(get_models(args.config, args.node_id)))


if __name__ == "__main__":
    main()

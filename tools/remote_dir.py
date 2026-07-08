"""config.yamlに記載されたリモートホスト上のデプロイ先ディレクトリを標準出力へ書き出す．

mise-tasks/{deploy,start,analyze,clean} が REMOTE_DIR を決定するために用いる．
"""

import argparse

import yaml

DEFAULT_REMOTE_DIR = "~/expert-mesh"


def get_remote_dir(config_path: str) -> str:
    """config.yamlのremote_dirを返す．未指定の場合はデフォルト値を返す．"""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("remote_dir", DEFAULT_REMOTE_DIR)


def main() -> None:
    """CLIエントリポイント．"""
    parser = argparse.ArgumentParser(description="config.yamlのremote_dirを出力する")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(get_remote_dir(args.config))


if __name__ == "__main__":
    main()

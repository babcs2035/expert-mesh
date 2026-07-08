# expert-meshアプリケーション（node.py）を実行するイメージ．uvで依存関係を解決する．
FROM python:3.12-slim

# ghcr.io経由のuvイメージ配布はネットワーク環境によって遮断されることがあるため，
# PyPI経由のpip installで導入する（挙動はイメージコピー方式と同等）．
RUN pip install --no-cache-dir uv

WORKDIR /app

# 依存関係のみを先にインストールすることで，アプリコード変更時のレイヤーキャッシュを効かせる
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY protocol.py expert_backend.py router.py aggregator.py http_client.py http_server.py node.py ./
# tools/healthcheck.py等は，操作端末がノードのLAN(192.168.15.0/24)に直接到達できない
# 環境で，いずれかのノードのコンテナ内から疎通確認・質問投入を行うために必要
COPY tools/ ./tools/

EXPOSE 8080

# `uv run` はコンテナ起動のたびにpyproject.tomlとの整合性を再チェックし，開発用依存
# （pytest/ruff等）まで再ダウンロードしようとするため，ビルド時に作成済みのvenvを
# 直接使う．これによりコンテナ起動時に追加のネットワークアクセスが発生しない．
# node-idはconfig.yaml以外にハードコードしないため，CMDのデフォルト引数は持たせず
# docker-compose.ymlのcommandで必ず明示的に指定させる．
ENTRYPOINT [".venv/bin/python", "node.py"]

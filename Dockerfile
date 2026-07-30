# Image for running the expert-mesh application (node.py). Resolves dependencies with uv.
FROM python:3.12-slim

# uv image distribution via ghcr.io may be blocked depending on network environment,
# so install via pip install instead (same behavior as the image copy approach).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first to leverage layer caching when app code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY protocol.py expert_backend.py router.py aggregator.py http_client.py http_server.py node.py logging_utils.py classifier.py evaluation.py ./
COPY scripts/ ./scripts/
# tools/healthcheck.py etc. are needed to run connectivity checks and queries
# from within a node's container when the operator terminal cannot directly
# reach the node LAN (192.168.15.0/24).
COPY tools/ ./tools/
# run_experiment.py makes real /probe and /dispatch calls against
# the mesh, so it must run inside a node's container for the same reason as
# tools/healthcheck.py above.
COPY run_experiment.py build_dataset.py metrics.py ./
COPY data/ ./data/

# Baked in at build time (mise.toml's setup task passes --build-arg
# GIT_HEAD=$(git rev-parse HEAD)) since .git itself is not copied into the
# image. run_experiment.py reads this to stamp each experiment's results
# directory with the exact commit the running image was built from —
# without it, a stale-deploy incident (docs/d0002 §6-C/§6-D, Iter22) can
# only be reconstructed after the fact by cross-referencing journal.md
# commit hashes against dates.
ARG GIT_HEAD=unknown
ENV GIT_HEAD=${GIT_HEAD}

EXPOSE 8080

# `uv run` rechecks pyproject.toml consistency on every container start and
# re-downloads dev dependencies (pytest, ruff, etc.), so we use the pre-built
# venv directly to avoid additional network access at container startup.
# node-id is not hardcoded anywhere except config.yaml, so the CMD has no
# default arguments — docker-compose.yml must always specify it explicitly.
ENTRYPOINT [".venv/bin/python", "node.py"]

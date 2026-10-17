# syntax=docker/dockerfile:1
# findex web service: two stages, dependencies resolved from uv.lock.
# Build:  docker build -t findex .
# Run:    docker run --rm -p 8000:8000 findex
#         docker run --rm -p 8000:8000 -e INDEX_PATH=/data/my.bin -v "$PWD/data:/data" findex

# ---------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Dependencies only (cached layer: changes to src/ do not reinstall them).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --no-editable

# 2) The project itself, installed into .venv as a regular (non-editable) package.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# 3) Bake the demo index into the image (simple and fine for a portfolio demo).
COPY corpus ./corpus
RUN mkdir -p data \
 && gunzip -c corpus/peps.jsonl.gz > data/peps.jsonl \
 && .venv/bin/findex index data/peps.jsonl --out data/peps.bin --positions

# ---------------------------------------------------------------- runtime
FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --system --gid 10001 findex \
 && useradd --system --uid 10001 --gid findex --home-dir /app --no-create-home findex

WORKDIR /app
COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/data /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/app/data/peps.bin \
    LOG_FORMAT=json \
    LOG_LEVEL=INFO \
    PORT=8000

USER findex
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health\", timeout=2)"]

# The CLI stays the entry point; PORT and WEB_CONCURRENCY come from the platform.
CMD ["findex", "serve", "--host", "0.0.0.0"]

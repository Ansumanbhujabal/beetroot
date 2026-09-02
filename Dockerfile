# syntax=docker/dockerfile:1
#
# beatroot — packaged for HF Spaces (Docker SDK, port 7860) and for a plain
# `docker run`/`docker compose up` on a developer's machine.
#
# The whole project's premise is that this boots with NO credentials.
# That guarantee lives in TWO places, deliberately redundant:
#
#   1. `beatroot.settings.Settings` (the real fix — see its
#      `_default_to_offline_without_credentials` validator): if neither the
#      configured completion model nor the embedding model has credentials
#      in the environment, `offline` flips to True itself, logged at
#      WARNING. This is what actually matters, because a bare `uvicorn`
#      run outside Docker has the identical problem and gets the identical
#      fix — `build_container()` eagerly embeds the whole catalog to build
#      the vector store, before `/health` is even reachable, so a
#      credential-less real `LLMClient` used to take the whole process
#      down at startup, not on first request.
#   2. `BEATROOT_OFFLINE=1` below, belt-and-braces: makes the containerised
#      path explicitly offline unless someone supplies real provider
#      credentials and sets `BEATROOT_OFFLINE=0` (`docker run -e ...`, or
#      the same via `.env`/compose — see README.md). Redundant with (1) on
#      purpose — this must never be the ONLY thing keeping it keyless.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    BEATROOT_OFFLINE=1

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

# HEALTHCHECK below shells out to curl rather than depending on the app's
# own venv/interpreter path, so it stays correct regardless of how `uv`
# lays out the virtualenv.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not invalidate it. Both extras
# matter here, not just for parity:
#
#   --extra qdrant  docker-compose.yml sets QDRANT_URL for the containerised
#                   run, and beatroot.retrieval.dense switches to the real
#                   QdrantVectorStore whenever that's set — so qdrant-client
#                   has to be in this image for `docker compose up` to work.
#   --extra obs     the Langfuse SDK. Without it, prompts silently fall back
#                   to the local files and tracing is a no-op — the container
#                   keeps serving, which is exactly what makes the omission
#                   easy to miss. `beatroot prompts status` names the source,
#                   so the fallback is visible rather than assumed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra qdrant --extra obs

COPY . .
RUN uv sync --frozen --no-dev --extra qdrant --extra obs

# HF Spaces runs as a non-root user and expects port 7860.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uv", "run", "uvicorn", "beatroot.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

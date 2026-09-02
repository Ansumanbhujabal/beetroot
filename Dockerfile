# syntax=docker/dockerfile:1
#
# beatroot — a two-stage build producing a runtime image that carries the
# virtualenv and the application's data, and nothing that was only needed to
# assemble them.
#
# WHY TWO STAGES, AND WHAT THE SINGLE-STAGE VERSION COST
# ------------------------------------------------------
# The previous single-stage image was 1.32 GB against a 348 MB virtualenv.
# Almost all of the excess came from one line:
#
#     RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
#
# `chown -R` rewrites the ownership metadata of every file under /app, and a
# Docker layer records a changed file as a whole new copy — so that single
# RUN added a 408 MB layer that duplicated the entire application and its
# virtualenv. The image carried both copies forever, because a later layer
# can never shrink an earlier one. Creating the user FIRST and copying with
# `--chown` sets the ownership as the files land, so there is nothing to
# rewrite afterwards.
#
# The build stage also drops out entirely: `uv` itself, the apt lists, and
# any intermediate build state exist only in the builder and are never part
# of what ships.

# ---------------------------------------------------------------- builder --
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

COPY --from=ghcr.io/astral-sh/uv:0.7.19 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first so source edits do not invalidate it. Both extras
# matter, not just for parity with local development:
#
#   --extra qdrant  docker-compose.yml sets QDRANT_URL, and
#                   beatroot.retrieval.dense switches to the real
#                   QdrantVectorStore whenever that is set — so qdrant-client
#                   has to be present for `docker compose up` to work.
#   --extra obs     the Langfuse SDK, for prompt fetching and tracing.
#                   Without it prompts silently fall back to the local files
#                   and tracing is a no-op; the container keeps serving,
#                   which is exactly what makes the omission easy to miss.
#                   `beatroot prompts status` names the source either way.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra qdrant --extra obs

# The project is installed as an editable pointer to /app/src, so the source
# has to be present for the install to resolve — and the runtime stage has to
# keep /app/src at the same path for that pointer to stay valid.
COPY src ./src
RUN uv sync --frozen --no-dev --extra qdrant --extra obs

# ---------------------------------------------------------------- runtime --
FROM python:3.12-slim AS runtime

# Put the virtualenv on PATH so `uvicorn` and `beatroot` are called directly.
# The old image shelled out through `uv run`, which meant shipping the uv
# binary and paying a dependency-resolution check on every container start,
# to launch an environment that was already fully built.
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Created BEFORE anything is copied, so every COPY --chown below lands with
# the right ownership and no recursive rewrite is ever needed.
#
# `/app` itself is chowned too, and that is not incidental: the application
# creates `beatroot.db` and `beatroot.checkpoints.db` in this directory at
# startup, so the DIRECTORY has to be writable by the runtime user or the
# container dies in `lifespan` with `sqlite3.OperationalError: unable to
# open database file`. This is one inode, not a recursive rewrite — the
# contents arrive already-owned via `COPY --chown`, which is the whole
# point of the split.
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app \
    && chown appuser:appuser /app

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv ./.venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser pyproject.toml ./

# Runtime data the application reads from disk, by path, at startup. Each of
# these resolves relative to the repository root (see `container.ROOT` and
# `settings.ROOT`), so the layout here has to mirror the repository's.
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser prompts ./prompts
COPY --chown=appuser:appuser skills ./skills
COPY --chown=appuser:appuser skills-lock.json ./
COPY --chown=appuser:appuser eval ./eval

# Documents and diagrams the `/docs` page serves. These were excluded from
# the old image by `.dockerignore`, which meant the page rendered with a
# broken diagram and dead download links in every containerised run — the
# one deployment a person actually looks at.
COPY --chown=appuser:appuser docs/diagrams ./docs/diagrams
COPY --chown=appuser:appuser docs/WALKTHROUGH.md docs/PRODUCTION_READINESS.md ./docs/
COPY --chown=appuser:appuser README.md ARCHITECTURE.md CUT_LIST.md EVAL_RESULTS.md EVAL_HISTORY.md ./

USER appuser

EXPOSE 7860

# Uses the interpreter that is already in the image rather than installing
# curl, which cost an apt layer and its lists purely to make one HTTP
# request. `urllib` raises a non-zero exit on any non-2xx, which is exactly
# the semantics a healthcheck needs.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:7860/health', timeout=4)" \
        || exit 1

CMD ["uvicorn", "beatroot.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

"""The beatroot HTTP API. Spec §15.

Production shape, not a demo: every dependency is built once, at process
startup, by `beatroot.container.build_container` and handed to the app via
`lifespan` — there is no bare module-level `_agent = None` that a request
handler lazily populates on first use. A correlation id travels with every
request (`x-request-id`, generated if the caller didn't send one, always
echoed back) so a single request can be traced through logs even without
Task 19's tracing landed yet. An unhandled exception is always logged with
its traceback and never leaks one back to the caller — the two are
independent failure modes and this module refuses to trade one for the
other.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from beatroot.container import DATA_DIR, ROOT, Container, build_container
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.obs.cost import CostLedger
from beatroot.obs.logging import RING_BUFFER_SIZE, bind_request, recent_logs
from beatroot.obs.tracing import start_trace_flusher

log = logging.getLogger("beatroot.api")

# The dashboard (now three pages — PROFILES/UI task): one self-contained
# file per page, no build step. Served by path rather than embedded as a
# string so each stays ordinary, editable HTML.
_WEB_DIR = Path(__file__).resolve().parents[1] / "web"
_INDEX_PATH = _WEB_DIR / "index.html"
_INCIDENTS_PATH = _WEB_DIR / "incidents.html"
_EVALS_PATH = _WEB_DIR / "evals.html"
_DOCS_PATH = _WEB_DIR / "docs.html"

# Preset dietary profiles — PRESET PROFILES task. data/profiles.yaml, same
# repo-root DATA_DIR the Container seeds the catalog from.
_PROFILES_PATH = DATA_DIR / "profiles.yaml"


@lru_cache(maxsize=1)
def _load_profiles() -> list[dict[str, Any]]:
    """`data/profiles.yaml`, parsed and validated once per process.

    Each entry's `constraints` round-trips through `Constraint` (the exact
    model `POST /recommend` validates against) so a value the model would
    reject can never reach a client silently — this fails at first request
    with a normal Pydantic error instead. `@lru_cache` rather than a
    module-level constant so a bad file fails on first use, not at import
    time (consistent with every other lazy, request-time load in this
    module).
    """
    raw = yaml.safe_load(_PROFILES_PATH.read_text()) or []
    profiles: list[dict[str, Any]] = []
    for entry in raw:
        constraints = [Constraint(**c) for c in entry.get("constraints", [])]
        profiles.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "description": entry["description"],
                "constraints": [c.model_dump(mode="json") for c in constraints],
            }
        )
    return profiles


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = build_container()
    app.state.container = container
    # Spans reach Langfuse on a timer, not per request. LiteLLM creates the
    # span AFTER the response is sent, so flushing from response middleware
    # flushes an empty provider; and OpenTelemetry only force-flushes at
    # process exit, which a server never reaches. Together those two facts
    # meant a fully "configured" server exported nothing at all. See
    # `obs.tracing._PeriodicFlusher`. `None` when Langfuse is unconfigured.
    flusher = start_trace_flusher()
    log.info(
        "startup",
        extra={"health": container.health(), "trace_flusher": flusher is not None},
    )
    try:
        yield
    finally:
        if flusher is not None:
            # Flushes once more on the way out, so the last few seconds of
            # traffic are not lost to a restart.
            flusher.stop()
        container.close()


# `docs_url="/api-docs"`: FastAPI mounts Swagger UI at "/docs" BY DEFAULT and
# registers it during __init__, before any route below — so it would silently
# shadow this project's own documentation page. Moved rather than renaming our
# page: "/docs" is what a reviewer types looking for the architecture, and the
# OpenAPI explorer is the less-visited of the two.
app = FastAPI(title="beatroot", version="0.1.0", lifespan=lifespan, docs_url="/api-docs")


def container(request: Request) -> Container:
    """The one seam route handlers use to reach the composition root.

    Tests substitute a Container by overriding this dependency
    (`app.dependency_overrides[container] = lambda: my_container`) —
    never by monkeypatching an import.

    Production (`uvicorn` -> `lifespan`) always has `app.state.container`
    set before the first request is even accepted. The fallback to
    `get_container()` below exists only for a caller that drives `app`
    directly with a plain `TestClient(app)` and never enters it as a
    context manager, so `lifespan` never ran — a real scenario (this
    module's own smoke test does exactly that), not a hypothetical one.
    It is the same memoized composition-root accessor `container.py`
    exposes for that purpose, never a bare module-level mutable global."""
    state_container: Container | None = getattr(request.app.state, "container", None)
    if state_container is not None:
        return state_container
    from beatroot.container import get_container

    return get_container()


ContainerDep = Annotated[Container, Depends(container)]


def _record_plan_cost(
    ledger: CostLedger, result: Recommendation | Negotiation | Escalation
) -> None:
    """Fold one plan's CostRecord (already computed by the graph via
    `agent.state.merge_cost`) into the process-wide CostLedger, so
    `/metrics` reports a real, accumulating cost-per-plan instead of a
    per-request number nobody ever aggregates. Every terminal result
    (COMMIT/NEGOTIATE/ESCALATE) carries a `.cost`, so this runs
    unconditionally whenever the graph actually reached one."""
    cost = result.cost
    for stage, usd in cost.per_stage.items():
        ledger.add(stage, usd)
    total_tokens = cost.prompt_tokens + cost.completion_tokens
    if total_tokens:
        ledger.tokens += total_tokens
    if cost.tokens_saved:
        ledger.record_short_circuit(cost.tokens_saved)
    ledger.record_plan()


def _terminal_envelope(
    c: Container,
    result: Recommendation | Negotiation | Escalation,
    trace: list[str],
) -> dict[str, Any]:
    """The response body for a real terminal (COMMIT/NEGOTIATE/ESCALATE) —
    shared by `/recommend` and `/resume/…` so the two routes can never
    disagree about the envelope's shape or about Task 23's split.

    A `Recommendation` with `explanation == ""` means `explain_node` took
    the async branch (`settings.async_explanation`) and only QUEUED the
    prose — the card itself (recipe, nutrition, trust, constraints
    satisfied) is already the full, verified answer. This overwrites the
    dumped `explanation` with `null` and adds `explanation_status`, rather
    than leaving callers to infer "pending" from an empty string, which is
    also what a real-but-blank completion would look like.
    """
    thread_id = c.agent.last_thread_id
    payload: dict[str, Any] = {
        "terminal_state": trace[-1] if trace else "UNKNOWN",
        "trace": trace,
        "audit_id": c.agent.last_audit_id,
        "thread_id": thread_id,
        "result": result.model_dump(mode="json"),
        # QUERY REWRITE task: `None` when RETRIEVE never ran (an infeasible
        # or vocabulary-escalated profile short-circuits before it) — the
        # UI shows nothing rather than a stale value from a previous run.
        "query_rewrite": c.agent.last_query_rewrite,
        # FREE-TEXT/NLP task: the FULL effective constraint set — every
        # constraint submitted PLUS whatever `compile_node` (GAP 1) parsed
        # out of free text and appended, each still carrying its own
        # `source` ("structured" vs "parsed_free_text"). This is what the
        # UI shows the user which constraints their free text actually
        # produced, instead of guessing from an id naming convention.
        "effective_constraints": (c.agent.last_constraint_set or {}).get("constraints", []),
    }
    if isinstance(result, Recommendation) and not result.explanation and c.explanation_queue:
        status = c.explanation_queue.status(thread_id) if thread_id else "pending"
        payload["result"]["explanation"] = None
        payload["result"]["explanation_status"] = status
    return payload


def _unhandled_response(request: Request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace to a caller, never lose one from the logs."""
    log.exception(
        "unhandled error",
        extra={"path": request.url.path, "request_id": getattr(request.state, "request_id", None)},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "The request could not be completed.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.middleware("http")
async def correlate(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Correlation id, always echoed back — even on a 500.

    Starlette's `ServerErrorMiddleware` (which is what `@app.exception_handler
    (Exception)` below actually attaches to) sits OUTSIDE this middleware, so
    a route exception that only that handler caught would propagate straight
    past this function and reach the caller with no `x-request-id` header at
    all. Catching here too is what guarantees the header — and the safe body
    — on every response, error or not.
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    # Task 23: bind the correlation id into obs.logging's ContextVar for
    # the lifetime of this request — not just onto `request.state`. A
    # route handler that hands work to `agent.async_explain.
    # ExplanationQueue` captures THIS context (`contextvars.copy_context()`)
    # before the background thread starts, which is the only reason the
    # worker's own log lines still carry `request_id` after this request
    # has already returned. ContextVars are not inherited by a
    # ThreadPoolExecutor worker for free — see that module's docstring.
    with bind_request(request_id):
        try:
            response = await call_next(request)
        except Exception as exc:
            response = _unhandled_response(request, exc)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Defense in depth for anything raised above `correlate` (routing
    itself, an ASGI-level failure) that the middleware's own try/except
    never gets a chance to see. HTTPException / RequestValidationError still
    go through Starlette's own more specific handlers."""
    return _unhandled_response(request, exc)


class RecommendRequest(BaseModel):
    """Bounded on purpose: unbounded input on a public endpoint is a defect,
    not a convenience. A `profile_id` has no legitimate reason to be longer
    than a UUID plus room to spare; 50 constraints and a 2000-character
    query are already generous for a single meal request."""

    profile_id: str = Field(min_length=1, max_length=128)
    constraints: list[Constraint] = Field(default_factory=list, max_length=50)
    query: str = Field("", max_length=2000)
    # GAP 1 + FREE-TEXT/NLP: optional free text compiled into constraints
    # (at whatever severity actually enforces them) by the `compile` node
    # before FEASIBILITY runs, and the only place scope is judged — a
    # request with no meal-planning content in this field terminates
    # straight to ESCALATE(reason="out_of_scope"). Distinct from `query`,
    # which only ever feeds retrieval ranking. Empty by default, which
    # costs zero tokens (`compile` degrades to a no-op — including the
    # scope check, which only ever runs when free text is present).
    preferences: str = Field("", max_length=2000)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Page 1/3 — profile picker, builder, free text, result, trust,
    ladder, trace. Static, self-contained, offline — see
    `beatroot/web/index.html`. Spec §15; PAGES task for the split."""
    return FileResponse(_INDEX_PATH, media_type="text/html")


@app.get("/incidents", include_in_schema=False)
def incidents_page() -> FileResponse:
    """Page 2/3 — the incident feed and drift ledger, moved off the main
    page (PAGES task). JSON data lives at `GET /api/incidents`; this route
    is the browsable page only — see `beatroot/web/incidents.html`."""
    return FileResponse(_INCIDENTS_PATH, media_type="text/html")


@app.get("/evals", include_in_schema=False)
def evals_page() -> FileResponse:
    """Page 3/3 — eval results and live logs (PAGES task). JSON data lives
    at `GET /evals/summary` and `GET /evals/logs`; this route is the
    browsable page only — see `beatroot/web/evals.html`."""
    return FileResponse(_EVALS_PATH, media_type="text/html")


@app.get("/docs", include_in_schema=False)
def docs_page() -> FileResponse:
    """Page 4/4 — the architecture diagram (pan/zoom) and downloadable docs.

    The FastAPI interactive docs live at `/api-docs` (see `docs_url` on the
    `FastAPI(...)` constructor); this path is the project's OWN documentation
    page, which is what a reviewer means by "/docs".
    """
    return FileResponse(_DOCS_PATH, media_type="text/html")


# An explicit ALLOW-LIST, keyed on the exact filename, mapping to a resolved
# path. Not a directory server and not a path parameter joined onto a base
# directory: `/docs/file/../../.env` cannot express anything this dict does not
# already contain, so there is no traversal to defend against — the attack
# surface is the dict, and the dict is six documents and four diagram exports.
#
# `INTERVIEW_PREP.md` and `VIDEO_SCRIPT.md` are deliberately NOT here. They
# are local-only preparation notes (gitignored, never pushed), and an
# allow-list entry would have served them from any deployment of this app —
# which is the same leak as committing them, reached by a different route.
# Keeping them out of the dict is what makes "local-only" true of the
# running system, not just of the repository.
_DOC_FILES: dict[str, tuple[Path, str]] = {
    "README.md": (ROOT / "README.md", "text/markdown"),
    "ARCHITECTURE.md": (ROOT / "ARCHITECTURE.md", "text/markdown"),
    "EVAL_RESULTS.md": (ROOT / "EVAL_RESULTS.md", "text/markdown"),
    "CUT_LIST.md": (ROOT / "CUT_LIST.md", "text/markdown"),
    "WALKTHROUGH.md": (ROOT / "docs" / "WALKTHROUGH.md", "text/markdown"),
    "PRODUCTION_READINESS.md": (ROOT / "docs" / "PRODUCTION_READINESS.md", "text/markdown"),
    "EVAL_HISTORY.md": (ROOT / "EVAL_HISTORY.md", "text/markdown"),
    "beatroot-architecture.svg": (
        ROOT / "docs" / "diagrams" / "beatroot-architecture.svg",
        "image/svg+xml",
    ),
    "beatroot-architecture.drawio.png": (
        ROOT / "docs" / "diagrams" / "beatroot-architecture.drawio.png",
        "image/png",
    ),
    "beatroot-architecture.pdf": (
        ROOT / "docs" / "diagrams" / "beatroot-architecture.pdf",
        "application/pdf",
    ),
    "beatroot-architecture.drawio": (
        ROOT / "docs" / "diagrams" / "beatroot-architecture.drawio",
        "application/xml",
    ),
}


@app.get("/docs/file/{name}", include_in_schema=False, response_model=None)
def docs_file(name: str) -> FileResponse | JSONResponse:
    """Serve one allow-listed document or diagram export, read-only.

    A name outside `_DOC_FILES` is a 404 — including one that resolves to a
    real file elsewhere in the repo. A listed file that is missing on disk is
    also a 404 rather than a 500: the diagram exports are build artifacts, and
    a checkout without them should degrade to a dead link, not an error page.
    """
    entry = _DOC_FILES.get(name)
    if entry is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    path, media_type = entry
    if not path.is_file():
        return JSONResponse({"detail": "not built"}, status_code=404)
    return FileResponse(path, media_type=media_type, filename=name)


@app.get("/health")
def health(c: ContainerDep) -> dict[str, Any]:
    return c.health()


@app.get("/profiles")
def profiles() -> dict[str, Any]:
    """Preset dietary profiles for the dropdown — PRESET PROFILES task.

    Loads `data/profiles.yaml` once and validates every profile's
    `constraints` through the SAME `Constraint` model `POST /recommend`
    accepts, so a malformed preset fails loudly at first request rather
    than reaching a client as a silently-broken dropdown entry. Picking one
    fills the existing builder — it never bypasses it.
    """
    return {"profiles": _load_profiles()}


@app.post("/recommend")
def recommend(req: RecommendRequest, c: ContainerDep) -> dict[str, Any]:
    cs = ConstraintSet(profile_id=req.profile_id, constraints=req.constraints)
    result = c.agent.run(cs, query=req.query, preferences=req.preferences)
    trace = list(c.agent.trace)

    if result is None:
        # The graph genuinely paused at interrupt_before=["commit"]: trust
        # landed in the medical review band and this profile carries a
        # MEDICAL constraint. Durable state sits at `last_thread_id`,
        # waiting on `.resume()` — no meal was produced, and none should
        # be, until a human looks at it. There is no /resume route yet
        # (a later task's job); this still must never be reported as one
        # of the three real terminals.
        return {
            "terminal_state": "PENDING_REVIEW",
            "trace": trace,
            "audit_id": c.agent.last_audit_id,
            "result": {
                "thread_id": c.agent.last_thread_id,
                "detail": (
                    "trust landed in the medical review band; a human "
                    "must approve before this can commit."
                ),
            },
        }

    _record_plan_cost(c.cost_ledger, result)
    return _terminal_envelope(c, result, trace)


class ResumeRequest(BaseModel):
    """`thread_id` comes from the path, not the body — a `PENDING_REVIEW`
    `/recommend` response is the only legitimate source of one."""

    approved: bool


@app.post("/recommend/{thread_id}/resume", response_model=None)
def resume(thread_id: str, req: ResumeRequest, c: ContainerDep) -> dict[str, Any] | JSONResponse:
    """Continue a thread paused at the medical review gate. Spec §6.

    `agent.graph.MealPlanningAgent.resume()` already existed (Task 11) —
    this is the only surface that reaches it. Every response is checked
    against `agent.thread_state()` BEFORE `.resume()` is ever called, so
    an unknown or already-settled thread_id never reaches `.resume()` at
    all and never has a chance to produce a 500.
    """
    status = c.agent.thread_state(thread_id)
    if status == "unknown":
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "thread_id": thread_id,
                "detail": "no paused recommendation was found for this thread_id.",
            },
        )
    if status == "resolved":
        return JSONResponse(
            status_code=409,
            content={
                "error": "already_resolved",
                "thread_id": thread_id,
                "detail": (
                    "this thread is not awaiting approval — it either never "
                    "paused or has already been resumed."
                ),
            },
        )

    result = c.agent.resume(thread_id, req.approved)
    trace = list(c.agent.trace)
    if result is None:
        return {
            "terminal_state": trace[-1] if trace else "UNKNOWN",
            "trace": trace,
            "audit_id": c.agent.last_audit_id,
            "thread_id": c.agent.last_thread_id,
            "result": None,
        }
    _record_plan_cost(c.cost_ledger, result)
    return _terminal_envelope(c, result, trace)


@app.get("/recommend/{thread_id}/explanation", response_model=None)
def recommend_explanation(thread_id: str, c: ContainerDep) -> dict[str, Any] | JSONResponse:
    """Poll for the prose a `POST /recommend` response deferred. Task 23.

    Only meaningful when `settings.async_explanation` is on — `POST
    /recommend` already returns the full explanation inline otherwise, so
    there is nothing this route would ever have to offer a caller that
    isn't already in the recommendation's own body.
    """
    queue = c.explanation_queue
    if queue is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "detail": "async explanation is not enabled for this deployment.",
            },
        )
    status = queue.status(thread_id)
    # A short bounded wait, not a long-poll: if the job already finished,
    # `.get()` returns instantly (the future is already done); if it is
    # still genuinely pending, this is a small allowance for the
    # ready-flip race between the entry and the future, not a mechanism
    # this route relies on to make an unfinished job look finished.
    text = queue.get(thread_id, timeout=1.0) if status == "ready" else None
    return {"thread_id": thread_id, "explanation_status": status, "explanation": text}


@app.get("/metrics")
def metrics(c: ContainerDep) -> dict[str, Any]:
    """Cache hit rates and incident count — the numbers that make the
    caching and safety story measurable rather than merely asserted."""
    return {
        "feasibility_cache": {
            "hits": c.feasibility_cache.hits,
            "misses": c.feasibility_cache.misses,
            "hit_rate": c.feasibility_cache.hit_rate,
        },
        "embedding_cache": {
            "hits": c.embedding_cache.hits,
            "misses": c.embedding_cache.misses,
            "hit_rate": c.embedding_cache.hit_rate,
        },
        "incidents": len(c.incidents.all()),
        "cost": {
            "per_stage_usd": c.cost_ledger.per_stage,
            "total_usd": c.cost_ledger.total_usd,
            "plans": c.cost_ledger.plans,
            "per_plan_usd": c.cost_ledger.per_plan_usd,
            "tokens": c.cost_ledger.tokens,
            # GAP 2: never a measured spend — the calls it stands in for
            # never happened. See `agent.nodes._estimate_skipped_tokens`
            # for the derivation this method string names.
            "tokens_saved": c.cost_ledger.tokens_saved,
            "tokens_saved_estimate_method": (
                "chars(rendered rerank + explain prompts, real catalog sample) / 4 chars-per-token"
            ),
        },
    }


@app.get("/api/incidents")
def incidents(c: ContainerDep) -> dict[str, Any]:
    """JSON data for the incidents page — `GET /incidents` (plural, no
    `/api` prefix) is the HTML page itself (PAGES task); this is the data
    it (and the drift ledger built on top of it) actually fetches."""
    return {"incidents": [i.model_dump(mode="json") for i in c.incidents.all()]}


# Which CLI command produces which section of eval/last_run.json — named
# once, here, so both the "not computed yet" and "partial" branches of
# `evals_summary` below quote the exact same commands.
_EVAL_COMMANDS = {
    "system": "uv run beatroot eval system",
    "components": "uv run beatroot eval components",
}


def _section_age_seconds(computed_at: str) -> float:
    import datetime as dt

    computed = dt.datetime.fromisoformat(computed_at)
    return (dt.datetime.now(dt.UTC) - computed).total_seconds()


@app.get("/evals/summary")
def evals_summary() -> dict[str, Any]:
    """Latest system + component eval numbers — EVALS PAGE task.

    Reads `eval/last_run.json` (`eval.artifact.read_artifact`) and returns
    immediately. Never computes anything: the fix for the bug this route
    used to carry — the real system suite AND 200 synthetic profiles
    generated and run through the whole agent, both INSIDE the request
    handler, a batch job that never returned (>100s, unresolved) rather
    than a web request. `beatroot eval system` / `beatroot eval
    components` are the only things that ever produce this artifact
    (`eval.artifact.write_system_result` / `write_components_result`); this
    route has no refresh affordance and never will — running a suite
    on-demand from a request handler is exactly the mistake being fixed.

    No `ContainerDep` — this route never touches the agent, the catalog,
    or any other Container-owned dependency, which is itself proof it
    cannot block on one.

    Three shapes, always HTTP 200 (a caller with no data yet is not an
    error condition):

    - Neither section has ever been written: `status: "not_computed"`,
      both sections `null`, `commands` names the two CLI invocations that
      would produce them.
    - One section is present, the other never written (e.g. only
      `eval system` has ever run): `status: "partial"`, the present
      section's real numbers, `commands` names only what's still missing.
    - Both present: `status: "ok"`. `computed_at`/`age_seconds` are keyed
      off the OLDER of the two sections' timestamps — the honest "how
      stale is what you're looking at" answer, not the newer, more
      flattering one.
    """
    from beatroot.eval.artifact import read_artifact

    data = read_artifact()
    system_section = data.get("system") if data else None
    components_section = data.get("components") if data else None

    if system_section is None and components_section is None:
        return {
            "status": "not_computed",
            "detail": "No eval results yet — run the CLI to generate them.",
            "commands": list(_EVAL_COMMANDS.values()),
            "system": None,
            "components": None,
            "computed_at": None,
            "age_seconds": None,
        }

    missing = [
        name
        for name, section in (("system", system_section), ("components", components_section))
        if section is None
    ]
    present_ts = [s["computed_at"] for s in (system_section, components_section) if s]
    oldest = min(present_ts)

    return {
        "status": "partial" if missing else "ok",
        "detail": (
            "missing: " + ", ".join(f"{name} (run `{_EVAL_COMMANDS[name]}`)" for name in missing)
            if missing
            else None
        ),
        "commands": [_EVAL_COMMANDS[name] for name in missing],
        "system": system_section,
        "components": components_section,
        "computed_at": oldest,
        "age_seconds": round(_section_age_seconds(oldest), 1),
    }


@app.get("/evals/history")
def evals_history(limit: int = 100) -> dict[str, Any]:
    """Every persisted iteration snapshot (`eval.history.load_history`),
    oldest first, most-recent-`limit` kept — EVAL ITERATION LOOP task.

    Same posture as `GET /evals/summary`: reads whatever `beatroot eval
    iterate` already wrote to `eval/history/*.json` and returns
    immediately. It never runs a suite itself. An empty/never-run history
    is an empty `entries` list, not an error — the dashboard renders "no
    iterations recorded yet" for that case rather than failing.
    """
    from beatroot.eval.history import load_history

    entries = load_history()
    bounded = max(1, min(limit, 500))
    return {"entries": entries[-bounded:]}


@app.get("/evals/logs")
def evals_logs(limit: int = 200) -> dict[str, Any]:
    """The last `limit` structured log lines this process has emitted — a
    bounded in-memory ring buffer (`obs.logging.recent_logs`), NEVER full
    history. `ring_buffer_size` rides along explicitly so a caller (the
    evals page) can say "last N lines" honestly rather than implying this
    is everything."""
    bounded = max(1, min(limit, RING_BUFFER_SIZE))
    return {"ring_buffer_size": RING_BUFFER_SIZE, "lines": recent_logs(bounded)}


class FeedbackRequest(BaseModel):
    """Feedback on ONE previously recommended recipe. `recipe_id` must
    resolve in the catalog — the affinity update is keyed on that recipe's
    tags, not on anything the caller asserts about it."""

    profile_id: str = Field(min_length=1, max_length=128)
    recipe_id: str = Field(min_length=1, max_length=128)
    accepted: bool


@app.post("/feedback", response_model=None)
def feedback(req: FeedbackRequest, c: ContainerDep) -> dict[str, Any] | JSONResponse:
    """Record acceptance/rejection feedback and return the profile's
    updated per-tag affinity, so the effect of this one call is visible
    immediately rather than trusting it happened. Spec §9.
    """
    recipe = c.catalog.recipe(req.recipe_id)
    if recipe is None:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "recipe_id": req.recipe_id},
        )
    c.preferences.record(req.profile_id, recipe.tags, accepted=req.accepted)
    return {"ok": True, "affinity": c.preferences.affinity(req.profile_id)}


@app.get("/audit/{audit_id}", response_model=None)
def audit(audit_id: str, c: ContainerDep) -> dict[str, Any] | JSONResponse:
    row = c.audit.get(audit_id)
    if row is None:
        return JSONResponse(status_code=404, content={"error": "not_found", "audit_id": audit_id})
    return {
        **row,
        "payload": json.loads(row["payload"]),
        "skill_versions": json.loads(row["skill_versions"]),
    }

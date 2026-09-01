"""Tracing: one Langfuse generation span per model call. Spec §13.

WHY THIS INSTRUMENTS DIRECTLY INSTEAD OF USING LiteLLM'S CALLBACK
-----------------------------------------------------------------
The obvious design is `litellm.success_callback = ["langfuse_otel"]` — let
the client library emit spans, including for calls served by a `fallbacks`
entry. That is what this module did, and on this deployment it exported
nothing from the server while working perfectly from every script, which
took four wrong fixes to pin down:

1. The registered callback was `"langfuse"`, which drives the Langfuse **v2**
   SDK. Against the installed v4 SDK it fails inside LiteLLM's logger init,
   and LiteLLM swallows that as non-blocking — the app serves happily with
   tracing dark. `"langfuse_otel"` is the v3+ equivalent.
2. `LANGFUSE_HOST` defaults to the US cloud, so EU-region keys authenticate
   against nothing. The SDK's own `auth_check()` cannot detect this: it
   returns True against both regions. `verify_langfuse_auth()` makes a real
   authenticated API call instead.
3. LiteLLM logs a success **asynchronously, after the response is returned**,
   and builds its tracer provider lazily at that moment. OpenTelemetry only
   force-flushes at process exit — which a server never reaches. So
   short-lived processes exported every time and the long-running server
   exported nothing, with every diagnostic reporting healthy.
4. Flushing from response middleware therefore ran BEFORE the span existed,
   and flushed an empty provider while reporting success.

Instrumenting the call directly removes the ordering problem rather than
working around it: `observe_generation` opens the span before the request
and closes it after, on the calling thread. It also buys two things the
callback could not — each generation is linked to the exact Langfuse PROMPT
VERSION that produced it, and cost is reported from the figure this system
already computed rather than re-derived from a model name.

The LiteLLM callback is deliberately NOT registered alongside this. Two
independent instrumentations of the same call double-count cost and produce
duplicate observations, and a second mechanism that is currently silent is
worse than none: it looks like redundancy and behaves like a trap.

WHAT STILL NEEDS FLUSHING
-------------------------
The Langfuse SDK buffers spans and flushes on `flush()` or at process exit.
`_PeriodicFlusher` ticks so a long-running server does not hold traces
forever, and flushes once more on shutdown.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from beatroot.obs.logging import current_request_id
from beatroot.settings import get_settings

log = logging.getLogger("beatroot.obs.tracing")

# How model calls are traced. Named so `/health` and the CLI can report the
# mechanism without repeating a string, and so the deliberate absence of the
# LiteLLM callback is visible rather than implied.
INSTRUMENTATION = "langfuse-sdk (direct generation spans)"

# The LiteLLM callback this module deliberately does NOT register. Kept as a
# named constant so the decision is greppable and testable — a future author
# adding it back gets a failing test explaining the double-counting, rather
# than silence. See the module docstring.
UNUSED_LITELLM_CALLBACK = "langfuse_otel"


def configure_observability() -> None:
    """Prepare tracing if credentials exist. A clean no-op otherwise.

    A reviewer with a blank `.env` must never hit a credential error:
    `settings.obs.langfuse_enabled` is true only when BOTH Langfuse keys are
    present.

    Does NOT register any LiteLLM callback — spans come from
    `observe_generation`, wrapped around the call itself. It exports the
    credentials to the process environment because other OpenTelemetry-aware
    tooling reads them from there by convention, and warms the SDK client so
    the first request does not pay for constructing an exporter.
    """
    settings = get_settings()
    if not settings.obs.langfuse_enabled:
        return

    # `settings.py` owns every environment access in this codebase, reads
    # and writes alike, so the export lives there rather than here.
    settings.export_observability_env()
    _client()


def verify_langfuse_auth(timeout: float = 20.0) -> dict[str, Any]:
    """Prove the configured Langfuse credentials and host actually work,
    with a real authenticated request rather than a hopeful one.

    Returns a plain dict with the keys `configured`, `host`, `ok`, `detail`
    and `projects`, so both the CLI and a health surface can render it
    without either inventing its own shape. Never raises: a diagnostic that
    can take the caller down is not a diagnostic. A missing configuration
    reports `configured=False`, which is a legitimate state (the keyless
    path), not a failure.

    Deliberately NOT `Langfuse.auth_check()`: that method returns True
    against both the EU and US endpoints no matter which one holds the
    project, so it cannot distinguish a working configuration from a
    misdirected one — the single most likely Langfuse misconfiguration
    there is. `/api/public/projects` answers 200 only from the region that
    actually owns these keys, and names the project it found.
    """
    import base64

    import httpx

    settings = get_settings()
    obs = settings.obs
    if not obs.langfuse_enabled:
        return {
            "configured": False,
            "host": None,
            "ok": False,
            "detail": "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not both set.",
            "projects": [],
        }

    host = (obs.langfuse_host or "https://us.cloud.langfuse.com").rstrip("/")
    token = base64.b64encode(
        f"{obs.langfuse_public_key}:{obs.langfuse_secret_key}".encode()
    ).decode()
    try:
        response = httpx.get(
            f"{host}/api/public/projects",
            headers={"Authorization": f"Basic {token}"},
            timeout=timeout,
        )
    except Exception as exc:
        return {
            "configured": True,
            "host": host,
            "ok": False,
            "detail": f"could not reach {host}: {type(exc).__name__}: {exc}",
            "projects": [],
        }

    if response.status_code != 200:
        # The overwhelmingly common cause is a right-keys/wrong-region
        # pairing, so the remedy is named rather than left to be guessed.
        return {
            "configured": True,
            "host": host,
            "ok": False,
            "detail": (
                f"{host} answered {response.status_code}. If this is 401, the keys are "
                "almost certainly valid for the OTHER Langfuse region — set LANGFUSE_HOST "
                "to https://cloud.langfuse.com (EU) or https://us.cloud.langfuse.com (US)."
            ),
            "projects": [],
        }

    try:
        names = [p.get("name", "?") for p in response.json().get("data", [])]
    except Exception:  # pragma: no cover - a 200 with an unparseable body
        names = []
    return {
        "configured": True,
        "host": host,
        "ok": True,
        "detail": f"authenticated against {host}",
        "projects": names,
    }


def _span_providers() -> list[Any]:
    """Every OpenTelemetry provider that might be holding beatroot spans.

    LiteLLM does NOT necessarily register its provider globally: it builds
    its own `TracerProvider` and keeps it on the logger instance
    (`_tracer_provider`, plus a `_tracer_provider_cache` of per-config
    providers). So `trace.get_tracer_provider()` — the obvious thing to
    flush — is usually the no-op default and flushing it achieves exactly
    nothing, silently. That is precisely the first version of this function,
    and it is why the fix looked applied while traces still never arrived.
    Collect the real providers off the logger instances instead.
    """
    import litellm

    candidates: list[Any] = []
    seen: set[int] = set()

    sources: list[Any] = []
    for attribute in (
        "success_callback",
        "failure_callback",
        "callbacks",
        "_async_success_callback",
    ):
        sources.extend(getattr(litellm, attribute, None) or [])
    try:
        from litellm.litellm_core_utils.litellm_logging import _in_memory_loggers

        sources.extend(_in_memory_loggers or [])
    except Exception as exc:  # pragma: no cover - internal layout differs by version
        log.debug("could not read litellm's in-memory loggers: %s", exc)

    for logger in sources:
        for provider in (
            getattr(logger, "_tracer_provider", None),
            *(getattr(logger, "_tracer_provider_cache", None) or {}).values(),
        ):
            if (
                provider is not None
                and hasattr(provider, "force_flush")
                and id(provider) not in seen
            ):
                seen.add(id(provider))
                candidates.append(provider)

    try:
        from opentelemetry import trace

        global_provider = trace.get_tracer_provider()
        if hasattr(global_provider, "force_flush") and id(global_provider) not in seen:
            candidates.append(global_provider)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not read the global tracer provider: %s", exc)

    return candidates


def flush_traces(timeout_millis: int = 5000) -> bool:
    """Force any buffered spans out to Langfuse. True if anything flushed.

    THIS IS NOT BELT-AND-BRACES; without it the server exports nothing.
    Measured on this deployment: a short-lived process (a CLI run, a probe
    script) exports fine, because OpenTelemetry force-flushes from an
    `atexit` hook when the process ends. The long-running uvicorn server
    never ends, so its spans sat in the batch processor's queue
    indefinitely — a live `/recommend` making three real model calls
    produced zero traces while an otherwise identical single-shot script
    produced one every time.

    That is the worst shape an observability bug can take: every place you
    would think to check says it is configured. The credentials
    authenticate, the callback is registered, the logger is constructed,
    the same code exports correctly from a script — and the product that
    actually serves users is dark.

    Called from the API's request middleware once the response is already
    on its way out, so the export cost never lands on the caller's latency.
    Never raises: an observability failure must not become a request
    failure.
    """
    flushed = False

    # The Langfuse SDK client owns its OWN exporter and buffers spans the
    # same way LiteLLM's provider does — flushing one and not the other
    # leaves exactly half the instrumentation dark, which is how this
    # function looked correct while still exporting nothing.
    client = _client()
    if client is not None:
        try:
            client.flush()
            flushed = True
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("langfuse client flush failed: %s", exc)

    for provider in _span_providers():
        try:
            provider.force_flush(timeout_millis)
            flushed = True
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("provider flush failed: %s", exc)
    return flushed


class _PeriodicFlusher:
    """A daemon thread that force-flushes buffered spans on an interval.

    WHY A TIMER AND NOT A FLUSH PER REQUEST — this is the whole finding, and
    it cost several wrong fixes to locate. LiteLLM logs a successful call
    **asynchronously, after the response has been returned**
    (`_async_success_handler_body`), and it builds the tracer provider
    lazily at that moment, per request, caching it on the logger. So a flush
    called from response middleware runs BEFORE the span it is trying to
    flush has been created — it dutifully flushes an empty provider and
    reports success. That version of the fix looked applied, logged nothing
    wrong, and changed nothing.

    Combined with the fact that OpenTelemetry only force-flushes on process
    exit, a long-running server exported nothing at all, while every
    short-lived script exported correctly. A timer sidesteps the ordering
    problem entirely: it does not care when the span was created, only that
    it exists by the next tick.

    `flush()` is also called once at shutdown, so spans from the final
    seconds of a process are not lost on a rolling restart.
    """

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="otel-flush", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            flush_traces()

    def stop(self) -> None:
        """Stop ticking and flush one last time, so the final requests
        before a shutdown still reach Langfuse."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        flush_traces()


def start_trace_flusher(interval_seconds: float = 5.0) -> _PeriodicFlusher | None:
    """Begin periodic span export, or `None` when tracing is not configured.

    Returning `None` on the keyless path keeps the caller's shutdown logic
    honest — there is nothing to stop, and no thread was started.
    """
    if not get_settings().obs.langfuse_enabled:
        return None
    flusher = _PeriodicFlusher(interval_seconds)
    flusher.start()
    return flusher


@lru_cache(maxsize=1)
def _client() -> Any | None:
    """A process-wide Langfuse SDK client, or `None` when unconfigured or
    the optional SDK is absent. Cached because constructing one opens an
    exporter; never raises, because tracing must not be able to break a
    request."""
    obs = get_settings().obs
    if not obs.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=obs.langfuse_public_key,
            secret_key=obs.langfuse_secret_key,
            host=obs.langfuse_host or None,
        )
    except Exception as exc:
        log.warning("Langfuse client unavailable (%s); tracing disabled", exc)
        return None


@contextmanager
def observe_generation(
    stage: str, prompt: str, model: str, prompt_ref: Any | None = None
) -> Iterator[Any | None]:
    """Open a Langfuse generation span around one model call.

    WHY THE SDK AND NOT LiteLLM'S CALLBACK — this replaced
    `success_callback=["langfuse_otel"]`, which was measured, on this
    deployment, to export from short-lived processes and NOT from the
    long-running server. The cause is a two-part ordering problem: LiteLLM
    logs a success asynchronously AFTER the response is returned and builds
    its tracer provider lazily at that moment, while OpenTelemetry only
    force-flushes at process exit — which a server never reaches. Scripts
    therefore exported every time and the actual product exported nothing,
    with every diagnostic (credentials, callback list, logger construction)
    reporting healthy.

    Instrumenting the call directly removes the ordering problem instead of
    working around it: the span is created before the request and closed
    after it, on the calling thread, so there is nothing to race. It also
    buys two things the callback could not — the generation is linked to the
    exact Langfuse PROMPT VERSION that produced it (`prompt_ref`), and cost
    is reported from the number this system already computed rather than
    re-derived from a model name.

    Yields `None` when tracing is off, so the caller's `with` block is a
    plain no-op on the keyless path.
    """
    client = _client()
    if client is None:
        yield None
        return

    request_id = current_request_id()
    linked = prompt_ref_client(prompt_ref)
    try:
        # `propagate_attributes` puts every generation made while serving one
        # request into a single Langfuse session, which is what makes
        # "cost of answering this question" a number you can read rather
        # than reassemble from timestamps across three separate calls.
        # `propagate_attributes` is a MODULE-level function, not a client
        # method. Calling it on the client raised AttributeError, which the
        # guard below turned into "continuing untraced" — so every request
        # ran with tracing silently disabled and one warning per call in the
        # logs. Worth stating plainly: the broad `except` here is load-
        # bearing for request safety, and it is also what let a plain typo
        # masquerade as a configuration problem for several rounds.
        from langfuse import propagate_attributes

        metadata = prompt_ref.trace_metadata() if prompt_ref is not None else {"stage": stage}
        with (
            propagate_attributes(session_id=request_id or None),
            client.start_as_current_observation(
                name=f"beatroot.{stage}",
                as_type="generation",
                input=prompt,
                model=model,
                metadata=metadata,
                prompt=linked,
            ) as generation,
        ):
            yield generation
    except Exception as exc:
        # Tracing must never take a request down. If the span cannot be
        # opened at all, run the call untraced rather than failing it.
        log.warning(
            "could not open a trace span for stage=%s (%s); continuing untraced", stage, exc
        )
        yield None


def prompt_ref_client(prompt_ref: Any | None) -> Any | None:
    """The raw Langfuse prompt object behind `prompt_ref`, if it was fetched
    remotely — this is what makes Langfuse's native prompt-to-generation
    linkage work. `None` for a locally-resolved prompt, which simply means
    the generation carries the provenance metadata but no link."""
    if prompt_ref is None:
        return None
    try:
        from beatroot.reasoning.prompts import remote_client

        return remote_client(getattr(prompt_ref, "id", ""))
    except Exception:  # pragma: no cover - defensive
        return None


def record_generation_result(
    generation: Any | None,
    *,
    output: str,
    prompt_tokens: int,
    completion_tokens: int,
    usd: float,
) -> None:
    """Attach the outcome to an open generation span. No-op when tracing is
    off, and never raises."""
    if generation is None:
        return
    try:
        generation.update(
            output=output,
            usage_details={
                "input": prompt_tokens,
                "output": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            cost_details={"total": usd},
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("could not record generation result: %s", exc)

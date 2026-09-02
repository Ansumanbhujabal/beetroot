"""Structured JSON logging with correlation ids. Spec §13.

Every log line is one JSON object on stderr. `bind_request` puts the
request id (and, when known, the profile id) into a `contextvars.ContextVar`
so it rides along with every log call made anywhere downstream — including
inside a LangGraph node — without threading a `request_id` parameter through
every function signature in the call graph.

Redaction is substring-based and recursive, on purpose. A first version of
this module matched an exact five-word set (`api_key`, `authorization`,
`secret_key`, `password`, `token`) case-insensitively against the whole key
and called that "unconditional" — it was not: `Api-Key`, `x-api-key`,
`openai_api_key`, `access_token` and `client_secret` all sailed through
unredacted, and a credential nested one level down — the realistic shape of
logging a settings object or a request body, under a wrapping key such as
`config` or `headers` — was never even inspected.

`_is_sensitive` now matches a normalised (lower-cased, separators stripped)
SUBSTRING against a marker list, and `_redact` walks dicts/lists/tuples
recursively (depth-capped) rather than only the top level of one `extra`
payload. `"key"` as a marker will over-redact benign names like `cache_key`
or `sort_key` — that is the correct trade: an over-redacted debug field
costs a developer a minute of squinting; an under-redacted credential costs
an incident. Do not narrow the marker list to make a cosmetic case pass.

Two things this still does NOT catch, by construction, and both are
mitigated rather than solved:

- A secret interpolated directly into the message string via an f-string
  or % formatting, rather than passed through `extra=`. No formatter can
  find a credential in an opaque string without a false-positive-prone
  content scan, and this module never attempts one on the `message` field.
  The only real fix is at the call site: pass secrets through `extra=`,
  never bake them into the message text. Documented limitation, not a
  hole nobody is aware of.
- Exception tracebacks (`formatException`) are free text from arbitrary
  library internals (`litellm`, `httpx`, ...), and a traceback CAN contain
  a header value or an interpolated credential verbatim. `_scrub_traceback`
  below is a best-effort regex pass over the formatted traceback for
  credential-SHAPED substrings (an API-key-style token, a bearer-style
  auth header, a long base64-ish run) — heuristic, not a guarantee. It
  catches the realistic accidents, not every conceivable shape a secret
  could take in free text.
"""

import contextvars
import json
import logging
import re
import sys
import warnings
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_profile_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("profile_id", default=None)

# Substrings, not exact matches, checked against a key with `-`, `_` and `.`
# stripped out — `openai_api_key`, `x-api-key`, `Api-Key` and `client_secret`
# are all the same hazard as `api_key`/`secret`, just spelled differently.
# Deliberately broad: "key" alone will also redact `cache_key`/`sort_key`.
# That over-redaction is intentional (see module docstring) — do not trim
# this list down to the minimal set that makes a particular test pass.
SENSITIVE_MARKERS = (
    "key",
    "secret",
    "token",
    "password",
    "passwd",
    "auth",
    "credential",
    "session",
    "cookie",
    "signature",
)

_MAX_REDACT_DEPTH = 6

# Everything a bare LogRecord already carries, plus the two computed fields
# `format()` adds itself (`asctime`, `message`) — anything else in
# `record.__dict__` came in through `extra={...}` at the call site and is a
# genuine payload field, not logging machinery.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime",
    "message",
}

# Best-effort, heuristic patterns for credential-shaped text inside a
# formatted traceback — see the module docstring's limitations section.
_TRACEBACK_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),  # OpenAI/Anthropic-style API keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{8,}"),  # `Authorization: Bearer <token>`
    re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b"),  # long base64-ish runs
)


def _is_sensitive(key: str) -> bool:
    normalised = key.lower().replace("-", "").replace("_", "").replace(".", "")
    return any(marker in normalised for marker in SENSITIVE_MARKERS)


def _redact(value: Any, depth: int = 0) -> Any:
    """Recursively replace sensitive-keyed values with `"[redacted]"`
    anywhere inside dicts, lists and tuples — not just at the top level of
    one `extra` payload. `depth` caps recursion so a cyclic or pathologically
    nested structure cannot hang the logger; anything past the cap is
    replaced wholesale rather than descended into further."""
    if depth > _MAX_REDACT_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        return {
            k: ("[redacted]" if _is_sensitive(str(k)) else _redact(v, depth + 1))
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(v, depth + 1) for v in value]
    return value


def _scrub_traceback(text: str) -> str:
    """Best-effort masking of credential-shaped substrings in a formatted
    traceback. Heuristic, not a guarantee — see the module docstring."""
    for pattern in _TRACEBACK_PATTERNS:
        text = pattern.sub("[redacted]", text)
    return text


class JSONFormatter(logging.Formatter):
    """Renders one `logging.LogRecord` as one JSON object."""

    def _build_payload(self, record: logging.LogRecord) -> dict[str, Any]:
        """The redacted, JSON-safe-VALUED dict `format()` serialises — split
        out so `RingBufferHandler` (EVALS PAGE task) can hold the same
        structured payload every log line already gets, without formatting
        to a string and parsing it back."""
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": _request_id.get(),
            "profile_id": _profile_id.get(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD:
                continue
            payload[key] = "[redacted]" if _is_sensitive(str(key)) else _redact(value)
        if record.exc_info:
            payload["exception"] = _scrub_traceback(self.formatException(record.exc_info))
        return payload

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(self._build_payload(record), default=str)


# EVALS PAGE task: a bounded, in-memory ring buffer of recent structured log
# lines — the LAST `RING_BUFFER_SIZE` records this PROCESS has emitted, not
# full history and not a durable store. A module-level deque (not something
# `configure_logging` recreates on every call) so the buffer survives across
# the repeat `configure_logging()` calls every `build_container()` makes —
# each CLI command and every test that builds a fresh Container would
# otherwise wipe it right back to empty.
RING_BUFFER_SIZE = 500
_log_ring: deque[dict[str, Any]] = deque(maxlen=RING_BUFFER_SIZE)


class RingBufferHandler(logging.Handler):
    """Appends the same redacted payload `JSONFormatter` would emit to a
    bounded `deque` instead of a stream — `recent_logs()` below reads it
    back. A formatting failure on one malformed record is swallowed (best
    effort observability, never a reason to break logging itself)."""

    def __init__(self, buffer: deque[dict[str, Any]], formatter: JSONFormatter) -> None:
        super().__init__()
        self._buffer = buffer
        self._json_formatter = formatter

    def emit(self, record: logging.LogRecord) -> None:
        # logging must never raise into the caller — a malformed record is
        # simply dropped from the ring buffer, same posture as every other
        # best-effort observability path in this module.
        with suppress(Exception):
            self._buffer.append(self._json_formatter._build_payload(record))


def recent_logs(limit: int = 200) -> list[dict[str, Any]]:
    """The most recent (oldest-first) `min(limit, RING_BUFFER_SIZE)`
    structured log records this process has emitted. Explicitly a
    bounded window, never a claim of full history — callers (GET
    /evals/logs) must say so plainly rather than let a UI imply this is
    everything."""
    items = list(_log_ring)
    return items[-limit:] if limit < len(items) else items


class _SuppressLiteLLMLoggingWorkerTeardownNoise(logging.Filter):
    """Silences one specific, benign `asyncio` ERROR line.

    LiteLLM runs its own background `LoggingWorker` (an `asyncio.Task`
    named `LoggingWorker._worker_loop`) to ship success/error callbacks
    without blocking the request path. When the interpreter (or a test's
    event loop) tears down while that task is still pending, CPython's
    default asyncio exception handler logs it at ERROR:
    "Task was destroyed but it is pending!" with a traceback pointing at
    `LoggingWorker._worker_loop`. That is LiteLLM's own housekeeping being
    garbage-collected, not a fault in this application — but at ERROR it
    reads exactly like one on the evals log page.

    Deliberately narrow: this filter drops only records whose message
    starts with asyncio's exact "Task was destroyed..." text AND mentions
    `LoggingWorker` — every other message on the `asyncio` logger (a real
    "Task exception was never retrieved", a genuine destroyed-task warning
    from OUR OWN code) still gets through at its normal level. Do not
    widen this to `logging.getLogger("asyncio").setLevel(...)` — that
    would blanket-silence real asyncio errors along with this one.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        is_known_noise = (
            message.startswith("Task was destroyed but it is pending!")
            and "LoggingWorker" in message
        )
        return not is_known_noise


def configure_logging(level: int = logging.INFO) -> None:
    """Point the root logger at one stderr handler emitting JSON, plus one
    in-memory ring-buffer handler (EVALS PAGE task) so recent log lines are
    pollable without tailing a file.

    Called once at startup (top of `Container.build_container`, which both
    the CLI and the FastAPI lifespan already go through) — never at import
    time, so test output never depends on import order. Safe to call more
    than once: it always leaves exactly the same two handlers on the root
    logger (the ring buffer's own deque, `_log_ring`, is a module global
    and is never recreated here, so its contents survive a repeat call).
    """
    json_formatter = JSONFormatter()
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(json_formatter)
    ring_handler = RingBufferHandler(_log_ring, JSONFormatter())
    root = logging.getLogger()
    root.handlers = [stream_handler, ring_handler]
    root.setLevel(level)

    # See `_SuppressLiteLLMLoggingWorkerTeardownNoise` — narrowly targets
    # LiteLLM's own logging-worker teardown message, not the `asyncio`
    # logger as a whole. Idempotent: skip re-adding on a repeat call.
    asyncio_logger = logging.getLogger("asyncio")
    if not any(
        isinstance(f, _SuppressLiteLLMLoggingWorkerTeardownNoise) for f in asyncio_logger.filters
    ):
        asyncio_logger.addFilter(_SuppressLiteLLMLoggingWorkerTeardownNoise())

    # The same LiteLLM teardown, arriving by a different route. Alongside the
    # asyncio ERROR above, CPython's garbage collector emits a RuntimeWarning
    # through the `warnings` module — not through logging, so the filter
    # above cannot see it:
    #
    #   RuntimeWarning: coroutine 'LoggingWorker._worker_loop' was never awaited
    #   RuntimeWarning: coroutine 'Logging.async_success_handler' was never awaited
    #
    # It is LiteLLM's own callback plumbing being collected at interpreter
    # exit. It costs this application nothing, because tracing here does not
    # go through LiteLLM's callbacks at all — spans are emitted directly by
    # `obs.tracing.observe_generation` around the call. So the un-awaited
    # handler had no work of ours to do.
    #
    # Narrow on purpose: matched on those two exact coroutine names, so any
    # other "never awaited" warning — including one from our own code, which
    # WOULD be a real bug — still surfaces. It is printed before any output
    # the user asked for, which on `beatroot eval system` means a clean run
    # opens with two tracebacks that look like faults and are not.
    warnings.filterwarnings(
        "ignore",
        message=r"coroutine '(LoggingWorker\._worker_loop|Logging\.async_success_handler)'"
        r" was never awaited",
        category=RuntimeWarning,
    )


def current_request_id() -> str | None:
    """The correlation id bound to the current context, or `None` outside a
    request (a CLI invocation, an eval run, a test).

    Exposed because tracing needs it too, not only log records: every model
    call made while serving one request carries this as the Langfuse
    `session_id`, which is what makes the three serial calls of a single
    plan (compile, rerank, explain) read as ONE unit of work with one cost
    rather than three unrelated generations. Reading the ContextVar keeps
    `LLMClient.complete()`'s signature free of a correlation-id parameter
    that every intermediate caller would otherwise have to thread through
    — the same reason `bind_request` uses a ContextVar in the first place.
    """
    return _request_id.get()


@contextmanager
def bind_request(request_id: str, profile_id: str | None = None) -> Iterator[None]:
    """Bind a correlation id (and optionally a profile id) for the duration
    of the `with` block. Uses `contextvars` rather than a thread-local or a
    module-global so it survives across `await` points and LangGraph node
    boundaries without every function in between needing a `request_id`
    parameter — and so it never leaks into a request that did not bind it,
    per-context isolation being exactly what `contextvars` is for.
    """
    request_token = _request_id.set(request_id)
    profile_token = _profile_id.set(profile_id)
    try:
        yield
    finally:
        _request_id.reset(request_token)
        _profile_id.reset(profile_token)

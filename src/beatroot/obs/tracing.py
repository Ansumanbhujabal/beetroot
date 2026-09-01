"""LiteLLM-native tracing. Spec §13.

Tracing is a LiteLLM callback, not a hand-written context manager around
`reasoning.llm.LLMClient`. LiteLLM already emits spans with model, latency,
token counts and cost for every call routed through it — including a call
served by a `fallbacks` entry, which a wrapper built around our own client's
`complete()`/`embed()` methods would silently miss entirely, and a fallback
is exactly the call whose trace matters most. Registering both
`success_callback` and `failure_callback` matters for the same reason:
failures need tracing more than successes do.

WHICH CALLBACK, AND WHY IT CHANGED
----------------------------------
This module used to register LiteLLM's `"langfuse"` callback. That callback
drives the Langfuse **v2** Python SDK, whose `Langfuse.trace()` /
`Langfuse.generation()` methods no longer exist in the v3+ SDK (4.x), which
is OpenTelemetry-based. With `langfuse==4.15.1` installed, the old callback
does not merely degrade — it raises on import inside LiteLLM's logger
initialisation and every call logs

    [Non-Blocking Error] Error initializing custom logger: No module named 'langfuse'

which is doubly misleading: langfuse IS installed, and because LiteLLM
swallows the error as non-blocking, the application keeps working with
tracing silently dark. Registering `"langfuse_otel"` instead targets the
OTLP endpoint the v3+ SDK and the Langfuse cloud ingestion API actually
speak. `CALLBACK` below names it once so nothing has to repeat the string.

THE HOST IS NOT OPTIONAL, AND THE DEFAULT IS A TRAP
---------------------------------------------------
LiteLLM defaults an unset `LANGFUSE_HOST` to the **US** cloud endpoint. A
project on the EU cloud therefore fails authentication with no error visible
to the application, because ingestion is fire-and-forget. Worse, the SDK's
own `Langfuse(...).auth_check()` returns True against BOTH regions
regardless of which one actually holds the project — it is not a usable
probe. The only reliable check is a real authenticated API call, and
`verify_langfuse_auth()` below is exactly that: it hits
`/api/public/projects` and reports what came back, so "tracing is
configured" can be demonstrated rather than assumed. `beatroot obs check`
runs it.
"""

import logging
from typing import Any

import litellm

from beatroot.settings import get_settings

log = logging.getLogger("beatroot.obs.tracing")

# The OpenTelemetry-based Langfuse callback (v3+ SDK). See the module
# docstring for why this is not `"langfuse"`.
CALLBACK = "langfuse_otel"


def configure_observability() -> None:
    """Register LiteLLM's Langfuse OTel callback if credentials exist.

    A reviewer with a blank `.env` must never hit a credential error:
    `settings.obs.langfuse_enabled` is true only when BOTH the public and
    secret Langfuse keys are present, so this is a clean no-op otherwise.

    Idempotent by construction, not by a module-level "already ran" flag: it
    checks the REAL state of `litellm.success_callback`/`failure_callback`
    every call and only appends the callback when it is actually missing. A
    bare boolean flag set on the first call would go stale the moment
    anything else resets those lists — a test, another module, litellm
    itself — and every later call would silently no-op forever, leaving
    tracing dark for the rest of the process with no error raised anywhere.
    Checking the list itself means a reset is self-healing: the very next
    `configure_observability()` call (this function already runs once at the
    top of every `Container` build) puts the callback straight back.
    """
    settings = get_settings()
    if not settings.obs.langfuse_enabled:
        return

    # LiteLLM's callback reads LANGFUSE_* straight from the process
    # environment and offers no other channel. `settings.py` owns every
    # environment access in this codebase, reads and writes alike, so the
    # export lives there rather than here — see its docstring.
    settings.export_observability_env()

    if CALLBACK not in litellm.success_callback:
        litellm.success_callback.append(CALLBACK)
    if CALLBACK not in litellm.failure_callback:
        litellm.failure_callback.append(CALLBACK)


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

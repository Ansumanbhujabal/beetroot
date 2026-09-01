"""Session-wide test fixtures.

**The suite is hermetic on purpose.** It must produce the same result on a
machine with real Azure and Langfuse credentials as on a fresh clone with
none, because a test that passes or fails depending on whose laptop it runs
on is not evidence of anything. Two autouse fixtures enforce that:

`_hermetic_environment` (session-scoped) hides the repo's `.env` from
`Settings` and strips provider credentials from the process environment.
This is not belt-and-braces — it closes a real hole. `beatroot.settings`
loads `.env` into `os.environ` at import so the offline fallback cannot
depend on import order (see `_assemble_process_environment`), which means
that without this fixture, a developer holding real keys silently ran the
WHOLE suite against live Azure: slower, billable, non-deterministic, and
quietly exercising different code paths than CI ever would. Forcing
`BEATROOT_OFFLINE=1` here restores the documented promise that the suite
runs keyless.

`_reset_vector_store_cache` (function-scoped) resets the process-global
vector-store cache in `dense.get_vector_store`. That cache is right for
production — building a store is startup work, not per-request work — but
in tests a store built from one test's tmp_path catalog would otherwise
keep answering a later test's queries against different data. Spec §17.

`QDRANT_URL` is deliberately NOT stripped: the Qdrant store tests are
designed to skip cleanly when it is unset and to run against a real
container when it is set, and taking that choice away from whoever runs the
suite would make those tests unrunnable rather than hermetic.
"""

import os

import pytest

from beatroot.retrieval.dense import reset_vector_store
from beatroot.settings import Settings, get_settings

# Credentials and model overrides that would otherwise leak a developer's
# `.env` into the suite. QDRANT_URL is excluded by design — see the module
# docstring.
_HOST_CREDENTIALS = (
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AZURE_API_VERSION",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "BEATROOT_LLM__MODEL",
    "BEATROOT_LLM__EMBEDDING_MODEL",
)


@pytest.fixture(autouse=True, scope="session")
def _hermetic_environment():
    saved_env = {name: os.environ.get(name) for name in (*_HOST_CREDENTIALS, "BEATROOT_OFFLINE")}
    saved_env_file = Settings.model_config.get("env_file")

    for name in _HOST_CREDENTIALS:
        os.environ.pop(name, None)
    # Explicit rather than relying on the credential-absence fallback: the
    # fallback is itself under test, and a test of a fallback must not be
    # the thing that arranges for it to fire.
    os.environ["BEATROOT_OFFLINE"] = "1"
    Settings.model_config["env_file"] = "/nonexistent/.env"
    get_settings.cache_clear()

    yield

    Settings.model_config["env_file"] = saved_env_file
    for name, value in saved_env.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_vector_store_cache():
    reset_vector_store()
    yield
    reset_vector_store()

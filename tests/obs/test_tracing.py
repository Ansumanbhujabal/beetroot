"""Tracing behaviour, tested against what the code actually does.

Three things these tests learned the hard way and now encode:

1. **Spans come from `observe_generation`, not a LiteLLM callback.** The
   callback approach exported from short-lived scripts and not from the
   long-running server, for reasons in `obs.tracing`'s module docstring.
   `test_no_litellm_callback_is_registered` pins the removal so a future
   author cannot quietly reintroduce double-counted cost.

2. **`monkeypatch.delenv` alone does not simulate a blank configuration.**
   `Settings` reads `.env` as a settings SOURCE, so on a machine with real
   credentials on disk, deleting the environment variable changed nothing
   and the "no credentials" tests failed. `_neutralise_dotenv` points the
   loader at a path that does not exist.

3. **Tracing must never break a request.** Every helper here degrades to a
   no-op rather than raising: an observability failure taking down a meal
   recommendation is a strictly worse outcome than losing the trace.
"""

import litellm
import pytest

from beatroot.obs.tracing import (
    INSTRUMENTATION,
    UNUSED_LITELLM_CALLBACK,
    configure_observability,
    observe_generation,
    record_generation_result,
    verify_langfuse_auth,
)
from beatroot.settings import Settings, get_settings

_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


@pytest.fixture(autouse=True)
def _neutralise_dotenv(monkeypatch):
    """Hide the repo's real `.env` from `Settings`, and undo any environment
    these tests write directly.

    `export_observability_env()` writes to `os.environ` by design, which
    `monkeypatch` cannot undo because it never saw the write: an exported
    key leaked out of this module once and made
    `tests/test_settings.py::test_obs_disabled_by_default` fail later in the
    session, a failure with no visible connection to its cause.
    """
    import os

    import beatroot.obs.tracing as tracing

    saved = {key: os.environ.get(key) for key in _KEYS}
    monkeypatch.setitem(Settings.model_config, "env_file", "/nonexistent/.env")
    get_settings.cache_clear()
    # `_client` is lru_cached for the process, so a client constructed by an
    # earlier test would answer a later "no credentials" one and try a real
    # export (observed as `Failed to export span batch code: 401`). Clearing
    # both caches is what makes each test start from genuinely blank state.
    tracing._client.cache_clear()
    yield
    # A test may have monkeypatched `_client` to a plain function; this
    # fixture tears down BEFORE monkeypatch restores it, so reach for
    # cache_clear defensively rather than assuming the lru_cache is back.
    getattr(tracing._client, "cache_clear", lambda: None)()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


def _blank(monkeypatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


def _configured(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    get_settings.cache_clear()


def test_configure_is_a_clean_no_op_without_credentials(monkeypatch):
    """A reviewer with a blank .env must never hit a credential error."""
    _blank(monkeypatch)
    configure_observability()  # must not raise


def test_no_litellm_callback_is_registered(monkeypatch):
    """Spans are emitted by `observe_generation`, wrapped around the call.

    Registering LiteLLM's Langfuse callback as well would instrument the
    same call twice: duplicate observations and double-counted cost, which
    would then disagree with `/metrics`. That is the reason the callback was
    removed, so it gets a test rather than a comment.
    """
    _configured(monkeypatch)
    litellm.success_callback = []
    litellm.failure_callback = []
    configure_observability()
    assert UNUSED_LITELLM_CALLBACK not in litellm.success_callback
    assert UNUSED_LITELLM_CALLBACK not in litellm.failure_callback


def test_instrumentation_is_named_for_health_reporting():
    assert "langfuse" in INSTRUMENTATION.lower()


def test_credentials_are_exported_to_the_environment(monkeypatch):
    """OpenTelemetry-aware tooling reads LANGFUSE_* from `os.environ` by
    convention, so configuring observability publishes them there."""
    import os

    _blank(monkeypatch)
    # Mutating the cached Settings directly is safe here: the autouse
    # fixture clears the lru_cache on the way out, so nothing leaks.
    settings = get_settings()
    settings.langfuse_public_key = "pk-exported"
    settings.langfuse_secret_key = "sk-exported"
    settings.langfuse_host = "https://us.cloud.langfuse.com"
    settings.export_observability_env()
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-exported"
    assert os.environ["LANGFUSE_HOST"] == "https://us.cloud.langfuse.com"


def test_an_existing_environment_value_is_never_overridden(monkeypatch):
    """An operator's `docker run -e` must outrank the config file."""
    import os

    _blank(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-from-operator")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-from-operator")
    settings = get_settings()
    settings.langfuse_public_key = "pk-from-config"
    settings.langfuse_secret_key = "sk-from-config"
    settings.export_observability_env()
    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-from-operator"


def test_observe_generation_is_a_no_op_without_credentials(monkeypatch):
    """The keyless path must still run the model call, untraced."""
    _blank(monkeypatch)
    with observe_generation("explain", "a prompt", "azure/gpt-4o") as generation:
        assert generation is None
    # And recording against that None must not raise.
    record_generation_result(None, output="text", prompt_tokens=1, completion_tokens=1, usd=0.0)


def test_observe_generation_never_raises_into_the_caller(monkeypatch):
    """Tracing failures must degrade to an untraced call, never propagate.

    This guard once hid a plain typo — `propagate_attributes` called on the
    client instead of the module — for several fix rounds, because a broad
    `except` turned "wrong attribute name" into "continuing untraced". The
    behaviour is still correct and still wanted; the test pins it so the
    degradation is deliberate rather than incidental.
    """
    _configured(monkeypatch)
    import beatroot.obs.tracing as tracing

    class _Exploding:
        def __getattr__(self, name):
            raise RuntimeError("tracing backend is down")

    monkeypatch.setattr(tracing, "_client", lambda: _Exploding())
    with observe_generation("explain", "a prompt", "azure/gpt-4o") as generation:
        assert generation is None, "a broken backend must yield an untraced call"


def test_auth_verification_reports_unconfigured_rather_than_failing(monkeypatch):
    """`verify_langfuse_auth` is a diagnostic, so the keyless path is a
    reportable state, not an error — and it must never raise."""
    _blank(monkeypatch)
    result = verify_langfuse_auth()
    assert result["configured"] is False
    assert result["ok"] is False
    assert result["projects"] == []

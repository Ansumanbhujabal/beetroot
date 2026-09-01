"""Tracing registration, tested against the callback the code actually uses.

Two things these tests learned the hard way and now encode:

1. **The callback name is `CALLBACK`, not a literal.** These tests used to
   assert on the string `"langfuse"`. That callback drives the Langfuse v2
   SDK, whose API no longer exists in the installed v3+ SDK — so the tests
   went on passing while tracing was dead in every real run. Asserting on
   the constant the production code registers means renaming it cannot
   leave a green suite behind a dark exporter.

2. **`monkeypatch.delenv` alone does not simulate a blank configuration.**
   `Settings` reads `.env` as a settings SOURCE, so on a developer machine
   with real credentials on disk, deleting the environment variable changed
   nothing and the "no credentials" tests failed. `_neutralise_dotenv`
   below points the settings loader at a path that does not exist, so
   "blank `.env`" means blank here rather than "blank unless the person
   running the suite happens to have keys".
"""

import litellm
import pytest

from beatroot.obs.tracing import CALLBACK, configure_observability, verify_langfuse_auth
from beatroot.settings import Settings, get_settings

_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


@pytest.fixture(autouse=True)
def _neutralise_dotenv(monkeypatch):
    """Make the repo's real `.env` invisible to `Settings` for these tests,
    and undo any environment these tests write directly.

    Two separate hazards, one fixture. Without the `env_file` override,
    every assertion below about "no credentials" is really an assertion
    about whether whoever ran the suite has Langfuse configured — not a
    property of this code. And `export_observability_env()` writes to
    `os.environ` by design, which `monkeypatch` cannot undo because it never
    saw the write: the exported `pk-exported` key leaked out of this module
    and made `tests/test_settings.py::test_obs_disabled_by_default` fail
    later in the session, a failure with no visible connection to its cause.
    Snapshotting the keys here contains it.
    """
    import os

    saved = {key: os.environ.get(key) for key in _KEYS}
    monkeypatch.setitem(Settings.model_config, "env_file", "/nonexistent/.env")
    get_settings.cache_clear()
    yield
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


def test_no_callbacks_registered_without_credentials(monkeypatch):
    """A reviewer with a blank .env must never hit a credential error."""
    _blank(monkeypatch)
    litellm.success_callback = []
    configure_observability()
    assert CALLBACK not in litellm.success_callback


def test_callbacks_registered_when_configured(monkeypatch):
    _configured(monkeypatch)
    litellm.success_callback = []
    litellm.failure_callback = []
    configure_observability()
    assert CALLBACK in litellm.success_callback
    assert CALLBACK in litellm.failure_callback, "failures need tracing most"


def test_the_registered_callback_is_the_otel_one(monkeypatch):
    """Pins the actual value, not just self-consistency with the constant.

    `"langfuse"` (the v2-SDK callback) is broken against the installed v3+
    SDK: LiteLLM catches the import failure as non-blocking, so the app
    keeps serving with tracing silently off. A test comparing the constant
    to itself would not have caught that; this one names the string.
    """
    assert CALLBACK == "langfuse_otel"


def test_configuration_is_idempotent(monkeypatch):
    _configured(monkeypatch)
    litellm.success_callback = []
    configure_observability()
    configure_observability()
    assert litellm.success_callback.count(CALLBACK) == 1


def test_callback_is_restored_after_an_external_reset(monkeypatch):
    """The real failure mode a bare 'already configured' flag misses: if
    anything else clears litellm.success_callback after the first
    configure_observability() call — a test, another module, litellm
    internals — a later call must put the callback back, not silently
    no-op forever because a flag says "already done"."""
    _configured(monkeypatch)
    litellm.success_callback = []
    litellm.failure_callback = []
    configure_observability()
    assert CALLBACK in litellm.success_callback
    assert CALLBACK in litellm.failure_callback

    # Something external clears the lists — tracing is now silently dark.
    litellm.success_callback = []
    litellm.failure_callback = []

    configure_observability()
    assert CALLBACK in litellm.success_callback, "must self-heal after an external reset"
    assert CALLBACK in litellm.failure_callback, "must self-heal after an external reset"


def test_late_credentials_are_picked_up_on_a_later_call(monkeypatch):
    """A first call with no credentials must not poison later calls once
    credentials do become available — there is no 'configured once,
    forever' state to get stuck in."""
    _blank(monkeypatch)
    litellm.success_callback = []
    configure_observability()
    assert CALLBACK not in litellm.success_callback

    _configured(monkeypatch)
    configure_observability()
    assert CALLBACK in litellm.success_callback


def test_credentials_are_exported_to_the_environment_for_litellm(monkeypatch):
    """LiteLLM's exporter reads LANGFUSE_* from `os.environ` and accepts no
    other channel, so configuring observability has to put them there —
    otherwise credentials that live only in `config`/`.env` would produce a
    registered callback that authenticates against nothing."""
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


def test_auth_verification_reports_unconfigured_rather_than_failing(monkeypatch):
    """`verify_langfuse_auth` is a diagnostic, so the keyless path is a
    reportable state, not an error — and it must never raise."""
    _blank(monkeypatch)
    result = verify_langfuse_auth()
    assert result["configured"] is False
    assert result["ok"] is False
    assert result["projects"] == []

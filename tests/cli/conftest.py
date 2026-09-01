import pytest

from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch):
    """Every CLI test runs fully offline — no credentials, no network."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _plain_console_output(monkeypatch):
    """Strip ambient colour forcing so assertions can match plain text.

    These tests assert on substrings like `"generated 10 profiles"`, and
    Rich's number highlighter renders that as
    `generated \x1b[1;36m10\x1b[0m profiles` the moment colour is enabled.
    Rich normally disables colour when stdout is not a TTY — which is the
    case under Click's CliRunner — but `FORCE_COLOR` overrides that
    detection, and plenty of environments set it: CI runners, and some
    developer terminals. The result was four tests that passed or failed
    purely on whose shell ran them, with a diff full of escape codes and no
    hint that the code was fine.

    `NO_COLOR` is the documented opt-out (no-color.org) and Rich honours it,
    so this pins the CLI's output to plain text regardless of the host.
    """
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")

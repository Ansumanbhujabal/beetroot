import pytest
from rich.console import Console

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
    """Pin CLI output to plain text so assertions can match substrings.

    These tests assert on strings like `"generated 10 profiles"`, and Rich's
    number highlighter renders that as
    `generated \x1b[1;36m10\x1b[0m profiles` the moment colour is on. Rich
    normally disables colour when stdout is not a TTY — true under Click's
    CliRunner — but `FORCE_COLOR` overrides that detection, and plenty of
    environments set it (CI runners, some developer terminals). The result
    was tests that passed or failed purely on whose shell ran them, with a
    diff full of escape codes and no hint that the code was fine.

    Setting `NO_COLOR` alone was not enough, and the reason is worth
    recording: `cli.main` builds its `Console` at IMPORT time, so by the
    time any fixture touches the environment the console has already
    resolved its colour settings. The environment variables are still
    cleared below for any console constructed later, but REPLACING the
    module-level console is what actually makes this deterministic.
    """
    import beatroot.cli.main as cli_main

    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli_main, "console", Console(no_color=True, force_terminal=False))

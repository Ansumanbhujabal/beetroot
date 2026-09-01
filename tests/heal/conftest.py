import pytest

from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch) -> object:
    """The end-to-end healing tests build a real Container/agent, so they
    need offline mode the same way `tests/eval/conftest.py` does —
    `get_settings()` is `@lru_cache`d, so the env var must be cleared before
    AND after."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

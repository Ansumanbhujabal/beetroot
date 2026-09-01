import pytest

from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch: pytest.MonkeyPatch) -> object:
    """Every eval test runs fully offline — no credentials, no network.
    `get_settings()` is `@lru_cache`d, so the env var must be cleared before
    AND after, same as `tests/cli/conftest.py` / `tests/test_container.py`."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

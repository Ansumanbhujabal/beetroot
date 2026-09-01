"""Isolation for the obs tests.

`configure_observability()` reads through the process-wide `get_settings()`
lru_cache, which needs clearing between tests or an earlier test's env
state leaks into a later one. Root-logger handlers are saved and restored
around each test so `configure_logging()` (which deliberately replaces
`root.handlers` every call — see its docstring) never leaks a handler bound
to one test's `capsys` stream into another test.
"""

import logging

import pytest

from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _isolate_observability_state():
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

    root.handlers = saved_handlers
    root.setLevel(saved_level)

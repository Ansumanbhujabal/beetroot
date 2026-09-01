import ast
import pathlib

import pytest

from beatroot.settings import Settings, get_settings


def test_trust_weights_sum_to_one():
    """A weighting that does not sum to 1 silently rescales every trust score."""
    w = get_settings().trust.weights
    total = w.catalog_coverage + w.constraint_completeness + w.model_self_assessment
    assert total == pytest.approx(1.0)


def test_invalid_weights_are_rejected_at_load_not_at_use():
    with pytest.raises(ValueError, match="must sum to 1"):
        Settings.model_validate(
            {
                "trust": {
                    "weights": {
                        "catalog_coverage": 0.9,
                        "constraint_completeness": 0.9,
                        "model_self_assessment": 0.9,
                    },
                    "refusal_threshold": 0.55,
                    "weak_signal_floor": 0.5,
                }
            }
        )


def test_env_overrides_yaml():
    import os

    os.environ["BEATROOT_TRUST__REFUSAL_THRESHOLD"] = "0.71"
    get_settings.cache_clear()
    try:
        assert get_settings().trust.refusal_threshold == 0.71
    finally:
        del os.environ["BEATROOT_TRUST__REFUSAL_THRESHOLD"]
        get_settings.cache_clear()


def _reads_env(path: pathlib.Path) -> bool:
    """True only for actual environment access — `os.getenv(...)`,
    `os.environ[...]`/`os.environ.get(...)`, or `from os import environ` /
    `from os import getenv` — never a mention of the words in a docstring or
    comment.

    A plain substring search (`"os.getenv" in text`) cannot tell code from
    prose: it once flagged `obs/__init__.py`'s own module docstring for
    *stating* that the module does not touch the environment. Parsing the
    AST and matching on `ast.Attribute`/`ast.ImportFrom` nodes only sees
    real access, so a module is free to document this rule without tripping
    the test that enforces it.

    This is also strictly stronger than the text search it replaces: `from
    os import getenv` (no `os.` prefix at the call site) would slip straight
    past `"os.getenv" in text`, but is caught here via the `ImportFrom`
    branch.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "getenv"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            return True
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name in {"environ", "getenv"} for alias in node.names)
        ):
            return True
    return False


def test_settings_is_the_only_module_reading_env():
    """Real env access anywhere else is configuration leaking out of its
    home — see `_reads_env` for why this is an AST walk, not a grep."""
    src = pathlib.Path(__file__).parents[1] / "src" / "beatroot"
    offenders = [
        str(f.relative_to(src))
        for f in src.rglob("*.py")
        if f.name != "settings.py" and _reads_env(f)
    ]
    assert not offenders, f"env access outside settings.py: {offenders}"


def test_reads_env_ignores_prose_mentions(tmp_path):
    """A docstring that merely talks about os.environ/os.getenv is not a
    violation — this is the exact case that broke the old text-search
    version of the check against `obs/__init__.py`."""
    mod = tmp_path / "mentions_env_in_prose.py"
    mod.write_text(
        '"""This module never touches os.environ or calls os.getenv — all '
        'configuration flows through get_settings()."""\n'
    )
    assert _reads_env(mod) is False


def test_reads_env_catches_os_dot_getenv(tmp_path):
    mod = tmp_path / "reads_via_os_dot.py"
    mod.write_text("import os\n\nvalue = os.getenv('SOME_VAR')\n")
    assert _reads_env(mod) is True


def test_reads_env_catches_from_os_import_getenv(tmp_path):
    mod = tmp_path / "reads_via_from_import.py"
    mod.write_text("from os import getenv\n\nvalue = getenv('SOME_VAR')\n")
    assert _reads_env(mod) is True


def test_offline_defaults_to_true_without_credentials(monkeypatch):
    """The core keyless-boot guarantee (Task 20 fix round): no
    `BEATROOT_OFFLINE` AND no provider credentials in the environment must
    still resolve `offline` to True, not construct a real `LLMClient` that
    throws on its first `.embed()`/`.complete()` call — which, for
    `build_container()`, is before `/health` is even reachable."""
    for var in ("BEATROOT_OFFLINE", "AZURE_API_KEY", "AZURE_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().offline is True
    finally:
        get_settings.cache_clear()


def test_offline_stays_false_when_credentials_are_present(monkeypatch):
    """The flip side: real credentials for the configured provider must
    still select the real client — the fallback only ever fires on
    genuine absence, never overriding a reviewer who has actually
    supplied keys."""
    monkeypatch.delenv("BEATROOT_OFFLINE", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    get_settings.cache_clear()
    try:
        assert get_settings().offline is False
    finally:
        get_settings.cache_clear()


def test_explicit_offline_is_unaffected_by_credential_presence(monkeypatch):
    """`BEATROOT_OFFLINE=1` must win regardless of what credentials also
    happen to be present — this only ever turns False into True, never
    the reverse."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    get_settings.cache_clear()
    try:
        assert get_settings().offline is True
    finally:
        get_settings.cache_clear()


def test_offline_can_be_set_via_env():
    import os

    os.environ["BEATROOT_OFFLINE"] = "1"
    get_settings.cache_clear()
    try:
        assert get_settings().offline is True
    finally:
        del os.environ["BEATROOT_OFFLINE"]
        get_settings.cache_clear()


def test_obs_disabled_by_default():
    """A blank .env must never enable tracing."""
    assert get_settings().obs.langfuse_enabled is False


def test_obs_reads_bare_langfuse_env_names():
    """Langfuse credentials are read from the bare LANGFUSE_* names (no
    BEATROOT_ prefix) — the names LiteLLM's own integration expects."""
    import os

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"
    os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
    get_settings.cache_clear()
    try:
        obs = get_settings().obs
        assert obs.langfuse_public_key == "pk-test"
        assert obs.langfuse_secret_key == "sk-test"
        assert obs.langfuse_host == "https://cloud.langfuse.com"
        assert obs.langfuse_enabled is True
    finally:
        del os.environ["LANGFUSE_PUBLIC_KEY"]
        del os.environ["LANGFUSE_SECRET_KEY"]
        del os.environ["LANGFUSE_HOST"]
        get_settings.cache_clear()


def test_obs_needs_both_keys_to_be_enabled():
    """Only a public key, or only a secret key, is not a configured Langfuse
    — half a credential pair must still no-op cleanly."""
    import os

    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    get_settings.cache_clear()
    try:
        assert get_settings().obs.langfuse_enabled is False
    finally:
        del os.environ["LANGFUSE_PUBLIC_KEY"]
        get_settings.cache_clear()

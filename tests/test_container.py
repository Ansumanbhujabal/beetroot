"""Tests for the composition root. Spec §15.

`build_container` raises before opening any connection or making any model
call if the skill files on disk have drifted from `skills-lock.json` (see
`beatroot.container`'s module docstring, point 2) — an audit record naming
skill versions that were never actually locked would be a false provenance
claim. That raise path had no test proving it actually fires; this file is
that proof.
"""

import shutil

import pytest

from beatroot.container import DATA_DIR, SKILLS_DIR, build_container
from beatroot.reasoning.llm import LLMClient
from beatroot.settings import get_settings


@pytest.fixture(autouse=True)
def _offline_mode(monkeypatch):
    """`get_settings()` is `@lru_cache`d — a monkeypatched env var has no
    effect unless the cache is cleared before AND after, or an earlier
    test's cached (non-offline) Settings instance silently wins here too,
    same as `tests/api/conftest.py`'s `test_container` fixture."""
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _copy_skills_with_one_drifted(tmp_path, drifted_id: str = "assess_trust"):
    """A throwaway copy of every real skill file, with one file's body
    mutated after copying — so its digest no longer matches the real,
    unmodified `skills-lock.json` while every other skill still does."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for src in SKILLS_DIR.glob("*.skill.md"):
        shutil.copy(src, skills_dir / src.name)

    drifted_path = skills_dir / f"{drifted_id}.skill.md"
    drifted_path.write_text(drifted_path.read_text() + "\nTHIS BODY WAS MUTATED FOR A TEST.\n")
    return skills_dir


def test_build_container_raises_on_skill_lock_drift(tmp_path):
    """Copy the six real skills into tmp_path, mutate one file's body, point
    `build_container` at that directory, and confirm it refuses to start —
    naming the drifted skill — instead of silently serving requests under
    rules that were never locked."""
    skills_dir = _copy_skills_with_one_drifted(tmp_path, "assess_trust")

    with pytest.raises(RuntimeError, match="assess_trust") as exc_info:
        build_container(db_path=tmp_path / "test.db", skills_dir=skills_dir)

    # The other five, untouched, copied skills must NOT be named as drifted.
    message = str(exc_info.value)
    for clean_id in (
        "check_feasibility",
        "compile_constraints",
        "compute_nutrition",
        "explain_choice",
        "retrieve_candidates",
    ):
        assert clean_id not in message


def test_build_container_succeeds_when_the_copied_skills_are_untouched(tmp_path):
    """Sanity check on the fixture above: an UNMODIFIED copy of the real
    skills must not itself trigger drift — proves the test above is
    catching the mutation, not merely the act of copying to a new path."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for src in SKILLS_DIR.glob("*.skill.md"):
        shutil.copy(src, skills_dir / src.name)

    container = build_container(db_path=tmp_path / "test.db", skills_dir=skills_dir)
    try:
        assert container.health()["skills_locked"] is True
    finally:
        container.close()


def test_langfuse_and_qdrant_are_real_dependencies_not_optional_extras():
    """A plain `uv sync` must produce an install where prompt management and
    tracing actually work.

    Both `langfuse` and `qdrant-client` used to be optional extras, and
    `uv sync` does not install extras — so the ordinary setup path produced an
    environment where prompts silently fell back to local files, tracing did
    nothing, and `QDRANT_URL` degraded to the in-memory store, while `.env`
    said all three were configured and `/health` reported
    `langfuse_configured: true`. Every signal said working; nothing was.

    The fallback in `reasoning.prompts.langfuse_client` exists for missing
    CREDENTIALS, which is a real runtime state. It should never be papering
    over a missing library, which is just a broken install.
    """
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    deps = " ".join(pyproject["project"]["dependencies"])
    for package in ("langfuse", "qdrant-client"):
        assert package in deps, f"{package} must be a real dependency, not an extra"

    extras = pyproject["project"].get("optional-dependencies", {})
    for name, packages in extras.items():
        for package in ("langfuse", "qdrant-client"):
            assert not any(package in p for p in packages), (
                f"{package} reappeared in the {name!r} extra; a plain `uv sync` "
                "would then produce a silently degraded install again"
            )

    # And they must genuinely import in this environment.
    import langfuse  # noqa: F401
    import qdrant_client  # noqa: F401


def test_a_missing_qdrant_client_degrades_instead_of_crashing(monkeypatch, tmp_path):
    """`QDRANT_URL` set without the optional client must fall back to NumPy.

    Qdrant is optional by design — the store sits behind a protocol with an
    in-memory implementation. But an ImportError here fires inside
    `build_container()`, which runs in the API's `lifespan`, so the whole
    process would die at startup before `/health` was reachable, over a
    dependency that has a working alternative sitting right below it.
    """
    import builtins

    from beatroot.container import _build_vector_store
    from beatroot.settings import get_settings
    from beatroot.store.db import connect, seed
    from beatroot.trusted.catalog import Catalog

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    get_settings.cache_clear()

    real_import = builtins.__import__

    def _no_qdrant(name, *args, **kwargs):
        if name.startswith("beatroot.retrieval.qdrant_store") or name.startswith("qdrant_client"):
            raise ImportError("qdrant-client is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_qdrant)

    conn = connect(tmp_path / "t.db")
    seed(conn, DATA_DIR)
    catalog = Catalog(conn)
    try:
        store = _build_vector_store(LLMClient.offline(), catalog, embedding_cache=None)
        assert getattr(store, "name", "") == "numpy", "must degrade to the in-memory store"
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        conn.close()
        get_settings.cache_clear()

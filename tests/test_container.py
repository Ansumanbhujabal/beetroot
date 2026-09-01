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

from beatroot.container import SKILLS_DIR, build_container
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

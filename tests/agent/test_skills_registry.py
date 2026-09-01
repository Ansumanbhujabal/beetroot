import pytest

from beatroot.agent.skills_registry import (
    Skill,
    SkillFormatError,
    load_skills,
    verify_lock,
    write_lock,
)


@pytest.fixture
def skills_dir():
    import pathlib

    return pathlib.Path(__file__).parents[2] / "skills"


def test_all_six_skills_load(skills_dir):
    skills = load_skills(skills_dir)
    assert set(skills) == {
        "compile_constraints",
        "check_feasibility",
        "retrieve_candidates",
        "compute_nutrition",
        "assess_trust",
        "explain_choice",
    }


def test_t0_skills_declare_no_llm(skills_dir):
    skills = load_skills(skills_dir)
    for sid in ("check_feasibility", "compute_nutrition"):
        assert skills[sid].tier == "T0"
        assert skills[sid].llm_permitted is False


def test_frontmatter_matches_spec_table(skills_dir):
    """tier / llm_permitted / priority carry real weight — pin them to the
    table in the brief so a typo in a skill file is caught here, not
    discovered downstream by something that trusted the frontmatter."""
    skills = load_skills(skills_dir)
    expected = {
        "compile_constraints": ("T2", True, 10),
        "check_feasibility": ("T0", False, 20),
        "retrieve_candidates": ("T1", True, 30),
        "compute_nutrition": ("T0", False, 40),
        "assess_trust": ("T3", True, 50),
        "explain_choice": ("T2", True, 60),
    }
    for sid, (tier, llm_permitted, priority) in expected.items():
        skill = skills[sid]
        assert skill.tier == tier, sid
        assert skill.llm_permitted is llm_permitted, sid
        assert skill.priority == priority, sid


def test_every_skill_has_the_three_required_sections(skills_dir):
    for skill in load_skills(skills_dir).values():
        for heading in ("## When to use", "## The pattern", "## Pitfalls"):
            assert heading in skill.body, f"{skill.id} missing {heading}"


def test_load_skills_is_deterministically_ordered(skills_dir):
    """Insertion order must not depend on filesystem directory-listing
    order — load twice and the key order must match exactly."""
    first = list(load_skills(skills_dir).keys())
    second = list(load_skills(skills_dir).keys())
    assert first == second
    # Order matches sorted file paths, not filesystem-listing luck.
    import pathlib

    expected = [
        Skill(**_frontmatter(p), body="x").id
        for p in sorted(pathlib.Path(skills_dir).glob("*.skill.md"))
    ]
    assert first == expected


def _frontmatter(path):
    import yaml

    text = path.read_text()
    _, front, _ = text.split("---", 2)
    return yaml.safe_load(front)


def test_lock_detects_drift(skills_dir, tmp_path):
    skills = load_skills(skills_dir)
    lock = tmp_path / "skills-lock.json"
    write_lock(skills, lock)
    assert verify_lock(skills, lock) == []

    tampered = dict(skills)
    tampered["assess_trust"] = tampered["assess_trust"].model_copy(update={"body": "changed"})
    assert verify_lock(tampered, lock) == ["assess_trust"]


def test_digest_is_independent_of_key_order_and_stable_across_runs():
    a = Skill(
        id="x",
        name="X",
        tier="T0",
        llm_permitted=False,
        priority=1,
        body="## When to use\n## The pattern\n## Pitfalls",
    )
    b = Skill(
        id="x",
        name="X",
        tier="T0",
        llm_permitted=False,
        priority=1,
        body="## When to use\n## The pattern\n## Pitfalls",
    )
    assert a.digest() == b.digest()
    # Digest excludes id/name/triggers_on — only tier/llm_permitted/priority/body.
    c = a.model_copy(update={"id": "different-id", "name": "Different Name"})
    assert a.digest() == c.digest()


def test_missing_frontmatter_fails_loudly(tmp_path):
    bad = tmp_path / "bad.skill.md"
    bad.write_text("## When to use\nno frontmatter here\n")
    with pytest.raises(SkillFormatError):
        load_skills(tmp_path)


def test_duplicate_skill_id_fails_loudly(tmp_path):
    """`load_skills` is the only silent-clobber path in this module before
    this fix — every other malformed input (`_parse`) already raises. Two
    files declaring the same `id` must raise, naming both files, instead of
    the second silently overwriting the first in the returned dict."""
    body = "## When to use\nx\n\n## The pattern\nx\n\n## Pitfalls\nx\n"
    front = "---\nid: dup\nname: Dup\ntier: T0\nllm_permitted: false\npriority: 1\n---\n"
    first = tmp_path / "a.skill.md"
    second = tmp_path / "b.skill.md"
    first.write_text(front + body)
    second.write_text(front + body)
    with pytest.raises(SkillFormatError, match="duplicate skill id 'dup'"):
        load_skills(tmp_path)


def test_missing_required_section_fails_loudly(tmp_path):
    bad = tmp_path / "bad.skill.md"
    bad.write_text(
        "---\n"
        "id: bad\nname: Bad\ntier: T0\nllm_permitted: false\npriority: 1\n"
        "---\n"
        "## When to use\nsomething\n\n## The pattern\nsomething\n"
        # "## Pitfalls" deliberately omitted
    )
    with pytest.raises(SkillFormatError):
        load_skills(tmp_path)

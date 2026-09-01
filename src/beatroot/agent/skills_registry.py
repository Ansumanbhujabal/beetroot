"""Skills registry and provenance lock. Spec §7.

A skill file is not internal plumbing here — it is read by a person deciding
whether to trust this system, and it is loaded at runtime to decide what the
agent is allowed to do at each state transition. `tier` and `llm_permitted`
are the two frontmatter fields that carry real weight: a T0 skill with
`llm_permitted: false` is a promise that no model call sits anywhere on that
skill's path. A skill file that fails to parse must fail LOUDLY at load —
never fall back to an empty body or a missing section, because a half-loaded
skill would silently misrepresent that promise to whatever reads the
registry next.

`Skill.digest()` is the provenance half. It hashes the fields that define
what the skill actually DOES — `tier`, `llm_permitted`, `priority`, `body`
— not `id` or `triggers_on`, which are addressing metadata rather than
content that changes what the skill authorizes. `json.dumps(..., sort_keys=
True)` makes the hash independent of dict key order, and hashing only
in-memory fields (never a file mtime, never a path) makes it independent of
where or when the file was read. Two runs on two machines produce the same
digest for the same skill content, which is the whole point of a lock file:
`verify_lock` can tell "the rules changed" from "nothing changed" with no
false positives from clock skew or filesystem noise.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

REQUIRED_SECTIONS = ("## When to use", "## The pattern", "## Pitfalls")

# skills/ lives at the repo root; this module lives at
# src/beatroot/agent/skills_registry.py, three levels below it.
DEFAULT_SKILLS_DIR = Path(__file__).parents[3] / "skills"


class SkillFormatError(ValueError):
    """A skill file that does not meet the documented shape.

    Raised instead of tolerated so a malformed skill fails at load time,
    not as a silent half-skill (missing frontmatter field, missing
    section) that only shows up later as a confusing downstream bug.
    """


class Skill(BaseModel):
    id: str
    name: str
    tier: str
    llm_permitted: bool
    triggers_on: list[str] = []
    priority: int = 100
    body: str

    def digest(self) -> str:
        """Stable content hash: tier, llm_permitted, priority, body only.

        Deliberately excludes `id`/`name`/`triggers_on` (addressing, not
        content) and anything filesystem-derived (mtime, path). Same
        semantic skill -> same digest, on any machine, on any run.
        """
        payload = json.dumps(
            {
                "tier": self.tier,
                "llm_permitted": self.llm_permitted,
                "priority": self.priority,
                "body": self.body,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _parse(text: str) -> tuple[dict[str, Any], str]:
    """Split a skill file into (frontmatter dict, body). Fails loudly.

    A skill file that lacks YAML frontmatter, or lacks any of the three
    required sections, is not a valid skill — never a valid skill with
    gaps. Both cases raise `SkillFormatError` rather than silently
    producing a `Skill` with an empty pattern or no pitfalls.
    """
    if not text.startswith("---"):
        raise SkillFormatError("skill file must begin with YAML frontmatter delimited by '---'")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillFormatError("skill file frontmatter is not closed with a second '---'")
    _, front, body = parts
    meta = yaml.safe_load(front)
    if not isinstance(meta, dict):
        raise SkillFormatError("skill frontmatter must be a YAML mapping")

    body = body.strip()
    missing = [h for h in REQUIRED_SECTIONS if h not in body]
    if missing:
        raise SkillFormatError(f"skill body is missing required section(s): {', '.join(missing)}")
    return meta, body


def load_skills(directory: Path | str | None = None) -> dict[str, Skill]:
    """Load every `*.skill.md` file in `directory` into `Skill` models.

    `directory` defaults to the repo's `skills/` directory — never
    hardcoded elsewhere, always overridable (a test fixture, a second
    skills pack). Iteration is over `sorted(...glob(...))`, so load order
    — and therefore dict insertion order — is deterministic regardless of
    the filesystem's own directory-listing order.

    A second file declaring an `id` already seen is a naming collision, not
    a silent overwrite: every other malformed-input path in this module
    (`_parse`) fails loudly, and `skills[skill.id] = skill` clobbering the
    first file's `Skill` with no trace it ever existed would be the one
    exception to that. Raising `SkillFormatError` names both files so the
    fix is obvious.
    """
    directory = Path(directory) if directory is not None else DEFAULT_SKILLS_DIR
    skills: dict[str, Skill] = {}
    sources: dict[str, Path] = {}
    for path in sorted(directory.glob("*.skill.md")):
        meta, body = _parse(path.read_text())
        skill = Skill(**meta, body=body)
        if skill.id in skills:
            raise SkillFormatError(
                f"duplicate skill id {skill.id!r}: declared in both {sources[skill.id]} and {path}"
            )
        skills[skill.id] = skill
        sources[skill.id] = path
    return skills


def write_lock(skills: dict[str, Skill], path: Path | str) -> None:
    """Provenance: every audit record names the skill versions that produced
    it, so a recommendation can be replayed against the exact rules that
    generated it. Spec §7.

    Skills are written in sorted id order so the lock file's own diff is
    stable and reviewable, independent of `skills` dict insertion order.
    """
    payload = {
        "version": 1,
        "skills": {sid: s.digest() for sid, s in sorted(skills.items())},
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def verify_lock(skills: dict[str, Skill], path: Path | str) -> list[str]:
    """Return the sorted list of skill ids whose digest no longer matches
    the lock file. Empty list means clean — every skill's rules are exactly
    what was locked, byte for byte.

    A skill present in `skills` but absent from the lock (never locked, or
    dropped from it) counts as drifted too: `locked.get(sid)` is `None`,
    which never equals a real digest.
    """
    locked = json.loads(Path(path).read_text())["skills"]
    return sorted(sid for sid, s in skills.items() if locked.get(sid) != s.digest())

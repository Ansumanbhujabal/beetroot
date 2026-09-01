"""Gates a linter cannot express but this project needs.

Note the raw strings in the regexes below — the patterns contain backslashes.
"""

import pathlib
import re

SRC = pathlib.Path(__file__).parents[1] / "src" / "beatroot"
ROOT = SRC.parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", ".sdd", "htmlcov", "node_modules"}
# SDD process transcripts (implementation plans/specs), not shipped code —
# they narrate this very test's regex and shell verification commands
# in prose, which self-matches the secret patterns below with no secret
# actually present. Excluded from the secrets scan only.
DOC_SKIP_PARTS = {"docs"}


def _sources() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_no_bare_except_swallows_a_safety_failure() -> None:
    offenders = [
        f"{p.relative_to(SRC)}:{i}"
        for p in _sources()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.match(r"\s*except\s*:", line)
    ]
    assert not offenders, f"bare except in safety-critical code: {offenders}"


def test_no_unfinished_markers_ship() -> None:
    offenders = [
        f"{p.relative_to(SRC)}:{i}"
        for p in _sources()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line)
    ]
    assert not offenders, f"unfinished markers: {offenders}"


def _git_ignored(path: pathlib.Path) -> bool:
    """Whether git ignores this path. A local .env holding real credentials is
    not a shipped file — but only while it stays ignored, which the test below
    pins so the exemption cannot quietly become a hole."""
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:  # pragma: no cover - git is always present in this repo
        return False
    return (
        subprocess.run(  # noqa: S603 - fixed argv, resolved binary, no shell
            [git, "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True
        ).returncode
        == 0
    )


def test_credential_files_are_git_ignored() -> None:
    """The secrets scan exempts git-ignored files, so the exemption itself needs
    a guard: if .env stopped being ignored, the scan would go quiet at exactly
    the moment it mattered."""
    for name in (".env", "beatroot.db", "beatroot.checkpoints.db"):
        assert _git_ignored(ROOT / name), f"{name} must stay git-ignored"


def test_no_secrets_in_any_shipped_file() -> None:
    patterns = [
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        # [ \t]* (not \s*) after the separator: keeps the match on one line so
        # an empty placeholder like "AZURE_API_KEY=" followed by the next
        # env var on the next line isn't swallowed into a false positive.
        r"AZURE_[A-Z_]*KEY\s*[:=][ \t]*\S+",
    ]
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or (SKIP_PARTS | DOC_SKIP_PARTS) & set(path.parts):
            continue
        if path.name == "test_quality_gates.py" or _git_ignored(path):
            # Excluded: this file's own source contains the patterns above,
            # which would otherwise match themselves.
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, PermissionError):
            continue
        for pattern in patterns:
            if re.search(pattern, text):
                offenders.append(f"{path.relative_to(ROOT)} ~ {pattern}")
    assert not offenders, f"possible secrets in shipped files: {offenders}"


def test_every_source_module_has_a_docstring() -> None:
    offenders = [
        str(p.relative_to(SRC))
        for p in _sources()
        if p.name != "__init__.py" and not p.read_text().lstrip().startswith(("'", '"'))
    ]
    assert not offenders, f"modules missing a docstring: {offenders}"


def test_no_model_provider_http_client_outside_litellm() -> None:
    """One LLM abstraction. A hand-rolled provider client is a design defect."""
    offenders = [
        str(p.relative_to(SRC))
        for p in _sources()
        if p.name != "llm.py" and re.search(r"openai\.com|/chat/completions", p.read_text())
    ]
    assert not offenders, f"hand-rolled provider HTTP: {offenders}"


def test_no_hand_written_model_provider_http_outside_llm() -> None:
    """The project has one LLM abstraction (LiteLLM). No module besides
    reasoning/llm.py may talk HTTP directly to a model provider — no raw
    `httpx`/`requests` calls to a provider host, and no provider SDK import.
    """
    provider_host_pattern = re.compile(
        r"api\.openai\.com|api\.anthropic\.com|generativelanguage\.googleapis\.com"
        r"|api\.cohere\.ai|api\.mistral\.ai|openai\.com/v1|anthropic\.com/v1"
    )
    provider_sdk_pattern = re.compile(
        r"^\s*(import|from)\s+(openai|anthropic|google\.generativeai|cohere|mistralai)\b",
        re.MULTILINE,
    )
    offenders = [
        str(p.relative_to(SRC))
        for p in _sources()
        if p.name != "llm.py"
        and (
            provider_host_pattern.search(p.read_text())
            or provider_sdk_pattern.search(p.read_text())
        )
    ]
    assert not offenders, f"hand-written model-provider HTTP outside llm.py: {offenders}"


# Note: the "os.getenv/os.environ only in settings.py" gate already exists as
# tests/test_settings.py::test_settings_is_the_only_module_reading_env — not
# duplicated here per the task brief.

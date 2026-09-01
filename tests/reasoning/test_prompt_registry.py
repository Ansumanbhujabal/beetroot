"""Prompt resolution: Langfuse when configured, the local file always.

The property under test throughout is that remote prompt management is an
ENHANCEMENT over reading a file off disk, and that nothing about it is
allowed to turn its absence — or its misbehaviour — into a broken
application. A prompt that cannot be fetched, or that comes back in a shape
this code will not run, must degrade to the local file with a log line, not
raise inside a graph node halfway through a request that has already spent
tokens.
"""

import pytest

from beatroot.reasoning.prompts import (
    PRODUCTION_LABEL,
    Prompt,
    _fetch_remote,
    _load_local,
    load_prompt,
    placeholders,
)

ALL_PROMPTS = ["rerank", "explain", "compile_constraints", "rewrite_query"]


@pytest.fixture(autouse=True)
def _clear_prompt_cache():
    load_prompt.cache_clear()
    yield
    load_prompt.cache_clear()


class _FakeFetched:
    def __init__(self, prompt: str, version: int = 7) -> None:
        self.prompt = prompt
        self.version = version


def test_escaped_braces_are_not_placeholders():
    """`{{` is a literal brace to `str.format`, not a variable.

    This is the whole reason rendering stays local rather than being handed
    to Langfuse's mustache compiler: `prompts/compile_constraints.md` ends
    with a JSON example written as `{{"in_scope": true, ...}}`, which
    `str.format` renders as `{"in_scope": ...}` and mustache would read as
    an interpolation. A regex-based placeholder scan got this wrong; using
    `string.Formatter` means this function is exactly as strict as the
    engine that will actually run the template.
    """
    template = 'Reply with JSON only:\n{{"in_scope": true, "value": "x"}}\nText: {free_text}'
    assert placeholders(template) == {"free_text"}


@pytest.mark.parametrize("name", ALL_PROMPTS)
def test_every_shipped_prompt_declares_all_its_placeholders(name):
    """The guard `_fetch_remote` applies to remote templates must also hold
    for the ones in the repo — otherwise the check would reject the very
    files it falls back to."""
    local = _load_local(name)
    undeclared = placeholders(local.template) - set(local.inputs)
    assert not undeclared, f"{name} uses undeclared input(s): {sorted(undeclared)}"


def test_a_directory_argument_never_consults_langfuse(tmp_path):
    """A test pointed at a fixture directory must not be silently answered
    by a real remote prompt — otherwise a passing test would be evidence
    about someone's Langfuse project, not about this code."""
    (tmp_path / "explain.md").write_text(
        "---\nid: explain\nversion: 1\nstage: explain\ninputs: [name]\n---\nHello {name}"
    )
    resolved = load_prompt("explain", tmp_path)
    assert resolved.source == "local"
    assert resolved.remote_version is None
    assert resolved.render(name="rice") == "Hello rice"


def test_resolution_falls_back_to_local_when_langfuse_is_absent(monkeypatch):
    """No client (no credentials, or the optional SDK not installed) is the
    ordinary keyless path, not an error."""
    import beatroot.reasoning.prompts as prompts

    monkeypatch.setattr(prompts, "langfuse_client", lambda: None)
    local = _load_local("explain")
    assert _fetch_remote("explain", local) is None


def test_a_fetch_failure_falls_back_rather_than_raising(monkeypatch):
    """A prompt that does not exist in Langfuse yet is the normal state of a
    fresh project — `beatroot prompts push` has not run. It must read as a
    fallback, never as a fault."""
    import beatroot.reasoning.prompts as prompts

    class _Failing:
        def get_prompt(self, name, label=None):
            raise RuntimeError("prompt not found")

    monkeypatch.setattr(prompts, "langfuse_client", lambda: _Failing())
    assert _fetch_remote("explain", _load_local("explain")) is None


def test_a_remote_prompt_referencing_an_undeclared_input_is_refused(monkeypatch):
    """The guard that keeps a web-UI edit from becoming a mid-request crash.

    `str.format` fails at RENDER time, which here means inside a graph node
    on a path that has already spent tokens. Checking placeholders at LOAD
    time converts that into a startup log line and a fallback.
    """
    import beatroot.reasoning.prompts as prompts

    local = _load_local("explain")
    tampered = local.template + "\nAlso hit {calorie_target}."

    class _Tampered:
        def get_prompt(self, name, label=None):
            return _FakeFetched(tampered)

    monkeypatch.setattr(prompts, "langfuse_client", lambda: _Tampered())
    assert _fetch_remote("explain", local) is None, "an undeclared input must be refused"


def test_an_empty_remote_template_is_refused(monkeypatch):
    import beatroot.reasoning.prompts as prompts

    class _Empty:
        def get_prompt(self, name, label=None):
            return _FakeFetched("   ")

    monkeypatch.setattr(prompts, "langfuse_client", lambda: _Empty())
    assert _fetch_remote("explain", _load_local("explain")) is None


def test_a_valid_remote_prompt_is_used_and_keeps_the_local_contract(monkeypatch):
    """Remote supplies the TEXT; the local file supplies the contract.

    `inputs` and `stage` are what the calling code was written against, so
    they are never taken from the remote payload — only the template body
    is.
    """
    import beatroot.reasoning.prompts as prompts

    local = _load_local("explain")

    class _Good:
        def get_prompt(self, name, label=None):
            assert label == PRODUCTION_LABEL, "must resolve at the production label"
            return _FakeFetched("Explain {name}. Facts: {facts}. Satisfies: {satisfied}", version=7)

    monkeypatch.setattr(prompts, "langfuse_client", lambda: _Good())
    resolved = _fetch_remote("explain", local)
    assert resolved is not None
    assert resolved.source == "langfuse"
    assert resolved.remote_version == 7
    assert resolved.inputs == local.inputs, "the contract stays local"
    assert resolved.stage == local.stage


def test_ref_and_trace_metadata_name_the_version_that_actually_ran():
    """Provenance has to distinguish "which contract does the code expect"
    from "which text actually ran" — they are different questions and a
    single version number cannot answer both."""
    local = Prompt(
        id="explain", version=1, stage="explain", inputs=["name"], template="Hi {name}"
    )
    assert local.ref == "explain@local:v1"
    assert local.trace_metadata()["prompt_version"] == 1

    remote = local.model_copy(update={"source": "langfuse", "remote_version": 9})
    assert remote.ref == "explain@langfuse:v9"
    assert remote.trace_metadata()["prompt_source"] == "langfuse"
    assert remote.trace_metadata()["prompt_version"] == 9


def test_render_still_rejects_missing_inputs():
    with pytest.raises(KeyError):
        _load_local("explain").render(name="x")

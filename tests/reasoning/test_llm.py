from typing import ClassVar

from beatroot.reasoning.llm import LLMClient, _parse_llm_json, get_llm_client
from beatroot.settings import get_settings


def test_completion_carries_cost_from_litellm(monkeypatch):
    """Cost comes from LiteLLM's own accounting — never a hand-maintained
    price table that silently rots when a deployment changes."""
    client = LLMClient(model="azure/gpt-4o-mini")

    class _Msg:  # minimal LiteLLM-shaped response
        content = '{"choice_index": 0, "self_assessment": 0.8}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices: ClassVar = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _Resp())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.00042)

    out = client.complete("hi", stage="rerank")
    assert out.cost.usd == 0.00042
    assert out.cost.prompt_tokens == 10
    assert out.self_assessment == 0.8
    assert out.parsed["choice_index"] == 0


def test_fallbacks_are_passed_to_litellm(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "beatroot.reasoning.llm.litellm.completion", lambda **k: seen.update(k) or _stub()
    )
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)
    LLMClient(model="azure/x", fallbacks=["ollama/llama3.2"]).complete("hi")
    assert seen["fallbacks"] == ["ollama/llama3.2"]
    assert seen["num_retries"] >= 1


def _stub():
    class _M:
        content = "{}"

    class _C:
        message = _M()

    class _R:
        choices: ClassVar = [_C()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()

    return _R()


def test_malformed_json_does_not_raise(monkeypatch):
    """A model returning prose where JSON was demanded is a Tuesday, not a crash."""
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _prose())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)
    out = LLMClient(model="azure/x").complete("hi", schema=dict)
    assert out.parsed is None
    assert out.text


def _prose():
    class _M:
        content = "I'm afraid I can't do that."

    class _C:
        message = _M()

    class _R:
        choices: ClassVar = [_C()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()

    return _R()


def test_non_object_json_does_not_raise(monkeypatch):
    """A bare JSON array or scalar is not the object shape a schema demands.
    It must be treated as unparsed, not smuggled into `parsed: dict | None`
    (which would blow up pydantic validation) or crash on `.get`."""
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _bare_array())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)
    out = LLMClient(model="azure/x").complete("hi", schema=dict)
    assert out.parsed is None
    assert out.text


def _bare_array():
    class _M:
        content = "[1, 2, 3]"

    class _C:
        message = _M()

    class _R:
        choices: ClassVar = [_C()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()

    return _R()


# ---- BUG 1: real models fence their JSON --------------------------------
#
# Recorded (not live-network) response strings. gpt-4o, against the live
# `rerank` prompt, actually returns the first one verbatim — see the bug
# report for the paired live before/after.

_FENCED_WITH_TAG = (
    '```json\n{"choice_index": 0, "rationale": "Jeera rice better matches '
    'the request for something light and fragrant.", "self_assessment": 1.0}\n```'
)
_FENCED_BARE = '```\n{"choice_index": 1, "self_assessment": 0.7}\n```'
_UNFENCED = '{"choice_index": 2, "self_assessment": 0.4}'
_FENCED_WITH_PROSE = (
    "Sure, here's my pick:\n\n"
    '```json\n{"choice_index": 0, "self_assessment": 0.9}\n```\n\n'
    "Let me know if you'd like another option."
)
_UNPARSEABLE_PROSE = "I'm afraid I can't help rank these — not enough information."


def test_parse_llm_json_unwraps_fence_with_language_tag():
    parsed = _parse_llm_json(_FENCED_WITH_TAG)
    assert parsed == {
        "choice_index": 0,
        "rationale": "Jeera rice better matches the request for something light and fragrant.",
        "self_assessment": 1.0,
    }


def test_parse_llm_json_unwraps_bare_fence():
    assert _parse_llm_json(_FENCED_BARE) == {"choice_index": 1, "self_assessment": 0.7}


def test_parse_llm_json_handles_plain_unfenced_json():
    assert _parse_llm_json(_UNFENCED) == {"choice_index": 2, "self_assessment": 0.4}


def test_parse_llm_json_unwraps_fence_surrounded_by_prose():
    assert _parse_llm_json(_FENCED_WITH_PROSE) == {"choice_index": 0, "self_assessment": 0.9}


def test_parse_llm_json_returns_none_for_genuinely_unparseable_prose():
    assert _parse_llm_json(_UNPARSEABLE_PROSE) is None


def test_complete_unwraps_fenced_json_and_captures_self_assessment(monkeypatch):
    """End-to-end through `.complete()`: a fenced reply (the actual live
    gpt-4o shape for the `rerank` prompt) must populate `parsed` and a
    real, non-neutral `self_assessment` — not fall back to `parsed=None`."""

    class _Msg:
        content = _FENCED_WITH_TAG

    class _Choice:
        message = _Msg()

    class _Resp:
        choices: ClassVar = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _Resp())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.001)

    out = LLMClient(model="azure/gpt-4o").complete("rank these", stage="rerank")
    assert out.parsed is not None
    assert out.parsed["choice_index"] == 0
    assert out.self_assessment == 1.0


def test_complete_unparseable_prose_still_returns_parsed_none_without_raising(monkeypatch, caplog):
    class _Msg:
        content = _UNPARSEABLE_PROSE

    class _Choice:
        message = _Msg()

    class _Resp:
        choices: ClassVar = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _Resp())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)

    out = LLMClient(model="azure/gpt-4o").complete("rank these", schema=dict, stage="rerank")
    assert out.parsed is None
    assert out.self_assessment is None


# ---- BUG 2: the non-JSON warning must only fire for schema-bound stages --


def test_no_warning_for_prose_stage_that_never_requested_a_schema(monkeypatch, caplog):
    """`explain` returns prose by design and never passes `schema=`. It
    must not log the "non-JSON" warning that a schema-bound stage would."""

    class _Msg:
        content = "This dish is a great fit because it is light and fragrant."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices: ClassVar = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _Resp())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)

    with caplog.at_level("WARNING", logger="beatroot.llm"):
        out = LLMClient(model="azure/gpt-4o").complete("explain this", stage="explain")
    assert out.parsed is None
    assert "non-JSON" not in caplog.text


def test_warning_still_fires_for_schema_bound_stage_on_unparseable_reply(monkeypatch, caplog):
    class _Msg:
        content = _UNPARSEABLE_PROSE

    class _Choice:
        message = _Msg()

    class _Resp:
        choices: ClassVar = [_Choice()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _Resp())
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", lambda **k: 0.0)

    with caplog.at_level("WARNING", logger="beatroot.llm"):
        LLMClient(model="azure/gpt-4o").complete("rank these", schema=dict, stage="rerank")
    assert "stage=rerank" in caplog.text
    assert "non-JSON" in caplog.text


def test_completion_cost_failure_falls_back_to_zero(monkeypatch, caplog):
    """litellm.completion_cost can raise for an unrecognised model. Cost
    accounting must not take down a request — but the reason must be logged,
    not silently swallowed."""
    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion", lambda **k: _stub())

    def _raise(**k):
        raise ValueError("unrecognised model")

    monkeypatch.setattr("beatroot.reasoning.llm.litellm.completion_cost", _raise)
    with caplog.at_level("WARNING", logger="beatroot.llm"):
        out = LLMClient(model="azure/unknown-model").complete("hi", stage="rerank")
    assert out.cost.usd == 0.0
    assert "rerank" in caplog.text


def test_offline_mode_needs_no_network():
    client = LLMClient.offline()
    a = client.complete("same prompt")
    b = client.complete("same prompt")
    assert a.text == b.text and a.cost.usd == 0.0
    v = client.embed(["paneer"])[0]
    assert abs(sum(x * x for x in v) ** 0.5 - 1.0) < 1e-6


def test_offline_compile_stage_extracts_known_tags_mentioned_in_free_text():
    """The offline stub is not a fixed no-op for the `compile` stage: it
    extracts every catalog tag literally mentioned in the free text (read
    off the prompt's own rendered "Known tags:"/"User text:" sections) and
    returns it as `exclude_tags` — deterministic, no network, no catalog
    access — so `--preferences` visibly does something offline and
    `compile_node`'s parse/filter logic is exercised by every
    credential-free run instead of skipped by a fixed stub payload with no
    `exclude_tags` key at all."""
    from beatroot.reasoning.prompts import load_prompt

    prompt = load_prompt("compile_constraints").render(
        free_text="no dairy please, and nothing with peanuts",
        known_tags="dairy, peanut, gluten, vegan",
        known_cuisines="",
        known_kinds="",
        kind_shapes="",
    )
    out = LLMClient.offline().complete(prompt, stage="compile")
    assert set(out.parsed["exclude_tags"]) == {"dairy", "peanut"}
    assert out.parsed["prefer_tags"] == []


def test_offline_compile_stage_handles_underscore_tags_spelled_with_a_space():
    from beatroot.reasoning.prompts import load_prompt

    prompt = load_prompt("compile_constraints").render(
        free_text="please avoid tree nuts",
        known_tags="tree_nut, dairy",
        known_cuisines="",
        known_kinds="",
        kind_shapes="",
    )
    out = LLMClient.offline().complete(prompt, stage="compile")
    assert out.parsed["exclude_tags"] == ["tree_nut"]


def test_offline_compile_stage_ignores_tags_not_mentioned():
    from beatroot.reasoning.prompts import load_prompt

    prompt = load_prompt("compile_constraints").render(
        free_text="surprise me, I trust the chef",
        known_tags="dairy, peanut, gluten, vegan",
        known_cuisines="",
        known_kinds="",
        kind_shapes="",
    )
    out = LLMClient.offline().complete(prompt, stage="compile")
    assert out.parsed["exclude_tags"] == []


def test_offline_non_compile_stage_never_gets_an_exclude_tags_key():
    out = LLMClient.offline().complete("hi", stage="rerank")
    assert "exclude_tags" not in out.parsed


def test_get_llm_client_reads_offline_from_settings_not_env():
    """The BEATROOT_OFFLINE toggle is a settings field — get_llm_client must
    consult get_settings(), never os.getenv, to pick offline vs live."""
    get_settings.cache_clear()
    import os

    os.environ["BEATROOT_OFFLINE"] = "1"
    try:
        get_settings.cache_clear()
        client = get_llm_client()
        assert client._offline is True
    finally:
        del os.environ["BEATROOT_OFFLINE"]
        get_settings.cache_clear()


def test_get_llm_client_is_live_when_credentials_are_present(monkeypatch):
    """`Settings._default_to_offline_without_credentials` (Task 20 fix
    round) only ever flips offline ON for a missing-credentials process —
    a process that actually has the configured provider's credentials
    must still get a live client, not be forced offline regardless."""
    monkeypatch.delenv("BEATROOT_OFFLINE", raising=False)
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.openai.azure.com")
    get_settings.cache_clear()
    try:
        client = get_llm_client()
        assert client._offline is False
    finally:
        get_settings.cache_clear()


def test_get_llm_client_is_offline_by_default_with_no_credentials(monkeypatch):
    """The keyless-boot guarantee at the `get_llm_client()` seam directly:
    no `BEATROOT_OFFLINE` and no provider credentials must still produce
    an offline client — this used to construct a live one that only threw
    on its first real call, which for `build_container()` was before
    `/health` was ever reachable."""
    for var in ("BEATROOT_OFFLINE", "AZURE_API_KEY", "AZURE_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    try:
        client = get_llm_client()
        assert client._offline is True
    finally:
        get_settings.cache_clear()

"""LiteLLM must not leak a socket per model call.

Reported from a running server: dozens of `asyncio: Unclosed client session`
and `Unclosed connector` ERRORs, all carrying the SAME request_id — one
request leaking a session per model call, surfacing only when the GC finalised
them much later. Measured before the fix: 3 real calls -> 3 live
`aiohttp.ClientSession` objects, none closed. A /recommend makes ~3 model
calls, so the server leaked roughly a socket per call until it ran out of file
descriptors. The log lines were the symptom; descriptor exhaustion was the bug.

WHAT THIS FILE CAN AND CANNOT TEST, stated because the first version of it got
this wrong in both directions:

  - A before/after COUNT of live sessions was order-dependent: any unrelated
    session created or collected inside the window moved the total, so it
    failed in the full suite for reasons unconnected to leaking.
  - Worse, it was VACUOUS in isolation. With no credentials `settings.offline`
    is True, so `complete()` never opens a socket at all and the assertion
    passed without exercising anything — the same failure A6 had.

So the invariant tested here is the one that is actually deterministic: the
transport is disabled BY THE TIME any client can be constructed. The leak
itself was verified directly (3 -> 0 sessions on a real HTTP path, and 0
`Unclosed` errors on a live server serving real recommendations); that check
needs a network round trip and does not belong in the unit suite.
"""

import litellm


def test_litellm_exposes_the_flag_this_fix_depends_on() -> None:
    """If LiteLLM renames or removes it, fail loudly here rather than silently
    resume leaking a socket per call."""
    assert hasattr(litellm, "disable_aiohttp_transport"), (
        "litellm no longer exposes disable_aiohttp_transport — re-verify the "
        "socket-leak fix against the new API before assuming it still holds"
    )


def test_importing_the_llm_module_disables_the_leaking_transport() -> None:
    import beatroot.reasoning.llm  # noqa: F401

    assert litellm.disable_aiohttp_transport is True


def test_transport_is_disabled_before_any_client_can_be_built() -> None:
    """The ordering is the whole fix.

    `beatroot.container` imports `reasoning.llm` lazily, inside the function,
    so the flag is still False at container-module import time and only flips
    when a container is actually built. That is fine — but only because the
    flip happens BEFORE any completion can be issued. This pins that ordering,
    which is what a future refactor could quietly break while every other test
    stays green.
    """
    from beatroot.container import build_container

    container = build_container(async_explanation=False)
    try:
        assert litellm.disable_aiohttp_transport is True
    finally:
        container.close()

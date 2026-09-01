import json
import logging

from beatroot.obs.logging import RING_BUFFER_SIZE, bind_request, configure_logging, recent_logs


def test_logs_are_json_with_correlation_id(capsys):
    configure_logging()
    with bind_request(request_id="req-1", profile_id="p-9"):
        logging.getLogger("beatroot").info("planning", extra={"stage": "feasibility"})
    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line["request_id"] == "req-1"
    assert line["profile_id"] == "p-9"
    assert line["stage"] == "feasibility"
    assert line["message"] == "planning"


def test_context_does_not_leak_between_requests(capsys):
    configure_logging()
    with bind_request(request_id="req-1", profile_id="p-1"):
        pass
    logging.getLogger("beatroot").info("outside")
    line = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert line.get("request_id") is None
    assert line.get("profile_id") is None


def test_secrets_are_never_logged(capsys):
    configure_logging()
    logging.getLogger("beatroot").info("call", extra={"api_key": "sk-live-abcdef"})
    out = capsys.readouterr().err
    assert "sk-live-abcdef" not in out
    assert "[redacted]" in out


def test_secret_like_keys_are_all_redacted(capsys):
    """Every field named like a credential is redacted, not just api_key."""
    configure_logging()
    logging.getLogger("beatroot").info(
        "call",
        extra={
            "authorization": "Bearer sk-abc123",
            "secret_key": "shh-dont-tell",
            "password": "hunter2",
            "token": "eyJhbGciOi",
            "stage": "retrieve",  # not a secret; must survive untouched
        },
    )
    out = capsys.readouterr().err
    for secret in ("sk-abc123", "shh-dont-tell", "hunter2", "eyJhbGciOi"):
        assert secret not in out
    line = json.loads(out.strip().splitlines()[-1])
    assert line["authorization"] == "[redacted]"
    assert line["secret_key"] == "[redacted]"
    assert line["password"] == "[redacted]"
    assert line["token"] == "[redacted]"
    assert line["stage"] == "retrieve"


def test_nested_binds_restore_the_outer_context():
    """bind_request is reentrant: an inner block's exit restores whatever
    the outer block had bound, rather than clearing it to None."""
    import beatroot.obs.logging as obs_logging

    with bind_request(request_id="outer", profile_id="p-outer"):
        with bind_request(request_id="inner", profile_id="p-inner"):
            assert obs_logging._request_id.get() == "inner"
        assert obs_logging._request_id.get() == "outer"
        assert obs_logging._profile_id.get() == "p-outer"


# ---------------------------------------------------------------------------
# Adversarial redaction — every leak the review found against the first
# version of this module (exact five-word match, top-level-only). The
# property under test is that the VALUE is absent from output, not merely
# that some key renders as "[redacted]" — a formatter could satisfy the
# weaker property while still leaking the secret under a different key.
# ---------------------------------------------------------------------------

_ADVERSARIAL_KEYS = [
    "Api-Key",
    "x-api-key",
    "openai_api_key",
    "access_token",
    "client_secret",
]


def test_adversarial_key_variants_are_all_redacted(capsys):
    configure_logging()
    secrets = {key: f"SECRET-VALUE-FOR-{key}" for key in _ADVERSARIAL_KEYS}
    logging.getLogger("beatroot").info("call", extra=secrets)
    out = capsys.readouterr().err
    for key, secret in secrets.items():
        assert secret not in out, f"{key!r} leaked its value into the log line"
    assert "[redacted]" in out


def test_nested_dict_secret_is_redacted(capsys):
    """The realistic shape: logging a settings object or a request body,
    where the credential is one level down, not at the top of `extra`."""
    configure_logging()
    logging.getLogger("beatroot").info(
        "provider config",
        extra={"config": {"provider": "azure", "api_key": "sk-live-nested-secret"}},
    )
    out = capsys.readouterr().err
    assert "sk-live-nested-secret" not in out
    line = json.loads(out.strip().splitlines()[-1])
    assert line["config"]["api_key"] == "[redacted]"
    assert line["config"]["provider"] == "azure"  # non-secret siblings survive


def test_list_of_dicts_secret_is_redacted(capsys):
    """extra={"headers": [{"Authorization": "..."}]} — the realistic shape
    of logging a batch of provider requests/responses."""
    configure_logging()
    logging.getLogger("beatroot").info(
        "batch",
        extra={
            "requests": [
                {"url": "https://api.example.com", "Authorization": "Bearer sk-batch-secret"},
                {"url": "https://api.example.com/2", "Authorization": "Bearer sk-batch-secret-2"},
            ]
        },
    )
    out = capsys.readouterr().err
    assert "sk-batch-secret" not in out
    assert "sk-batch-secret-2" not in out
    line = json.loads(out.strip().splitlines()[-1])
    assert line["requests"][0]["Authorization"] == "[redacted]"
    assert line["requests"][1]["Authorization"] == "[redacted]"
    assert line["requests"][0]["url"] == "https://api.example.com"


def test_traceback_with_fake_bearer_token_is_scrubbed(capsys):
    """A litellm/httpx-style exception echoing an Authorization header must
    not leak that header's value through `log.exception`."""
    configure_logging()
    log = logging.getLogger("beatroot")
    try:
        raise RuntimeError(
            "provider call failed: Authorization: Bearer sk-live-traceback-secret-12345"
        )
    except RuntimeError:
        log.exception("upstream call failed")
    out = capsys.readouterr().err
    assert "sk-live-traceback-secret-12345" not in out
    line = json.loads(out.strip().splitlines()[-1])
    assert "[redacted]" in line["exception"]


def test_traceback_with_raw_api_key_string_is_scrubbed(capsys):
    configure_logging()
    log = logging.getLogger("beatroot")
    try:
        raise ValueError("rejected key sk-abcd1234-efgh5678-ijkl9012")
    except ValueError:
        log.exception("bad key")
    out = capsys.readouterr().err
    assert "sk-abcd1234-efgh5678-ijkl9012" not in out


def test_deeply_nested_structure_does_not_hang_and_still_redacts(capsys):
    """A depth cap protects against a cyclic or pathologically deep
    structure; a secret within the cap is still caught."""
    configure_logging()
    nested: dict = {"api_key": "deep-secret"}
    for _ in range(20):
        nested = {"child": nested}
    logging.getLogger("beatroot").info("deep", extra={"payload": nested})
    out = capsys.readouterr().err
    assert "deep-secret" not in out


def test_message_interpolated_secret_is_a_documented_gap(capsys):
    """Known limitation, not a silent hole: a secret baked directly into
    the message string (rather than passed via `extra=`) is NOT redacted.
    This test documents that boundary rather than asserting false safety."""
    configure_logging()
    logging.getLogger("beatroot").info("using key sk-message-interpolated-secret")
    out = capsys.readouterr().err
    assert "sk-message-interpolated-secret" in out


# ---------------------------------------------------------------------------
# Ring buffer (EVALS PAGE task) — recent_logs() reads a bounded, in-memory
# window of what this process has emitted, independent of stderr.
# ---------------------------------------------------------------------------


def test_recent_logs_returns_emitted_records():
    configure_logging()
    logging.getLogger("beatroot").info("ring-buffer-marker", extra={"stage": "test"})
    lines = recent_logs(limit=RING_BUFFER_SIZE)
    assert any(
        line["message"] == "ring-buffer-marker" and line["stage"] == "test" for line in lines
    )


def test_recent_logs_redacts_the_same_as_stderr():
    configure_logging()
    logging.getLogger("beatroot").info(
        "ring-secret-check", extra={"api_key": "sk-ring-buffer-secret"}
    )
    lines = recent_logs(limit=RING_BUFFER_SIZE)
    marker = next(line for line in lines if line["message"] == "ring-secret-check")
    assert marker["api_key"] == "[redacted]"


def test_recent_logs_is_bounded_and_honest_about_its_size():
    configure_logging()
    for i in range(RING_BUFFER_SIZE + 20):
        logging.getLogger("beatroot").info("filler", extra={"i": i})
    lines = recent_logs(limit=RING_BUFFER_SIZE + 100)
    assert len(lines) <= RING_BUFFER_SIZE


def test_recent_logs_respects_a_smaller_limit():
    configure_logging()
    for i in range(10):
        logging.getLogger("beatroot").info("limited", extra={"i": i})
    lines = recent_logs(limit=3)
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# BUG 3 — LiteLLM's own logging-worker teardown noise must be silenced at
# ERROR without blanket-silencing the `asyncio` logger.
# ---------------------------------------------------------------------------


def test_litellm_logging_worker_teardown_noise_is_suppressed(capsys):
    configure_logging()
    logging.getLogger("asyncio").error(
        "Task was destroyed but it is pending!\n"
        "task: <Task pending name='Task-266' "
        "coro=<LoggingWorker._worker_loop() running at "
        ".../litellm/litellm_core_utils/logging_worker.py:113>>"
    )
    assert capsys.readouterr().err == ""


def test_other_asyncio_errors_still_reach_the_log(capsys):
    """The suppression must be narrow — a real asyncio ERROR must still
    surface, never a blanket `asyncio` logger silence."""
    configure_logging()
    logging.getLogger("asyncio").error("Exception in callback something_real")
    err = capsys.readouterr().err
    assert "Exception in callback something_real" in err


def test_configure_logging_does_not_duplicate_the_asyncio_filter(capsys):
    """`configure_logging()` is called repeatedly (once per CLI command,
    once per Container build) — the filter must stay singular, not stack
    up and (harmlessly, but wastefully) re-check the message N times."""
    configure_logging()
    configure_logging()
    configure_logging()
    from beatroot.obs.logging import _SuppressLiteLLMLoggingWorkerTeardownNoise

    filters = logging.getLogger("asyncio").filters
    assert sum(isinstance(f, _SuppressLiteLLMLoggingWorkerTeardownNoise) for f in filters) == 1

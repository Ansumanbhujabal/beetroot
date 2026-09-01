"""Observability: LiteLLM-native tracing, structured logging, cost accounting.

Spec §13. Three small, independent pieces:

- `obs.tracing` registers LiteLLM's own success/failure callbacks (Langfuse)
  so every model call gets a span — including one served by a fallback —
  without a hand-rolled wrapper around `reasoning.llm.LLMClient`.
- `obs.logging` gives every log line a JSON shape and a correlation id that
  survives across LangGraph nodes via `contextvars`, with secret redaction
  applied unconditionally.
- `obs.cost` aggregates per-stage cost and tokens-not-spent across plans.

Nothing in this package calls `os.getenv` or touches `os.environ` directly.
`configure_observability()` reads credentials through
`beatroot.settings.get_settings().obs` instead, keeping `settings.py` the
one module in the codebase permitted to read the process environment —
`tests/test_settings.py::test_settings_is_the_only_module_reading_env`
enforces that with an AST walk (not a text search) precisely so a module
can state this rule, as this docstring just did, without tripping it.
"""

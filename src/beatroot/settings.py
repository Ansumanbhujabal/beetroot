"""Configuration has exactly one home.

`Settings` is the ONLY module in this codebase allowed to read the process
environment. Every threshold, weight, model name, retry count, and tag
vocabulary used by any other module must flow from `get_settings()` — never
a literal, never a direct `os.getenv`. `tests/test_settings.py` enforces the
`os.getenv`/`os.environ` restriction with a grep-level test; do not add an
exemption to it. Spec §14.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "config" / "beatroot.yaml"

log = logging.getLogger("beatroot.settings")


def _assemble_process_environment() -> None:
    """Load `.env` into the process environment ONCE, at import of this
    module, without overriding anything already set.

    This exists because the alternative was an import-order bug, and a
    nasty one: `_provider_credentials_present` below probes `os.environ`,
    but `.env` only ever reached `os.environ` as a SIDE EFFECT of importing
    `litellm`, which calls `load_dotenv()` itself. So

        from beatroot.settings import get_settings   -> offline = True
        import litellm; from beatroot.settings ...   -> offline = False

    with identical credentials on disk — the same command would silently
    run against the offline stub or against a real provider depending on
    which module a caller happened to import first. `beatroot prompts push`
    hit exactly this and reported "no credentials found" while holding
    valid keys.

    Doing it here, at import of the one module permitted to touch the
    environment, makes the process environment fully assembled before
    anything reads it, and makes the answer the same no matter who imports
    what first. `override=False` keeps the real environment authoritative,
    so a `docker run -e` or a compose `environment:` block still outranks
    the file — and a test's `monkeypatch.delenv` still takes effect,
    because this runs at import, long before any test body does.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv ships with pydantic-settings
        return
    for candidate in (Path.cwd() / ".env", ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_assemble_process_environment()

# Minimal provider-prefix -> required-env-var(s) map, mirroring the exact
# names LiteLLM itself reads for each provider. Not exhaustive by design: an
# unrecognised prefix falls back to the "{PREFIX}_API_KEY" convention
# LiteLLM uses for the large majority of providers it supports, and
# `ollama` needs no key at all (only a reachable server — see
# `_provider_credentials_present` below for why that's not checked here).
_KNOWN_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    # "local" is an explicit choice to use the built-in deterministic embedder,
    # not a missing credential. It lets a real chat model run without also
    # requiring an embedding deployment on the same resource.
    "local": (),
    "bedrock": ("AWS_ACCESS_KEY_ID",),
    "gemini": ("GEMINI_API_KEY",),
    "vertex_ai": ("VERTEXAI_PROJECT",),
    "mistral": ("MISTRAL_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
    "together_ai": ("TOGETHERAI_API_KEY",),
}


def _provider_credentials_present(model: str) -> bool:
    """Whether the env vars LiteLLM would need to actually CALL `model`
    look present, WITHOUT making any network call of our own — this only
    ever checks `os.environ`, matching every other read in this module.

    `ollama` always reports True: it needs no API key, only a reachable
    server, and reachability is a network question this module (settings
    load, no I/O) has no business asking. An unreachable Ollama still
    fails — just later, at the actual call site, exactly like any other
    unreachable network dependency, and no worse than before this
    function existed.
    """
    prefix = model.split("/", 1)[0] if "/" in model else model
    if prefix == "ollama":
        return True
    required = _KNOWN_PROVIDER_ENV_VARS.get(prefix, (f"{prefix.upper()}_API_KEY",))
    return all(os.getenv(name) for name in required) if required else True


class LLMConfig(BaseModel):
    model: str
    fallbacks: list[str] = Field(default_factory=list)
    embedding_model: str
    embedding_fallbacks: list[str] = Field(default_factory=list)
    temperature: float = 0.2
    max_retries: int = 3
    timeout_seconds: int = 30


class RetrievalConfig(BaseModel):
    rrf_k: int
    lexical_weight: float
    dense_weight: float
    affinity_weight: float
    candidate_limit: int
    top_k: int


class TrustWeights(BaseModel):
    catalog_coverage: float
    constraint_completeness: float
    model_self_assessment: float

    @model_validator(mode="after")
    def _sum_to_one(self) -> "TrustWeights":
        """A weighting that does not sum to 1 silently rescales every trust
        score in the system — reject it at load, not at first use."""
        total = self.catalog_coverage + self.constraint_completeness + self.model_self_assessment
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"trust weights must sum to 1, got {total}")
        return self


class TrustConfig(BaseModel):
    weights: TrustWeights
    refusal_threshold: float
    weak_signal_floor: float
    # Distinct from weak_signal_floor: that's the veto threshold below which
    # a deterministic signal is named as failing. This is what "the model
    # had no opinion" means when self-assessment is missing or malformed.
    # Both happen to be 0.5 today; tuning one must never silently retune
    # the other.
    neutral_model_default: float = Field(0.5, ge=0.0, le=1.0)
    # Task 11: width of the grey band above refusal_threshold in which a
    # MEDICAL-bearing profile pauses for human approval (interrupt_before)
    # rather than auto-committing. A profile carrying no MEDICAL constraint
    # auto-resumes through the same band — the interrupt is a safety gate
    # for irreversible-if-wrong medical calls, not a tax on every request.
    medical_review_band: float = Field(0.15, ge=0.0, le=1.0)


class FeasibilityConfig(BaseModel):
    max_relaxation_subset_size: int = 2


class PreferencesConfig(BaseModel):
    ema_alpha: float = Field(0.3, gt=0, le=1)


class SynthConfig(BaseModel):
    min_constraints: int = 1
    max_constraints: int = 4
    tag_constraint_probability: float = Field(0.7, ge=0, le=1)
    default_profiles: int = 200
    default_adversarial: int = 100
    # Target share of generated profiles that are FORCED infeasible (see
    # eval.synth.profiles._force_infeasible_constraints) rather than left to
    # a purely random draw — a handful of random tag exclusions over a
    # ~100-recipe catalog almost never rules out every recipe, so without
    # this the infeasible branch of feasibility_accuracy/calibration is
    # exercised on a small handful of profiles rather than a real fraction.
    infeasible_fraction: float = Field(0.25, ge=0, le=1)


class EvalConfig(BaseModel):
    axis_by_family: dict[str, str]


class HealingConfig(BaseModel):
    rule_proposal_min_cluster: int = 3


class ObsConfig(BaseModel):
    """Optional tracing/logging configuration. Everything here defaults to
    a clean no-op: `langfuse_enabled` is true only when BOTH Langfuse keys
    are present, so a reviewer with a blank `.env` never hits a credential
    error. Spec §13."""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    log_level: str = "INFO"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


class Settings(BaseSettings):
    """The ONE place configuration lives. Env overrides YAML with a `BEATROOT_`
    prefix and `__` nesting, e.g. BEATROOT_TRUST__REFUSAL_THRESHOLD=0.7."""

    model_config = SettingsConfigDict(
        env_prefix="BEATROOT_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        yaml_file=CONFIG_PATH,
    )

    llm: LLMConfig
    retrieval: RetrievalConfig
    trust: TrustConfig
    feasibility: FeasibilityConfig = FeasibilityConfig()
    preferences: PreferencesConfig = PreferencesConfig()
    healing: HealingConfig = HealingConfig()
    synth: SynthConfig = SynthConfig()
    eval: EvalConfig

    # Not part of config/beatroot.yaml — a pure runtime toggle. Kept as a
    # settings field (rather than a scattered os.getenv) so every module that
    # needs to know whether we're offline reads it through get_settings().
    offline: bool = False

    # Task 23: whether `explain_node` generates prose synchronously (the
    # default, unchanged behaviour every existing caller — CLI, eval
    # runners, golden cases — already depends on) or hands the job to
    # `agent.async_explain.ExplanationQueue` and returns immediately with
    # `explanation=""`, letting `POST /recommend` answer before the model
    # is ever called. Opt-in rather than a new default: every caller that
    # reads `Recommendation.explanation` synchronously today keeps working
    # with zero changes unless this is explicitly turned on.
    async_explanation: bool = False

    # Selects the production Qdrant vector store over the NumPy dev fallback.
    # Read from the bare `QDRANT_URL` (docker compose sets this, unprefixed)
    # rather than `BEATROOT_QDRANT_URL` — the alias opts this one field out
    # of env_prefix so `os.getenv` still appears in exactly this one module.
    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")

    # Optional tracing credentials, read from the bare LANGFUSE_* names (no
    # BEATROOT_ prefix, same reasoning as `qdrant_url` above) because
    # LiteLLM's own Langfuse integration and every other tool in this
    # ecosystem already expects those exact names. Exposed to callers only
    # through the computed `obs` property below, never read directly —
    # `obs.tracing.configure_observability()` is the one consumer.
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="", validation_alias="LANGFUSE_HOST")
    log_level: str = "INFO"

    @property
    def obs(self) -> ObsConfig:
        return ObsConfig(
            langfuse_public_key=self.langfuse_public_key,
            langfuse_secret_key=self.langfuse_secret_key,
            langfuse_host=self.langfuse_host,
            log_level=self.log_level,
        )

    def export_observability_env(self) -> None:
        """Publish the resolved Langfuse credentials into the process
        environment, because LiteLLM's `langfuse_otel` callback reads them
        from there and accepts no other configuration channel.

        This is the ONLY write-to-environment in the codebase, and it lives
        here for the same reason every read does: `settings.py` is the one
        module permitted to touch `os.environ`
        (`tests/test_settings.py::test_settings_is_the_only_module_reading_env`,
        an AST walk that catches writes as well as reads). Putting this
        helper in `obs/tracing.py` — where it would read more naturally —
        would fail that test, and the right response to that is to keep the
        rule, not to carve an exemption into it.

        Only ever fills a GAP: a variable already present in the environment
        is left exactly as it is, so an operator's `docker run -e` or a
        compose `environment:` block always outranks the config file. Does
        nothing at all when Langfuse is not configured — the keyless path
        must stay a clean no-op.
        """
        if not self.obs.langfuse_enabled:
            return
        exported = {
            "LANGFUSE_PUBLIC_KEY": self.langfuse_public_key,
            "LANGFUSE_SECRET_KEY": self.langfuse_secret_key,
            "LANGFUSE_HOST": self.langfuse_host,
        }
        for name, value in exported.items():
            if value and not os.environ.get(name):
                os.environ[name] = value

    @model_validator(mode="after")
    def _default_to_offline_without_credentials(self) -> "Settings":
        """The whole submission's premise is that a reviewer with a blank
        `.env` gets a fully working, keyless run. `offline` alone used to
        be a purely opt-in toggle (`BEATROOT_OFFLINE=1`) — everything else
        constructed a real `LLMClient` that only failed once something
        actually called it, which turned out to be `build_container()`
        itself: it eagerly embeds the whole catalog to build the vector
        store, before a single request (including `/health`) is ever
        reachable, so a credential-less real client took the whole process
        down before startup finished, in a container with no `.env`.

        `offline` is therefore not purely a user request any more: if
        neither the configured completion model nor the embedding model
        has credentials available, this flips it to True itself — as if
        `BEATROOT_OFFLINE=1` had been set — and says so at WARNING so it
        is never ambiguous which provider actually answered. An explicit
        `BEATROOT_OFFLINE=1` is unaffected; this only ever turns False
        into True, never the reverse.
        """
        if not self.offline and not (
            _provider_credentials_present(self.llm.model)
            and _provider_credentials_present(self.llm.embedding_model)
        ):
            log.warning(
                "no credentials found for llm.model=%s / llm.embedding_model=%s "
                "-- defaulting to the offline provider; supply the provider's "
                "credentials (see .env.example) to use a real model",
                self.llm.model,
                self.llm.embedding_model,
            )
            self.offline = True
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """YAML supplies the defaults; env (and an explicit `.env` file) must
        be able to override any individual leaf value without having to
        repeat the rest of the document. Pydantic-settings merges sources in
        priority order — highest first — so the YAML source goes *after* env
        and dotenv here, making it the base layer rather than the winner."""
        yaml_settings = YamlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

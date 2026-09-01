"""Prompts are content, not code — and versioned content at that.

Every prompt lives in `prompts/*.md` with YAML frontmatter declaring its
`inputs`. Loading and rendering goes through `load_prompt(name).render(**kw)`
— no prompt-shaped string literal may survive inside a Python file
(`tests/reasoning/test_prompts.py` enforces this). Spec §14.

WHERE A PROMPT COMES FROM
-------------------------
Two sources, in a fixed order:

1. **Langfuse**, when it is configured — the prompt is fetched by name at
   the `production` label, so a prompt can be revised and rolled back
   without a redeploy, and every generation in a trace can be attributed to
   the exact prompt version that produced it.
2. **The local `prompts/*.md` file**, always — as the fallback for every
   failure mode there is: no credentials, no network, a fetch error, a
   prompt that does not exist in Langfuse yet, or a remote prompt this code
   refuses (see below).

The local file is never merely a bootstrap copy. It is the authoritative
declaration of the prompt's CONTRACT — its `inputs` and its `stage` — and it
is what a bare clone with no credentials runs on. `beatroot prompts push`
publishes the local files to Langfuse; `beatroot prompts status` shows which
source each prompt actually resolved from, so "we use prompt management" is
observable rather than asserted.

WHY RENDERING STAYS LOCAL
-------------------------
Langfuse can compile a prompt itself, using mustache (`{{var}}`). This code
does not use that, and the reason is concrete rather than stylistic:
`prompts/compile_constraints.md` ends with a literal JSON example whose
braces are escaped for `str.format` (`{{"in_scope": true, ...}}` renders as
`{"in_scope": true, ...}`). Under mustache the exact same characters mean a
variable interpolation, so handing these templates to a second templating
engine would silently corrupt the one prompt whose output shape the whole
free-text path depends on. Langfuse is therefore used as the versioned
STORE for prompt text, and rendering stays with `Prompt.render` — one
templating engine, the same one offline and online, with the same escaping
rules in both.

WHY A FETCHED PROMPT CAN BE REFUSED
-----------------------------------
A remote prompt is text that someone can edit in a web UI, outside code
review, and `str.format` fails at RENDER time — which here means inside a
graph node, mid-request, on a path that has already spent tokens. So a
fetched template is checked BEFORE it is trusted: every `{placeholder}` in
it must be declared in the local file's `inputs`. A remote edit introducing
`{calorie_target}` where no such input exists is rejected at load, logged at
WARNING, and the local file is used instead. The failure becomes a startup
log line rather than a mid-request `KeyError`, which is the same posture
every other degradation in this codebase takes: refuse the untrusted thing,
keep serving, and say so.
"""

import logging
from functools import cache
from pathlib import Path
from string import Formatter
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from beatroot.settings import get_settings

log = logging.getLogger("beatroot.prompts")

PROMPTS_DIR = Path(__file__).parents[3] / "prompts"

# The Langfuse label this application resolves prompts at. A prompt version
# only reaches production by being promoted to this label — publishing a new
# version alone does not change what runs, which is the point of having a
# label at all.
PRODUCTION_LABEL = "production"

PromptSource = Literal["local", "langfuse"]


class Prompt(BaseModel):
    """One prompt, resolved from wherever it actually came from.

    `version` is the local file's own frontmatter version — the contract
    version, bumped in code review. `remote_version` is Langfuse's integer
    version for the fetched text, and is `None` for a locally-resolved
    prompt. Both are carried because they answer different questions: the
    first is "which contract does this code expect", the second is "which
    text actually ran".
    """

    id: str
    version: int
    stage: str
    inputs: list[str]
    template: str
    source: PromptSource = "local"
    remote_version: int | None = None

    def render(self, **kwargs: object) -> str:
        missing = set(self.inputs) - set(kwargs)
        if missing:
            raise KeyError(f"prompt {self.id} missing inputs: {sorted(missing)}")
        return self.template.format(**kwargs)

    @property
    def ref(self) -> str:
        """Compact provenance string for logs and trace metadata, e.g.
        `explain@langfuse:v4` or `explain@local:v1`. One token that answers
        "which text produced this generation", which is the whole reason a
        prompt registry is worth wiring in."""
        version = self.remote_version if self.source == "langfuse" else self.version
        return f"{self.id}@{self.source}:v{version}"

    def trace_metadata(self) -> dict[str, Any]:
        """The prompt-provenance fields attached to every model call made
        with this prompt, so a generation in Langfuse can be traced back to
        the exact version that produced it rather than to a prompt name that
        has since been edited."""
        return {
            "prompt_name": self.id,
            "prompt_ref": self.ref,
            "prompt_source": self.source,
            "prompt_version": self.remote_version if self.source == "langfuse" else self.version,
        }


def placeholders(template: str) -> set[str]:
    """Every `{name}` `str.format` would try to substitute.

    `Formatter().parse` is used rather than a regex precisely because it
    applies `str.format`'s own escaping rules: `{{` is a literal brace and
    yields no field, which is what keeps `compile_constraints`'s JSON
    example from reading as four undeclared variables. A regex over `\\{(\\w+)\\}`
    got this wrong, and the whole point of this function is to be exactly as
    strict as the engine that will run the template.

    Positional/auto-numbered fields (`{}` or `{0}`) are not a thing any
    prompt here uses; an empty or numeric field name is returned as-is so
    the caller's subset check rejects it rather than silently allowing it.
    """
    return {name for _, name, _, _ in Formatter().parse(template) if name is not None}


def _load_local(name: str, directory: Path | None = None) -> Prompt:
    """The local file, parsed. This is the contract: `inputs` and `stage`
    come from here even when the TEXT came from Langfuse, because they are
    what the calling code was written against."""
    text = ((directory or PROMPTS_DIR) / f"{name}.md").read_text()
    _, front, body = text.split("---", 2)
    return Prompt(**yaml.safe_load(front), template=body.strip(), source="local")


def langfuse_client() -> Any | None:
    """A configured Langfuse client, or `None` when Langfuse is not set up
    or the SDK is not installed (it is an optional `obs` extra, not a hard
    dependency).

    Returns `None` rather than raising on every failure path: prompt
    management is an enhancement over reading a file off disk, and nothing
    here is permitted to turn its absence into a broken application.
    """
    obs = get_settings().obs
    if not obs.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("langfuse credentials are set but the SDK is not installed; using local prompts")
        return None
    try:
        return Langfuse(
            public_key=obs.langfuse_public_key,
            secret_key=obs.langfuse_secret_key,
            host=obs.langfuse_host or None,
        )
    except Exception as exc:
        log.warning("could not construct a Langfuse client (%s); using local prompts", exc)
        return None


def _fetch_remote(name: str, local: Prompt) -> Prompt | None:
    """The `production`-labelled Langfuse version of `name`, validated
    against `local`'s declared inputs, or `None` to fall back.

    Every return of `None` here is a deliberate fallback, never an error the
    caller has to handle: an unconfigured Langfuse, an absent prompt, a
    network failure, and a remote template this code will not run all land
    in the same place — the local file.
    """
    client = langfuse_client()
    if client is None:
        return None
    try:
        fetched = client.get_prompt(name, label=PRODUCTION_LABEL)
    except Exception as exc:
        # Includes the ordinary "not published yet" case, which is why this
        # is INFO-with-a-reason rather than an error: a fresh Langfuse
        # project has no prompts until `beatroot prompts push` runs, and
        # that must not read as a fault.
        log.info("prompt %r not resolved from Langfuse (%s); using the local file", name, exc)
        return None

    template = (getattr(fetched, "prompt", "") or "").strip()
    if not template:
        log.warning("Langfuse returned an empty template for %r; using the local file", name)
        return None

    undeclared = placeholders(template) - set(local.inputs)
    if undeclared:
        # The guard the module docstring describes: a remote edit that would
        # raise KeyError mid-request is refused at load instead.
        log.warning(
            "Langfuse prompt %r references undeclared input(s) %s (declared: %s); "
            "refusing it and using the local file",
            name,
            sorted(undeclared),
            sorted(local.inputs),
        )
        return None

    return local.model_copy(
        update={
            "template": template,
            "source": "langfuse",
            "remote_version": getattr(fetched, "version", None),
        }
    )


@cache
def load_prompt(name: str, directory: Path | None = None) -> Prompt:
    """Resolve `name` from Langfuse if it is configured, else from disk.

    Cached for the process lifetime, like the file-only version it replaces:
    a prompt fetch on every model call would add a network round trip to the
    request path to save a redeploy, which is the wrong trade for a system
    whose latency is already dominated by serial model calls. `beatroot
    prompts refresh` (and `load_prompt.cache_clear()`) is how a promoted
    version is picked up without a restart.

    `directory` forces the local file location and is used by tests; it also
    skips Langfuse entirely, so a test pointed at a fixture directory can
    never be silently answered by a real remote prompt.
    """
    local = _load_local(name, directory)
    if directory is not None:
        return local
    return _fetch_remote(name, local) or local


def resolved_sources() -> dict[str, str]:
    """`{prompt_name: ref}` for every local prompt, resolved the same way a
    real call resolves it. Powers `beatroot prompts status` and the health
    surface, so which prompt text is actually live is something you can read
    rather than infer."""
    return {path.stem: load_prompt(path.stem).ref for path in sorted(PROMPTS_DIR.glob("*.md"))}


def push_prompts(directory: Path | None = None) -> list[dict[str, Any]]:
    """Publish every local prompt to Langfuse at the `production` label.

    Idempotent in effect: Langfuse creates a new version per call and moves
    the label to it, so re-running produces a new version number with
    identical text rather than a duplicate or an error. The local `version`,
    `stage` and `inputs` ride along in the prompt's `config`, so someone
    editing text in the Langfuse UI can see which inputs are legal without
    having to open this repository.

    Returns one result dict per prompt (`name`, `ok`, `detail`) rather than
    raising on the first failure — a partial push should report exactly
    which prompts landed, not abort halfway with no account of what changed.
    """
    directory = directory or PROMPTS_DIR
    client = langfuse_client()
    if client is None:
        return [
            {
                "name": "-",
                "ok": False,
                "detail": "Langfuse is not configured (set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY).",
            }
        ]

    results: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.md")):
        name = path.stem
        local = _load_local(name, directory)
        try:
            created = client.create_prompt(
                name=name,
                prompt=local.template,
                labels=[PRODUCTION_LABEL],
                type="text",
                config={
                    "stage": local.stage,
                    "inputs": local.inputs,
                    "local_version": local.version,
                    # Stated in the stored config because the Langfuse UI
                    # renders mustache by default, and someone editing this
                    # text needs to know which engine will actually run it.
                    "templating": "python str.format ({name}), NOT mustache ({{name}})",
                },
            )
            results.append(
                {"name": name, "ok": True, "detail": f"v{getattr(created, 'version', '?')}"}
            )
        except Exception as exc:
            results.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})
    return results

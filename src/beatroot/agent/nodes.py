"""Node functions for the meal-planning `StateGraph`. Spec §6, §9.

Every node is a pure `PlanState -> partial PlanState` function closed over a
frozen `Deps` dataclass — no globals, no service locator. Dependencies
(catalog, LLM client, vector store, tag index, incident/audit logs, the
skills registry, preference/feasibility caches) are supplied once, at graph
build time, by whoever assembles this process (a CLI entrypoint, a test
fixture, eventually Task 12's Container).

Three boundary properties matter more than any single node's body:

- **Zero-token infeasibility.** `compile` runs first (GAP 1 + FREE-TEXT/NLP:
  it turns optional free text into constraints at whatever severity
  actually enforces them, and can itself refuse an out-of-scope request)
  but is a genuine no-op — no trace entry, no model call — when there is no
  free text, so `feasibility` is still effectively the first node to spend
  anything. When no recipe survives, it routes straight to `negotiate` —
  `retrieve`, `score` (the only nodes that ever call the model on the happy
  path) never run. Feature 1's whole economic argument depends on this
  being a routing fact, not a hope.
- **The model never gets the last word.** `verify` runs after `explain` and
  re-checks the CHOSEN recipe against the same `ConstraintSet` that produced
  the candidates (`check_recipe`, deterministic, no model), and diffs every
  number the explanation stated against catalog truth (`detect_drift`). Both
  can route to `escalate` even after a trust-gated model has already spoken.
- **No node's exception ever escapes `graph.invoke()`.** Every node returned
  from `make_nodes` is wrapped so an unhandled exception routes to ESCALATE
  (logged, incident recorded) rather than leaving a thread's checkpoint stuck
  mid-graph with no terminal and no record that anything happened at all —
  the one path where the system neither succeeds nor declines is exactly the
  one a system built around "decline safely" cannot afford to leave open.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import ValidationError

from beatroot.agent.async_explain import ExplanationQueue
from beatroot.agent.skills_registry import Skill
from beatroot.agent.state import PlanState
from beatroot.confirm.escalation import gate
from beatroot.confirm.trust_score import load_thresholds, score
from beatroot.contracts.core import (
    HARD_SEVERITIES,
    Constraint,
    ConstraintKind,
    ConstraintSet,
    Severity,
)
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.contracts.result import Escalation, Negotiation, RecipeIngredient, Recommendation
from beatroot.contracts.trust import CostRecord, TrustReport
from beatroot.eval.verifiers.nutrition_drift import detect_drift
from beatroot.obs.cost import estimate_tokens
from beatroot.reasoning.llm import LLMClient
from beatroot.reasoning.prompts import load_prompt
from beatroot.retrieval.dense import VectorStore
from beatroot.retrieval.query_rewrite import rewrite_query
from beatroot.retrieval.rerank import llm_rerank, retrieve
from beatroot.settings import Settings, get_settings
from beatroot.store.audit import AuditLog
from beatroot.store.cache import FeasibilityCache
from beatroot.store.incidents import IncidentLog
from beatroot.t0_invariants.constraints import (
    CheckResult,
    ConstraintVocabulary,
    check_recipe,
    get_parser,
    kind_shapes,
    registered_kinds,
)
from beatroot.t0_invariants.feasibility import assess, rank_relaxations
from beatroot.t0_invariants.vocabulary import unknown_vocabulary
from beatroot.trusted.catalog import Catalog, Recipe
from beatroot.trusted.index import TagIndex

log = logging.getLogger("beatroot.agent")

# Nodes whose outgoing edge is UNCONDITIONAL straight to END (Spec §6's three
# terminals). A guarded exception here has no downstream `escalate` node to
# fall through to, so unlike every other node it has to record the incident
# and audit itself rather than relying on `escalate_node` running next.
_TERMINAL_STAGES = frozenset({"commit", "negotiate", "escalate"})


class _Preferences(Protocol):
    """Structural type for `store.preferences.PreferenceMemory` (Task 17).

    `agent/` stays untangled from `store.preferences`'s concrete class this
    way — `Deps.preferences` only needs the one method `retrieve_node`
    actually calls. `build_container` wires in a real `PreferenceMemory`,
    which satisfies this structurally; tests that don't care about
    preference memory can pass `None` or any object shaped like this.
    """

    def affinity(self, profile_id: str) -> dict[str, float]: ...


@dataclass(frozen=True)
class Deps:
    """Everything a node needs, closed over once at graph build time."""

    catalog: Catalog
    llm: LLMClient
    vector_store: VectorStore
    tag_index: TagIndex
    incidents: IncidentLog
    audit: AuditLog
    skills: dict[str, Skill]
    preferences: _Preferences | None = None
    feasibility_cache: FeasibilityCache | None = None
    # Task 23: present only when `settings.async_explanation` is on
    # (`container.build_container` wires the two together). `explain_node`
    # checks BOTH the setting and this being non-None before taking the
    # async branch, so a Deps built by hand (every test in this file's
    # conftest) without this field keeps running EXPLAIN synchronously,
    # exactly as before.
    explanation_queue: ExplanationQueue | None = None


def skill_versions(deps: Deps) -> dict[str, str]:
    """`{skill_id: digest[:12]}` for every audit record — provenance, not a
    hand-typed version string. Public (not `_`-prefixed): both `nodes.py` and
    `graph.py` (the pause-audit path) need it."""
    return {sid: s.digest()[:12] for sid, s in deps.skills.items()}


def _require_recipe(catalog: Catalog, recipe_id: str) -> Recipe:
    """A `chosen_id` that no longer resolves in the catalog (a reseed racing
    this run, or a corrupted/replayed checkpoint) is an internal
    inconsistency, not a constraint failure. Raising here lets the existing
    node guards (`_guarded`/`_guarded_terminal`) turn it into a named
    ESCALATE with a full traceback in the log, instead of an opaque
    `AttributeError` several call frames deeper."""
    recipe = catalog.recipe(recipe_id)
    if recipe is None:
        raise LookupError(f"chosen_id '{recipe_id}' no longer resolves in the catalog")
    return recipe


def _catalog_tag_vocabulary(catalog: Catalog) -> set[str]:
    """Every tag any recipe in the catalog carries — the same "known tag"
    universe `t0_invariants.vocabulary._known_tags` validates structured
    constraints against. Recomputed independently here rather than shared:
    this module already depends on `reasoning` (GAP 1's `compile` node calls
    the model), which `t0_invariants` must never do
    (`tests/test_boundaries.py`), so the two cannot import one shared
    helper without pulling `reasoning` into a T0 module."""
    tags: set[str] = set()
    for recipe in catalog.recipes():
        tags |= recipe.tags
    return tags


def _catalog_cuisine_vocabulary(catalog: Catalog) -> set[str]:
    """Every cuisine any recipe in the catalog actually carries — read at
    runtime, exactly like `_catalog_tag_vocabulary`, so `compile_node`'s
    prompt (and the validation in `_parse_compiled_constraint`) reflect
    whatever cuisines the catalog holds today. No separately maintained
    list to drift when a cuisine is added or removed from the data."""
    return {r.cuisine for r in catalog.recipes() if r.cuisine}


def _render_kind_shapes() -> str:
    """The KIND SHAPES section of `prompts/compile_constraints.md`,
    generated from `t0_invariants.constraints.kind_shapes()` — the single
    registration `@evaluator(kind, shape=..., parse=...)` — rather than a
    separately maintained bullet list in the prompt file. A kind's shape
    text lives exactly once, next to the evaluator and parser it describes;
    a kind registered with no shape text is impossible by construction
    (`evaluator()` requires `shape=`), so there is nothing here to drift out
    of sync with `registered_kinds()`."""
    shapes = kind_shapes()
    return "\n".join(f"  - {kind}: {shapes[kind]}" for kind in sorted(shapes))


# FREE-TEXT/NLP task: the model classifies WHAT KIND of statement a parsed
# constraint is (its own open-ended judgement, in its own words mapped onto
# this fixed vocabulary); this table is what turns that classification into
# an enforcement `Severity`, deterministically, in code. It is NOT a lookup
# keyed on tag/ingredient/phrase content — it is the same five-member
# `Severity` taxonomy `contracts/core.py` already defines, so a new dietary
# identity or allergen needs no change here, only an honest category label
# from the model. The model never gets to write a raw `severity` string
# that bypasses this — see `_parse_compiled_constraint`.
_CATEGORY_SEVERITY: dict[str, Severity] = {
    "medical": Severity.MEDICAL,
    "religious": Severity.RELIGIOUS,
    "dietary": Severity.DIETARY,
    "goal": Severity.GOAL,
    "preference": Severity.PREFERENCE,
}

# The deterministic RATCHET, content-independent: `require_tag`/
# `require_any_tag` exist specifically as the allowlist primitive for a
# categorical dietary identity (t0_invariants.constraints' own docstring).
# A model that chose one of these KINDS has already made an identity claim
# by the shape of what it emitted, not by any phrase it used — so that
# claim is floored at DIETARY severity regardless of which `category` label
# came with it. This generalises to any identity the model expresses this
# way (halal, kosher, one nobody wrote a test for) because it keys on the
# constraint's kind, never on its value.
_IDENTITY_KINDS = frozenset({"require_tag", "require_any_tag"})


def _parse_compiled_constraint(
    index: int,
    item: object,
    known_kinds: set[str],
    known_tags: set[str],
    known_cuisines: set[str],
) -> Constraint | None:
    """One item from the model's open-vocabulary `constraints` list -> a
    `Constraint`, or `None` if ANY check fails — never guessed past, never
    silently coerced into something the model didn't actually say.

    This function no longer knows any kind's value SHAPE itself — that lives
    entirely in the T0 registry (`t0_invariants.constraints`'s
    `@evaluator(kind, shape=..., parse=...)`), reached here via
    `get_parser(kind)`. What stays HERE, deliberately not registration-
    driven: the id, and the severity ratchet (`_CATEGORY_SEVERITY` /
    `_IDENTITY_KINDS`) — a model-proposed category becomes a `Severity` only
    through code an evaluator cannot influence, so no kind can declare
    itself soft. `exclude_ingredient` is deliberately NOT checked against a
    vocabulary: `t0_invariants.vocabulary.unknown_vocabulary` already
    resolves it against the real ingredient/synonym table at FEASIBILITY and
    escalates if it names nothing real — reused, not re-implemented.
    """
    if not isinstance(item, dict):
        return None
    kind = item.get("kind")
    if not isinstance(kind, str) or kind not in known_kinds:
        return None

    category = item.get("category")
    severity = _CATEGORY_SEVERITY.get(category) if isinstance(category, str) else None
    if severity is None:
        # The model didn't clearly state a category this system knows how
        # to enforce — drop the item rather than default it to soft (which
        # would silently under-enforce a real allergy) or to hard (which
        # would silently over-refuse a real dislike). Neither is a guess
        # this function gets to make.
        return None
    if kind in _IDENTITY_KINDS and severity not in HARD_SEVERITIES:
        severity = Severity.DIETARY

    parser = get_parser(kind)
    if parser is None:
        # `kind` is in `known_kinds` (== registered_kinds()), so `evaluator`
        # requiring `parse=` on every registration means this is genuinely
        # unreachable — kept as a guard, not a silent drop, in case that
        # invariant is ever weakened.
        return None
    vocabulary = ConstraintVocabulary(
        known_tags=frozenset(known_tags), known_cuisines=frozenset(known_cuisines)
    )
    parsed = parser(item, vocabulary)
    if parsed is None:
        return None
    value, nutrient = parsed

    try:
        return Constraint(
            id=f"parsed_{index}_{kind}",
            kind=cast(ConstraintKind, kind),
            severity=severity,
            value=value,
            nutrient=nutrient,
            source="parsed_free_text",
        )
    except ValidationError:
        return None


def _estimate_skipped_tokens(deps: Deps, cfg: Settings) -> int:
    """GAP 2: a defensible ESTIMATE of the tokens a short-circuit
    (infeasible -> NEGOTIATE, unknown vocabulary -> ESCALATE) avoided
    spending — grounded in the two prompts that never ran (`rerank` and
    `explain`), not an invented constant.

    Renders both prompts against real catalog data standing in for what
    those stages would actually have seen: up to `settings.retrieval.
    top_k` real recipes for the rerank candidate listing (the exact
    listing shape `retrieval.rerank.llm_rerank` builds), and one
    representative recipe's real, computed nutrition facts for explain.
    Only that small sample is hydrated — never the whole catalog — so this
    stays cheap on every call, including the zero-token paths it is
    computing an estimate FOR.

    Character count is converted to a token estimate via
    `obs.cost.estimate_tokens` (~4 characters/token, the commonly cited
    rule of thumb for English text under BPE-style tokenizers) — an
    approximation, stated as one. `/metrics` reports the resulting ledger
    total as `tokens_saved` alongside `tokens_saved_estimate_method`,
    never presenting it as a measured spend.
    """
    catalog_recipes = deps.catalog.recipes()
    if not catalog_recipes:
        return 0
    top_k = max(1, cfg.retrieval.top_k)
    sample = [deps.catalog.hydrate(r) for r in catalog_recipes[:top_k]]

    listing = "\n".join(
        f"{i}. {r.name} ({r.cuisine}, {r.prep_minutes} min)" for i, r in enumerate(sample)
    )
    rerank_text = load_prompt("rerank").render(query="", preferences="", candidates=listing)

    representative = sample[0]
    if representative.nutrition is not None:
        n = representative.nutrition
        facts = ", ".join(
            f"{f}={getattr(n, f)}"
            for f in ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")
        )
    else:
        facts = ""
    explain_text = load_prompt("explain").render(
        name=representative.name, facts=facts, satisfied=""
    )

    return estimate_tokens(rerank_text) + estimate_tokens(explain_text)


def _internal_error_escalation(stage: str, exc: Exception) -> Escalation:
    """Shape an unhandled exception into a valid `Escalation`.

    `Escalation.reason` is a closed `Literal` (Task 1's `contracts/`) that
    does not include an "internal_error" member, and widening it is outside
    this fix round's file scope — `constraint_uncheckable` is the closest
    existing member ("we could not reliably determine this"), so `reason`
    stays there and the actually-diagnostic label lives in `failing_signal`,
    which is a free string. The full traceback goes to the log, keyed by
    stage, rather than into `detail` (which lands in the audit/incident
    tables and does not need a stack trace bloating every row).
    """
    log.exception("stage=%s raised an unhandled exception", stage)
    return Escalation(
        reason="constraint_uncheckable",
        failing_signal="internal_error",
        detail=f"unhandled exception in stage '{stage}': {type(exc).__name__}: {exc}",
    )


def _guarded(stage: str, fn: Callable[[PlanState], PlanState]) -> Callable[[PlanState], PlanState]:
    """Wrap a non-terminal node: on an unhandled exception, log it and
    return an ESCALATE-shaped partial state. The conditional edge after
    every non-terminal node already sends `terminal == "ESCALATE"` to
    `escalate_node`, which records the incident and audit — this wrapper
    only has to log and shape the escalation, not duplicate that
    bookkeeping."""

    def wrapped(state: PlanState) -> PlanState:
        try:
            return fn(state)
        except Exception as exc:
            esc = _internal_error_escalation(stage, exc)
            return {
                "trace": [stage.upper()],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }

    wrapped.__name__ = getattr(fn, "__name__", stage)
    return wrapped


def _guarded_terminal(
    stage: str, deps: Deps, fn: Callable[[PlanState], PlanState]
) -> Callable[[PlanState], PlanState]:
    """Wrap `commit`/`negotiate`/`escalate`. Their outgoing edge is
    unconditional straight to END (Spec §6's three terminals), so nothing
    downstream will ever see a failure here — this records the incident and
    audit itself instead of counting on `escalate_node` to run."""

    def wrapped(state: PlanState) -> PlanState:
        try:
            return fn(state)
        except Exception as exc:
            esc = _internal_error_escalation(stage, exc)
            raw_cs = state.get("constraint_set") or {}
            profile_id = raw_cs.get("profile_id", "unknown")
            deps.incidents.record(
                "escalation",
                profile_id,
                esc.detail,
                {
                    "reason": esc.reason,
                    "failing_signal": esc.failing_signal,
                    # Already exactly `ConstraintSet.model_dump(mode="json")`
                    # shape (graph.py sets state["constraint_set"] that way at
                    # run start) — no re-validation needed to carry it along.
                    "constraint_set": raw_cs,
                    "terminal": "ESCALATE",
                },
            )
            aid = deps.audit.record(
                profile_id,
                "ESCALATE",
                esc.model_dump(mode="json"),
                skill_versions(deps),
                esc.cost.usd,
            )
            return {
                "trace": [stage.upper()],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
                "audit_id": aid,
            }

    wrapped.__name__ = getattr(fn, "__name__", stage)
    return wrapped


def _ingredient_lines(catalog: Catalog, recipe: Recipe) -> list[RecipeIngredient]:
    """The recipe's ingredients, resolved to display names and grams.

    `Recipe.ingredient_ids` carries ids only; the grams live in the raw recipe
    payload and the human name on the ingredient payload, so both are read
    here rather than reconstructed anywhere downstream. An id the catalog
    cannot resolve still appears, under its own id — omitting an ingredient
    from a list a diner uses to check for allergens is the one failure mode
    worth avoiding, so an unresolvable line degrades to visible rather than
    absent.
    """
    payload = catalog.recipe_payload(recipe.id) or {}
    grams_by_id: dict[str, float] = {}
    for entry in payload.get("ingredients") or []:
        if isinstance(entry, dict):
            iid, g = entry.get("ingredient_id"), entry.get("grams")
            if isinstance(iid, str) and isinstance(g, int | float) and not isinstance(g, bool):
                grams_by_id[iid] = float(g)

    lines: list[RecipeIngredient] = []
    for iid in recipe.ingredient_ids:
        ing = catalog.ingredient_payload(iid) or {}
        name = ing.get("name")
        lines.append(
            RecipeIngredient(
                ingredient_id=iid,
                name=name if isinstance(name, str) and name else iid,
                grams=grams_by_id.get(iid),
            )
        )
    return lines


def _describe_satisfied(cs_data: object, satisfied_ids: list[str]) -> str:
    """Human-readable constraint descriptions for the explanation prompt.

    Handing the model raw ids made it write "satisfies the med0 constraint"
    into user-facing prose — an internal identifier leaking through the one
    surface a diner actually reads. The model can only repeat what it is
    given, so it is given language rather than keys.
    """
    # Graph state carries the ConstraintSet as a serialised dict so it can
    # round-trip through SqliteSaver; accept either form.
    if cs_data is None:
        return "no explicit dietary constraints"
    try:
        cs = (
            cs_data if isinstance(cs_data, ConstraintSet) else ConstraintSet.model_validate(cs_data)
        )
    except ValidationError:
        # Cosmetic phrasing must never cost a recommendation. If the state
        # shape is unexpected, fall back to neutral wording rather than
        # escalating a meal that is already verified and safe.
        return "the stated dietary constraints"
    by_id = {c.id: c for c in cs.constraints}
    parts: list[str] = []
    for cid in satisfied_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        if c.kind in ("exclude_tag", "exclude_ingredient"):
            parts.append(f"contains no {c.value}")
        elif c.kind == "require_tag" and isinstance(c.value, str):
            parts.append(f"is {c.value}")
        elif c.kind == "require_any_tag" and isinstance(c.value, list) and c.value:
            parts.append(f"is {' or '.join(str(v) for v in c.value)}")
        elif c.kind == "max_prep_minutes" and isinstance(c.value, int | float):
            parts.append(f"ready within {c.value:g} minutes")
        elif c.kind == "budget_max" and isinstance(c.value, int | float):
            parts.append(f"costs no more than {c.value:g}")
        elif c.kind == "exclude_cuisine":
            names = c.value if isinstance(c.value, list) else [c.value]
            parts.append(f"is not {' or '.join(str(v) for v in names)}")
        elif c.kind == "cuisine_affinity" and isinstance(c.value, str):
            parts.append(f"is {c.value}")
        elif c.kind == "nutrient_range" and isinstance(c.value, list | tuple):
            lo, hi = c.value[0], c.value[1]
            parts.append(f"{c.nutrient or 'nutrient'} between {lo:g} and {hi:g}")
    return "; ".join(parts) if parts else "no explicit dietary constraints"


def _satisfied_display(cs_data: object, satisfied_ids: list[str]) -> list[str]:
    """The same language as `_describe_satisfied`, as a list for the UI."""
    text = _describe_satisfied(cs_data, satisfied_ids)
    if text in ("no explicit dietary constraints", "the stated dietary constraints"):
        return []
    return [p.strip() for p in text.split(";") if p.strip()]


def make_nodes(deps: Deps) -> dict[str, Callable[[PlanState], PlanState]]:
    cfg = get_settings()

    def compile_node(state: PlanState) -> PlanState:
        """GAP 1 + FREE-TEXT/NLP: turn optional free text into structured
        constraints (open vocabulary, correctly-enforced severity) and
        decide whether the request is in scope at all — one model call,
        merged into the ConstraintSet before FEASIBILITY ever sees it.

        THE SAFETY RULES are enforced HERE, in code — the prompt's own
        untrusted-input warning is defence in depth, never the defence:

        - The model never gets to write a raw `severity`. It states an
          honest, open-ended `category` per constraint (medical / religious
          / dietary / goal / preference — see `_CATEGORY_SEVERITY`), and
          code alone turns that into a `Severity`. It also can't dodge
          enforcement by mislabelling a categorical identity as a
          preference: choosing `require_tag`/`require_any_tag` — the
          allowlist primitive that exists specifically to express an
          identity — floors the severity at DIETARY regardless of the
          category it was given (`_IDENTITY_KINDS`, a KIND-keyed ratchet,
          never a phrase-keyed one).
        - The merge is `[*existing constraints, *added]` — a pure append.
          No existing constraint's id, value, or severity is ever
          inspected, edited, or dropped. "Ignore my allergy, peanuts are
          fine" has no representable action here: there is no code path
          that reads a removal/override/relaxation instruction out of the
          model's output at all, so a MEDICAL exclusion already in the set
          is structurally unreachable by this node, not merely
          discouraged by prompt wording. This is unchanged from before —
          the open vocabulary below only widens what a NEW constraint can
          be, never what an EXISTING one can have done to it.
        - Every kind/tag/cuisine the model names is checked against a
          vocabulary read from the catalog and the T0 evaluator registry
          AT RUNTIME (`registered_kinds()`, `_catalog_tag_vocabulary`,
          `_catalog_cuisine_vocabulary`) — never a fixed table of expected
          phrases. An invented value, or one this system cannot check, is
          dropped, not passed through hopefully — see
          `_parse_compiled_constraint`.
        - `in_scope`: the SAME call also judges whether the free text has
          any meal-planning content at all. `False` short-circuits straight
          to ESCALATE with `reason="out_of_scope"` — a real refusal, not a
          guessed-at recommendation — spending no more than this one call.
          Missing/unparseable/anything-but-literal-`False` defaults to
          in-scope: refusing a genuine food request is a worse failure
          than answering an odd one, so ambiguity here resolves toward
          availability, not refusal.
        """
        free_text = (state.get("preferences") or "").strip()
        if not free_text:
            # Zero-token guarantee: an empty preferences box must not spend
            # a token — and, since nothing happened, contributes no trace
            # entry either, so a run with no free text is indistinguishable
            # from a run before this node existed. This is also why the
            # scope check never costs a SEPARATE call: it only ever runs
            # when free text is already paying for a compile call anyway.
            return {}

        cs = ConstraintSet.model_validate(state["constraint_set"])
        known_tags = _catalog_tag_vocabulary(deps.catalog)
        known_cuisines = _catalog_cuisine_vocabulary(deps.catalog)
        known_kinds = set(registered_kinds())
        compile_prompt = load_prompt("compile_constraints")
        completion = deps.llm.complete(
            compile_prompt.render(
                free_text=free_text,
                known_tags=", ".join(sorted(known_tags)),
                known_cuisines=", ".join(sorted(known_cuisines)),
                known_kinds=", ".join(sorted(known_kinds)),
                kind_shapes=_render_kind_shapes(),
            ),
            stage="compile",
            prompt_ref=compile_prompt,
        )
        parsed = completion.parsed or {}
        result: PlanState = {"trace": ["COMPILE"], "cost": completion.cost.model_dump(mode="json")}

        if parsed.get("in_scope") is False:
            esc = Escalation(
                reason="out_of_scope",
                failing_signal="scope",
                detail=(
                    "this request has no meal-planning content to act on — "
                    "refusing rather than guessing at a recommendation."
                ),
                cost=completion.cost,
            )
            result["escalation"] = esc.model_dump(mode="json")
            result["terminal"] = "ESCALATE"
            return result

        added: list[Constraint] = []
        raw_items = parsed.get("constraints")
        if isinstance(raw_items, list):
            for i, item in enumerate(raw_items[:50]):  # bounded, same as RecommendRequest
                constraint = _parse_compiled_constraint(
                    i, item, known_kinds, known_tags, known_cuisines
                )
                if constraint is not None:
                    added.append(constraint)
        else:
            # Legacy/degraded shape — the offline stub, or any reply that
            # skipped the open-vocabulary schema entirely: a flat
            # `exclude_tags` list, the same narrow behaviour this node had
            # before the schema above existed. Every accepted tag becomes
            # an `exclude_tag` constraint at PREFERENCE severity, nothing
            # else — never guessed harder than that from a degraded reply.
            raw_tags = parsed.get("exclude_tags")
            if isinstance(raw_tags, list):
                for i, tag in enumerate(raw_tags):
                    if isinstance(tag, str) and tag in known_tags:
                        added.append(
                            Constraint(
                                id=f"parsed_{i}_{tag}",
                                kind="exclude_tag",
                                severity=Severity.PREFERENCE,
                                value=tag,
                                source="parsed_free_text",
                            )
                        )

        if added:
            merged = cs.model_copy(update={"constraints": [*cs.constraints, *added]})
            result["constraint_set"] = merged.model_dump(mode="json")
        return result

    def feasibility(state: PlanState) -> PlanState:
        cs = ConstraintSet.model_validate(state["constraint_set"])

        # Vocabulary check runs FIRST, before a single recipe is scanned or
        # a token spent: exclude_tag/exclude_ingredient are pure membership
        # tests (t0_invariants.constraints), so a value naming a tag/id the
        # catalog has never heard of is vacuously "satisfied" by every
        # recipe rather than flagged as unverifiable. A constraint we
        # cannot check is not a constraint we satisfied — see
        # t0_invariants.vocabulary's module docstring. Checked regardless
        # of severity: an unverifiable PREFERENCE must escalate exactly
        # like an unverifiable MEDICAL constraint would, or this
        # reintroduces the same silent-pass hole one severity down.
        unknown = unknown_vocabulary(cs, deps.catalog)
        if unknown:
            ids = ", ".join(c.id for c in unknown)
            detail = "; ".join(
                f"{c.id} ({c.severity}) names an unrecognised value: {c.value!r}" for c in unknown
            )
            deps.incidents.record(
                "unknown_ingredient",
                cs.profile_id,
                detail,
                {
                    "constraint_ids": [c.id for c in unknown],
                    # Carried so the healing loop (Task 16) can replay this
                    # exact ConstraintSet instead of an empty one.
                    "constraint_set": cs.model_dump(mode="json"),
                    "terminal": "ESCALATE",
                },
            )
            esc = Escalation(
                reason="unknown_ingredient",
                failing_signal=ids,
                detail=f"cannot verify constraint(s) against the catalog vocabulary: {detail}",
                # GAP 2: RETRIEVE/SCORE/EXPLAIN never ran — this is a
                # defensible ESTIMATE of what they would have cost, not a
                # measured spend. See `_estimate_skipped_tokens`.
                cost=CostRecord(tokens_saved=_estimate_skipped_tokens(deps, cfg)),
            )
            return {
                "trace": ["FEASIBILITY"],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }

        recipes = deps.catalog.hydrated()

        cached_ids = deps.feasibility_cache.get(cs) if deps.feasibility_cache else None
        if cached_ids is not None:
            # Cache HIT. A cached empty list is a legitimate answer for a
            # profile with no feasible recipes (store.cache.FeasibilityCache
            # docstring) — it must NOT be treated as a miss and recomputed
            # via a full assess(), or every infeasible profile pays the
            # O(catalog) scan again on every request, defeating the cache
            # for exactly the case it exists to speed up.
            surviving = [r for r in recipes if r.id in set(cached_ids)]
            if surviving:
                return {"trace": ["FEASIBILITY"], "surviving_ids": [r.id for r in surviving]}
            negotiation = Negotiation(
                total_candidates=len(recipes),
                surviving=0,
                relaxations=rank_relaxations(
                    cs, recipes, deps.tag_index, cfg.feasibility.max_relaxation_subset_size
                ),
                locked=[c.id for c in cs.hard()],
                # GAP 2: a cache-hit infeasible short-circuit skips
                # RETRIEVE/SCORE/EXPLAIN just as surely as a freshly
                # computed one — see `_estimate_skipped_tokens`.
                cost=CostRecord(tokens_saved=_estimate_skipped_tokens(deps, cfg)),
            )
        else:
            result = assess(cs, recipes, deps.tag_index, cfg.feasibility.max_relaxation_subset_size)
            if deps.feasibility_cache:
                deps.feasibility_cache.put(cs, [r.id for r in result.surviving])
            if result.feasible:
                return {"trace": ["FEASIBILITY"], "surviving_ids": [r.id for r in result.surviving]}
            if result.negotiation is None:
                # assess() always sets .negotiation when .feasible is False
                # — checked, not just assumed, so a future assess()
                # regression that drops it fails loudly here (this node is
                # `_guarded`, so the exception still becomes a clean
                # ESCALATE) instead of silently continuing on a
                # negotiation of None.
                raise RuntimeError("assess() returned infeasible with no negotiation")
            negotiation = result.negotiation
            # `assess()` (t0_invariants, no model) never sets `cost` — fill
            # in the GAP 2 estimate here rather than relying on the
            # Pydantic default silently leaving `tokens_saved` at 0, so
            # "we spent nothing on RETRIEVE/SCORE/EXPLAIN, and here is a
            # defensible estimate of what that would have cost" reads as
            # an asserted fact in code, not an absence.
            negotiation = negotiation.model_copy(
                update={"cost": CostRecord(tokens_saved=_estimate_skipped_tokens(deps, cfg))}
            )

        deps.incidents.record(
            "infeasible",
            cs.profile_id,
            "no meal satisfies the profile",
            {
                "total": len(recipes),
                "constraint_set": cs.model_dump(mode="json"),
                "terminal": "NEGOTIATE",
            },
        )
        return {
            "trace": ["FEASIBILITY"],
            "negotiation": negotiation.model_dump(mode="json"),
            "terminal": "NEGOTIATE",
        }

    def retrieve_node(state: PlanState) -> PlanState:
        """QUERY REWRITE task: expand the user's query into better
        retrieval terms BEFORE it reaches the lexical/dense stores —
        `rewrite_query` (retrieval.query_rewrite) itself guarantees the
        zero-token skip on empty input and the degrade-to-original on any
        failure; this node only decides what to pass to `retrieve()` and
        surfaces the result via `query_rewrite` so a caller (the API) can
        show both the original and the rewritten query, not just apply one
        silently. The FALLBACK query ("a balanced meal" for a genuinely
        empty request) is never rewritten — there is nothing there for a
        rewrite step to expand.
        """
        cs = ConstraintSet.model_validate(state["constraint_set"])
        raw_query = (state.get("query") or "").strip()
        qr = rewrite_query(raw_query, deps.llm)
        search_query = qr.rewritten if raw_query else "a balanced meal"
        found = retrieve(
            search_query,
            cs,
            deps.catalog,
            deps.llm,
            vector_store=deps.vector_store,
            top_k=cfg.retrieval.top_k,
            affinity=(deps.preferences.affinity(cs.profile_id) if deps.preferences else None),
        )
        result: PlanState = {"trace": ["RETRIEVE"], "query_rewrite": qr.model_dump(mode="json")}
        if raw_query:
            # Zero-token guarantee mirrors compile_node: a cost key is only
            # ever contributed when a call was actually attempted.
            result["cost"] = qr.cost.model_dump(mode="json")
        if not found:
            esc = Escalation(
                reason="constraint_uncheckable",
                failing_signal="retrieval",
                detail="feasibility found survivors but retrieval returned none",
            )
            result["escalation"] = esc.model_dump(mode="json")
            result["terminal"] = "ESCALATE"
            return result
        result["candidates"] = [r.id for r in found]
        return result

    def score_node(state: PlanState) -> PlanState:
        resolved = [deps.catalog.recipe(i) for i in state["candidates"]]
        missing = [i for i, r in zip(state["candidates"], resolved, strict=True) if r is None]
        if missing:
            # A candidate id retrieve_node just produced no longer resolves
            # in the catalog — a reseed racing this run, or a corrupted/
            # replayed checkpoint. Escalate with a named reason rather than
            # letting `llm_rerank`/`hydrate` crash on a `None` recipe below.
            esc = Escalation(
                reason="constraint_uncheckable",
                failing_signal="candidate_missing",
                detail=f"candidate id(s) no longer resolve in the catalog: {missing}",
            )
            return {
                "trace": ["SCORE"],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }
        candidates = [r for r in resolved if r is not None]
        # `llm_rerank` (retrieval/rerank.py) returns the `Completion.cost`
        # it actually spent as its 4th element — a COMMIT spends real
        # money on two model calls (rerank here, explain later), and
        # dropping this one used to silently understate `per_plan_usd` by
        # roughly half against a real provider. Contributed below via the
        # `"cost"` key, same as every other cost-spending node — the
        # `merge_cost` reducer (agent.state) sums it into `PlanState.cost`.
        chosen, _rationale, self_assessment, rerank_cost = llm_rerank(
            state.get("query", ""), candidates, deps.llm
        )
        if chosen is None:
            # Defensive only: retrieve_node already guarantees `candidates`
            # is non-empty, and llm_rerank only returns None for an empty
            # list — this branch should be unreachable, but a routing edge
            # exists for it rather than letting `trust` crash on missing
            # `check`/`nutrition` if it ever is.
            esc = Escalation(
                reason="constraint_uncheckable",
                failing_signal="rerank",
                detail="retrieval produced candidates but none survived reranking",
            )
            return {
                "trace": ["SCORE"],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }

        chosen = deps.catalog.hydrate(chosen)
        if chosen.nutrition is None:
            # hydrate() leaves nutrition unset only when the catalog has no
            # payload for this recipe id — an internal catalog
            # inconsistency, not a user-facing constraint failure. Escalate
            # by name rather than crashing on `.model_dump()` below.
            esc = Escalation(
                reason="constraint_uncheckable",
                failing_signal="nutrition_unavailable",
                detail=f"catalog has no payload to compute nutrition for '{chosen.id}'",
            )
            return {
                "trace": ["SCORE"],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }
        cs = ConstraintSet.model_validate(state["constraint_set"])
        check = check_recipe(chosen, cs)
        return {
            "trace": ["SCORE"],
            "chosen_id": chosen.id,
            "nutrition": chosen.nutrition.model_dump(mode="json"),
            "check": check.model_dump(mode="json"),
            "self_assessment": self_assessment,
            "cost": rerank_cost.model_dump(mode="json"),
        }

    def trust_node(state: PlanState) -> PlanState:
        check = CheckResult.model_validate(state["check"])
        nutrition = NutritionFacts.model_validate(state["nutrition"])
        cs = ConstraintSet.model_validate(state["constraint_set"])
        report = score(nutrition, check, cs, state.get("self_assessment"))
        escalation = gate(report, cfg)
        if escalation is not None:
            return {
                "trace": ["TRUST"],
                "trust": report.model_dump(mode="json"),
                "escalation": escalation.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }

        # Grey band on a MEDICAL profile is where a human belongs — trust
        # cleared the bar but not by much, and the constraint at stake is
        # not one where "probably fine" is an acceptable answer. Any other
        # profile in the same grey band auto-resumes: an agent that pauses
        # on every request is not a safety gate, it's a broken product.
        # Spec §6, §9.
        grey = report.composite < (cfg.trust.refusal_threshold + cfg.trust.medical_review_band)
        medical = any(c.severity == Severity.MEDICAL for c in cs.hard())
        return {
            "trace": ["TRUST"],
            "trust": report.model_dump(mode="json"),
            "needs_approval": bool(grey and medical),
        }

    def explain_node(state: PlanState) -> PlanState:
        n = NutritionFacts.model_validate(state["nutrition"])
        check = CheckResult.model_validate(state["check"])
        recipe = _require_recipe(deps.catalog, state["chosen_id"])

        if cfg.async_explanation and deps.explanation_queue is not None:
            # Task 23: the recommendation is already fully determined —
            # everything COMMIT needs (recipe, nutrition, trust, the
            # constraint check) exists before this branch runs. Prose is
            # the only thing left, so it leaves the request path entirely:
            # submit and return immediately with no completion, no cost
            # spent HERE (the queue's own worker accounts for it once the
            # job actually finishes — see `ExplanationQueue.cost`).
            # `state["thread_id"]` is the same id the API hands back to a
            # caller, so `GET /recommend/{id}/explanation` can look this
            # exact job up later.
            # Render HERE, with `_describe_satisfied`, and hand the finished
            # prompt over. `ExplanationQueue._render_prompt` has its own
            # renderer whose docstring claims to produce "the exact prompt
            # explain_node renders synchronously" — it did not: it joined the
            # raw `check.satisfied` IDS. So a diner on the async path read
            # "aligning with constraints c0, c1, and c2", the exact internal
            # -identifier leak `_describe_satisfied` exists to prevent, while
            # the sync path was clean and the docstring asserted they matched.
            #
            # Two renderers that must agree will eventually disagree. The
            # queue cannot describe constraints itself — it is handed a
            # `CheckResult` (ids only), never the `ConstraintSet` — so the
            # only place the description CAN be built is here.
            deps.explanation_queue.submit(
                state["thread_id"],
                recipe=recipe,
                nutrition=n,
                check=check,
                prompt=load_prompt("explain").render(
                    name=recipe.name,
                    facts=", ".join(
                        f"{f}={getattr(n, f)}"
                        for f in ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")
                    ),
                    satisfied=_describe_satisfied(state.get("constraint_set"), check.satisfied),
                ),
            )
            return {"trace": ["EXPLAIN"], "explanation": "", "cost": {}}

        facts = ", ".join(
            f"{f}={getattr(n, f)}"
            for f in ("kcal", "protein_g", "carbs_g", "fat_g", "sodium_mg", "fibre_g")
        )
        explain_prompt = load_prompt("explain")
        completion = deps.llm.complete(
            explain_prompt.render(
                name=recipe.name,
                facts=facts,
                satisfied=_describe_satisfied(state.get("constraint_set"), check.satisfied),
            ),
            stage="explain",
            prompt_ref=explain_prompt,
        )
        return {
            "trace": ["EXPLAIN"],
            "explanation": completion.text,
            "cost": completion.cost.model_dump(mode="json"),
        }

    def verify_node(state: PlanState) -> PlanState:
        """The model never gets the last word. Re-checks the chosen recipe
        against the same ConstraintSet (no model involved), and diffs every
        number the explanation stated against catalog truth. Spec §9.

        Task 23: with `async_explanation` on, `state["explanation"]` is
        `""` here — the prose has not been generated yet, only queued. An
        empty string states no numbers, so `detect_drift` would trivially
        find nothing either way; skipping it when there is genuinely
        nothing to check makes that explicit rather than relying on the
        regex incidentally agreeing. Nothing else about VERIFY changes:
        the constraint recheck (the part that never depended on the
        model) still runs unconditionally.
        """
        cs = ConstraintSet.model_validate(state["constraint_set"])
        recipe = deps.catalog.hydrate(_require_recipe(deps.catalog, state["chosen_id"]))
        recheck = check_recipe(recipe, cs)
        nutrition = NutritionFacts.model_validate(state["nutrition"])
        tolerance = load_thresholds().verifiers.nutrition_drift_pct
        explanation = state["explanation"]
        drift = detect_drift(explanation, nutrition, tolerance=tolerance) if explanation else []

        if recheck.violated or drift:
            for f in drift:
                deps.incidents.record(
                    "drift",
                    cs.profile_id,
                    f"{f.field} stated {f.stated} vs computed {f.computed}",
                    {
                        "delta_pct": f.delta_pct,
                        "constraint_set": cs.model_dump(mode="json"),
                        "terminal": "ESCALATE",
                    },
                )
            trust = TrustReport.model_validate(state["trust"]) if state.get("trust") else None
            # Tokens were genuinely spent getting here (at minimum, EXPLAIN
            # ran) — an escalation that reports 0.0 here would be exactly
            # the same "invites the wrong question" failure this fix round
            # exists to close for COMMIT. Whatever `PlanState.cost` has
            # accumulated so far rides along on the Escalation itself.
            spent = CostRecord.model_validate(state.get("cost") or {})
            esc = Escalation(
                reason="verification_failed",
                failing_signal="drift" if drift else "constraint_recheck",
                trust=trust,
                cost=spent,
                detail=(
                    f"post-generation verification failed: {len(drift)} drift "
                    f"finding(s), {len(recheck.violated)} constraint violation(s)"
                ),
            )
            return {
                "trace": ["VERIFY"],
                "escalation": esc.model_dump(mode="json"),
                "terminal": "ESCALATE",
            }
        return {"trace": ["VERIFY"]}

    def commit_node(state: PlanState) -> PlanState:
        recipe = _require_recipe(deps.catalog, state["chosen_id"])
        check = CheckResult.model_validate(state["check"])
        cs = ConstraintSet.model_validate(state["constraint_set"])
        rec = Recommendation(
            recipe_id=recipe.id,
            recipe_name=recipe.name,
            nutrition=NutritionFacts.model_validate(state["nutrition"]),
            trust=TrustReport.model_validate(state["trust"]),
            explanation=state["explanation"],
            constraints_satisfied=check.satisfied,
            constraints_satisfied_display=_satisfied_display(
                state.get("constraint_set"), check.satisfied
            ),
            ingredients=_ingredient_lines(deps.catalog, recipe),
            skill_versions=skill_versions(deps),
            cost=CostRecord.model_validate(state.get("cost") or {}),
        )
        aid = deps.audit.record(
            cs.profile_id, "COMMIT", rec.model_dump(mode="json"), skill_versions(deps), rec.cost.usd
        )
        return {
            "trace": ["COMMIT"],
            "recommendation": rec.model_dump(mode="json"),
            "terminal": "COMMIT",
            "audit_id": aid,
        }

    def negotiate_node(state: PlanState) -> PlanState:
        cs = ConstraintSet.model_validate(state["constraint_set"])
        aid = deps.audit.record(
            cs.profile_id, "NEGOTIATE", state["negotiation"], skill_versions(deps), 0.0
        )
        return {"trace": ["NEGOTIATE"], "terminal": "NEGOTIATE", "audit_id": aid}

    def escalate_node(state: PlanState) -> PlanState:
        esc = Escalation.model_validate(state["escalation"])
        cs = ConstraintSet.model_validate(state["constraint_set"])
        deps.incidents.record(
            "escalation",
            cs.profile_id,
            esc.detail,
            {
                "reason": esc.reason,
                "failing_signal": esc.failing_signal,
                "constraint_set": cs.model_dump(mode="json"),
                "terminal": "ESCALATE",
            },
        )
        aid = deps.audit.record(
            cs.profile_id,
            "ESCALATE",
            esc.model_dump(mode="json"),
            skill_versions(deps),
            esc.cost.usd,
        )
        return {"trace": ["ESCALATE"], "terminal": "ESCALATE", "audit_id": aid}

    raw = {
        "compile": compile_node,
        "feasibility": feasibility,
        "retrieve": retrieve_node,
        "score": score_node,
        "trust": trust_node,
        "explain": explain_node,
        "verify": verify_node,
        "commit": commit_node,
        "negotiate": negotiate_node,
        "escalate": escalate_node,
    }
    return {
        stage: (
            _guarded_terminal(stage, deps, fn) if stage in _TERMINAL_STAGES else _guarded(stage, fn)
        )
        for stage, fn in raw.items()
    }

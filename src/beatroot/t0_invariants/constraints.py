"""T0 invariant layer: deterministic hard-constraint enforcement.

Constraint evaluation is POLICY AS DATA, not control flow. Each evaluator is a
pure function registered against a constraint `kind` via `@evaluator(...)`.
Adding a new kind means registering a function here — never editing a branch
of an `if`/`match` chain. Spec §3, §2 (guarded failure mode).

The registration is the SINGLE source of truth for a kind's whole vocabulary,
not just its evaluation: `@evaluator(kind, shape=..., parse=...)` also
declares the one-line, model-facing description of the value shape
(`kind_shapes()`, rendered into `prompts/compile_constraints.md`) and the
function that turns a raw model value into a validated `ConstraintValue`
(`get_parser()`, used by `agent.nodes._parse_compiled_constraint`). Both
`shape` and `parse` are required keyword arguments — a kind registered
without either fails immediately at import time, so "advertised but silently
dropped" (the `exclude_cuisine` incident) is impossible by construction
rather than merely tested against.

The one thing NOT registration-driven, deliberately: the severity ratchet
(model-proposed category -> `Severity`, and the DIETARY floor for identity
kinds) stays hardcoded in `agent.nodes` — an evaluator declares how to check
and how to parse a kind, never how "hard" it gets to consider itself.

This module (and every module under t0_invariants/) must NEVER import
`beatroot.reasoning`, `beatroot.agent`, or a catalog LOADER (something that
reads `data/*` off disk). Task 5 adds a test that enforces the `reasoning`
boundary at the import-graph level; do not weaken that test to make anything
here pass. Vocabulary a parser needs (known tags, known cuisines) is handed
in as plain data via `ConstraintVocabulary`, never fetched by importing a
loader.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from beatroot.contracts.core import Constraint, ConstraintSet, ConstraintValue
from beatroot.trusted.canonical import resolve_ingredient_id
from beatroot.trusted.catalog import Recipe


@dataclass(frozen=True)
class ConstraintVocabulary:
    """The catalog-derived vocabulary a parse function may need — plain data,
    handed in by the caller (`agent.nodes`), never fetched by this module
    importing a catalog loader. Keeps the t0/agent boundary intact while
    still letting a parser validate a tag or cuisine name for real."""

    known_tags: frozenset[str]
    known_cuisines: frozenset[str]


# A parse function turns one raw model item (the dict the model emitted for
# one constraint, e.g. {"kind": ..., "value": ..., "nutrient": ...}) into a
# validated (value, nutrient) pair, or None if nothing about it can be
# trusted. It receives the whole item (not just `value`) because exactly one
# kind (`nutrient_range`) needs a second field (`nutrient`) alongside value —
# giving every parser the same signature keeps the registry uniform rather
# than special-casing that one kind's call shape.
ParsedValue = tuple[ConstraintValue, str | None]
ParseFn = Callable[[dict[str, object], ConstraintVocabulary], ParsedValue | None]


class CheckResult(BaseModel):
    """A validated model, not a bare dataclass — this crosses into LangGraph
    state (Task 11) via SqliteSaver, so it must serialize/deserialize cleanly
    instead of round-tripping through `__dict__` and losing type validation."""

    ok: bool
    violated: list[str] = Field(default_factory=list)
    uncheckable: list[str] = Field(default_factory=list)
    satisfied: list[str] = Field(default_factory=list)


Outcome = Literal["satisfied", "violated", "uncheckable"]

_Evaluator = Callable[[Recipe, Constraint], Outcome]

# Policy as DATA, not as control flow. Each evaluator is a pure function
# registered against a constraint kind; adding a kind means registering a
# function, never editing a branch in this module. Spec §3.
#
# `_REGISTRY` keeps its original shape (kind -> bare evaluator function) on
# purpose: several existing tests reach in and monkeypatch a single entry
# directly (`monkeypatch.setitem(_REGISTRY, "exclude_tag", ...)`), and that
# must keep working unchanged. The two new dicts below carry the rest of the
# vocabulary a kind now declares; all three are always populated together by
# `evaluator(...)`, never independently.
_REGISTRY: dict[str, _Evaluator] = {}
_SHAPES: dict[str, str] = {}
_PARSERS: dict[str, ParseFn] = {}


def evaluator(kind: str, *, shape: str, parse: ParseFn) -> Callable[[_Evaluator], _Evaluator]:
    """Register everything this kind is, in one call: `fn` is how to
    EVALUATE it against a recipe; `shape` is the one-line, model-facing
    description of its value shape (`kind_shapes()`); `parse` is how to
    validate/parse a raw value the model proposed for it
    (`get_parser()`). Both `shape` and `parse` are required — there is no
    default that would let a kind register with only an evaluator and
    silently have no shape text or no parser, which is exactly how
    `exclude_cuisine` got advertised to the model and then dropped on the
    floor.
    """

    def register(fn: _Evaluator) -> _Evaluator:
        _REGISTRY[kind] = fn
        _SHAPES[kind] = shape
        _PARSERS[kind] = parse
        return fn

    return register


def _as_number(value: object) -> float | None:
    """`Constraint.value` is `str | float | tuple[float, float] | list[str]` —
    a constraint author can put anything shaped like that value in. A
    scalar-numeric evaluator (`max_prep_minutes`, `budget_max`) must not
    hand a tuple or list to `float()` and crash; it degrades to
    `uncheckable` instead, exactly like a nutrient name the catalog does
    not have. `bool` is excluded even though it is an `int` subclass — a
    constraint value is never meant to carry a `True`/`False`.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _parse_single_known_tag(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """Shared by `exclude_tag`/`require_tag`: value is ONE tag, and it must
    come from the live catalog tag vocabulary — never invented, never a tag
    the catalog has never heard of."""
    raw = item.get("value")
    if not isinstance(raw, str) or raw not in vocab.known_tags:
        return None
    return raw, None


@evaluator(
    "exclude_tag",
    shape="ONE catalog tag the recipe must not carry.",
    parse=_parse_single_known_tag,
)
def _exclude_tag(recipe: Recipe, c: Constraint) -> Outcome:
    """No canonicalisation here — unlike `_exclude_ingredient` below, tags
    have no synonym layer at all: `Recipe.tags` are catalog-canonical tag
    strings from the moment `trusted.tags.derive_recipe_tags` writes them
    (from each ingredient's `allergen_tags`/`religious_tags`/
    `dietary_tags`), so a real `exclude_tag` constraint value is already
    the thing being compared against — no resolution step is missing.

    The `str` guard IS the fix this evaluator needed: a non-`str` value
    (`Constraint.value` also allows `float`/`tuple[float, float]`/
    `list[str]`) has nothing meaningful to compare against a `set[str]` of
    tags. Before this guard, `c.value in recipe.tags` degraded a malformed
    value to `"satisfied"` for any non-str scalar, and would raise
    `TypeError: unhashable type` outright for a `list` value — either way
    exactly the asymmetry `_exclude_ingredient` was already careful to
    avoid: a constraint that cannot be evaluated must read as
    `"uncheckable"`, never `"satisfied"`, and never crash the whole check.
    """
    if not isinstance(c.value, str):
        return "uncheckable"
    return "violated" if c.value in recipe.tags else "satisfied"


@evaluator(
    "require_tag",
    shape='ONE tag the recipe must carry — a strict identity (e.g. "must be vegan").',
    parse=_parse_single_known_tag,
)
def _require_tag(recipe: Recipe, c: Constraint) -> Outcome:
    """The ALLOWLIST primitive, and the absence of it was a real safety bug.

    Every other evaluator here says what is forbidden. Nothing could say what
    is REQUIRED, so a categorical dietary identity ("this user is vegan") had
    to be approximated as a denylist of animal-derived tags — and a denylist
    over an open world is incomplete by construction. This catalog carries no
    `chicken`/`poultry` tag, so `Chicken satay` (tags: peanut) matched none of
    the vegan preset's seven exclusions and was served to a vegan user. The
    dishes that WERE blocked were blocked incidentally, for containing dairy.

    A positive requirement inverts the failure direction, which is the whole
    point: an unknown or newly added dish is excluded until it is positively
    tagged vegan, rather than admitted until someone remembers to add a tag
    for it. Denylists fail open; allowlists fail closed. For a dietary-safety
    boundary the second is the only defensible default.

    Same "never guess" posture as its siblings: a non-`str` value cannot be
    compared against a `set[str]` and reads as `uncheckable`, never
    `satisfied`.
    """
    if not isinstance(c.value, str):
        return "uncheckable"
    return "satisfied" if c.value in recipe.tags else "violated"


def _parse_any_known_tag(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """Value is a LIST of tags; every entry not in the live catalog
    vocabulary is dropped rather than failing the whole constraint — same
    "give back what IS real" posture as `_parse_exclude_cuisine` below. An
    empty result after filtering is `None`: a disjunction over nothing is
    malformed input, not a constraint permitting everything."""
    raw = item.get("value")
    if not isinstance(raw, list):
        return None
    tags = [v for v in raw if isinstance(v, str) and v in vocab.known_tags]
    if not tags:
        return None
    return tags, None


@evaluator(
    "require_any_tag",
    shape=(
        "a LIST of tags; the recipe must carry AT LEAST ONE — a disjunctive "
        "identity (e.g. vegetarian OR fish, for pescetarian). Use this whenever "
        "a single required tag would misrepresent the identity."
    ),
    parse=_parse_any_known_tag,
)
def _require_any_tag(recipe: Recipe, c: Constraint) -> Outcome:
    """Disjunction: satisfied when the recipe carries AT LEAST ONE of the
    listed tags. `Constraint.value` already admitted `list[str]`; nothing
    consumed it as a set of alternatives until now.

    This exists because some real dietary identities are genuinely a union,
    and forcing them into a single `require_tag` misrepresents them. A
    pescetarian eats vegetarian food OR fish — in this catalog those sets are
    strictly disjoint (0 recipes carry both tags), so no single required tag
    can express it. The previous encoding fell back to excluding beef and
    red_meat at PREFERENCE severity, which admitted chicken.

    Deliberately NOT a generic boolean expression language. `require_tag` for
    "must be X" and `require_any_tag` for "must be one of X, Y" cover the
    dietary identities this catalog can express, and each stays a pure
    O(len(value)) set membership test that a reader can verify by reading.
    An arbitrary AND/OR/NOT tree over tags would be more expressive and far
    harder to audit, and this is the layer where auditability outranks
    expressiveness.

    An empty list is `uncheckable`, never `satisfied`: a requirement that
    permits nothing is malformed input, and silently passing it would admit
    every recipe — the exact fail-open direction this whole primitive exists
    to close.
    """
    if not isinstance(c.value, list) or not c.value:
        return "uncheckable"
    if not all(isinstance(v, str) for v in c.value):
        return "uncheckable"
    return "satisfied" if recipe.tags & set(c.value) else "violated"


def _parse_ingredient_name(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """Value is free text, deliberately NOT checked against a vocabulary
    here — `t0_invariants.vocabulary.unknown_vocabulary` already resolves it
    against the real ingredient/synonym table at FEASIBILITY and escalates
    if it names nothing real; this only rejects the empty/non-string case."""
    raw = item.get("value")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip(), None


@evaluator(
    "exclude_ingredient",
    shape=(
        "ONE ingredient, as its own specific common name (e.g. a particular oil, "
        "spice, or dish component) — a synonym is fine, it is resolved against the "
        "catalog separately, but a bare category word that could mean several "
        "distinct ingredients may fail to resolve at all. Prefer the most specific "
        "name the user's own words support."
    ),
    parse=_parse_ingredient_name,
)
def _exclude_ingredient(recipe: Recipe, c: Constraint) -> Outcome:
    """`c.value` is user-facing text — often a synonym, like "groundnut
    oil", the common Indian name recorded against `ing_peanut_oil` in
    `data/ingredients.yaml` — while `recipe.ingredient_ids` holds catalog
    CANONICAL ids. Comparing the two literally is the bug this evaluator
    shipped with: `unknown_vocabulary` canonicalises "groundnut oil" to
    validate it as a real, known ingredient, then this evaluator compared
    the raw string and never matched, so the checked-in profile was
    accepted and then never enforced. Resolve `c.value` to its canonical
    id FIRST, via the SAME resolution `unknown_vocabulary` already used
    to validate this value (`trusted.canonical.resolve_ingredient_id`),
    then compare ids to ids — never raw text to ids.

    A value that resolves to nothing is `uncheckable`, never `satisfied`:
    an ingredient exclusion `unknown_vocabulary` is meant to catch
    upstream must not be silently waved through here if it ever gets past
    that check anyway — same "never guess, never silently pass" posture
    as every other evaluator in this module. A non-`str` value is the
    identical case (nothing to resolve) and takes the same path.
    """
    if not isinstance(c.value, str):
        return "uncheckable"
    resolved = resolve_ingredient_id(c.value)
    if resolved is None:
        return "uncheckable"
    return "violated" if resolved in recipe.ingredient_ids else "satisfied"


def _parse_number(item: dict[str, object], vocab: ConstraintVocabulary) -> ParsedValue | None:
    """Shared by `max_prep_minutes`/`budget_max`: value is a single number.
    `bool` is excluded even though it is an `int` subclass — same posture as
    `_as_number` above."""
    raw = item.get("value")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw), None


@evaluator(
    "max_prep_minutes",
    shape="a single number: the maximum prep time in minutes.",
    parse=_parse_number,
)
def _max_prep(recipe: Recipe, c: Constraint) -> Outcome:
    limit = _as_number(c.value)
    if recipe.prep_minutes is None or limit is None:
        return "uncheckable"
    return "violated" if recipe.prep_minutes > limit else "satisfied"


def _parse_nutrient_range(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """The one kind whose parsed value needs a SECOND field alongside
    `value` — `nutrient` names which `NutritionFacts` field the [low, high]
    pair constrains. Both must be present and well-shaped or the whole
    constraint is dropped; never guessed as protein."""
    raw_nutrient = item.get("nutrient")
    if not isinstance(raw_nutrient, str) or not raw_nutrient.strip():
        return None
    raw_value = item.get("value")
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        return None
    lo, hi = raw_value
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in (lo, hi)):
        return None
    return (float(lo), float(hi)), raw_nutrient.strip()


@evaluator(
    "nutrient_range",
    shape="[low, high]; `nutrient` names which NutritionFacts field it constrains.",
    parse=_parse_nutrient_range,
)
def _nutrient_range(recipe: Recipe, c: Constraint) -> Outcome:
    """Never guesses which nutrient is meant. A constraint that doesn't name
    a nutrient, or names one `NutritionFacts` doesn't have, is uncheckable —
    not silently checked against protein."""
    if recipe.nutrition is None or c.nutrient is None:
        return "uncheckable"
    if not isinstance(c.value, tuple) or len(c.value) != 2:
        # A nutrient_range constraint whose value is not a (lo, hi) pair is
        # malformed input, not a crash — same "never guess" posture as an
        # unknown nutrient name.
        return "uncheckable"
    lo, hi = c.value
    value = getattr(recipe.nutrition, c.nutrient, None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        # unknown nutrient name, or a real-but-non-numeric field (e.g.
        # "provenance") — never guess, never crash, never pass.
        return "uncheckable"
    return "satisfied" if lo <= value <= hi else "violated"


@evaluator(
    "budget_max",
    shape="a single number: the maximum cost, in the catalog's currency units.",
    parse=_parse_number,
)
def _budget_max(recipe: Recipe, c: Constraint) -> Outcome:
    limit = _as_number(c.value)
    if recipe.cost_inr is None or limit is None:
        return "uncheckable"
    return "violated" if recipe.cost_inr > limit else "satisfied"


def _parse_single_known_cuisine(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """Value is ONE cuisine, and it must come from the live catalog cuisine
    vocabulary — never invented, never a cuisine the catalog has never
    heard of."""
    raw = item.get("value")
    if not isinstance(raw, str) or raw not in vocab.known_cuisines:
        return None
    return raw, None


@evaluator(
    "cuisine_affinity",
    shape=(
        "ONE cuisine this diner favours or wants to avoid — a ranking preference, "
        "never a hard filter. This means ATTRACTION toward a cuisine, the OPPOSITE "
        "of avoidance — use `exclude_cuisine` to express a dislike, never this."
    ),
    parse=_parse_single_known_cuisine,
)
def _cuisine_affinity(recipe: Recipe, c: Constraint) -> Outcome:
    return "satisfied"  # a ranking signal, never an enforcement gate


def _parse_exclude_cuisine(
    item: dict[str, object], vocab: ConstraintVocabulary
) -> ParsedValue | None:
    """Accepts one cuisine or a list of them, so a whole culinary region
    ("western food") is one constraint. Unknown cuisines are dropped from a
    list rather than failing the whole constraint — a model naming four
    cuisines of which three exist should still get those three, since the
    alternative is silently honouring none of them. A single unknown
    cuisine, or an empty list after filtering, is `None`."""
    raw = item.get("value")
    if isinstance(raw, str):
        if raw not in vocab.known_cuisines:
            return None
        return raw, None
    if isinstance(raw, list):
        cuisines = [v for v in raw if isinstance(v, str) and v in vocab.known_cuisines]
        if not cuisines:
            return None
        return cuisines, None
    return None


@evaluator(
    "exclude_cuisine",
    shape=(
        "ONE cuisine name, or a LIST of them, that the recipe's cuisine must not be. "
        "Use this for any stated dislike of a cuisine or of a culinary region — name "
        "the specific cuisines from the vocabulary you were given that make up that "
        "region."
    ),
    parse=_parse_exclude_cuisine,
)
def _exclude_cuisine(recipe: Recipe, c: Constraint) -> Outcome:
    """Cuisine AVOIDANCE, which had no representation at all until now.

    `cuisine_affinity` only ever expresses attraction. Asked for "dislikes
    western food", the compiler had no kind meaning "not these cuisines", so
    it emitted `cuisine_affinity: continental` — an affinity TOWARD a western
    cuisine, exactly inverting the request, and the diner was served Aglio e
    olio. A missing primitive does not make a model abstain; it makes it pick
    the nearest available shape, and the nearest shape here meant the
    opposite thing.

    `recipe.cuisine` is a scalar field, not a tag, so `exclude_tag` could
    never have covered this. A `list[str]` value lets one constraint carry a
    whole culinary region ("western" = italian/mediterranean/mexican/
    continental) without the catalog needing a region concept, and without
    this module hardcoding which cuisines count as western — the model names
    them from the live cuisine vocabulary it is handed.
    """
    if isinstance(c.value, str):
        return "violated" if recipe.cuisine == c.value else "satisfied"
    if isinstance(c.value, list) and c.value and all(isinstance(v, str) for v in c.value):
        return "violated" if recipe.cuisine in set(c.value) else "satisfied"
    return "uncheckable"


def registered_kinds() -> list[str]:
    """Every constraint kind this T0 layer actually knows how to enforce,
    sorted. The single source of truth `agent.nodes.compile_node` reads at
    runtime to tell the model what vocabulary it may emit — never a
    separately maintained list that can drift from what `_evaluate` actually
    dispatches on. Registering a new evaluator here is what it takes to add
    a kind the model is allowed to propose; nothing else needs editing."""
    return sorted(_REGISTRY)


def kind_shapes() -> dict[str, str]:
    """Maps each registered kind to its one-line model-facing shape
    description, from the same registration call (`@evaluator(...,
    shape=...)`) that declared its evaluator. `prompts/compile_constraints
    .md`'s KIND SHAPES section is rendered from this, via
    `agent.nodes.compile_node` — never a separately maintained prose list
    that can drift from what a kind's parser actually accepts."""
    return dict(_SHAPES)


def get_parser(kind: str) -> ParseFn | None:
    """The parse function registered for `kind`, or `None` if `kind` isn't
    registered at all. `evaluator(...)` requires `parse=` on every
    registration, so any registered kind is guaranteed to have one — `None`
    here only ever means "not a real kind", never "registered but nobody
    wrote a parser for it"."""
    return _PARSERS.get(kind)


def _evaluate(recipe: Recipe, c: Constraint) -> Outcome:
    """Dispatch through the registry. No LLM, ever — enforced by
    tests/test_boundaries.py.

    An UNREGISTERED kind returns 'uncheckable', never 'satisfied'. Silently
    passing a constraint nobody knows how to check is exactly the failure this
    project exists to prevent.
    """
    fn = _REGISTRY.get(c.kind)
    if fn is None:
        return "uncheckable"
    return fn(recipe, c)


def check_recipe(recipe: Recipe, cs: ConstraintSet) -> CheckResult:
    buckets: dict[str, list[str]] = {"satisfied": [], "violated": [], "uncheckable": []}
    for c in cs.constraints:
        outcome = _evaluate(recipe, c)
        buckets[outcome].append(c.id)
    return CheckResult(
        ok=not buckets["violated"],
        violated=buckets["violated"],
        uncheckable=buckets["uncheckable"],
        satisfied=buckets["satisfied"],
    )


def is_legal(recipe: Recipe, cs: ConstraintSet) -> bool:
    """Hard constraints only. This is the gate retrieval filters on.

    An uncheckable HARD constraint makes the recipe illegal — we do not serve
    a meal we cannot prove safe. Spec §2.
    """
    return all(_evaluate(recipe, c) == "satisfied" for c in cs.hard())

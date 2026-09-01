"""Preset dietary identities must actually exclude what they name. Regression.

Reported from the running app: a user picked the **Vegan** preset, asked for
chicken, and got it.

The investigation found this was not a failure of the invariant engine — that
did exactly what it is specified to do. It was a failure of how a dietary
identity was *expressed*, and it had two independent causes that compounded:

1. **The vocabulary had no allowlist primitive.** `t0_invariants.constraints`
   registered only `exclude_tag` / `exclude_ingredient` — ways to say what is
   forbidden, never a way to say what is REQUIRED. So "vegan" had to be
   approximated as a denylist of animal-derived tags, and a denylist over an
   open world is incomplete by construction. This catalog carries no
   `chicken`/`poultry` tag at all, so `Chicken hakka noodles` (gluten,
   root_vegetable, sesame, soy) and `Chicken satay` (peanut) matched NONE of
   the seven exclusions and passed cleanly. `Butter chicken` was blocked, but
   only incidentally, for containing `dairy` — safety by coincidence.

2. **Severity.** Every one of those constraints was `preference`, and
   `HARD_SEVERITIES` is `{medical, religious}`, so `is_legal()` never enforced
   them — soft constraints are ranking's job, not filtering's, by design.

The comparison that makes the inconsistency plain: `jain` expressed its
identity at `religious` severity and was enforced, so a Jain user never saw
root vegetables — while a vegan user saw chicken. Same class of categorical
dietary rule, opposite outcomes, for no principled reason.

These tests are written against the SHIPPED preset data, not a fixture, because
the bug lived in that data plus the missing primitive. A fixture would have
passed throughout.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from beatroot.container import build_container
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.t0_invariants.constraints import is_legal

PROFILES = Path(__file__).resolve().parents[1] / "data" / "profiles.yaml"


def _profiles() -> list[dict[str, Any]]:
    return yaml.safe_load(PROFILES.read_text()) or []


def _profile(pid: str) -> dict[str, Any]:
    for p in _profiles():
        if p["id"] == pid:
            return p
    raise AssertionError(f"preset profile {pid!r} not found in {PROFILES}")


def _cs(pid: str) -> ConstraintSet:
    p = _profile(pid)
    return ConstraintSet(
        profile_id=pid, constraints=[Constraint(**c) for c in p.get("constraints", [])]
    )


@pytest.fixture(scope="module")
def recipes() -> list:
    catalog = build_container().catalog
    return [catalog.hydrate(r) if hasattr(catalog, "hydrate") else r for r in catalog.recipes()]


def _animal_dishes(recipes: list) -> list:
    """Recipes that are unambiguously not vegan, by name."""
    meat_words = ("chicken", "mutton", "fish", "prawn", "egg", "keema", "beef", "lamb")
    return [r for r in recipes if any(w in r.name.lower() for w in meat_words)]


def test_catalog_actually_has_a_vegan_tag() -> None:
    """The vegan preset's own description asserted 'the catalog has no direct
    "vegan" filter, so veganism is expressed as excluding every animal-derived
    tag it does track'. That premise was false, and the false premise is what
    justified the denylist. Pin the truth so it cannot be re-assumed."""
    catalog = build_container().catalog
    tags: set[str] = set()
    for r in catalog.recipes():
        h = catalog.hydrate(r) if hasattr(catalog, "hydrate") else r
        tags |= set(h.tags)
    assert "vegan" in tags
    assert "vegetarian" in tags


@pytest.mark.parametrize(
    "pid",
    ["vegan", "eggetarian", "vegetarian_lactose_intolerant", "pescetarian_peanut_allergy"],
)
def test_dietary_identity_is_enforced_not_merely_preferred(pid: str) -> None:
    """A categorical dietary identity must carry at least one HARD constraint.

    Without one, `is_legal()` lets every dish through and the identity is
    decoration — which is exactly how a vegan was served chicken.
    """
    cs = _cs(pid)
    assert cs.hard(), (
        f"preset {pid!r} has no hard constraint, so is_legal() enforces nothing "
        f"for it; severities present: {sorted({str(c.severity) for c in cs.constraints})}"
    )


def test_vegan_profile_rejects_every_meat_dish_in_the_catalog(recipes: list) -> None:
    """The headline regression: no dish with meat in its name may be legal for
    the vegan preset. Chicken hakka noodles and Chicken satay were the two that
    slipped through the denylist entirely."""
    cs = _cs("vegan")
    served = [r.name for r in _animal_dishes(recipes) if is_legal(r, cs)]
    assert served == [], f"vegan preset admits animal dishes: {served}"


def test_vegetarian_identity_rejects_meat(recipes: list) -> None:
    """`vegetarian_lactose_intolerant` enforced only the LACTOSE half. The
    vegetarian half was preference-severity, so meat passed."""
    cs = _cs("vegetarian_lactose_intolerant")
    meat = [
        r.name
        for r in recipes
        if any(w in r.name.lower() for w in ("chicken", "mutton", "fish", "prawn", "keema"))
        and is_legal(r, cs)
    ]
    assert meat == [], f"vegetarian preset admits meat: {meat}"


def test_hindu_vegetarian_rejects_chicken_not_only_beef(recipes: list) -> None:
    """`hindu_vegetarian_no_beef` locked beef at `religious` severity but left
    the vegetarian part soft — so chicken passed while beef did not. Partial
    enforcement of a single identity is the same bug in a smaller costume."""
    cs = _cs("hindu_vegetarian_no_beef")
    chicken = [r.name for r in recipes if "chicken" in r.name.lower() and is_legal(r, cs)]
    assert chicken == [], f"hindu vegetarian preset admits chicken: {chicken}"


# --- egg, disjunction, and the Hindu non-vegetarian mirror ------------------
#
# Indian usage: "vegetarian" excludes egg. The catalog's `vegetarian` tag does
# NOT — `Mexican corn salad` and `Vegetable sandwich` carry both `vegetarian`
# and `egg` — so every vegetarian identity states the egg exclusion
# explicitly. `eggetarian` is the one profile that deliberately permits it,
# and that contrast is the point of having both.

EGG_FREE = [
    "vegan",
    "jain",
    "vegetarian_lactose_intolerant",
    "hindu_vegetarian_no_beef",
    "pescetarian_peanut_allergy",
]


@pytest.mark.parametrize("pid", EGG_FREE)
def test_vegetarian_identities_exclude_egg(pid: str, recipes: list) -> None:
    cs = _cs(pid)
    served = [r.name for r in recipes if "egg" in r.tags and is_legal(r, cs)]
    assert served == [], f"{pid} admits egg dishes: {served}"


def test_eggetarian_still_permits_egg(recipes: list) -> None:
    """The control. If this ever goes to zero the egg exclusions have been
    applied too broadly, and `eggetarian` has stopped meaning anything."""
    cs = _cs("eggetarian")
    assert [r.name for r in recipes if "egg" in r.tags and is_legal(r, cs)]


def test_pescetarian_is_a_real_disjunction(recipes: list) -> None:
    """Pescetarian = vegetarian OR fish. Those sets are strictly disjoint in
    this catalog (no recipe carries both tags), so a single `require_tag`
    cannot express it — this is what `require_any_tag` exists for. Both arms
    must actually admit dishes, or the disjunction has silently collapsed to
    one arm."""
    cs = _cs("pescetarian_peanut_allergy")
    legal = [r for r in recipes if is_legal(r, cs)]
    assert [r for r in legal if "fish" in r.tags], "fish arm admits nothing"
    assert [r for r in legal if "vegetarian" in r.tags], "vegetarian arm admits nothing"
    meat = [
        r.name for r in legal if any(w in r.name.lower() for w in ("chicken", "mutton", "keema"))
    ]
    assert meat == [], f"pescetarian admits non-fish meat: {meat}"


def test_hindu_nonvegetarian_blocks_only_beef(recipes: list) -> None:
    """The mirror of `hindu_vegetarian_no_beef`: same religious line, opposite
    default. A hard constraint is a precise instrument — a religious rule that
    also blocked chicken would misrepresent this user as badly as one that let
    beef through, so both directions are asserted."""
    cs = _cs("hindu_nonvegetarian")
    legal = [r for r in recipes if is_legal(r, cs)]

    beef = [r.name for r in legal if {"beef", "red_meat"} & set(r.tags)]
    assert beef == [], f"hindu non-vegetarian admits beef/red meat: {beef}"

    assert [r for r in legal if "chicken" in r.name.lower()], "chicken must be permitted"
    assert [r for r in legal if "fish" in r.tags], "fish must be permitted"
    assert [r for r in legal if "egg" in r.tags], "egg must be permitted"


def test_require_any_tag_empty_list_is_uncheckable_not_satisfied() -> None:
    """A requirement permitting nothing is malformed input. Returning
    `satisfied` would admit every recipe — the fail-open direction this whole
    primitive exists to close."""
    from beatroot.t0_invariants.constraints import _REGISTRY

    catalog = build_container().catalog
    r = next(iter(catalog.recipes()))
    recipe = catalog.hydrate(r) if hasattr(catalog, "hydrate") else r
    ev = _REGISTRY["require_any_tag"]
    assert ev(recipe, Constraint(id="x", kind="require_any_tag", severity="dietary", value=[])) == (
        "uncheckable"
    )
    assert ev(
        recipe, Constraint(id="x", kind="require_any_tag", severity="dietary", value="veg")
    ) == ("uncheckable")

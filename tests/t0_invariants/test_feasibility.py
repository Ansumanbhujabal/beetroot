from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.t0_invariants.feasibility import _survivors, assess, rank_relaxations
from beatroot.trusted.catalog import Recipe
from beatroot.trusted.index import TagIndex


def _r(rid, tags, prep=20):
    return Recipe(
        id=rid, name=rid, cuisine="x", prep_minutes=prep, tags=set(tags), ingredient_ids=[]
    )


def _cs(*cs):
    return ConstraintSet(profile_id="p", constraints=list(cs))


RECIPES = [
    _r("a", {"vegan", "peanut"}),
    _r("b", {"vegan", "dairy"}),
    _r("c", {"vegetarian", "dairy"}),
    _r("d", {"vegan"}, prep=90),
]


def test_feasible_profile_returns_survivors():
    cs = _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"))
    f = assess(cs, RECIPES)
    assert f.feasible is True
    assert {r.id for r in f.surviving} == {"b", "c", "d"}
    assert f.negotiation is None


def test_infeasible_profile_returns_ranked_relaxations():
    cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"),
        Constraint(id="c2", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c3", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=30),
    )
    f = assess(cs, RECIPES)
    assert f.feasible is False
    n = f.negotiation
    assert n.surviving == 0
    assert n.total_candidates == 4
    # dropping c2 unlocks b and c; dropping c3 unlocks d — both are single drops
    ids = [tuple(r.constraint_ids) for r in n.relaxations]
    assert ("c2",) in ids and ("c3",) in ids
    # ranked by yield, descending
    assert n.relaxations == sorted(n.relaxations, key=lambda r: -r.unlocks)


def test_medical_constraints_are_never_offered_for_relaxation():
    cs = _cs(
        Constraint(id="med", kind="exclude_tag", severity=Severity.MEDICAL, value="vegan"),
        Constraint(id="rel", kind="exclude_tag", severity=Severity.RELIGIOUS, value="vegetarian"),
    )
    f = assess(cs, RECIPES)
    assert f.feasible is False
    offered = {cid for r in f.negotiation.relaxations for cid in r.constraint_ids}
    assert "med" not in offered
    assert "rel" not in offered
    assert set(f.negotiation.locked) == {"med", "rel"}


def test_pairwise_relaxation_found_when_no_single_drop_helps():
    recipes = [_r("only", {"dairy", "slow"}, prep=90)]
    cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c2", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=30),
    )
    f = assess(cs, recipes)
    assert f.feasible is False
    ids = [tuple(sorted(r.constraint_ids)) for r in f.negotiation.relaxations]
    assert ("c1", "c2") in ids


def test_max_subset_size_is_honoured_not_decorative():
    """`settings.feasibility.max_relaxation_subset_size` shipped as a field
    that NOTHING read — `feasibility.py` used its own hardcoded literal, so
    changing the config changed nothing. A config field that silently does
    nothing is worse than no field: it tells an operator they have a control
    they do not have.

    This pins the wiring behaviourally. The same profile that yields a
    pairwise relaxation at size 2 must yield none at size 1 — proving the
    parameter reaches the branch, not merely that it is accepted.
    """
    recipes = [_r("only", {"dairy", "slow"}, prep=90)]
    cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c2", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=30),
    )

    pairs_allowed = rank_relaxations(cs, recipes, None, 2)
    assert ("c1", "c2") in [tuple(sorted(r.constraint_ids)) for r in pairs_allowed]

    pairs_forbidden = rank_relaxations(cs, recipes, None, 1)
    assert pairs_forbidden == [], "max_subset_size=1 must suppress pairwise relaxations"


def test_agent_passes_the_configured_subset_size_through():
    """The wiring above is only real if the NODE passes it. Guards against a
    future edit that reverts nodes.py to calling assess()/rank_relaxations()
    without the settings value — which would restore the dead-config bug
    while leaving the test above passing."""
    import inspect

    from beatroot.agent import nodes

    src = inspect.getsource(nodes)
    assert "cfg.feasibility.max_relaxation_subset_size" in src


def test_triples_are_not_explored():
    """If no single or pairwise drop helps, say so — never fall through to
    triples. Every constraint here is individually and pairwise necessary to
    exclude the only recipe."""
    recipes = [_r("only", {"dairy", "peanut", "gluten"}, prep=90)]
    cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c2", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut"),
        Constraint(id="c3", kind="exclude_tag", severity=Severity.PREFERENCE, value="gluten"),
    )
    f = assess(cs, recipes)
    assert f.feasible is False
    assert f.negotiation.relaxations == []


def test_spends_zero_tokens():
    """Feasibility must never touch the model. Asserted by construction:
    the module is under t0_invariants/, which test_boundaries.py polices."""
    import beatroot.t0_invariants.feasibility as mod

    assert "reasoning" not in mod.__file__


def test_descriptions_read_as_actions_not_machine_output():
    """Negotiation copy goes on camera in front of a user. A relaxation
    DROPS the constraint, so every numeric branch must read as "go beyond
    this limit" — never "set it to this limit", which falsely implies the
    limit becomes exactly that number. Numbers must not carry a trailing
    '.0', and singular/plural must agree (1 minute, 2 minutes)."""
    import re

    from beatroot.t0_invariants.feasibility import _describe

    prep_c = Constraint(id="c1", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=30)
    budget_c = Constraint(id="c2", kind="budget_max", severity=Severity.PREFERENCE, value=160)
    nutrient_c = Constraint(
        id="c3", kind="nutrient_range", severity=Severity.GOAL, value=(100, 500), nutrient="kcal"
    )
    one_minute_c = Constraint(
        id="c4", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=1
    )

    for desc in (_describe(prep_c), _describe(budget_c), _describe(nutrient_c)):
        assert ".0" not in desc
        # Direction semantics: no numeric branch may read "to <number>" — a
        # relaxation lifts a boundary, it does not set a new target.
        assert re.search(r"\bto\s+₹?\d", desc) is None, desc

    # Direct, unambiguous "go beyond" phrasing per branch:
    assert "more than 30" in _describe(prep_c)
    assert "above ₹160" in _describe(budget_c)
    assert "beyond 100" in _describe(nutrient_c) and "500" in _describe(nutrient_c)

    # Singular/plural agreement on the boundary number.
    assert _describe(one_minute_c) == "allow more than 1 minute of prep"
    assert _describe(prep_c) == "allow more than 30 minutes of prep"


def test_describe_falls_back_to_generic_nutrient_wording_when_nutrient_is_none():
    """`nutrient = c.nutrient or "nutrient"` — a nutrient_range constraint
    with no `nutrient` set must still produce readable copy, naming the
    axis generically rather than rendering `None` or crashing."""
    from beatroot.t0_invariants.feasibility import _describe

    c = Constraint(
        id="c5", kind="nutrient_range", severity=Severity.GOAL, value=(100, 500), nutrient=None
    )
    assert _describe(c) == "widen the nutrient range beyond 100-500"


def _build_survivor_fixture():
    recipes = [
        _r("a", {"vegan", "peanut"}, prep=10),
        _r("b", {"vegan", "dairy"}, prep=25),
        _r("c", {"vegetarian", "dairy"}, prep=45),
        _r("d", {"vegan"}, prep=90),
        _r("e", {"gluten", "dairy", "peanut"}, prep=15),
    ]
    recipes[0].nutrition = NutritionFacts(
        kcal=400, protein_g=20, carbs_g=40, fat_g=10, sodium_mg=300, fibre_g=5, coverage=1.0
    )
    recipes[0].cost_inr = 120.0
    recipes[1].nutrition = NutritionFacts(
        kcal=600, protein_g=10, carbs_g=60, fat_g=20, sodium_mg=500, fibre_g=3, coverage=1.0
    )
    recipes[1].cost_inr = 300.0
    recipes[2].nutrition = NutritionFacts(
        kcal=250, protein_g=5, carbs_g=30, fat_g=5, sodium_mg=200, fibre_g=6, coverage=1.0
    )
    recipes[2].cost_inr = 80.0
    recipes[3].cost_inr = 500.0
    recipes[4].nutrition = NutritionFacts(
        kcal=800, protein_g=30, carbs_g=90, fat_g=40, sodium_mg=900, fibre_g=2, coverage=1.0
    )
    return recipes


class _ExplodingCatalog(list):
    """Raises if anything iterates it. The indexed path must reach recipes
    only through the index, so touching this list at all is the regression
    the earlier check-recipe-call-counting test failed to catch: `and`
    short-circuits, so check_recipe was already only invoked for survivors
    even in the old, catalog-scanning body. The real O(catalog) cost lived
    in the membership scan (`for r in recipes if r.id in survivors`) and in
    `to_ids()`'s enumerate — neither of which the call-count test could see.
    This test makes iterating the catalog structurally impossible instead."""

    def __iter__(self):
        raise AssertionError(
            "indexed _survivors iterated the full catalog — the O(survivors) property has regressed"
        )


def test_indexed_path_never_iterates_the_catalog():
    recipes = [_r(f"r{i}", {"vegan"} if i % 2 else {"peanut"}) for i in range(2000)]
    index = TagIndex(recipes)  # construction may iterate; that is once, at build
    cs = _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="vegan"))
    survivors = _survivors(_ExplodingCatalog(recipes), cs, index)
    assert len(survivors) == 1000
    assert all("peanut" in r.tags for r in survivors)


def test_bitmask_path_agrees_with_scan_path():
    """The single most important test in this task: an optimisation whose
    results differ from the reference implementation is a bug, not an
    optimisation. Exercised across tag-only, nutrient, budget, prep, mixed,
    unknown-tag, empty, and nothing-survives constraint sets."""
    recipes = _build_survivor_fixture()
    index = TagIndex(recipes)

    constraint_sets = [
        _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")),
        _cs(
            Constraint(id="c1", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
            Constraint(id="c2", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut"),
        ),
        _cs(Constraint(id="c1", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=30)),
        _cs(Constraint(id="c1", kind="budget_max", severity=Severity.PREFERENCE, value=200)),
        _cs(
            Constraint(
                id="c1",
                kind="nutrient_range",
                severity=Severity.GOAL,
                value=(0, 500),
                nutrient="kcal",
            )
        ),
        _cs(
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="dairy"),
            Constraint(id="c2", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=20),
            Constraint(id="c3", kind="budget_max", severity=Severity.PREFERENCE, value=400),
        ),
        _cs(Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="unicorn")),
        _cs(),
        # Nothing survives: every recipe carries "vegan" or "dairy" or
        # "peanut" — exercises the `if mask == 0: return []` short-circuit.
        _cs(
            Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="vegan"),
            Constraint(id="c2", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
            Constraint(id="c3", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut"),
        ),
    ]

    for cs in constraint_sets:
        via_index = {r.id for r in _survivors(recipes, cs, index)}
        via_scan = {r.id for r in _survivors(recipes, cs, None)}
        assert via_index == via_scan, f"mismatch for {[c.id for c in cs.constraints]}"


def test_nothing_survives_returns_empty_via_both_paths():
    """Direct check on the mask==0 short-circuit, independent of the loop
    above: every recipe carries one of the three excluded tags."""
    recipes = _build_survivor_fixture()
    index = TagIndex(recipes)
    cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="vegan"),
        Constraint(id="c2", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c3", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut"),
    )
    assert _survivors(recipes, cs, index) == []
    assert _survivors(recipes, cs, None) == []


def test_assess_and_rank_relaxations_agree_with_and_without_index():
    """Important: exercise the PUBLIC entry points (assess, rank_relaxations)
    with a real TagIndex, not just the private _survivors helper — those are
    the specified seam and must agree end to end: feasibility, the
    surviving id set, and the full relaxation ladder (descriptions, unlocks,
    ordering)."""
    recipes = _build_survivor_fixture()
    index = TagIndex(recipes)

    feasible_cs = _cs(
        Constraint(id="c1", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
    )
    f_idx = assess(feasible_cs, recipes, index)
    f_scan = assess(feasible_cs, recipes, None)
    assert f_idx.feasible == f_scan.feasible is True
    assert {r.id for r in f_idx.surviving} == {r.id for r in f_scan.surviving}
    assert f_idx.negotiation is f_scan.negotiation is None

    infeasible_cs = _cs(
        Constraint(id="med", kind="exclude_tag", severity=Severity.MEDICAL, value="vegan"),
        Constraint(id="c1", kind="exclude_tag", severity=Severity.PREFERENCE, value="dairy"),
        Constraint(id="c2", kind="max_prep_minutes", severity=Severity.PREFERENCE, value=12),
    )
    f_idx = assess(infeasible_cs, recipes, index)
    f_scan = assess(infeasible_cs, recipes, None)
    assert f_idx.feasible == f_scan.feasible is False
    assert f_idx.negotiation.locked == f_scan.negotiation.locked
    idx_relax = [
        (r.constraint_ids, r.description, r.unlocks) for r in f_idx.negotiation.relaxations
    ]
    scan_relax = [
        (r.constraint_ids, r.description, r.unlocks) for r in f_scan.negotiation.relaxations
    ]
    assert idx_relax == scan_relax

    rr_idx = rank_relaxations(infeasible_cs, recipes, index)
    rr_scan = rank_relaxations(infeasible_cs, recipes, None)
    assert [(r.constraint_ids, r.description, r.unlocks) for r in rr_idx] == [
        (r.constraint_ids, r.description, r.unlocks) for r in rr_scan
    ]

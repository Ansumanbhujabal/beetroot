from pathlib import Path

from beatroot.container import build_container
from beatroot.eval.runners.components import _derive_query, run_components
from beatroot.eval.synth.profiles import generate_profiles


def test_retrieval_leakage_is_zero(tmp_path: Path) -> None:
    """The headline component metric: hybrid retrieval must never surface an
    illegal candidate, at any profile."""
    container = build_container(tmp_path / "c.db")
    cases = generate_profiles(container.catalog, n=60, seed=3)
    report = run_components(container, cases)
    assert report.retrieval_leakage == 0


def test_feasibility_matches_the_oracle_exactly(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    cases = generate_profiles(container.catalog, n=60, seed=4)
    assert run_components(container, cases).feasibility_accuracy == 1.0


def test_nutrition_is_deterministic(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    assert run_components(container, []).nutrition_exact_match == 1.0


def test_recall_at_k_is_between_zero_and_one(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    cases = generate_profiles(container.catalog, n=40, seed=8)
    report = run_components(container, cases)
    assert 0.0 <= report.retrieval_recall_at_k <= 1.0


def test_empty_cases_do_not_crash_and_score_zero_recall(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    report = run_components(container, [])
    assert report.retrieval_recall_at_k == 0.0
    assert report.retrieval_recall_at_k_hard_only == 0.0
    assert report.retrieval_leakage == 0
    assert report.feasibility_accuracy == 0.0


# ---- BENCHMARK FIX: per-case query derivation, dual-contract recall -----


def test_derive_query_uses_cuisine_affinity_when_the_profile_carries_one(tmp_path: Path) -> None:
    from beatroot.contracts.core import Constraint, ConstraintSet
    from beatroot.eval.synth.profiles import SyntheticCase

    container = build_container(tmp_path / "c.db")
    cs = ConstraintSet(
        profile_id="p1",
        constraints=[
            Constraint(
                id="c0", kind="cuisine_affinity", severity="preference", value="north_indian"
            )
        ],
    )
    case = SyntheticCase(id="p1", constraint_set=cs, oracle_valid_ids={"x"})
    assert _derive_query(case, container.catalog) == "north indian"


def test_derive_query_falls_back_to_a_real_catalog_cuisine_deterministically(
    tmp_path: Path,
) -> None:
    from beatroot.contracts.core import ConstraintSet
    from beatroot.eval.synth.profiles import SyntheticCase

    container = build_container(tmp_path / "c.db")
    cs = ConstraintSet(profile_id="p2", constraints=[])
    case = SyntheticCase(id="p2", constraint_set=cs, oracle_valid_ids={"x"})

    known_cuisines = {r.cuisine for r in container.catalog.recipes() if r.cuisine}
    query = _derive_query(case, container.catalog)
    assert query.replace(" ", "_") in known_cuisines

    # Deterministic: the same case id always derives the same query.
    assert _derive_query(case, container.catalog) == query


def test_derive_query_never_looks_at_the_oracle(tmp_path: Path) -> None:
    """Two cases with identical constraints but DIFFERENT oracle sets must
    derive the identical query — proves the derivation is a pure function
    of (case.id, constraint_set), never of oracle_valid_ids, which would
    bias the query toward the answers recall is graded against."""
    from beatroot.contracts.core import ConstraintSet
    from beatroot.eval.synth.profiles import SyntheticCase

    container = build_container(tmp_path / "c.db")
    cs = ConstraintSet(profile_id="same_id", constraints=[])
    case_a = SyntheticCase(id="same_id", constraint_set=cs, oracle_valid_ids={"a", "b"})
    case_b = SyntheticCase(id="same_id", constraint_set=cs, oracle_valid_ids={"z"})
    assert _derive_query(case_a, container.catalog) == _derive_query(case_b, container.catalog)


def test_recall_at_k_hard_only_is_between_zero_and_one(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    cases = generate_profiles(container.catalog, n=40, seed=8)
    report = run_components(container, cases)
    assert 0.0 <= report.retrieval_recall_at_k_hard_only <= 1.0


def test_component_report_notes_disclose_the_benchmark_change(tmp_path: Path) -> None:
    container = build_container(tmp_path / "c.db")
    cases = generate_profiles(container.catalog, n=20, seed=11)
    report = run_components(container, cases)
    joined = " ".join(report.notes)
    assert "NEW BASELINE" in joined
    assert "not comparable" in joined
    assert "hard_only" in joined or "hard-only" in joined


def test_vector_store_is_built_once_not_per_case(tmp_path: Path) -> None:
    """`run_components` must reuse `container.vector_store` rather than
    constructing a fresh index per case — the same object identity before
    and after a run proves no rebuild happened."""
    container = build_container(tmp_path / "c.db")
    before = container.vector_store
    cases = generate_profiles(container.catalog, n=10, seed=9)
    run_components(container, cases)
    assert container.vector_store is before

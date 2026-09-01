"""Each safety layer must hold ALONE.

Mutation-testing the eval suite showed that disabling T0's `exclude_tag`
evaluator changed nothing observable: retrieval's filter pushdown blocked every
peanut-bearing recipe by itself. That is defence in depth working — but it also
means the system eval cannot tell the two layers apart, and a layer whose
failure is invisible is a layer nobody will notice losing.

These tests disable one layer at a time and assert the other still holds, so
each is verified on its own rather than only in company.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

import beatroot.t0_invariants.constraints as constraints_module
from beatroot.container import build_container
from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.retrieval.rerank import retrieve
from beatroot.t0_invariants.constraints import is_legal

MEDICAL_PEANUT = Constraint(
    id="m", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut"
)


@pytest.fixture
def container():
    with tempfile.TemporaryDirectory() as d:
        yield build_container(pathlib.Path(d) / "t.db")


def _cs() -> ConstraintSet:
    return ConstraintSet(profile_id="p", constraints=[MEDICAL_PEANUT])


def test_retrieval_pushdown_holds_when_t0_is_disabled(container, monkeypatch):
    """Layer 1 alone: with T0's tag evaluator neutered, the stores must still
    refuse to surface a peanut-bearing recipe."""
    monkeypatch.setitem(
        constraints_module._REGISTRY, "exclude_tag", lambda recipe, c: "satisfied"
    )
    got = retrieve(
        "aloo tikki peanut",
        _cs(),
        container.catalog,
        container.llm,
        vector_store=container.vector_store,
        top_k=20,
    )
    assert got, "sanity: retrieval should still return candidates"
    assert not [r for r in got if "peanut" in r.tags]


def test_t0_holds_when_retrieval_is_bypassed(container):
    """Layer 2 alone: hand T0 the peanut-bearing recipes directly, as though
    retrieval had never filtered, and it must still call them illegal."""
    peanut = [r for r in container.catalog.recipes() if "peanut" in r.tags]
    assert peanut, "catalog fixture must contain peanut-bearing recipes"
    assert not [r for r in peanut if is_legal(r, _cs())]


def test_both_layers_target_the_same_recipes(container):
    """The layers are only redundant if they agree on what is unsafe. If one
    considered a different set illegal, 'defence in depth' would be two partial
    filters wearing a trench coat."""
    cs = _cs()
    by_tag = {r.id for r in container.catalog.recipes() if "peanut" in r.tags}
    by_t0 = {r.id for r in container.catalog.recipes() if not is_legal(r, cs)}
    assert by_tag == by_t0

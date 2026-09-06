import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from beatroot.agent.graph import MealPlanningAgent
from beatroot.agent.nodes import Deps
from beatroot.agent.skills_registry import load_skills
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.trust import CostRecord
from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.dense import DenseIndex
from beatroot.store.audit import AuditLog
from beatroot.store.cache import FeasibilityCache
from beatroot.store.db import connect, seed
from beatroot.store.incidents import IncidentLog
from beatroot.trusted.catalog import Catalog
from beatroot.trusted.index import TagIndex

DATA = Path(__file__).parents[2] / "data"


class _CountingLLM(LLMClient):
    """Wraps the offline client and records every model call. The whole
    point of Feature 1 is that an infeasible profile makes zero of these."""

    def __init__(self) -> None:
        super().__init__(offline=True)
        self.calls: list[tuple] = []

    def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
        self.calls.append(("complete", stage))
        return super().complete(prompt, schema=schema, stage=stage)

    def embed(self, texts):
        self.calls.append(("embed", len(texts)))
        return super().embed(texts)


class _LiveishLLM(LLMClient):
    """Offline determinism, but every completion carries a synthetic
    NON-ZERO cost — standing in for what `litellm.completion_cost` returns
    against a real provider, so the cost-accumulation tests don't depend on
    live network access or credentials to prove `PlanState.cost` actually
    sums instead of silently defaulting to zero."""

    _USD_PER_CALL = 0.0021
    _PROMPT_TOKENS = 42
    _COMPLETION_TOKENS = 17

    def __init__(self) -> None:
        super().__init__(offline=True)

    def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
        completion = super().complete(prompt, schema=schema, stage=stage)
        completion.cost = CostRecord(
            prompt_tokens=self._PROMPT_TOKENS,
            completion_tokens=self._COMPLETION_TOKENS,
            usd=self._USD_PER_CALL,
            per_stage={stage: self._USD_PER_CALL} if stage else {},
        )
        return completion


class _RaisingLLM(LLMClient):
    """Raises on `.complete()` calls at one chosen stage, offline-normal
    everywhere else — used to prove a SPECIFIC node's own exception routes
    to ESCALATE instead of escaping `graph.invoke()`. Defaults to the
    "explain" stage (the deepest node before VERIFY/COMMIT) so a failure
    there, not the first LLM-touching node (`score`'s rerank), is what gets
    exercised."""

    def __init__(self, fail_at_stage: str = "explain") -> None:
        super().__init__(offline=True)
        self._fail_at_stage = fail_at_stage

    def complete(self, prompt, *, schema=None, stage="", prompt_ref=None):
        if stage == self._fail_at_stage:
            raise RuntimeError(f"simulated provider failure at stage={stage!r}")
        return super().complete(prompt, schema=schema, stage=stage)


def _build_deps(conn: sqlite3.Connection, llm=None) -> Deps:
    catalog = Catalog(conn)
    llm = llm or LLMClient.offline()
    vector_store = DenseIndex(llm, catalog)
    tag_index = TagIndex(catalog.recipes())
    return Deps(
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=tag_index,
        incidents=IncidentLog(conn),
        audit=AuditLog(conn),
        skills=load_skills(),
        feasibility_cache=FeasibilityCache(conn),
    )


@pytest.fixture
def agent_deps(tmp_path) -> Deps:
    conn = connect(tmp_path / "agent.db")
    seed(conn, DATA)
    return _build_deps(conn)


@pytest.fixture
def agent(agent_deps) -> MealPlanningAgent:
    return MealPlanningAgent(agent_deps)


@pytest.fixture
def preset_cs() -> Callable[[str], ConstraintSet]:
    """Build a `ConstraintSet` from a shipped preset in `data/profiles.yaml`.

    `data/profiles.yaml` is data, not a fixture, and a test that wants "the
    pescetarian peanut-allergy profile" otherwise hand-maps
    `kind`/`severity`/`value` dicts — including the list-valued
    `require_any_tag` case — before it can make its first assertion. These
    are the profiles the API's own `/profiles` route serves and the ones a
    real caller picks from, so a test asserting against them is asserting
    against something that ships.

    `ConstraintSet` also requires a `profile_id`; this uses the preset's own
    id, so two different presets never collide on one identity.
    """

    def build(profile_id: str) -> ConstraintSet:
        raw = yaml.safe_load((DATA / "profiles.yaml").read_text()) or []
        entry = next((e for e in raw if e["id"] == profile_id), None)
        if entry is None:
            known = ", ".join(sorted(e["id"] for e in raw))
            raise KeyError(f"no preset profile {profile_id!r} in profiles.yaml; have: {known}")
        return ConstraintSet(
            profile_id=entry["id"],
            constraints=[Constraint(**c) for c in entry.get("constraints", [])],
        )

    return build


@pytest.fixture
def counting_llm() -> _CountingLLM:
    return _CountingLLM()


@pytest.fixture
def liveish_llm() -> _LiveishLLM:
    return _LiveishLLM()


@pytest.fixture
def raising_llm() -> _RaisingLLM:
    return _RaisingLLM()


@pytest.fixture
def agent_with_checkpointer(agent_deps, tmp_path) -> MealPlanningAgent:
    """A FILE-backed checkpointer (`tmp_path/checkpoints.db`) — distinct from
    the in-memory one every default `agent` fixture gets, so a test built on
    this fixture actually exercises `SqliteSaver`'s on-disk path rather than
    a connection indistinguishable from the default. This fixture alone does
    NOT prove durability across a fresh connection (a fresh Python object
    reopening this same file, with no reference to the writer surviving) —
    `test_checkpoint_survives_a_fresh_connection_to_the_same_file` builds
    that scenario end to end instead of trusting this fixture's file-ness to
    imply it."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(str(tmp_path / "checkpoints.db"), check_same_thread=False)
    return MealPlanningAgent(agent_deps, checkpointer=SqliteSaver(conn))


# ---------------------------------------------------------------------------
# A tiny, hand-built catalog for the interrupt/grey-band tests below. These
# need EXACT control over `NutritionFacts.coverage` to land trust in the
# grey band deterministically — the real ~100-recipe catalog does not offer
# that control without depending on incidental data shape. `store.db.seed`
# validates every ingredient's per_100g at seed time, so partial coverage is
# engineered post-seed by deleting one ingredient row outright: the recipe
# that referenced it still resolves (Recipe.ingredient_ids is set at seed
# time from the payload), but `Catalog.ingredient_payload` now returns None
# for it, exactly like an ingredient the catalog never learned about.
# ---------------------------------------------------------------------------

_GREY_INGREDIENTS = [
    {
        "id": "gb_rice",
        "name": "Grey Band Rice",
        "per_100g": {
            "kcal": 130.0,
            "protein_g": 2.7,
            "carbs_g": 28.0,
            "fat_g": 0.3,
            "sodium_mg": 1.0,
            "fibre_g": 0.4,
        },
        "cost_per_100g_inr": 8.0,
    },
    {
        "id": "gb_dal",
        "name": "Grey Band Dal",
        "per_100g": {
            "kcal": 116.0,
            "protein_g": 9.0,
            "carbs_g": 20.0,
            "fat_g": 0.4,
            "sodium_mg": 2.0,
            "fibre_g": 8.0,
        },
        "cost_per_100g_inr": 12.0,
    },
    {
        # Exists ONLY so "peanut" is a real, checkable tag in this
        # micro-catalog's vocabulary — t0_invariants.vocabulary.
        # unknown_vocabulary would otherwise escalate the grey-band tests'
        # `exclude_tag peanut` constraint straight out of FEASIBILITY
        # (correctly: a 2-ingredient catalog that never mentions peanut
        # anywhere really doesn't have that tag). Never referenced by
        # gb_target/gb_filler, so it changes nothing about which recipe
        # gets chosen or what its tags are.
        "id": "gb_peanut_marker",
        "name": "Unrelated Peanut Garnish",
        "allergen_tags": ["peanut"],
        "per_100g": {
            "kcal": 50.0,
            "protein_g": 2.0,
            "carbs_g": 2.0,
            "fat_g": 4.0,
            "sodium_mg": 1.0,
            "fibre_g": 1.0,
        },
        "cost_per_100g_inr": 5.0,
    },
]

_GREY_RECIPES = [
    {
        "id": "gb_target",
        "name": "Grey Band Rice Bowl",
        "cuisine": "test",
        "prep_minutes": 20,
        "ingredients": [
            {"ingredient_id": "gb_rice", "grams": 100.0},
            {"ingredient_id": "gb_dal", "grams": 100.0},
        ],
    },
    {
        "id": "gb_filler",
        "name": "Filler Curry",
        "cuisine": "test",
        "prep_minutes": 15,
        "ingredients": [{"ingredient_id": "gb_rice", "grams": 50.0}],
    },
    {
        # Only here to carry gb_peanut_marker into a Recipe.tags set so
        # "peanut" shows up in `_known_tags` — see the ingredient's own
        # comment above. Named and worded to share no tokens with the
        # queries these tests actually use ("Grey Band Rice Bowl",
        # "Filler Curry"), so lexical/dense retrieval never prefers it.
        "id": "gb_unrelated",
        "name": "Zzz Unrelated Peanut Snack Zzz",
        "cuisine": "test",
        "prep_minutes": 5,
        "ingredients": [{"ingredient_id": "gb_peanut_marker", "grams": 20.0}],
    },
]


@pytest.fixture
def grey_band_deps(tmp_path) -> Deps:
    """One legal recipe ("Grey Band Rice Bowl") whose catalog coverage is
    pinned to 0.5 by deleting `gb_dal` after seeding, plus a filler recipe so
    reranking actually calls the model (a single-candidate list short-circuits
    `llm_rerank` to `self_assessment=1.0`, which would push trust out of the
    grey band). The query names the target recipe so lexical+dense fusion
    puts it at rank 0, which the offline model's fixed `choice_index=0`
    then confirms.
    """
    data_dir = tmp_path / "grey_data"
    data_dir.mkdir()
    (data_dir / "ingredients.yaml").write_text(yaml.dump(_GREY_INGREDIENTS))
    (data_dir / "recipes.yaml").write_text(yaml.dump(_GREY_RECIPES))

    conn = connect(tmp_path / "grey.db")
    seed(conn, data_dir)
    conn.execute("DELETE FROM ingredients WHERE id = 'gb_dal'")
    conn.commit()

    return _build_deps(conn)

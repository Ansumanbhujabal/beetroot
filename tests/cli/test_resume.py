"""Tests for the CLI `resume` command. Spec §6, §15.

Each CLI command (`recommend`, `resume`, `incidents`) calls
`build_container()` fresh and closes it before exiting — a real,
independent process lifetime, not a long-lived server. `_pause_a_thread`
below simulates a PRIOR, separate `beatroot recommend --medical peanut`
invocation directly: it builds `Deps`/`MealPlanningAgent` by hand (like
`tests/agent/conftest.py`'s `grey_band_deps` fixture) rather than through
`build_container()`, because engineering the partial-coverage grey band
needs deleting `gb_dal` from the catalog AFTER seeding but BEFORE
`Catalog._load()` ever runs — and `build_container()` itself already
triggers that load (building `TagIndex(catalog.recipes())`) before this
test ever gets a chance to touch the connection. It uses the SAME
file-backed checkpointer helper (`container._build_checkpointer`)
`build_container()` uses, on the SAME `DEFAULT_DB` path the fixture below
patches — that shared path is the only thing connecting it to the REAL
`resume` command every test here drives through `CliRunner`, building its
own, later, unrelated `Container`. The only way that later `Container`
finds `thread_id` at all is by reading the checkpoint back off disk, which
is the exact fix this proves.
"""

import pytest
import yaml
from typer.testing import CliRunner

import beatroot.container as container_module
from beatroot.agent.graph import MealPlanningAgent
from beatroot.agent.nodes import Deps
from beatroot.agent.skills_registry import load_skills
from beatroot.cli.main import app
from beatroot.contracts.core import Constraint, ConstraintSet, Severity
from beatroot.contracts.result import Recommendation
from beatroot.reasoning.llm import LLMClient
from beatroot.retrieval.dense import DenseIndex
from beatroot.settings import get_settings
from beatroot.store.audit import AuditLog
from beatroot.store.cache import EmbeddingCache, FeasibilityCache
from beatroot.store.db import connect, seed
from beatroot.store.incidents import IncidentLog
from beatroot.trusted.catalog import Catalog
from beatroot.trusted.index import TagIndex
from tests.agent.conftest import _GREY_INGREDIENTS, _GREY_RECIPES

runner = CliRunner()


@pytest.fixture(autouse=True)
def _grey_boot(tmp_path, monkeypatch):
    data_dir = tmp_path / "grey_data"
    data_dir.mkdir()
    (data_dir / "ingredients.yaml").write_text(yaml.dump(_GREY_INGREDIENTS))
    (data_dir / "recipes.yaml").write_text(yaml.dump(_GREY_RECIPES))

    monkeypatch.setattr(container_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(container_module, "DEFAULT_DB", tmp_path / "cli.db")
    monkeypatch.setenv("BEATROOT_OFFLINE", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _build_agent_by_hand(delete_gb_dal: bool) -> MealPlanningAgent:
    """Mirrors `tests/agent/conftest.py::grey_band_deps` exactly, but on
    `container_module.DEFAULT_DB`/`DATA_DIR` (patched above) and with the
    SAME file-backed checkpointer `build_container()` gives a real process
    — so a checkpoint written here is exactly as durable as one a real
    `beatroot recommend` process would have written."""
    conn = connect(container_module.DEFAULT_DB)
    seed(conn, container_module.DATA_DIR)
    if delete_gb_dal:
        conn.execute("DELETE FROM ingredients WHERE id = 'gb_dal'")
        conn.commit()

    catalog = Catalog(conn)
    llm = LLMClient.offline()
    embedding_cache = EmbeddingCache(conn)
    vector_store = DenseIndex(llm, catalog, embedding_cache=embedding_cache)
    deps = Deps(
        catalog=catalog,
        llm=llm,
        vector_store=vector_store,
        tag_index=TagIndex(catalog.recipes()),
        incidents=IncidentLog(conn),
        audit=AuditLog(conn),
        skills=load_skills(),
        preferences=None,
        feasibility_cache=FeasibilityCache(conn),
    )
    checkpointer = container_module._build_checkpointer(container_module.DEFAULT_DB)
    return MealPlanningAgent(deps, checkpointer=checkpointer)


def _pause_a_thread(thread_id: str) -> None:
    agent = _build_agent_by_hand(delete_gb_dal=True)
    cs = ConstraintSet(
        profile_id="cli-grey",
        constraints=[
            Constraint(id="med", kind="exclude_tag", severity=Severity.MEDICAL, value="peanut")
        ],
    )
    result = agent.run(cs, query="Grey Band Rice Bowl", thread_id=thread_id)
    assert result is None, "fixture bug: must genuinely pause at PENDING_REVIEW"
    agent.deps.audit.conn.close()  # same conn as everything else built above
    agent.checkpointer.conn.close()  # type: ignore[union-attr]


def test_resume_approve_commits_across_a_separate_process():
    _pause_a_thread("cli-approve-1")
    result = runner.invoke(app, ["resume", "cli-approve-1", "--approve"])
    assert result.exit_code == 0, result.output
    assert "COMMIT" in result.output


def test_resume_reject_escalates_across_a_separate_process():
    _pause_a_thread("cli-reject-1")
    result = runner.invoke(app, ["resume", "cli-reject-1", "--reject"])
    assert result.exit_code == 0, result.output
    assert "ESCALATE" in result.output


def test_resume_unknown_thread_exits_nonzero_without_a_traceback():
    result = runner.invoke(app, ["resume", "does-not-exist"])
    assert result.exit_code != 0
    assert "unknown thread" in result.output.lower()
    assert "Traceback" not in result.output


def test_resume_on_a_thread_that_never_paused_exits_nonzero():
    agent = _build_agent_by_hand(delete_gb_dal=False)
    cs = ConstraintSet(
        profile_id="cli-committed",
        constraints=[
            Constraint(id="pref", kind="exclude_tag", severity=Severity.PREFERENCE, value="peanut")
        ],
    )
    result = agent.run(cs, query="Grey Band Rice Bowl", thread_id="cli-never-paused")
    assert isinstance(result, Recommendation), "fixture bug: must commit directly, no pause"
    agent.deps.audit.conn.close()
    agent.checkpointer.conn.close()  # type: ignore[union-attr]

    outcome = runner.invoke(app, ["resume", "cli-never-paused"])
    assert outcome.exit_code != 0
    assert "not awaiting approval" in outcome.output.lower()
    assert "Traceback" not in outcome.output


def test_resume_twice_exits_nonzero_the_second_time():
    _pause_a_thread("cli-twice-1")
    first = runner.invoke(app, ["resume", "cli-twice-1", "--approve"])
    assert first.exit_code == 0, first.output

    second = runner.invoke(app, ["resume", "cli-twice-1", "--approve"])
    assert second.exit_code != 0
    assert "not awaiting approval" in second.output.lower()

"""CLI tests. Spec §15.

Each terminal state must render legibly: COMMIT with the trust breakdown,
NEGOTIATE with the relaxation ladder and locked constraints marked, ESCALATE
with the failing signal named. NEGOTIATE and ESCALATE are not a meal, and
the output says so plainly.
"""

from rich.console import Console
from typer.testing import CliRunner

from beatroot.cli import main as cli_main
from beatroot.cli.main import app
from beatroot.contracts.core import Severity
from beatroot.contracts.nutrition import NutritionFacts
from beatroot.contracts.result import Escalation, Negotiation, Recommendation, Relaxation
from beatroot.contracts.trust import TrustReport

runner = CliRunner()


def _rendered(render_fn, obj) -> str:
    """Capture `render_fn(obj)`'s rich output as plain text, redirecting the
    module's shared `console` for the duration of the call — real data does
    not reliably produce all three terminals, so these tests build the
    contract objects directly rather than depending on which terminal a
    live catalog happens to land on."""
    buf = Console(record=True, width=100)
    original = cli_main.console
    cli_main.console = buf
    try:
        render_fn(obj)
    finally:
        cli_main.console = original
    return buf.export_text()


def test_recommend_runs_and_prints_a_terminal_state():
    result = runner.invoke(app, ["recommend", "rice"])
    assert result.exit_code == 0, result.output
    assert any(s in result.output for s in ("COMMIT", "NEGOTIATE", "ESCALATE"))


def test_commit_shows_the_trust_breakdown_when_it_happens():
    result = runner.invoke(app, ["recommend", "something warm with rice"])
    assert result.exit_code == 0, result.output
    if "COMMIT" in result.output:
        assert "trust" in result.output
        assert "catalog_coverage" in result.output
        assert "model_self_assessment" in result.output


def test_impossible_profile_prints_the_ladder_and_says_no_meal():
    result = runner.invoke(app, ["recommend", "anything", "--max-prep", "0"])
    assert result.exit_code == 0, result.output
    assert "NEGOTIATE" in result.output
    assert "no meal recommended" in result.output


def test_medical_constraint_is_marked_locked_never_a_relaxation_candidate():
    result = runner.invoke(app, ["recommend", "rice", "--medical", "peanut", "--max-prep", "0"])
    assert result.exit_code == 0, result.output
    if "NEGOTIATE" in result.output:
        assert "LOCKED" in result.output
        assert "med0" in result.output  # the medical constraint's id, struck-through/marked


def test_incidents_command_runs():
    result = runner.invoke(app, ["incidents"])
    assert result.exit_code == 0, result.output


def test_all_commands_are_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("recommend", "resume", "serve", "incidents", "heal", "eval", "synth"):
        assert cmd in result.output


# ---------------------------------------------------------------------------
# `heal`, `eval system|components`, `synth profiles|adversarial` — thin
# wrappers over entry points that already exist and are already tested
# (beatroot.heal.__main__, eval.runners.system/components,
# eval.synth.profiles/adversarial). See CUT_LIST.md for why these were not
# wired sooner.
# ---------------------------------------------------------------------------


def test_eval_system_subcommand_runs():
    result = runner.invoke(app, ["eval", "system"])
    assert result.exit_code == 0, result.output
    assert "overall: PASS" in result.output


def test_eval_components_subcommand_runs():
    result = runner.invoke(app, ["eval", "components"])
    assert result.exit_code == 0, result.output
    assert "retrieval recall@k" in result.output


def test_eval_simulation_subcommand_runs():
    result = runner.invoke(app, ["eval", "simulation", "--n", "40", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "overall: PASS" in result.output
    assert "total cases: 40" in result.output


def test_synth_profiles_subcommand_runs():
    result = runner.invoke(app, ["synth", "profiles", "--n", "10", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "generated 10 profiles" in result.output


def test_synth_adversarial_subcommand_runs():
    result = runner.invoke(app, ["synth", "adversarial", "--n", "10", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "generated 10 adversarial cases" in result.output


def test_prompts_status_reports_where_each_prompt_resolved_from():
    """"We use prompt management" is only honest if the fallback is visible.

    With no credentials every prompt resolves from the local file, and that
    is a supported, expected state — not a degraded one to hide. This
    command exists so the answer is readable rather than assumed.
    """
    result = runner.invoke(app, ["prompts", "status"])
    assert result.exit_code == 0, result.output
    for name in ("compile_constraints", "explain", "rerank", "rewrite_query"):
        assert name in result.output
    assert "@local:" in result.output, "the hermetic suite has no Langfuse credentials"


def test_obs_check_reports_unconfigured_without_credentials():
    """A diagnostic must report the keyless path as a state, not an error."""
    result = runner.invoke(app, ["obs", "check"])
    assert result.exit_code == 0, result.output
    assert "not configured" in result.output.lower()


def test_prompts_push_fails_loudly_without_credentials():
    """The opposite posture to `status`: pushing is an action with an
    intended effect, so silently doing nothing would be the wrong kind of
    quiet. It exits non-zero and says why."""
    result = runner.invoke(app, ["prompts", "push"])
    assert result.exit_code == 1
    assert "not configured" in result.output.lower()


def test_heal_subcommand_runs(tmp_path):
    out_dir = tmp_path / "healing"
    result = runner.invoke(app, ["heal", "--out-dir", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert "clusters" in result.output


# ---------------------------------------------------------------------------
# Direct renderer tests. A live catalog does not reliably produce all three
# terminals on demand (ESCALATE in particular needs an engineered low-trust
# scenario) — these construct the contract objects directly so every
# terminal's rendering is actually under test, not merely "if it happens to
# occur".
# ---------------------------------------------------------------------------


def test_render_commit_shows_the_full_trust_breakdown():
    rec = Recommendation(
        recipe_id="r1",
        recipe_name="Jeera Rice",
        nutrition=NutritionFacts(
            kcal=300, protein_g=5, carbs_g=50, fat_g=8, sodium_mg=400, fibre_g=2, coverage=1.0
        ),
        trust=TrustReport(
            composite=0.88,
            catalog_coverage=1.0,
            constraint_completeness=1.0,
            model_self_assessment=0.5,
        ),
        explanation="A warm rice dish.",
        constraints_satisfied=["c1"],
    )
    out = _rendered(cli_main._render_commit, rec)
    assert "COMMIT" in out
    assert "Jeera Rice" in out
    assert "0.88" in out
    assert "catalog_coverage=1.00" in out
    assert "constraint_completeness=1.00" in out
    assert "model_self_assessment=0.50" in out


def test_render_negotiate_shows_ladder_and_marks_locked_constraints():
    neg = Negotiation(
        total_candidates=100,
        surviving=0,
        relaxations=[
            Relaxation(
                constraint_ids=["prep"],
                description="allow more prep time",
                unlocks=40,
                severity=str(Severity.PREFERENCE),
            )
        ],
        locked=["med0"],
    )
    out = _rendered(cli_main._render_negotiate, neg)
    assert "NEGOTIATE" in out
    assert "no meal recommended" in out
    assert "allow more prep time" in out
    assert "LOCKED" in out
    assert "med0" in out


def test_render_escalate_names_the_failing_signal():
    esc = Escalation(
        reason="low_trust",
        failing_signal="catalog_coverage",
        detail="Composite trust 0.40 is below the 0.55 threshold.",
        trust=TrustReport(
            composite=0.40,
            catalog_coverage=0.20,
            constraint_completeness=1.0,
            model_self_assessment=0.5,
        ),
    )
    out = _rendered(cli_main._render_escalate, esc)
    assert "ESCALATE" in out
    assert "no meal recommended" in out
    assert "low_trust" in out
    assert "catalog_coverage" in out  # the named failing signal
    assert "0.40" in out

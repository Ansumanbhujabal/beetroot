"""The beatroot CLI. Spec §15.

`recommend` drives one profile through the agent and renders whichever of
the three terminal states it actually reached. Two of those three are NOT a
meal — `NEGOTIATE` (nothing satisfies this profile; here is the relaxation
ladder) and `ESCALATE` (trust or verification refused this recommendation;
here is the signal that failed) are both legitimate, expected outcomes, not
errors, and this CLI never dresses either of them up to look like a
successful recommendation.

`heal`, `eval system`/`eval components`, and `synth profiles`/`synth
adversarial` are thin wrappers over entry points that already exist and are
already tested (`beatroot.heal.__main__.main`, `beatroot.eval.runners.
system.main`, `beatroot.eval.runners.components.main`, `beatroot.eval.
synth.profiles.generate_profiles`, `beatroot.eval.synth.adversarial.
generate_adversarial`) — this module never reimplements any of their logic,
only renders it as a `beatroot` subcommand. See `CUT_LIST.md` for why these
were not wired sooner.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from beatroot.container import build_container
from beatroot.contracts.core import Constraint, ConstraintSet
from beatroot.contracts.result import Escalation, Negotiation, Recommendation
from beatroot.contracts.trust import TrustReport

app = typer.Typer(help="beatroot — meal planning with enforced boundaries")
eval_app = typer.Typer(help="Run eval suites (see eval/runners/).")
synth_app = typer.Typer(help="Generate synthetic eval cases with an exact oracle.")
prompts_app = typer.Typer(help="Publish and inspect versioned prompts (Langfuse).")
obs_app = typer.Typer(help="Observability diagnostics.")
app.add_typer(eval_app, name="eval")
app.add_typer(synth_app, name="synth")
app.add_typer(prompts_app, name="prompts")
app.add_typer(obs_app, name="obs")
console = Console()


def _print_trust(trust: TrustReport) -> None:
    console.print(
        f"  trust {trust.composite:.2f}  "
        f"(catalog_coverage={trust.catalog_coverage:.2f}, "
        f"constraint_completeness={trust.constraint_completeness:.2f}, "
        f"model_self_assessment={trust.model_self_assessment:.2f})"
    )
    if trust.failing_signal:
        console.print(f"  [yellow]weakest signal:[/yellow] {trust.failing_signal}")


def _render_commit(result: Recommendation) -> None:
    console.print(f"[bold green]COMMIT[/bold green] — {result.recipe_name}")
    console.print(result.explanation)
    n = result.nutrition
    console.print(
        f"  kcal={n.kcal:g} protein={n.protein_g:g}g carbs={n.carbs_g:g}g "
        f"fat={n.fat_g:g}g sodium={n.sodium_mg:g}mg fibre={n.fibre_g:g}g "
        f"(coverage {n.coverage:.2f}, provenance={n.provenance})"
    )
    _print_trust(result.trust)
    if result.constraints_satisfied:
        console.print(f"  constraints satisfied: {', '.join(result.constraints_satisfied)}")


def _render_negotiate(result: Negotiation) -> None:
    console.print(
        f"[bold yellow]NEGOTIATE[/bold yellow] — no meal recommended. "
        f"{result.surviving}/{result.total_candidates} recipes survived."
    )
    if result.relaxations:
        table = Table("relax", "unlocks", "severity", title="relaxation ladder")
        for r in result.relaxations:
            table.add_row(r.description, str(r.unlocks), r.severity)
        console.print(table)
    else:
        console.print("  no single- or paired-constraint relaxation helps.")
    if result.locked:
        struck = ", ".join(f"[strike]{cid}[/strike]" for cid in result.locked)
        console.print(
            f"  [bold red]LOCKED (never relaxed — medical/religious):[/bold red] {struck}"
        )


def _render_escalate(result: Escalation) -> None:
    console.print(f"[bold red]ESCALATE[/bold red] — no meal recommended. refused: {result.reason}")
    console.print(f"  failing signal: [bold]{result.failing_signal}[/bold]")
    console.print(f"  {result.detail}")
    if result.trust:
        _print_trust(result.trust)


def _build_constraints(
    exclude: list[str], medical: list[str], max_prep: int | None
) -> list[Constraint]:
    constraints = [
        Constraint(id=f"pref{i}", kind="exclude_tag", severity="preference", value=t)
        for i, t in enumerate(exclude)
    ] + [
        Constraint(id=f"med{i}", kind="exclude_tag", severity="medical", value=t)
        for i, t in enumerate(medical)
    ]
    if max_prep is not None:
        constraints.append(
            Constraint(id="prep", kind="max_prep_minutes", severity="preference", value=max_prep)
        )
    return constraints


@app.command()
def recommend(
    query: Annotated[str, typer.Argument()] = "a balanced meal",
    exclude: Annotated[
        list[str], typer.Option("--exclude", help="tag to exclude (relaxable preference)")
    ] = [],  # noqa: B006 — typer reads this list identically to typer.Option([], ...); never mutated
    medical: Annotated[
        list[str], typer.Option("--medical", help="tag to exclude (medical, never relaxed)")
    ] = [],  # noqa: B006 — see `exclude` above
    max_prep: Annotated[int | None, typer.Option("--max-prep", help="maximum prep minutes")] = None,
    profile_id: Annotated[str, typer.Option("--profile-id")] = "cli",
    preferences: Annotated[
        str,
        typer.Option(
            "--preferences",
            help=(
                "free-text preferences, compiled into PREFERENCE-only "
                "constraints before feasibility (never medical/religious, "
                "never removes a structured constraint) — distinct from "
                "the query argument, which only feeds retrieval ranking"
            ),
        ),
    ] = "",
) -> None:
    """Run one profile through the agent and render whichever terminal
    state it actually reached — COMMIT, NEGOTIATE, or ESCALATE."""
    constraints = _build_constraints(exclude, medical, max_prep)
    container = build_container()
    try:
        cs = ConstraintSet(profile_id=profile_id, constraints=constraints)
        result = container.agent.run(cs, query=query, preferences=preferences)
        trace = container.agent.trace
        audit_id = container.agent.last_audit_id
        console.print(f"[dim]trace: {' -> '.join(trace)} (audit_id={audit_id})[/dim]")
        rewrite = container.agent.last_query_rewrite
        if rewrite and rewrite.get("applied"):
            console.print(
                f"[dim]query rewritten: {rewrite['original']!r} -> {rewrite['rewritten']!r}[/dim]"
            )

        if result is None:
            console.print(
                "[bold magenta]PENDING_REVIEW[/bold magenta] — no meal recommended. "
                "trust landed in the medical review band; a human must approve "
                f"before this can commit (thread {container.agent.last_thread_id})."
            )
        elif isinstance(result, Recommendation):
            _render_commit(result)
        elif isinstance(result, Negotiation):
            _render_negotiate(result)
        elif isinstance(result, Escalation):
            _render_escalate(result)
        else:  # pragma: no cover
            # Genuinely unreachable under MealPlanningAgent.run()'s current
            # closed return type (mypy proves it, hence the ignore below) —
            # kept as defense-in-depth so a future terminal type added to
            # that union without a matching CLI branch prints something
            # instead of silently falling through with no output.
            console.print(f"[red]unrecognised terminal state:[/red] {result!r}")  # type: ignore[unreachable]
    finally:
        container.close()


@app.command()
def resume(
    thread_id: Annotated[str, typer.Argument(help="thread_id from a PENDING_REVIEW run")],
    approved: Annotated[
        bool,
        typer.Option("--approve/--reject", help="approve or reject the paused recommendation"),
    ] = True,
) -> None:
    """Continue a thread paused at the medical review gate (PENDING_REVIEW).

    Unknown thread_ids and threads that are not currently paused (never
    needed approval, or already resumed) are reported and exit non-zero —
    never a stack trace.
    """
    container = build_container()
    try:
        status = container.agent.thread_state(thread_id)
        if status == "unknown":
            console.print(f"[bold red]unknown thread:[/bold red] {thread_id!r}")
            raise typer.Exit(code=1)
        if status == "resolved":
            console.print(
                f"[bold red]not awaiting approval:[/bold red] thread {thread_id!r} either "
                "never paused or was already resumed."
            )
            raise typer.Exit(code=1)

        result = container.agent.resume(thread_id, approved)
        trace = container.agent.trace
        audit_id = container.agent.last_audit_id
        console.print(f"[dim]trace: {' -> '.join(trace)} (audit_id={audit_id})[/dim]")

        if isinstance(result, Recommendation):
            _render_commit(result)
        elif isinstance(result, Negotiation):
            _render_negotiate(result)
        elif isinstance(result, Escalation):
            _render_escalate(result)
        else:  # pragma: no cover — .resume() never re-pauses; result is always one of the above
            console.print(f"[red]unrecognised terminal state:[/red] {result!r}")
    finally:
        container.close()


@app.command()
def serve(
    host: Annotated[
        str, typer.Option("--host", help="bind address; 0.0.0.0 for containerised runs")
    ] = "0.0.0.0",  # noqa: S104 — deliberate: the standard docker-compose bind-all address
    port: Annotated[int, typer.Option("--port")] = 7860,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    uvicorn.run("beatroot.api.main:app", host=host, port=port, reload=reload)


@app.command()
def incidents(
    limit: Annotated[int, typer.Option("--limit", help="most recent N incidents")] = 20,
) -> None:
    """List recorded incidents — every escalation, refusal, drift finding,
    and infeasibility beatroot has logged."""
    container = build_container()
    try:
        rows = container.incidents.all()[-limit:]
        table = Table("kind", "profile", "detail", "when")
        for i in rows:
            table.add_row(
                i.kind, i.profile_id, i.detail[:70], i.created_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        console.print(table)
        console.print(f"{len(rows)} of {len(container.incidents.all())} incident(s) shown")
    finally:
        container.close()


@app.command()
def heal(
    out_dir: Annotated[
        Path | None,
        typer.Option("--out-dir", help="where to write proposals (default: eval/healing/)"),
    ] = None,
) -> None:
    """Cluster every recorded incident and write healing proposals — the
    same `beatroot.heal.__main__.main` used by `uv run python -m
    beatroot.heal`, wired here as a real subcommand."""
    from beatroot.heal.__main__ import DEFAULT_OUT_DIR
    from beatroot.heal.__main__ import main as heal_main

    raise typer.Exit(code=heal_main(out_dir if out_dir is not None else DEFAULT_OUT_DIR))


@eval_app.command("system")
def eval_system() -> None:
    """Run the system-level safety eval suite — six adversarial-family
    axes against a catalog-derived oracle, zero credentials required. Same
    entry point as `uv run python -m beatroot.eval.runners.system`."""
    from beatroot.eval.runners.system import main as system_main

    raise typer.Exit(code=system_main())


@eval_app.command("components")
def eval_components() -> None:
    """Run the component-level eval suite — retrieval recall/leakage,
    feasibility accuracy, nutrition determinism. Same entry point as `uv
    run python -m beatroot.eval.runners.components`."""
    from beatroot.eval.runners.components import main as components_main

    raise typer.Exit(code=components_main())


@eval_app.command("simulation")
def eval_simulation(
    n: Annotated[
        int | None, typer.Option("--n", help="how many adversarial cases (default: settings.synth)")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="reproducibility seed")] = 0,
) -> None:
    """Run the large-scale adversarial simulation — every `eval.synth.
    adversarial` family, generated at scale and run through the real
    agent, reported as a per-family pass rate rather than one blended
    number. Same entry point as `uv run python -m beatroot.eval.runners.
    simulation`."""
    from beatroot.eval.runners.simulation import main as simulation_main

    raise typer.Exit(code=simulation_main(n=n, seed=seed))


@eval_app.command("iterate")
def eval_iterate(
    label: Annotated[str, typer.Option("--label", help="what this run is (e.g. 'baseline')")],
    note: Annotated[str, typer.Option("--note", help="what changed and why")],
    verdict: Annotated[
        str, typer.Option("--verdict", help="e.g. 'kept', 'reverted', 'diagnostic'")
    ] = "",
    reason: Annotated[str, typer.Option("--reason", help="one-line reason for the verdict")] = "",
    profiles_seed: Annotated[int, typer.Option("--profiles-seed")] = 0,
    n_adversarial: Annotated[
        int | None, typer.Option("--n-adversarial", help="default: settings.synth")
    ] = None,
    adversarial_seed: Annotated[int, typer.Option("--adversarial-seed")] = 0,
    calibration_n: Annotated[
        int | None, typer.Option("--calibration-n", help="default: settings.synth")
    ] = None,
) -> None:
    """Run the FULL suite (system, components, simulation, calibration)
    against the real agent, and persist an attributed snapshot to
    `eval/history/<timestamp>.json` — regenerating `EVAL_HISTORY.md` from
    every snapshot on disk. Whether this run is offline or live follows
    `settings.offline` exactly like every other command; nothing here
    overrides it."""
    from beatroot.eval.runners.iterate import main as iterate_main

    raise typer.Exit(
        code=iterate_main(
            label=label,
            note=note,
            verdict=verdict,
            reason=reason,
            profiles_seed=profiles_seed,
            n_adversarial=n_adversarial,
            adversarial_seed=adversarial_seed,
            calibration_n=calibration_n,
        )
    )


@synth_app.command("profiles")
def synth_profiles(
    n: Annotated[
        int | None, typer.Option("--n", help="how many profiles (default: settings.synth)")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="reproducibility seed")] = 0,
) -> None:
    """Generate synthetic constraint profiles with an exact,
    catalog-derived oracle (`eval.synth.profiles.generate_profiles`) — no
    LLM, no human labelling."""
    container = build_container()
    try:
        from beatroot.eval.synth.profiles import generate_profiles

        cases = generate_profiles(container.catalog, n=n, seed=seed)
        feasible = sum(1 for c in cases if c.oracle_valid_ids)
        console.print(f"generated {len(cases)} profiles (seed={seed})")
        console.print(f"  feasible: {feasible}  infeasible: {len(cases) - feasible}")
    finally:
        container.close()


@synth_app.command("adversarial")
def synth_adversarial(
    n: Annotated[
        int | None, typer.Option("--n", help="how many cases (default: settings.synth)")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="reproducibility seed")] = 0,
) -> None:
    """Generate synthetic adversarial cases — injection attempts and
    synonym evasion against real catalog vocabulary
    (`eval.synth.adversarial.generate_adversarial`)."""
    container = build_container()
    try:
        from collections import Counter

        from beatroot.eval.synth.adversarial import generate_adversarial

        cases = generate_adversarial(container.catalog, n=n, seed=seed)
        families = Counter(c["family"] for c in cases)
        console.print(f"generated {len(cases)} adversarial cases (seed={seed})")
        for family, count in sorted(families.items()):
            console.print(f"  {family}: {count}")
    finally:
        container.close()


@prompts_app.command("push")
def prompts_push() -> None:
    """Publish every local `prompts/*.md` to Langfuse at the `production`
    label, so the running app resolves versioned prompt text instead of
    whatever happens to be on this machine's disk."""
    from beatroot.reasoning.prompts import push_prompts

    results = push_prompts()
    table = Table("prompt", "status", "detail")
    for r in results:
        mark = "[green]pushed[/green]" if r["ok"] else "[red]FAILED[/red]"
        table.add_row(r["name"], mark, r["detail"])
    console.print(table)
    if not all(r["ok"] for r in results):
        raise typer.Exit(1)


@prompts_app.command("status")
def prompts_status() -> None:
    """Show where each prompt ACTUALLY resolved from — `langfuse:vN` or
    `local:vN`.

    This is the honest version of "we use prompt management": a fallback to
    the local file is a normal, supported outcome, and it should be visible
    rather than hidden behind a claim that Langfuse is in use.
    """
    from beatroot.reasoning.prompts import load_prompt, resolved_sources

    load_prompt.cache_clear()
    table = Table("prompt", "resolved from")
    for name, ref in resolved_sources().items():
        colour = "green" if "@langfuse:" in ref else "yellow"
        table.add_row(name, f"[{colour}]{ref}[/{colour}]")
    console.print(table)


@obs_app.command("check")
def obs_check() -> None:
    """Prove the Langfuse configuration works, with a real authenticated
    API call rather than a hopeful one.

    The SDK's own `auth_check()` answers True against BOTH cloud regions no
    matter which one holds the project, so it cannot catch the single most
    common Langfuse misconfiguration — right keys, wrong host. This hits
    `/api/public/projects` and names the project that came back.
    """
    from beatroot.obs.tracing import CALLBACK, verify_langfuse_auth

    result = verify_langfuse_auth()
    if not result["configured"]:
        console.print("[yellow]Langfuse is not configured[/yellow] — tracing is a no-op.")
        console.print(result["detail"])
        return
    colour = "green" if result["ok"] else "red"
    console.print(f"host:     {result['host']}")
    console.print(f"auth:     [{colour}]{'OK' if result['ok'] else 'FAILED'}[/{colour}]")
    console.print(f"detail:   {result['detail']}")
    if result["projects"]:
        console.print(f"projects: {', '.join(result['projects'])}")
    console.print(f"callback: {CALLBACK}")
    if not result["ok"]:
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()

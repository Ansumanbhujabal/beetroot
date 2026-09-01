"""Run the healing loop end to end: `uv run python -m beatroot.heal`.

Reads every incident recorded so far (`IncidentLog.all()`), clusters them,
and writes proposals to `eval/healing/` (`proposals/` for human-gated
`threshold`/`meta_tag` changes, `generated/` for auto-applied eval cases).

Wired into `beatroot.cli.main` as the `heal` Typer command (see
`cli/main.py`). This docstring previously claimed the opposite, which was
true when written and went stale when the command landed. Kept as a module
entry point too, so it stays runnable as `python -m beatroot.heal`: Task 16 runs
alongside concurrent work in `cli/main.py` and `eval/`, so this module gives
`heal` its own runnable entry point instead of touching either. Wiring a
`beatroot heal` Typer command is deferred to whichever task next owns
`cli/main.py`.
"""

from __future__ import annotations

from pathlib import Path

from beatroot.container import ROOT, build_container
from beatroot.heal.cluster import cluster
from beatroot.heal.proposals import propose

DEFAULT_OUT_DIR = ROOT / "eval" / "healing"


def main(out_dir: Path = DEFAULT_OUT_DIR) -> int:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    container = build_container()
    try:
        clusters = cluster(container.incidents.all())
        proposals = propose(clusters, out_dir)
    finally:
        container.close()

    table = Table("kind", "auto-applied", "path")
    for p in proposals:
        table.add_row(
            p.kind,
            "yes" if p.auto_applied else "[yellow]NEEDS REVIEW[/yellow]",
            str(p.path),
        )
    console.print(table)

    pending = sum(1 for p in proposals if not p.auto_applied)
    console.print(
        f"{len(clusters)} clusters, {len(proposals)} proposals, "
        f"[bold]{pending} awaiting human approval[/bold]"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

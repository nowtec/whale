import tempfile
import webbrowser
from pathlib import Path

import click
import typer
from typer.core import TyperGroup

import agent_loop
from memory import describe_graph, evolve_from_trajectory, render_graph_html
from session import initSession, listSessions


class DefaultCommandGroup(TyperGroup):
    """Route an unmatched positional argument to the default command."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["prompt", *args]
        return super().parse_args(ctx, args)


app = typer.Typer(cls=DefaultCommandGroup)


@app.command(hidden=True)
def prompt(
    text: str = typer.Argument(..., help="Prompt to execute."),
    no_loop: bool = typer.Option(
        False, "--no-loop", help="Single shot: one turn, no multi-turn loop."
    ),
) -> None:
    """Execute a prompt through the agent loop with procedural-graph guidance."""
    if no_loop:
        result = agent_loop.run_with_guidance(text)
    else:
        session = initSession()
        result = agent_loop.run(task=text, session_id=session.id, max_turns=3)
    typer.echo(result.final_answer)


@app.command()
def refine(
    session_id: str | None = typer.Option(
        None, "--session-id", help="Session whose trajectory drives refinement."
    ),
    rounds: int = typer.Option(
        10, "--round", help="Number of refinement rounds (K)."
    ),
    last: bool = typer.Option(
        False, "--last", help="Use the most recent session (the default)."
    ),
) -> None:
    """Evolve the procedural graph from a recorded session trajectory."""
    if rounds < 1:
        typer.echo("--round must be >= 1", err=True)
        raise typer.Exit(code=2)

    if session_id is None:
        sessions = listSessions()
        if not sessions:
            typer.echo("no sessions found; run a prompt first", err=True)
            raise typer.Exit(code=1)
        session_id = sessions[0]["id"]

    for round_idx in range(1, rounds + 1):
        try:
            result = evolve_from_trajectory(session_id, round_idx=round_idx)
        except FileNotFoundError:
            typer.echo(f"session not found: {session_id}", err=True)
            raise typer.Exit(code=1)
        after = "n/a" if result.score_after is None else f"{result.score_after:.3f}"
        typer.echo(
            f"round {round_idx}: accepted={result.accepted} "
            f"score {result.score_before:.3f} -> {after} "
            f"reason={result.rejection_reason or '-'}"
        )


memory_app = typer.Typer(help="Inspect the procedural graph memory.")
app.add_typer(memory_app, name="memory")


@memory_app.command("show")
def memory_show() -> None:
    """Print the graph's nodes, edges, guidance and pitfalls."""
    typer.echo(describe_graph())


@memory_app.command("plot")
def memory_plot(
    out: Path | None = typer.Option(
        None, "--out", help="HTML file to write (default: a temp file)."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Write the file but do not open the browser."
    ),
) -> None:
    """Render the graph as a self-contained HTML diagram and open it."""
    path = out or Path(tempfile.gettempdir()) / "whale-graph.html"
    path.write_text(render_graph_html(), encoding="utf-8")
    typer.echo(f"wrote {path}")
    if not no_open:
        webbrowser.open(path.resolve().as_uri())


@app.command()
def hello(name: str) -> None:
    typer.echo(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False) -> None:
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()

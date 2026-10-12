from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import main
from agent_loop import RunResult
from orchestrator_types import (
    CritiqueReport,
    OrchestratorResult,
    PlanNode,
    PlanTree,
    SubTaskResult,
)
from procedural_memory.refiner import EvolutionResult

runner = CliRunner()


@pytest.fixture
def loop_calls(monkeypatch):
    calls = {"run": [], "run_with_guidance": []}

    def fake_run(**kwargs):
        calls["run"].append(kwargs)
        return RunResult("sid", [], "FINAL-ANSWER", True)

    def fake_single(*args, **kwargs):
        calls["run_with_guidance"].append((args, kwargs))
        return RunResult("sid", [], "SINGLE-ANSWER", True)

    monkeypatch.setattr(main.agent_loop, "run", fake_run)
    monkeypatch.setattr(main.agent_loop, "run_with_guidance", fake_single)
    monkeypatch.setattr(main, "initSession", lambda: SimpleNamespace(id="sid"))
    return calls


@pytest.fixture
def evolve_calls(monkeypatch):
    calls: list[tuple] = []

    def fake_evolve(session_id, *, round_idx=None, **kw):
        calls.append((session_id, round_idx))
        return EvolutionResult(round_idx, round_idx % 2 == 1, 0.9, 1.0, 3, 1, None)

    monkeypatch.setattr(main, "evolve_from_trajectory", fake_evolve)
    return calls


def test_prompt_routes_through_loop(loop_calls):
    result = runner.invoke(main.app, ["prompt", "hello"])
    assert result.exit_code == 0
    assert loop_calls["run"] == [{"task": "hello", "session_id": "sid", "max_turns": 3}]
    assert "FINAL-ANSWER" in result.output
    assert loop_calls["run_with_guidance"] == []


def test_prompt_no_loop(loop_calls):
    result = runner.invoke(main.app, ["prompt", "hello", "--no-loop"])
    assert result.exit_code == 0
    assert loop_calls["run"] == []
    assert loop_calls["run_with_guidance"] == [(("hello",), {})]
    assert "SINGLE-ANSWER" in result.output


def test_bare_text_routes_to_prompt(loop_calls):
    result = runner.invoke(main.app, ["do something useful"])
    assert result.exit_code == 0
    assert loop_calls["run"][0]["task"] == "do something useful"


@pytest.fixture
def orchestrate_calls(monkeypatch):
    calls: list[tuple] = []
    outcome = {"status": "ok", "reason": None}

    def fake_run(task, **kwargs):
        calls.append((task, kwargs))
        plan = PlanTree(PlanNode("root", "root", task, "done"))
        executions = [
            SubTaskResult("s1", "sess1", "first answer", "ok", 2),
            SubTaskResult("s2", "sess2", "second answer", "max_turns", 8),
        ]
        return OrchestratorResult(
            outcome["status"],
            outcome["reason"],
            plan,
            CritiqueReport([], [], "advisory"),
            executions,
            None,
        )

    monkeypatch.setattr(main.orchestrator, "run", fake_run)
    return calls, outcome


def test_plan_flag_plans_before_executing(loop_calls, orchestrate_calls):
    calls, _ = orchestrate_calls
    result = runner.invoke(main.app, ["prompt", "build it", "-p"])

    assert result.exit_code == 0
    task, kwargs = calls[0]
    assert task == "build it"
    assert kwargs["strict"] is False
    assert kwargs["on_human_approval"] is main.orchestrator.default_human_approval
    assert loop_calls["run"] == [] and loop_calls["run_with_guidance"] == []


def test_plan_flag_reports_status_and_each_subtask(loop_calls, orchestrate_calls):
    result = runner.invoke(main.app, ["prompt", "build it", "--plan"])
    assert "ok" in result.output.splitlines()[0]
    assert "- s1: ok (2 turns)" in result.output
    assert "first answer" in result.output
    assert "- s2: max_turns (8 turns)" in result.output


def test_plan_flag_works_before_the_text(loop_calls, orchestrate_calls):
    calls, _ = orchestrate_calls
    result = runner.invoke(main.app, ["-p", "build it"])
    assert result.exit_code == 0
    assert calls[0][0] == "build it"


def test_plan_strict_and_no_human_are_forwarded(loop_calls, orchestrate_calls):
    calls, _ = orchestrate_calls
    runner.invoke(main.app, ["build it", "-p", "--strict", "--no-human"])
    _, kwargs = calls[0]
    assert kwargs["strict"] is True
    assert kwargs["on_human_approval"] is None


def test_plan_failure_exits_1_with_reason(loop_calls, orchestrate_calls):
    _, outcome = orchestrate_calls
    outcome.update(status="rejected_by_human", reason="human_rejected_root_plan")
    result = runner.invoke(main.app, ["build it", "-p"])
    assert result.exit_code == 1
    assert "rejected_by_human" in result.output
    assert "human_rejected_root_plan" in result.output


def test_without_plan_flag_orchestrator_is_not_used(loop_calls, orchestrate_calls):
    calls, _ = orchestrate_calls
    runner.invoke(main.app, ["prompt", "hello"])
    assert calls == []
    assert len(loop_calls["run"]) == 1


def test_refine_runs_k_rounds_with_round_idx(evolve_calls):
    result = runner.invoke(
        main.app, ["refine", "--session-id", "abc", "--round", "3"]
    )
    assert result.exit_code == 0
    assert evolve_calls == [("abc", 1), ("abc", 2), ("abc", 3)]
    assert "round 1: accepted=True score 0.900 -> 1.000 reason=-" in result.output
    assert "round 2: accepted=False" in result.output


def test_refine_default_is_ten_rounds_on_most_recent_session(monkeypatch, evolve_calls):
    monkeypatch.setattr(
        main, "listSessions", lambda: [{"id": "newest"}, {"id": "older"}]
    )
    result = runner.invoke(main.app, ["refine"])
    assert result.exit_code == 0
    assert evolve_calls == [("newest", k) for k in range(1, 11)]


def test_refine_last_flag(monkeypatch, evolve_calls):
    monkeypatch.setattr(main, "listSessions", lambda: [{"id": "newest"}])
    result = runner.invoke(main.app, ["refine", "--last", "--round", "1"])
    assert result.exit_code == 0
    assert evolve_calls == [("newest", 1)]


def test_refine_no_sessions_exits_1(monkeypatch, evolve_calls):
    monkeypatch.setattr(main, "listSessions", lambda: [])
    result = runner.invoke(main.app, ["refine"])
    assert result.exit_code == 1
    assert "no sessions" in result.output
    assert evolve_calls == []


def test_refine_unknown_session_exits_1(monkeypatch):
    def boom(session_id, **kw):
        raise FileNotFoundError(f"session not found: {session_id}")

    monkeypatch.setattr(main, "evolve_from_trajectory", boom)
    result = runner.invoke(main.app, ["refine", "--session-id", "nope"])
    assert result.exit_code == 1
    assert "session not found: nope" in result.output


def test_refine_rejects_non_positive_rounds(evolve_calls):
    result = runner.invoke(main.app, ["refine", "--session-id", "x", "--round", "0"])
    assert result.exit_code == 2
    assert evolve_calls == []


def test_refine_prints_na_when_no_score_after(monkeypatch):
    monkeypatch.setattr(
        main,
        "evolve_from_trajectory",
        lambda sid, **kw: EvolutionResult(
            1, False, 0.5, None, 0, 0, "parse_error:bad"
        ),
    )
    result = runner.invoke(main.app, ["refine", "--session-id", "x", "--round", "1"])
    assert "round 1: accepted=False score 0.500 -> n/a reason=parse_error:bad" in result.output


def test_memory_show_prints_summary(monkeypatch):
    monkeypatch.setattr(main, "describe_graph", lambda: "GRAPH-SUMMARY")
    result = runner.invoke(main.app, ["memory", "show"])
    assert result.exit_code == 0
    assert "GRAPH-SUMMARY" in result.output


def test_memory_plot_writes_html_and_opens_browser(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(main, "render_graph_html", lambda: "<html>PLOT</html>")
    monkeypatch.setattr(main.webbrowser, "open", lambda url: opened.append(url))
    out = tmp_path / "g.html"
    result = runner.invoke(main.app, ["memory", "plot", "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_text() == "<html>PLOT</html>"
    assert str(out) in result.output
    assert opened == [out.resolve().as_uri()]


def test_memory_plot_no_open(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(main, "render_graph_html", lambda: "<html>PLOT</html>")
    monkeypatch.setattr(main.webbrowser, "open", lambda url: opened.append(url))
    out = tmp_path / "g.html"
    result = runner.invoke(main.app, ["memory", "plot", "--out", str(out), "--no-open"])
    assert result.exit_code == 0 and out.exists() and opened == []


def test_memory_plot_default_path_is_in_temp_dir(monkeypatch):
    monkeypatch.setattr(main, "render_graph_html", lambda: "<html>PLOT</html>")
    monkeypatch.setattr(main.webbrowser, "open", lambda url: None)
    result = runner.invoke(main.app, ["memory", "plot"])
    assert result.exit_code == 0
    assert "whale-graph.html" in result.output


def test_plan_max_turns_and_progress_are_forwarded(loop_calls, orchestrate_calls):
    calls, _ = orchestrate_calls
    runner.invoke(main.app, ["build it", "-p", "--max-turns", "2"])
    _, kwargs = calls[0]
    assert kwargs["max_turns"] == 2
    assert kwargs["on_progress"] is main.typer.echo

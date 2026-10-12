from __future__ import annotations

import json

import pytest

import orchestrator
from agent_loop import RunResult, TurnOutput
from memory import MemoryEngine
from orchestrator_types import CritiqueReport, PlanNode, PlanTree, SubTaskResult
from session import SessionManager


class StubLlm:
    """Replays queued replies in order; an Exception reply is raised."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def node_json(node_id: str, *children: dict, **extra) -> dict:
    return {
        "id": node_id,
        "title": node_id,
        "task": f"do {node_id}",
        "acceptance": f"{node_id} done",
        "children": list(children),
        **extra,
    }


def plan_json(root: dict) -> dict:
    return {"root": root}


def simple_tree(*leaf_ids: str) -> PlanTree:
    leaves = [
        PlanNode(id=i, title=i, task=f"do {i}", acceptance=f"{i} done")
        for i in leaf_ids or ("s1",)
    ]
    return PlanTree(
        root=PlanNode(
            id="r", title="r", task="r", acceptance="r done", children=leaves
        )
    )


# --- Recall ---------------------------------------------------------------


class FakeEngine:
    def __init__(self, guidance="graph guidance", error=None):
        self.guidance = guidance
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def retrieve_guidance(self, task, recent_actions):
        self.calls.append((task, list(recent_actions)))
        if self.error:
            raise self.error
        return self.guidance


def test_recall_queries_graph_with_no_recent_actions():
    engine = FakeEngine()
    assert orchestrator.Recall.recall("refactor auth", engine) == "graph guidance"
    assert engine.calls == [("refactor auth", [])]


def test_recall_degrades_to_empty_string_when_graph_fails():
    engine = FakeEngine(error=RuntimeError("graph unreadable"))
    assert orchestrator.Recall.recall("anything", engine) == ""


def test_default_constants():
    assert orchestrator.DEFAULT_MAX_TURNS == 3
    assert orchestrator.DEFAULT_MAX_OUTER_ITERATIONS == 20
    assert orchestrator.DEFAULT_MAX_REPLAN == 2


# --- PlanAgent ------------------------------------------------------------


def test_plan_agent_builds_tree_from_json():
    reply = plan_json(
        node_json("root", node_json("s1", max_turns=4), node_json("s2", strict=True))
    )
    tree = orchestrator.PlanAgent.run("Do X", "", llm=StubLlm(reply))

    assert [c.id for c in tree.root.children] == ["s1", "s2"]
    assert tree.root.children[0].max_turns == 4
    assert tree.root.children[1].strict is True


def test_plan_agent_prompt_carries_task_and_recall_context():
    llm = StubLlm(plan_json(node_json("r")))
    orchestrator.PlanAgent.run("Do X", "prior pitfall: Y", llm=llm)
    assert "Do X" in llm.prompts[0]
    assert "prior pitfall: Y" in llm.prompts[0]


def test_plan_agent_prompt_carries_blocking_feedback_on_replan():
    llm = StubLlm(plan_json(node_json("r")))
    orchestrator.PlanAgent.run("Do X", "", feedback=["no rollback step"], llm=llm)
    assert "no rollback step" in llm.prompts[0]


def test_plan_agent_retries_once_after_invalid_reply():
    llm = StubLlm("not json", plan_json(node_json("r")))
    tree = orchestrator.PlanAgent.run("r", "", llm=llm)
    assert tree.root.id == "r"
    assert len(llm.prompts) == 2


def test_plan_agent_retries_when_llm_raises():
    llm = StubLlm(ValueError("boom"), plan_json(node_json("r")))
    assert orchestrator.PlanAgent.run("r", "", llm=llm).root.id == "r"


def test_plan_agent_returns_none_after_two_failures():
    llm = StubLlm("nope", "still nope")
    assert orchestrator.PlanAgent.run("r", "", llm=llm) is None
    assert len(llm.prompts) == 2


def test_plan_agent_rejects_tree_deeper_than_two_levels():
    too_deep = plan_json(node_json("a", node_json("b", node_json("c", node_json("d")))))
    assert orchestrator.PlanAgent.run("r", "", llm=StubLlm(too_deep)) is None


def test_plan_agent_rejects_node_missing_required_field():
    broken = node_json("r")
    del broken["acceptance"]
    assert orchestrator.PlanAgent.run("r", "", llm=StubLlm(plan_json(broken))) is None


def test_plan_agent_ignores_non_positive_max_turns():
    reply = plan_json(node_json("r", max_turns=0))
    tree = orchestrator.PlanAgent.run("r", "", llm=StubLlm(reply))
    assert tree.root.max_turns is None


# --- CriticAgent ----------------------------------------------------------


def critique(mode, reply):
    return orchestrator.CriticAgent.run(
        "r", simple_tree(), "", mode=mode, llm=StubLlm(reply)
    )


def test_critic_advisory_reports_findings_and_never_blocks():
    report = critique("advisory", {"findings": ["note"], "blocking": ["defect"]})
    assert report.findings == ["note"]
    assert report.blocking == []
    assert report.mode == "advisory"


def test_critic_strict_reports_blocking_defects():
    report = critique("strict", {"findings": ["note"], "blocking": ["no rollback"]})
    assert report.blocking == ["no rollback"]
    assert report.mode == "strict"


def test_critic_prompt_names_mode_and_includes_plan():
    llm = StubLlm({"findings": [], "blocking": []})
    orchestrator.CriticAgent.run("r", simple_tree("s1"), "", mode="strict", llm=llm)
    assert "strict" in llm.prompts[0]
    assert '"s1"' in llm.prompts[0]


def test_critic_returns_empty_report_on_unusable_reply():
    report = critique("strict", "nope")
    assert (report.findings, report.blocking, report.mode) == ([], [], "strict")


def test_critic_returns_empty_report_when_llm_raises():
    report = orchestrator.CriticAgent.run(
        "r", simple_tree(), "", mode="strict", llm=StubLlm(RuntimeError("down"))
    )
    assert report.blocking == []


# --- dispatch -------------------------------------------------------------


@pytest.fixture
def stores(tmp_path):
    return MemoryEngine(tmp_path), SessionManager(tmp_path)


def turn(action: str) -> TurnOutput:
    return TurnOutput(thought="t", action=action, raw="")


@pytest.fixture
def fake_loop(monkeypatch):
    """Stand-in for agent_loop.run; attributes describe what the executor 'did'."""

    class Loop:
        calls: list[dict] = []
        turns = [turn('FINISH(answer="done")')]
        final_answer = "done"
        error: Exception | None = None

    def run(**kwargs):
        Loop.calls.append(kwargs)
        if Loop.error:
            raise Loop.error
        return RunResult(
            kwargs["session_id"], Loop.turns, Loop.final_answer, closed=True
        )

    Loop.calls = []
    monkeypatch.setattr(orchestrator.agent_loop, "run", run)
    return Loop


def leaf(node_id="s1", **kwargs) -> PlanNode:
    return PlanNode(
        id=node_id,
        title=node_id,
        task=f"do {node_id}",
        acceptance=f"{node_id} done",
        **kwargs,
    )


def test_dispatch_runs_agent_loop_in_fresh_session(stores, fake_loop):
    engine, sessions = stores
    result = orchestrator._dispatch_subtask(
        leaf(), memory_engine=engine, session_manager=sessions
    )

    call = fake_loop.calls[0]
    assert call["task"] == "do s1"
    assert call["max_turns"] == orchestrator.DEFAULT_MAX_TURNS
    assert call["memory_engine"] is engine
    assert call["session_manager"] is sessions
    assert sessions.loadSession(call["session_id"]) == []
    assert result == SubTaskResult("s1", call["session_id"], "done", "ok", 1)


def test_dispatch_honours_node_max_turns(stores, fake_loop):
    engine, sessions = stores
    orchestrator._dispatch_subtask(
        leaf(max_turns=4), memory_engine=engine, session_manager=sessions
    )
    assert fake_loop.calls[0]["max_turns"] == 4


def test_dispatch_reports_max_turns_when_executor_never_finished(stores, fake_loop):
    engine, sessions = stores
    fake_loop.turns = [turn("list_dir(path='.')")]
    result = orchestrator._dispatch_subtask(
        leaf(), memory_engine=engine, session_manager=sessions
    )
    assert result.status == "max_turns"


def test_dispatch_reports_failed_when_executor_raises(stores, fake_loop):
    engine, sessions = stores
    fake_loop.error = RuntimeError("tool crashed")
    result = orchestrator._dispatch_subtask(
        leaf(), memory_engine=engine, session_manager=sessions
    )
    assert (result.status, result.turns, result.final_answer) == ("failed", 0, "")


# --- records --------------------------------------------------------------


def edges_between(engine, src, tgt):
    data = json.loads(engine.graph_path.read_text())
    return [e for e in data["edges"] if (e["source"], e["target"]) == (src, tgt)]


def test_subtask_triplet_follows_record_contract(stores):
    engine, _ = stores
    result = SubTaskResult("audit_auth", "exec-1", "a" * 1000, "ok", 3)
    orchestrator._record_subtask_triplet(
        "Refactor auth to JWT", leaf("audit_auth"), result, engine
    )

    (edge,) = edges_between(engine, "refactor_auth_to_jwt", "audit_auth")
    assert edge["relation"] == "CONVERGES_TO"
    assert edge["guidance"] == "a" * 500
    assert edge["pitfalls"] == ""
    assert edge["condition"] == "audit_auth done"
    assert edge["provenance"]["source_session_id"] == "exec-1"


def test_subtask_triplet_notes_pitfall_for_unfinished_subtask(stores):
    engine, _ = stores
    result = SubTaskResult("s1", "exec-1", "", "max_turns", 8)
    orchestrator._record_subtask_triplet("root", leaf(), result, engine)

    (edge,) = edges_between(engine, "root", "s1")
    assert "max_turns" in edge["pitfalls"]


def test_plan_summary_links_root_task_to_end(stores):
    engine, _ = stores
    executions = [SubTaskResult("s1", "x", "ok", "ok", 2)]
    edge_id = orchestrator._record_plan_summary(
        "Refactor auth", executions, "ok", engine
    )

    (edge,) = edges_between(engine, "refactor_auth", "End")
    assert edge["relation"] == "LEADS_TO"
    assert edge["condition"] is None
    assert "ok" in edge["guidance"]
    assert edge["pitfalls"] == ""
    assert edge_id == "refactor_auth->End#LEADS_TO"


def test_plan_summary_lists_unfinished_subtasks_as_pitfalls(stores):
    engine, _ = stores
    executions = [
        SubTaskResult("s1", "x", "", "failed", 0),
        SubTaskResult("s2", "y", "fine", "ok", 2),
    ]
    orchestrator._record_plan_summary("r", executions, "partial", engine)

    (edge,) = edges_between(engine, "r", "End")
    assert "s1" in edge["pitfalls"] and "failed" in edge["pitfalls"]
    assert "s2" not in edge["pitfalls"]


# --- run() ----------------------------------------------------------------

OK_CRITIQUE = {"findings": [], "blocking": []}


@pytest.fixture
def dispatched(monkeypatch):
    """Replace the executor; `statuses` maps node id -> status (default ok)."""

    class Dispatch:
        nodes: list[str] = []
        turn_budgets: list[int] = []
        statuses: dict[str, str] = {}

    def fake(plan_node, *, memory_engine, session_manager, default_max_turns):
        Dispatch.nodes.append(plan_node.id)
        Dispatch.turn_budgets.append(default_max_turns)
        return SubTaskResult(
            plan_node.id,
            f"sess-{plan_node.id}",
            f"answer {plan_node.id}",
            Dispatch.statuses.get(plan_node.id, "ok"),
            2,
        )

    Dispatch.nodes = []
    Dispatch.turn_budgets = []
    Dispatch.statuses = {}
    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake)
    return Dispatch


def run(stores, *replies, approve=lambda plan, critique: True, **kwargs):
    engine, sessions = stores
    return orchestrator.run(
        "Refactor auth",
        memory_engine=engine,
        session_manager=sessions,
        on_human_approval=approve,
        llm=StubLlm(*replies),
        **kwargs,
    )


def two_step_plan() -> dict:
    return plan_json(node_json("root", node_json("s1"), node_json("s2")))


def test_run_executes_leaves_in_order_and_records_summary(stores, dispatched):
    result = run(stores, two_step_plan(), OK_CRITIQUE)

    assert result.status == "ok"
    assert dispatched.nodes == ["s1", "s2"]
    assert [e.plan_node_id for e in result.executions] == ["s1", "s2"]
    assert result.plan_summary_triplet_id == "refactor_auth->End#LEADS_TO"
    engine, _ = stores
    assert edges_between(engine, "refactor_auth", "s1")
    assert edges_between(engine, "refactor_auth", "End")


def test_run_executes_root_when_plan_has_no_children(stores, dispatched):
    result = run(stores, plan_json(node_json("root")), OK_CRITIQUE)
    assert dispatched.nodes == ["root"]
    assert result.status == "ok"


def test_run_executes_depth_two_leaves_not_their_parents(stores, dispatched):
    nested = plan_json(
        node_json("root", node_json("a", node_json("a1"), node_json("a2")), node_json("b"))
    )
    run(stores, nested, OK_CRITIQUE)
    assert dispatched.nodes == ["a1", "a2", "b"]


def test_run_is_partial_when_a_subtask_does_not_finish(stores, dispatched):
    dispatched.statuses["s1"] = "max_turns"
    result = run(stores, two_step_plan(), OK_CRITIQUE)
    assert result.status == "partial"
    assert dispatched.nodes == ["s1", "s2"]


def test_run_asks_human_exactly_once_with_plan_and_critique(stores, dispatched):
    seen = []

    def approve(plan, critique):
        seen.append((plan.root.id, critique.mode))
        return True

    run(stores, two_step_plan(), OK_CRITIQUE, approve=approve)
    assert seen == [("root", "advisory")]


def test_run_human_rejection_dispatches_nothing(stores, dispatched):
    result = run(
        stores, two_step_plan(), OK_CRITIQUE, approve=lambda plan, critique: False
    )
    assert (result.status, result.reason) == (
        "rejected_by_human",
        "human_rejected_root_plan",
    )
    assert result.executions == []
    assert dispatched.nodes == []


def test_run_treats_failing_approval_gate_as_rejection(stores, dispatched):
    def broken(plan, critique):
        raise EOFError

    result = run(stores, two_step_plan(), OK_CRITIQUE, approve=broken)
    assert result.status == "rejected_by_human"


def test_run_without_gate_executes_directly(stores, dispatched):
    result = run(stores, two_step_plan(), OK_CRITIQUE, approve=None)
    assert result.status == "ok"


def test_run_fails_when_plan_cannot_be_parsed(stores, dispatched):
    result = run(stores, "nope", "still nope")
    assert (result.status, result.reason) == ("failed", "plan_parse_error")
    assert result.executions == []


def test_run_passes_recall_context_to_planner(stores, dispatched, monkeypatch):
    monkeypatch.setattr(orchestrator.Recall, "recall", lambda task, engine: "PRIOR")
    engine, sessions = stores
    llm = StubLlm(two_step_plan(), OK_CRITIQUE)
    orchestrator.run(
        "t", memory_engine=engine, session_manager=sessions, llm=llm
    )
    assert "PRIOR" in llm.prompts[0]


def test_advisory_blocking_defects_do_not_halt(stores, dispatched):
    critique = {"findings": ["note"], "blocking": ["defect"]}
    result = run(stores, two_step_plan(), critique, strict=False)
    assert result.status == "ok"
    assert result.critique.blocking == []


def test_strict_replans_with_blocking_feedback_then_executes(stores, dispatched):
    engine, sessions = stores
    llm = StubLlm(
        two_step_plan(),
        {"findings": [], "blocking": ["no rollback"]},
        plan_json(node_json("root", node_json("rollback"))),
        OK_CRITIQUE,
    )
    result = orchestrator.run(
        "Refactor auth",
        strict=True,
        memory_engine=engine,
        session_manager=sessions,
        llm=llm,
    )
    assert result.status == "ok"
    assert dispatched.nodes == ["rollback"]
    assert "no rollback" in llm.prompts[2]


def test_strict_halts_when_replan_budget_is_exhausted(stores, dispatched):
    blocking = {"findings": [], "blocking": ["still wrong"]}
    plan_then_critique = [two_step_plan(), blocking] * 3  # initial plan + 2 replans
    result = run(stores, *plan_then_critique, strict=True, max_replan=2)
    assert (result.status, result.reason) == ("failed", "strict_replan_cap_reached")
    assert result.critique.blocking == ["still wrong"]
    assert result.executions == []
    assert dispatched.nodes == []


def test_strict_subtask_failure_halts_remaining_work(stores, dispatched):
    plan = plan_json(
        node_json("root", node_json("s1", strict=True), node_json("s2"))
    )
    dispatched.statuses["s1"] = "failed"
    result = run(stores, plan, OK_CRITIQUE)

    assert (result.status, result.reason) == ("failed", "strict_subtask_failed:s1")
    assert dispatched.nodes == ["s1"]
    assert result.plan_summary_triplet_id is not None


def test_run_stops_at_max_outer_iterations(stores, dispatched):
    result = run(stores, two_step_plan(), OK_CRITIQUE, max_outer_iterations=1)
    assert (result.status, result.reason) == (
        "max_outer_iterations",
        "max_outer_iterations",
    )
    assert dispatched.nodes == ["s1"]


def test_run_completing_exactly_at_the_cap_is_ok(stores, dispatched):
    result = run(stores, two_step_plan(), OK_CRITIQUE, max_outer_iterations=2)
    assert result.status == "ok"


# --- default human gate ----------------------------------------------------


class FakeStdin:
    def __init__(self, line="", tty=True):
        self.line = line
        self.tty = tty

    def isatty(self):
        return self.tty

    def readline(self):
        return self.line


def gate(monkeypatch, stdin):
    monkeypatch.setattr(orchestrator.sys, "stdin", stdin)
    report = CritiqueReport(["watch the tokens"], [], "advisory")
    return orchestrator.default_human_approval(simple_tree("s1", "s2"), report)


@pytest.mark.parametrize("answer, approved", [("y\n", True), ("YES\n", True), ("n\n", False), ("\n", False), ("", False)])
def test_default_gate_reads_yes_no(monkeypatch, answer, approved):
    assert gate(monkeypatch, FakeStdin(answer)) is approved


def test_default_gate_rejects_without_a_tty(monkeypatch):
    assert gate(monkeypatch, FakeStdin("y\n", tty=False)) is False


def test_default_gate_shows_plan_and_findings(monkeypatch, capsys):
    gate(monkeypatch, FakeStdin("n\n"))
    out = capsys.readouterr().out
    assert "s1" in out and "s2" in out and "watch the tokens" in out


# --- planner prompt framing --------------------------------------------------


def planner_prompt(task="How can I learn to code efficiently?"):
    llm = StubLlm(plan_json(node_json("r")))
    orchestrator.PlanAgent.run(task, "Start->End; skip Goal_Clarification", llm=llm)
    return llm.prompts[0]


def test_planner_prompt_treats_recall_as_background_not_requirements():
    prompt = planner_prompt()
    assert "background" in prompt.lower()
    assert "never" in prompt.lower() and "graph" in prompt.lower()


def test_planner_prompt_asks_for_steps_that_accomplish_the_task_itself():
    prompt = planner_prompt()
    assert "accomplish the user's task" in prompt


def test_planner_prompt_turns_ambiguity_into_clarifying_questions():
    assert "clarifying question" in planner_prompt()


def test_run_passes_turn_budget_to_every_dispatch(stores, dispatched):
    run(stores, two_step_plan(), OK_CRITIQUE, max_turns=5)
    assert dispatched.turn_budgets == [5, 5]


def test_run_defaults_turn_budget_to_default_max_turns(stores, dispatched):
    run(stores, two_step_plan(), OK_CRITIQUE)
    assert dispatched.turn_budgets == [orchestrator.DEFAULT_MAX_TURNS] * 2


def test_dispatch_uses_given_default_when_node_has_no_budget(stores, fake_loop):
    engine, sessions = stores
    orchestrator._dispatch_subtask(
        leaf(), memory_engine=engine, session_manager=sessions, default_max_turns=2
    )
    assert fake_loop.calls[0]["max_turns"] == 2


def test_run_reports_progress_for_each_subtask(stores, dispatched):
    lines: list[str] = []
    run(stores, two_step_plan(), OK_CRITIQUE, on_progress=lines.append)

    assert lines == [
        "[1/2] s1: s1",
        "      -> ok (2 turns)",
        "[2/2] s2: s2",
        "      -> ok (2 turns)",
    ]

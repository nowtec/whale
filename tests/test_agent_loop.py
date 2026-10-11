from __future__ import annotations

import json

import pytest

from agent_loop import RunResult, TurnOutput, run, run_with_guidance
from memory import MemoryEngine
from procedural_memory.refiner import EvolutionResult
from session import SessionManager


class _Solver:
    """Replies in order; repeats the last reply when exhausted."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class _FakeEngine:
    def __init__(self, guidance=None):
        self.calls: list[tuple] = []
        self.guidance = guidance or (lambda n: f"G-{n}")
        self.retrieved: list[list[str]] = []
        self.turns: list[tuple] = []

    def retrieve_guidance(self, task, recent_actions):
        self.retrieved.append(list(recent_actions))
        self.calls.append(("retrieve",))
        return self.guidance(len(self.retrieved))

    def record_turn(self, session_id, thought, action, observation):
        self.turns.append((session_id, thought, action, observation))
        self.calls.append(("record",))
        return []

    def evolve_from_trajectory(self, session_id, **kw):
        self.calls.append(("evolve", session_id))
        return EvolutionResult(1, True, 1.0, 1.0, 0, 0, None)


@pytest.fixture
def sm(tmp_path):
    return SessionManager(tmp_path)


def _go(sm, solver, engine, task="do it", max_turns=3, **kw):
    session = sm.initSession()
    result = run(
        task,
        session.id,
        max_turns,
        solver=solver,
        memory_engine=engine,
        session_manager=sm,
        **kw,
    )
    return session, result


@pytest.mark.parametrize(
    "msg, thought, action",
    [
        ("Thought: look it up\nAction: search(q=1)", "look it up", "search(q=1)"),
        ("Thought: only thinking", "only thinking", ""),
        ("Thought: line one\nline two\nAction: read(id=2)", "line one\nline two", "read(id=2)"),
        ("Action: lone()", "", "lone()"),
        ("hello there", "", ""),
        ("", "", ""),
        ("Thought: t\nAction: a()\nObservation: fake\nAction: b()", "t", "a()"),
    ],
)
def test_from_message_variants(msg, thought, action):
    turn = TurnOutput.from_message(msg)
    assert (turn.thought, turn.action, turn.raw) == (thought, action, msg)


def test_runs_k_turns_and_retrieves_k_times(sm):
    engine = _FakeEngine()
    solver = _Solver("Thought: t\nAction: search(q=1)")
    session, result = _go(sm, solver, engine, max_turns=3)

    assert isinstance(result, RunResult)
    assert result.session_id == session.id and result.closed is True
    assert len(result.turns) == 3
    assert [c for c in engine.calls if c[0] == "retrieve"] == [("retrieve",)] * 3
    assert len(engine.turns) == 3
    roles = [m["role"] for m in sm.loadSession(session.id)]
    assert roles == ["user"] + ["assistant", "tool"] * 3
    assert "closed_at" in json.loads(session.path.read_text())


def test_retrieve_gets_last_three_actions(sm):
    engine = _FakeEngine()
    replies = [f"Thought: t\nAction: a{i}()" for i in range(1, 6)]
    _go(sm, _Solver(*replies), engine, max_turns=5)
    assert engine.retrieved[0] == []
    assert engine.retrieved[1] == ["a1()"]
    assert engine.retrieved[4] == ["a2()", "a3()", "a4()"]


def test_finish_stops_early(sm):
    engine = _FakeEngine()
    solver = _Solver("Thought: a\nAction: search(q=1)", "Thought: done\nAction: FINISH()")
    _, result = _go(sm, solver, engine, max_turns=5)
    assert len(result.turns) == 2
    assert result.final_answer == "done"
    assert len(engine.turns) == 2


def test_done_is_case_insensitive(sm):
    engine = _FakeEngine()
    _, result = _go(sm, _Solver("Thought: x\nAction: done()"), engine, max_turns=5)
    assert len(result.turns) == 1


def test_guidance_in_solver_prompt(sm):
    engine = _FakeEngine(guidance=lambda n: "" if n == 1 else f"G-{n}")
    solver = _Solver("Thought: t\nAction: search(q=1)")
    _go(sm, solver, engine, max_turns=2)
    assert "Procedural Graph Guidance: (none)" in solver.prompts[0]
    assert "Procedural Graph Guidance: G-2" in solver.prompts[1]
    assert solver.prompts[0].rstrip().endswith("Thought:")
    assert "user: do it" in solver.prompts[0]
    assert "tool: [stub-obs] search(q=1)" in solver.prompts[1]


def test_stub_observation_default_and_custom(sm):
    engine = _FakeEngine()
    _, _ = _go(sm, _Solver("Thought: t\nAction: search(q=1)"), engine, max_turns=1)
    assert engine.turns[0][3] == "[stub-obs] search(q=1)"

    engine2 = _FakeEngine()
    seen = []
    _go(
        sm,
        _Solver("Thought: t\nAction: search(q=1)"),
        engine2,
        max_turns=1,
        on_observation=lambda action: seen.append(action) or "REAL-OBS",
    )
    assert seen == ["search(q=1)"]
    assert engine2.turns[0][3] == "REAL-OBS"


@pytest.mark.parametrize("reply", [None, "", "hello"])
def test_solver_garbage_does_not_crash_or_write_triplets(tmp_path, reply):
    manager = SessionManager(tmp_path)
    engine = MemoryEngine(tmp_path)
    _go(manager, _Solver(reply), engine, max_turns=2)
    assert not engine.graph_path.exists()


def test_close_session_after_all_turns_and_evolve_flag(sm):
    engine = _FakeEngine()
    session, _ = _go(sm, _Solver("Thought: t\nAction: a()"), engine, max_turns=2)
    assert all(c[0] != "evolve" for c in engine.calls)

    engine = _FakeEngine()
    session, _ = _go(
        sm,
        _Solver("Thought: t\nAction: a()"),
        engine,
        max_turns=2,
        evolve_on_close=True,
    )
    kinds = [c[0] for c in engine.calls]
    assert kinds.count("evolve") == 1
    assert kinds[-1] == "evolve"
    assert engine.calls[-1] == ("evolve", session.id)


def test_final_answer_is_last_nonempty_thought(sm):
    engine = _FakeEngine()
    solver = _Solver("Thought: first\nAction: a()", "Action: b()")
    _, result = _go(sm, solver, engine, max_turns=2)
    assert result.final_answer == "first"


def test_run_with_guidance_defaults_to_single_turn(sm):
    engine = _FakeEngine()
    result = run_with_guidance(
        "task",
        solver=_Solver("Thought: t\nAction: a()"),
        memory_engine=engine,
        session_manager=sm,
    )
    assert len(result.turns) == 1
    assert [s["id"] for s in sm.listSessions()] == [result.session_id]


def test_end_to_end_with_real_memory_engine(tmp_path):
    manager = SessionManager(tmp_path)
    prompts: list[str] = []
    engine = MemoryEngine(
        tmp_path, llm=lambda p: prompts.append(p) or {"guidance": "from graph"}
    )

    first = _Solver(
        "Thought: look\nAction: search(q=1)", "Thought: done\nAction: FINISH()"
    )
    _go(manager, first, engine, max_turns=3)
    edges = {
        (e["source"], e["target"])
        for e in json.loads(engine.graph_path.read_text())["edges"]
    }
    assert {("Start", "search"), ("search", "End")} <= edges

    second = _Solver("Thought: again\nAction: FINISH()")
    _go(manager, second, engine, max_turns=1)
    assert "Procedural Graph Guidance: from graph" in second.prompts[0]
    assert "[Start] → [search]" in prompts[-1]


def test_turns_without_action_are_not_in_recent_actions(sm):
    engine = _FakeEngine()
    solver = _Solver(
        "Thought: a\nAction: a1()",
        "Thought: just thinking",
        "Thought: b\nAction: a2()",
        "Thought: c\nAction: a3()",
    )
    _go(sm, solver, engine, max_turns=4)
    assert engine.retrieved[2] == ["a1()"]
    assert engine.retrieved[3] == ["a1()", "a2()"]


@pytest.mark.parametrize(
    "action, expected",
    [
        ('FINISH(answer="Yes, I can code.")', "Yes, I can code."),
        ("DONE(answer='Sure (really)')", "Sure (really)"),
        ("FINISH(Yes I can)", "Yes I can"),
        ("FINISH()", "my thought"),
    ],
)
def test_final_answer_prefers_finish_argument(sm, action, expected):
    solver = _Solver(f"Thought: my thought\nAction: {action}")
    _, result = _go(sm, solver, _FakeEngine(), max_turns=3)
    assert result.final_answer == expected


def test_system_prompt_names_the_finish_action(sm):
    solver = _Solver("Thought: t\nAction: FINISH()")
    _go(sm, solver, _FakeEngine(), max_turns=1)
    assert "FINISH(answer=" in solver.prompts[0]

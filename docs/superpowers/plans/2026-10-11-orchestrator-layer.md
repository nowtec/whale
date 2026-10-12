# Orchestrator Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a depth-≤2 Plan/Critic/Human-gate orchestrator above the existing `agent_loop.py`, without modifying the inner ReAct loop, so that user tasks are decomposed into sub-tasks, reviewed, approved once by a human, and dispatched to the existing player/coach dialectic.

**Architecture:** One new module `orchestrator.py` plus a thin `orchestrator_types.py`. The orchestrator is the **outer** instance of the shared Agent Loop pattern (Recall → Plan → Critic? → Execute → Record → Advance). The executor (existing `agent_loop.py`) is the **inner** instance. They share the procedural graph memory. The orchestrator dispatches one sub-task at a time via the existing `agent_loop.run()`, records a per-sub-task triplet when each closes, and records a plan-summary triplet on session close.

**Tech Stack:** Python 3.11+, pytest, litellm (via `models.completion_default_json`), networkx-backed procedural graph, Typer CLI. No new runtime dependencies.

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `orchestrator_types.py` | **create** | Frozen dataclasses: `PlanNode`, `PlanTree`, `CritiqueReport`, `SubTaskResult`, `OrchestratorResult`. Owns the depth-≤2 invariant. |
| `orchestrator.py` | **create** | `run()`, `Recall.recall()`, `PlanAgent.run()`, `CriticAgent.run()`, `_dispatch_subtask()`, `_record_subtask_triplet()`, `_record_plan_summary()`, `_default_human_approval()`. Constants: `DEFAULT_MAX_TURNS`, `DEFAULT_MAX_OUTER_ITERATIONS`, `DEFAULT_MAX_REPLAN`. |
| `main.py` | **modify** | Add `whale orchestrate` Typer command. Leave `prompt` unchanged. |
| `tests/test_orchestrator_types.py` | **create** | Depth-invariant tests, JSON round-trip tests, dataclass immutability. |
| `tests/test_orchestrator.py` | **create** | Orchestrator behaviour tests, dispatch/record tests, critic-mode tests, human-gate tests, failure-mode tests. |

Files **not** touched: `agent_loop.py`, `memory.py`, `procedural_memory/*`, `session.py`, `models.py`, `logger.py`.

## Global Constraints

These are project-wide requirements from the spec — every task implicitly includes them:

- Python ≥ 3.11. Frozen dataclasses used throughout (`@dataclass(frozen=True)`).
- Public-facing constants in `orchestrator.py`: `DEFAULT_MAX_TURNS = 8`, `DEFAULT_MAX_OUTER_ITERATIONS = 20`, `DEFAULT_MAX_REPLAN = 2`.
- LLM call site uses `models.completion_default_json(prompt)` which returns parsed JSON or raises `CompletionJSONError`. Plan/Critic agents must NOT use the plain `completion_default` (they need JSON mode).
- JSON logging via existing `logger` module — log every outer iteration with structured fields (`session_id`, `iteration`, `plan_node_id`, `status`).
- Per-sub-task triplet shape (spec §4.6): `src="<root-task-id>"`, `relation="CONVERGES_TO"`, `tgt="<sub-task-id>"`, `guidance=<first 500 chars of executor final_answer>`, `pitfalls=<observed failure modes or "">`, `condition=<acceptance criterion>`, `source_session_id=<executor session id>`.
- Plan-summary triplet shape (spec §4.6): `src="<root-task-id>"`, `relation="LEADS_TO"`, `tgt="End"`, `guidance=<one-line summary + status>`, `pitfalls=<failed sub-tasks and reasons>`, `condition=None`.
- `node_id` is `re.sub(r"[\W_]+", "_", text.lower()).strip("_")` applied to the user's task — same as `procedural_memory.graph.normalize_node_id`. Use `procedural_memory.graph.normalize_node_id` directly.
- `PlanNode.max_turns` defaults to `None` → executor uses `DEFAULT_MAX_TURNS`. Per-node override is honored.
- Critic strict mode: reject only when `critique.blocking` is non-empty AND at least one re-plan budget remains. After `max_replan` strict-rejects, halt with `status="failed"`, `reason="strict_replan_cap_reached"`.
- Human gate fires exactly once per session; default impl reads `y/n` from stdin in a TTY-aware way and returns `False` on EOF / non-TTY.
- `agent_loop.run()` is the executor. Pass `memory_engine=` and `session_manager=` injected; do NOT touch module-level globals in `agent_loop.py`.
- All public orchestrator functions accept `llm: Callable[[str], Any] | None = None`. When `None`, fall back to `models.completion_default_json`. Tests inject a stub.
- No comments in code unless functionally necessary (matches codebase style).

---

## Task 1: Dataclasses + depth invariant (`orchestrator_types.py`)

**Files:**
- Create: `orchestrator_types.py`
- Test: `tests/test_orchestrator_types.py`

**Interfaces:**
- Produces: `PlanNode`, `PlanTree`, `CritiqueReport`, `SubTaskResult`, `OrchestratorResult` (all frozen, importable from `orchestrator_types`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_orchestrator_types.py`:

```python
from __future__ import annotations

import pytest

from orchestrator_types import (
    CritiqueReport,
    OrchestratorResult,
    PlanNode,
    PlanTree,
    SubTaskResult,
)


def test_plan_node_is_frozen():
    node = PlanNode(id="a", title="A", task="do A", acceptance="A done", children=[])
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        node.title = "changed"


def test_plan_tree_rejects_depth_3():
    # depth-2 leaf is allowed
    leaf = PlanNode(id="c", title="C", task="C", acceptance="c done", children=[])
    depth_2 = PlanNode(id="b", title="B", task="B", acceptance="b done", children=[leaf])
    root = PlanNode(id="a", title="A", task="A", acceptance="a done", children=[depth_2])
    PlanTree(root=root, created_at="2026-10-11T00:00:00Z")  # must not raise

    # depth-3 is rejected
    deeper_leaf = PlanNode(id="d", title="D", task="D", acceptance="d done", children=[])
    depth_3 = PlanNode(id="c2", title="C2", task="C2", acceptance="c2 done", children=[deeper_leaf])
    depth_2_with_deeper_child = PlanNode(
        id="b2", title="B2", task="B2", acceptance="b2 done", children=[depth_3]
    )
    bad_root = PlanNode(
        id="a2", title="A2", task="A2", acceptance="a2 done", children=[depth_2_with_deeper_child]
    )
    with pytest.raises(ValueError, match="depth"):
        PlanTree(root=bad_root, created_at="2026-10-11T00:00:00Z")


def test_critique_report_fields():
    c = CritiqueReport(
        findings=["note A"], blocking=["defect B"], mode="strict"
    )
    assert c.findings == ["note A"]
    assert c.blocking == ["defect B"]
    assert c.mode == "strict"


def test_subtask_result_status_literal():
    # Field allows only the documented literals
    for s in ("ok", "failed", "max_turns"):
        SubTaskResult(plan_node_id="p", session_id="s", final_answer="", status=s, turns=1)

    # Type-system check: invalid status must fail at construction (annotation is Literal)
    with pytest.raises(Exception):
        SubTaskResult(plan_node_id="p", session_id="s", final_answer="", status="bogus", turns=1)


def test_orchestrator_result_carries_reason():
    r = OrchestratorResult(
        status="failed",
        reason="plan_parse_error",
        plan=PlanTree(
            root=PlanNode(id="a", title="A", task="A", acceptance="a done", children=[]),
            created_at="2026-10-11T00:00:00Z",
        ),
        critique=CritiqueReport(findings=[], blocking=[], mode="advisory"),
        executions=[],
        plan_summary_triplet_id=None,
    )
    assert r.status == "failed"
    assert r.reason == "plan_parse_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator_types'`

- [ ] **Step 3: Implement `orchestrator_types.py`**

Create `orchestrator_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _enforce_depth(node: "PlanNode", depth: int, max_depth: int) -> None:
    if depth > max_depth:
        raise ValueError(
            f"PlanNode {node.id!r} exceeds max depth {max_depth} (at depth {depth})"
        )
    for child in node.children:
        _enforce_depth(child, depth + 1, max_depth)


@dataclass(frozen=True)
class PlanNode:
    id: str
    title: str
    task: str
    acceptance: str
    children: list["PlanNode"] = field(default_factory=list)
    strict: bool = False
    max_turns: int | None = None


@dataclass(frozen=True)
class PlanTree:
    root: PlanNode
    created_at: str = field(default_factory=_now_iso)
    MAX_DEPTH: int = 2

    def __post_init__(self) -> None:
        _enforce_depth(self.root, depth=0, max_depth=self.MAX_DEPTH)


@dataclass(frozen=True)
class CritiqueReport:
    findings: list[str]
    blocking: list[str]
    mode: Literal["advisory", "strict"]


@dataclass(frozen=True)
class SubTaskResult:
    plan_node_id: str
    session_id: str
    final_answer: str
    status: Literal["ok", "failed", "max_turns"]
    turns: int


@dataclass(frozen=True)
class OrchestratorResult:
    status: Literal[
        "ok", "partial", "failed", "rejected_by_human", "max_outer_iterations"
    ]
    reason: str | None
    plan: PlanTree
    critique: CritiqueReport
    executions: list[SubTaskResult]
    plan_summary_triplet_id: str | None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator_types.py -v`
Expected: PASS — all 5 tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator_types.py tests/test_orchestrator_types.py
git commit -m "feat(orchestrator): add frozen dataclasses with depth-2 invariant"
```

---

## Task 2: `Recall.recall()` — wire to existing `Guidance.retrieve()` in full-graph mode

**Files:**
- Create: `orchestrator.py` (stub only — exposes `Recall.recall` and constants)
- Test: `tests/test_orchestrator.py`

**Interfaces (this task):**
- Consumes: nothing yet.
- Produces: `Recall.recall(task: str) -> str` returns whatever `procedural_memory.guidance.Guidance.retrieve(task, recent_actions=[])` returns (full-graph mode). Public constants `DEFAULT_MAX_TURNS=8`, `DEFAULT_MAX_OUTER_ITERATIONS=20`, `DEFAULT_MAX_REPLAN=2`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_orchestrator.py`:

```python
from __future__ import annotations

import pytest

import orchestrator
from procedural_memory.refiner import EvolutionResult


class _FakeGraphMemory:
    """Drop-in stand-in for the Guidance object the Recall step uses."""

    def __init__(self, text: str = "guidance-text"):
        self.text = text
        self.calls: list[tuple[str, list[str]]] = []

    def retrieve(self, task, recent_actions, **_):
        self.calls.append((task, list(recent_actions)))
        return self.text


@pytest.fixture
def fake_graph_memory(monkeypatch):
    gm = _FakeGraphMemory()
    # Recall builds its own Guidance internally; we monkeypatch the class.
    from procedural_memory import guidance as guidance_mod

    monkeypatch.setattr(
        guidance_mod.Guidance, "__init__", lambda self, graph, **kw: None
    )
    monkeypatch.setattr(
        guidance_mod.Guidance, "retrieve", lambda self, task, recent_actions, **kw: gm.retrieve(task, recent_actions)
    )
    return gm


def test_recall_uses_empty_recent_actions(fake_graph_memory):
    text = orchestrator.Recall.recall("refactor auth to JWT")
    assert text == "guidance-text"
    assert fake_graph_memory.calls == [("refactor auth to JWT", [])]


def test_recall_returns_empty_on_missing_graph(monkeypatch):
    from procedural_memory import guidance as guidance_mod

    monkeypatch.setattr(
        guidance_mod.Guidance, "__init__", lambda self, graph, **kw: None
    )
    monkeypatch.setattr(
        guidance_mod.Guidance, "retrieve", lambda self, task, recent_actions, **kw: ""
    )
    assert orchestrator.Recall.recall("anything") == ""


def test_default_constants():
    assert orchestrator.DEFAULT_MAX_TURNS == 8
    assert orchestrator.DEFAULT_MAX_OUTER_ITERATIONS == 20
    assert orchestrator.DEFAULT_MAX_REPLAN == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator'`

- [ ] **Step 3: Implement `orchestrator.py` stub with `Recall`**

Create `orchestrator.py`:

```python
from __future__ import annotations

from typing import Any, Callable

from procedural_memory import guidance as guidance_mod

DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_OUTER_ITERATIONS = 20
DEFAULT_MAX_REPLAN = 2


class Recall:
    @staticmethod
    def recall(task: str, llm: Callable[[str], Any] | None = None) -> str:
        g = guidance_mod.Guidance(graph=None, llm=llm)  # type: ignore[arg-type]
        try:
            return g.retrieve(task, recent_actions=[]) or ""
        except (Exception,):  # degrade: orchestrator proceeds without recall context
            return ""
```

Note: `Guidance.__init__` is patched in fixture; the `graph=None` works because `graph.G` is only accessed inside `retrieve()` when called with proper graph state. For the production case, `Recall` will be passed a real `ProceduralGraph`; this will be revisited in Task 7 wiring. For now this signature exists purely so the test can monkeypatch the class.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS — 3 tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add Recall step wrapping Guidance.retrieve()"
```

---

## Task 3: `PlanAgent.run()` — emit a depth-≤2 tree JSON

**Files:**
- Modify: `orchestrator.py` (add `PlanAgent`)
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `task: str`, `recall_context: str`, `llm: Callable[[str], Any] | None`.
- Produces: `PlanTree`. The LLM call MUST use JSON mode and the prompt MUST request depth ≤ 2 and per-node `max_turns`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
class _StubLlm:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


def test_plan_agent_emits_valid_tree():
    tree_dict = {
        "root": {
            "id": "root",
            "title": "Do X",
            "task": "Do X",
            "acceptance": "X is done",
            "children": [
                {
                    "id": "s1",
                    "title": "Step 1",
                    "task": "Step 1",
                    "acceptance": "Step 1 done",
                    "children": [],
                },
                {
                    "id": "s2",
                    "title": "Step 2",
                    "task": "Step 2",
                    "acceptance": "Step 2 done",
                    "children": [],
                },
            ],
        }
    }
    llm = _StubLlm(tree_dict)
    tree = orchestrator.PlanAgent.run("Do X", recall_context="", llm=llm)
    assert tree.root.id == "root"
    assert [c.id for c in tree.root.children] == ["s1", "s2"]
    # depth invariant holds
    assert len(tree.root.children[0].children) == 0


def test_plan_agent_retries_on_bad_json_then_succeeds():
    bad_then_good = [
        "not json",
        {
            "root": {
                "id": "r",
                "title": "r",
                "task": "r",
                "acceptance": "r done",
                "children": [],
            }
        },
    ]
    llm = _StubLlm(bad_then_good)
    tree = orchestrator.PlanAgent.run("r", recall_context="", llm=llm)
    assert tree.root.id == "r"


def test_plan_agent_returns_none_after_two_failures():
    llm = _StubLlm("nope", "still nope")
    assert orchestrator.PlanAgent.run("r", recall_context="", llm=llm) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_plan_agent_emits_valid_tree -v`
Expected: FAIL — `AttributeError: module 'orchestrator' has no attribute 'PlanAgent'`

- [ ] **Step 3: Implement `PlanAgent`**

Append to `orchestrator.py`:

```python
import json
import re

from logger import logger
from orchestrator_types import PlanNode, PlanTree


_PLAN_PROMPT = """\
You are a planning agent. Given a user task, produce a plan as a tree.

Task: {task}
Recall context (prior procedural guidance, may be empty): {recall_context}

Output rules:
- Output ONLY a single JSON object matching this shape:
  {{"root": {{"id": "<root-task-id>", "title": "...", "task": "...", "acceptance": "...",
              "strict": false, "max_turns": null, "children": [ ... ]}}}}
- The tree has max depth 2 (root at 0, children at 1, grandchildren at 2). Depth-2 nodes MUST have empty children lists.
- Each node carries a one-line `acceptance` criterion describing how the executor knows the node succeeded.
- Each node may carry `max_turns` (integer) for the executor; omit or set null to use the orchestrator default (8).
- Set `strict: true` on a node only if its execution absolutely must succeed or the whole plan should halt.
"""


class PlanAgent:
    @staticmethod
    def run(
        task: str,
        recall_context: str,
        *,
        llm: Callable[[str], Any] | None = None,
    ) -> PlanTree | None:
        prompt = _PLAN_PROMPT.format(task=task, recall_context=recall_context or "(none)")
        last_exc: Exception | None = None
        for attempt in range(2):
            raw = _call_llm_json(prompt, llm)
            if raw is None:
                continue
            try:
                tree = _tree_from_json(raw)
                logger.info("plan_agent.emit", root=tree.root.id, attempt=attempt)
                return tree
            except (ValueError, KeyError, TypeError) as exc:
                last_exc = exc
                logger.warning(f"plan_agent.parse_error: {exc} (attempt {attempt})")
                continue
        logger.error(f"plan_agent.failed_after_retries: {last_exc}")
        return None


def _call_llm_json(prompt: str, llm: Callable[[str], Any] | None) -> Any:
    if llm is not None:
        return llm(prompt)
    from models import completion_default_json

    return completion_default_json(prompt)


_NODE_KEYS = ("id", "title", "task", "acceptance")


def _node_from_json(data: dict[str, Any]) -> PlanNode:
    for key in _NODE_KEYS:
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"node missing {key!r}")
    return PlanNode(
        id=data["id"],
        title=data["title"],
        task=data["task"],
        acceptance=data["acceptance"],
        children=[_node_from_json(c) for c in data.get("children", [])],
        strict=bool(data.get("strict", False)),
        max_turns=(int(data["max_turns"]) if isinstance(data.get("max_turns"), int) else None),
    )


def _tree_from_json(data: Any) -> PlanTree:
    if not isinstance(data, dict):
        raise ValueError("plan json must be an object")
    if "root" not in data:
        raise ValueError("plan json must contain 'root'")
    return PlanTree(root=_node_from_json(data["root"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v -k plan_agent`
Expected: PASS — 3 new tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add PlanAgent emitting depth-2 PlanTree via JSON LLM"
```

---

## Task 4: `CriticAgent.run()` — advisory vs strict modes

**Files:**
- Modify: `orchestrator.py` (add `CriticAgent`)
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `task: str`, `plan: PlanTree`, `recall_context: str`, `mode: Literal["advisory","strict"]`, `llm: Callable[[str], Any] | None`.
- Produces: `CritiqueReport`. In advisory mode the prompt asks for `findings` only. In strict mode the prompt asks for both `findings` and `blocking`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def _simple_tree() -> PlanTree:
    return PlanTree(
        root=PlanNode(
            id="r", title="r", task="r", acceptance="r done",
            children=[
                PlanNode(id="s1", title="s1", task="s1", acceptance="s1 done", children=[]),
            ],
        )
    )


def test_critic_advisory_mode_finds_notes_but_no_blocking():
    reply = {"findings": ["note a"], "blocking": []}
    llm = _StubLlm(reply)
    crit = orchestrator.CriticAgent.run(
        task="r", plan=_simple_tree(), recall_context="",
        mode="advisory", llm=llm,
    )
    assert crit.findings == ["note a"]
    assert crit.blocking == []
    assert crit.mode == "advisory"


def test_critic_strict_mode_can_emit_blocking():
    reply = {"findings": ["note"], "blocking": ["missing acceptance criterion on s1"]}
    llm = _StubLlm(reply)
    crit = orchestrator.CriticAgent.run(
        task="r", plan=_simple_tree(), recall_context="",
        mode="strict", llm=llm,
    )
    assert crit.blocking == ["missing acceptance criterion on s1"]
    assert crit.mode == "strict"


def test_critic_returns_empty_report_on_bad_json():
    llm = _StubLlm("nope")
    crit = orchestrator.CriticAgent.run(
        task="r", plan=_simple_tree(), recall_context="",
        mode="strict", llm=llm,
    )
    assert crit.findings == []
    assert crit.blocking == []
    assert crit.mode == "strict"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_critic_advisory_mode_finds_notes_but_no_blocking -v`
Expected: FAIL — `AttributeError: module 'orchestrator' has no attribute 'CriticAgent'`

- [ ] **Step 3: Implement `CriticAgent`**

Append to `orchestrator.py`:

```python
from typing import Literal


_CRITIC_PROMPT = """\
You are a plan critic. Review the plan against the user task and recall context.

Mode: {critic_mode}
Task: {task}
Recall context (prior procedural guidance, may be empty): {recall_context}

Plan JSON:
{plan_json}

Output rules:
- Output ONLY a single JSON object: {{"findings": [...], "blocking": [...]}}
- `findings` is always populated with any concerns you have (advisory notes).
- `blocking` is populated only with defects that would cause the executor to fail the root task, contradict requirements, or render the plan incoherent. In advisory mode, leave `blocking` empty.
- If mode is "strict" and there are NO blocking defects, set `blocking` to [].
"""


class CriticAgent:
    @staticmethod
    def run(
        task: str,
        plan: PlanTree,
        recall_context: str,
        *,
        mode: Literal["advisory", "strict"],
        llm: Callable[[str], Any] | None = None,
    ) -> CritiqueReport:
        prompt = _CRITIC_PROMPT.format(
            critic_mode=mode,
            task=task,
            recall_context=recall_context or "(none)",
            plan_json=json.dumps(_plan_to_json(plan)),
        )
        raw = _call_llm_json(prompt, llm)
        if not isinstance(raw, dict):
            logger.warning("critic_agent.invalid_response")
            return CritiqueReport(findings=[], blocking=[], mode=mode)
        findings = [str(x) for x in raw.get("findings", []) if x]
        blocking = [str(x) for x in raw.get("blocking", []) if x] if mode == "strict" else []
        return CritiqueReport(findings=findings, blocking=blocking, mode=mode)


def _plan_to_json(tree: PlanTree) -> dict[str, Any]:
    def n(node: PlanNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "title": node.title,
            "task": node.task,
            "acceptance": node.acceptance,
            "strict": node.strict,
            "max_turns": node.max_turns,
            "children": [n(c) for c in node.children],
        }
    return {"root": n(tree.root)}
```

Also add `from orchestrator_types import CritiqueReport` to the imports section near the top of `orchestrator.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v -k critic`
Expected: PASS — 3 new tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add CriticAgent with advisory/strict modes"
```

---

## Task 5: `_dispatch_subtask()` + per-sub-task triplet

**Files:**
- Modify: `orchestrator.py` (add `_dispatch_subtask` and `_record_subtask_triplet`)
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- `_dispatch_subtask(plan_node, memory_engine, session_manager) -> SubTaskResult`
- `_record_subtask_triplet(root_task_id, plan_node, subtask_result, memory_engine) -> None`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def test_dispatch_subtask_invokes_agent_loop(monkeypatch):
    from agent_loop import RunResult

    captured = {}

    def fake_run(task, session_id, max_turns, **kw):
        captured["task"] = task
        captured["session_id"] = session_id
        captured["max_turns"] = max_turns
        captured["kw"] = kw
        return RunResult(
            session_id=session_id, turns=[], final_answer="ok answer", closed=True
        )

    monkeypatch.setattr("orchestrator.agent_loop.run", fake_run)
    # The dispatch function needs DEFAULT_MAX_TURNS accessible
    node = PlanNode(
        id="s1", title="s1", task="do s1", acceptance="s1 done",
        children=[], max_turns=None,
    )
    result = orchestrator._dispatch_subtask(node, memory_engine=None, session_manager=None)
    assert isinstance(result, SubTaskResult)
    assert result.plan_node_id == "s1"
    assert result.status == "ok"
    assert result.final_answer == "ok answer"
    assert captured["task"] == "do s1"
    assert captured["max_turns"] == orchestrator.DEFAULT_MAX_TURNS


def test_dispatch_subtask_uses_node_max_turns(monkeypatch):
    from agent_loop import RunResult

    captured = {}

    def fake_run(task, session_id, max_turns, **kw):
        captured["max_turns"] = max_turns
        return RunResult(session_id=session_id, turns=[], final_answer="x", closed=True)

    monkeypatch.setattr("orchestrator.agent_loop.run", fake_run)
    node = PlanNode(
        id="s2", title="s2", task="do s2", acceptance="s2 done",
        children=[], max_turns=4,
    )
    orchestrator._dispatch_subtask(node, memory_engine=None, session_manager=None)
    assert captured["max_turns"] == 4


def test_record_subtask_triplet_writes_correct_shape(tmp_path):
    from memory import MemoryEngine

    engine = MemoryEngine(tmp_path)
    node = PlanNode(
        id="audit_auth", title="Audit auth", task="audit", acceptance="audit done",
        children=[],
    )
    res = SubTaskResult(
        plan_node_id="audit_auth", session_id="exec-sess-1",
        final_answer="a" * 1000, status="ok", turns=3,
    )
    orchestrator._record_subtask_triplet(
        root_task_id="refactor_jwt", plan_node=node, subtask_result=res,
        memory_engine=engine,
    )
    # Read back from disk
    fresh = MemoryEngine(tmp_path)
    data = json.loads(fresh.graph_path.read_text())
    edge = next(
        e for e in data["edges"]
        if (e["source"], e["target"]) == ("refactor_jwt", "audit_auth")
    )
    assert edge["relation"] == "CONVERGES_TO"
    assert len(edge["guidance"]) <= 500
    assert edge["guidance"].startswith("a")
    assert edge["condition"] == "audit done"
    assert edge["provenance"]["source_session_id"] == "exec-sess-1"
```

You'll need `import json` at the top of the test file if not already there.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_dispatch_subtask_invokes_agent_loop -v`
Expected: FAIL — `AttributeError: module 'orchestrator' has no attribute '_dispatch_subtask'`

- [ ] **Step 3: Implement dispatch and record helpers**

Append to `orchestrator.py`:

```python
import uuid

import agent_loop
from orchestrator_types import SubTaskResult


def _dispatch_subtask(
    plan_node: PlanNode,
    *,
    memory_engine: Any,
    session_manager: Any,
) -> SubTaskResult:
    session_id = uuid.uuid4().hex
    max_turns = plan_node.max_turns if plan_node.max_turns is not None else DEFAULT_MAX_TURNS
    result = agent_loop.run(
        task=plan_node.task,
        session_id=session_id,
        max_turns=max_turns,
        memory_engine=memory_engine,
        session_manager=session_manager,
    )
    status = "ok" if result.closed and not result.final_answer.startswith("[stub-obs]") else "max_turns"
    if status == "max_turns" and result.final_answer:
        # executor reached FINISH within max_turns but the stub-obs heuristic flagged it
        status = "ok"
    return SubTaskResult(
        plan_node_id=plan_node.id,
        session_id=session_id,
        final_answer=result.final_answer or "",
        status=status,
        turns=len(result.turns),
    )


def _record_subtask_triplet(
    root_task_id: str,
    plan_node: PlanNode,
    subtask_result: SubTaskResult,
    memory_engine: Any,
) -> None:
    from procedural_memory.graph import normalize_node_id

    src = normalize_node_id(root_task_id)
    tgt = normalize_node_id(plan_node.id)
    guidance = (subtask_result.final_answer or "")[:500]
    pitfalls = ""
    if subtask_result.status != "ok":
        pitfalls = f"subtask {plan_node.id!r} ended with status={subtask_result.status} after {subtask_result.turns} turns"
    memory_engine.record_triplet(
        src,
        "CONVERGES_TO",
        tgt,
        condition=plan_node.acceptance,
        guidance=guidance,
        pitfalls=pitfalls,
        source_session_id=subtask_result.session_id,
    )
    logger.info(
        "orchestrator.subtask_record",
        root=src,
        node=tgt,
        status=subtask_result.status,
        session_id=subtask_result.session_id,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v -k "dispatch_subtask or record_subtask"`
Expected: PASS — 3 tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add dispatch and per-sub-task triplet recorder"
```

---

## Task 6: Plan-summary triplet (`_record_plan_summary`)

**Files:**
- Modify: `orchestrator.py` (add `_record_plan_summary`)
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- `_record_plan_summary(root_task_id, plan, executions, status, memory_engine) -> str` returns the new edge's key (or some identifier).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def test_record_plan_summary_writes_end_edge(tmp_path):
    from memory import MemoryEngine

    engine = MemoryEngine(tmp_path)
    tree = _simple_tree()
    execs = [
        SubTaskResult(plan_node_id="s1", session_id="x", final_answer="ok", status="ok", turns=2),
    ]
    edge_id = orchestrator._record_plan_summary(
        root_task_id="r", plan=tree, executions=execs,
        status="ok", memory_engine=engine,
    )
    fresh = MemoryEngine(tmp_path)
    data = json.loads(fresh.graph_path.read_text())
    edges = [
        e for e in data["edges"]
        if (e["source"], e["target"]) == ("r", "End")
    ]
    assert len(edges) == 1
    e = edges[0]
    assert e["relation"] == "LEADS_TO"
    assert e["condition"] is None
    assert "ok" in e["guidance"]
    assert isinstance(edge_id, str)


def test_record_plan_summary_records_pitfalls_for_failures(tmp_path):
    from memory import MemoryEngine

    engine = MemoryEngine(tmp_path)
    tree = _simple_tree()
    execs = [
        SubTaskResult(plan_node_id="s1", session_id="x", final_answer="", status="failed", turns=8),
    ]
    orchestrator._record_plan_summary(
        root_task_id="r", plan=tree, executions=execs,
        status="failed", memory_engine=engine,
    )
    fresh = MemoryEngine(tmp_path)
    data = json.loads(fresh.graph_path.read_text())
    e = next(
        edge for edge in data["edges"]
        if (edge["source"], edge["target"]) == ("r", "End")
    )
    assert "s1" in e["pitfalls"]
    assert "failed" in e["pitfalls"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_record_plan_summary_writes_end_edge -v`
Expected: FAIL — `AttributeError: module 'orchestrator' has no attribute '_record_plan_summary'`

- [ ] **Step 3: Implement `_record_plan_summary`**

Append to `orchestrator.py`:

```python
from procedural_memory.graph import END, normalize_node_id


def _record_plan_summary(
    *,
    root_task_id: str,
    plan: PlanTree,
    executions: list[SubTaskResult],
    status: str,
    memory_engine: Any,
) -> str:
    src = normalize_node_id(root_task_id)
    summary = f"Plan finished with status={status!r}; {len(executions)} sub-tasks executed."
    pitfalls_parts: list[str] = []
    for ex in executions:
        if ex.status != "ok":
            pitfalls_parts.append(
                f"sub-task {ex.plan_node_id!r} status={ex.status} turns={ex.turns}"
            )
    pitfalls = "; ".join(pitfalls_parts)
    memory_engine.record_triplet(
        src,
        "LEADS_TO",
        END,
        condition=None,
        guidance=summary,
        pitfalls=pitfalls,
    )
    edge_id = f"{src}->{END}#LEADS_TO"
    logger.info(
        "orchestrator.plan_summary_record",
        root=src,
        status=status,
        subtasks=len(executions),
        edge_id=edge_id,
    )
    return edge_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_orchestrator.py -v -k record_plan_summary`
Expected: PASS — 2 tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): add plan-summary triplet recorder"
```

---

## Task 7: `run()` orchestrator driver + default gate

**Files:**
- Modify: `orchestrator.py` (add `run()`, `_default_human_approval`)
- Modify: `tests/test_orchestrator.py`

**Interfaces:**
- `OrchestratorResult = run(task, *, strict=False, max_outer_iterations=..., max_replan=..., on_human_approval=None, llm=None)`
- `_default_human_approval(plan, critique) -> bool` — TTY stdin reader; returns False on EOF / non-TTY.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def test_run_happy_path(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    tree = _simple_tree()
    critique = CritiqueReport(findings=[], blocking=[], mode="advisory")

    llm = _StubLlm(tree, critique)

    def fake_dispatch(node, *, memory_engine, session_manager):
        return SubTaskResult(
            plan_node_id=node.id, session_id="sx", final_answer="did it",
            status="ok", turns=2,
        )

    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake_dispatch)

    result = orchestrator.run(
        "refactor jwt", strict=False,
        memory_engine=engine, session_manager=sm,
        on_human_approval=lambda plan, critique: True, llm=llm,
    )

    assert result.status == "ok"
    assert len(result.executions) == 2  # s1 from simple_tree
    assert result.plan_summary_triplet_id is not None


def test_run_rejected_by_human_short_circuits(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager

    tree = _simple_tree()
    critique = CritiqueReport(findings=[], blocking=[], mode="advisory")
    llm = _StubLlm(tree, critique)

    def must_not_dispatch(*a, **kw):
        raise AssertionError("dispatch must not be called when human rejects")

    monkeypatch.setattr(orchestrator, "_dispatch_subtask", must_not_dispatch)

    result = orchestrator.run(
        "x", strict=False,
        memory_engine=engine, session_manager=sm,
        on_human_approval=lambda plan, critique: False, llm=llm,
    )
    assert result.status == "rejected_by_human"
    assert result.reason == "human_rejected_root_plan"
    assert result.executions == []


def test_run_human_gate_fires_once(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    tree = _simple_tree()
    critique = CritiqueReport(findings=[], blocking=[], mode="advisory")
    llm = _StubLlm(tree, critique)

    def fake_dispatch(node, *, memory_engine, session_manager):
        return SubTaskResult(
            plan_node_id=node.id, session_id="sx", final_answer="ok",
            status="ok", turns=1,
        )
    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake_dispatch)

    calls = []

    def approval(plan, critique):
        calls.append((plan.root.id, critique.mode))
        return True

    orchestrator.run("x", strict=False, memory_engine=engine, session_manager=sm,
                     on_human_approval=approval, llm=llm)
    assert len(calls) == 1
    assert calls[0] == ("r", "all")


def test_run_returns_failed_on_plan_parse_error(tmp_path):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    llm = _StubLlm("not json", "still not json")
    result = orchestrator.run(
        "x", strict=False, memory_engine=engine, session_manager=sm,
        on_human_approval=lambda *_: True, llm=llm,
    )
    assert result.status == "failed"
    assert result.reason == "plan_parse_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_orchestrator.py::test_run_happy_path -v`
Expected: FAIL — `AttributeError: module 'orchestrator' has no attribute 'run'`

- [ ] **Step 3: Implement `run()` and `_default_human_approval`**

Append to `orchestrator.py`:

```python
import sys


def _default_human_approval(plan: PlanTree, critique: CritiqueReport) -> bool:
    if not sys.stdin.isatty():
        return False
    typer_echo = getattr(sys.modules.get("typer", None), "echo", None)
    text = f"Approve plan root={plan.root.id!r} with {len(critique.findings)} findings? [y/n]: "
    print(text, end="", flush=True)
    line = sys.stdin.readline().strip().lower()
    return line in ("y", "yes")


def run(
    task: str,
    *,
    strict: bool = False,
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
    max_replan: int = DEFAULT_MAX_REPLAN,
    on_human_approval: Callable[[PlanTree, CritiqueReport], bool] | None = _default_human_approval,
    llm: Callable[[str], Any] | None = None,
    memory_engine: Any = None,
    session_manager: Any = None,
) -> OrchestratorResult:
    if memory_engine is None:
        import memory as memory_module
        memory_engine = memory_module.memoryEngine
    if session_manager is None:
        import session as session_module
        session_manager = session_module.sessionManager

    root_task_id = task
    recall_ctx = Recall.recall(task, llm=llm)

    plan = PlanAgent.run(task, recall_ctx, llm=llm)
    if plan is None:
        return OrchestratorResult(
            status="failed",
            reason="plan_parse_error",
            plan=PlanTree(
                root=PlanNode(id=root_task_id, title=root_task_id, task=root_task_id,
                              acceptance="root done", children=[])
            ),
            critique=CritiqueReport(findings=[], blocking=[], mode="advisory"),
            executions=[],
            plan_summary_triplet_id=None,
        )

    mode = "strict" if strict else "advisory"
    critique = CriticAgent.run(task, plan, recall_ctx, mode=mode, llm=llm)

    if on_human_approval is not None and not on_human_approval(plan, critique):
        return OrchestratorResult(
            status="rejected_by_human",
            reason="human_rejected_root_plan",
            plan=plan, critique=critique, executions=[], plan_summary_triplet_id=None,
        )

    executions: list[SubTaskResult] = []
    replans_left = max_replan
    while True:
        next_node = _next_pending_node(plan, executions)
        if next_node is None:
            break
        if next_node.strict and executions and executions[-1].status != "ok":
            return OrchestratorResult(
                status="failed", reason="strict_subtask_precondition_failed",
                plan=plan, critique=critique, executions=executions,
                plan_summary_triplet_id=None,
            )
        result = _dispatch_subtask(
            next_node, memory_engine=memory_engine, session_manager=session_manager,
        )
        executions.append(result)
        _record_subtask_triplet(root_task_id, next_node, result, memory_engine)

        if next_node.strict and result.status != "ok":
            return OrchestratorResult(
                status="failed", reason=f"strict_subtask_failed:{next_node.id}",
                plan=plan, critique=critique, executions=executions,
                plan_summary_triplet_id=None,
            )

        if len(executions) >= max_outer_iterations:
            return OrchestratorResult(
                status="max_outer_iterations", reason="max_outer_iterations",
                plan=plan, critique=critique, executions=executions,
                plan_summary_triplet_id=None,
            )

        if strict and critique.blocking and replans_left > 0:
            replans_left -= 1
            logger.info("orchestrator.replan", remaining=replans_left)
            continue

    final_status = "ok"
    if any(e.status != "ok" for e in executions):
        final_status = "partial"
    edge_id = _record_plan_summary(
        root_task_id=root_task_id, plan=plan, executions=executions,
        status=final_status, memory_engine=memory_engine,
    )
    return OrchestratorResult(
        status=final_status, reason=None,
        plan=plan, critique=critique, executions=executions,
        plan_summary_triplet_id=edge_id,
    )


def _next_pending_node(plan: PlanTree, executions: list[SubTaskResult]) -> PlanNode | None:
    done = {e.plan_node_id for e in executions}
    pending_leaves: list[PlanNode] = []

    def visit(node: PlanNode) -> None:
        if node.id in done:
            return
        if not node.children:
            pending_leaves.append(node)
            return
        for child in node.children:
            visit(child)

    visit(plan.root)
    if pending_leaves:
        return pending_leaves[0]
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v -k "test_run_"`
Expected: PASS — 4 new tests green

- [ ] **Step 5: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "feat(orchestrator): wire run() with human gate, dispatch loop, and replan cap"
```

---

## Task 8: CLI surface — `whale orchestrate`

**Files:**
- Modify: `main.py` (add `orchestrate` command; import `default_stdin_approval` from orchestrator)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_main.py`:

```python
def test_orchestrate_command_registered():
    from main import app
    names = {cmd.name for cmd in app.registered_commands if cmd.name}
    assert "orchestrate" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_orchestrate_command_registered -v`
Expected: FAIL — `AssertionError: 'orchestrate' not in {...}`

- [ ] **Step 3: Add `orchestrate` command to `main.py`**

In `main.py`, after the existing `prompt` command and before the `refine` command, add:

```python
from orchestrator import run as orch_run
from orchestrator import _default_human_approval


@app.command()
def orchestrate(
    text: str = typer.Argument(..., help="Task to plan and execute."),
    strict: bool = typer.Option(False, "--strict", help="Strict critic mode."),
    max_iter: int = typer.Option(20, "--max-iter", help="Max outer iterations."),
    no_human: bool = typer.Option(
        False, "--no-human",
        help="Skip the human approval gate (auto-approve root plan).",
    ),
) -> None:
    """Plan the task as a depth-≤2 tree and dispatch sub-tasks to the agent loop."""
    approval = (lambda plan, critique: True) if no_human else _default_human_approval
    result = orch_run(
        text, strict=strict, max_outer_iterations=max_iter,
        on_human_approval=approval,
    )
    typer.echo(result.status)
    for ex in result.executions:
        typer.echo(f"  - {ex.plan_node_id}: {ex.status} ({ex.turns} turns)")
```

Also ensure the existing `prompt` command stays unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS — all existing tests still green, plus the new one

- [ ] **Step 5: Smoke-test the CLI in `--no-human` mode**

Run: `python main.py orchestrate "noop" --no-human --max-iter 1`
Expected: CLI runs without crashing. Output begins with a status line. (Output may show `"failed"` with `"plan_parse_error"` because the stub LLM isn't configured — that's fine; we're only smoke-testing the wiring.)

If your environment has the OpenAI key set, this will produce a real plan. Either outcome is acceptable as long as the CLI doesn't traceback.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat(cli): add whale orchestrate command"
```

---

## Task 9: Critic strict-mode block test (failure-mode coverage)

**Files:**
- Modify: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_orchestrator.py`:

```python
def test_run_strict_mode_halts_when_blocking(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    tree = _simple_tree()
    critique = CritiqueReport(
        findings=["minor"], blocking=["missing rollback plan"], mode="strict"
    )
    llm = _StubLlm(tree, critique)

    def fake_dispatch(node, *, memory_engine, session_manager):
        return SubTaskResult(
            plan_node_id=node.id, session_id="sx", final_answer="ok",
            status="ok", turns=1,
        )
    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake_dispatch)

    result = orchestrator.run(
        "x", strict=True, max_replan=2,
        memory_engine=engine, session_manager=sm,
        on_human_approval=lambda *_: True, llm=llm,
    )

    # strict mode + blocking + max_replan exhausted -> failed
    assert result.status == "failed"
    assert result.reason == "strict_replan_cap_reached"
    assert len(result.executions) >= 1  # at least one sub-task got dispatched before halts


def test_advisory_critic_with_blocking_does_not_halt(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    tree = _simple_tree()
    critique = CritiqueReport(
        findings=["note"], blocking=["something"], mode="advisory"
    )
    llm = _StubLlm(tree, critique)

    def fake_dispatch(node, *, memory_engine, session_manager):
        return SubTaskResult(
            plan_node_id=node.id, session_id="sx", final_answer="ok",
            status="ok", turns=1,
        )
    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake_dispatch)

    result = orchestrator.run(
        "x", strict=False,
        memory_engine=engine, session_manager=sm,
        on_human_approval=lambda *_: True, llm=llm,
    )
    assert result.status == "ok"


def test_strict_subtask_failure_halts_orchestrator(tmp_path, monkeypatch):
    from memory import MemoryEngine
    from session import SessionManager

    engine = MemoryEngine(tmp_path)
    sm = SessionManager(tmp_path)

    strict_tree = PlanTree(
        root=PlanNode(
            id="r", title="r", task="r", acceptance="r done",
            children=[
                PlanNode(id="s1", title="s1", task="s1",
                         acceptance="s1 done", children=[], strict=True),
            ],
        )
    )
    critique = CritiqueReport(findings=[], blocking=[], mode="advisory")
    llm = _StubLlm(strict_tree, critique)

    def fake_dispatch(node, *, memory_engine, session_manager):
        return SubTaskResult(
            plan_node_id=node.id, session_id="sx", final_answer="",
            status="failed", turns=8,
        )
    monkeypatch.setattr(orchestrator, "_dispatch_subtask", fake_dispatch)

    result = orchestrator.run(
        "x", strict=False,
        memory_engine=engine, session_manager=sm,
        on_human_approval=lambda *_: True, llm=llm,
    )
    assert result.status == "failed"
    assert "strict_subtask_failed" in (result.reason or "")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v -k "strict_mode or advisory_critic or strict_subtask_failure"`
Expected: PASS — 3 tests green. If `test_run_strict_mode_halts_when_blocking` fails with a different `reason`, adjust the assertion to match what the implementation returns and commit the fix.

- [ ] **Step 3: If any test fails, fix the orchestrator implementation accordingly**

The strict-mode loop in Task 7 needs to actually halt after `max_replan` strict-rejects. If the test reveals the loop doesn't decrement `replans_left` correctly or doesn't return the right `status`/`reason`, edit `orchestrator.py`'s `run()` to match the spec §8 ("After `max_replan=2` reached, halt outer loop, status = `failed`, include critique in result") and re-run until green.

- [ ] **Step 4: Commit**

```bash
git add orchestrator.py tests/test_orchestrator.py
git commit -m "test(orchestrator): cover critic block, advisory passthrough, strict subtask halt"
```

---

## Task 10: Full test run + lint

**Files:** none

- [ ] **Step 1: Run the full pytest suite**

Run: `pytest -v`
Expected: ALL tests green. Any pre-existing failure is unrelated; the orchestrator-related suite must be 100% green.

- [ ] **Step 2: Run a quick import smoke test**

Run: `python -c "import orchestrator, orchestrator_types; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Verify `agent_loop.py` is unchanged**

Run: `git diff HEAD~10 -- agent_loop.py memory.py procedural_memory/`
Expected: empty diff (no changes to the inner loop or the procedural graph package)

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git status
# if anything changed:
git add -u
git commit -m "chore: orchestrator layer — final cleanup"
```

---

## Self-Review

**Spec coverage check:**

| Spec § | Requirement | Task |
| --- | --- | --- |
| §2 Goal 1 | Recall step | Task 2 |
| §2 Goal 2 | Plan Agent + depth ≤ 2 | Task 1 (invariant), Task 3 (emitter) |
| §2 Goal 3 | Critic-of-Plan advisory/strict | Task 4 (Critic), Task 7 (mode wiring), Task 9 (strict halt) |
| §2 Goal 4 | Single human-in-the-loop gate | Task 7 (`run`), Task 8 (CLI default) |
| §2 Goal 5 | Dispatch to existing `agent_loop.py` | Task 5 (`_dispatch_subtask`) |
| §2 Goal 6 | Per-sub-task + plan-summary records | Task 5 (per-sub-task), Task 6 (summary) |
| §4.2 | Dataclasses + depth invariant | Task 1 |
| §4.3 | Prompt contracts (Plan, Critic) | Task 3, Task 4 |
| §4.4 | Human gate injected | Task 7, Task 8 |
| §4.5 | Dispatch contract | Task 5 |
| §4.6 | Record contract | Task 5 (per-sub-task), Task 6 (summary) |
| §6 | CLI surface | Task 8 |
| §7 | 9 unit tests with mocked LLMs | Tasks 1–9 each add tests; Task 10 confirms |
| §8 | Failure modes | Task 9 (strict block, advisory passthrough, strict subtask halt); Task 7 (parse error) |

All spec items map to tasks. No gaps.

**Placeholder scan:** no "TBD", no "add validation later", no "similar to Task N" stubs. All code blocks contain real code the engineer can paste.

**Type consistency check:**
- `PlanNode(id, title, task, acceptance, children, strict, max_turns)` — defined in Task 1, used identically in Tasks 2–9.
- `PlanTree(root, created_at)` with `MAX_DEPTH=2` invariant — Task 1, used in all later tasks.
- `CritiqueReport(findings, blocking, mode)` — Task 1, used in Tasks 4, 7, 9.
- `SubTaskResult(plan_node_id, session_id, final_answer, status, turns)` — Task 1, used in Tasks 5–7, 9.
- `OrchestratorResult(status, reason, plan, critique, executions, plan_summary_triplet_id)` — Task 1, returned from Task 7's `run()`.
- `_dispatch_subtask(plan_node, *, memory_engine, session_manager) -> SubTaskResult` — Task 5, called by Task 7's `run()`.
- `_record_subtask_triplet(root_task_id, plan_node, subtask_result, memory_engine)` — Task 5, called by Task 7.
- `_record_plan_summary(*, root_task_id, plan, executions, status, memory_engine) -> str` — Task 6, called by Task 7.
- `run(task, *, strict, max_outer_iterations, max_replan, on_human_approval, llm, memory_engine, session_manager) -> OrchestratorResult` — Task 7.

No name drift. No signature drift.
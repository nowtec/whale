# Orchestrator Layer (Recall → Plan → Critic → Execute) — Design Spec

**Date:** 2026-10-11
**Status:** Approved (brainstorming), pre-implementation
**Paper reference:** *Adversarial Cooperation In Code Synthesis*, Block AI Research (2025). PDF at `docs/adversarial-cooperation-in-code-synthesis.pdf`. §3 ("Adversarial Cooperation") and §4 ("Context Window Management") are the conceptual anchors; this spec generalises the coach/player dyad into a layered system.

---

## 1. Summary

Add a new module `orchestrator.py` that wraps the existing `agent_loop.py` with a depth-≤2 `PlanTree`, an Advisory/Strict `Critic-of-Plan`, a single human-approval gate at session start, and per-sub-task + plan-summary records on the procedural graph. The inner ReAct loop is **not modified**. The orchestrator and the executor are the same Agent Loop pattern (Recall → Plan → Critic? → Execute → Record → Advance) instantiated at two scales.

---

## 2. Goals & Non-Goals

### Goals

Add a **layer above** the existing `agent_loop.py` that implements the user's architecture description:
*pick up → learn → plan → research → execute → register completion → loop*.

Concretely, introduce an **Orchestrator** that wraps the existing player/coach dialectic with:

1. **Recall** — a deliberate pre-plan retrieval step against the procedural graph memory, distinct from the per-turn `Guidance.retrieve()` already used inside the executor.
2. **Plan** — a `Plan Agent` that produces a **depth ≤ 2 tree** of sub-tasks (root → depth-1 → depth-2). Each non-root node names exactly one executor and its acceptance criteria.
3. **Critic-of-Plan** — a `Critic Agent` that reviews the tree before dispatch. **Advisory** by default (emits critique, does not block). **Strict** on tasks the user has flagged, or on sub-tasks explicitly marked strict (rejects on any blocking defect).
4. **Human-in-the-loop** — a single approval gate at session start on the root plan tree. No further human interruptions during execution.
5. **Dispatch** — one sub-task at a time to the **existing** `agent_loop.py` as the executor. The executor's per-turn `Guidance.retrieve()` IS the inner-loop Recall; the inner loop is unchanged.
6. **Records** — per-sub-task triplet on the procedural graph when each sub-task finishes, plus a single plan-summary triplet when the session closes.

### Non-Goals (this iteration)

- Modifying `agent_loop.py`. The inner ReAct loop and its `Guidance` call stay exactly as they are.
- Modifying the player/coach dialectic semantics inside the executor. The g3 paper's pattern is preserved verbatim.
- A swarm of multiple Planner LLMs voting on the tree. The Plan Agent is one LLM call per outer iteration. Tree extension via swarm is a later spec.
- Automatic strict-mode detection. Tasks are flagged strict explicitly (`--strict` flag, or a `strict` field in the Plan tree).
- Multi-session planning. Each session plans its own root tree. Cross-session learning is already handled by the procedural graph memory.
- Replacing the procedural graph. The orchestrator **uses** the existing `procedural_memory/` package; it does not redefine the graph.

---

## 3. Architectural insight: one Agent Loop pattern, two instantiations

The single most important framing decision in this spec:

> **The orchestrator and the executor are the same Agent Loop pattern instantiated at two scales.**

The cycle is identical at both levels: **Recall → Plan → Critic? → Execute → Record → Advance**. The differences are only in *scale of unit-of-work*, *presence of Critic*, *scope of Recall*, and *shape of Record*.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AgentLoop  (the pattern)                          │
│                                                                      │
│   while not done:                                                     │
│       1. RECALL   — query procedural graph for guidance              │
│       2. PLAN     — Planner LLM emits a unit-of-work artifact       │
│       3. CRITIC   — (optional) Critic LLM reviews; advisory/strict   │
│       4. EXECUTE  — apply the unit; collect result                   │
│       5. RECORD   — append triplet(s) to the procedural graph        │
│       6. ADVANCE  — update "current state" pointer                   │
└──────────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴────────────┐
                │                          │
        ┌───────▼────────┐        ┌────────▼─────────┐
        │   OUTER LOOP   │        │   INNER LOOP     │
        │  (Orchestrator)│        │   (Executor)     │
        │  orchestrator  │        │   agent_loop.py  │
        │      .py       │        │   (UNCHANGED)    │
        └────────────────┘        └──────────────────┘
```

### 3.1 Pattern parameters (the only differences)

| Aspect                  | Outer (Orchestrator)                                     | Inner (Executor)                                                |
| ----------------------- | -------------------------------------------------------- | --------------------------------------------------------------- |
| Loop driver             | `Orchestrator.run(task, strict=False)`                   | `agent_loop.run(task, session_id, max_turns)`                   |
| **Unit of work**        | one sub-task (a node in the depth-≤2 tree)               | one ReAct turn (Thought + Action)                               |
| Recall                  | `Orchestrator.recall(task, plan_state)`                  | `procedural_memory.guidance.Guidance.retrieve(...)`             |
| Recall scope            | full graph, query = root task + current sub-task         | local h-hop neighbourhood, query = recent actions               |
| Planner                 | `Plan Agent` → emits a `PlanTree`                        | `SOLVER_PROMPT` → emits one `Action:`                           |
| Critic                  | `Critic-of-Plan Agent` (advisory or strict)              | **none** — the g3 paper's Coach is the critic + approver fused  |
| Execute                 | dispatch one sub-task to a fresh executor                | run one tool, return tool output                                 |
| Record                  | per-sub-task triplet + plan-summary triplet at close     | per-turn `memory.record_turn` triplet                           |
| Advance                 | mark sub-task complete; pointer → next unstarted node    | recent-actions list updated                                     |
| Human gate              | yes, on root plan, first outer iteration only            | none                                       |
| Done when               | all sub-tasks complete, or max outer iterations, or root failure | FINISH verb, or max_turns, or coach approves (paper semantics) |

### 3.2 Why the inner loop is unchanged

The g3 paper's adversarial cooperation (§3 of the PDF) is a player/coach dyad inside a bounded turn loop with fresh context per turn. **That whole mechanism already lives in `agent_loop.py` plus `procedural_memory/guidance.py`.** The user's stated architecture describes adding *a layer before the agentic loop* — i.e. an orchestrator that uses the loop, not one that replaces it.

Touching `agent_loop.py` would also break the existing test surface and the per-turn `Guidance.retrieve()` contract that already implements "fresh context per turn" (§4 of the paper). The Critic lives **only in the outer loop** because adding a thin Critic in the inner loop would either:
- duplicate the Coach's job (the Coach already does critique + approve inside the dyad), or
- double the LLM-call count per turn for marginal benefit.

The requirement that "the dialectic happens at almost every iteration of the Agentic Loop" is satisfied because **each outer iteration dispatches one sub-task, and that sub-task runs its own player/coach dialectic** (the inner loop). The dialectic is preserved at execution scale; the orchestrator adds planning-scale structure above it.

---

## 4. Components

### 4.1 `orchestrator.py` (new, ~300–450 LOC target)

Single public entry point:

```python
DEFAULT_MAX_TURNS = 8         # executor max_turns when PlanNode.max_turns is None
DEFAULT_MAX_OUTER_ITERATIONS = 20
DEFAULT_MAX_REPLAN = 2

def run(
    task: str,
    *,
    strict: bool = False,             # default: advisory critic
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
    max_replan: int = DEFAULT_MAX_REPLAN,
    on_human_approval: Callable[[PlanTree, CritiqueReport], bool] | None = None,
    llm: Callable[[str], Any] | None = None,
) -> OrchestratorResult: ...
```

Internal flow:

1. `Recall.recall(task, plan_state=None)` → string used in Plan prompt.
2. `plan_state = PlanAgent.run(task, recall_context)` → initial `PlanTree` (root → depth ≤ 2).
3. `critique = CriticAgent.run(plan_state, recall_context, mode=strict_or_advisory)` → `CritiqueReport`.
4. If `on_human_approval` is provided and returns `False` → return `OrchestratorResult(status="rejected_by_human", reason="human_rejected_root_plan", plan=plan_state, critique=critique, executions=[], plan_summary_triplet_id=None)`. This fires once per session.
5. Outer loop:
   - pick next unstarted sub-task in DFS order,
   - dispatch to `agent_loop.run(sub_task.task, session_id=new, max_turns=sub_task.max_turns or DEFAULT_MAX_TURNS)`,
   - on executor close: `record_subtask_triplet(plan_node, executor_result)`,
   - if any strict-flagged sub-task failed → halt and surface to caller,
   - if critic mode is strict and the *next* plan delta produces a blocking defect → re-plan (cap re-plans at `max_replan`).
6. On outer loop exit: `record_plan_summary(plan_state, executed_subtasks, outcomes)` → one summary triplet on the graph.
7. Return `OrchestratorResult(status, reason, plan, critique, executions, plan_summary_triplet_id)`.

### 4.2 `orchestrator_types.py` (new, dataclasses only)

```python
@dataclass(frozen=True)
class PlanNode:
    id: str                    # stable node id within the tree
    title: str                 # human-readable
    task: str                  # the prompt handed to the executor
    acceptance: str            # criteria the executor must satisfy
    children: list[PlanNode]   # depth ≤ 2 enforced at construction
    strict: bool = False       # per-node override of orchestrator.strict
    max_turns: int | None = None  # override executor max_turns; None -> DEFAULT_MAX_TURNS

@dataclass(frozen=True)
class PlanTree:
    root: PlanNode             # always depth 0; children depth 1; their children depth 2
    created_at: str

@dataclass(frozen=True)
class CritiqueReport:
    findings: list[str]        # advisory notes (always populated)
    blocking: list[str]        # only populated in strict mode; empty -> no drop
    mode: Literal["advisory", "strict"]

@dataclass(frozen=True)
class SubTaskResult:
    plan_node_id: str
    session_id: str            # executor session
    final_answer: str
    status: Literal["ok", "failed", "max_turns"]
    turns: int

@dataclass(frozen=True)
class OrchestratorResult:
    status: Literal["ok", "partial", "failed", "rejected_by_human", "max_outer_iterations"]
    reason: str | None         # machine-readable explanation for failure statuses
    plan: PlanTree
    critique: CritiqueReport
    executions: list[SubTaskResult]
    plan_summary_triplet_id: str | None
```

**Depth invariant:** `PlanTree` constructor enforces `len(node.children) == 0` for any node whose parent has `depth>1`. Equivalently: depth-2 nodes are leaves.

### 4.3 Prompt contracts (three LLM call sites)

All three prompts are simple string-templates returning JSON. Reuse the JSON-mode `models.completion_default_json` helper.

**(a) Plan Agent prompt** — input: `task`, `recall_context`, optional `previous_plan_node` + `subtask_outcome` (for re-plan deltas). Output: a `PlanTree` JSON matching the dataclass shape. **Constraint in the prompt:** "produce a tree of depth at most 2; depth-2 nodes are leaves; each node names exactly one acceptance criterion; each node may carry an integer `max_turns` for the executor (omit to use the orchestrator default)."

**(b) Critic-of-Plan Agent prompt** — input: `task`, `plan_tree_json`, `recall_context`, `mode` (`"advisory" | "strict"`). Output: a `CritiqueReport` JSON. **Advisory mode instruction:** "list any defects you observe, but do not block execution." **Strict mode instruction:** "list any *blocking* defects that would cause the executor to fail the root task or contradict the requirements. Empty `blocking` list means approve."

### 4.4 Human gate

The human gate is a single `on_human_approval(plan, critique) -> bool` callback, injected by the caller. Default implementation (used by `main.py`) prints the plan tree + critique to stdout and reads `y/n` from stdin. This is the **only** human interruption per session.

Rationale for "approve root only": the g3 paper's main efficiency claim is that autonomous cooperation eliminates the "humans must be present when the agent turn completes" cost. Heavy HITL would defeat that. The single root gate gives the human veto power over the plan without paying the per-turn cost.

### 4.5 Dispatch contract (Orchestrator → Executor)

```python
# orchestrator.py → agent_loop.py
agent_loop.run(
    task=plan_node.task,
    session_id=new_uuid(),         # fresh session per sub-task
    max_turns=plan_node.max_turns or DEFAULT_MAX_TURNS,  # 8 default
)
```

The executor is **not aware** of the orchestrator. It runs the same loop it runs today. The orchestrator only:
- picks which `task` to hand it,
- decides how many `max_turns`,
- captures `result.final_answer` and `result.turns`.

### 4.6 Record contract (Orchestrator → Procedural Graph)

**Per-sub-task record** (when each sub-task closes, before the next sub-task starts):

```
src = "<root-task-id>"
relation = "CONVERGES_TO"   (one of the RELATIONS in procedural_memory/graph.py)
tgt = "<sub-task-id>"
guidance = "<executor final answer, first 500 chars>"
pitfalls = "<failure modes observed, if any>"
condition = "<acceptance criterion from PlanNode>"
source_session_id = "<executor-session-id>"
```

**Plan-summary record** (when the outer loop closes):

```
src = "<root-task-id>"
relation = "LEADS_TO"
tgt = "End"   (the reserved END node from procedural_memory/graph.py)
guidance = "<one-line summary of the executed plan + final status>"
pitfalls = "<list of failed sub-tasks and why>"
condition = None
```

Both reuse the existing `memory.record_triplet()` API. No new graph API is introduced.

---

## 5. Data flow (one outer iteration)

```
User task: "Refactor the auth module to use JWT"
                │
                ▼
       ┌────────────────────────────────────────────────┐
       │  Orchestrator.run(task)                        │
       │                                                │
       │  1. Recall.recall(task)                        │
       │     → reads procedural graph via Guidance.retrieve│
       │       with empty recent_actions (full graph)   │
       │     → "prior sessions refactored similar auth  │
       │        modules; common pitfalls: missing token │
       │        expiry check"                           │
       │                                                │
       │  2. PlanAgent.run(task, recall_context)        │
       │     → PlanTree (root = task):                  │
       │       [0] Refactor auth to use JWT             │
       │           [1] Audit current auth module        │
       │           [1] Design JWT schema                │
       │           [1] Implement JWT middleware         │
       │           [1] Write integration tests          │
       │       (depth 1 → 4 sub-tasks; each is itself  │
       │        a leaf because max depth = 2)           │
       │                                                │
       │  3. CriticAgent.run(plan, recall, mode)        │
       │     → CritiqueReport (advisory)                │
       │       "Plan looks coherent. Note: no 'Rollback
       │        plan' sub-task — consider adding."       │
       │                                                │
       │  4. on_human_approval(plan, critique)          │
       │     → True (user typed 'y')                    │
       │                                                │
       │  5. outer loop iteration 1:                    │
       │     sub_task = audit-current-auth              │
       │     result = agent_loop.run(task=audit, ...)   │
       │     record_subtask_triplet(...)                │
       │                                                │
       │     outer loop iteration 2:                    │
       │     sub_task = design-jwt-schema               │
       │     ...                                        │
       │                                                │
       │  6. all sub-tasks complete                     │
       │     record_plan_summary(...)                   │
       │     return OrchestratorResult(status=ok, ...)  │
       └────────────────────────────────────────────────┘
```

---

## 6. CLI surface

Add to `main.py` (Typer):

```python
@app.command()
def orchestrate(
    text: str = typer.Argument(..., help="Task to plan and execute."),
    strict: bool = typer.Option(False, "--strict", help="Strict critic mode."),
    max_iter: int = typer.Option(20, "--max-iter", help="Max outer iterations."),
    no_human: bool = typer.Option(False, "--no-human",
        help="Skip the human approval gate (auto-approve root plan)."),
) -> None:
    """Plan the task as a depth-≤2 tree and dispatch sub-tasks to the agent loop."""
    from orchestrator import run as orch_run
    approval = (lambda plan, critique: True) if no_human else default_stdin_approval
    result = orch_run(text, strict=strict, max_outer_iterations=max_iter,
                      on_human_approval=approval)
    typer.echo(result.status)
    for ex in result.executions:
        typer.echo(f"  - {ex.plan_node_id}: {ex.status} ({ex.turns} turns)")
```

The existing `prompt` command stays unchanged.

---

## 7. Testing strategy

This spec's behaviour lives in `orchestrator.py` (mostly) and `orchestrator_types.py` (depth invariant). Tests target the boundaries that can be unit-tested without LLM calls:

| Test                                                           | What it asserts                                              |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| `test_plan_tree_depth_invariant.py`                            | depth-3 nodes rejected at construction                          |
| `test_orchestrator_recall_uses_existing_guidance.py`          | `Recall.recall()` calls `Guidance.retrieve()` with empty recent_actions (full-graph mode) |
| `test_orchestrator_dispatch_calls_agent_loop.py`              | dispatch invokes `agent_loop.run()` with correct args        |
| `test_orchestrator_records_subtask_triplet.py`                | each sub-task closure calls `memory.record_triplet` exactly once with the contract in §4.6 |
| `test_orchestrator_records_plan_summary_on_close.py`          | session close calls `record_triplet` to `End` with the summary contract |
| `test_critic_advisory_does_not_block.py`                      | strict-flag off + blocking finding in critique → still proceeds |
| `test_critic_strict_blocks_on_blocking.py`                    | strict-flag on + non-empty `blocking` → re-plan up to `max_replan` |
| `test_human_gate_fires_once_per_session.py`                   | `on_human_approval` called exactly once                      |
| `test_human_gate_rejection_short_circuits.py`                 | human returns False → no sub-tasks dispatched, status = `rejected_by_human` |

LLM responses are mocked with stub `llm` callables injected into `Orchestrator.run(..., llm=stub)`.

---

## 8. Failure modes & edge cases

| Failure                                              | Behaviour                                                                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Plan Agent returns invalid JSON                      | One retry with stricter prompt; on second failure, return `OrchestratorResult(status="failed", reason="plan_parse_error")` |
| Plan Agent returns tree with depth > 2               | Clamp at construction in `PlanTree` (reject depth-3 nodes; if any depth-3 returned, surface error) |
| Critic strict mode + blocking finding + replan cap   | After `max_replan=2` reached, halt outer loop, status = `failed`, include critique in result      |
| Executor hits max_turns without FINISH                | Sub-task status = `max_turns`; orchestrator continues to next sub-task; surfaces in plan summary     |
| Sub-task explicitly marked `strict` and fails        | Halt outer loop, status = `failed`, surface failing node                                           |
| Procedural graph empty (skeleton only)               | Recall returns empty string; Plan Agent prompt degrades gracefully (no graph guidance); same as `Guidance.retrieve` today |
| Human gate times out / non-TTY stdin                 | `default_stdin_approval` raises → orchestrator returns `rejected_by_human`                        |
| Two strict sub-tasks both fail                       | First failure halts; second never runs                                                              |

---

## 9. Out of scope (deferred)

These are flagged explicitly so they don't sneak into implementation:

- **Swarm of planner sub-agents** voting on tree branches. Today's spec has one Plan Agent per outer iteration. Adding a swarm is a later spec.
- **Tree re-shaping** beyond `max_replan=2`. If the planner wants to restructure the tree after sub-task feedback beyond two re-plans, halt and surface to the user instead.
- **Parallel sub-task execution.** Sub-tasks run sequentially. Parallelism changes failure semantics and is not needed for correctness.
- **Multi-Plan-tree sessions.** One session = one PlanTree.
- **Tool surface expansion.** The executor's tool surface stays text-shaped (the existing constraint from the procedural-graph-memory spec).
- **Migration of existing sessions.** Existing sessions in `~/.whale/sessions/` are not retroactively planned. New tasks get an orchestrator; old ones stay as raw agent-loop trajectories.
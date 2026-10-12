from __future__ import annotations

import json
import sys
from typing import Any, Callable

import agent_loop
from logger import logger
from orchestrator_types import (
    CriticMode,
    CritiqueReport,
    OrchestratorResult,
    OrchestratorStatus,
    PlanNode,
    PlanTree,
    SubTaskResult,
)
from procedural_memory.graph import END, normalize_node_id

DEFAULT_MAX_TURNS = 3
DEFAULT_MAX_OUTER_ITERATIONS = 20
DEFAULT_MAX_REPLAN = 2

_PLAN_ATTEMPTS = 2
_GUIDANCE_LIMIT = 500

Llm = Callable[[str], Any]

_PLAN_PROMPT = """\
You are a planning agent. Write a plan whose steps accomplish the user's task \
for the user, as a tree of concrete steps.

Task: {task}
{feedback}
Background from past sessions (may be empty or unrelated; use it only if it \
clearly helps, and never mention its graph nodes, transitions or wording in the plan):
{recall_context}

Planning rules:
- Every step is something to do or produce for the user's task, in the user's domain.
- If the task is ambiguous or missing facts you need (goal, level, time budget, \
constraints), make the first steps clarifying questions to put to the user, \
written as concrete questions.

Output rules:
- Output ONLY one JSON object of this shape:
  {{"root": {{"id": "...", "title": "...", "task": "...", "acceptance": "...",
              "strict": false, "max_turns": null, "children": [ ... ]}}}}
- The tree has max depth 2 (root at depth 0, children at 1, grandchildren at 2). \
Depth-2 nodes must have empty children lists.
- Node ids are unique. Each node has exactly one `acceptance` criterion.
- `max_turns` is an integer executor budget; null uses the orchestrator default ({default_turns}).
- Set `strict: true` only on a node whose failure must halt the whole plan.
"""

_PLAN_RETRY_SUFFIX = (
    "\nYour previous reply was not a valid plan. Reply with the JSON object only, "
    "no prose and no code fences."
)

_CRITIC_PROMPT = """\
You are a plan critic. Review the plan against the user task.

Mode: {mode}
Task: {task}
Recall context (prior procedural guidance, may be empty): {recall_context}

Plan JSON:
{plan_json}

Output rules:
- Output ONLY one JSON object: {{"findings": [...], "blocking": [...]}}
- `findings`: every concern you have, as short advisory notes.
- `blocking`: only defects that would make the executor fail the root task, \
contradict the requirements, or leave the plan incoherent. Empty list approves.
"""


def _call_llm_json(prompt: str, llm: Llm | None) -> Any:
    """The model's parsed JSON reply, or None when the call itself fails."""
    try:
        if llm is not None:
            return llm(prompt)
        from models import completion_default_json

        return completion_default_json(prompt)
    except Exception as exc:
        logger.warning(f"orchestrator.llm_error: {exc}")
        return None


class Recall:
    @staticmethod
    def recall(task: str, memory_engine: Any) -> str:
        try:
            return memory_engine.retrieve_guidance(task, []) or ""
        except Exception as exc:  # degrade: plan without prior guidance
            logger.warning(f"orchestrator.recall_failed: {exc}")
            return ""


def _node_from_json(data: Any) -> PlanNode:
    if not isinstance(data, dict):
        raise ValueError("plan node must be an object")
    for key in ("id", "title", "task", "acceptance"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"plan node missing {key!r}")
    children = data.get("children") or []
    if not isinstance(children, list):
        raise ValueError("plan node 'children' must be a list")
    max_turns = data.get("max_turns")
    return PlanNode(
        id=data["id"],
        title=data["title"],
        task=data["task"],
        acceptance=data["acceptance"],
        children=[_node_from_json(child) for child in children],
        strict=data.get("strict") is True,
        max_turns=max_turns
        if isinstance(max_turns, int) and not isinstance(max_turns, bool) and max_turns > 0
        else None,
    )


def _tree_from_json(data: Any) -> PlanTree:
    if not isinstance(data, dict) or "root" not in data:
        raise ValueError("plan must be an object with a 'root' node")
    return PlanTree(root=_node_from_json(data["root"]))


def _plan_to_json(plan: PlanTree) -> dict[str, Any]:
    def encode(node: PlanNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "title": node.title,
            "task": node.task,
            "acceptance": node.acceptance,
            "strict": node.strict,
            "max_turns": node.max_turns,
            "children": [encode(child) for child in node.children],
        }

    return {"root": encode(plan.root)}


def _feedback_block(feedback: list[str] | None) -> str:
    if not feedback:
        return ""
    defects = "\n".join(f"- {item}" for item in feedback)
    return f"Fix these blocking defects found in your previous plan:\n{defects}\n"


class PlanAgent:
    @staticmethod
    def run(
        task: str,
        recall_context: str,
        *,
        feedback: list[str] | None = None,
        llm: Llm | None = None,
    ) -> PlanTree | None:
        prompt = _PLAN_PROMPT.format(
            task=task,
            recall_context=recall_context or "(none)",
            feedback=_feedback_block(feedback),
            default_turns=DEFAULT_MAX_TURNS,
        )
        for attempt in range(1, _PLAN_ATTEMPTS + 1):
            reply = _call_llm_json(prompt, llm)
            try:
                plan = _tree_from_json(reply)
            except ValueError as exc:
                logger.warning(f"orchestrator.plan_invalid: {exc} (attempt {attempt})")
                prompt += _PLAN_RETRY_SUFFIX
                continue
            logger.info("orchestrator.plan", root=plan.root.id, attempt=attempt)
            return plan
        return None


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


class CriticAgent:
    @staticmethod
    def run(
        task: str,
        plan: PlanTree,
        recall_context: str,
        *,
        mode: CriticMode,
        llm: Llm | None = None,
    ) -> CritiqueReport:
        prompt = _CRITIC_PROMPT.format(
            mode=mode,
            task=task,
            recall_context=recall_context or "(none)",
            plan_json=json.dumps(_plan_to_json(plan)),
        )
        reply = _call_llm_json(prompt, llm)
        if not isinstance(reply, dict):
            logger.warning("orchestrator.critic_unusable_reply")
            return CritiqueReport(findings=[], blocking=[], mode=mode)
        return CritiqueReport(
            findings=_strings(reply.get("findings")),
            blocking=_strings(reply.get("blocking")) if mode == "strict" else [],
            mode=mode,
        )


def _graph_id(text: str) -> str:
    return normalize_node_id(text) or "task"


def _dispatch_subtask(
    plan_node: PlanNode,
    *,
    memory_engine: Any,
    session_manager: Any,
    default_max_turns: int = DEFAULT_MAX_TURNS,
) -> SubTaskResult:
    session_id = session_manager.initSession().id
    max_turns = (
        plan_node.max_turns if plan_node.max_turns is not None else default_max_turns
    )
    try:
        result = agent_loop.run(
            task=plan_node.task,
            session_id=session_id,
            max_turns=max_turns,
            memory_engine=memory_engine,
            session_manager=session_manager,
        )
    except Exception as exc:
        logger.error(f"orchestrator.executor_error: {exc}", node=plan_node.id)
        return SubTaskResult(plan_node.id, session_id, "", "failed", 0)

    finished = bool(result.turns) and (
        agent_loop._verb(result.turns[-1].action) in agent_loop._FINISH_VERBS
    )
    return SubTaskResult(
        plan_node_id=plan_node.id,
        session_id=session_id,
        final_answer=result.final_answer,
        status="ok" if finished else "max_turns",
        turns=len(result.turns),
    )


def _record_subtask_triplet(
    root_task_id: str,
    plan_node: PlanNode,
    result: SubTaskResult,
    memory_engine: Any,
) -> None:
    pitfalls = (
        ""
        if result.status == "ok"
        else f"sub-task {plan_node.id!r} ended with status={result.status} "
        f"after {result.turns} turns"
    )
    memory_engine.record_triplet(
        _graph_id(root_task_id),
        "CONVERGES_TO",
        _graph_id(plan_node.id),
        condition=plan_node.acceptance,
        guidance=result.final_answer[:_GUIDANCE_LIMIT],
        pitfalls=pitfalls,
        source_session_id=result.session_id,
    )


def _record_plan_summary(
    root_task_id: str,
    executions: list[SubTaskResult],
    status: str,
    memory_engine: Any,
) -> str:
    src = _graph_id(root_task_id)
    pitfalls = "; ".join(
        f"sub-task {ex.plan_node_id!r} status={ex.status} turns={ex.turns}"
        for ex in executions
        if ex.status != "ok"
    )
    memory_engine.record_triplet(
        src,
        "LEADS_TO",
        END,
        condition=None,
        guidance=f"Plan finished with status={status}; "
        f"{len(executions)} sub-tasks executed.",
        pitfalls=pitfalls,
    )
    return f"{src}->{END}#LEADS_TO"


def _leaves(node: PlanNode) -> list[PlanNode]:
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _leaves(child)]


def render_plan(plan: PlanTree, critique: CritiqueReport) -> str:
    lines = ["Plan:"]

    def visit(node: PlanNode, depth: int) -> None:
        marker = " [strict]" if node.strict else ""
        lines.append(f"{'  ' * (depth + 1)}- {node.id}: {node.title}{marker}")
        lines.append(f"{'  ' * (depth + 2)}done when: {node.acceptance}")
        for child in node.children:
            visit(child, depth + 1)

    visit(plan.root, 0)
    lines.append(f"Critique ({critique.mode}):")
    lines.extend(f"  - {item}" for item in critique.findings or ["(no findings)"])
    lines.extend(f"  ! blocking: {item}" for item in critique.blocking)
    return "\n".join(lines)


def default_human_approval(plan: PlanTree, critique: CritiqueReport) -> bool:
    if not sys.stdin.isatty():
        return False
    print(render_plan(plan, critique))
    print("Approve this plan? [y/N]: ", end="", flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _failure(
    task: str,
    reason: str,
    plan: PlanTree | None = None,
    critique: CritiqueReport | None = None,
) -> OrchestratorResult:
    root = _graph_id(task)
    placeholder = PlanNode(id=root, title=task, task=task, acceptance="task done")
    return OrchestratorResult(
        status="failed",
        reason=reason,
        plan=plan or PlanTree(root=placeholder),
        critique=critique or CritiqueReport([], [], "advisory"),
        executions=[],
        plan_summary_triplet_id=None,
    )


def _plan_with_critique(
    task: str,
    recall_context: str,
    *,
    mode: CriticMode,
    max_replan: int,
    llm: Llm | None,
) -> tuple[PlanTree, CritiqueReport] | OrchestratorResult:
    """Plan and critique; in strict mode re-plan on blocking defects up to the cap."""
    feedback: list[str] | None = None
    for replan in range(max_replan + 1):
        plan = PlanAgent.run(task, recall_context, feedback=feedback, llm=llm)
        if plan is None:
            return _failure(task, "plan_parse_error")
        critique = CriticAgent.run(task, plan, recall_context, mode=mode, llm=llm)
        if not critique.blocking:
            return plan, critique
        logger.info("orchestrator.replan", replan=replan, blocking=len(critique.blocking))
        feedback = critique.blocking
    return _failure(task, "strict_replan_cap_reached", plan, critique)


def _execute(
    task: str,
    plan: PlanTree,
    *,
    max_outer_iterations: int,
    max_turns: int,
    memory_engine: Any,
    session_manager: Any,
    on_progress: Callable[[str], None],
) -> tuple[list[SubTaskResult], OrchestratorStatus, str | None]:
    executions: list[SubTaskResult] = []
    leaves = _leaves(plan.root)
    for node in leaves:
        if len(executions) >= max_outer_iterations:
            return executions, "max_outer_iterations", "max_outer_iterations"
        on_progress(f"[{len(executions) + 1}/{len(leaves)}] {node.id}: {node.title}")
        result = _dispatch_subtask(
            node,
            memory_engine=memory_engine,
            session_manager=session_manager,
            default_max_turns=max_turns,
        )
        executions.append(result)
        on_progress(f"      -> {result.status} ({result.turns} turns)")
        _record_subtask_triplet(task, node, result, memory_engine)
        logger.info(
            "orchestrator.iteration",
            iteration=len(executions),
            plan_node_id=node.id,
            status=result.status,
            executor_session_id=result.session_id,
        )
        if node.strict and result.status != "ok":
            return executions, "failed", f"strict_subtask_failed:{node.id}"
    all_ok = all(ex.status == "ok" for ex in executions)
    return executions, "ok" if all_ok else "partial", None


def run(
    task: str,
    *,
    strict: bool = False,
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
    max_replan: int = DEFAULT_MAX_REPLAN,
    max_turns: int = DEFAULT_MAX_TURNS,
    on_human_approval: Callable[[PlanTree, CritiqueReport], bool] | None = None,
    on_progress: Callable[[str], None] | None = None,
    llm: Llm | None = None,
    memory_engine: Any = None,
    session_manager: Any = None,
) -> OrchestratorResult:
    """Recall -> Plan -> Critic -> human gate -> dispatch each leaf -> record."""
    if memory_engine is None:
        import memory

        memory_engine = memory.memoryEngine
    if session_manager is None:
        import session

        session_manager = session.sessionManager

    recall_context = Recall.recall(task, memory_engine)
    planned = _plan_with_critique(
        task,
        recall_context,
        mode="strict" if strict else "advisory",
        max_replan=max_replan,
        llm=llm,
    )
    if isinstance(planned, OrchestratorResult):
        return planned
    plan, critique = planned

    if on_human_approval is not None and not _approved(on_human_approval, plan, critique):
        return OrchestratorResult(
            status="rejected_by_human",
            reason="human_rejected_root_plan",
            plan=plan,
            critique=critique,
            executions=[],
            plan_summary_triplet_id=None,
        )

    executions, status, reason = _execute(
        task,
        plan,
        max_outer_iterations=max_outer_iterations,
        max_turns=max_turns,
        memory_engine=memory_engine,
        session_manager=session_manager,
        on_progress=on_progress or (lambda _line: None),
    )
    summary_id = _record_plan_summary(task, executions, status, memory_engine)
    return OrchestratorResult(
        status=status,
        reason=reason,
        plan=plan,
        critique=critique,
        executions=executions,
        plan_summary_triplet_id=summary_id,
    )


def _approved(
    gate: Callable[[PlanTree, CritiqueReport], bool],
    plan: PlanTree,
    critique: CritiqueReport,
) -> bool:
    try:
        return bool(gate(plan, critique))
    except Exception as exc:  # EOF / interrupted prompt counts as no approval
        logger.warning(f"orchestrator.approval_failed: {exc}")
        return False

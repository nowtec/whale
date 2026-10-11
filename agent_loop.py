from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import memory as memory_module
import session as session_module

SYSTEM_PROMPT = (
    "You are an autonomous agent. Solve the user's task step by step, "
    "using one tool action per step. When you can answer the task, finish "
    'with Action: FINISH(answer="<your final answer>").'
)

# Paper §B.5, "Solver Execution Prompt (ReAct loop)".
SOLVER_PROMPT = """\
{system_prompt}
Procedural Graph Guidance: {procedural_graph_guidance}
You must interleave Thought and Action. Your output format must be exactly:
Thought: <your reasoning about what to do next>
Action: <tool_name>(arg1=val1, arg2=val2, ...)
Example:
Thought: I need to check the files in the workspace directory to locate the source documents.
Action: list_dir(path=".")
DO NOT write any “Observation:” block or any subsequent steps. Only output exactly one Thought and one Action block. Do NOT simulate the environment’s responses.
Current Trajectory: {trajectory}
Thought:"""

_FINISH_VERBS = {"FINISH", "DONE"}
_RECENT_ACTIONS = 3


@dataclass
class TurnOutput:
    thought: str
    action: str
    raw: str

    @classmethod
    def from_message(cls, msg: str) -> TurnOutput:
        thought_lines: list[str] = []
        action = ""
        in_thought = False
        for line in msg.splitlines():
            stripped = line.strip()
            if stripped.startswith("Action:"):
                action = stripped[len("Action:") :].strip()
                break  # anything after the first Action is simulated output
            if stripped.startswith("Thought:"):
                in_thought = True
                thought_lines.append(stripped[len("Thought:") :].strip())
            elif in_thought:
                thought_lines.append(stripped)
        return cls(thought="\n".join(thought_lines).strip(), action=action, raw=msg)


@dataclass
class RunResult:
    session_id: str
    turns: list[TurnOutput] = field(default_factory=list)
    final_answer: str = ""
    closed: bool = False


def _verb(action: str) -> str:
    return action.split("(", 1)[0].strip().upper()


def _finish_answer(action: str) -> str:
    """The answer carried by FINISH(...)/DONE(...); '' when there is none."""
    if _verb(action) not in _FINISH_VERBS or "(" not in action:
        return ""
    args = action.split("(", 1)[1].rsplit(")", 1)[0].strip()
    if args.lower().startswith("answer="):
        args = args[len("answer=") :].strip()
    if len(args) >= 2 and args[0] == args[-1] and args[0] in "\"'":
        args = args[1:-1]
    return args.strip()


def _render_trajectory(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)


def _default_solver(prompt: str) -> str | None:
    from models import completion_default

    return completion_default(prompt)


def run(
    task: str,
    session_id: str,
    max_turns: int = 3,
    *,
    on_observation: Callable[[str], str] | None = None,
    evolve_on_close: bool = False,
    solver: Callable[[str], str | None] | None = None,
    memory_engine: Any | None = None,
    session_manager: Any | None = None,
) -> RunResult:
    engine = memory_engine if memory_engine is not None else memory_module.memoryEngine
    sessions = (
        session_manager
        if session_manager is not None
        else session_module.sessionManager
    )
    solve = solver or _default_solver

    sessions.appendMessage(session_id, {"role": "user", "content": task})

    turns: list[TurnOutput] = []
    actions: list[str] = []
    for turn_no in range(1, max_turns + 1):
        guidance = engine.retrieve_guidance(task, actions[-_RECENT_ACTIONS:])
        prompt = SOLVER_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            procedural_graph_guidance=guidance or "(none)",
            trajectory=_render_trajectory(sessions.loadSession(session_id)),
        )
        message = solve(prompt) or ""
        turn = TurnOutput.from_message(message)

        observation = (
            on_observation(turn.action)
            if on_observation is not None
            else f"[stub-obs] {turn.action}"
        )
        sessions.appendMessage(session_id, {"role": "assistant", "content": message})
        sessions.appendMessage(session_id, {"role": "tool", "content": observation})
        engine.record_turn(
            session_id=session_id,
            thought=turn.thought,
            action=turn.action,
            observation=observation,
        )

        turns.append(turn)
        if turn.action:
            actions.append(turn.action)
        if _verb(turn.action) in _FINISH_VERBS or turn_no == max_turns:
            break

    sessions.close_session(
        session_id, evolve=evolve_on_close, memory_engine=engine
    )
    final_answer = next(
        (_finish_answer(t.action) for t in reversed(turns) if _finish_answer(t.action)),
        "",
    ) or next((t.thought for t in reversed(turns) if t.thought), "")
    return RunResult(
        session_id=session_id, turns=turns, final_answer=final_answer, closed=True
    )


def run_with_guidance(
    task: str,
    session_id: str | None = None,
    max_turns: int = 1,
    **kwargs: Any,
) -> RunResult:
    """Single-shot (no loop by default) run that still uses graph guidance."""
    if session_id is None:
        manager = kwargs.get("session_manager") or session_module.sessionManager
        session_id = manager.initSession().id
    return run(task, session_id, max_turns, **kwargs)

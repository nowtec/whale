from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, get_args

MAX_PLAN_DEPTH = 2

SubTaskStatus = Literal["ok", "failed", "max_turns"]
OrchestratorStatus = Literal[
    "ok", "partial", "failed", "rejected_by_human", "max_outer_iterations"
]
CriticMode = Literal["advisory", "strict"]


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class PlanNode:
    id: str
    title: str
    task: str
    acceptance: str
    children: list[PlanNode] = field(default_factory=list)
    strict: bool = False
    max_turns: int | None = None


def _validate(node: PlanNode, depth: int, seen: set[str]) -> None:
    if depth > MAX_PLAN_DEPTH:
        raise ValueError(
            f"plan node {node.id!r} at depth {depth} exceeds max depth {MAX_PLAN_DEPTH}"
        )
    if node.id in seen:
        raise ValueError(f"duplicate plan node id {node.id!r}")
    seen.add(node.id)
    for child in node.children:
        _validate(child, depth + 1, seen)


@dataclass(frozen=True)
class PlanTree:
    root: PlanNode
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        _validate(self.root, 0, set())


@dataclass(frozen=True)
class CritiqueReport:
    findings: list[str]
    blocking: list[str]
    mode: CriticMode


@dataclass(frozen=True)
class SubTaskResult:
    plan_node_id: str
    session_id: str
    final_answer: str
    status: SubTaskStatus
    turns: int

    def __post_init__(self) -> None:
        if self.status not in get_args(SubTaskStatus):
            raise ValueError(f"invalid sub-task status {self.status!r}")


@dataclass(frozen=True)
class OrchestratorResult:
    status: OrchestratorStatus
    reason: str | None
    plan: PlanTree
    critique: CritiqueReport
    executions: list[SubTaskResult]
    plan_summary_triplet_id: str | None

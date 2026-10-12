from __future__ import annotations

import pytest

from orchestrator_types import (
    CritiqueReport,
    OrchestratorResult,
    PlanNode,
    PlanTree,
    SubTaskResult,
)


def _node(node_id: str, *children: PlanNode) -> PlanNode:
    return PlanNode(
        id=node_id,
        title=node_id,
        task=node_id,
        acceptance=f"{node_id} done",
        children=list(children),
    )


def test_plan_node_is_frozen():
    node = _node("a")
    with pytest.raises(AttributeError):
        node.title = "changed"


def test_plan_tree_accepts_depth_2():
    PlanTree(root=_node("a", _node("b", _node("c"))))


def test_plan_tree_rejects_depth_3():
    too_deep = _node("a", _node("b", _node("c", _node("d"))))
    with pytest.raises(ValueError, match="depth"):
        PlanTree(root=too_deep)


def test_plan_tree_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        PlanTree(root=_node("a", _node("b"), _node("b")))


def test_plan_tree_stamps_created_at():
    assert PlanTree(root=_node("a")).created_at.endswith("Z")


def test_critique_report_fields():
    report = CritiqueReport(findings=["note"], blocking=["defect"], mode="strict")
    assert (report.findings, report.blocking, report.mode) == (
        ["note"],
        ["defect"],
        "strict",
    )


@pytest.mark.parametrize("status", ["ok", "failed", "max_turns"])
def test_subtask_result_accepts_documented_statuses(status):
    SubTaskResult("p", "s", "", status, 1)


def test_subtask_result_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        SubTaskResult("p", "s", "", "bogus", 1)


def test_orchestrator_result_carries_reason():
    result = OrchestratorResult(
        status="failed",
        reason="plan_parse_error",
        plan=PlanTree(root=_node("a")),
        critique=CritiqueReport([], [], "advisory"),
        executions=[],
        plan_summary_triplet_id=None,
    )
    assert (result.status, result.reason) == ("failed", "plan_parse_error")

from __future__ import annotations

import json

import networkx as nx
import pytest

from procedural_memory import refiner as refiner_mod
from procedural_memory.graph import END, START, ProceduralGraph, new_skeleton
from procedural_memory.refiner import (
    EditDelta,
    RefinerParseError,
    Refiner,
    RejectionLog,
    prepare_candidate,
    tail_clip,
)

PREFIX = "Could not validate a linked graph memory"


def _edge(src, tgt, relation="LEADS_TO", guidance="Do it.", pitfalls="Careful."):
    return {
        "source": src,
        "target": tgt,
        "relation": relation,
        "condition": None,
        "guidance": guidance,
        "pitfalls": pitfalls,
    }


def _node(node_id, node_type="ACTION"):
    return {"id": node_id, "type": node_type, "description": f"{node_id} node"}


VALID = {
    "add_nodes": [_node("search")],
    "delete_nodes": [],
    "add_edges": [_edge(START, "search"), _edge("search", END)],
    "delete_edges": [{"source": START, "target": END}],
}

DANGLING = {
    "add_nodes": [],
    "delete_nodes": [],
    "add_edges": [_edge(START, "ghost")],
    "delete_edges": [],
}


class _Llm:
    def __init__(self, *replies, exc=None):
        self.replies = list(replies)
        self.exc = exc
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture
def warnings(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        refiner_mod.logger, "warning", lambda msg, *a, **k: seen.append(msg)
    )
    return seen


def _make(tmp_path, llm, **kw):
    pg = ProceduralGraph(tmp_path / "g.json")
    return pg, Refiner(pg, tmp_path / "rej.jsonl", llm=llm, **kw)


def _rejections(tmp_path):
    path = tmp_path / "rej.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# --- tail_clip -------------------------------------------------------------


def test_tail_clip_keeps_ending():
    text = "".join(f"line-{i}\n" for i in range(5000))
    clipped = tail_clip(text, max_tokens=100)
    assert len(clipped) <= 400
    assert clipped.endswith(text[-50:])


def test_tail_clip_short_unchanged():
    assert tail_clip("short", max_tokens=100) == "short"


# --- EditDelta -------------------------------------------------------------


def test_edit_delta_from_json_ok():
    delta = EditDelta.from_json(VALID)
    assert [n.id for n in delta.add_nodes] == ["search"]
    assert delta.delete_edges == [(START, END)]
    assert delta.add_edges[0].relation == "LEADS_TO"
    assert delta.to_json()["add_nodes"][0]["id"] == "search"


def test_edit_delta_missing_arrays_are_empty():
    delta = EditDelta.from_json({})
    assert delta.add_nodes == [] and delta.delete_edges == []


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"add_nodes": "x"},
        {"add_edges": [{"source": "a", "target": "b", "relation": "LEADS_TO"}]},
        {"delete_edges": [{"source": "a"}]},
        {"add_nodes": [{"type": "ACTION"}]},
        {"add_edges": [{**_edge("a", "b"), "condition": 5}]},
    ],
)
def test_edit_delta_from_json_rejects_shapes(data):
    with pytest.raises(RefinerParseError):
        EditDelta.from_json(data)


# --- prepare_candidate -----------------------------------------------------


def test_prepare_valid_candidate():
    candidate, diags = prepare_candidate(
        new_skeleton(), EditDelta.from_json(VALID), refiner_round=4
    )
    assert diags == []
    assert not candidate.has_edge(START, END)
    assert candidate.has_edge(START, "search", key="LEADS_TO")
    assert candidate.nodes["search"]["provenance"]["refiner_round"] == 4


def test_delete_edges_spans_relations_and_add_edges_wins(tmp_path):
    pg = ProceduralGraph(tmp_path / "g.json")
    for relation in ("LEADS_TO", "TRIGGERS"):
        pg.add_triplet(
            "a", relation, "b", condition=None, guidance="g", pitfalls="p"
        )
    delta = EditDelta.from_json(
        {
            "delete_edges": [{"source": "a", "target": "b"}],
            "add_edges": [_edge("a", "b")],
        }
    )
    candidate, diags = prepare_candidate(pg.G, delta)
    assert diags == []
    assert list(candidate["a"]["b"]) == ["LEADS_TO"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(DANGLING, id="missing-endpoint"),
        pytest.param(
            {"add_edges": [_edge(START, END, relation="FOO")]}, id="unknown-relation"
        ),
        pytest.param(
            {"add_nodes": [_node("x", node_type="BOGUS")]}, id="unknown-node-type"
        ),
        pytest.param({"delete_nodes": [START]}, id="delete-start"),
        pytest.param({"delete_nodes": [END]}, id="delete-end"),
    ],
)
def test_prepare_rejects(data):
    _, diags = prepare_candidate(new_skeleton(), EditDelta.from_json(data))
    assert diags


def test_exit_less_cycle_rejected_when_cycles_allowed():
    delta = EditDelta.from_json(
        {
            "add_nodes": [_node("a"), _node("b")],
            "add_edges": [_edge(START, "a"), _edge("a", "b"), _edge("b", "a")],
        }
    )
    _, diags = prepare_candidate(new_skeleton(), delta, cycles_forbidden=False)
    assert diags and "a" in diags[-1] and "b" in diags[-1]


def _cycle_delta():
    return EditDelta.from_json(
        {
            "add_nodes": [_node("a"), _node("b")],
            "add_edges": [
                _edge(START, "a"),
                _edge("a", "b"),
                _edge("b", "a"),
                _edge("b", END),
            ],
        }
    )


def test_cycle_closing_edge_removed_when_forbidden():
    candidate, diags = prepare_candidate(new_skeleton(), _cycle_delta())
    assert diags == []
    assert nx.is_directed_acyclic_graph(candidate)
    assert candidate.has_edge("a", "b") and not candidate.has_edge("b", "a")


def test_cycle_kept_when_allowed():
    candidate, diags = prepare_candidate(
        new_skeleton(), _cycle_delta(), cycles_forbidden=False
    )
    assert diags == []
    assert candidate.has_edge("b", "a")


def test_auto_stub_missing_nodes():
    candidate, diags = prepare_candidate(
        new_skeleton(),
        EditDelta.from_json(DANGLING),
        auto_stub_missing_nodes=True,
    )
    assert diags == []
    assert candidate.nodes["ghost"]["node_type"] == "STATUS"


# --- Refiner.evolve --------------------------------------------------------


def test_evolve_accepts_valid_edit(tmp_path):
    pg, refiner = _make(tmp_path, _Llm(VALID))
    result = refiner.evolve("user: hi", round_idx=3)
    assert result.accepted is True
    assert result.score_after >= result.score_before
    assert (result.candidate_added, result.candidate_removed) == (3, 1)
    reloaded = ProceduralGraph(tmp_path / "g.json")
    assert "search" in reloaded.G
    assert reloaded.G.nodes["search"]["provenance"]["refiner_round"] == 3
    assert _rejections(tmp_path) == []


def test_evolve_structural_reject_leaves_file_untouched(tmp_path, warnings):
    pg, refiner = _make(tmp_path, _Llm(DANGLING))
    pg.save()
    before = (tmp_path / "g.json").read_bytes()
    result = refiner.evolve("user: hi", round_idx=2)
    assert result.accepted is False
    assert result.rejection_reason.startswith("structural:")
    assert (tmp_path / "g.json").read_bytes() == before
    entries = _rejections(tmp_path)
    assert len(entries) == 1
    assert entries[0]["round"] == 2
    assert entries[0]["reason"] == result.rejection_reason
    assert any(w.startswith(f"{PREFIX}: structural reject:") for w in warnings)


def test_evolve_score_reject_with_strict_false(tmp_path):
    pg, refiner = _make(tmp_path, _Llm(DANGLING), strict=False)
    pg.save()
    before = (tmp_path / "g.json").read_bytes()
    result = refiner.evolve("user: hi")
    assert result.accepted is False
    assert result.rejection_reason.startswith("score:")
    assert result.score_after < result.score_before
    assert (tmp_path / "g.json").read_bytes() == before
    assert len(_rejections(tmp_path)) == 1


def test_evolve_llm_error(tmp_path, warnings):
    pg, refiner = _make(tmp_path, _Llm(exc=RuntimeError("down")))
    result = refiner.evolve("user: hi")
    assert result.accepted is False
    assert result.rejection_reason.startswith("refiner_unreachable")
    assert any(w.startswith(f"{PREFIX}: refiner unreachable: ") for w in warnings)
    assert _rejections(tmp_path) == []
    assert not (tmp_path / "g.json").exists()


@pytest.mark.parametrize("reply", [["x"], {"add_nodes": "x"}, "text"])
def test_evolve_parse_error(tmp_path, reply):
    pg, refiner = _make(tmp_path, _Llm(reply))
    result = refiner.evolve("user: hi")
    assert result.accepted is False
    assert result.rejection_reason.startswith("parse_error")
    assert _rejections(tmp_path) == []


def test_evolve_malformed_json_from_provider_is_parse_error(tmp_path):
    pg, refiner = _make(tmp_path, _Llm(exc=ValueError("invalid JSON")))
    result = refiner.evolve("user: hi")
    assert result.rejection_reason.startswith("parse_error")


def test_evolve_corrupt_graph_file(tmp_path, warnings):
    (tmp_path / "g.json").write_text("{not json")
    llm = _Llm(VALID)
    pg, refiner = _make(tmp_path, llm)
    result = refiner.evolve("user: hi")
    assert result.rejection_reason == "load_failed"
    assert llm.prompts == []
    assert (tmp_path / "g.json").read_text() == "{not json"
    assert any("load failed" in w for w in warnings)


def test_rejected_history_in_next_prompt(tmp_path):
    llm = _Llm(DANGLING, VALID)
    pg, refiner = _make(tmp_path, llm)
    first = refiner.evolve("user: hi", round_idx=1)
    refiner.evolve("user: hi", round_idx=2)
    assert first.rejection_reason in llm.prompts[1]
    assert first.rejection_reason not in llm.prompts[0]


def test_rejections_corrupt_line_skipped(tmp_path):
    path = tmp_path / "rej.jsonl"
    path.write_text("garbage\n" + json.dumps({"reason": "structural:x"}) + "\n[1]\n")
    assert RejectionLog(path).recent() == [{"reason": "structural:x"}]
    pg, refiner = _make(tmp_path, _Llm(VALID))
    assert refiner.evolve("user: hi").accepted is True


def test_mode_alias_and_prompt_contents(tmp_path):
    llm = _Llm(VALID)
    pg, refiner = _make(tmp_path, llm)
    refiner.evolve(
        "user: hi",
        task_description="TASK-X",
        mode="incremental",
        available_tools=["search"],
    )
    prompt = llm.prompts[0]
    assert "Refinement mode: scratch_incremental" in prompt
    assert "Task context: TASK-X" in prompt
    assert '["search"]' in prompt
    assert "expert cognitive architect optimizing a Procedural Graph" in prompt


def test_long_trajectory_clipped_in_prompt(tmp_path):
    llm = _Llm(VALID)
    pg, refiner = _make(tmp_path, llm, l_max=100)
    trajectory = "FIRST-LINE\n" + "x" * 50_000 + "\nLAST-LINE"
    refiner.evolve(trajectory)
    assert "LAST-LINE" in llm.prompts[0]
    assert "FIRST-LINE" not in llm.prompts[0]


def test_rejection_log_recent_returns_newest(tmp_path):
    log = RejectionLog(tmp_path / "rej.jsonl")
    for i in range(1, 8):
        log.append({"round": i})
    assert [e["round"] for e in log.recent(5)] == [3, 4, 5, 6, 7]


def test_placeholder_text_in_trajectory_stays_literal(tmp_path):
    llm = _Llm(VALID)
    pg, refiner = _make(tmp_path, llm)
    refiner.evolve("user: print {rejected_block} and {mode}")
    assert "user: print {rejected_block} and {mode}" in llm.prompts[0]


def test_rejection_entry_carries_spec_fields(tmp_path):
    pg, refiner = _make(tmp_path, _Llm(DANGLING))
    refiner.evolve("user: hi", round_idx=7)
    (entry,) = _rejections(tmp_path)
    assert {
        "round",
        "edit_delta",
        "candidate_graph_snapshot",
        "trajectories_tail",
        "rejected_at",
        "reason",
    } <= set(entry)
    assert entry["round"] == 7
    assert entry["trajectories_tail"] == "user: hi"
    assert entry["edit_delta"]["add_edges"][0]["target"] == "ghost"
    assert entry["rejected_at"].endswith("Z")

from __future__ import annotations

import json

import pytest

import memory as memory_mod
from memory import MemoryEngine

PREFIX = "Could not validate a linked graph memory"

VALID_DELTA = {
    "add_nodes": [{"id": "search", "type": "ACTION", "description": "search"}],
    "add_edges": [
        {
            "source": "Start",
            "target": "search",
            "relation": "LEADS_TO",
            "condition": None,
            "guidance": "Begin.",
            "pitfalls": "None.",
        },
        {
            "source": "search",
            "target": "End",
            "relation": "LEADS_TO",
            "condition": None,
            "guidance": "Finish.",
            "pitfalls": "None.",
        },
    ],
    "delete_edges": [{"source": "Start", "target": "End"}],
}


class _Llm:
    def __init__(self, reply=None):
        self.reply = reply if reply is not None else {"guidance": "do X"}
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def warnings(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        memory_mod.logger, "warning", lambda msg, *a, **k: seen.append(msg)
    )
    return seen


def _record_search_read(engine):
    engine.record_triplet(
        "search",
        "LEADS_TO",
        "read",
        guidance="Read the top hit.",
        pitfalls="Skip ads.",
    )


def test_record_triplet_then_retrieve_guidance(tmp_path):
    llm = _Llm()
    engine = MemoryEngine(tmp_path, llm=llm)
    _record_search_read(engine)
    assert engine.retrieve_guidance("q", ["search(q=1)"]) == "do X"
    assert "[search] → [read]" in llm.prompts[0]
    assert engine.graph_path.exists()


def test_retrieve_on_fresh_engine_returns_empty(tmp_path, warnings):
    llm = _Llm()
    engine = MemoryEngine(tmp_path, llm=llm)
    assert engine.retrieve_guidance("q", []) == ""
    assert llm.prompts == []
    assert warnings == [f"{PREFIX}: graph has no procedural guidance yet"]


def test_corrupt_graph_file_degrades(tmp_path, warnings):
    llm = _Llm()
    engine = MemoryEngine(tmp_path, llm=llm)
    engine.graph_path.write_text("{bad")
    assert engine.retrieve_guidance("q", ["search()"]) == ""
    _record_search_read(engine)  # must not raise or clobber the file
    assert engine.record_turn("s1", "t", "search(q=1)", "obs") == []
    assert engine.graph_path.read_text() == "{bad"
    assert llm.prompts == []
    assert len(warnings) == 3
    assert all(w.startswith(f"{PREFIX}: load failed: ") for w in warnings)


def test_record_turn_chain(tmp_path):
    engine = MemoryEngine(tmp_path)
    triples = []
    for thought, action in [
        ("look it up", "search(q=1)"),
        ("open result", "read(id=2)"),
        ("done", "FINISH()"),
    ]:
        triples += engine.record_turn("s1", thought, action, "obs")
    assert triples == [
        ("Start", "LEADS_TO", "search"),
        ("search", "LEADS_TO", "read"),
        ("read", "LEADS_TO", "End"),
    ]
    data = json.loads(MemoryEngine(tmp_path).graph_path.read_text())  # persisted
    nodes = {n["id"]: n for n in data["nodes"]}
    assert nodes["search"]["description"] == "search(q=1)"
    edge = next(
        e for e in data["edges"] if (e["source"], e["target"]) == ("Start", "search")
    )
    assert edge["guidance"] == "look it up"
    assert edge["condition"] is None


def test_record_turn_edge_cases(tmp_path):
    engine = MemoryEngine(tmp_path)
    assert engine.record_turn("s1", "t", "", "o") == []
    assert engine.record_turn("s1", "t", "()", "o") == []
    assert not engine.graph_path.exists()
    assert engine.record_turn("s1", "t", "search(q=1)", "o") == [
        ("Start", "LEADS_TO", "search")
    ]
    assert engine.record_turn("s1", "t", "search(q=2)", "o") == []  # no self-loop
    assert engine.record_turn("s1", "", "read()", "o") == [
        ("search", "LEADS_TO", "read")
    ]
    data = json.loads(engine.graph_path.read_text())
    guidance = {(e["source"], e["target"]): e["guidance"] for e in data["edges"]}
    assert guidance[("search", "read")] == "Proceed with read()"


def test_record_turn_sessions_are_independent(tmp_path):
    engine = MemoryEngine(tmp_path)
    assert engine.record_turn("a", "t", "search()", "o") == [
        ("Start", "LEADS_TO", "search")
    ]
    assert engine.record_turn("b", "t", "read()", "o") == [
        ("Start", "LEADS_TO", "read")
    ]


def test_evolve_from_trajectory_uses_session_messages(tmp_path):
    messages = [
        {"role": "user", "content": "Find the capital of France"},
        {"role": "assistant", "content": "Thought: look\nAction: Search(q=1)"},
        {"role": "tool", "content": "[stub-obs] Search(q=1)"},
        {"role": "assistant", "content": "Thought: done\nAction: FINISH()"},
    ]
    seen = []
    llm = _Llm(reply={})
    engine = MemoryEngine(
        tmp_path,
        llm=llm,
        session_loader=lambda sid: seen.append(sid) or messages,
    )
    result = engine.evolve_from_trajectory("s1", round_idx=4)
    assert seen == ["s1"]
    assert result.accepted is True and result.round_idx == 4
    prompt = llm.prompts[0]
    assert "Task context: Find the capital of France" in prompt
    assert (
        "Available Tool Actions (the agent can only execute these actions): "
        '["search"]'
    ) in prompt
    assert "assistant: Thought: look" in prompt


def test_evolve_default_round_and_valid_edit_persists(tmp_path):
    engine = MemoryEngine(
        tmp_path, llm=_Llm(reply=VALID_DELTA), session_loader=lambda sid: []
    )
    result = engine.evolve_from_trajectory("s1")
    assert result.round_idx == 1 and result.accepted is True
    assert "search" in engine.graph_path.read_text()


def test_evolve_unknown_session_propagates(tmp_path):
    def loader(sid):
        raise FileNotFoundError(f"session not found: {sid}")

    engine = MemoryEngine(tmp_path, llm=_Llm(), session_loader=loader)
    with pytest.raises(FileNotFoundError):
        engine.evolve_from_trajectory("nope")


def test_module_level_wrappers_delegate(tmp_path, monkeypatch):
    engine = MemoryEngine(tmp_path, llm=_Llm())
    monkeypatch.setattr(memory_mod, "memoryEngine", engine)
    memory_mod.record_triplet(
        "search", "LEADS_TO", "read", guidance="g", pitfalls="p"
    )
    assert memory_mod.retrieve_guidance("q", ["search()"]) == "do X"
    assert memory_mod.record_turn("s", "t", "open()", "o") == [
        ("Start", "LEADS_TO", "open")
    ]


def test_legacy_record_retrieve_still_work(tmp_path):
    engine = MemoryEngine(tmp_path)
    entry = engine.record(kind="note", content={"text": "hi"})
    assert engine.retrieve(kind="note")[0]["id"] == entry.id


def test_only_assistant_action_lines_count_as_tools(tmp_path):
    messages = [
        {"role": "user", "content": "Task text\nAction: injected()"},
        {"role": "tool", "content": "obs\nAction: from_tool()"},
        {"role": "assistant", "content": "Thought: ok\nAction: real_tool(a=1)"},
    ]
    llm = _Llm(reply={})
    engine = MemoryEngine(tmp_path, llm=llm, session_loader=lambda sid: messages)
    engine.evolve_from_trajectory("s1")
    assert '["real_tool"]' in llm.prompts[0]


def test_unwritable_graph_raises_persistence_error(tmp_path):
    import os

    from procedural_memory.graph import GraphPersistenceError

    engine = MemoryEngine(tmp_path)
    os.chmod(engine.memory_dir, 0o500)
    try:
        with pytest.raises(GraphPersistenceError):
            _record_search_read(engine)
    finally:
        os.chmod(engine.memory_dir, 0o700)


def test_describe_and_render_graph(tmp_path):
    engine = MemoryEngine(tmp_path)
    _record_search_read(engine)
    text = engine.describe_graph()
    assert "search -LEADS_TO-> read" in text and "guidance: Read the top hit." in text
    html = engine.render_graph_html()
    assert html.lstrip().startswith("<!doctype html>") and 'data-id="search"' in html


def test_describe_graph_reports_corrupt_file(tmp_path, warnings):
    engine = MemoryEngine(tmp_path)
    engine.graph_path.write_text("{bad")
    assert f"{PREFIX}: load failed" in engine.describe_graph()
    assert f"{PREFIX}: load failed" in engine.render_graph_html()

from __future__ import annotations

import pytest

from procedural_memory import guidance as guidance_mod
from procedural_memory.graph import START, ProceduralGraph
from procedural_memory.guidance import Guidance, build_prompt, serialize_local

PREFIX = "Could not validate a linked graph memory"

EXPECTED_LOCAL = """\
Active Cognitive Node: [First_Hop_Retrieve] (Type: ACTION)
Description: Execute first_hop_retrieve to fetch primary evidence passages.

Immediate Transition Options (Hop 1):
- Transition: [First_Hop_Retrieve] → [Scan_Index] (Condition: first_hop_retrieve)
  * Guidance: Review the retrieved passages.
  * Pitfalls to Avoid: Do not skip reading evidence details.

Subsequent Horizon (Hop 2):
- Transition: [Scan_Index] → [Bridge_Extract] (Condition: (always))
  * Guidance: Extract the bridge term.
  * Pitfalls to Avoid: Match Wikipedia capitalization."""


@pytest.fixture
def warnings(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        guidance_mod.logger, "warning", lambda msg, *a, **k: seen.append(msg)
    )
    return seen


@pytest.fixture
def pg(tmp_path):
    return ProceduralGraph(tmp_path / "g.json")


def _hotpot(pg):
    pg.add_triplet(
        "First_Hop_Retrieve",
        "LEADS_TO",
        "Scan_Index",
        condition="first_hop_retrieve",
        guidance="Review the retrieved passages.",
        pitfalls="Do not skip reading evidence details.",
    )
    pg.add_triplet(
        "Scan_Index",
        "PROVIDES_INPUT_FOR",
        "Bridge_Extract",
        condition=None,
        guidance="Extract the bridge term.",
        pitfalls="Match Wikipedia capitalization.",
    )
    pg.G.nodes["First_Hop_Retrieve"][
        "description"
    ] = "Execute first_hop_retrieve to fetch primary evidence passages."


def _simple(pg):
    pg.add_triplet(
        "search",
        "LEADS_TO",
        "read",
        condition=None,
        guidance="Read the top hit.",
        pitfalls="Skip ads.",
    )


class _Llm:
    def __init__(self, reply=None, exc=None):
        self.reply = reply if reply is not None else {"guidance": "do X"}
        self.exc = exc
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if self.exc is not None:
            raise self.exc
        return self.reply


def test_serialize_local_exact(pg):
    _hotpot(pg)
    G_t = pg.h_hop_neighborhood("First_Hop_Retrieve", 2)
    assert serialize_local("First_Hop_Retrieve", G_t) == EXPECTED_LOCAL


def test_serialize_hides_relation_labels(pg):
    _hotpot(pg)
    text = serialize_local(
        "First_Hop_Retrieve", pg.h_hop_neighborhood("First_Hop_Retrieve", 2)
    )
    assert "LEADS_TO" not in text
    assert "PROVIDES_INPUT_FOR" not in text


def test_serialize_full_graph_when_active_none(pg):
    _simple(pg)
    text = serialize_local(None, pg.G)
    assert "Active Cognitive Node" not in text
    assert text.startswith("Immediate Transition Options (Hop 1):")
    assert "- Transition: [search] → [read] (Condition: (always))" in text


def test_build_prompt_variants():
    local = build_prompt("T", "GRAPH", "Q", "CTX", full_graph=False)
    full = build_prompt("T", "GRAPH", "Q", "CTX", full_graph=True)
    for prompt in (local, full):
        assert "GRAPH" in prompt and "Q" in prompt and "CTX" in prompt
        assert "expert cognitive architect and execution guide" in prompt
        assert '"guidance"' in prompt
    assert "complete Procedural Graph" in full
    assert "complete Procedural Graph" not in local


def test_retrieve_success(pg, warnings):
    _simple(pg)
    llm = _Llm()
    g = Guidance(pg, llm=llm)
    out = g.retrieve("my task", ["search(q=1)"], recent_context="ctx-line")
    assert out == "do X"
    prompt = llm.prompts[0]
    assert "my task" in prompt
    assert "- Transition: [search] → [read]" in prompt
    assert "ctx-line" in prompt
    assert warnings == []


def test_retrieve_skeleton_returns_empty_and_warns(pg, warnings):
    llm = _Llm()
    assert Guidance(pg, llm=llm).retrieve("t", []) == ""
    assert warnings == [f"{PREFIX}: graph has no procedural guidance yet"]
    assert llm.prompts == []


def test_retrieve_miss_falls_back_to_full_graph(pg, warnings):
    _simple(pg)
    llm = _Llm()
    assert Guidance(pg, llm=llm).retrieve("t", ["zzz()"]) == "do X"
    assert "complete Procedural Graph" in llm.prompts[0]
    assert warnings == [
        f"{PREFIX}: localized node not found, falling back to full graph"
    ]


def test_retrieve_empty_actions_is_hit_on_start(pg, warnings):
    pg.add_triplet(
        START, "LEADS_TO", "search", condition=None, guidance="Begin.", pitfalls=""
    )
    llm = _Llm()
    assert Guidance(pg, llm=llm).retrieve("t", []) == "do X"
    assert "[Start] → [search]" in llm.prompts[0]
    assert warnings == []


def test_retrieve_terminal_node_has_no_guidance(pg, warnings):
    _simple(pg)
    llm = _Llm()
    assert Guidance(pg, llm=llm).retrieve("t", ["read(id=1)"]) == ""
    assert warnings == [f"{PREFIX}: graph has no procedural guidance yet"]
    assert llm.prompts == []


def test_retrieve_llm_raises(pg, warnings):
    _simple(pg)
    g = Guidance(pg, llm=_Llm(exc=RuntimeError("down")))
    assert g.retrieve("t", ["search(q=1)"]) == ""
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{PREFIX}: guidance LLM error: ")


@pytest.mark.parametrize("reply", [[], {}, {"guidance": 5}, {"guidance": ""}])
def test_retrieve_wrong_shape(pg, warnings, reply):
    _simple(pg)
    g = Guidance(pg, llm=_Llm(reply=reply))
    assert g.retrieve("t", ["search(q=1)"]) == ""
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{PREFIX}: guidance LLM error: ")


def test_retrieve_corrupt_file(tmp_path, warnings):
    path = tmp_path / "g.json"
    path.write_text("{not json")
    llm = _Llm()
    assert Guidance(ProceduralGraph(path), llm=llm).retrieve("t", []) == ""
    assert len(warnings) == 1
    assert warnings[0].startswith(f"{PREFIX}: load failed: ")
    assert llm.prompts == []


def test_h_config_limits_neighborhood(pg):
    for src, tgt in [("u", "a"), ("a", "b")]:
        pg.add_triplet(src, "LEADS_TO", tgt, condition=None, guidance="g", pitfalls="p")
    llm = _Llm()
    Guidance(pg, h=1, llm=llm).retrieve("t", ["u()"])
    assert "[u] → [a]" in llm.prompts[0]
    assert "[a] → [b]" not in llm.prompts[0]
    llm2 = _Llm()
    Guidance(pg, h=2, llm=llm2).retrieve("t", ["u()"])
    assert "[a] → [b]" in llm2.prompts[0]

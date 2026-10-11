from __future__ import annotations

import pytest

from procedural_memory.graph import (
    END,
    START,
    GraphLoadError,
    GraphPersistenceError,
    ProceduralGraph,
    load_graph,
    new_skeleton,
    save_graph,
    structural_score,
    unreachable_nodes,
)


@pytest.fixture
def pg(tmp_path):
    return ProceduralGraph(tmp_path / "g.json")


def _triplet(pg, src, tgt, guidance="g", pitfalls="p", **kw):
    pg.add_triplet(
        src, "LEADS_TO", tgt, condition=None, guidance=guidance, pitfalls=pitfalls, **kw
    )


def test_skeleton_when_file_absent(pg, tmp_path):
    assert list(pg.G.edges(keys=True)) == [(START, END, "LEADS_TO")]
    assert not (tmp_path / "g.json").exists()
    assert pg.is_skeleton()


def test_round_trip(pg, tmp_path):
    _triplet(pg, "a", "b", source_session_id="s1", refiner_round=2)
    pg.save()
    fresh = ProceduralGraph(tmp_path / "g.json")
    data = fresh.G.edges["a", "b", "LEADS_TO"]
    assert data["condition"] is None
    assert data["guidance"] == "g"
    assert data["pitfalls"] == "p"
    assert data["provenance"]["source_session_id"] == "s1"
    assert fresh.G.nodes["a"]["node_type"] == "ACTION"
    assert "id" not in fresh.G.nodes["a"]
    assert not fresh.is_skeleton()


def test_add_triplet_is_upsert(pg):
    _triplet(pg, "a", "b", guidance="old")
    _triplet(pg, "a", "b", guidance="new")
    assert pg.G.number_of_edges("a", "b") == 1
    assert pg.G.edges["a", "b", "LEADS_TO"]["guidance"] == "new"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relation": "NOPE"},
        {"node_type": "BOGUS"},
        {"guidance": None},
    ],
)
def test_add_triplet_validation(pg, kwargs):
    args = {
        "relation": "LEADS_TO",
        "condition": None,
        "guidance": "g",
        "pitfalls": "p",
    }
    args.update(kwargs)
    relation = args.pop("relation")
    with pytest.raises(ValueError):
        pg.add_triplet("a", relation, "b", **args)


def test_save_failure_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(GraphPersistenceError):
        save_graph(new_skeleton(), blocker / "g.json")


def test_save_leaves_no_tmp_file(pg, tmp_path):
    pg.save()
    assert [p.name for p in tmp_path.iterdir()] == ["g.json"]


def test_load_invalid_json_raises(tmp_path):
    path = tmp_path / "g.json"
    path.write_text("{not json")
    with pytest.raises(GraphLoadError):
        load_graph(path)


def test_load_truncated_file_raises(pg, tmp_path):
    _triplet(pg, "a", "b")
    pg.save()
    path = tmp_path / "g.json"
    path.write_text(path.read_text()[:40])
    with pytest.raises(GraphLoadError):
        load_graph(path)


def test_localize_latest_action_wins(pg):
    _triplet(pg, "search", "read")
    assert pg.localize_active(["search(q=1)", "read(id=2)"]) == "read"


def test_localize_window_clip(pg):
    _triplet(pg, "search", "read")
    actions = ["search(q=1)", "x()", "y()", "z()"]
    assert pg.localize_active(actions, window_w=3) is None


def test_localize_miss_is_none(pg):
    _triplet(pg, "search", "read")
    assert pg.localize_active(["unknown(a=1)"]) is None


def test_localize_empty_window_is_start(pg):
    assert pg.localize_active([]) == START


def test_localize_tiebreak_newest_created_at(pg):
    _triplet(pg, "older", "next")
    _triplet(pg, "newer", "next")
    pg.G.nodes["older"]["description"] = "do_it()"
    pg.G.nodes["newer"]["description"] = "do_it()"
    pg.G.nodes["older"]["created_at"] = "2026-01-01T00:00:00.000Z"
    pg.G.nodes["newer"]["created_at"] = "2026-02-01T00:00:00.000Z"
    assert pg.localize_active(["do_it()"]) == "newer"


def test_neighborhood_hops(pg):
    for src, tgt in [("u", "a"), ("a", "b"), ("b", "c")]:
        _triplet(pg, src, tgt)
    h0 = pg.h_hop_neighborhood("u", 0)
    assert set(h0.nodes) == {"u"} and h0.number_of_edges() == 0
    h1 = pg.h_hop_neighborhood("u", 1)
    assert set(h1.nodes) == {"u", "a"}
    assert {(s, t) for s, t, _ in h1.edges(keys=True)} == {("u", "a")}
    h2 = pg.h_hop_neighborhood("u", 2)
    assert set(h2.nodes) == {"u", "a", "b"}
    assert {(s, t) for s, t, _ in h2.edges(keys=True)} == {("u", "a"), ("a", "b")}


def test_score_skeleton_is_one():
    assert structural_score(new_skeleton()) == pytest.approx(1.0)


def test_score_dangling_edge():
    G = new_skeleton()
    G.add_edge("x", END, key="LEADS_TO", relation="LEADS_TO")
    assert structural_score(G) == pytest.approx(0.88)


def _cyclic():
    G = new_skeleton()
    for node in ("a", "b"):
        G.add_node(node, node_type="ACTION", description=node)
    for src, tgt in [(START, "a"), ("a", "b"), ("b", "a"), ("b", END)]:
        G.add_edge(src, tgt, key="LEADS_TO", relation="LEADS_TO")
    return G


def test_score_cycle_forbidden():
    G = _cyclic()
    assert structural_score(G) == pytest.approx(0.8)
    assert structural_score(G, cycles_forbidden=False) == pytest.approx(1.0)


def test_score_cycle_without_exit_lowers_reach():
    G = new_skeleton()
    for node in ("a", "b"):
        G.add_node(node, node_type="ACTION", description=node)
    G.add_edge("a", "b", key="LEADS_TO", relation="LEADS_TO")
    G.add_edge("b", "a", key="LEADS_TO", relation="LEADS_TO")
    assert sorted(unreachable_nodes(G)) == ["a", "b"]
    # reach 2/4 (0.25) + acyclic 0 + edit quality 1.0 (0.2) + type validity 1.0 (0.1)
    assert structural_score(G) == pytest.approx(0.55)


def test_localize_matches_capitalized_node_ids(pg):
    _triplet(pg, "First_Hop_Retrieve", "Scan_Index")
    assert pg.localize_active(["first_hop_retrieve(q=1)"]) == "First_Hop_Retrieve"


def test_commit_persists_then_replaces(pg, tmp_path):
    candidate = new_skeleton()
    candidate.add_node("x", node_type="ACTION", description="x")
    candidate.add_edge("x", END, key="LEADS_TO", relation="LEADS_TO")
    pg.commit(candidate)
    assert "x" in pg.G
    assert "x" in ProceduralGraph(tmp_path / "g.json").G


def test_commit_failure_keeps_previous_graph(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    pg = ProceduralGraph(blocker / "g.json")
    before = pg.G
    with pytest.raises(GraphPersistenceError):
        pg.commit(new_skeleton())
    assert pg.G is before

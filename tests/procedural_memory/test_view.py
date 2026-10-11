from __future__ import annotations

import re

import pytest

from procedural_memory.graph import ProceduralGraph, new_skeleton
from procedural_memory.view import render_html, summarize


@pytest.fixture
def pg(tmp_path):
    return ProceduralGraph(tmp_path / "g.json")


def _chain(pg):
    pg.add_triplet(
        "Start", "LEADS_TO", "search", condition="always at start",
        guidance="Search broadly.", pitfalls="Avoid vague queries.",
    )
    pg.add_triplet(
        "search", "TRIGGERS", "End", condition=None,
        guidance="Finish with the answer.", pitfalls="Do not loop.",
    )


def test_summarize_skeleton():
    text = summarize(new_skeleton())
    assert "[Start] STATUS" in text
    assert "[End] STATUS" in text
    assert "Start -LEADS_TO-> End (always)" in text


def test_summarize_lists_nodes_edges_guidance(pg):
    _chain(pg)
    text = summarize(pg.G)
    assert "[search] ACTION: search" in text
    assert "Start -LEADS_TO-> search (always at start)" in text
    assert "search -TRIGGERS-> End (always)" in text
    assert "guidance: Search broadly." in text
    assert "pitfalls: Avoid vague queries." in text


def test_html_layers_start_left_end_right(pg):
    _chain(pg)
    html = render_html(pg.G)
    layers = dict(re.findall(r'data-id="([^"]+)" data-layer="(\d+)"', html))
    assert layers == {"Start": "0", "search": "1", "End": "2"}


def test_html_end_is_last_layer_even_with_direct_edge(pg):
    _chain(pg)
    pg.add_triplet(
        "Start", "LEADS_TO", "End", condition=None, guidance="g", pitfalls="p"
    )
    layers = dict(
        re.findall(r'data-id="([^"]+)" data-layer="(\d+)"', render_html(pg.G))
    )
    assert layers["End"] == "2"


def test_html_escapes_untrusted_text(pg):
    pg.add_triplet(
        "Start", "LEADS_TO", "<script>alert(1)</script>", condition=None,
        guidance="<img src=x onerror=alert(2)>", pitfalls='"quoted" & more',
    )
    html = render_html(pg.G)
    assert "<script>alert(1)" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&quot;quoted&quot; &amp; more" in html


def test_html_edge_details_in_tooltip_and_table(pg):
    _chain(pg)
    html = render_html(pg.G)
    assert "<title>" in html and "Search broadly." in html
    assert "Avoid vague queries." in html
    assert "<table" in html and "TRIGGERS" in html


def test_html_handles_cycles_self_loops_and_unreachable(pg):
    _chain(pg)
    pg.add_triplet("search", "LEADS_TO", "search", condition=None, guidance="g", pitfalls="p")
    pg.add_triplet("x", "LEADS_TO", "y", condition=None, guidance="g", pitfalls="p")
    pg.add_triplet("y", "LEADS_TO", "x", condition=None, guidance="g", pitfalls="p")
    layers = dict(
        re.findall(r'data-id="([^"]+)" data-layer="(\d+)"', render_html(pg.G))
    )
    assert set(layers) == {"Start", "search", "End", "x", "y"}
    assert layers["x"] == layers["y"] == "2"  # unreachable: after reachable nodes
    assert layers["End"] == "3"

from __future__ import annotations

from html import escape
from typing import Any

import networkx as nx

from procedural_memory.graph import END, START

_NODE_W, _NODE_H = 150, 40
_X_GAP, _Y_GAP, _PAD = 240, 84, 40
_LABEL_MAX = 20


def _condition(data: dict[str, Any]) -> str:
    return data.get("condition") or "always"


def summarize(G: nx.MultiDiGraph) -> str:
    """Plain-text dump of nodes, edges, and per-edge guidance/pitfalls."""
    lines = [
        f"Procedural graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
        "",
        "Nodes:",
    ]
    for node, data in G.nodes(data=True):
        lines.append(
            f"  [{node}] {data.get('node_type', '?')}: {data.get('description', '')}"
        )
    lines += ["", "Edges:"]
    for src, tgt, key, data in G.edges(keys=True, data=True):
        lines.append(f"  {src} -{key}-> {tgt} ({_condition(data)})")
        lines.append(f"      guidance: {data.get('guidance', '')}")
        lines.append(f"      pitfalls: {data.get('pitfalls', '')}")
    return "\n".join(lines)


def _layers(G: nx.MultiDiGraph) -> dict[str, int]:
    """Start at 0, BFS distance for reachable nodes, unreachable next, End last."""
    layers: dict[str, int] = {}
    if START in G:
        layers = dict(nx.single_source_shortest_path_length(G, START))
    layers.pop(END, None)
    deepest = max(layers.values(), default=0)
    for node in G:
        if node not in layers and node != END:
            layers[node] = deepest + 1
    if END in G:
        layers[END] = max(layers.values(), default=0) + 1
    return layers


def _label(text: str) -> str:
    return text if len(text) <= _LABEL_MAX else text[: _LABEL_MAX - 1] + "…"


def _edge_path(
    sx: float, sy: float, tx: float, ty: float, forward: bool, self_loop: bool
) -> tuple[str, float, float]:
    if self_loop:
        x, y = sx - _NODE_W / 2, sy - _NODE_H / 2
        d = f"M {x - 18:.0f} {y:.0f} C {x - 18:.0f} {y - 46:.0f}, {x + 18:.0f} {y - 46:.0f}, {x + 18:.0f} {y:.0f}"
        return d, x, y - 36
    if forward:
        mx = (sx + tx) / 2
        d = f"M {sx:.0f} {sy:.0f} C {mx:.0f} {sy:.0f}, {mx:.0f} {ty:.0f}, {tx:.0f} {ty:.0f}"
        return d, mx, (sy + ty) / 2 - 6
    lift = 56 + abs(ty - sy) * 0.2
    top = min(sy, ty) - lift
    d = f"M {sx:.0f} {sy:.0f} C {sx + 70:.0f} {top:.0f}, {tx - 70:.0f} {top:.0f}, {tx:.0f} {ty:.0f}"
    return d, (sx + tx) / 2, top + 10


_CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#a8a29e;--node:#fff;--stroke:#d6d3d1;
--action:#2563eb;--status:#16a34a;--reasoning:#9333ea;--skill:#d97706;--row:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#fafaf9;--muted:#a8a29e;--line:#78716c;
--node:#292524;--stroke:#44403c;--action:#60a5fa;--status:#4ade80;--reasoning:#c084fc;--skill:#fbbf24;--row:#292524}}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
h1{font-size:18px;margin:0 0 4px}.meta{color:var(--muted);margin:0 0 16px}
.canvas{overflow:auto;border:1px solid var(--stroke);border-radius:8px;background:var(--node)}
.node rect{fill:var(--node);stroke-width:2}.node text{fill:var(--fg);font-size:12px}
.type-ACTION rect{stroke:var(--action)}.type-STATUS rect{stroke:var(--status)}
.type-REASONING rect{stroke:var(--reasoning)}.type-SKILL rect{stroke:var(--skill)}
.edge path{fill:none;stroke:var(--line);stroke-width:1.5}.edge:hover path{stroke:var(--action);stroke-width:2.5}
.edge text{fill:var(--muted);font-size:10px}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{text-align:left;vertical-align:top;padding:6px 8px;border-bottom:1px solid var(--stroke)}
tr:nth-child(even){background:var(--row)}
"""


def render_html(G: nx.MultiDiGraph) -> str:
    """Self-contained HTML page: layered SVG diagram plus a transitions table."""
    layers = _layers(G)
    by_layer: dict[int, list[str]] = {}
    for node in G:
        by_layer.setdefault(layers[node], []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in by_layer.items():
        for row, node in enumerate(nodes):
            pos[node] = (_PAD + layer * _X_GAP, _PAD + row * _Y_GAP)

    width = _PAD * 2 + (max(by_layer, default=0)) * _X_GAP + _NODE_W
    height = _PAD * 2 + max((len(v) for v in by_layer.values()), default=1) * _Y_GAP + 40

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Procedural graph">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
        'fill="#a8a29e"/></marker></defs>',
    ]

    seen_pairs: dict[tuple[str, str], int] = {}
    for src, tgt, key, data in G.edges(keys=True, data=True):
        sx, sy = pos[src]
        tx, ty = pos[tgt]
        n = seen_pairs.get((src, tgt), 0)
        seen_pairs[(src, tgt)] = n + 1
        forward = layers[tgt] > layers[src]
        d, lx, ly = _edge_path(
            sx + _NODE_W if forward else sx + _NODE_W / 2,
            sy + _NODE_H / 2 + n * 6,
            tx if forward else tx + _NODE_W / 2,
            ty + _NODE_H / 2 + n * 6,
            forward,
            src == tgt,
        )
        tip = (
            f"{src} -{key}-> {tgt}\ncondition: {_condition(data)}\n"
            f"guidance: {data.get('guidance', '')}\npitfalls: {data.get('pitfalls', '')}"
        )
        parts.append(
            f'<g class="edge"><title>{escape(tip)}</title>'
            f'<path d="{d}" marker-end="url(#arrow)"/>'
            f'<text x="{lx:.0f}" y="{ly + n * 10:.0f}" text-anchor="middle">{escape(key)}</text></g>'
        )

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_type = str(data.get("node_type", "UNKNOWN"))
        parts.append(
            f'<g class="node type-{escape(node_type)}" data-id="{escape(node)}" '
            f'data-layer="{layers[node]}" transform="translate({x:.0f},{y:.0f})">'
            f'<title>{escape(node)} ({escape(node_type)}): '
            f'{escape(str(data.get("description", "")))}</title>'
            f'<rect width="{_NODE_W}" height="{_NODE_H}" rx="8"/>'
            f'<text x="{_NODE_W / 2}" y="{_NODE_H / 2 + 4}" text-anchor="middle">'
            f"{escape(_label(node))}</text></g>"
        )
    parts.append("</svg>")

    rows = []
    for src, tgt, key, data in G.edges(keys=True, data=True):
        rows.append(
            "<tr>"
            f"<td>{escape(src)} → {escape(tgt)}</td><td>{escape(key)}</td>"
            f"<td>{escape(_condition(data))}</td>"
            f"<td>{escape(str(data.get('guidance', '')))}</td>"
            f"<td>{escape(str(data.get('pitfalls', '')))}</td></tr>"
        )

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Whale procedural graph</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Procedural graph</h1>"
        f'<p class="meta">{G.number_of_nodes()} nodes · {G.number_of_edges()} edges · '
        "hover an edge for its guidance and pitfalls</p>"
        f'<div class="canvas">{"".join(parts)}</div>'
        "<h2>Transitions</h2>"
        "<table><thead><tr><th>Transition</th><th>Relation</th><th>Condition</th>"
        "<th>Guidance</th><th>Pitfalls</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table>'
        "</body></html>\n"
    )

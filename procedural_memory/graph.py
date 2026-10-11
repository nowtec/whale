from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
from networkx.readwrite import json_graph

RELATIONS: tuple[str, ...] = (
    "LEADS_TO",
    "TRIGGERS",
    "PROVIDES_INPUT_FOR",
    "CONVERGES_TO",
)
NODE_TYPES: tuple[str, ...] = ("ACTION", "REASONING", "STATUS", "SKILL")
START = "Start"
END = "End"
_RESERVED = (START, END)

# Literal grep token for every degraded-path WARN from the engine (spec §5.3).
WARN_PREFIX = "Could not validate a linked graph memory"


class GraphLoadError(Exception):
    """procedural_graph.json is unreadable or not a valid node-link graph."""


class GraphPersistenceError(Exception):
    """procedural_graph.json could not be written."""


@dataclass(frozen=True)
class Weights:
    reachability: float = 0.5
    acyclicity: float = 0.2
    edit_quality: float = 0.2
    type_validity: float = 0.1


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def normalize_node_id(raw: str) -> str:
    return re.sub(r"[\W_]+", "_", raw.lower()).strip("_")


def action_verb(action: str) -> str:
    return action.split("(", 1)[0].strip()


def new_skeleton() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for node in _RESERVED:
        G.add_node(
            node,
            node_type="STATUS",
            description=node,
            created_at=_now(),
            provenance={"source_session_id": None, "refiner_round": None},
        )
    G.add_edge(
        START,
        END,
        key="LEADS_TO",
        relation="LEADS_TO",
        condition=None,
        guidance="",
        pitfalls="",
        provenance={
            "source_session_id": None,
            "refiner_round": None,
            "created_at": _now(),
        },
    )
    return G


def load_graph(path: Path) -> nx.MultiDiGraph:
    path = Path(path)
    if not path.exists():
        return new_skeleton()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        G = json_graph.node_link_graph(data, edges="edges")
    except (OSError, ValueError, KeyError, TypeError, nx.NetworkXException) as exc:
        raise GraphLoadError(f"{path}: {exc}") from exc
    if not isinstance(G, nx.MultiDiGraph):
        raise GraphLoadError(f"{path}: not a directed multigraph")
    return G


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(G, edges="edges")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise GraphPersistenceError(f"{path}: {exc}") from exc


def unreachable_nodes(G: nx.MultiDiGraph) -> list[str]:
    terminals = [n for n in G if G.out_degree(n) == 0]
    reachable = set(terminals)
    for terminal in terminals:
        reachable |= nx.ancestors(G, terminal)
    return [n for n in G if n not in reachable]


def _has_valid_type(attrs: dict[str, Any]) -> bool:
    return attrs.get("node_type") in NODE_TYPES


def structural_score(
    G: nx.MultiDiGraph,
    *,
    cycles_forbidden: bool = True,
    weights: Weights = Weights(),
) -> float:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    reach = 1.0 - len(unreachable_nodes(G)) / max(1, n_nodes)
    acyclic = (
        0.0
        if cycles_forbidden and not nx.is_directed_acyclic_graph(G)
        else 1.0
    )

    dangling = sum(
        1
        for src, tgt, _ in G.edges(keys=True)
        if not (_has_valid_type(G.nodes[src]) and _has_valid_type(G.nodes[tgt]))
    )
    edit_q = 1.0 - dangling / max(1, n_edges)

    invalid_nodes = sum(1 for _, d in G.nodes(data=True) if not _has_valid_type(d))
    invalid_relations = sum(1 for _, _, k in G.edges(keys=True) if k not in RELATIONS)
    type_valid = 1.0 - (invalid_nodes + invalid_relations) / max(
        1, n_nodes + n_edges
    )

    return (
        weights.reachability * reach
        + weights.acyclicity * acyclic
        + weights.edit_quality * edit_q
        + weights.type_validity * type_valid
    )


class ProceduralGraph:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._G: nx.MultiDiGraph | None = None

    @property
    def G(self) -> nx.MultiDiGraph:
        if self._G is None:
            self.load()
        assert self._G is not None
        return self._G

    def load(self) -> None:
        self._G = load_graph(self.path)

    def save(self) -> None:
        save_graph(self.G, self.path)

    def commit(self, G: nx.MultiDiGraph) -> None:
        """Persist G, then make it the live graph. Raises on write failure."""
        save_graph(G, self.path)
        self._G = G

    def is_skeleton(self) -> bool:
        return set(self.G.nodes) <= set(_RESERVED) and set(
            self.G.edges(keys=True)
        ) <= {(START, END, "LEADS_TO")}

    def _ensure_node(
        self,
        name: str,
        node_type: str,
        source_session_id: str | None,
        refiner_round: int | None,
    ) -> None:
        if name in self.G and _has_valid_type(self.G.nodes[name]):
            return
        self.G.add_node(
            name,
            node_type="STATUS" if name in _RESERVED else node_type,
            description=name,
            created_at=_now(),
            provenance={
                "source_session_id": source_session_id,
                "refiner_round": refiner_round,
            },
        )

    def add_triplet(
        self,
        src: str,
        relation: str,
        tgt: str,
        *,
        condition: str | None,
        guidance: str,
        pitfalls: str,
        source_session_id: str | None = None,
        refiner_round: int | None = None,
        node_type: str = "ACTION",
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation: {relation!r}")
        if node_type not in NODE_TYPES:
            raise ValueError(f"unknown node_type: {node_type!r}")
        if not isinstance(guidance, str) or not isinstance(pitfalls, str):
            raise ValueError("guidance and pitfalls must be str")
        if condition is not None and not isinstance(condition, str):
            raise ValueError("condition must be str or None")

        self._ensure_node(src, node_type, source_session_id, refiner_round)
        self._ensure_node(tgt, node_type, source_session_id, refiner_round)

        created_at = _now()
        if self.G.has_edge(src, tgt, key=relation):
            created_at = self.G.edges[src, tgt, relation]["provenance"].get(
                "created_at", created_at
            )
        self.G.add_edge(
            src,
            tgt,
            key=relation,
            relation=relation,
            condition=condition,
            guidance=guidance,
            pitfalls=pitfalls,
            provenance={
                "source_session_id": source_session_id,
                "refiner_round": refiner_round,
                "created_at": created_at,
            },
        )

    def localize_active(
        self, recent_actions: Sequence[str], window_w: int = 3
    ) -> str | None:
        if not recent_actions:
            return START
        for action in reversed(list(recent_actions)[-window_w:]):
            verb_id = normalize_node_id(action_verb(action))
            matches = [
                n
                for n, d in self.G.nodes(data=True)
                if n not in _RESERVED
                and (
                    (verb_id and normalize_node_id(n) == verb_id)
                    or d.get("description") == action
                )
            ]
            if matches:
                return max(
                    matches, key=lambda n: self.G.nodes[n].get("created_at") or ""
                )
        return None

    def h_hop_neighborhood(self, u: str, h: int = 2) -> nx.MultiDiGraph:
        sub = nx.MultiDiGraph()
        if u not in self.G:
            return sub
        dist = nx.single_source_shortest_path_length(self.G, u, cutoff=h)
        for node in dist:
            sub.add_node(node, **self.G.nodes[node])
        for src, tgt, key, data in self.G.edges(keys=True, data=True):
            if src in dist and dist[src] < h:
                sub.add_edge(src, tgt, key=key, **data)
        return sub

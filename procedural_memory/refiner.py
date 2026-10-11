from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import networkx as nx
from networkx.readwrite import json_graph

from logger import logger
from procedural_memory.graph import (
    END,
    NODE_TYPES,
    RELATIONS,
    START,
    WARN_PREFIX,
    GraphLoadError,
    ProceduralGraph,
    Weights,
    structural_score,
    unreachable_nodes,
)

_RESERVED = (START, END)
_CHARS_PER_TOKEN = 4
_REJECTED_IN_PROMPT = 5

_MODES = {
    "static_onetime",
    "static_incremental",
    "scratch_onetime",
    "scratch_incremental",
}
_MODE_ALIASES = {
    "incremental": "scratch_incremental",
    "onetime": "scratch_onetime",
}

# Paper §B.5, "Refiner Prompt (Self-evolution)". Placeholders are {name}.
_REFINER_PROMPT = """\
You are an expert cognitive architect optimizing a Procedural Graph for an intelligent agent. The Procedural Graph encodes structured procedural guidance.
Task context: {task_description}
Refinement mode: {mode}
Available Tool Actions (the agent can only execute these actions): {available_tools_list}
Recent execution trajectories: {attempts_block}
Current Procedural Graph representation: {current_graph_json}
Previously rejected candidates: {rejected_block}
Your job is to refine the Procedural Graph. Follow these guidelines based on the mode:
• static_onetime / static_incremental: Prune edges/nodes that lead to loops, deadlocks, or failures. Add missing nodes and edges that could fix the failures and improve performance for future tasks.
• scratch_onetime / scratch_incremental: If starting from scratch (the graph contains only Start → End), synthesize a brand new, complete Procedural Graph using the Available Tool Actions list, Status, and successful patterns in the trajectories. Otherwise, prune edges/nodes that lead to loops, deadlocks, or failures, and add missing nodes and edges based on the given graph.
Rules for nodes and edges. Rules 2–4 describe the edge attributes in Φ(e): condition, guidance, and pitfalls. The remaining rules govern node compatibility, generality, and graph structure.
1. Action Nodes. Any node of type ACTION must match one of the action/tool names in the "Available Tool Actions" list above.
2. Transition Conditions. If an edge has a condition, provide a natural-language semantic precondition under which this transition should fire (e.g., "When dialogue history has been parsed but target constraints are unknown"). Use null if the transition is unconditional.
3. Execution Guidance. For every edge added in add_edges, you MUST provide a guidance string detailing exactly what action to take next and the strategic rationale behind it.
4. Pitfalls. Provide a pitfalls string warning about premature actions, forbidden words, or common formatting pitfalls to avoid during this step.
5. Generality & Leak Prevention. The updated Procedural Graph must guide the agent effectively without overfitting to specific details of a single trajectory. Use high-level conceptual descriptions.
6. Node ID Compatibility. If refining an existing graph (static modes), you MUST preserve the existing node IDs (such as Month_Start, Decide_Capital, and the tool names) so they remain compatible with the environment's state tracker. Do not rename them.
7. Graph Structure. Follow the task's configured cycle policy. Every edge must reference existing nodes, and every node must have a directed path to a terminal node. The environment loop handles repetition across simulation cycles.
Please propose the exact set of edits to perform. You must output your edits as a single valid JSON block containing four arrays: add_nodes, delete_nodes, add_edges, and delete_edges. Output format must be exactly:
{ "add_nodes": [{"id":..., "type": "ACTION", "description":...}],
  "delete_nodes": ["node_id"],
  "add_edges": [{"source":..., "target":..., "relation":...,
                 "condition":..., "guidance":..., "pitfalls":...}],
  "delete_edges": [{"source":..., "target":...}] }
Make sure to output ONLY the raw JSON block.
Each entry in delete_edges removes all edges with the specified source and target, regardless of relation. To retain selected transitions between the same endpoints, include them in add_edges, which is applied after deletion."""


class RefinerParseError(Exception):
    """The refiner's reply is not a well-formed EditDelta."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _warn(suffix: str) -> None:
    logger.warning(f"{WARN_PREFIX}: {suffix}")


@dataclass(frozen=True)
class NodeSpec:
    id: str
    type: str
    description: str


@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str
    relation: str
    condition: str | None
    guidance: str
    pitfalls: str


@dataclass(frozen=True)
class EditDelta:
    add_nodes: list[NodeSpec] = field(default_factory=list)
    delete_nodes: list[str] = field(default_factory=list)
    add_edges: list[EdgeSpec] = field(default_factory=list)
    delete_edges: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Any) -> EditDelta:
        if not isinstance(data, dict):
            raise RefinerParseError(
                f"expected a JSON object, got {type(data).__name__}"
            )

        def items(key: str) -> list[Any]:
            value = data.get(key)
            if value is None:
                return []
            if not isinstance(value, list):
                raise RefinerParseError(f"{key} must be a list")
            return value

        def text(item: Any, name: str, where: str) -> str:
            value = item.get(name) if isinstance(item, dict) else None
            required_id = name in ("id", "source", "target")
            if not isinstance(value, str) or (required_id and not value):
                raise RefinerParseError(f"{where}: missing or invalid {name!r}")
            return value

        add_nodes = []
        for item in items("add_nodes"):
            description = item.get("description", "") if isinstance(item, dict) else ""
            if not isinstance(description, str):
                raise RefinerParseError("add_nodes: description must be a string")
            add_nodes.append(
                NodeSpec(
                    id=text(item, "id", "add_nodes"),
                    type=text(item, "type", "add_nodes"),
                    description=description,
                )
            )

        delete_nodes = []
        for item in items("delete_nodes"):
            if not isinstance(item, str):
                raise RefinerParseError("delete_nodes: entries must be strings")
            delete_nodes.append(item)

        add_edges = []
        for item in items("add_edges"):
            condition = item.get("condition") if isinstance(item, dict) else None
            if condition is not None and not isinstance(condition, str):
                raise RefinerParseError("add_edges: condition must be string or null")
            add_edges.append(
                EdgeSpec(
                    source=text(item, "source", "add_edges"),
                    target=text(item, "target", "add_edges"),
                    relation=text(item, "relation", "add_edges"),
                    condition=condition,
                    guidance=text(item, "guidance", "add_edges"),
                    pitfalls=text(item, "pitfalls", "add_edges"),
                )
            )

        delete_edges = [
            (text(item, "source", "delete_edges"), text(item, "target", "delete_edges"))
            for item in items("delete_edges")
        ]
        return cls(add_nodes, delete_nodes, add_edges, delete_edges)

    def to_json(self) -> dict[str, Any]:
        return {
            "add_nodes": [vars(n) for n in self.add_nodes],
            "delete_nodes": list(self.delete_nodes),
            "add_edges": [vars(e) for e in self.add_edges],
            "delete_edges": [
                {"source": s, "target": t} for s, t in self.delete_edges
            ],
        }


@dataclass
class EvolutionResult:
    round_idx: int
    accepted: bool
    score_before: float
    score_after: float | None
    candidate_added: int
    candidate_removed: int
    rejection_reason: str | None


def tail_clip(text: str, max_tokens: int = 4000) -> str:
    """Keep the ending; drop the start. ~4 characters per token."""
    budget = max_tokens * _CHARS_PER_TOKEN
    return text if len(text) <= budget else text[-budget:]


def normalize_mode(mode: str) -> str:
    if mode in _MODES:
        return mode
    return _MODE_ALIASES.get(mode, "scratch_incremental")


def _snapshot(G: nx.MultiDiGraph | None) -> str | None:
    if G is None:
        return None
    return json.dumps(json_graph.node_link_data(G, edges="edges"), ensure_ascii=False)


def prepare_candidate(
    G: nx.MultiDiGraph,
    delta: EditDelta,
    *,
    cycles_forbidden: bool = True,
    strict: bool = True,
    auto_stub_missing_nodes: bool = False,
    source_session_id: str | None = None,
    refiner_round: int | None = None,
) -> tuple[nx.MultiDiGraph | None, list[str]]:
    """Apply delta to a copy of G (paper §B.6 PrepareCandidate).

    Returns (candidate, diagnostics). Non-empty diagnostics mean rejected;
    candidate is None only when the delta failed vocabulary checks.
    """
    diagnostics: list[str] = []
    for node in delta.delete_nodes:
        if node in _RESERVED:
            diagnostics.append(f"cannot delete reserved node {node!r}")
    for spec in delta.add_nodes:
        if spec.id in _RESERVED:
            diagnostics.append(f"cannot redefine reserved node {spec.id!r}")
        if spec.type not in NODE_TYPES:
            diagnostics.append(f"node {spec.id!r}: unknown type {spec.type!r}")
    for edge in delta.add_edges:
        if edge.relation not in RELATIONS:
            diagnostics.append(
                f"edge {edge.source}->{edge.target}: unknown relation {edge.relation!r}"
            )
    if diagnostics:
        return None, diagnostics

    C = G.copy()
    provenance = {"source_session_id": source_session_id, "refiner_round": refiner_round}

    for src, tgt in delta.delete_edges:
        if C.has_edge(src, tgt):
            for key in list(C[src][tgt]):
                C.remove_edge(src, tgt, key)
    for node in delta.delete_nodes:
        if node in C:
            C.remove_node(node)
    for spec in delta.add_nodes:
        if spec.id in C:
            C.nodes[spec.id].update(node_type=spec.type, description=spec.description)
        else:
            C.add_node(
                spec.id,
                node_type=spec.type,
                description=spec.description,
                created_at=_now(),
                provenance=dict(provenance),
            )
    for edge in delta.add_edges:
        missing = [n for n in (edge.source, edge.target) if n not in C]
        if missing and auto_stub_missing_nodes:
            for node in missing:
                C.add_node(
                    node,
                    node_type="STATUS",
                    description="auto-stubbed",
                    created_at=_now(),
                    provenance=dict(provenance),
                )
        elif missing and strict:
            diagnostics.append(
                f"edge {edge.source}->{edge.target} references missing node(s) "
                f"{', '.join(repr(m) for m in missing)}"
            )
            continue
        C.add_edge(
            edge.source,
            edge.target,
            key=edge.relation,
            relation=edge.relation,
            condition=edge.condition,
            guidance=edge.guidance,
            pitfalls=edge.pitfalls,
            provenance={**provenance, "created_at": _now()},
        )

    if diagnostics:
        return C, diagnostics

    if cycles_forbidden:
        while True:
            try:
                cycle = nx.find_cycle(C)
            except nx.NetworkXNoCycle:
                break
            src, tgt, key = cycle[-1]
            C.remove_edge(src, tgt, key)

    if strict:
        stranded = unreachable_nodes(C)
        if stranded:
            diagnostics.append(
                "no directed path to a terminal node from: " + ", ".join(sorted(stranded))
            )
    return C, diagnostics


class RejectionLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, entry: dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("rejections.append_failed", path=str(self.path), error=str(exc))

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        entries: list[dict[str, Any]] = []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
        return entries[-n:]


class Refiner:
    def __init__(
        self,
        graph: ProceduralGraph,
        rejections_path: Path,
        *,
        llm: Callable[[str], Any] | None = None,
        l_max: int = 4000,
        cycles_forbidden: bool = True,
        strict: bool = True,
        auto_stub_missing_nodes: bool = False,
        weights: Weights = Weights(),
    ) -> None:
        self.graph = graph
        self.rejections = RejectionLog(rejections_path)
        self._llm = llm
        self.l_max = l_max
        self.cycles_forbidden = cycles_forbidden
        self.strict = strict
        self.auto_stub_missing_nodes = auto_stub_missing_nodes
        self.weights = weights

    def _call_llm(self, prompt: str) -> Any:
        if self._llm is not None:
            return self._llm(prompt)
        from models import completion_default_json

        return completion_default_json(prompt)

    def _score(self, G: nx.MultiDiGraph) -> float:
        return structural_score(
            G, cycles_forbidden=self.cycles_forbidden, weights=self.weights
        )

    def propose(
        self,
        *,
        task_description: str,
        mode: str,
        available_tools: Sequence[str],
        trajectory: str,
        rejected_block: str,
    ) -> EditDelta:
        values = {
            "task_description": task_description,
            "mode": normalize_mode(mode),
            "available_tools_list": json.dumps(list(available_tools), ensure_ascii=False),
            "attempts_block": tail_clip(trajectory, self.l_max),
            "current_graph_json": _snapshot(self.graph.G),
            "rejected_block": rejected_block,
        }
        # single pass so placeholders inside substituted text stay literal
        prompt = re.sub(
            r"\{(\w+)\}",
            lambda m: values.get(m.group(1), m.group(0)),
            _REFINER_PROMPT,
        )
        try:
            raw = self._call_llm(prompt)
        except ValueError as exc:  # malformed JSON from the provider
            raise RefinerParseError(str(exc)) from exc
        return EditDelta.from_json(raw)

    def _reject(
        self,
        round_idx: int,
        delta: EditDelta,
        candidate: nx.MultiDiGraph | None,
        tail: str,
        reason: str,
        score: float | None = None,
    ) -> None:
        self.rejections.append(
            {
                "round": round_idx,
                "edit_delta": delta.to_json(),
                "candidate_graph_snapshot": _snapshot(candidate),
                "trajectories_tail": tail,
                "rejected_at": _now(),
                "reason": reason,
                "score": score,
            }
        )

    def evolve(
        self,
        trajectory: str,
        *,
        task_description: str = "",
        mode: str = "incremental",
        round_idx: int = 1,
        session_id: str | None = None,
        available_tools: Sequence[str] | None = None,
    ) -> EvolutionResult:
        def result(accepted, before, after=None, added=0, removed=0, reason=None):
            return EvolutionResult(
                round_idx, accepted, before, after, added, removed, reason
            )

        try:
            retained = self.graph.G
        except GraphLoadError as exc:
            _warn(f"load failed: {exc}")
            return result(False, 0.0, reason="load_failed")

        score_before = self._score(retained)
        tail = tail_clip(trajectory, self.l_max)
        tools = (
            list(available_tools)
            if available_tools is not None
            else [n for n, d in retained.nodes(data=True) if d.get("node_type") == "ACTION"]
        )
        rejected_block = json.dumps(
            [
                {k: e.get(k) for k in ("round", "reason", "edit_delta")}
                for e in self.rejections.recent(_REJECTED_IN_PROMPT)
            ],
            ensure_ascii=False,
        )

        try:
            delta = self.propose(
                task_description=task_description,
                mode=mode,
                available_tools=tools,
                trajectory=trajectory,
                rejected_block=rejected_block,
            )
        except RefinerParseError as exc:
            _warn(f"refiner response invalid: {exc}")
            return result(False, score_before, reason=f"parse_error:{exc}")
        except Exception as exc:
            _warn(f"refiner unreachable: {exc}")
            return result(False, score_before, reason=f"refiner_unreachable:{exc}")

        added = len(delta.add_nodes) + len(delta.add_edges)
        removed = len(delta.delete_nodes) + len(delta.delete_edges)

        candidate, diagnostics = prepare_candidate(
            retained,
            delta,
            cycles_forbidden=self.cycles_forbidden,
            strict=self.strict,
            auto_stub_missing_nodes=self.auto_stub_missing_nodes,
            source_session_id=session_id,
            refiner_round=round_idx,
        )
        if diagnostics:
            detail = "; ".join(diagnostics)
            _warn(f"structural reject: {detail}")
            reason = f"structural:{detail}"
            self._reject(round_idx, delta, candidate, tail, reason)
            logger.info("evolve.reject", round=round_idx, reason=reason)
            return result(False, score_before, None, added, removed, reason)

        score_after = self._score(candidate)
        if score_after + 1e-9 >= score_before:
            self.graph.commit(candidate)
            logger.info(
                "evolve.accept",
                round=round_idx,
                score_before=score_before,
                score_after=score_after,
            )
            return result(True, score_before, score_after, added, removed)

        reason = f"score:{score_after:.4f} < {score_before:.4f}"
        self._reject(round_idx, delta, candidate, tail, reason, score=score_after)
        logger.info(
            "evolve.reject",
            round=round_idx,
            reason=reason,
            score_before=score_before,
            score_after=score_after,
        )
        return result(False, score_before, score_after, added, removed, reason)

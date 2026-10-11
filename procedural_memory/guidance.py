from __future__ import annotations

from collections import deque
from typing import Any, Callable, Sequence

import networkx as nx

from logger import logger
from procedural_memory.graph import WARN_PREFIX, GraphLoadError, ProceduralGraph

# Paper §B.5, "Guidance Generation Prompt (Local subgraph; default)".
_PROMPT = (
    "You are an expert cognitive architect and execution guide for an AI agent "
    "solving the task: {task_description}\n"
    "Here is {graph_context_desc}: {subgraph_summary}\n"
    "Here is the current active query / observation: {query}\n"
    "Here is the agent's recent execution trajectory: {recent_context}\n"
    "Analyze this {graph_source} in the context of the agent's current progress. "
    "Using the condition, guidance, and pitfalls attributes carried by the edges "
    "in the graph context, generate clear, detailed, and actionable guidance "
    "advising the agent on exactly what step or strategy to pursue next, what "
    "pitfalls to avoid, and how to recover from recent failures if any. You must "
    "include any specific command patterns, file paths, tools, or arguments "
    "defined in the graph context if they are relevant to the next steps.\n"
    'Respond with a JSON object: {{"guidance": "<your guidance text>"}}'
)

_LOCAL_DESC = "the local procedural subgraph around the agent's active node"
_LOCAL_SOURCE = "local procedural subgraph"
_FULL_DESC = (
    "the complete Procedural Graph governing the task structure and "
    "strategic guidance"
)
_FULL_SOURCE = "complete Procedural Graph"


def _warn(suffix: str) -> None:
    logger.warning(f"{WARN_PREFIX}: {suffix}")


def _edge_lines(src: str, tgt: str, data: dict[str, Any]) -> list[str]:
    condition = data.get("condition") or "(always)"
    return [
        f"- Transition: [{src}] → [{tgt}] (Condition: {condition})",
        f"  * Guidance: {data.get('guidance', '')}",
        f"  * Pitfalls to Avoid: {data.get('pitfalls', '')}",
    ]


def serialize_local(active: str | None, G_t: nx.MultiDiGraph) -> str:
    """Paper §B.5 local serializer. Relation labels are never printed."""
    hop_of: dict[str, int] = {}
    if active is not None and active in G_t:
        hop_of[active] = 0
        queue = deque([active])
        while queue:
            node = queue.popleft()
            for succ in G_t.successors(node):
                if succ not in hop_of:
                    hop_of[succ] = hop_of[node] + 1
                    queue.append(succ)

    hop1: list[str] = []
    hop2: list[str] = []
    for src, tgt, _, data in G_t.edges(keys=True, data=True):
        if not hop_of:
            hop1.extend(_edge_lines(src, tgt, data))
        elif hop_of.get(src) == 0:
            hop1.extend(_edge_lines(src, tgt, data))
        elif hop_of.get(src) == 1:
            hop2.extend(_edge_lines(src, tgt, data))

    sections: list[str] = []
    if hop_of:
        attrs = G_t.nodes[active]
        sections.append(
            f"Active Cognitive Node: [{active}] "
            f"(Type: {attrs.get('node_type', 'ACTION')})\n"
            f"Description: {attrs.get('description', '')}"
        )
    if hop1:
        sections.append(
            "\n".join(["Immediate Transition Options (Hop 1):", *hop1])
        )
    if hop2:
        sections.append("\n".join(["Subsequent Horizon (Hop 2):", *hop2]))
    return "\n\n".join(sections)


def build_prompt(
    task: str,
    graph_text: str,
    query: str,
    recent_context: str,
    *,
    full_graph: bool,
) -> str:
    return _PROMPT.format(
        task_description=task,
        graph_context_desc=_FULL_DESC if full_graph else _LOCAL_DESC,
        subgraph_summary=graph_text,
        query=query,
        recent_context=recent_context,
        graph_source=_FULL_SOURCE if full_graph else _LOCAL_SOURCE,
    )


class Guidance:
    def __init__(
        self,
        graph: ProceduralGraph,
        *,
        h: int = 2,
        window_w: int = 3,
        model: str | None = None,
        llm: Callable[[str], Any] | None = None,
    ) -> None:
        self.graph = graph
        self.h = h
        self.window_w = window_w
        self.model = model
        self._llm = llm

    def _call_llm(self, prompt: str) -> Any:
        if self._llm is not None:
            return self._llm(prompt)
        from models import completion_default_json

        return completion_default_json(prompt, model=self.model)

    def retrieve(
        self,
        task: str,
        recent_actions: Sequence[str],
        *,
        recent_context: str = "",
    ) -> str:
        try:
            G = self.graph.G
        except GraphLoadError as exc:
            _warn(f"load failed: {exc}")
            return ""

        if self.graph.is_skeleton():
            _warn("graph has no procedural guidance yet")
            return ""

        active = self.graph.localize_active(recent_actions, self.window_w)
        full_graph = active is None
        if full_graph:
            _warn("localized node not found, falling back to full graph")
            G_t = G
        else:
            G_t = self.graph.h_hop_neighborhood(active, self.h)
            logger.info("guidance.localize_hit", node=active, hops=self.h)

        if G_t.number_of_edges() == 0:
            _warn("graph has no procedural guidance yet")
            return ""

        graph_text = serialize_local(None if full_graph else active, G_t)
        logger.debug(
            "retrieval.serialize_subgraph",
            nodes=G_t.number_of_nodes(),
            edges=G_t.number_of_edges(),
            hops=self.h,
        )
        prompt = build_prompt(
            task, graph_text, task, recent_context, full_graph=full_graph
        )

        try:
            reply = self._call_llm(prompt)
        except Exception as exc:  # degrade: the solver proceeds unguided
            _warn(f"guidance LLM error: {exc}")
            return ""

        text = reply.get("guidance") if isinstance(reply, dict) else None
        if not isinstance(text, str) or not text.strip():
            _warn(
                "guidance LLM error: unexpected response shape "
                f"({type(reply).__name__})"
            )
            return ""
        return text.strip()

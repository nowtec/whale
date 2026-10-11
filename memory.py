from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import re
from pathlib import Path
from typing import Any, Callable

from logger import logger
from procedural_memory.graph import (
    END,
    START,
    WARN_PREFIX,
    GraphLoadError,
    ProceduralGraph,
    action_verb,
    normalize_node_id,
)
from procedural_memory.guidance import Guidance
from procedural_memory.refiner import EvolutionResult, Refiner
from procedural_memory.view import render_html, summarize

DEFAULT_ROOT = Path.home() / ".whale"
MEMORY_DIR_NAME = "memory"
ENTRIES_FILE = "entries.json"
GRAPH_FILE = "procedural_graph.json"
REJECTIONS_FILE = "rejections.jsonl"

_FINISH_VERBS = {"FINISH", "DONE"}
_ACTION_LINE = re.compile(r"^Action:\s*(.+)$", re.MULTILINE)


@dataclass
class MemoryEntry:
    id: str
    kind: str
    content: dict[str, Any]
    created_at: str
    tags: list[str] = field(default_factory=list)
    source: str | None = None


class MemoryEngine:
    def __init__(
        self,
        root: Path | None = None,
        *,
        llm: Callable[[str], Any] | None = None,
        session_loader: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.root = (root or DEFAULT_ROOT).expanduser()
        self.memory_dir = self.root / MEMORY_DIR_NAME
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.memory_dir / ENTRIES_FILE
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

        self.graph_path = self.memory_dir / GRAPH_FILE
        self.rejections_path = self.memory_dir / REJECTIONS_FILE
        self._session_loader = session_loader
        self._graph = ProceduralGraph(self.graph_path)
        self._guidance = Guidance(self._graph, llm=llm)
        self._refiner = Refiner(self._graph, self.rejections_path, llm=llm)
        self._last_node: dict[str, str] = {}

    # --- procedural graph memory -------------------------------------------

    def retrieve_guidance(self, task: str, recent_actions: list[str]) -> str:
        return self._guidance.retrieve(
            task, recent_actions, recent_context="\n".join(recent_actions)
        )

    def record_triplet(
        self,
        src: str,
        relation: str,
        tgt: str,
        *,
        condition: str | None = None,
        guidance: str,
        pitfalls: str,
        source_session_id: str | None = None,
        refiner_round: int | None = None,
    ) -> None:
        try:
            self._graph.add_triplet(
                src,
                relation,
                tgt,
                condition=condition,
                guidance=guidance,
                pitfalls=pitfalls,
                source_session_id=source_session_id,
                refiner_round=refiner_round,
            )
        except GraphLoadError as exc:
            logger.warning(f"{WARN_PREFIX}: load failed: {exc}")
            return
        self._graph.save()
        logger.info(
            "memory.record_triplet", src=src, relation=relation, tgt=tgt
        )

    def record_turn(
        self, session_id: str, thought: str, action: str, observation: str
    ) -> list[tuple[str, str, str]]:
        verb = action_verb(action)
        node = normalize_node_id(verb)
        if not node:
            return []
        tgt = END if verb.upper() in _FINISH_VERBS else node
        src = self._last_node.get(session_id, START)
        if src == tgt:
            return []

        logger.debug("memory.record_turn", session_id=session_id, observation=observation)
        try:
            is_new_node = tgt not in self._graph.G
            self._graph.add_triplet(
                src,
                "LEADS_TO",
                tgt,
                condition=None,
                guidance=thought or f"Proceed with {action}",
                pitfalls="",
                source_session_id=session_id,
            )
        except GraphLoadError as exc:
            logger.warning(f"{WARN_PREFIX}: load failed: {exc}")
            return []
        if is_new_node and tgt != END:
            self._graph.G.nodes[tgt]["description"] = action
        self._graph.save()
        self._last_node[session_id] = tgt
        logger.info("memory.record_triplet", src=src, relation="LEADS_TO", tgt=tgt)
        return [(src, "LEADS_TO", tgt)]

    def evolve_from_trajectory(
        self,
        session_id: str,
        *,
        mode: str = "incremental",
        round_idx: int | None = None,
    ) -> EvolutionResult:
        loader = self._session_loader
        if loader is None:
            from session import loadSession as loader  # lazy: avoids import cycle
        messages = loader(session_id)

        task = next(
            (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
            "",
        )
        trajectory = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages
        )

        tools: list[str] = []
        try:
            tools = [
                n
                for n, d in self._graph.G.nodes(data=True)
                if d.get("node_type") == "ACTION"
            ]
        except GraphLoadError:
            pass  # Refiner.evolve reports the load failure itself
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for match in _ACTION_LINE.finditer(str(message.get("content", ""))):
                verb = action_verb(match.group(1))
                name = normalize_node_id(verb)
                if name and verb.upper() not in _FINISH_VERBS and name not in tools:
                    tools.append(name)

        return self._refiner.evolve(
            trajectory,
            task_description=task,
            mode=mode,
            round_idx=round_idx or 1,
            session_id=session_id,
            available_tools=tools,
        )

    def describe_graph(self) -> str:
        """Plain-text view of the procedural graph (for `memory show`)."""
        try:
            return summarize(self._graph.G)
        except GraphLoadError as exc:
            message = f"{WARN_PREFIX}: load failed: {exc}"
            logger.warning(message)
            return message

    def render_graph_html(self) -> str:
        """Self-contained HTML/SVG diagram of the graph (for `memory plot`)."""
        try:
            return render_html(self._graph.G)
        except GraphLoadError as exc:
            message = f"{WARN_PREFIX}: load failed: {exc}"
            logger.warning(message)
            return f"<!doctype html><html><body><p>{html.escape(message)}</p></body></html>"

    # --- legacy flat entries (unused by the procedural path) ----------------

    def _now(self) -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

    def _read(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, entries: list[dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def record(
        self,
        kind: str,
        content: dict[str, Any],
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            kind=kind,
            content=content,
            created_at=self._now(),
            tags=list(tags or []),
            source=source,
        )
        entries = self._read()
        entries.append(entry.__dict__)
        self._write(entries)
        logger.info(
            "memory.record",
            kind=kind,
            entry_id=entry.id,
            source=source,
        )
        return entry

    def retrieve(
        self, kind: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        entries = self._read()
        if kind is not None:
            entries = [e for e in entries if e.get("kind") == kind]
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return entries[:limit]


memoryEngine = MemoryEngine()


def retrieve_guidance(task: str, recent_actions: list[str]) -> str:
    return memoryEngine.retrieve_guidance(task, recent_actions)


def record_triplet(
    src: str,
    relation: str,
    tgt: str,
    *,
    condition: str | None = None,
    guidance: str,
    pitfalls: str,
    source_session_id: str | None = None,
    refiner_round: int | None = None,
) -> None:
    memoryEngine.record_triplet(
        src,
        relation,
        tgt,
        condition=condition,
        guidance=guidance,
        pitfalls=pitfalls,
        source_session_id=source_session_id,
        refiner_round=refiner_round,
    )


def record_turn(
    session_id: str, thought: str, action: str, observation: str
) -> list[tuple[str, str, str]]:
    return memoryEngine.record_turn(session_id, thought, action, observation)


def describe_graph() -> str:
    return memoryEngine.describe_graph()


def render_graph_html() -> str:
    return memoryEngine.render_graph_html()


def evolve_from_trajectory(
    session_id: str, *, mode: str = "incremental", round_idx: int | None = None
) -> EvolutionResult:
    return memoryEngine.evolve_from_trajectory(
        session_id, mode=mode, round_idx=round_idx
    )


def _hook_record(record: dict[str, Any] | str) -> MemoryEntry | None:
    if isinstance(record, str):
        payload: dict[str, Any] = {"text": record}
    else:
        payload = record
    kind = payload.pop("kind", "note")
    tags = payload.pop("tags", None)
    source = payload.pop("source", None)
    return memoryEngine.record(kind=kind, content=payload, tags=tags, source=source)


def _hook_retrieve(kind: str | None = None, limit: int = 20) -> list[str]:
    return [
        json.dumps(entry, ensure_ascii=False)
        for entry in memoryEngine.retrieve(kind=kind, limit=limit)
    ]

"""End-to-end smoke check of the procedural graph memory (stub LLM, temp dir).

Run: uv run python scripts/smoke_pg.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory import MemoryEngine  # noqa: E402

DELTA = {
    "add_nodes": [{"id": "search", "type": "ACTION", "description": "search"}],
    "delete_nodes": [],
    "add_edges": [
        {
            "source": "Start",
            "target": "search",
            "relation": "LEADS_TO",
            "condition": None,
            "guidance": "Begin by searching.",
            "pitfalls": "Avoid vague queries.",
        },
        {
            "source": "search",
            "target": "End",
            "relation": "LEADS_TO",
            "condition": None,
            "guidance": "Finish once you have the answer.",
            "pitfalls": "Do not loop.",
        },
    ],
    "delete_edges": [{"source": "Start", "target": "End"}],
}


def stub_llm(prompt: str):
    if "cognitive architect optimizing" in prompt:
        return DELTA
    return {"guidance": "stub guidance"}


def check(step: int, label: str, ok: bool) -> None:
    print(f"{step}. {label}: {'ok' if ok else 'FAILED'}")
    if not ok:
        sys.exit(1)


def edge_set(engine: MemoryEngine) -> set[tuple[str, str, str]]:
    data = json.loads(engine.graph_path.read_text())
    return {(e["source"], e["target"], e["relation"]) for e in data["edges"]}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        messages = [
            {"role": "user", "content": "smoke task"},
            {"role": "assistant", "content": "Thought: go\nAction: search(q=1)"},
        ]
        engine = MemoryEngine(root, llm=stub_llm, session_loader=lambda sid: messages)

        check(
            1,
            "fresh graph is the Start -LEADS_TO-> End skeleton",
            engine._graph.is_skeleton() and not engine.graph_path.exists(),
        )

        engine.record_triplet(
            "search", "LEADS_TO", "read", guidance="Read it.", pitfalls="Skip ads."
        )
        reloaded = MemoryEngine(root, llm=stub_llm)
        check(
            2,
            "record_triplet persists and survives a reload",
            ("search", "read", "LEADS_TO") in edge_set(reloaded),
        )

        guidance = reloaded.retrieve_guidance("smoke task", ["search(q=1)"])
        check(3, "retrieve_guidance returns guidance for a recorded node", bool(guidance))

        result = engine.evolve_from_trajectory("smoke-session")
        check(
            4,
            "evolve_from_trajectory accepts a structurally valid edit",
            result.accepted and ("Start", "search", "LEADS_TO") in edge_set(engine),
        )
    print("OK")


if __name__ == "__main__":
    main()

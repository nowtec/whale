# Procedural Graph Memory Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `whale`'s flat `MemoryEngine` with a NetworkX-backed Procedural Graph memory (graph store, online guidance, offline self-evolution) behind the existing `memory.py` module.

**Architecture:** A new `procedural_memory/` package holds three focused modules (`graph.py`, `guidance.py`, `refiner.py`); `memory.py` is a thin facade over them; `agent_loop.py` is a minimal K-turn ReAct primitive that only touches the facade. All LLM calls go through `models.completion_default_json` and are injectable (`llm=` callable) so every test uses a stub.

**Tech Stack:** Python 3.11, networkx, litellm, typer, pytest (run with `uv run pytest`).

**Spec:** `docs/superpowers/specs/2026-10-11-procedural-graph-memory-design.md` (paper PDF beside it in `docs/`; prompt texts are paper §B.5, Algorithm 1 is §B.6).

## Global Constraints

- Relations (exact strings): `LEADS_TO`, `TRIGGERS`, `PROVIDES_INPUT_FOR`, `CONVERGES_TO`. Node types: `ACTION`, `REASONING`, `STATUS`, `SKILL`. Reserved node ids: `Start`, `End` (type `STATUS`).
- Skeleton when `procedural_graph.json` is absent: `Start —LEADS_TO→ End`.
- Defaults: `h=2`, `window_w=3`, `L_max=4000` tokens, `whale refine --round` default `10`, `Refiner.evolve` runs one round per call.
- Score weights: reachability `0.5`, acyclicity `0.2`, edit quality `0.2`, type validity `0.1`. Accept a candidate iff `S_cand >= S_before`.
- Every non-fatal engine degradation logs at WARN via `logger.warning` with the literal prefix `Could not validate a linked graph memory` followed by `: <suffix>` (suffixes in spec §5.3). Disk-write failure raises `GraphPersistenceError`.
- On-disk: `~/.whale/memory/procedural_graph.json` (node-link JSON, atomic tmp + `os.replace`), `~/.whale/memory/rejections.jsonl` (append-only). `entries.json` and the `_hook_*` shims stay but are unused by the new path. No migrations.
- `agent_loop.py` and `main.py` import only `memory`/`session`/`models`, never `procedural_memory`.
- No commits unless the user asks.

## Decisions Where the Spec Is Silent or Conflicts

| # | Decision | Why |
|---|----------|-----|
| D1 | Evolution does **not** fire automatically. `session.close_session(sid, *, evolve=False)` and `agent_loop.run(..., evolve_on_close=False)`; `whale refine` is the trigger. | Goal 3 (latest edit) says "Only by calling `refine` command"; §6.4/§7.2 describe auto-firing. The opt-in flag keeps the §7.2 wiring available. |
| D2 | `networkx>=3.4`, `node_link_data(..., edges="edges")`. | `edges=` kwarg avoids a FutureWarning; spec floor 3.0 lacks it. |
| D3 | Node id is the NetworkX node key, never an attribute named `id`. Edge key = relation, so `(u, v, relation)` is unique and `record_triplet` is an upsert. | `node_link_data` overwrites a node attr called `id`. |
| D4 | Serializer label is `* Pitfalls to Avoid:` (paper excerpt), not `* Pitfalls:`. | Spec says "verbatim from paper". |
| D5 | `mode="incremental"` is an alias for the paper's `scratch_incremental` (`onetime` → `scratch_onetime`); the paper's four mode names are also accepted. | The paper's refiner prompt only defines the four names. |
| D6 | `record_turn` returns `list[tuple[src, relation, tgt]]` (not `list[None]`). | Lets tests assert what was written. |
| D7 | Empty `recent_actions` localizes to `Start` (a hit, no WARN). | First turn has no actions yet. |
| D8 | `cycles_forbidden` defaults to `True`; "dangling edge" = edge with an endpoint lacking a valid `node_type` (implicitly created by `add_edge`). | Spec leaves both undefined. |
| D9 | `LOG_LEVEL=WARN` accepted as an alias for `WARNING`. | Spec §5.3 uses `WARN`; logger currently rejects it. |
| D10 | `whale refine` with neither `--session-id` nor `--last` uses the most recent session. | Avoids a usage error for the common case. |
| D11 | `Refiner(strict=True)`; with `strict=False` invalid candidates are scored instead of rejected structurally. | Without it the "score decreased" reject path in Algorithm 1 is unreachable in tests. |

## Review Focus

Inputs the spec implies but no spec'd test exercises; each has a test in the named task.

1. Corrupt/truncated `procedural_graph.json` → `retrieve_guidance` returns `""`, writes do not clobber the file (Tasks 4, 6).
2. LLM returns valid JSON of the wrong shape (list, missing `guidance`, non-string) → no crash, `""` guidance / round reported as parse failure (Tasks 4, 5).
3. Actions that normalize to nothing (`""`, `"()"`), repeated identical actions (self-loop), and `FINISH`/`DONE` → no junk nodes, no self-loops, edge to `End` (Task 6).
4. Refiner proposes deleting `Start`/`End`, an edge to a missing node, or an exit-less cycle → rejected, graph file byte-identical (Task 5).
5. Trajectory longer than `L_max` keeps its ending; a corrupt line in `rejections.jsonl` is skipped, not fatal (Task 5).

## File Structure

| File | Responsibility |
|------|----------------|
| `procedural_memory/__init__.py` | Package marker. |
| `procedural_memory/graph.py` | Schema constants, load/save, `ProceduralGraph` (triplet upsert, `localize_active`, `h_hop_neighborhood`), `structural_score`. |
| `procedural_memory/guidance.py` | `serialize_local`, guidance prompt, `Guidance.retrieve`. |
| `procedural_memory/refiner.py` | `EditDelta`, `prepare_candidate`, `RejectionLog`, `tail_clip`, `Refiner.propose/evolve`. |
| `memory.py` | `MemoryEngine` facade gains the four spec'd methods; module-level wrappers. |
| `session.py` | `close_session`. |
| `agent_loop.py` | `TurnOutput`, `RunResult`, `run`, `run_with_guidance`. |
| `models.py` | `completion_default_json`, `CompletionJSONError`. |
| `main.py` | `prompt` through the loop (`--no-loop`), `refine` command. |
| `scripts/smoke_pg.py` | End-to-end stub demo. |
| `tests/procedural_memory/test_{graph,guidance,refiner}.py`, `tests/test_{memory,agent_loop,session,models,main}.py` | Tests. |

Tests never touch `~/.whale`: engines take `root=tmp_path`; LLMs are stubs. WARN assertions use `monkeypatch.setattr(<module>.logger, "warning", recorder)` (the `whale` logger has `propagate=False`, so `caplog` sees nothing).

---

### Task 1: Dependency and LOG_LEVEL alias

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`), `logger.py:14-20`
- Test: `tests/test_logger.py` (extend)

**Interfaces:**
- Produces: `import networkx` available; `LOG_LEVEL=WARN` resolves to `logging.WARNING`.

- [ ] **Step 1: Failing test** in `tests/test_logger.py`: `test_log_level_warn_alias` — follow how existing tests reload the module; assert `_resolve_level()` returns `logging.WARNING` with `LOG_LEVEL=WARN`.
- [ ] **Step 2:** `uv run pytest tests/test_logger.py -k warn_alias -v` → FAIL (falls back to INFO).
- [ ] **Step 3:** Add `"WARN": logging.WARNING` to `_VALID_LEVELS`.
- [ ] **Step 4:** `uv add "networkx>=3.4"`; `uv run python -c "import networkx"` exits 0.
- [ ] **Step 5:** `uv run pytest tests/test_logger.py -v` → all PASS.

### Task 2: `completion_default_json`

**Files:**
- Modify: `models.py`
- Test: `tests/test_models.py` (create)

**Interfaces:**
- Produces: `class CompletionJSONError(Exception)`; `completion_default_json(prompt: str, response_format: dict | None = None, *, model: str | None = None) -> Any`. Default `response_format={"type": "json_object"}`. `model=None` ⇒ `openai/$OPENAI_MODEL`. `completion_default` is unchanged.

- [ ] **Step 1: Failing tests** (monkeypatch `models.completion` with a fake returning an object whose `.choices[0].message.content` is set; set `OPENAI_MODEL=m`):
  - `test_json_parsed`: content `'{"a": 1}'` → `{"a": 1}`; fake received `response_format={"type": "json_object"}` and `model="openai/m"`.
  - `test_model_override`: `model="x/y"` → fake received `model="x/y"`.
  - `test_invalid_json_raises`: content `"nope"` → `CompletionJSONError`.
  - `test_none_content_raises`: content `None` → `CompletionJSONError`.
  - `test_provider_error_propagates`: fake raises `RuntimeError` → same `RuntimeError`.
- [ ] **Step 2:** Run → FAIL (`ImportError`).
- [ ] **Step 3:** Implement with the same `completion request/response/error` logging as `completion_default`; share the request/log body through a private helper rather than copying it.
- [ ] **Step 4:** `uv run pytest tests/test_models.py -v` → PASS.

### Task 3: `procedural_memory/graph.py`

**Files:**
- Create: `procedural_memory/__init__.py`, `procedural_memory/graph.py`
- Test: `tests/procedural_memory/test_graph.py`

**Interfaces:**
- Produces (all in `graph.py`):
  - `RELATIONS: tuple[str, ...]`, `NODE_TYPES: tuple[str, ...]`, `START = "Start"`, `END = "End"`
  - `class GraphLoadError(Exception)`, `class GraphPersistenceError(Exception)`
  - `@dataclass(frozen=True) class Weights: reachability: float = 0.5; acyclicity: float = 0.2; edit_quality: float = 0.2; type_validity: float = 0.1`
  - `new_skeleton() -> nx.MultiDiGraph`
  - `load_graph(path: Path) -> nx.MultiDiGraph` (missing file → skeleton, nothing written; unreadable/invalid → `GraphLoadError`)
  - `save_graph(G: nx.MultiDiGraph, path: Path) -> None` (tmp + `os.replace`; any `OSError` → `GraphPersistenceError`; no tmp file left behind)
  - `normalize_node_id(raw: str) -> str` — lower-case, runs of non `[a-z0-9]` → `_`, stripped of `_`; may return `""`
  - `action_verb(action: str) -> str` — text before the first `(`, stripped
  - `structural_score(G, *, cycles_forbidden: bool = True, weights: Weights = Weights()) -> float`
  - `unreachable_nodes(G) -> list[str]` — nodes with no directed path to a zero-out-degree node (a terminal counts itself)
  - `class ProceduralGraph(path: Path)` with `.G` (lazy-loaded property), `.load()`, `.save()`, `.is_skeleton() -> bool`, `.add_triplet(src, relation, tgt, *, condition: str | None, guidance: str, pitfalls: str, source_session_id: str | None = None, refiner_round: int | None = None, node_type: str = "ACTION") -> None`, `.localize_active(recent_actions: Sequence[str], window_w: int = 3) -> str | None`, `.h_hop_neighborhood(u: str, h: int = 2) -> nx.MultiDiGraph`
- Storage shape: node attrs `node_type`, `description`, `created_at`, `provenance={source_session_id, refiner_round}`; edge key = relation; edge attrs `relation`, `condition`, `guidance`, `pitfalls`, `provenance={source_session_id, refiner_round, created_at}`. Serialize with `node_link_data(G, edges="edges")`.

Semantics to implement exactly:
- `structural_score` = `0.5*reach + 0.2*acyclic + 0.2*edit_q + 0.1*type_valid`; `reach` = fraction of nodes not in `unreachable_nodes`; `acyclic` = `0.0` iff `cycles_forbidden and not is_directed_acyclic_graph`; `edit_q = 1 - dangling_edges / max(1, edges)`; `type_valid = 1 - (nodes lacking a valid node_type + edges with a relation outside RELATIONS) / max(1, nodes + edges)`.
- `add_triplet` validates `relation ∈ RELATIONS`, `node_type ∈ NODE_TYPES`, `guidance`/`pitfalls` are `str` (empty allowed), else `ValueError`. New nodes get `description=<node id>`; existing nodes are never overwritten. Same `(src, relation, tgt)` updates the edge in place.
- `localize_active`: examine `recent_actions[-window_w:]` newest first; an action matches node `n` (not `Start`/`End`) if `normalize_node_id(action_verb(action)) == n` or `description == action`; several matches for one action → newest `created_at`. Empty `recent_actions` → `START`. No match → `None`.
- `h_hop_neighborhood(u, h)`: nodes within `h` forward hops of `u`, plus edges whose source is at distance `< h` (so `h=0` is just `u`).

- [ ] **Step 1: Failing tests** in `test_graph.py`:
  - `test_skeleton_when_file_absent`: new `ProceduralGraph(tmp/"g.json")` → `.G` has exactly edge `("Start","End","LEADS_TO")`; file not created; `.is_skeleton()` is `True`.
  - `test_round_trip`: add `a →LEADS_TO→ b` with `condition=None, guidance="g", pitfalls="p"`, save, load via a fresh instance → same edge attrs, `node_type == "ACTION"`, `"id" not in G.nodes["a"]`.
  - `test_add_triplet_is_upsert`: same triplet twice with different guidance → one edge, newer guidance.
  - `test_add_triplet_validation`: unknown relation, unknown node_type, `guidance=None` → `ValueError`.
  - `test_save_failure_raises`: path whose parent is a regular file → `GraphPersistenceError`.
  - `test_load_invalid_json_raises` and `test_load_truncated_file_raises` → `GraphLoadError`.
  - `test_localize_latest_action_wins`, `test_localize_window_clip` (match older than the last 3 actions → `None`), `test_localize_miss_is_none`, `test_localize_empty_window_is_start`, `test_localize_tiebreak_newest_created_at`.
  - `test_neighborhood_hops`: chain `u→a→b→c`: `h=0` nodes `{u}` no edges; `h=1` nodes `{u,a}` edges `{(u,a)}`; `h=2` nodes `{u,a,b}` edges `{(u,a),(a,b)}`.
  - `test_score_skeleton_is_one`; `test_score_dangling_edge`: skeleton plus `G.add_edge("x","End",key="LEADS_TO",relation="LEADS_TO")` (no node attrs on `x`) scores `0.88` (±1e-9); `test_score_cycle_forbidden`: skeleton plus `Start→a, a→b, b→a, b→End` scores `0.8`, and `1.0` with `cycles_forbidden=False`; `test_score_cycle_without_exit_lowers_reach`: `a↔b` with no path to a terminal → `unreachable_nodes == ["a","b"]` (order-insensitive) and score `< 1`.
- [ ] **Step 2:** `uv run pytest tests/procedural_memory/test_graph.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3:** Implement `graph.py` to the interface above.
- [ ] **Step 4:** Run → all PASS.

### Task 4: `procedural_memory/guidance.py`

**Files:**
- Create: `procedural_memory/guidance.py`
- Test: `tests/procedural_memory/test_guidance.py`

**Interfaces:**
- Consumes: Task 3's `ProceduralGraph`, `GraphLoadError`; Task 2's `completion_default_json`.
- Produces:
  - `serialize_local(active: str | None, G_t: nx.MultiDiGraph) -> str`
  - `build_prompt(task: str, graph_text: str, query: str, recent_context: str, *, full_graph: bool) -> str`
  - `class Guidance(graph: ProceduralGraph, *, h: int = 2, window_w: int = 3, model: str | None = None, llm: Callable[[str], Any] | None = None)` with `.retrieve(task: str, recent_actions: Sequence[str], *, recent_context: str = "") -> str`. Default `llm` is `lambda p: completion_default_json(p, model=model)`.

Exact serializer output (blank line between sections; hop = BFS distance of the edge's **source** from `active`, 0 → Hop 1, 1 → Hop 2; relation labels never printed; `condition is None` → `(always)`; when `active is None` print no header and list every edge under Hop 1):

```
Active Cognitive Node: [First_Hop_Retrieve] (Type: ACTION)
Description: Execute first_hop_retrieve to fetch primary evidence passages.

Immediate Transition Options (Hop 1):
- Transition: [First_Hop_Retrieve] → [Scan_Index] (Condition: first_hop_retrieve)
  * Guidance: Review the retrieved passages.
  * Pitfalls to Avoid: Do not skip reading evidence details.

Subsequent Horizon (Hop 2):
- Transition: [Scan_Index] → [Bridge_Extract] (Condition: (always))
  * Guidance: Extract the bridge term.
  * Pitfalls to Avoid: Match Wikipedia capitalization.
```

`build_prompt` uses the paper §B.5 "Guidance Generation Prompt" text verbatim with slots `{task_description}`, `{graph_context_desc}`, `{subgraph_summary}`, `{query}`, `{recent_context}`, `{graph_source}`; local variant fills the slots with "the local procedural subgraph around the agent's active node" / "local procedural subgraph"; full variant uses the paper's phrases "the complete Procedural Graph governing the task structure and strategic guidance" / "complete Procedural Graph". Append one line: `Respond with a JSON object: {"guidance": "<your guidance text>"}`.

`retrieve` flow: skeleton → WARN `graph has no procedural guidance yet`, `""`. `localize_active` hit → `G_t = h_hop_neighborhood(u, h)`; miss → WARN `localized node not found, falling back to full graph`, `G_t = G`, `full_graph=True`. `G_t` with no edges → same "no procedural guidance yet" WARN, `""`. LLM raise / non-dict / missing, non-str, or empty `guidance` → WARN `guidance LLM error: <reason>`, `""`. `GraphLoadError` → WARN `load failed: <reason>`, `""`. INFO log `guidance.localize_hit` on hit; DEBUG `retrieval.serialize_subgraph`.

- [ ] **Step 1: Failing tests:**
  - `test_serialize_local_exact`: build the HotpotQA-shaped graph above and assert equality with the block above.
  - `test_serialize_hides_relation_labels`: `"LEADS_TO"`/`"PROVIDES_INPUT_FOR"` absent.
  - `test_serialize_full_graph_when_active_none`.
  - `test_retrieve_success`: stub llm returns `{"guidance": "do X"}` → `"do X"`; captured prompt contains the task, the serialized block, and the recent context.
  - `test_retrieve_skeleton_returns_empty_and_warns`: recorder sees a message starting `Could not validate a linked graph memory: graph has no procedural guidance yet`; llm never called.
  - `test_retrieve_miss_falls_back_to_full_graph`: prompt contains `complete Procedural Graph`; WARN recorded once.
  - `test_retrieve_empty_actions_is_hit_on_start`: no WARN.
  - `test_retrieve_llm_raises`, `test_retrieve_wrong_shape` (parametrized: `[]`, `{}`, `{"guidance": 5}`, `{"guidance": ""}`) → `""` + WARN containing `guidance LLM error`.
  - `test_retrieve_corrupt_file`: graph path holds `{not json` → `""` + WARN `load failed`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 5: `procedural_memory/refiner.py`

**Files:**
- Create: `procedural_memory/refiner.py`
- Test: `tests/procedural_memory/test_refiner.py`

**Interfaces:**
- Consumes: Task 3 (`ProceduralGraph`, `Weights`, `structural_score`, `unreachable_nodes`, `save_graph`, constants, `GraphLoadError`).
- Produces:
  - `class RefinerParseError(Exception)`
  - `@dataclass(frozen=True) NodeSpec(id: str, type: str, description: str)`, `EdgeSpec(source: str, target: str, relation: str, condition: str | None, guidance: str, pitfalls: str)`, `EditDelta(add_nodes: list[NodeSpec], delete_nodes: list[str], add_edges: list[EdgeSpec], delete_edges: list[tuple[str, str]])` with `EditDelta.from_json(data: Any) -> EditDelta` (missing arrays ⇒ empty; non-dict / non-list arrays / malformed items ⇒ `RefinerParseError`) and `.to_json() -> dict`
  - `@dataclass EvolutionResult(round_idx: int, accepted: bool, score_before: float, score_after: float | None, candidate_added: int, candidate_removed: int, rejection_reason: str | None)`
  - `tail_clip(text: str, max_tokens: int = 4000) -> str` — budget is `max_tokens * 4` characters; keeps the **end**; shorter input unchanged
  - `prepare_candidate(G, delta, *, cycles_forbidden: bool = True, strict: bool = True, auto_stub_missing_nodes: bool = False, source_session_id: str | None = None, refiner_round: int | None = None) -> tuple[nx.MultiDiGraph | None, list[str]]` — `(candidate, diagnostics)`; non-empty diagnostics ⇒ rejected
  - `class RejectionLog(path: Path)` — `.append(entry: dict) -> None`, `.recent(n: int = 5) -> list[dict]` (skips undecodable lines)
  - `class Refiner(graph: ProceduralGraph, rejections_path: Path, *, llm: Callable[[str], Any] | None = None, l_max: int = 4000, cycles_forbidden: bool = True, strict: bool = True, auto_stub_missing_nodes: bool = False, weights: Weights = Weights())` with `.propose(*, task_description: str, mode: str, available_tools: Sequence[str], trajectory: str, rejected_block: str) -> EditDelta` and `.evolve(trajectory: str, *, task_description: str = "", mode: str = "incremental", round_idx: int = 1, session_id: str | None = None, available_tools: Sequence[str] | None = None) -> EvolutionResult`

`prepare_candidate` order (paper §B.6): copy `G`; delete edges `(source, target)` across all relations; delete nodes (and incident edges); add nodes; add edges (upsert by relation). Diagnostics (strict): malformed or unknown `type`/`relation`; deleting `Start`/`End`; edge endpoint missing after adds (unless `auto_stub_missing_nodes`, which adds a `STATUS` stub with description `auto-stubbed`); then, if `cycles_forbidden`, repeatedly `nx.find_cycle` and drop the cycle's last edge until acyclic; finally any `unreachable_nodes` ⇒ a diagnostic naming them. With `strict=False` only malformed/unknown-vocabulary and reserved-node deletions stay fatal; dangling endpoints are created implicitly and left to the score.

`evolve` steps (spec §6.2): load retained graph (`GraphLoadError` → WARN `load failed`, result `rejection_reason="load_failed"`); `score_before`; `propose` (prompt = paper §B.5 Refiner Prompt verbatim, trajectory through `tail_clip`, rejected block = JSON of `RejectionLog.recent(5)` carrying `edit_delta`, `reason`); mode normalized per D5; llm exception → WARN `refiner unreachable: <reason>`, `accepted=False`, `rejection_reason="refiner_unreachable:<reason>"`, nothing appended, nothing written; `RefinerParseError` → `rejection_reason="parse_error:<reason>"`, nothing appended. Then `prepare_candidate`; diagnostics ⇒ WARN `structural reject: <diagnostics>`, append `{round, edit_delta, candidate_graph_snapshot, trajectories_tail, rejected_at, reason: "structural:<diag>"}`, `accepted=False`. Else `score_after = structural_score(candidate)`; `>= score_before` ⇒ `save_graph` + INFO `evolve.accept`; otherwise append with `reason: "score:<after> < <before>"` + INFO `evolve.reject`. `candidate_added = len(add_nodes)+len(add_edges)`, `candidate_removed = len(delete_nodes)+len(delete_edges)`.

- [ ] **Step 1: Failing tests** (stub `llm` is a callable returning a dict / raising; fixtures build a `Refiner` over `tmp_path`):
  - `test_tail_clip_keeps_ending` (long text → result ends with the input's last 50 chars and `len <= max_tokens*4`) and `test_tail_clip_short_unchanged`.
  - `test_edit_delta_from_json_ok`; `test_edit_delta_from_json_rejects_shapes` (`[]`, `{"add_nodes": "x"}`, an edge missing `guidance`).
  - `test_prepare_valid_candidate`: delta adds node `search` (ACTION), edges `Start→search`, `search→End`, deletes `(Start, End)` → no diagnostics; `Start→End` gone.
  - `test_delete_edges_spans_relations_and_add_edges_wins`: both `LEADS_TO` and `TRIGGERS` between `a`,`b`; delta deletes `(a,b)` and re-adds `LEADS_TO` → only `LEADS_TO` remains.
  - `test_prepare_rejects`: parametrized — edge to missing node; unknown relation; unknown node type; delete `Start`; delete `End`; exit-less cycle `a↔b` → diagnostics non-empty.
  - `test_cycle_closing_edge_removed_when_forbidden`: `a→b`, `b→a`, `b→End` → candidate is a DAG; with `cycles_forbidden=False` both kept.
  - `test_auto_stub_missing_nodes`.
  - `test_evolve_accepts_valid_edit`: `accepted=True`, `score_after >= score_before`, reloaded graph file contains `search`, node `provenance["refiner_round"] == round_idx`.
  - `test_evolve_structural_reject_leaves_file_untouched`: file bytes identical before/after; one line appended to `rejections.jsonl` with `reason` starting `structural:`; WARN recorded starting `Could not validate a linked graph memory: structural reject:`.
  - `test_evolve_score_reject_with_strict_false`: delta adds edge to missing node → `accepted=False`, `rejection_reason` starts `score:`, file untouched, line appended.
  - `test_evolve_llm_error`: stub raises → `accepted=False`, reason starts `refiner_unreachable`, WARN `refiner unreachable`, no rejections content, graph untouched.
  - `test_evolve_parse_error`: stub returns `["x"]` → reason starts `parse_error`.
  - `test_rejected_history_in_next_prompt`: after one rejection, the second `evolve` prompt contains the first reason string.
  - `test_rejections_corrupt_line_skipped`: file with `garbage\n` + one valid JSON line → `recent()` returns one entry; `evolve` does not raise.
  - `test_mode_alias`: `mode="incremental"` → prompt contains `scratch_incremental`.
  - `test_long_trajectory_clipped_in_prompt`: prompt contains the trajectory's last line and not its first line.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 6: `memory.py` facade

**Files:**
- Modify: `memory.py` (keep `MemoryEntry`, `record`, `retrieve`, `_hook_record`, `_hook_retrieve` untouched)
- Test: `tests/test_memory.py` (create)

**Interfaces:**
- Consumes: Tasks 3–5; `session.loadSession` (imported lazily inside `evolve_from_trajectory` to avoid an import cycle with `session.py`).
- Produces: `MemoryEngine(root: Path | None = None, *, llm: Callable[[str], Any] | None = None, session_loader: Callable[[str], list[dict]] | None = None)` gains
  - `retrieve_guidance(task: str, recent_actions: list[str]) -> str`
  - `record_triplet(src: str, relation: str, tgt: str, *, condition: str | None = None, guidance: str, pitfalls: str, source_session_id: str | None = None, refiner_round: int | None = None) -> None` (INFO `memory.record_triplet`; `GraphLoadError` → WARN `load failed`, no write)
  - `record_turn(session_id: str, thought: str, action: str, observation: str) -> list[tuple[str, str, str]]`
  - `evolve_from_trajectory(session_id: str, *, mode: str = "incremental", round_idx: int | None = None) -> EvolutionResult`
  - module-level functions `retrieve_guidance`, `record_triplet`, `record_turn`, `evolve_from_trajectory` delegating to `memoryEngine`.
  - Attributes `graph_path = memory_dir / "procedural_graph.json"`, `rejections_path = memory_dir / "rejections.jsonl"`.

`record_turn` rules: `verb = action_verb(action)`; `node = normalize_node_id(verb)`; `node == ""` ⇒ return `[]`; `verb.upper() in {"FINISH","DONE"}` ⇒ target is `End`. Source is the engine's remembered last node for `session_id` (in-memory dict) else `Start`. `src == tgt` ⇒ skip. Triplet: `LEADS_TO`, `condition=None`, `guidance=thought or f"Proceed with {action}"`, `pitfalls=""`, `source_session_id=session_id`; the new node's description is the raw `action`. Remember `tgt` as last node. `observation` is logged at DEBUG only.

`evolve_from_trajectory`: `messages = session_loader(session_id)`; `task_description` = first `user` message content; trajectory = `"{role}: {content}"` lines; `available_tools` = ids of current `ACTION` nodes; `round_idx` default `1`; delegate to `Refiner.evolve`.

The engine builds `ProceduralGraph`, `Guidance`, `Refiner` lazily so importing `memory` stays cheap.

- [ ] **Step 1: Failing tests** (`MemoryEngine(tmp_path, llm=stub, session_loader=lambda sid: [...])`):
  - `test_record_triplet_then_retrieve_guidance`: record `search →LEADS_TO→ read`; `retrieve_guidance("q", ["search(q=1)"])` calls the stub with a prompt containing `[search] → [read]` and returns its guidance.
  - `test_retrieve_on_fresh_engine_returns_empty` (skeleton; stub never called).
  - `test_corrupt_graph_file_degrades`: write `{bad` to `graph_path`; `retrieve_guidance` → `""`; `record_triplet` does not raise and the file content is unchanged; WARN recorded.
  - `test_record_turn_chain`: turns `search(q=1)`, `read(id=2)`, `FINISH()` → triples `[("Start","LEADS_TO","search"), ("search","LEADS_TO","read"), ("read","LEADS_TO","End")]`; node `search` description is `search(q=1)`; guidance equals the thought.
  - `test_record_turn_edge_cases`: empty action → `[]`, graph unchanged; same action twice → no self-loop; `"()"` → `[]`.
  - `test_record_turn_sessions_are_independent`: two session ids each start from `Start`.
  - `test_evolve_from_trajectory_uses_session_messages`: stub llm returns an empty-array delta; result `accepted=True`; stub prompt contains the user task text.
  - `test_legacy_record_retrieve_still_work`: instance `record`/`retrieve` roundtrip.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** `uv run pytest tests/test_memory.py tests/procedural_memory -v` → PASS.

### Task 7: `session.close_session`

**Files:**
- Modify: `session.py`
- Test: `tests/test_session.py` (create)

**Interfaces:**
- Consumes: Task 6 (`memory.memoryEngine`, `MemoryEngine.evolve_from_trajectory`).
- Produces: `SessionManager.close_session(session_id: str, *, evolve: bool = False, memory_engine: Any | None = None) -> EvolutionResult | None` and module-level `close_session(session_id, *, evolve=False)`. Always writes `closed_at` (same format as `_now`) into the session file; unknown session ⇒ `FileNotFoundError`. `evolve=True`: `(memory_engine or memory.memoryEngine).evolve_from_trajectory(session_id)`, INFO `session.close` with `accepted`, `score_before`, `score_after`, return the result; otherwise INFO `session.close` and return `None`. `import memory` happens inside the method.

- [ ] **Step 1: Failing tests** (`SessionManager(tmp_path)` + a fake engine recording calls):
  - `test_close_marks_closed_at_and_keeps_messages`.
  - `test_close_does_not_evolve_by_default`: fake engine never called, returns `None`.
  - `test_close_with_evolve_calls_engine_once`: called with `session_id`; returns its result.
  - `test_close_unknown_session_raises`.
  - `test_load_session_still_works_after_close`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 8: `agent_loop.py`

**Files:**
- Create: `agent_loop.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: Task 6 `MemoryEngine`; Task 7 `SessionManager.close_session`; `models.completion_default`.
- Produces:
  - `@dataclass TurnOutput(thought: str, action: str, raw: str)` with classmethod `from_message(msg: str) -> TurnOutput` — `Thought:` text runs until an `Action:` line; `Action:` is the rest of that line; missing parts ⇒ `""`.
  - `@dataclass RunResult(session_id: str, turns: list[TurnOutput], final_answer: str, closed: bool)` — `final_answer` = last non-empty `thought`.
  - `run(task: str, session_id: str, max_turns: int = 3, *, on_observation: Callable[[str], str] | None = None, evolve_on_close: bool = False, solver: Callable[[str], str | None] | None = None, memory_engine: MemoryEngine | None = None, session_manager: SessionManager | None = None) -> RunResult`
  - `run_with_guidance(task: str, session_id: str | None = None, max_turns: int = 1, **kw) -> RunResult` — creates a session when `session_id is None`, then `run`.
  - `SOLVER_PROMPT` = paper §B.5 "Solver Execution Prompt" verbatim with slots `{system_prompt}`, `{procedural_graph_guidance}`, `{trajectory}`; `SYSTEM_PROMPT` a one-line constant.

Loop per spec §7.4: append `{"role": "user", "content": task}` once before turn 1; per turn `retrieve_guidance(task, last 3 non-empty actions)` → `solver(prompt)` (`None` ⇒ `""`) → parse → observation (`on_observation(action)` or `f"[stub-obs] {action}"`) → append assistant + tool messages → `record_turn` → stop on action verb `FINISH`/`DONE` (case-insensitive) or `t == max_turns`. Then `session_manager.close_session(session_id, evolve=evolve_on_close, memory_engine=memory_engine)`; `closed=True`. Empty guidance renders as `(none)`. `trajectory` = `"{role}: {content}"` lines of the session so far.

- [ ] **Step 1: Failing tests** (stub solver returning scripted messages; `MemoryEngine(tmp_path)` with a counting stub llm; `SessionManager(tmp_path)`):
  - `test_from_message_variants`: both parts; thought only; multi-line thought; no markers → empties except `raw`.
  - `test_runs_k_turns_and_retrieves_k_times`: `max_turns=3`, solver never finishes → 3 `retrieve_guidance` calls, 3 assistant + 3 tool messages (+1 user), `closed=True`.
  - `test_finish_stops_early`: second reply `Action: FINISH()` → 2 turns.
  - `test_guidance_in_solver_prompt`: after a recorded triplet, the next solver prompt contains the stub guidance text.
  - `test_stub_observation_default_and_custom`.
  - `test_solver_none_and_garbage_do_not_crash`: `None`, `""`, `"hello"` replies → loop completes, no triplets written.
  - `test_close_session_called_last_and_evolve_flag`: default → engine's `evolve_from_trajectory` not called; `evolve_on_close=True` → called exactly once, after all turns.
  - `test_final_answer_is_last_nonempty_thought`.
  - `test_run_with_guidance_defaults_single_turn`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.

### Task 9: `main.py` — prompt via loop, `whale refine`

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py` (create; typer `CliRunner`)

**Interfaces:**
- Consumes: `agent_loop.run`, `agent_loop.run_with_guidance`, `session.initSession`, `session.listSessions`, `memory.evolve_from_trajectory`.
- Produces:
  - `prompt(text, no_loop: bool = Option(False, "--no-loop"))` — `no_loop` ⇒ `run_with_guidance(text)`; else `run(text, session.id, max_turns=3)`; echoes `result.final_answer`. Drops the old direct `completion_default` / `_hook_record` use.
  - `refine(session_id: str | None = Option(None, "--session-id"), rounds: int = Option(10, "--round"), last: bool = Option(False, "--last"))` — resolves the session (explicit id, else most recent per D10; none exist ⇒ echo error, exit 1); for `k in 1..rounds` calls `evolve_from_trajectory(sid, round_idx=k)` and echoes `round k: accepted=<bool> score <before> -> <after|n/a> reason=<reason|->`. Unknown `--session-id` ⇒ echo `session not found: <id>`, exit 1. `rounds < 1` ⇒ exit 2.

- [ ] **Step 1: Failing tests** (monkeypatch `main.run`/`main.run_with_guidance`, `main.evolve_from_trajectory`, `main.listSessions`, `main.initSession`):
  - `test_prompt_routes_through_loop`: `max_turns=3`; output is the stub's `final_answer`.
  - `test_prompt_no_loop`: `run_with_guidance` called.
  - `test_default_command_routing`: bare `"hello there"` reaches `prompt`; `refine --last` reaches `refine`.
  - `test_refine_runs_k_rounds_with_round_idx`: `--round 3` → calls with `round_idx` 1, 2, 3.
  - `test_refine_defaults_to_most_recent_session`, `test_refine_no_sessions_exits_1`, `test_refine_unknown_session_exits_1`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** `uv run pytest -v` (whole suite) → PASS.

### Task 10: Smoke script and README

**Files:**
- Create: `scripts/smoke_pg.py`
- Modify: `README.md` (append a "Procedural graph memory — next steps" list under the 11.10.2026 entry)

**Interfaces:**
- Consumes: `memory.MemoryEngine`.
- Produces: `python scripts/smoke_pg.py` prints four numbered check lines and `OK`, exit 0; exit 1 naming the failing step otherwise. Uses a `tempfile.TemporaryDirectory()` root and a deterministic stub llm returning `{"guidance": "stub"}` for guidance prompts and a fixed valid delta (add node `search`, edges `Start→search`, `search→End`, delete `(Start, End)`) when the prompt contains `cognitive architect optimizing`.

Checks (spec §9): (1) fresh graph is `Start →LEADS_TO End`; (2) `record_triplet` persists and survives a reload by a second engine; (3) `retrieve_guidance` returns a non-empty string for a recorded node; (4) `evolve_from_trajectory` returns `accepted=True`.

README list (from spec §11): legacy `entries.json` migration, real ReAct tool dispatcher, embedding-based localization backup, embedding-based `H_rejected` retrieval, per-task typed schema file.

- [ ] **Step 1:** Write the script. **Step 2:** `uv run python scripts/smoke_pg.py` → prints `OK`, exit 0. **Step 3:** Update README. **Step 4:** `uv run pytest -v` → full suite PASS.

---

## Self-Review

- **Spec coverage:** §2 facade → T6; §3 layout → T3/T5/T6; §4.1 schema → T3 (D3); §4.2 → T3; §4.3 score + strict rules → T3/T5; §5 guidance + WARN table → T4 (rows 1–4), T5 (rows 5–6), T6 (row 1 for writes); §6 refiner/rejections → T5; §6.4 wiring → T7/T8/T9 (D1); §7.1 → T2; §7.2 → T7; §7.3 → T9; §7.4 → T8; §7.5 → T9; §8 levels → T1 + per-task logging; §9 tests + smoke → every task + T10; §10 file plan → all files covered, `pyproject.toml` in T1; §11 → T10.
- **Type consistency:** `ProceduralGraph`, `Weights`, `EditDelta`, `EvolutionResult`, `MemoryEngine.evolve_from_trajectory(session_id, *, mode, round_idx)`, `SessionManager.close_session(..., evolve, memory_engine)` are spelled identically wherever consumed.
- **Proportion:** the plan carries decisions, signatures, test names/assertions, and the one exact string the spec fixes (serializer output); bodies are left to the implementer.

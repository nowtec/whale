# Procedural Graph Memory Engine — Design Spec

**Date:** 2026-10-11
**Status:** Approved (brainstorming), pre-implementation
**Paper:** *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents*, Lu et al. (2026). Source PDF at `docs/Procedural Graphs Self-Evolving Execution Structures for LLM Agents .pdf`. Sections 3.1, 3.2, 3.3 + Algorithm 1 are in scope.

---

## 1. Goals & Non-Goals

### Goals

Replace `whale`'s flat `MemoryEngine` with a **Procedural Graph** (PG) memory engine implementing the paper's three pillars:

1. **Graph store** — directed, attributed graph `G = (V, R, E, Φ)` of `(procedure, relation, procedure)` triplets carrying `condition / guidance / pitfalls` attributes (paper §3.1).
2. **Online Generative Guidance** — localize the active node from the agent's trajectory, extract the 2-hop neighborhood `N_h(u_t)`, and translate it into step-level situational guidance appended to the solver's prompt (paper §3.2).
3. **Offline Self-Evolution** — at the end of each agent run, an LLM refiner contrasts successful vs. failing trajectories and proposes edits (`add_nodes`, `delete_nodes`, `add_edges`, `delete_edges`); candidates are committed when a structural-validation score does not decrease; rejected candidates are retained as negative evidence (paper §3.3 + Algorithm 1). Only by calling `refine` command

### Non-Goals (this iteration)

- Full ReAct/Tool dispatcher — we add a **minimal** K-turn AgentLoop primitive only to expose the right memory boundaries (§4.2). The tool surface stays text-shaped.
- We do no need any migrations
- Multi-task training/eval splits. The K-round evolution loop is exposed via `whale refine` for callers to drive manually if they want larger runs.

---

## 2. Architecture

```
+----------------+         +----------------------------------+
|   AgentLoop    | <-----  |            memory.py             |
|   agent_loop.py|         |     (facade — 3 methods)         |
+--------+-------+         +-----------------+----------------+
         |                                   |
         | retrieve_guidance(task, turns)     | record_triplet / record_turn
         | record_turn(...)                   | evolve_from_trajectory(...)
         | close_session -> evolve(...)       |
         v                                   v
+----------------------------------+     +----------------------------------+
|        procedural_memory/        |     |        procedural_memory/        |
|  graph.py      (paper §3.1)      |     |  guidance.py    (paper §3.2)     |
|  - MultiDiGraph via networkx     |<----+  - localize → N_h → Psi → g_t    |
|  - load/save (node_link JSON)    |     |                                  |
|  - structural_score(NX-based)    |     +-----------------+----------------+
+-----------------+----------------+                       |
                  |                                        v
                  |              +-------------+   models.completion_default_json()
                  |              |             |   (litellm structured JSON)
                  +------------> | refiner.py  | <--------+
                                 | (paper §3.3)|
                                 | Algorithm 1 |
                                 +------+------+
                                        |
                                        v
                       ~/.whale/memory/procedural_graph.json
                       ~/.whale/memory/rejections.jsonl
```

**Three-method facade** (the public API of `memory.py`):

```python
# §5 — online
memory.retrieve_guidance(task: str, recent_actions: list[str]) -> str

# §4 — graph writes
memory.record_triplet(src: str, relation: str, tgt: str, *, condition=None,
                      guidance: str, pitfalls: str,
                      source_session_id: str | None = None,
                      refiner_round: int | None = None) -> None

# §4 — turn convenience (used by AgentLoop; backed by record_triplet)
memory.record_turn(session_id: str, thought: str, action: str,
                   observation: str) -> list[None]
# Derives one or more (action, LEADS_TO, action) triplets from the turn
# and calls record_triplet for each. Returns the list of write effects
# (mostly so tests can assert).

# §6 — offline
memory.evolve_from_trajectory(session_id: str, *, mode: str = "incremental",
                              round_idx: int | None = None) -> EvolutionResult
```

`AgentLoop` does **not** import from `procedural_memory/`. The subpackage is an internal implementation detail behind `memory.py`. `record_triplet` is the data-level write; `record_turn` is the convenience that `AgentLoop` calls once per turn and that internally derives triplets from the action sequence.

---

## 3. On-Disk Layout

```
~/.whale/
├── memory/
│   ├── entries.json             # legacy flat entries; won't be used
│   ├── procedural_graph.json    # node-link JSON, written by networkx
│   └── rejections.jsonl         # H_rejected — one JSON object per line
├── sessions/
│   └── <session_id>.json
└── logs/
    └── whale.log
```

- `procedural_graph.json` is the round-trip output of `networkx.readwrite.json_graph.node_link_data(G)`. Loaded with `node_link_graph`. Single atomic write per change (`Path.write_text` to tmp + `os.replace`).
- `rejections.jsonl` is append-only and grows slowly; a future compaction pass is out of scope.

---

## 4. Data Model

### 4.1 Graph Structure

`networkx.MultiDiGraph`. Nodes and edges carry dictionaries; the schema is enforced at `record_triplet` and at `Refiner.evolve`'s `PrepareCandidate` step.

**Node** `v ∈ V`:

| Field         | Type                                            | Notes                                              |
| ------------- | ----------------------------------------------- | -------------------------------------------------- |
| `id`          | `str` (snake_case)                              | Required unique. Reserved: `Start`, `End`.         |
| `node_type`   | `enum {ACTION, REASONING, STATUS, SKILL}`       | `Start` and `End` use a synthesized `STATUS` type.  |
| `description` | `str`                                           | Free text.                                         |
| `created_at`  | `iso8601`                                       | Set on first record.                               |
| `provenance`  | `{source_session_id, refiner_round}` | Empty when seeded manually.                        |

**Edge** `e = (u, r, v) ∈ E ⊆ V × R × V`:

| Field       | Type                                                      | Notes                                                                               |
| ----------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `relation`  | `enum R = {LEADS_TO, TRIGGERS, PROVIDES_INPUT_FOR, CONVERGES_TO}` | Required. Exact strings, capitalized with underscores.                             |
| `condition` | `str \| None`                                             | Natural-language precondition. `None` ⇒ unconditional (rendered as `(always)`).     |
| `guidance`  | `str`                                                     | Required. What to do next and why.                                                  |
| `pitfalls`  | `str`                                                     | Required. Common mistakes / formatting traps to avoid.                              |
| `provenance`| `{source_session_id, refiner_round, created_at}`          | Audit trail.                                                                       |

**Skeleton**: when `procedural_graph.json` is absent, the engine creates `Start →LEADS_TO End` on first load (paper Mode 4 / 5).

### 4.2 Subgraph Extraction (paper §3.2 eq. 2)

```python
def h_hop_neighborhood(self, u, h=2):
    """N_h(u) — directed forward neighborhood reachable from u within h hops."""
```

If `localize_active` returns `None`, `G_t = G` (full graph fallback).

### 4.3 Structural Validation (the NetworkX stand-in for `S_val`)

```python
def structural_score(G) -> float:
    """Composite [0, 1] weighted score using only networkx diagnostics.
       Stands in for paper's empirical S_val until a benchmark harness ships.
    """
    reach      = fraction_of_nodes_with_path_to_terminal(G)        # weight 0.5
    acyclic    = 0.0 if (cycles_forbidden and not nx.is_directed_acyclic_graph(G)) else 1.0  # weight 0.2
    edit_q     = 1 - dangling_edges(G) / max(1, G.number_of_edges())                        # weight 0.2
    type_valid = 1 - invalid_typed_nodes_or_relations(G) / max(1, G.number_of_nodes() + G.number_of_edges())  # weight 0.1
    return 0.5 * reach + 0.2 * acyclic + 0.2 * edit_q + 0.1 * type_valid
```

Weights live in `Refiner(...)` constructor (`reachability_w`, `acyclicity_w`, `edit_quality_w`, `type_validity_w`).

**Per-edit rules (strict mode, default `True`):**

1. Edge endpoints must already exist as nodes, *or* appear in the same `EditDelta` via `add_nodes`. Otherwise the candidate fails validation → appended to `H_rejected`.
2. Reachability: every node must have a directed path to a terminal (zero-out-degree) node. If cycles are disallowed and the candidate introduces a cycle, remove the cycle-closing edge last (deepest edge first by edge-key tiebreak) before scoring. If unreachable nodes remain → fail.
3. Relation vocabulary: must be one of the four in `R`.
4. Node type vocabulary: must be one of the four (excluding `Start`/`End`'s synthesized `STATUS`).

`auto_stub_missing_nodes: bool = False` is exposed for future callers (a dataset builder, a CLI flag) without changing `memory.py`'s public API.

---

## 5. Online Generative Guidance (`procedural_memory/guidance.py`)

Implements `Psi(G_t, q, T_window) -> g_t`. Wired into `memory.retrieve_guidance(task, recent_actions)`.

### 5.1 Pipeline

```
input  q, T_window (last w=3 turns)
       │
       ▼
localize_active(T_window)   →   u_t  (or None on miss)
       │
       ├── hit ──► G_t = N_h(u_t, h=2)
       │           (paper's hop-2 default)
       │
       └── miss ─► G_t = G   (fallback per paper §3.2)
       │
       ▼
serialize_local(u_t, G_t, q, T_window)   # paper's local serializer (see §5.2)
       │
       ▼
completion_default_json(SUBGRAPH_PROMPT, response_format={type: json_object})
       │
       ▼
g_t  (rendered into solver prompt as: "Procedural Graph Guidance: {g_t}")
```

### 5.2 Local Serializer (verbatim from paper §B.5)

```
Active Cognitive Node: [<id>] (Type: <node_type>)
Description: <description>

Immediate Transition Options (Hop 1):
- Transition: [<u>] → [<v>] (Condition: <cond or (always)>)
  * Guidance: <Φ(e).guidance>
  * Pitfalls: <Φ(e).pitfalls>

Subsequent Horizon (Hop 2):
- Transition: [<u>] → [<v>] (Condition: ...)
  * Guidance: ...
  * Pitfalls: ...
```

Relation labels are **not** printed (matches the paper's local serializer).

### 5.3 Failure Modes → WARN logs

Every non-fatal engine failure is logged at WARN through `logger.py`. **All WARN messages from the procedural-memory engine are prefixed with the literal phrase `Could not validate a linked graph memory`** so the user can grep one string and find every degraded-path event. The variable suffix distinguishes the cause:

| Condition                                                                            | Log message (WARN)                                                                    | Engine behavior            |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------- |
| JSON file unreadable or invalid                                                      | `Could not validate a linked graph memory: load failed: <reason>`                     | `retrieve_guidance` returns `""`; engine degrades |
| `localize_active` finds no match                                                      | `Could not validate a linked graph memory: localized node not found, falling back to full graph` | Fallback to full `G`       |
| Graph is the empty skeleton (`Start →LEADS_TO End` only)                              | `Could not validate a linked graph memory: graph has no procedural guidance yet`      | Returns empty string       |
| `completion_default_json` returns non-JSON or raises                                  | `Could not validate a linked graph memory: guidance LLM error: <reason>`              | Returns empty string; solver proceeds |
| Refiner LLM error during `evolve_from_trajectory`                                    | `Could not validate a linked graph memory: refiner unreachable: <reason>`             | Skip evolution, no mutation |
| Structural validation rejects refiner candidate                                       | `Could not validate a linked graph memory: structural reject: <diagnostics>`         | Append to `H_rejected`, evolution round reports `accepted=False` |

Display is governed by `LOG_LEVEL`. At `LOG_LEVEL=WARN` (default), each row above is printed exactly once per occurrence. At `LOG_LEVEL=INFO` the same messages still print (WARN ≥ INFO). At `LOG_LEVEL=ERROR` they are suppressed. INFO-level operational events (`memory.record_triplet`, `guidance.localize_hit`, `evolve.accept`, `evolve.reject`) and DEBUG-level telemetry (`retrieval.serialize_subgraph`, neighborhood hop counts) follow the existing `logger.py` level filter.

The phrase `Could not validate a linked graph memory` (case-sensitive, lowercase) is the single user-grep token.

### 5.4 Configuration

```python
Guidance(
    h: int = 2,                # paper default h=2
    window_w: int = 3,         # paper default w=3
    model: str | None = None,  # defaults to env-configured completion_default
)
```

---

## 6. Offline Self-Evolution (`procedural_memory/refiner.py`)

Implements Algorithm 1. Two public functions:

### 6.1 `Refiner.propose(...) -> EditDelta`

Drives Step 2 of Algorithm 1 using the paper's Section B.5 Refiner prompt verbatim:

```
You are an expert cognitive architect optimizing a Procedural Graph for an intelligent agent.
Task context: {task_description}
Refinement mode: {mode}
Available Tool Actions: {available_tools_list}
Recent execution trajectories: {attempts_block}
Current Procedural Graph representation: {current_graph_json}
Previously rejected candidates: {rejected_block}
…
Please propose the exact set of edits to perform. Output a single valid JSON block:

{ "add_nodes":   [{"id":..., "type": "ACTION", "description":...}],
  "delete_nodes": ["node_id"],
  "add_edges":   [{"source":..., "target":..., "relation":...,
                    "condition":..., "guidance":..., "pitfalls":...}],
  "delete_edges":[{"source":..., "target":...}] }
```

The `{attempts_block}` is the user's session trajectory, tail-clipped to `L_max` (default 4000 tokens) preserving the *ending*. The `{rejected_block}` is the serialized `H_rejected` history. The prompt mandates `add_nodes` and `add_edges` be emitted together — that's what makes strict-mode structural validation work.

```python
@dataclass(frozen=True)
class EditDelta:
    add_nodes:   list[NodeSpec]
    delete_nodes: list[str]
    add_edges:   list[EdgeSpec]
    delete_edges: list[tuple[str, str]]   # (source, target) — applies across relations
```

`completion_default_json` parses the response. Malformed JSON raises `RefinerParseError`; the evolution loop catches this and short-circuits the round.

### 6.2 `Refiner.evolve(...) -> EvolutionResult`

Implements Algorithm 1's main loop.

```python
@dataclass
class EvolutionResult:
    round_idx:           int
    accepted:            bool
    score_before:        float
    score_after:         float | None   # populated only when validation ran
    candidate_added:     int            # nodes + edges
    candidate_removed:  int
    rejection_reason:    str | None
```

```text
1. snapshot G_retained from disk + its score
2. ΔG = refiner.propose(G_retained, C_k, R_k)       # Step 2 of Algorithm 1
3. G_cand = G_retained.copy() applied with ΔG      # delete-edges, delete-nodes,
                                                     # add-nodes, add-edges
4. structural checks (§4.3):
   on failure: append (ΔG, G_cand, E_k, diagnostics) to H_rejected
                return EvolutionResult(accepted=False, rejection_reason="structural:<reason>")
5. S_cand = structural_score(G_cand)
6. accept iff S_cand ≥ S_before:
     on accept : G_cand → disk; update retained state
     on reject : append (ΔG, G_cand, E_k, S_cand) to H_rejected
7. return EvolutionResult(...)
```

`Refiner.evolve` defaults to `K=1` per invocation. `whale refine` exposes `K=10` (paper's round budget for EnterpriseArena self-evolution).

### 6.3 Rejection Memory

Persisted to `~/.whale/memory/rejections.jsonl`. One JSON object per line:

```json
{ "round": 4, "edit_delta": {...}, "candidate_graph_snapshot": "...",
  "trajectories_tail": "...", "rejected_at": "...", "reason": "structural:..." }
```

`Tail_{L_max}` from the paper is reused — clips from the *start*, preserves the *end*, never silently truncates the ending.

### 6.4 Boundary Wiring

| Caller                              | When                              | What fires                                                            |
| ----------------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| `agent_loop.run(task, max_turns=K)` | After K turns (or task completion) | `session.close_session(sid)` → `memory.evolve_from_trajectory(sid)`   |
| `whale refine [--session-id SID]`   | Manually on demand                | same path; `--last` re-runs the most recent session                   |
| Tests                               | N/A                               | Direct calls to `Refiner.evolve(...)` / `memory.evolve_from_trajectory(...)` |

A future ReAct/AgentLoop replaces `session.close_session()` with `agent_run_complete()`; `memory.evolve_from_trajectory` is the only thing that needs to know.

---

## 7. Models, Sessions, CLI

### 7.1 `models.py` — adds `completion_default_json(prompt: str, response_format=None) -> Any`

- `response_format={"type": "json_object"}` by default for refiner + guidance callers.
- Returns the parsed JSON.
- Raises `CompletionJSONError` on parse failure.
- Reuses `litellm.completion(...)` and `OPENAI_MODEL`.

### 7.2 `session.py` — adds `close_session(session_id: str) -> None`

- Marks `closed_at` on the session file.
- Loads the full trajectory.
- Calls `memory.evolve_from_trajectory(session_id)`.
- Logs the `EvolutionResult` at INFO (`session.close`, `accepted=...`, `score_before=...`, `score_after=...`).
- Existing `appendMessage` is unchanged.

### 7.3 `main.py` — `prompt` routes through `agent_loop.run()`

```python
@app.command(hidden=True)
def prompt(text: str = typer.Argument(..., help="Prompt to execute.")) -> None:
    session = initSession()
    result = agent_loop.run(task=text, session_id=session.id, max_turns=3)
    typer.echo(result.final_answer)
```

`run_with_guidance(task, session_id=None, max_turns=1)` is exposed for callers that want to keep single-shot behavior without the loop (`--no-loop` flag).

### 7.4 `agent_loop.py` — minimal K-turn ReAct

```python
@dataclass
class TurnOutput:
    thought: str
    action:  str        # parsed from `Action: <verb>(...)` line; empty if none
    raw:     str        # full solver message

@dataclass
class RunResult:
    session_id:    str
    turns:         list[TurnOutput]
    final_answer:  str         # last non-empty `thought` text
    closed:        bool        # True once close_session ran

def run(task: str, session_id: str, max_turns: int = 3, *,
        on_observation: Callable[[str], str] | None = None) -> RunResult:
    """Per turn t in 1..max_turns:
        g_t        = memory.retrieve_guidance(task, recent_actions=last 3 actions)
        msg_t      = solver(task, history=session.messages, g_t)        # ReAct prompt
        parsed_t   = TurnOutput.from_message(msg_t)                     # 'Thought:' & 'Action:' lines
        obs_t      = on_observation(parsed_t.action) if on_observation else f"[stub-obs] {parsed_t.action}"
        session.append_message({"role": "assistant", "content": msg_t})  # via session.appendMessage
        session.append_message({"role": "tool",       "content": obs_t}) # stub observation channel
        memory.record_turn(session_id=session_id, thought=parsed_t.thought, action=parsed_t.action, observation=obs_t)
        if parsed_t.action in {"FINISH", "DONE"} or t == max_turns:
            break
    session_manager.close_session(session_id)        # triggers memory.evolve_from_trajectory
    return RunResult(...)
```

Stub defaults: `on_observation=None` echoes the action back as `[stub-obs] <action>` so the loop self-terminates cleanly without a real tool surface. The ReAct message format — `Thought: ...\nAction: <verb>(...)` — is parsed line-prefix aware; if the LLM emits only a Thought the action is `""` and the loop continues with the stub observation `""`.

`on_observation` defaults to a stub that echoes the action back as observation. Real tools will plug in later without touching memory.

### 7.5 CLI addition — `whale refine`

```
whale refine [--session-id SID] [--round K] [--last]
```

Reuses `Refiner.evolve(...)` with the chosen session as input trajectory and `K` rounds.

---

## 8. Errors, Logging, Failure Modes

| Level   | Trigger                                                                                       | Effect                                                       |
| ------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| ERROR   | Disk write of `procedural_graph.json` fails (atomic rename target missing or unwritable)      | Raise `GraphPersistenceError`; operation aborted             |
| WARN    | All six `Could not validate a linked graph memory: <suffix>` rows in §5.3                     | Engine degrades gracefully; guidance/evolution skipped       |
| INFO    | `memory.record_triplet`, `guidance.localize_hit`, `evolve.accept`, `evolve.reject`            | Routed to `~/.whale/logs/whale.log`                          |
| DEBUG   | `retrieval.serialize_subgraph`, token-count telemetry, neighborhood hop counts                | Opt-in via `LOG_LEVEL=DEBUG`                                 |

Display is governed by `LOG_LEVEL`: at default `LOG_LEVEL=INFO` (set in `logger.py`), WARN rows from §5.3 are emitted on every occurrence. Suppressing them requires `LOG_LEVEL=ERROR`. The user can flip `LOG_LEVEL=WARN` to silence INFO but keep the WARN events visible, or `LOG_LEVEL=ERROR` to silence both WARN and INFO — the engine behavior does not change, only the visibility of the diagnostic.

The exact phrase `Could not validate a linked graph memory` is the user-discriminator: grepable, single source of truth for "memory engine degraded gracefully."

---

## 9. Testing (pytest)

```
tests/procedural_memory/test_graph.py
tests/procedural_memory/test_guidance.py
tests/procedural_memory/test_refiner.py
tests/test_memory.py
tests/test_agent_loop.py
tests/test_session.py   # extended with close_session
```

- Graph tests: node-link round-trip; `localize_active` exact-match with time tiebreak; `h_hop_neighborhood` (h=0,1,2); `structural_score` known inputs (empty skeleton = 1.0, single dangling edge = 0.5 + edit_q penalty, etc.).
- Guidance tests: local serializer output matches paper's HotpotQA excerpt shape; full-G fallback when localize misses; prompt assembly contains expected placeholders. Stub LLM.
- Refiner tests: Algorithm 1 round-trip with a stub proposer returning `noop`/fixed edits; accept/reject paths; rejection memory append; structural-only gate works on synthetic trajectories.
- `memory.py` facade tests: `retrieve_guidance` callable end-to-end; `evolve_from_trajectory` invoked from `session.close_session`.
- `agent_loop` tests: K=3 loop with stub solver; per-turn `retrieve_guidance` invoked exactly K times; final call is `close_session → evolve_from_trajectory`.

### Smoke script — `scripts/smoke_pg.py`

Runs the new stack end-to-end with a deterministic stub refiner. User can verify:

1. Skeleton `Start →LEADS_TO End` is materialized on first run.
2. `record_triplet` writes a new edge and reloads.
3. `retrieve_guidance` returns the local serializer's output for a query matching a recorded node.
4. `evolve_from_trajectory` returns an `EvolutionResult` with `accepted=True` when the stub refiner proposes a structurally valid edit.

---

## 10. File Plan

**New**

- `procedural_memory/__init__.py`
- `procedural_memory/graph.py`
- `procedural_memory/guidance.py`
- `procedural_memory/refiner.py`
- `agent_loop.py`
- `scripts/smoke_pg.py`

**Modified**

- `memory.py` — facade over `procedural_memory/`. Keeps `_hook_record`/`_hook_retrieve` shims for the legacy path.
- `models.py` — adds `completion_default_json`.
- `session.py` — adds `close_session`, surfaces `evolve_from_trajectory` integration.
- `main.py` — `prompt` routes through `agent_loop.run()`; adds `whale refine`.
- `pyproject.toml` — `networkx>=3.0`.

---

## 11. Open Decisions Deferred

- Migration script for legacy `entries.json` triplets — out of scope this iteration.
- Real ReAct tool dispatcher — out of scope; `agent_loop` is a thin primitive.
- Embedding-based localization backup (paper notes exact-match can miss paraphrased actions).
- Embedding-based similarity for `deserialize` of `H_rejected` (large rejection mems).
- A separate `_typed_schema.json` for per-task attribute fields beyond `condition/guidance/pitfalls` (paper §3.1 mentions schema configurable per task).

All five are noted in the README's "Next steps" once this spec lands.

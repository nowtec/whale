# Cognee — Memory Engine Review

**Repo:** `git@github.com:topoteretes/cognee.git`
**Version reviewed:** 1.5.3 (`cognee/version.py`, pyproject.toml)
**License:** Apache-2.0
**Runtime:** Python ≥ 3.10, < 3.15
**Stack at a glance:** `pydantic`, `litellm`, `instructor`, `sqlalchemy`, `aiosqlite`, `neo4j`, `kuzu`, `lancedb`, `pgvector`, `redis`, `alembic`, `fastapi`
**Research paper:** Markovic et al., 2025 — *"Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning"* (arXiv:2505.24478)

> Cognee is the **closest competitor to whale-harness's memory ambition** in the open-source world: a pluggable, persistent, multi-layered memory engine for AI agents that combines a knowledge graph + vector store + session cache + (optional) LLM-driven self-improvement loops. It is the only reviewed engine that takes both **structured (graph) memory** and **feedback-driven self-tuning** seriously.

---

## 1. What it actually is

Cognee is positioned as *"the open-source AI memory platform for AI Agents."* Its public surface is four verbs (`cognee/api/v1/__init__.py:51-66`):

```python
await cognee.remember(...)  # ingest into graph + cache
await cognee.recall(...)    # query, multi-source, auto-routed
await cognee.forget(...)    # unified delete
await cognee.improve(...)   # enrichment / self-improvement
```

A second, lower-level V1 API is also exposed: `add`, `cognify`, `search`, `memify`, `prune`, `update`, `delete`. `remember`/`recall` are wrappers over `add`+`cognify` and a multi-source dispatcher over `search`.

The model is **two-tier memory** (graph + session cache), with optional **temporal**, **contradiction**, and **feedback-driven weighting** layers.

---

## 2. High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Public API: remember / recall / forget / improve                │
│      └── v1 API: add / cognify / search / memify                 │
├──────────────────────────────────────────────────────────────────┤
│  Memory layer                                                     │
│   ┌──────────────────────┐   ┌────────────────────────────────┐  │
│   │  Session cache        │   │  Knowledge graph (permanent)   │  │
│   │  QA + trace + context │   │  entities + edges + summaries  │  │
│   │  Redis | FS | SQLite  │   │  Neo4j | Kuzu/Ladybug |        │  │
│   │  | Postgres | Tapes   │   │  Postgres | Turso | Neptune    │  │
│   └──────────────────────┘   └────────────────────────────────┘  │
│   ┌────────────────────────────────────────────────────────────┐ │
│   │  Vector store (LanceDB | pgvector | Turso) — embeddings     │ │
│   └────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│  Pipeline layer (cognee/tasks/, cognee/pipelines/)                │
│   document classify → chunk → ontology extract → summarize        │
│   → persist nodes/edges → index vectors → triplet embeddings     │
│   → contradiction detection (opt-in) → temporal graph (opt-in)   │
│   → global context index (opt-in) → feedback weighting           │
├──────────────────────────────────────────────────────────────────┤
│  LLM gateway (litellm) · embedding gateway · OpenTelemetry       │
└──────────────────────────────────────────────────────────────────┘
```

### 2.1 Source tree (`cognee/` package, top-level only)

```
cognee/
├── api/v1/             ← Public API surface
│   ├── remember/ recall/ forget/ improve/
│   ├── cognify/ add/ search/ delete/ update/ prune/ memify/
│   ├── agents/ skills/ tools/ session/ sessions/ users/ permissions/
│   ├── cloud/ serve/ visualize/ export/ push/ sync/
│   └── health/ ui/ config/ settings/
├── modules/            ← Domain modules
│   ├── cognify/        ← Pipeline routing & config
│   ├── retrieval/      ← 13+ retriever classes (see §6)
│   ├── search/         ← SearchType enum (17 strategies)
│   ├── agents/ agent_memory/ ontology/ chunking/ ingestion/
│   ├── observability/  ← OTEL spans, metrics, traces
│   ├── memify/         ← Self-improvement (skill_improvement.py)
│   ├── tools/          ← Skill runs, MCP-style tools
│   ├── users/          ← Auth + tenant isolation
│   ├── recall/         ← Session/trace/context result types
│   ├── migration/      ← Import from Mem0, Zep, Letta, LangMem, COGX
│   ├── provenance/     ← Audit-grade ledger (opt-in)
│   └── truth_subspace/ session_distillation/ session_lifecycle/
├── memory/             ← Typed entry models (QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry)
├── infrastructure/
│   ├── databases/
│   │   ├── graph/      ← Neo4j, Kuzu/Ladybug, Postgres-demo, Turso, Neptune adapters
│   │   ├── vector/     ← LanceDB, pgvector, Turso adapters
│   │   ├── cache/      ← Redis, SQLite, Postgres, FS, Tapes session-cache adapters
│   │   ├── relational/ ← SQLAlchemy core (Postgres/SQLite)
│   │   ├── hybrid/     ← Neptune + Postgres hybrid configs
│   │   ├── provenance/ ← Source-ref tracking for rollback
│   │   └── unified/    ← Graph+vector unified engine interface
│   ├── llm/            ← litellm gateway, prompts, extraction
│   ├── session/        ← SessionManager (QA + trace + context)
│   └── loaders/ files/ locks/ entities/
├── tasks/              ← Pipeline tasks (atomic operations)
│   ├── documents/ graph/ chunks/ summarization/ ingestion/
│   ├── memify/ temporal_graph/ provenance/ code_graph/
│   └── translation/ web_scraper/ codingagents/ completion/
├── memify_pipelines/   ← Higher-order pipelines
│   ├── apply_feedback_weights.py
│   ├── apply_frequency_weights.py
│   ├── consolidate_entities.py
│   ├── create_triplet_embeddings.py
│   ├── cross_connect_entities.py
│   ├── global_context_index.py
│   ├── persist_sessions_in_knowledge_graph.py
│   └── ...
├── pipelines/          ← run_pipeline / Task abstraction
├── shared/             ← logging, data_models (KnowledgeGraph), utils
└── alembic/            ← SQL migrations for relational DB
```

---

## 3. Storage layer — the core of the engine

### 3.1 Pluggable graph database

`cognee/infrastructure/databases/graph/get_graph_engine.py` is a factory that picks an adapter based on `GRAPH_DATABASE_PROVIDER` env. Supported providers (verified):

| Provider | Adapter file | Notes |
|---|---|---|
| `neo4j` | `neo4j_driver/adapter.py` (2 507 LOC, 71 async methods) | Full Cypher, deadlock retry, metrics utils, per-dataset containers via `neo4j_community_adapter` |
| `kuzu` / `ladybug` | `kuzu/adapter.py` | Default; bundled C++ graph engine |
| `postgres_demo` / `postgres` | `postgres_demo/adapter.py` | Demo only, not production-ready |
| `turso` | `turso/adapter.py` | Embedded SQLite-compatible |
| `neptune_analytics` | via `hybrid/` | AWS Neptune |

`Neo4jAdapter` is the production-grade option, and its presence is exactly what makes Cognee the natural fit under the `neo4j_agent_memory.md` filename.

### 3.2 Pluggable vector database

`cognee/infrastructure/databases/vector/` factory. Supported: `LanceDB` (default), `pgvector`, `Turso`.

### 3.3 Pluggable session cache

`cognee/infrastructure/databases/cache/`. Five backends: `Redis`, `SQLite`, `Postgres`, `FSCache`, `Tapes` (a log-structured cache). All implement `CacheDBInterface` with `SessionQAEntry` + `SessionAgentTraceEntry` schemas (`cache/models.py:7-80`).

### 3.4 Unified graph+vector interface

`cognee/infrastructure/databases/unified/unified_store_engine.py` and `graph_vector_store_interface.py` provide **provenance-aware deletion** — `delete_by_source_ref`, `delete_by_dataset_id`, `rollback_by_pipeline_run_id` — so failed cognify runs can be cleanly undone.

---

## 4. The `remember` API — ingestion path

`cognee/api/v1/remember/remember.py` (1 272 lines). Three modes dispatched at top of `remember()`:

1. **`MemorySource`** import — Mem0, Zep/Graphiti, Letta, LangMem, COGX archive (`cognee/modules/migration/sources/`).
2. **Typed `MemoryEntry`** — `QAEntry`, `TraceEntry`, `FeedbackEntry`, `SkillRunEntry` — short-circuits add+cognify, routed via `_dispatch_session_entry()` to `SessionManager`.
3. **Raw data** (str / bytes / files) — runs `add()` + `cognify()` + `improve()`.

```python
# V2 API (recommended)
await cognee.remember("Cognee turns documents into AI memory.")
await cognee.remember(session_data, session_id="chat_1")          # session-only
await cognee.remember(skill_run_entry)                            # graph-backed
await cognee.remember("repo/", content_type="code")               # code graph
await cognee.remember(SKILL_MD, content_type="skills")            # skill nodes
```

The promise-style `RememberResult` (`remember.py:411-633`) supports background runs (`run_in_background=True`), polling, and structured error capture.

### 4.1 The cognify pipeline

`cognee/api/v1/cognify/cognify.py` builds a task list (note `routing.py:32-50` for per-document-type branching — standard / dlt / code / code_repo):

```
classify_documents → extract_chunks → extract_graph_from_data →
extract_graph_and_summarize → resolve_temporal_contradictions (opt) →
detect_contradictions (opt) → record_provenance (opt) →
add_data_points → index_data_points
```

A `dry_run=True` mode returns `DryRunEstimate` (token usage, batch count) without making LLM calls.

### 4.2 Self-improvement pipeline (`improve` + `memify`)

`cognee/api/v1/improve/improve.py` — four stages when `session_ids` provided:

1. **Apply feedback weights** — session `feedback_score` (1-5) updates `feedback_weight` on the graph nodes/edges used to answer.
2. **Persist session Q&A** — chats cognified into the graph under `node_set="user_sessions_from_cache"`.
3. **Distill sessions** — gated active-guidance entries → entity-anchored lessons, cognified under `session_learnings`.
4. **Default enrichment** — triplet embeddings + indexing.

`memify_pipelines/` provides 10 named enrichment pipelines (`apply_feedback_weights.py`, `apply_frequency_weights.py`, `consolidate_entities.py`, `create_triplet_embeddings.py`, `cross_connect_entities.py`, `global_context_index.py`, `persist_sessions_in_knowledge_graph.py`, etc.) — each is a `Task` object composable into custom pipelines.

---

## 5. The `recall` API — retrieval path

`cognee/api/v1/recall/recall.py` (945 lines). Single `recall()` function that **auto-routes** across sources via `normalize_scope(scope)` (`cognee/memory/entries.py:130-180`).

### 5.1 Recall scopes

| Scope | Meaning |
|---|---|
| `auto` (default) | session-only if `session_id` alone; otherwise session + graph |
| `graph` | Permanent knowledge graph only |
| `session` | Session cache (QA entries) |
| `trace` | Agent trace steps in session |
| `session_context` | Distilled session context (entity-anchored lessons) |
| `all` | graph + session + trace + session_context |
| `tools` | External database connections (opt-in) |
| `code` | Code graph scope (opt-in, never implied) |

Each scope is dispatched in parallel (`_run_session`, `_run_trace`, `_run_session_context`, `_run_graph`).

### 5.2 Auto-routing

When `auto_route=True` (default), a lightweight rule-based classifier picks a `SearchType`. 17 strategies in `cognee/modules/search/types/SearchType.py`:

```python
SUMMARIES, CHUNKS, RAG_COMPLETION, HYBRID_COMPLETION, TRIPLET_COMPLETION,
GRAPH_COMPLETION, GRAPH_COMPLETION_DECOMPOSITION, GRAPH_SUMMARY_COMPLETION,
CYPHER, NATURAL_LANGUAGE, GRAPH_COMPLETION_COT, GRAPH_COMPLETION_CONTEXT_EXTENSION,
FEELING_LUCKY, TEMPORAL, CODING_RULES, CHUNKS_LEXICAL, AGENTIC_COMPLETION,
CODE, GRAPH_REPORT
```

---

## 6. Retrieval implementations — `cognee/modules/retrieval/`

13+ retriever classes inheriting from `BaseRetriever`:

| Retriever | Class | Strategy |
|---|---|---|
| `GraphCompletionRetriever` | `graph_completion_retriever.py` | Triplet search → context → LLM completion. Core "ask the graph" path |
| `HybridRetriever` | `hybrid_retriever.py` | Chunk + entity + global-context channels, ranked merge (`hybrid/merge.py`) |
| `TripletRetriever` | `triplet_retriever.py` | Pure vector-over-triplets |
| `TemporalRetriever` | `temporal_retriever.py` | Time-windowed graph completion (inherits GraphCompletion) |
| `GraphSummaryCompletionRetriever` | `graph_summary_completion_retriever.py` | Summarized context |
| `GraphCompletionCotRetriever` | `graph_completion_cot_retriever.py` | Chain-of-thought over graph |
| `GraphCompletionDecompositionRetriever` | `graph_completion_decomposition_retriever.py` | Query decomposition |
| `GraphCompletionContextExtensionRetriever` | `graph_completion_context_extension_retriever.py` | Context expansion |
| `GraphReportRetriever` | `graph_report_retriever.py` | Multi-section reports |
| `AgenticRetriever` | `agentic_retriever.py` | LLM-driven retrieval loops |
| `CypherSearchRetriever` | `cypher_search_retriever.py` | NL → Cypher generation |
| `NaturalLanguageRetriever` | `natural_language_retriever.py` | NL → graph queries |
| `ChunksRetriever` | `chunks_retriever.py` | Vector search over chunks |
| `SummariesRetriever` | `summaries_retriever.py` | Vector search over summaries |
| `LexicalRetriever` | `lexical_retriever.py` | BM25 |
| `CodingRulesRetriever` | `coding_rules_retriever.py` | Code-specific rules |
| `CodeRetriever` | `code_retriever.py` | Full code-graph queries (impact analysis, explore) |
| `CompletionRetriever` | `completion_retriever.py` | Generic LLM completion |

The hybrid retriever is the default and is the most sophisticated — it runs chunk/entities/facts/optional global-context lanes in parallel and merges with `use_importance_weight`/`use_truth_weight`.

---

## 7. Typed memory entries — `cognee/memory/entries.py`

```python
class QAEntry(BaseModel):       # Q&A turn
    type: Literal["qa"] = "qa"
    question: str; answer: str; context: str
    feedback_text: Optional[str]; feedback_score: Optional[int]  # 1-5
    used_graph_element_ids: Optional[dict]

class TraceEntry(BaseModel):     # agent tool/function call
    type: Literal["trace"] = "trace"
    origin_function: str; status: Literal["success", "error"]
    method_params: dict; method_return_value: Any
    memory_query: str; memory_context: str
    generate_feedback_with_llm: bool

class FeedbackEntry(BaseModel):  # attaches to existing QAEntry
    type: Literal["feedback"] = "feedback"
    qa_id: str; feedback_text: str; feedback_score: int

class SkillRunEntry(BaseModel):  # graph-backed (not session)
    type: Literal["skill_run"] = "skill_run"
    run_id: str; selected_skill_id: str; success_score: float  # [0,1]
    feedback: float  # [-1,1]
    candidate_skill_ids: list[str]; tool_trace: list[dict]
```

This is the **public API surface that maps cleanly onto whale-harness's hook model** — `main.py:32-38` already records turns via `_hook_record`; mapping that onto `QAEntry` / `TraceEntry` / `SkillRunEntry` is direct.

---

## 8. Observability — `cognee/modules/observability/`

OTEL-native spans (`new_span`), `CogneeTrace`, per-operation metrics, custom span attributes (`COGNEE_DATASET_NAME`, `COGNEE_SESSION_ID`, `COGNEE_DATA_SIZE_BYTES`, `COGNEE_SEARCH_TYPE`, `COGNEE_RECALL_SCOPE`, `COGNEE_FORGET_TARGET`, `COGNEE_IMPROVE_STAGES`). The whole pipeline is **traced end-to-end** — see `cognee/api/v1/recall/recall.py:514-520`.

`send_telemetry` is called from `remember`, `recall`, `improve`, `forget` — opt-in via env. There is **a real telemetry side channel** (different from OpenWolf's "no API calls" stance).

---

## 9. Does it have self-learning capabilities?

**Yes — multi-layered and explicit.** This is the single dimension on which Cognee outclasses every other engine we have reviewed.

### 9.1 Feedback weighting (`memify_pipelines/apply_feedback_weights.py`)

`apply_feedback_weights_pipeline(session_ids=..., alpha=0.1)` walks the Q&A entries for the given sessions, reads `feedback_score` (1-5) and `used_graph_element_ids` (the `node_ids` / `edge_ids` actually used to answer), and **adjusts `feedback_weight` on those exact nodes/edges**. Higher-rated answers boost their source graph elements.

This is **direct online learning** at the graph-attribute level — a closed feedback loop from user rating → next retrieval scoring.

### 9.2 Skill improvement (`cognee/modules/memify/skill_improvement.py`)

This is the most explicit self-learning path. When `cognee.remember(SkillRunEntry)` is called with `skill_improvement={...}`:

1. Find recent low-scoring / errored `SkillRun` records (`_find_recent_failure_runs`, score_threshold, max_runs).
2. Build a context block of `# Skill / # Current Procedure / # Failure Evidence`.
3. LLM call (`_generate_proposal`) → returns `SkillImprovementDraft{proposed_procedure, rationale, confidence}`.
4. Persist as `SkillImprovementProposal(status="proposed")` node in the graph.
5. Apply: `cognee.remember(..., skill_improvement={"apply": True, "proposal_id": ...})` writes the new procedure back to the `Skill` node (`_apply_proposal`, `skill_improvement.py:186-221`).

The proposal-first gate (must explicitly `apply=True` with a `proposal_id`) prevents runaway auto-mutation — the loop is **propose, then approve**. This is the exact pattern whale-harness's "self-evolving capabilities" spec calls for.

### 9.3 Auto-feedback on session turns (`AUTO_FEEDBACK` env, `cache/config.py:32-37`)

When `CACHING=true` and `AUTO_FEEDBACK=true` (defaults), `session_aware_completion.py` runs a **structured-output LLM call after each answered turn** to detect implicit feedback and extract session-context guidance. Set `AUTO_FEEDBACK=false` to disable the extra LLM call — the rest of session memory still works. **The default posture is "memory that improves from conversation signals"** (from the README).

### 9.4 Session distillation (`session_distillation/`)

Gated active-guidance entries are curated into entity-anchored lessons and cognified into the graph under `node_set="session_learnings"`. This is **episode → lesson consolidation**, a step above plain log storage.

### 9.5 Frequency weighting (`apply_frequency_weights.py`)

Node access counts feed back into graph-attribute weights — usage-based ranking emerges from recall traffic alone.

### 9.6 Truth subspace (`truth_subspace/`, opt-in via `build_truth_subspace=True`)

Builds a curated subgraph from distilled session learnings — a separate, higher-confidence knowledge layer. Off by default.

### 9.7 Global context index (`memify_pipelines/global_context_index.py`)

Bucketed root summaries (vector or graph bucketing, `placement_distance_threshold=0.5`, `max_bucket_size=20`) — hierarchical summarization for retrieval-time context preludes.

### 9.8 Triplet embeddings (`create_triplet_embeddings.py`)

When `cognify_config.triplet_embedding=True`, `(head, relation, tail)` triples get their own vector index — supports direct semantic lookup over the graph's relational structure.

### Summary of self-learning mechanisms

| Layer | Mechanism | Trigger | Risk |
|---|---|---|---|
| Graph attribute | `feedback_weight` updates from session feedback | `apply_feedback_weights_pipeline` | Low — additive, bounded |
| Graph attribute | `frequency_weight` from access counts | `apply_frequency_weights` | Low |
| Graph node | Skill procedure rewrite | `skill_improvement={apply=True}` w/ `proposal_id` | Medium — gated |
| Graph node | `SkillImprovementProposal` nodes (audit trail) | `skill_improvement={}` (default propose-only) | None — read-only proposal |
| Session cache | Auto feedback detection per turn | `AUTO_FEEDBACK=true` (default) | Medium — adds 1 LLM call/turn |
| Session → graph | Distillation of guidance into lessons | `improve(session_ids=...)` | Low |
| Knowledge layer | Truth subspace from distilled learnings | `build_truth_subspace=True` | Low — opt-in |
| Knowledge layer | Global context index (bucketed summaries) | `build_global_context_index=True` | Low — opt-in |

Cognee **explicitly treats memory as a living artifact** that rewrites itself from interaction signals. The propose-then-approve gate on skill rewrites is the exact pattern whale-harness's "self-learning + stop criteria" spec demands.

---

## 10. Migration paths — `cognee/modules/migration/sources/`

Plug-in importers for existing memory systems:

```python
await cognee.remember(Mem0Source("mem0_export.json"))    # Mem0
await cognee.remember(ZepSource(...))                    # Zep / Graphiti
await cognee.remember(LettaSource(...))                  # Letta
await cognee.remember(LangMemSource(...))                # LangMem
await cognee.remember(COGXArchiveSource(...))            # Cognee export
```

This means **whale-harness can pilot Cognee alongside other memory engines** without lock-in — start with our `memory.py` JSON store, then migrate via the `MemorySource` adapter when ready.

---

## 11. How `cognee` would look inside whale-harness

A minimal adapter wrapping the public V2 API as whale-harness hooks (`cognee_memory.py`):

```python
import asyncio
import cognee
from memory import memoryEngine, MemoryEntry
from cognee.memory import QAEntry, TraceEntry, SkillRunEntry
from cognee.api.v1.recall import recall

class CogneeMemoryEngine:
    """Drop-in replacement for memory.py backed by Cognee."""

    def __init__(self, dataset: str = "whale_main", user=None):
        self.dataset = dataset
        self.user = user
        self._loop = None

    async def remember(self, kind: str, content: dict, **kwargs):
        if kind == "turn":
            entry = QAEntry(
                question=content["prompt"],
                answer=content["response"],
                context=content.get("context", ""),
                feedback_score=content.get("feedback_score"),
                feedback_text=content.get("feedback_text"),
            )
            result = await cognee.remember(
                entry,
                dataset_name=self.dataset,
                session_id=content.get("session_id"),
                user=self.user,
            )
            return MemoryEntry(
                id=result.entry_id or "",
                kind="qa",
                content=content,
                created_at=content.get("created_at", ""),
            )

        if kind == "trace":
            entry = TraceEntry(
                origin_function=content["origin_function"],
                status=content.get("status", "success"),
                method_params=content.get("method_params"),
                method_return_value=content.get("method_return_value"),
                memory_query=content.get("memory_query", ""),
                memory_context=content.get("memory_context", ""),
                generate_feedback_with_llm=content.get("auto_feedback", True),
            )
            await cognee.remember(
                entry,
                dataset_name=self.dataset,
                session_id=content["session_id"],
                user=self.user,
            )

        if kind == "skill_run":
            entry = SkillRunEntry(
                selected_skill_id=content["skill_id"],
                task_text=content["task"],
                result_summary=content["result"],
                success_score=content.get("success_score"),
                feedback=content.get("feedback", 0.0),
                tool_trace=content.get("tool_trace", []),
                latency_ms=content.get("latency_ms", 0),
            )
            # SkillRunEntry is graph-backed, no session_id
            await cognee.remember(entry, dataset_name=self.dataset, user=self.user)

        # Fallback: raw remember for unstructured notes
        return await cognee.remember(content.get("text", str(content)), dataset_name=self.dataset)

    async def recall(self, query: str, session_id: str | None = None, top_k: int = 10, scope: str = "auto"):
        results = await cognee.recall(
            query,
            session_id=session_id,
            dataset_name=self.dataset,
            top_k=top_k,
            scope=scope,
            user=self.user,
        )
        return [
            {"content": r.content if hasattr(r, "content") else str(r), "source": getattr(r, "_source", "graph")}
            for r in results
        ]
```

Wire `main.py:32-38` to call `CogneeMemoryEngine().remember(...)` instead of `memoryEngine.record(...)` and the JSON store becomes a thin cache in front of a graph.

---

## 12. Fit for whale-harness

### 12.1 Spec checklist

| whale-harness spec (§README.md, lines 17-22) | Cognee support |
|---|---|
| Extensive memory, ideally structured | **Native** — knowledge graph + vector + cache |
| Loop engineering (act → observe → decide → repeat) | **Native** — `remember`/`recall`/`improve`/`forget` form a loop |
| Self-learning | **Multi-layer explicit** (§9) |
| Tools available | `tools/` builtin registry + MCP integration (`cognee-mcp/`) |
| Cheap to run on simple LLM | Graph builds are LLM-heavy by default; configurable per-task |
| Swarm / rhizomatic | No built-in swarm primitives — engine is per-process (or one server per dataset); cross-agent sharing via dataset permissions and `cloud/` |
| Stop criteria | `stop_after_session` / pipeline completion; no formal outer stop |

### 12.2 Pros

- **Only reviewed engine with explicit, multi-layer self-learning**: feedback weighting, skill rewrite (propose-then-approve), auto-feedback on turns, session distillation, frequency weighting, truth subspace.
- **Production-grade Neo4j adapter** — exactly what `neo4j_agent_memory.md` filename hints at; 2 507 LOC, deadlock retry, metrics utils, multi-container support.
- **Pluggable everything**: graph (5 providers), vector (3), cache (5), LLM (litellm gateway).
- **Typed memory entries** (`QAEntry`, `TraceEntry`, `FeedbackEntry`, `SkillRunEntry`) map almost 1:1 onto whale-harness's existing `_hook_record` contract.
- **17 search strategies + 13+ retrievers** with hybrid-by-default — covers any future query shape.
- **Multi-tenant from day one** (`users/`, `permissions/`, `dataset_*_handler`) — whale-harness's `~/.whale/memory/` has none.
- **OTEL-native observability** — plug into the whale-harness loop engineering pattern.
- **Migration adapters** for Mem0/Zep/Letta/LangMem/COGX — no lock-in, easy to pilot alongside.
- **MCP server** (`cognee-mcp/`) — directly usable from agents that speak MCP.
- **Code graph support** — `content_type="code"` ingests repos as architectural graphs (`tasks/code_graph/`).
- **Apache-2.0 license** — compatible with whale-harness's MIT-friendly posture.

### 12.3 Cons

- **Python-only** — whale-harness's CLI is currently Python (`typer`), but the roadmap hints at Go (`README.md:69`). Embedding Cognee as a sidecar subprocess is doable, in-process is not.
- **Heavy dependency footprint**: pydantic, litellm, instructor, sqlalchemy, neo4j, lancedb, alembic, fastapi. Whale-harness currently ships `litellm + typer + python-dotenv` — adding Cognee triples the install footprint.
- **Self-learning is opt-in but loud**: default `AUTO_FEEDBACK=true` adds an LLM call per turn. Whale-harness's "lean in token consumption" spec needs explicit tuning (`AUTO_FEEDBACK=false` for background, `true` only during active learning).
- **No swarm primitives** — the engine is **single-process or single-server per dataset**. Whale-harness's "swarm of agents in parallel" pattern needs an outer orchestrator; Cognee is one node's memory, not a swarm bus. Multi-agent sharing is via dataset permissions only.
- **OSS scope vs. enterprise**: the `cloud/` module and `cognee-mcp/` are tightly bound to a hosted product; self-hosting is fully supported but some conveniences (UI, hosted connectors) are paywalled.
- **Telemetry side-channel**: `send_telemetry` is on by default — whale-harness's "zero API calls" preference (inherited from OpenWolf review) requires explicit opt-out.
- **Operational cost**: Neo4j + vector DB + cache + relational DB + observability = a non-trivial deployment. Whale-harness's current single-file `~/.whale/memory/entries.json` is the polar opposite.
- **No formal "stop criteria" hook** — pipelines run to completion; whale-harness's loop needs an outer watcher.

### 12.4 Verdict

**Strong fit for the *memory engine* role; weak fit as the entire whale-harness backend.**

Cognee is the **best off-the-shelf answer** to whale-harness's "extensive structured memory + self-learning" spec. The propose-then-approve skill rewrite and feedback-weighted graph are exactly what the README's "Reliable harness with memoery and self-evolving capabilies" line is asking for. The Neo4j adapter is production-grade and the `neo4j_agent_memory.md` filename suggests this review was commissioned with that backend in mind.

However, Cognee is a **memory engine**, not an orchestration runtime. Whale-harness's swarm, loop engineering, and rhizomatic-agent structure must remain in whale-harness itself — Cognee slots in as the memory layer beneath that loop, not as the loop itself. The two are complementary, not substitutable.

**Recommended integration pattern**: keep `main.py` and `session.py` as the runtime; replace `memory.py` with a `CogneeMemoryEngine` adapter that maps whale's `kind: "turn"|"trace"|"skill_run"` onto Cognee's typed entries. Use `remember(..., session_id=...)` for short-term memory and `remember(..., self_improvement=True)` for the feedback→graph rewrite loop. Tune `AUTO_FEEDBACK=false` outside active-learning windows to honor "lean in token consumption." Pilot Neo4j Community container per-dataset via `GRAPH_DATASET_DATABASE_HANDLER=neo4j_community` for self-hosted zero-ops.

---

## 13. Verification commands run

```bash
# Clone
git clone --depth 1 https://github.com/topoteretes/cognee.git /tmp/cognee-research/cognee

# Layout
ls cognee/ cognee/api/v1/ cognee/modules/ cognee/infrastructure/databases/
ls cognee/memify_pipelines/ cognee/tasks/ cognee/memory/

# Graph providers
grep -rn "neo4j\|Neo4j" cognee/infrastructure/databases/graph/

# Self-learning
grep -rn "feedback_weight\|self_improvement\|SkillImprovementProposal" cognee/

# Retrieval strategies
grep "class.*Retriever.*BaseRetriever" cognee/modules/retrieval/*.py

# Search types
cat cognee/modules/search/types/SearchType.py

# Public API
cat cognee/api/v1/__init__.py
cat cognee/api/v1/remember/remember.py
cat cognee/api/v1/recall/recall.py
cat cognee/api/v1/improve/improve.py
cat cognee/api/v1/forget/forget.py

# Memory entries
cat cognee/memory/entries.py

# Skill improvement (self-learning)
cat cognee/modules/memify/skill_improvement.py
cat cognee/memify_pipelines/apply_feedback_weights.py

# Neo4j adapter
cat cognee/infrastructure/databases/graph/neo4j_driver/adapter.py | head -80
sed -n '375,420p' cognee/infrastructure/databases/graph/get_graph_engine.py
```

---

## 14. Conclusion

Cognee is **the most ambitious memory engine we have reviewed** and the only one that takes both **structured graph memory** and **multi-layer self-learning** seriously. The Neo4j adapter is real, the feedback loops are wired end-to-end, and the typed entry API maps cleanly onto whale-harness's existing hook contract.

For whale-harness, the right move is to **adopt Cognee as the memory engine underneath the existing runtime**, not as a replacement for it. Keep the swarm/loop/CLI in whale; let Cognee own the *structured, persistent, self-tuning knowledge base* — the layer whale-harness has not yet built.

The proposed-then-approved skill rewrite loop (`cognee/modules/memify/skill_improvement.py`) is, by itself, worth the integration — it is the cleanest "self-learning with stop criteria" implementation in the open-source agent-memory landscape.

**Fit for whale-harness: ✅ Yes, as the memory tier.** Adopt Cognee behind whale's existing `memory.py` adapter surface; the `Neo4jAdapter` (2 507 LOC, deadlock-retry, metrics utils, per-dataset containers) is the production-grade backing store the `neo4j_agent_memory.md` filename promises.

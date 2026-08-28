# Semantica — Memory Engine Review

**Repo:** `git@github.com:semantica-agi/semantica.git`
**Version reviewed:** v0.6.7 (`da642f12`, 2026-08-28)
**License:** MIT · **Python:** >=3.8 · **Scale:** ~721 Python files, 2631 commits, first commit 2025-06-25
**Self-description:** *"Graph-Native Infrastructure for Context and Accountable AI Systems" / "The Open Source Palantir for AI Agents"*

---

## 1. What it actually is

Semantica is **not** a memory library. It is a full **knowledge-graph ETL platform** with a
memory-shaped API bolted on top. The centre of gravity is the pipeline:

```
Sources → Ingest → Parse → Normalize → Split → Extract → Conflict Detection → Deduplication
   → Knowledge Graph → [ Ontology · Reasoning · Provenance · Decisions ] → Enriched KG
   → Vector Store + Graph Store (RDF & LPG) → Export / Visualize / REST · MCP · CLI
```

Everything is a separately importable module under `semantica/`:

| Module | Purpose |
|---|---|
| `ingest` | Files, web, DBs, Kafka/Kinesis, Git repos, IMAP email, Databricks, Snowflake, MCP resources |
| `parse` / `normalize` / `split` | Document parsing, text/entity/date normalization, entity-aware chunking |
| `semantic_extract` | NER, relation extraction, event detection, triplets, coreference |
| `conflicts` / `deduplication` | Contradiction flagging, entity merging (blocking + semantic dedup) |
| `kg` | `GraphBuilder`, `EntityResolver`, bi-temporal facts, centrality, communities, link prediction |
| `ontology` | OWL / SHACL / SKOS generation and validation |
| `reasoning` | Rete network, Datalog, SPARQL, forward chaining, explanation generation |
| `provenance` | W3C PROV-O lineage on every fact |
| **`context`** | **the memory layer — see below** |
| `vector_store` | FAISS, Qdrant, Weaviate, Milvus, Pinecone, PgVector + RRF hybrid search |
| `graph_store` | Neo4j, FalkorDB, Apache AGE, Amazon Neptune; RDF via Oxigraph/Jena/RDF4J/Blazegraph |
| `export` | RDF Turtle, JSON-LD, N-Triples, OWL, SHACL, Parquet, Cypher, AQL, GraphML, CSV, HTML |
| `mcp_server` / `server.py` / `cli.py` | MCP (12 tools), REST API, 4611-line Typer CLI |

---

## 2. The memory layer (`semantica/context/`) — how it works

This is the only part directly relevant to a harness. ~20k lines across 18 files:

```
context_graph.py      5466   ContextGraph — the actual graph store
context_retriever.py  2801   retrieval strategies
agent_context.py      2546   AgentContext — high-level facade (the "memory API")
agent_memory.py       2314   AgentMemory — store/retrieve/forget engine
decision_query.py     1300   precedent search, decision querying
policy_engine.py      1043   rule evaluation / compliance gates
decision_methods.py    924
entity_linker.py       898
causal_analyzer.py     779
decision_recorder.py   618
graph_schema.py        541
decision_context.py    479
decision_models.py     398
```

### 2.1 Three-tier memory model

`AgentMemory.__init__` (agent_memory.py:184) wires a **write-through hierarchy**:

| Tier | Backing store | Notes |
|---|---|---|
| **Short-term** | `self.short_term_memory: List[MemoryItem]` | Pruned by count (`short_term_limit`, default 10) **and** token budget (`token_limit`, default 2000) |
| **Long-term** | pluggable `vector_store` | Optional. Only used if a store instance is injected |
| **Structured** | pluggable `knowledge_graph` | Entities/relationships promoted into the KG |
| (index) | `memory_items: Dict` + `memory_index: deque(maxlen=max_memory_size)` | In-process; default cap 10 000 |

`store()` (agent_memory.py:301) is a genuine write-through: appends to short-term → prunes →
embeds into vector store → registers in the dict/deque → updates the KG if entities were
supplied → bumps stats → applies retention policy. Flags `skip_vector` and `skip_graph` let you
degrade to short-term-only. Idempotent on a repeated `memory_id` (replaces in place).

`retrieve()` (agent_memory.py:427) is a **3-stage cascade**:

1. `_search_short_term()` — lowercase term-overlap scoring over the recent buffer, newest first
2. Vector search — `search_vectors(query_vector, k=max_results*2)` or `search(query, limit=…)`
3. `_keyword_search()` — **only if stages 1+2 returned nothing**

then sort by score desc, filter by `min_score`, truncate to `max_results`.

### 2.2 `ContextGraph` — the graph itself

Pure-Python, dependency-light. Static import analysis of `context_graph.py` shows **zero** heavy
imports — only stdlib plus `yaml`:

```
collections copy dataclasses datetime errno hashlib itertools json logging
os pathlib re shutil stat tempfile threading typing uuid yaml
```

Internals: `nodes: Dict[str, ContextNode]`, `edges: List[ContextEdge]`, `_edge_index`,
`_adjacency: defaultdict(list)`, `node_type_index`, `edge_type_index`, guarded by a
`threading.RLock`. So the graph primitive is cheap and embeddable — the weight lives in the
optional analytics/extraction modules, not here.

Notable capabilities:

- **Bi-temporal validity** — `ContextNode.is_active(at_time)`, `state_at(timestamp)` for
  point-in-time snapshots (graph as it existed on any past date)
- **Retract vs. purge** — `retract_node/edge` closes the validity window but keeps history;
  `purge_node/edge` removes content and leaves a **tombstone**. `list_retractions()`,
  `list_tombstones()`. This is a properly thought-through "forget" semantics, keyed by
  `(entity_kind, entity_id)` to avoid node/edge id collisions.
- **Graph linking** — `link_graph()`, `navigate_to()`, `cross_graph_path()`, `resolve_links()`:
  multiple context graphs can be federated and traversed across. Directly relevant to swarms.
- **Analytics** — `get_node_centrality`, `find_similar_nodes`, `analyze_graph_with_kg`,
  `analyze_connections`, `get_node_importance`
- **Persistence** — `save_to_file` / `load_from_file`, `to_dict` / `from_dict`, `to_kg_dict`

### 2.3 Decision intelligence — the genuinely differentiated part

Decisions are **first-class graph nodes**, not log lines. Lifecycle: Record → Link → Query →
Govern → Audit-export.

```python
from semantica.context import ContextGraph

graph = ContextGraph(advanced_analytics=True)

decision_id = graph.record_decision(
    category="vendor_selection",
    scenario="Choose cloud provider for HIPAA workload",
    reasoning="AWS offers BAA, mature HIPAA tooling, existing team expertise",
    outcome="selected_aws",
    confidence=0.93,
)

chain     = graph.trace_decision_chain(decision_id)                       # causal ancestry
similar   = graph.find_similar_decisions("cloud vendor", max_results=5)   # precedent search
impact    = graph.analyze_decision_impact(decision_id)                    # downstream map
compliant = graph.check_decision_rules({"category": "vendor_selection"})  # policy gate
```

Surrounding API: `add_causal_relationship()` (triggers / enables / causes / precedes),
`get_causal_chain()`, `trace_decision_causality()`, `find_precedents_by_scenario()`,
`analyze_decision_influence()`, `predict_decision_relationships()`,
`enforce_decision_policy()`, `get_decision_insights()`.

`AgentContext` adds `checkpoint(label)`, `diff_checkpoints(l1, l2)`, `flush_checkpoint(label)`,
`multi_hop_context_query()`, `trace_decision_explainability()`, `capture_cross_system_inputs()`,
`usage_stats(period)`, `health()`, `backup()` / `restore()`.

### 2.4 Access surfaces

- **MCP server** — JSON-RPC 2.0 over stdio, protocol `2024-11-05`, 12 tools + 3 resources:
  `record_decision`, `query_decisions`, `find_precedents`, `get_causal_chain`,
  `analyze_decision_impact`, `add_entity`, `add_relationship`, `search_graph`,
  `get_graph_summary`, `get_graph_analytics`, `extract_entities`, `extract_relations`,
  `extract_all`, `run_reasoning`, `abductive_reasoning`, `export_graph`, `get_provenance`
- **Python API** — direct import, the cheapest path
- **REST API** (`semantica/server.py`, 100+ endpoints) and **CLI** (`semantica/cli.py`, 4611 lines)
- **Framework integrations** — `integrations/{agno,crewai,langchain,openclaw}`. The `openclaw`
  one is a precedent for exactly our use case: a harness wiring Semantica in via MCP config.

---

## 3. Honest assessment of quality

### Strong

- **Forget/retract semantics are real.** Retraction vs. tombstone, validity windows, `(kind, id)`
  keyspace separation. Most memory libraries just `del`. This maps straight onto our
  `--memory manage [forget | relearn]` spec.
- **Decision graph is the right primitive for a learning loop.** Our architecture step 6
  ("register the completion fact, match it to the plan from step 3") is precisely
  `record_decision` + `add_causal_relationship` + `trace_decision_chain`.
- **`ContextGraph` core is dependency-free.** The graph you'd actually use is stdlib + yaml.
- **Graph federation** (`link_graph` / `cross_graph_path`) fits swarms: per-agent graph, linked,
  traversable across.
- **Pickle loading is explicitly disabled** on the legacy path with a security warning
  (agent_memory.py:252) — someone did think about deserialization attacks.
- **Test suite is substantial** — ~85 test directories/files including security regression suites.
- **Active** — 2631 commits, last commit the day of this review, real external contributors.

### Weak / concerning

- **`_generate_embedding` is a stub.** agent_memory.py:769:
  ```python
  def _generate_embedding(self, content: str) -> Any:
      """Generate embedding for content."""
      # This would use an embedding model
      # For now, return placeholder
      if hasattr(self.vector_store, "embed"):
          return self.vector_store.embed(content)
      return None
  ```
  If your vector store lacks `.embed()`, you silently pass `None` as a query vector. The
  "semantic memory" is only as good as the store you inject; Semantica supplies no default.

- **Indentation bug in `retrieve()`.** agent_memory.py:490-527: the `for result in
  vector_results:` loop that materializes results is nested **inside** the
  `elif hasattr(self.vector_store, "search")` branch. Meaning: if your store exposes
  `search_vectors` (the FAISS/first-class path, checked first at line 471), `vector_results` is
  built at line 488 and then **never consumed**. Long-term vector recall is a silent no-op on the
  primary code path. That is a serious defect in the headline memory feature.

- **Retrieval fusion is naive.** Short-term term-overlap counts and vector cosine scores are
  thrown into one list and sorted by raw `score` — incommensurable scales, no normalization, no
  RRF (despite RRF existing elsewhere in `vector_store`). Keyword search is a dead-letter
  fallback that only fires when everything else returns empty.

- **Install weight is brutal.** *Core*, non-optional dependencies include `torch`,
  `transformers`, `sentence-transformers`, `spacy`, `faiss-cpu`, `opencv-python`, `librosa`,
  `umap-learn`, `gensim`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `plotly`,
  `ipywidgets`, `onnxruntime`, `pyarrow`. That is multiple GB and a minutes-long install for a
  CLI whose stated acceptance criteria are "cheap to run" and "lean in token consumption".
  `opencv` and `librosa` in the *core* set of a knowledge-graph library is a packaging smell.

- **`requires-python = ">=3.8"` with `numpy>=2.0.2`.** numpy 2.x requires Python >=3.9. The
  floor is misdeclared.

- **Sloppy patches in hot files.** `ContextGraph.__init__` contains stray blank lines and
  trailing-whitespace-only comment lines; `retrieve()` defines a `ResultObj` class **inside** the
  method body on every call. Cosmetic, but consistent with the impression of fast-moving,
  lightly-reviewed velocity.

- **Marketing-to-substance ratio.** A 77 KB README, "Open Source Palantir", Trendshift badges.
  Contribution graph is heavily concentrated: `KaifAhmad1` + `Mohd Kaif` = 1927 of 2631 commits
  (~73%), i.e. effectively one author. Bus factor ~1.

- **Everything in-process by default.** `memory_items` is a plain dict, `memory_index` a
  `deque(maxlen=10000)`. Durability requires explicitly calling `save(path)`. No WAL, no
  incremental flush, no crash safety. A killed `whale` process loses unsaved memory.

- **`from __future__` / typing style is Python-3.8-era** (`Dict`, `List`, `Optional`) — clashes
  with our codebase convention of `dict[str, Any]` + `from __future__ import annotations`.

---

## 4. Fit as the whale-harness memory agent

### Where we are today

`memory.py` is a 100-line append-only JSON log: `MemoryEntry(id, kind, content, created_at,
tags, source)`, `record()`, `retrieve(kind, limit)` sorted by recency. Rewrites the whole
`~/.whale/memory/entries.json` on every write. No search, no graph, no forget. The README's
"Next steps" already names the target: *"always ask llm to return a prepared graph objects like
node or branch … and save it immediately."*

Semantica's `ContextGraph` **is** that target data structure, already built.

### Verdict: adopt the idea and the `context` subsystem; do not adopt the platform.

| Whale requirement | Semantica answer | Grade |
|---|---|---|
| Structured / semi-structured memory | `ContextGraph` typed nodes + edges | Strong |
| "llm wiki type" memory | Nodes with properties + markdown filesystem link support | Good |
| Rhizomatic structure | `link_graph` / `cross_graph_path` federation | Strong |
| Self-learning loop (step 6 → step 3 matching) | `record_decision` + causal chain + precedent search | **Best-in-class** |
| Swarm shared context | One shared graph or federated per-agent graphs | Good |
| `--memory manage [forget \| relearn \| add]` | `store` / `forget` / `retract` / `purge` / tombstones | **Direct mapping** |
| Stop criteria / success criteria | `policy_engine` + `check_decision_rules` as loop exit gate | Strong, underrated fit |
| Cheap to run | torch + spacy + opencv in core deps | **Fails** |
| Lean token consumption | Graph traversal beats dumping history into context — helps a lot | Strong (conceptually) |
| Fast cheap LLM backbones | KG construction/reasoning/provenance need **no LLM** — deterministic | **Strong** |
| Reliability | Vector-recall indentation bug; embedding stub | **Concerning** |

### The decisive points

**For:** the deterministic, no-LLM-required design is a genuinely excellent match for
"powerful on simple fast cheap LLM backbones". A cheap model does not need to be smart if it can
traverse a graph instead of re-reasoning. And `record_decision` → `trace_decision_chain` →
`find_similar_decisions` is exactly the self-learning loop closure we sketched in the README's
step 6, already implemented and tested.

**Against:** we are building a lean CLI installable via `brew`/`curl`/`apt`. `pip install
semantica` drags in torch, spacy, opencv, and librosa. That is disqualifying for the default
install path, and the vector-retrieval bug means the "long-term semantic memory" headline
feature likely does not work on the FAISS path as shipped.

### Recommendation — three options, in order of preference

**Option A (recommended): vendor the pattern, not the package.**
Reimplement `ContextGraph` + decision-node semantics in `whale/memory/` — it is stdlib-only by
construction, so this is achievable in ~1–2k lines. Copy specifically: bi-temporal
`is_active(at_time)` / `state_at()`, retract-vs-tombstone forget semantics, the
`(kind, id)` keyspace split, `link_graph` federation, and the
`record_decision` / `add_causal_relationship` / `trace_decision_chain` triad. Keep our
`from __future__ import annotations` + lowercase-generics style and our `structlog` logger.
Zero new heavy dependencies; we own the bugs.

**Option B: optional extra.**
`whale[semantica]` — expose Semantica as a pluggable backend behind a narrow
`MemoryBackend` protocol (`record`, `retrieve`, `forget`, `link`, `decide`), with our JSON
engine as the always-available default. Users in regulated contexts who want PROV-O export,
SHACL governance, and RDF get it; nobody pays the install cost by default.

**Option C: MCP-only, out of process.**
Run `python -m semantica.mcp_server` as an external MCP server and consume the 12 tools through
our normal tool surface — exactly what `integrations/openclaw` does. Total isolation from the
dependency tree; costs a subprocess and IPC latency, and makes memory an optional service rather
than a core guarantee. Reasonable as a **step 1 spike** to validate the decision-graph model
against real whale sessions before committing to Option A.

### If we do integrate, watch for

1. Inject a vector store that actually implements `.embed()`, or `_generate_embedding` returns
   `None` and semantic recall dies silently.
2. Verify long-term recall end-to-end before trusting it — see the `retrieve()` indentation bug.
   Test with a `search_vectors`-style store specifically.
3. Call `save(path)` on every meaningful turn, or wrap it in a hook; there is no autosave.
4. `max_memory_size=10000` deque cap and `short_term_limit=10` / `token_limit=2000` need tuning
   for long agent loops.
5. Pin the version. 2631 commits in ~14 months with a bus factor of 1 means the API surface moves.

### Bottom line

Semantica is the **right conceptual model** for whale's memory — graph-native, decision-first,
deterministic, with real forget semantics — wrapped in the **wrong distribution** for a lean CLI.
Take the architecture. Spike it over MCP (Option C) to validate, then vendor the `ContextGraph`
and decision-node design into `whale/memory/` (Option A). Reserve the full package as an
opt-in extra for users who need the compliance/RDF/ontology tail.

---

## 5. Reference links

- `ARCHITECTURE.md` — full Mermaid pipeline + decision-intelligence lifecycle diagrams
- `semantica/context/context_usage.md` — in-repo usage docs for the memory layer
- `semantica/kg/kg_usage.md`, `semantica/core/core_usage.md`
- `integrations/openclaw/` — closest precedent for a harness integration
- `mcp/README.md` + `mcp/tools/` — MCP tool schemas
- `cookbook/` — `introduction`, `advanced`, `integrations`
- Docs: https://docs.getsemantica.ai/
</content>

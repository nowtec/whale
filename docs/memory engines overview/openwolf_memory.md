# OpenWolf — Memory Engine Review

**Repo:** `git@github.com:cytostack/openwolf.git`
**Version reviewed:** v2.5.0 (`package.json`, dated 2026-08-22 per CHANGELOG)
**License:** AGPL-3.0 · **Runtime:** Node.js ≥ 20 · **Language:** TypeScript (ESM)
**Size:** ~14k lines of source across ~70 modules; 24 vitest-style unit suites
**Self-description:** *"One project memory across Claude Code, Codex and OpenCode. Token usage measured, not estimated. Zero API calls."*

> OpenWolf is **not** a knowledge-graph memory engine (like Semantica, Neo4j, Cogee).
> It is **agent-side persistent project memory + context-engineering middleware**
> for coding-agent CLIs. The closest mental model is "second brain for an AI coding
> harness", implemented entirely as plain Node.js file I/O — no vector DB, no LLM
> calls, no telemetry.

---

## 1. What it actually is

OpenWolf sits between you and a coding agent (Claude Code, Codex CLI, OpenCode,
and to a lesser degree Gemini / Cursor / Antigravity). It installs **lifecycle
hooks** into the agent's own hook system, then maintains a `.wolf/` directory of
markdown/JSON state that survives sessions, agents, and machines.

Three-layer architecture (per `docs/how-it-works.md`):

```
┌─────────────────────────────────────────────────────────────┐
│  Coding Agent (Claude Code / Codex / OpenCode / …)          │
│      ↓ invokes on lifecycle events                         │
│  Hook layer (.wolf/hooks/*.js — 12 hooks)                   │
│      ↓ reads / writes                                       │
│  State layer (.wolf/ — markdown + JSON, git-tracked state)  │
│      ↑ runs                                               │
│  Optional daemon (Express + WS, scheduler, dashboard)       │
└─────────────────────────────────────────────────────────────┘
```

The tagline that matters: **"Pure local file I/O: no API calls, no telemetry, no
added latency."** This was hard-enforced in v2.5 — the legacy `ai_task` cron
action that shelled out to `claude -p` was removed, and a stale manifest entry
of that type now throws an error rather than silently reaching the network
(`src/daemon/cron-engine.ts:261`).

---

## 2. The `.wolf/` state directory

This is OpenWolf's "memory". All files are plain markdown or JSON, mostly
git-tracked, partly machine-local (the `.gitignore` enforces the split).

| File | Type | Purpose | Budget |
|------|------|---------|--------|
| `cerebrum.md` | markdown | **Learned preferences, conventions, Do-Not-Repeat list, Decision Log** | 2 000 tok |
| `STATUS.md` | markdown | Session handoff — "read this FIRST when resuming" | 1 000 tok |
| `memory.md` | markdown | Chronological action log, one block per session | unbounded, consolidated weekly |
| `anatomy-index.json` + `anatomy.md` | json+md | Durable project index: descriptions, sizes, symbols, import graph, **personalized PageRank importance** | unbounded |
| `buglog.json` | json | **Bug + fix memory** with auto-detection and SQLite FTS5 retrieval | unbounded |
| `token-ledger.json` | json | **Measured + verified** per-session token usage (real transcripts) | unbounded |
| `cron-manifest.json` / `cron-state.json` | json | Scheduled tasks (scan / consolidate / audit) + execution log + dead-letter queue | n/a |
| `config.json` | json | Tunables: governor, budgets, cadences, ports | n/a |
| `hooks/_heartbeat.json` | json | Per-hook last_ok / last_error / consecutive_failures | n/a |
| `hooks/sessions/<id>.json` | json | Per-session read/write tracking, anatomy hits, pending reminders | n/a |
| `cache/bash/<id>.log` | verbatim | Full stdout of every governed Bash call (full output always preserved) | 200 files / 50 MB |
| `cache/buglog.db` | SQLite | FTS5 index over buglog (derived, disposable) | n/a |
| `OPENWOLF.md` | markdown | Operating protocol for agents without a native skill surface | n/a |
| `identity.md` | markdown | Persona block (default: "Wolf") | n/a |

The **committed-vs-machine-local split** is a first-class concern:

```
committed by .wolf/.gitignore      ignored (machine-local)
─────────────────────────           ─────────────────────────────
cerebrum.md  STATUS.md              token-ledger.json
memory.md    buglog.json            cache/
anatomy-index.json                  hooks/_heartbeat.json
cron-manifest.json                  hooks/sessions/
OPENWOLF.md   identity.md           hooks/<compiled>.js
config.json
```

This means **conventions, handoff, and bug fixes travel through git and reach
every teammate and every agent** — but ledgers and runtime caches stay local.

---

## 3. The hook layer — 12 lifecycle hooks

Registered through the agent's own hook system (`src/hooks/`, run by `src/cli/init.ts:141-153`).
For Claude Code: `.claude/settings.json`. For Codex: `.codex/hooks.json`. For OpenCode: a
multi-file native plugin under `.opencode/plugin/openwolf/`.

| Hook event | Script | What it does |
|------------|--------|--------------|
| `SessionStart` | `session-start.js` | Builds a **budget-capped digest** (~400 tok) of `.wolf/` state; injects it via `additionalContext`. On resume/compact: restores scoped rules + in-flight state. |
| `UserPromptSubmit` | `user-prompt-submit.js` | Drains `pending_reminders` queued by `Stop` so they ride along the next prompt without burning an extra turn. |
| `PreToolUse` (Read) | `pre-read.js` | Surfaces anatomy description + symbol map; warns on duplicate full reads. |
| `PreToolUse` (Write/Edit) | `pre-write.js` | Greps `cerebrum.md` Do-Not-Repeat against the edit content; FTS-searches `buglog.json` for relevant past fixes; surfaces as a **factual advisory** (not a directive). |
| `PreToolUse` (Bash) | `pre-bash.js` | Suggests narrower variants for flood-prone commands. |
| `PostToolUse` (Read) | `post-read.js` | Records real read sizes in the session file. |
| `PostToolUse` (Write) | `post-write.js` | Updates anatomy store under a cross-process lock; appends to `memory.md`; auto-detects bug-fix patterns and **appends them to `buglog.json` automatically**; emits budget warning if `.wolf/cerebrum.md` or `STATUS.md` grow past their budgets. |
| `PostToolUse` (Bash) | `post-bash.js` | The **Bash output governor** (flagship 2.3 feature). See §5. |
| `PostToolBatch` | `post-batch.js` | Every N tool batches (default 25), re-surfaces top Do-Not-Repeat rules. The "instruction-decay countermeasure" — counter to the empirical 5.6% per-function compliance decay measured in arXiv 2605.10039 (1 650 sessions). |
| `PreCompact` | `precompact.js` | Snapshots session JSON for post-mortem; `SessionStart` with `source: "compact"` then restores scoped rules + in-flight files + top rules. |
| `Stop` | `stop.js` | Flushes ledger with **measured** transcript usage; queues 3 end-of-turn reminders (buglog, cerebrum freshness, semantic summary) at most once each per session. |
| `SessionEnd` | `session-end.js` | Final flush + session summary. |

Every hook runs through `hookMain()` (`src/hooks/shared.ts:104`) which:
- Runs the body, then writes a heartbeat (`last_ok`) regardless of outcome.
- Never `exit(1)`s — a broken hook never blocks the agent.
- Supports `--selfcheck`: exits with `ok <hookname>` proving all static
  imports resolved, which is the failure class that once silently killed a
  hook for **440 invocations over 3 weeks** (CHANGELOG 2.2).

---

## 4. How learning actually works — "cerebrum"

This is the closest thing OpenWolf has to **self-learning**. It is *not* online
gradient descent or vector embedding — it's structured markdown bookkeeping
that the agent maintains while the user/hooks nudge it.

### 4.1 What is in cerebrum.md

```
## User Preferences
## Key Learnings
## Do-Not-Repeat        ← auto-mirrored into Claude Code auto-memory (v2.5)
## Decision Log
```

- **Do-Not-Repeat** entries are dated `[YYYY-MM-DD]` mistakes and the fix.
  `pre-write.js` greps this section against the next edit using regex extraction
  of quoted tokens / "never use X" patterns. If a match hits, an advisory is
  injected before the write fires. `post-batch.js` re-surfaces the top-K rules
  every 25 batches (decay countermeasure, §3).
- **Budget enforcement** (`src/hooks/post-write.ts:75-95`): writing
  `.wolf/cerebrum.md` or `STATUS.md` past its token budget (defaults 2k/1k)
  triggers **one factual warning per session** — the same
  measure-after-write loop native Claude auto-memory uses.

### 4.2 What triggers a write

There is **no automatic reflection job**. In v2.4 the `cerebrum-reflection`
cron action existed and ran weekly via `claude -p` — that **was** removed in
v2.5 as the one violation of the "no API calls" claim. Today the only writers
are:

1. **The user / agent together** — `OPENWOLF.md` instructs the agent:
   *"Update cerebrum.md whenever you learn something: a user correction or
   preference, a project convention not obvious from code, an API surprise…
   The bar is LOW; a redundant entry costs nothing, a missing one repeats the
   discovery next session."*
2. **Hook nudges**: `session-start.js` emits a startup note if cerebrum has <3
   entries or hasn't been touched in 3 days (`src/hooks/session-start.ts:255-265`).
3. **`Stop` reminder**: if `files_written >= 3` and cerebrum mtime > 24 h ago,
   a once-per-session reminder is queued to update it.
4. **`autoDetectBugFix`** in `post-write.js` — regex-classifies an Edit's
   `old_string → new_string` into categories (error-handling, null-safety,
   guard-clause, wrong-value, operator-fix, missing-import, async-fix,
   type-fix, refactor) and **writes a new bug entry** to `buglog.json` with
   dedup within a 5-minute window (`src/hooks/post-write.ts:359-412`).

### 4.3 Cross-agent bridge

`src/cli/memory-migrate.ts` runs in two directions:

- **cerebrum → Claude Code auto-memory**: mirrors the three sections into
  `~/.claude/projects/<slug>/memory/openwolf-*.md` with marker fences, so
  Claude recalls Do-Not-Repeat natively without the digest double-injecting it.
- **Native MEMORY.md → cerebrum**: the reverse bridge (2.5) writes the Claude
  native memory's INDEX (pointers, not bodies) into a marker-fenced
  `## Native memory index (Claude Code, this machine)` section of cerebrum,
  so other agents and teammates at least know what topics Claude captured.

### 4.4 Buglog FTS5 retrieval

`src/hooks/bug-index.ts` uses `node:sqlite` (builtin, no dependency) to build
a disposable FTS5 index at `.wolf/cache/buglog.db` whenever `buglog.json`'s
mtime changes. Normalized signature strips hex ids, paths, digits, punctuation.
On Node < 22.5 the module returns null and `pre-write.js` falls back to a
Jaccard-overlap matcher. Search is gated to same-file hits and cross-file
hits only with a tag match or ≥3-token overlap (precision gate).

### 4.5 PageRank importance

`src/anatomy/importance.ts` builds a directed graph from per-language import
regexes (TS, JS, Py, Go, Rust, PHP), runs a hand-rolled power-method PageRank
(damping 0.85, 20 iterations, dangling-mass follows the restart vector), and
**persists** both `importance` (0..1) and `imports` (resolved edges) per file
on every scan. The `restart` vector can be personalized with seeds, which is
how `openwolf map --focus <terms>` and session-touched files bias the
ranking (the aider repo-map trick).

### 4.6 Net effect on "learning"

OpenWolf is honest about not being an ML system. The system "learns" in four
narrow senses:

1. **Adds entries** to cerebrum.md / buglog.json when humans/agents prompt it.
2. **Re-uses** prior entries via grep/FTS5 retrieval at the next decision point.
3. **Consolidates** old memory.md rows weekly into `> Consolidated session
   (N actions)` markers (`src/daemon/cron-engine.ts:293-350`).
4. **Reweights** PageRank by what this session touched (restart vector bias).

There is **no automatic summarization, no embedding model, no clustering,
no model feedback loop, no RL-style improvement**. Every "learning" is a
file write the agent was instructed to make.

---

## 5. The Bash output governor (flagship 2.3)

Empirical grounding (CHANGELOG 2.3): Bash carries **48.3%** of all
tool-result tokens; results over 2 k tokens are **25.8%** of bash output.
Native platform truncation is positional head/tail with zero semantic
filtering.

When Bash stdout > `threshold_tokens` (default 2000), `post-bash.js`
classifies the command into one of six families via regex
(`src/hooks/bash-output-governor.ts:48-60`) and **rewrites the tool result**
via the harness's `updatedToolOutput` channel:

| Family | Default action | Condensation |
|--------|----------------|--------------|
| `grep_flood` (grep -rn / rg) | `replace` | First K matches per file + per-file counts + total |
| `file_print` (cat / head / tail / sed -n) | `replace` | Head-and-tail window |
| `git_show` (git show / log -p / diff) | `replace` | Commit header + per-file diff stats; hunks elided with counts |
| `test` (jest, pytest, go test, …) | `suggest` | Never auto-replaced — failure detail matters more |
| `build` (tsc, vite build, cargo build, …) | `suggest` | Same |
| `unknown` | `suggest` | Generic head/tail |

Hard rules (everywhere in the file):
- **stderr is never modified** — errors must reach the model verbatim.
- **The full stdout is always preserved** at `.wolf/cache/bash/<id>.log` with a
  pointer in the condensed output.
- **Condensation only fires when it saves ≥30%** (the `result` check).
- **Ground-truth measurement**: for every governed call the ledger records
  original vs entered tokens. This delta is unique — only the hook doing the
  rewrite can know what actually entered context, since platform telemetry
  logs pre-hook output.

Bash-channel read dedupe: `cat/head/tail/sed -n` are also parsed and
registered in `session.files_read`, closing the blind spot where all
measured duplicate reads actually lived.

---

## 6. Token accounting — measured, estimated, verified

Three levels of truth, surfaced in `openwolf report` (`docs/how-it-works.md:150-166`):

1. **Estimates** — character-ratio heuristic (3.5 chars/tok code, 4.0 prose).
2. **Measured** — real input/output/cache tokens parsed from the harness's
   JSONL transcript at `Stop`. Includes subagent sidechains.
3. **Verified** — the same transcript records every hook invocation as an
   attachment line; `verifyHookDelivery()` cross-checks which hooks fired,
   which failed, and which injected context provably entered the conversation
   (`src/hooks/hook-attachments.ts`).

The dashboard hero number is **"tokens verifiably kept out of context"** —
the governed delta plus the verified injection count, displayed with
**overhead as a ratio** ("1.6% of what it kept out") and priced at Anthropic's
list rates per model, broken into cache reads / output / cache writes / fresh
input (v2.5 cost panel).

`cache-attribution.ts` also detects prompt-cache rebuilds from the usage
sequence and names the trigger (model switch, compaction, version change,
cache expiry, or honestly unattributed) — rebuilds are the most expensive
single event in an agent session because they re-pay the entire context at
the write rate instead of the 0.1× read rate.

---

## 7. Daemon, cron, dashboard

Optional background process (`src/daemon/wolf-daemon.ts`):

- **Express + ws** server bound to `127.0.0.1:18791` with **per-project bearer
  token auth** (printed by `openwolf dashboard`, stored at
  `.wolf/dashboard-token`). Origin-locked to localhost / 127.0.0.1 / ::1.
- **Cron engine** (`src/daemon/cron-engine.ts`): three default tasks wired by
  `cron-manifest.json`:
  - `anatomy-rescan` — every 6 h, **gated on actual staleness** (stat sweep or
    git HEAD move). Blind rescans were abandoned because 6 h loses to normal
    commit cadence.
  - `memory-consolidation` — daily 02:00, compresses `memory.md` rows older
    than 7 days.
  - `token-audit` — weekly Monday 00:00, generates waste report and merges
    measured transcripts into the ledger.
  - Retry with exponential/linear backoff, dead-letter queue, alert after
    consecutive failures, locked read-modify-write on state.
- **File watcher** pushes state changes to the dashboard via WebSocket.
- **Health heartbeat** — `heartbeat.json` per hook; cron-state heartbeat
  every `heartbeat_interval_minutes` (default 30).
- **Dashboard** — React + Tailwind + Recharts SPA at `src/dashboard/app/`,
  Vite-built to `dist/dashboard/`. Panels: measured tokens, per-agent usage,
  cache rebuild attribution, hook health, anatomy browser, cron control.

`openwolf update` is the upgrade path: backup, re-run all registered agents'
adapters, refresh bundled skills, run a selfcheck on every installed hook,
fail loudly instead of leaving a broken install. v2.5 also removes hand-
patched Windows-path hook entries that survived previous filters.

---

## 8. What it does NOT do

These are deliberately absent and called out by the authors as principled:

- **No vector embeddings, no RAG, no semantic search** beyond the
  regex-based PageRank on the import graph and the FTS5 over buglog.json.
- **No LLM calls of any kind** — the entire codebase is dependency-light
  (chalk, chokidar, commander, express, node-cron, open, tree-sitter, ws)
  and no `anthropic` / `openai` / `@google/generative-ai` dependency exists.
  `bench.ts` shells out to `claude -p` only for the optional A/B benchmark
  and only with `--yes` confirming the user wants real API spend.
- **No automatic reflection, summarization, or knowledge-graph extraction**.
  The weekly cerebrum reflection that did this was removed in v2.5 as a
  single-source contradiction to the "no API calls" claim. CHANGELOG is
  candid about it: *"the only path in OpenWolf that reached a model… gone,
  along with the Insights dashboard panel, suggestions.json, and the
  cron.use_claude_p / cron.api_key_env settings."*
- **No content-level intelligence about what the user is doing**. It knows
  file paths, hashes, and import edges; it does not parse semantic intent
  out of conversations.
- **Not a knowledge graph** (no triples, no ontology, no reasoning engine).
  The "graph" is the import DAG + PageRank.
- **Scope**: project-local only. No global cross-project knowledge.
  The `[features]` registry exists (`src/cli/registry.ts`) for upgrade UX,
  not as a shared corpus.

---

## 9. Does it fit the WHALE Harness as a memory agent?

**Verdict: it fits as ONE layer in the memory stack — specifically as the
"project-local episodic + procedural memory" layer — but it does not by itself
satisfy WHALE's stated specs around "structured memory", "self-learning", or
"rhizomatic-like structure".**

Mapping to the WHALE README's spec (whale-harness/README.md:11-47):

| WHALE requirement | OpenWolf coverage | Notes |
|---|---|---|
| Multiagent | ✅ First-class | Adapters for Claude Code, Codex, OpenCode, Gemini CLI, Cursor, Antigravity. Same `.wolf/` shared by all of them. |
| Memory (semistructured, llm-wiki, Semantica-style) | ⚠️ Partial | Markdown + JSON with budgeted sections, FTS5 bug search, import-graph. No ontology, no RDF, no semantic triples. Could be **complemented** by Semantica; the two don't conflict. |
| **Self-learning** | ⚠️ **Weak** — see §10 | All "learning" is file writes the agent was told to make. No model feedback loop, no automatic reflection, no embedding updates. |
| Powerful on cheap backbones | ✅ | Hashes PageRank + FTS + size heuristics run in microseconds. |
| Swarm capabilities | ❌ | Single-agent harness middleware. No spawn / fanout primitive. |
| Rhizomatic-like structure | ❌ | Strictly tree-like per project (root → folders → files), with one graph on top (import DAG). No mesh, no cross-cutting edges. |
| Loop engineering (act → observe → decide → repeat) | ⚠️ Partial | Hooks ARE the loop, but only at the agent-tool boundary, not at the harness-orchestration boundary. |
| Capability surface (skills, plugins, MCP, tools) | ✅ Skills | Ships `/handoff`, `/security-audit`, `/reframe` as Claude skills. No MCP server. |
| Hooks/Middleware for deterministic execution | ✅ Canonical example | This is literally what it does best. |
| Cheap to run, lean token consumption | ✅ **Flagship feature** | The entire reason it exists. |
| Self-evolving capabilities (Acceptance criteria, line 134) | ⚠️ Conditional | See §10 — it can grow its own memory **if the agent is instructed to**, but it does not autonomously discover new skills/capabilities. |
| Native integrations: "memory engines like openwolf" | ✅ The README already names it (line 142) | `whale` is documented as integrating "memory engines like openwolf" as a native integration. |

### Where OpenWolf shines inside WHALE

1. **Project-scoped episodic memory** that survives sessions, agents, and
   teammates. Drop it in as a sidecar at `.wolf/` of any project whale
   orchestrates agents against; the .gitignore split means
   conventions + bug fixes + anatomy commit, ledgers stay local.
2. **Token economics** — the Bash governor + duplicate-read advisory +
   anatomy shortlist + budget-enforced state files are the best in class
   for keeping agents cheap. WHALE's "lean in token consumption" line maps
   directly onto OpenWolf's hero metric.
3. **Cross-agent continuity** — if WHALE spawns a swarm that mixes Claude
   and Codex and OpenCode workers, the shared `.wolf/` is genuinely portable
   in a way the native memories of each agent are not.
4. **Hooks are reusable** — the lifecycle-hook pattern is well-designed
   (`hookMain`, heartbeat, self-test, never-`exit(1)`) and could be lifted
   as the middleware layer WHALE itself uses between its orchestrator and
   any subagent harness.
5. **Measured not estimated** — the verification loop (`verifyHookDelivery`,
   transcript-based real usage, governor delta at the rewrite point) is
   exactly the kind of honest accounting WHALE needs to report cost back
   to a user.

### Where OpenWolf is insufficient for WHALE on its own

1. **No real self-learning.** There is no feedback signal that updates
   future behavior beyond "files that survive into next session". A WHALE
   loop that wants *evolving* behavior needs something on top — Semantica
   for the semantic side, an external reward/critic, or an explicit
   reflection step the orchestrator drives.
2. **No shared global memory** — project-local only. WHALE's "pickup /
   retrieve from memory" step across the user's whole history needs a
   cross-project layer (e.g. Neo4j/Cogee reviewed in sibling docs).
3. **No semantic memory** — FTS5 over bug messages and PageRank over
   imports are the only retrieval primitives. No embeddings, no similarity
   recall over arbitrary past sessions, no entity linking.
4. **No swarm primitives** — nothing about spawning, fanning out,
   consensus, or result aggregation.
5. **Markdown-first** — beautiful for human inspection, painful for
   machine reasoning at scale. The 2 000-token cerebrum budget is a
   hard ceiling; if WHALE wants long-form memory across many projects,
   this won't carry it.
6. **AGPL-3.0** — copyleft. WHALE should be sure the integration tier
   (an MCP wrapper, a whale-native plugin) is the boundary that takes
   the license, not the whale core.

---

## 10. Self-learning — does OpenWolf actually have it?

**Direct answer: no, not in the modern sense.**

OpenWolf has:

- ✅ **Cumulative memory** (cerebrum.md, buglog.json, anatomy-index.json
  grow over time).
- ✅ **Retrieval** at the next decision point (Do-Not-Repeat grep, FTS5 bug
  search, anatomy hints, PageRank-seeded map).
- ✅ **Compaction survival** (precompact snapshot + session-start restore).
- ✅ **Decay countermeasure** (post-batch rule re-injection every 25 calls).
- ✅ **Cross-session continuity** (state files persist, git-tracked,
  committed; lessons travel with the repo).
- ✅ **Verified accounting** (it can prove what worked — governor delta,
  transcript-verified injections).

OpenWolf does **NOT** have:

- ❌ **Online learning / weight updates / fine-tuning**.
- ❌ **Embedding-based semantic recall**.
- ❌ **Automatic reflection or summarization** (the weekly reflection cron
  that *did* this was removed in v2.5).
- ❌ **Closed-loop feedback** — it can't observe "the agent made a mistake
  because X" and tune future behavior beyond writing a note in cerebrum
  the next time the same trigger fires.
- ❌ **Skill / tool autodiscovery** — `/security-audit`, `/reframe`,
  `/handoff` are hand-bundled; nothing discovers or generates new ones.
- ❌ **Reinforcement / reward signal**. The waste-detector flags
  inefficiencies but does not change policy in response.

In short, **OpenWolf is a self-recording, self-retrieving notebook for an
agent — not a self-improving agent.** The "learning" curve is whatever the
human + agent put into the markdown files over time. The instruction-decay
re-injection is the closest thing to a learned behavior, and even that is
a hard-coded cadence, not a learned schedule.

For WHALE's spec line *"It should be self-learning!"* (README line 20) and
the acceptance criterion *"Reliable harness with memory and self-evolving
capabilities"* (line 134), OpenWolf contributes the **memory half** of that
sentence, but the **self-evolving** half must come from somewhere else —
likely the WHALE orchestrator loop itself (act → observe → decide →
reflect → write back to OpenWolf as the storage layer, with Semantica or a
similar engine handling the semantic layer).

---

## 11. Quick file index (paths in the cloned source)

```
README.md                              ← one-page pitch
docs/how-it-works.md                   ← architecture (cleanest narrative)
docs/hooks.md                          ← hook reference
docs/commands.md                       ← CLI reference
src/templates/cerebrum.md              ← the actual schema of "memory"
src/templates/anatomy.md               ← durable project index format
src/templates/config.json              ← every tunable default
src/templates/cron-manifest.json       ← scheduled-task format
src/hooks/session-start.ts             ← digest builder + compact-restore
src/hooks/post-write.ts                ← anatomy update + auto bug detect
src/hooks/pre-write.ts                 ← Do-Not-Repeat + bug search advisories
src/hooks/post-bash.ts                 ← Bash output governor entry
src/hooks/bash-output-governor.ts      ← family classifier + condensers
src/hooks/rule-reinjection.ts          ← topRules + scoped rules
src/hooks/bug-index.ts                 ← FTS5 over buglog (node:sqlite)
src/hooks/ledger.ts                    ← measured + verified token usage
src/anatomy/importance.ts              ← PageRank over import graph
src/scanner/anatomy-scanner.ts         ← walkDir + buildAnatomy
src/buglog/bug-tracker.ts              ← Jaccard dedup + search
src/cli/init.ts                        ← `openwolf init` lifecycle
src/cli/memory-migrate.ts              ← cerebrum ↔ Claude auto-memory bridge
src/daemon/wolf-daemon.ts              ← Express + WS + cron + watcher
src/daemon/cron-engine.ts              ← scheduler + dead-letter
src/tracker/waste-detector.ts          ← patterns: repeated reads, bloat, …
src/tracker/cache-attribution.ts       ← prompt-cache rebuild reasons
src/agents/index.ts                    ← adapter registry + auto-detect
src/agents/codex.ts, opencode.ts, …    ← per-agent install logic
src/templates/opencode-plugin/         ← OpenCode native plugin source
```

---

## 12. TL;DR for WHALE

- OpenWolf is **agent-side persistent project memory + token economy
  middleware**. Pure local file I/O, zero model calls, very fast.
- Excellent for **cumulative cross-session project memory**, **cross-agent
  continuity** (Claude + Codex + OpenCode share one `.wolf/`), and
  **measured token accounting** with honest governor-delta math.
- **Not** an ontology, **not** a knowledge graph, **not** a self-learning
  system, **not** a swarm orchestrator.
- Self-learning: **weak** — file writes by instructed agents, no
  autonomous reflection (the one reflection loop was removed in v2.5),
  no embedding feedback. The decay re-injection is the only learned
  behavior and it is a hard-coded cadence.
- Best fit inside WHALE: **storage layer for the episodic + procedural
  memory + hook middleware** that the orchestrator loop reads and writes.
  Pair with Semantica (semantic / KG layer, sibling doc) for the
  "structured memory" half and with a WHALE-native reflection step for
  the "self-evolving" half.
- License: **AGPL-3.0** — mind the boundary at which OpenWolf code is
  invoked from whale so the copyleft does not bleed into whale core.
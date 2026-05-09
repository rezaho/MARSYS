# 10 — Memory Architecture

> Synthesizes a deep-research pass on April-2026 best practice (Letta / mem0 / Zep / Generative Agents / Cline / Cursor / Claude.ai memory / ChatGPT memory / Auto Dream / OpenInference) with an independent first-principles design pass. The two converged on most major calls, with one killer empirical anchor: **Letta's own benchmark shows plain filesystem + `grep`/`open` beat their graph memory (Mem0g) on LoCoMo, 74.0% vs 68.5%.** A memory-systems company published evidence that their own product was outperformed by markdown files. **Markdown-as-substrate wins.**

This doc covers the storage tiers, write paths, read paths, consolidation cadence, multi-agent memory bubbling, security defenses (memory poisoning is a real attack class), and user-control UX. The meta-agent and sandbox / permission model live in [`09-meta-agent.md`](./09-meta-agent.md).

---

## 1. Four storage tiers (not five, not one)

The tiers are distinguished by **access pattern and lifecycle**, not by cognitive-science labels (working / episodic / semantic / procedural — operationally over-fit).

### Tier 1 — Working memory (in-process, ephemeral)
- **Format:** in-process Python data + the active branch's message buffer (Marsys's `ConversationMemory` already)
- **Contents:** current task's running thought stream, intermediate tool outputs, plan state, retrieved context
- **Lifecycle:** dies when the branch / agent turn completes. A *summary* is bubbled up; raw scratchpad is discarded.
- **Token cap:** by tokens, not turns. On overflow: anchored iterative summarization.

### Tier 2 — Session log (append-only, durable)
- **Format:** SQLite (`workbench.db`'s `events` table) **plus** JSONL files under `~/.spren/sandbox/logs/events/YYYY/MM/DD.jsonl` for cold storage
- **Contents:** every external stimulus and every agent action — incoming messages, tool calls, agent spawns, workflow transitions, fact extractions, user edits to memory itself. SQLite FTS5 enables BM25 over this for free.
- **Lifecycle:** never mutated. Compressed after 30 days. Cold-archived after 1 year. **This is the only tier that is "true" in a strong sense — every higher tier is rebuildable from this.**
- **Why immutable:** without it, every higher-tier mistake is unrecoverable. Memory bugs at month 6 need to be debuggable.

### Tier 3 — Markdown knowledge base (the consolidated, human-editable layer)
- **Format:** plain markdown under `~/.spren/sandbox/shared/memory/`. Each file is human-readable and editable.
- **Layout:**
  ```
  shared/memory/
  ├── profile.md                    # who the user is — facts, preferences, identity
  ├── projects/
  │   └── <project-slug>.md         # one file per ongoing project
  ├── relationships.md              # people the user mentions; companies, etc.
  ├── procedures/
  │   └── <slug>.md                 # learned how-to's (one per procedure)
  ├── decisions/
  │   └── ADR-<n>-<slug>.md         # decisions the agent + user made together
  ├── active_context.md             # current focus state — the file the agent reads on every turn
  └── journal/
      └── YYYY-MM-DD.md             # daily journal entries written by the consolidation pass
  ```
- **Contents:** structured human-language. Each file uses YAML frontmatter for facts that need to be queryable; prose for narrative.
  Example `projects/spren-launch.md`:
  ```yaml
  ---
  status: active
  started: 2026-04-15
  current_focus: meta-agent design
  team: ["user", "Reza"]
  blockers: []
  ---

  # Spren launch

  Working on the meta-agent architecture as of 2026-04-28. Memory architecture
  research came back with a strong recommendation for markdown-as-substrate...
  ```
- **Lifecycle:** edited by the consolidation pass (one git commit per pass). Edited by the user via `$EDITOR`. NEVER edited live by an agent during a turn (poisoning defense — see § 5).
- **This is the only tier the user is allowed to think about.** Everything else is plumbing.

### Tier 4 — Hybrid discovery index (rebuilt from log + KB)
- **Format:** SQLite + `sqlite-vec` for embeddings (pin **0.1.9**, the latest stable; **do NOT use 0.1.10 alphas** as of 2026-04). Optionally `tantivy` Python bindings (pin `>=0.25.1,<0.26`) as a separate column-store index if FTS5 isn't fast enough at 100k+ rows.
- **Contents:** indexed chunks of the markdown KB + indexed messages from the session log. Each chunk carries `{source: file_path|event_id, timestamp, entity_refs[], chunk_id}`.
- **Lifecycle:** **fully derivable from Tiers 2 and 3.** Rebuilt from scratch on schema change. Old chunks not deleted on update — re-scored: a chunk last referenced a year ago and never retrieved drops out of the hot index but remains.
- **Why a discovery aid, not source of truth:** vector indexes are the easiest tier to poison and the hardest to debug. Authoritative answers come from markdown + session log; the index just helps the agent *find* them.

**Skip the knowledge graph.** Zep/Graphiti's bi-temporal model is impressive for enterprise CRMs where many agents argue over the same entity. For one user on a laptop it's over-engineering and a maintenance liability. If we ever need temporal queries, we use `valid_from` / `valid_to` YAML fields on facts in markdown.

## 2. Three write paths, in order of trust

### A. Explicit user override (highest trust)
- User says "remember that..." OR uses CLI `spren memory remember "<fact>"` OR uses the web UI "remember this" button on a chat message
- The agent writes directly to the relevant markdown file with a provenance line:
  > `> source: user, 2026-04-28, msg-id` followed by the fact
- Bypasses the consolidation cycle. User edits are authoritative.

### B. LLM-judged extraction (medium trust)
- During each agent turn, a cheap classifier prompt scores whether anything in the user's last message is durable ("user prefers X", "deadline is Y", "user's preferred editor is Zed")
- **If yes, queue it in `pending_facts` (a SQLite table). Do NOT write to markdown live.**
- Reason: live extraction-on-write is the main vector for memory poisoning ([MINJA, NeurIPS 2025](https://openreview.net/forum?id=QVX6hcJ2um) shows >95% injection success against production agents via query-only interaction). Queueing means a single poisoned message can't immediately corrupt the knowledge base.

### C. Async consolidation (lowest per-event trust, highest aggregate value)
- A scheduled "dream" job that reviews `pending_facts` + recent journal + sub-agent outputs and produces edits to the markdown KB
- See § 6 for cadence and pipeline detail
- Run by a **dedicated, weaker model** — cheaper, and structurally separates "doing" from "reflecting"

**Sub-agents (per the permission tiers in [`09-meta-agent.md`](./09-meta-agent.md)) NEVER write to the shared KB directly.** They write to their own scoped `instances/<id>/scratchpad/` and emit a `BranchSummary` artifact:

```
{
  "goal": "...",
  "actions_taken": [...],
  "results": {...},
  "new_facts_proposed": [
    {"predicate": "...", "value": "...", "confidence": 0.x, "source": "..."}
  ],
  "procedural_lessons": [...],
  "open_questions": [...]
}
```

The parent agent (or the consolidation pass) decides which `new_facts_proposed` get committed. **Parent acts as gate.** A hallucinated fact from a sub-agent should not pollute the global store.

## 3. Three read tools, not one magic memory

The agent does NOT have a single auto-injecting memory layer. Instead, three explicit tools:

| Tool | Cost | Use for |
|---|---|---|
| `read_file(path)` | Cheapest | Reading `active_context.md` on session start; reading a specific known doc |
| `grep(pattern, scope)` | Cheap | Exact-match search over markdown + session log; the agent's first reach when it knows what string to look for |
| `recall(query, k=5, time_window=None)` | Expensive | Hybrid BM25+vector+rerank; used only when structured/grep search isn't enough; reformulatable by the agent |

**Why three tools, not one:** Letta's benchmark showed agents win when they can **reformulate queries and iterate**, which requires giving them tools they understand from training (POSIX read, grep). Resist building a single auto-injecting "memory layer." Force the agent to retrieve explicitly. This also makes traces auditable (every retrieval is a tool call, visible in the trace).

The agent is prompted to prefer `read_file` and `grep` over `recall`. `recall` returned items include `{path, snippet, asserted_at, volatility, citation}` — the agent reasons about staleness rather than the system pre-filtering. Trust the agent with metadata; don't pre-chew.

## 4. Volatility — explicit, not silent decay

Every fact in the markdown KB carries a `volatility` tag in YAML frontmatter:

| Volatility | Re-verification cadence | Examples |
|---|---|---|
| `stable` | Never (only on direct contradiction) | birthday, allergy, PII, language preference |
| `slow` | Quarterly | preferred editor, current city, primary email |
| `volatile` | Weekly or per-touch | current project, current focus, what user worked on yesterday |

**The rule:** volatility is set at extraction time based on predicate type. We maintain a hand-coded mapping in `src/spren/memory/volatility.py` (e.g., `prefers_editor → slow`, `current_project → volatile`, `birthday → stable`); unknown predicates default to `slow`. The user can override per-fact via the markdown frontmatter. Reads of volatile facts older than the threshold get a flag — `[possibly stale: last verified 2026-04-12]`. The agent then either re-verifies opportunistically ("Are you still working on Q3-launch?") or surfaces the uncertainty in its reasoning.

**No silent decay.** Silent decay confuses "old" with "wrong." A user's allergy from 2024 is still true in 2026. Decay only volatile predicates, and even then *flag* rather than delete.

## 5. Strict supersession — the deterministic algorithm

When a new fact `f_new = (entity, predicate, value)` is committed (from explicit user, or from the consolidation pass), the system follows a **deterministic algorithm** — no LLM in the supersession path itself. LLMs propose facts; pure code commits them.

### Algorithm

```python
def commit_fact(f_new):
    # 1. Look up active fact for the same (entity, predicate)
    f_old = facts.find_active(entity=f_new.entity, predicate=f_new.predicate)

    # 2. No prior fact: just insert
    if f_old is None:
        facts.insert(f_new, status='active')
        return

    # 3. Same value: no new row, just update last-seen timestamp
    if f_old.value == f_new.value:
        facts.update(f_old, last_seen_at=now())
        return

    # 4. Stable fact, non-user source — refuse to silently auto-replace
    if f_new.volatility == 'stable' and f_new.source != 'user_direct':
        facts.insert(f_new, status='disputed', supersedes=f_old.id)
        facts.update(f_old, status='disputed', superseded_by=f_new.id)
        enqueue_user_decision(f_old, f_new)
        return

    # 5. Standard supersession: old becomes 'superseded', new becomes 'active'
    facts.insert(f_new, status='active', supersedes=f_old.id)
    facts.update(f_old, status='superseded', superseded_by=f_new.id)
```

Status values: `active` (current truth), `superseded` (replaced; kept for audit), `disputed` (conflicting stable facts awaiting user), `tombstoned` (explicitly forgotten by user; never re-extract).

### What this looks like in markdown

Both fact blocks remain in the file. The consolidation pass (or the explicit user-write path) writes the new block AND mutates the old block's frontmatter to mark it superseded. One git commit per consolidation captures all changes atomically.

```yaml
::: fact id=f-2025-10-15-spouse
predicate: spouse_name
value: Alex
status: superseded
superseded_by: f-2026-04-28-spouse
asserted_at: 2025-10-15
volatility: stable
source: user, msg-old123
:::

# Updated below (2026-04-28)

::: fact id=f-2026-04-28-spouse
predicate: spouse_name
value: Alex Chen
status: active
supersedes: f-2025-10-15-spouse
asserted_at: 2026-04-28
volatility: stable
source: user, msg-xyz789
:::
```

### Reading current truth

- **From SQLite:** `SELECT * FROM facts WHERE entity=? AND predicate=? AND status='active'`
- **From markdown:** find the fact block with `status: active` for that (entity, predicate)

Both must agree. The `MarkdownIndexer` (see § 11) is what keeps them aligned.

### Audit chain

`spren memory why <claim>` walks the `supersedes_id` pointers backward to the original assertion, then to the source event log entry, then to the channel/turn the assertion came from. Full traceability — works whether the user queries SQL or reads the markdown directly.

**Refuse to auto-merge contradictory stable facts silently.** At scale this is what produces the "agent thinks I live in two cities" failure mode. The `disputed` state forces a human decision rather than a silent guess.

## 6. Consolidation — daily-ish, hybrid trigger, git-committed

**Trigger:** **24 hours elapsed AND ≥5 sessions accumulated since last run.** Both conditions must hold. Auto Dream's rule, validated by their failure logs: pure 24-hour cron triggers on quiet days where nothing happened (wasting tokens); pure session-count triggers fire too often on intense days and produce conflicting partial summaries.

**Pipeline (4 stages, runs in idle hours, by a cheaper model):**

1. **Inventory** — list all current markdown files; hash them; load `pending_facts` since last run; load sub-agent `BranchSummary` artifacts from completed instances since last run
2. **Extract & merge** — LLM pass over the day's session log + pending_facts; produce *diff proposals* against existing markdown
3. **Conflict resolution** — for each new fact, retrieve overlapping existing facts; if contradiction, follow the supersession rules in § 5
4. **Apply & log** — write the diffs as a normal **git commit on `~/.spren/sandbox/shared/memory/`**. Free version control, free audit trail, free rollback. Auto Dream's biggest documented failure mode is "no consolidated log of what changed" — we get this for free with git.

**Critical rules:**
- **Always consolidate the raw delta only** — never re-process already-consolidated markdown. Re-summarizing already-summarized material erodes facts (the "transcript replay drift" failure).
- **Every consolidated fact must cite a `session_id:event_id` from the session log.** Reject facts without provenance — that's the only defense against consolidation hallucination.
- **Rebuild the discovery index** at the end of consolidation (or schedule for next idle) so it reflects the new KB.

**Failure modes (from the literature, plan for these):**
- *Too frequent (hourly):* premature compression destroys context that turns out to matter the next day. Token cost balloons.
- *Too rare (weekly+):* `pending_facts` queue grows past the LLM's context, you have to chunk, and chunked consolidation produces conflicting summaries.
- *Consolidation hallucination:* mitigated by required provenance.
- *Drift through repeated re-summarization:* mitigated by delta-only rule.

## 7. Memory poisoning — defenses

[MINJA](https://openreview.net/forum?id=QVX6hcJ2um) and [MemoryGraft](https://arxiv.org/html/2512.16962v1) demonstrate >95% memory-injection success against production agents via query-only interaction. We MUST plan for this from the first Spren release (v0.3).

**Defenses, all in our model:**
- **Queue-then-consolidate** (not live write) — already covered in § 2. A poisoned message can't immediately corrupt the KB.
- **Per-channel trust scoring** — facts originated from the user terminal score higher than facts from a Slack DM, which score higher than facts from inbound webhook. Trust score factored into supersession (a low-trust fact can't auto-supersede a higher-trust one).
- **Provenance metadata on every fact** — `source: <channel-id>:<event_id>`, `extracted_by: <agent-id>`, `confidence: 0.x`.
- **Consolidation pass acts as a second-look** — sees the full context of where the candidate fact came from; can reject suspicious ones.
- **User-visible audit log** — `spren memory why <claim>` shows the supersession chain back to source events. Lets the user spot poisoned facts and `forget` them.

**SP-015 (new principle, see [`08-design-principles.md`](./08-design-principles.md)):** Untrusted-channel writes never touch the KB live. Implementers MUST NOT add code paths that bypass the `pending_facts` queue for any non-user-direct source.

## 8. Multi-agent memory — bubbling up, not sharing across

When sub-agents (working instances or team managers' subordinates) do work in parallel, their memory must NOT pollute the global KB directly. Pattern:

- Each sub-agent gets a **read-only snapshot** of relevant Tier 3 facts at spawn (filtered by task scope), plus their own Tier 1 working memory and a Tier 2-equivalent local log
- They do their work in their scratchpad
- On completion they emit `BranchSummary` (see § 2.C)
- The parent agent reads the summary, reasons about it, and decides which `new_facts_proposed` to commit (either directly via the user-trust path if the parent is the main agent acting on user-direct intent, or via `pending_facts` queue otherwise)

**Avoid mesh-style shared memory** between sub-agents. Multi-agent memory architecture papers warn this creates cache-coherence problems with no clean protocol yet. A later release may add cross-team-manager direct queries if patterns emerge.

## 9. User control — three operations, all on the markdown

The killer feature of file-based memory: the user can `vim` the file. Cline / Cursor users already do this and it's the single most-loved property of file-based memory in 2026 deployments.

CLI:
```
spren memory show <topic>            # cat the relevant markdown
spren memory edit <topic>            # opens the relevant markdown in $EDITOR
spren memory remember "<fact>"       # appends to the appropriate markdown with a "source: user" provenance line
spren memory forget <fact-id>        # tombstones the fact (see below)
spren memory why <claim>             # traces the supersession chain back to source events
spren memory journal <date>          # opens the day's journal entry
spren memory rebuild-index           # rebuild the Tier 4 hybrid index from scratch
```

**`forget` semantics matter:** true deletion would let the agent re-extract the forgotten fact from the session log tomorrow. Instead:
- The fact's YAML frontmatter gets `tombstoned: true, tombstoned_at: <ts>, tombstoned_by: user`
- The fact entry is moved to `~/.spren/sandbox/archive/forgotten/YYYY-MM-DD.md` with a backlink
- The consolidation pass respects tombstones — it won't re-extract a tombstoned fact even if it appears in new events
- Reversible: `spren memory restore <fact-id>` brings it back

**Honest forgetting requires remembering that you forgot.** This satisfies "right to be forgotten" without losing the audit trail that makes the system trustworthy.

Web UI surfaces (settings → memory): same operations as the CLI, plus a search/browse view of the markdown KB rendered as a wiki.

## 10. Cost expectations

Reference numbers (mem0 paper, April 2025; should be similar order in 2026):
- Hybrid memory architectures vs full-context: ~91% p95 latency reduction, ~90% token savings
- For Spren, the dominant memory cost will be the **consolidation pass**, not retrieval. Budget that as a single deliberate ~$0.10–0.50/day expense rather than nickel-and-diming every turn with embedding calls.
- Per-turn live extraction (Tier 2 write path B) is cheap (single small-model classifier call per user message) and worth keeping.
- Embedding generation: only happens during index rebuild, infrequent.

The per-day budget cap (default $10/day, see [`09-meta-agent.md`](./09-meta-agent.md) § Cost ceiling) covers all this with substantial room.

## 11. Markdown → SQL indexer (deterministic, no LLM)

Markdown is the source of truth (SP-016); SQLite is a derived index, rebuilt from markdown by a deterministic process. This section specifies that process precisely so it can be implemented without ambiguity.

### Fact block syntax

Two ways to write a fact in markdown, both parseable by the indexer:

**1) YAML frontmatter at the top of a file** — for file-scoped facts (e.g., a project file's status):

```yaml
---
status: active
started: 2026-04-15
current_focus: meta-agent design
---

# Spren launch

(prose body...)
```

**2) Inline fact blocks within prose** — for facts about specific entities/predicates:

```yaml
::: fact id=f-2026-04-28-editor
predicate: prefers_editor
value: Zed
status: active
asserted_at: 2026-04-28
volatility: slow
source: user, msg-abc123
:::
```

The `::: fact ... :::` syntax is a standard "fenced div" pattern (also used in pandoc and some markdown extensions). It's parseable with a simple regex; works everywhere markdown is rendered (renders as a code block in viewers that don't recognize it, which is acceptable graceful degradation).

### MarkdownIndexer process

A long-running Python service that watches `~/.spren/sandbox/shared/memory/`. Implementation:

```python
class MarkdownIndexer:
    """Watches markdown KB; updates derived SQLite index. NO LLM calls."""

    def __init__(self, memory_dir: Path, db: SQLiteConnection):
        self.memory_dir = memory_dir
        self.db = db
        self.observer = Observer()  # watchdog: inotify/FSEvents/ReadDirectoryChangesW

    def start(self):
        self.observer.schedule(MemoryFSHandler(self), self.memory_dir, recursive=True)
        self.observer.start()
        # Also a fallback periodic full scan every N minutes
        schedule_periodic(self._full_scan, interval_minutes=5)

    def reindex_file(self, path: Path):
        """Called on file change OR full-scan visit."""
        content = path.read_text()
        frontmatter = parse_yaml_frontmatter(content)
        fact_blocks = parse_fact_blocks(content)  # regex on ::: fact ... :::

        # Compose all facts found in this file
        facts_in_file = []
        if frontmatter:
            facts_in_file.extend(frontmatter_to_facts(frontmatter, source_file=path))
        for block in fact_blocks:
            facts_in_file.append(block_to_fact(block, source_file=path, source_line=block.line))

        # UPSERT each by id
        with self.db.transaction():
            for f in facts_in_file:
                self.db.upsert("facts", f, key="id")

            # Mark facts that USED to be in this file but are no longer
            current_ids = {f.id for f in facts_in_file}
            stale = self.db.query(
                "SELECT id FROM facts WHERE source_file=? AND id NOT IN (?)",
                (str(path), current_ids))
            for s in stale:
                self.db.update("facts", s.id, {"status": "orphaned"})
```

### Properties

- **Deterministic.** Same markdown input → same SQL state. No LLM judgment in this path.
- **Cheap.** Pure parsing + a few SQL UPSERTs. Sub-millisecond per fact block.
- **Fast.** File-system events trigger reindex within milliseconds (inotify-class).
- **Resilient.** A 5-minute periodic full scan catches anything missed by the watcher (e.g., during daemon restart).
- **User-friendly.** User edits markdown in `vim` → indexer picks it up → SQL reflects → all reads see the change.

### Drift detection

`spren memory verify-index` command walks every fact in SQLite and confirms its source file still contains a matching block. Any drift is reported (not auto-fixed; user reviews and decides). Defends against parser bugs and edge cases.

### What does NOT use the indexer

- The Tier 4 vector index for semantic discovery is built from prose body text, not from fact blocks. That path uses embedding API calls. Cheap, infrequent, separate from the structured-fact indexer.
- The session log (Tier 2) is written directly by the runtime; the indexer doesn't touch it.

### Why no LLM in this path

The user must be able to predict what their markdown edits will produce in the index. An LLM in the loop would mean an edit might or might not be indexed depending on the model's mood. Pure parsing makes the contract explicit: this YAML/fenced syntax → this SQL row. Nothing else.

LLMs are involved upstream — when proposing facts during extraction, when consolidating, when interpreting prose for the discovery index. But the markdown→SQL pipeline is bedrock: deterministic, debuggable, testable.

## 12. Open questions for next round

- **Embedding model drift over years.** When we re-embed in 2027 with a new model, old vectors are dead. Plan for periodic full re-embedding (annual?). Document the migration path.
- **Concrete trust-score computation per channel.** Manual config for v0.3 (settings UI per channel). Learned over time? Heuristic? Open.
- **Cross-device sync** isn't addressed. If Spren ever runs on a phone too, the markdown layer makes sync trivial (git/Syncthing); the SQLite layer needs CRDT or single-writer discipline. v0.4 territory.
- **The `pending_facts` queue size** — what's the threshold at which we trigger consolidation early (independent of the 24h+5-session rule)? Probably token-count-based: if pending_facts would exceed half the consolidation model's context, trigger now.
- **What goes in the discovery index from the session log** — full text of every event would balloon. Probably: every user message + every BranchSummary + every consolidated journal entry. Other event types (heartbeat ticks, tool calls) stay in the session log only.

## References

- [Letta filesystem benchmark, April 2026](https://www.letta.com/blog/benchmarking-ai-agent-memory) — the killer evidence
- [mem0 paper, arXiv 2504.19413](https://arxiv.org/abs/2504.19413) — extraction-on-write technique, hybrid retrieval; we adopt the technique but defer writes via queue
- [MINJA, NeurIPS 2025](https://openreview.net/forum?id=QVX6hcJ2um) — memory poisoning attacks
- [Generative Agents (Park et al.), arXiv 2304.03442](https://arxiv.org/abs/2304.03442) — reflection-on-cumulative-importance trigger model
- [Voyager, arXiv 2305.16291](https://arxiv.org/abs/2305.16291) — procedural memory pattern
- [Cline Memory Bank docs](https://docs.cline.bot/features/memory-bank) — direct template for our markdown layout
- [Cursor rules docs](https://cursor.com/docs/context/rules) — `active_context.md` precedent
- [Claude Code Auto Dream](https://bregg.com/post.php?slug=claude-code-auto-dream-memory-consolidation) — 24h+5-session trigger
- [Memory in the Age of AI Agents survey, arXiv 2512.13564](https://arxiv.org/abs/2512.13564) — operational typology arguments
- [Multi-agent memory architecture, arXiv 2603.10062](https://arxiv.org/html/2603.10062v1) — cross-agent caching warnings
- [VectorHub hybrid retrieval](https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking) — BM25 + dense + reranker pattern
- [Tribe AI on context-aware memory](https://www.tribe.ai/applied-ai/beyond-the-bubble-how-context-aware-memory-systems-are-changing-the-game-in-2025) — transcript replay drift

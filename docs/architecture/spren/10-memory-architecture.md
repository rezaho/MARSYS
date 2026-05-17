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
- **Layout** — two parallel decompositions, both materialized in markdown, both derived from one source of truth into one canonical SQL projection:
  ```
  shared/memory/
  ├── people/                          # entity-typed records — one file per entity
  │   ├── _self.md                     # the user as an entity (references profile/)
  │   ├── p-<slug>.md                  # other people
  │   └── org-<slug>.md                # organizations
  ├── projects/
  │   └── proj-<slug>.md               # entity records — one per project
  ├── procedures/
  │   └── proc-<slug>/                 # SKILL.md format reserved for v0.4 — see § Procedural memory
  │       └── SKILL.md
  ├── decisions/
  │   └── dec-ADR-<n>-<slug>.md
  ├── profile/                         # user-ABOUT facts, by category
  │   ├── identity.md                  # name, pronouns, languages — referenced by _self.md
  │   ├── constraints.md               # GDPR Article-9 sensitive: medical, legal, financial, accessibility (sensitive: true)
  │   ├── values.md                    # ethics, stances, principles
  │   ├── preferences.md               # operational preferences (tooling, comm, decision-style)
  │   ├── capabilities.md              # skills, expertise, what they can / can't do
  │   ├── rhythm.md                    # daily / weekly cadence, working hours, locations, season-of-life
  │   └── goals.md                     # short / medium / long-term goals
  ├── personas/
  │   ├── main.yaml                    # Spren's persona; archetype + 8 sub-axes; evolves through bond
  │   ├── pending_persona_changes/     # persona-evolution proposals awaiting user decision
  │   └── journal/                     # past personas + rejected proposals + bond reset archives
  ├── journal/
  │   └── YYYY-MM-DD.md                # daily journal entries written by the consolidation pass
  ├── active_context.md                # current focus + active-session adjustments — read every turn
  ├── active_todos.md                  # long-term TODO / planning list
  ├── vault/                           # reserved for v0.4 (encrypted; not parsed by indexer)
  └── security/
      └── poisoning_patterns.yaml      # v0.3 empty; v0.4 seeded for SP-022 trajectory review
  ```
- **Two decompositions, not redundant:** Entity records (`people/`, `projects/`, `procedures/`, `decisions/`) answer "who is this entity and what facts attach to it?" Profile categories (`profile/identity.md`, etc.) answer "what kind of fact is this and how do I treat it?" Different categories have different volatility, sensitivity, and read cadence — `constraints.md` (GDPR Article-9) is read on every action that might violate one; `values.md` is read rarely; `rhythm.md` is read on every "should I send this now?" decision. The same fact lives once in the canonical source and is referenced from both views via the indexer; consistency is automatic.
- **`active_context.md` does double duty:** current task focus AND active-session adjustments to behavior or persona that haven't earned consolidation yet. Re-read every turn (eager-loaded in axis #5 of `09-meta-agent.md`). Cleared / condensed by the consolidation pass. Soft cap on size (default 2K tokens); on overflow the agent compresses older entries into a summary line.
- **Single `::: facts` block per entity / category file** (replaces the prior many-blocks-with-IDs syntax). One block at the top of each file, after YAML frontmatter, before the prose body. Each line is `predicate: value [metadata]`. Multi-valued facts are multiple lines with the same predicate. The indexer parses with a small regex.
- **Contents:** structured human-language. Each file has YAML frontmatter (entity metadata: `type`, `id`, `status`, `schema_version`, `created_at`, `updated_at`, `sensitive`), one `::: facts` block, and free-prose body sections.
  Example `projects/proj-spren-launch.md`:
  ```markdown
  ---
  type: project
  id: proj-spren-launch
  status: active
  started: 2026-04-15
  ended: null
  schema_version: 1
  created_at: 2026-04-15
  updated_at: 2026-05-13
  ---

  # Spren launch

  ::: facts
  current_focus: meta-agent design [volatility: volatile, asserted: 2026-04-28, confidence: 0.95]
  team: user [volatility: stable]
  team: p-mike [volatility: stable]
  deadline: 2026-06-01 [volatility: slow]
  blockers: []
  :::

  ## Narrative

  Working on the meta-agent architecture as of 2026-04-28. Memory architecture
  research came back with a strong recommendation for markdown-as-substrate...

  ## Decisions
  - [ADR-12 — Single-process daemon](../decisions/dec-ADR-12-single-process-daemon.md) — 2026-04-20
  ```
- **Lifecycle — two write paths:**
  - **Async default path** (the bulk of writes). The consolidation pass (24h+5sessions trigger) reads the immutable session log + `pending_facts` queue + sub-agent `BranchSummary.procedural_lessons` queue, runs extraction-with-adjudication (CUPMem four-state output: KEEP / STALE / REPLACE / UNKNOWN per candidate), runs the Truth Maintenance gate (SP-020) against `core_facts`, applies routing rules to land each fact in the right file, and commits the diffs as one git commit on `~/.spren/sandbox/shared/memory/`.
  - **Sync escape-valve path** (rare, gated). When the agent identifies a critical fact during a turn, it can call `commit_fact_now(fact, justification)` — hard-rail confirmation prompt to the user shows the fact + justification before commit. Subject to the same TMS gate and supersession algorithm as the consolidation path. Logs as `live_commit_event` in the session log so consolidation doesn't re-extract. Source MUST be `user_direct` — the agent cannot escape-valve facts from sub-agent observations or untrusted channels (SP-015 holds).
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

### D. Sync escape valve (rare, hard-rail-gated)

Sometimes a fact arrives mid-task that's load-bearing enough that waiting until the next consolidation pass is wrong — a critical constraint, a procedural change that affects the next action, an explicit "remember this" from the user. The agent has `commit_fact_now(fact, justification)` for this case.

Mechanics:
- **Hard rail by default** — confirmation prompt to the user shows the fact + justification before commit. User approves; agent commits. No standing approvals — escape valve always confirms.
- **Subject to the same TMS gate (§ 6 step 3) and supersession algorithm (§ 5)** as the consolidation path — escape valve doesn't bypass safety checks.
- **Source MUST be `user_direct`** (SP-015 holds — the agent cannot escape-valve facts from sub-agent observations or untrusted channels).
- **Logs as `live_commit_event`** in the session log so the next consolidation pass sees it and doesn't re-extract.

Use cases:
- "Remember that I'm allergic to penicillin" — critical constraint; can't wait.
- "The deploy procedure changed — it's now `pnpm deploy:prod` not `pnpm push`" — procedural fact that affects the next action.
- "Stop being so cautious about retries" — `PersonaFeedback` event with two-tier handling (active-session softening via `active_context.md` + queue for next consolidation; see § Persona-evolution in `09-meta-agent.md`).

NOT escape-valve cases:
- Random observations the agent extracts ("user mentioned they like Zed").
- Speculative facts ("seems like the user might prefer morning meetings").
- Anything that reasonably waits until consolidation.

The escape valve is rare-by-design. Default is async-via-consolidation. Implementer telemetry should surface escape-valve usage rate as a watch — if it's frequently invoked, that's a signal the consolidation cadence is too slow OR the agent is over-using the valve, both worth investigation.

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

## 3. Read tools — explicit retrieval, not auto-injection

The agent does NOT have a single auto-injecting memory layer. Five explicit read tools plus one gated write tool:

| Tool | Cost | Use for |
|---|---|---|
| `read_file(path)` | Cheapest | Reading `active_context.md` on session start; reading a specific known doc |
| `grep(pattern, scope)` | Cheap | Exact-match search over markdown + session log; the agent's first reach when it knows what string to look for |
| `lookup_facts(entity_id, predicate=None, time=None)` | Cheap | Typed retrieval over the SQL projection. The agent uses this when it knows the entity (which it usually does — entities are referenced by id everywhere). Returns Fact records with metadata: `{path, value, asserted_at, last_verified_at, volatility, cardinality, confidence, status, content_hash, source}`. The optional `time` parameter returns historical state at a timestamp. |
| `recall(query, k=5, time_window=None)` | Expensive | Hybrid BM25+vector+rerank over prose body text. Used only when structured/grep/lookup_facts search isn't enough; reformulatable by the agent. |
| `verify_fact(fact_id, current_context)` | Cheap | NLI check between a fact and the current context. On confirmed: refresh `last_verified_at`. On contradicted: mark `possibly_stale`. The agent uses this opportunistically when retrieved facts come back stale-flagged. |
| `confirm_with_user(fact_id, draft_phrasing)` | (sync) | Produces a draft "are you still working on Q3-launch?" message and routes through the inbox-confirmation flow per the authority-tier system in `09-meta-agent.md`. The user's response becomes the next supersession-eligible event. |
| `commit_fact_now(fact, justification)` | (sync) | The escape-valve write path. Hard-rail-confirmed; gated; runs the same TMS gate + supersession algorithm as the consolidation path. See § 2.D below. Source MUST be `user_direct` (SP-015). |

**Why explicit retrieval over auto-injection:** Letta's benchmark showed agents win when they can **reformulate queries and iterate**, which requires giving them tools they understand from training (POSIX read, grep, typed-fact lookup). Resist building a single auto-injecting "memory layer." Force the agent to retrieve explicitly. This also makes traces auditable (every retrieval is a tool call, visible in the trace).

The agent is prompted to prefer `read_file` and `grep` first, `lookup_facts` when it has the entity in hand, `recall` last. The tools are independent; the agent picks per query.

**Retrieval items carry freshness + status metadata.** Every result includes `volatility`, `last_verified_at`, and `status`. The agent's doctrine (axis #2 of the system prompt in `09-meta-agent.md`) says: "When you retrieve a fact with `status != 'active'` OR `last_verified_at` older than the volatility threshold, treat as historical context. Either re-verify with `verify_fact` OR explicitly note the uncertainty in your response. Never silently propagate stale state." Trust the agent with metadata; don't pre-chew.

`commit_fact_now` is rare-by-design. The default write path is async-via-consolidation; the escape valve exists for critical facts that can't wait. See § 2.D.

## 4. Predicate metadata — volatility, cardinality, default channel-trust

Every fact in the markdown KB carries metadata derived from its predicate. Three dimensions:

### 4.1 Volatility

| Volatility | Re-verification cadence | Examples |
|---|---|---|
| `stable` | Never (only on direct contradiction) | birthday, allergy, PII, language preference |
| `slow` | Quarterly | preferred editor, current city, primary email |
| `volatile` | Weekly or per-touch | current project, current focus, what user worked on yesterday |

**The rule:** volatility is set at extraction time based on predicate type. The user can override per-fact via the markdown frontmatter. Reads of volatile facts older than the threshold get a flag — `[possibly stale: last verified 2026-04-12]`. The agent then either re-verifies opportunistically ("Are you still working on Q3-launch?") or surfaces the uncertainty in its reasoning.

**No silent decay.** Silent decay confuses "old" with "wrong." A user's allergy from 2024 is still true in 2026. Flag rather than delete. We deliberately reject Weibull-style continuous decay (SSGM Eq. for `w(Δτ)`) for v0.3 and v0.4 — opaque to the user, harder to test, harder to debug. Re-evaluate at v0.5 if multi-user / hosted contexts emerge.

### 4.2 Cardinality

| Cardinality | Behavior on new value | Examples |
|---|---|---|
| `functional` | New value supersedes old (one active fact at a time per `(entity, predicate)`) | `birthday`, `current_employer`, `current_city`, `prefers_editor` |
| `multi_valued` | New value adds to the set (multiple active facts per `(entity, predicate)`) | `has_dog`, `is_member_of`, `attended_event`, `team` |

This dimension fixes the [Memory-R1 Buddy/Scout failure mode](https://arxiv.org/abs/2508.19828): when a user adopts a second dog, the heuristic system superseding-by-`(user, has_dog)` loses Buddy. With `has_dog` declared `multi_valued`, the supersession algorithm dispatches to the additive path and both facts coexist as `active`. Unknown predicates default to `functional` (the safer choice — a missed cardinality call produces a supersession we can audit and revert, not a silent accumulation).

### 4.3 Default channel-trust

| Source | Default trust | Treatment |
|---|---|---|
| `user_direct` | High | Bypasses queue (Path A); eligible for escape-valve path |
| `user_vim` | High | Direct file edit; honored by indexer on next reindex |
| `meta_agent_observed` | Medium | Queues to `pending_facts`; consolidation reviews |
| `sub_agent_branch_summary` | Medium | Queues to `procedural_lessons` or `pending_facts`; consolidation reviews |
| `channel_telegram` / `channel_slack` / `channel_discord` | Low | Queues; consolidation must also pass NLI gate before committing high-volatility writes |
| `webhook_inbound` | Low | Queues; non-critical writes only |

Trust factored into supersession (a low-trust fact can't auto-supersede a higher-trust one) and into the TMS gate (a low-trust contradiction of a `core_fact` is dropped silently rather than enqueued for user decision; it's noise, not signal).

### 4.4 Implementation

`src/spren/memory/predicate_metadata.py` (renamed from `volatility.py`) maps predicate → `{volatility, cardinality, default_trust_threshold}`. Starter table covers ~30 common predicates; expanded organically. Unknown predicates default to `slow / functional / medium`. The user can override per-fact via the `[volatility: …, cardinality: …]` metadata block on the fact line.

## 5. Strict supersession — the deterministic algorithm

When a new fact `f_new = (entity, predicate, value)` is committed (from explicit user, or from the consolidation pass), the system follows a **deterministic algorithm** — no LLM in the supersession path itself. LLMs propose facts; pure code commits them.

### Algorithm

```python
def commit_fact(f_new):
    # 1. Multi-valued predicate? Always additive; no supersession.
    if predicate_metadata(f_new.predicate).cardinality == 'multi_valued':
        facts.insert(f_new, status='active')
        return

    # 2. Functional predicate. Look up active fact for the same (entity, predicate).
    f_old = facts.find_active(entity=f_new.entity, predicate=f_new.predicate)

    # 3. No prior fact: just insert.
    if f_old is None:
        facts.insert(f_new, status='active')
        return

    # 4. Same value: no new row, just update last-seen timestamp.
    if f_old.value == f_new.value:
        facts.update(f_old, last_seen_at=now())
        return

    # 5. Confidence-floor check: a low-confidence fact never auto-supersedes a high-confidence one.
    if f_new.confidence < 0.6 or f_new.confidence < f_old.confidence:
        facts.insert(f_new, status='disputed', supersedes=f_old.id)
        facts.update(f_old, status='disputed', superseded_by=f_new.id)
        enqueue_user_decision(f_old, f_new, kind='confidence')
        return

    # 6. Stable fact, non-user source — refuse to silently auto-replace.
    if f_new.volatility == 'stable' and f_new.source != 'user_direct':
        facts.insert(f_new, status='disputed', supersedes=f_old.id)
        facts.update(f_old, status='disputed', superseded_by=f_new.id)
        enqueue_user_decision(f_old, f_new, kind='stable')
        return

    # 7. Trust-level check: low-trust fact cannot auto-supersede a higher-trust one.
    if trust_level(f_new.source) < trust_level(f_old.source):
        facts.insert(f_new, status='disputed', supersedes=f_old.id)
        facts.update(f_old, status='disputed', superseded_by=f_new.id)
        enqueue_user_decision(f_old, f_new, kind='trust')
        return

    # 8. Standard supersession: old becomes 'superseded', new becomes 'active'.
    facts.insert(f_new, status='active', supersedes=f_old.id)
    facts.update(f_old, status='superseded', superseded_by=f_new.id)
```

Status values: `active` (current truth), `superseded` (replaced; kept for audit), `disputed` (conflicting facts awaiting user — kind: `stable | confidence | trust | implicit | tms`), `tombstoned` (explicitly forgotten by user; never re-extract), `orphaned` (parent file removed it without explicit tombstone — needs reconciliation).

**Note** — this algorithm runs *after* the consolidation pass's CUPMem four-state adjudication (KEEP / STALE / REPLACE / UNKNOWN) and Truth Maintenance gate (§ 6). The supersession algorithm consumes verdicts; it doesn't re-derive them. See § 6 for the upstream pipeline.

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

**Trigger:** `(24h elapsed AND ≥5 sessions accumulated)` OR `(≥1 PersonaFeedback queued AND ≥1 hour elapsed)`. Both clauses guard against opposing failure modes — the first against quiet-day waste, the second against same-conversation knee-jerk persona changes. Auto Dream's pure-time and pure-session-count triggers both fail in different ways.

**Pipeline (6 stages, runs in idle hours, by a cheaper model — see § 10 model floor):**

1. **Inventory** — list all current markdown files; hash them; load `pending_facts` since last run; load sub-agent `BranchSummary.procedural_lessons` from completed instances since last run; load `pending_persona_changes` queue.
2. **Extract & adjudicate** — LLM pass over the day's session-log delta + pending_facts. For each candidate fact, the LLM explicitly outputs a CUPMem four-state verdict: `KEEP existing | STALE existing | REPLACE existing | UNKNOWN`. The supersession algorithm consumes these verdicts in stage 4 rather than re-deriving them. This addresses the [STALE / LightMem failure mode](https://arxiv.org/abs/2605.06527) — agents retrieve new evidence in 77% of cases but adjudicate only 3%; explicit four-state output forces the adjudication.
3. **Truth Maintenance gate** (SP-020). For each candidate fact, run an NLI classifier against `core_facts` — the protected belief set, defined as facts in `profile/` with `volatility: stable` AND `source: user_direct`. On contradiction: candidate enters `disputed` state with `kind: tms`, `enqueue_user_decision()` fires, candidate does NOT auto-commit. Cost: ~50 facts to check at typical scale, one small-model call per candidate, runs offline so latency is irrelevant.
4. **Routing** — apply the bubble-up routing rules (§ 8.2) to land each fact in the right entity / category file. Lookup table; no LLM.
5. **Conflict resolution** — for each candidate fact that survived TMS, run the supersession algorithm in § 5 (cardinality dispatch + confidence floor + stable-source check + trust-level check). On disputed: enqueue user decision, do not auto-commit.
6. **Apply & log** — write the diffs as one **git commit on `~/.spren/sandbox/shared/memory/`**. Free version control, free audit trail, free rollback. Compute `content_hash: sha256(timestamp + source_event_id + content)` for every committed fact (SP-022 foundational defense; verified at every indexer read). Emit a `ConsolidationCompleted` event into the meta-agent's inbox with `tokens_in`, `tokens_out`, `cost_usd`, `facts_committed`, `facts_disputed`, `duration_ms` (SP-010).

**Critical rules:**
- **Always consolidate the raw delta only** — never re-process already-consolidated markdown. Re-summarizing already-summarized material erodes facts (the "transcript replay drift" failure).
- **Every consolidated fact must cite a `session_id:event_id` from the session log.** Reject facts without provenance — that's the only defense against consolidation hallucination.
- **Sub-agent `procedural_lessons` get a higher commit threshold** than semantic facts: a procedural lesson must appear in N≥2 successful trajectories before it commits as a procedure (defense against [MemoryGraft](https://arxiv.org/abs/2512.16962) experience-poisoning at the consolidation layer; SP-022 reinforces with retrieval-time defense in v0.4+). The threshold is low (N=2 fits a single-user system); community-skill systems would need higher.
- **Procedural lessons land in the SKILL.md format** — see § Procedural memory below.
- **Rebuild the discovery index** at the end of consolidation (or schedule for next idle) so it reflects the new KB.

**Failure modes (from the literature, plan for these):**
- *Too frequent (hourly):* premature compression destroys context that turns out to matter the next day. Token cost balloons.
- *Too rare (weekly+):* `pending_facts` queue grows past the LLM's context, you have to chunk, and chunked consolidation produces conflicting summaries.
- *Consolidation hallucination:* mitigated by required provenance + TMS gate.
- *Drift through repeated re-summarization:* mitigated by delta-only rule.
- *Format errors at write boundary:* per [Anatomy of Agentic Memory](https://arxiv.org/html/2602.19320), 1.2-30.4% rates across systems; mitigated by the format validator at the indexer (see § 11).
- *Implicit conflict missed at adjudication:* STALE (best frontier 55.2%) shows this is hard. The TMS gate catches stable-fact contradictions; the v0.4 staleness watcher (note for later) catches volatile-fact implicit conflicts via periodic NLI sweep over recent session-log windows.

## 7. Memory poisoning — defenses

[MINJA](https://openreview.net/forum?id=QVX6hcJ2um) and [MemoryGraft](https://arxiv.org/html/2512.16962v1) demonstrate >95% memory-injection success against production agents via query-only interaction. We MUST plan for this from the first Spren release (v0.3).

**Defenses, all in our model:**

- **Queue-then-consolidate** (not live write) — already covered in § 2. A poisoned message can't immediately corrupt the KB. (SP-015.)
- **Per-channel trust scoring** — facts originated from the user terminal score higher than facts from a Slack DM, which score higher than facts from inbound webhook. Trust score factored into supersession (a low-trust fact can't auto-supersede a higher-trust one) and into the TMS gate (a low-trust contradiction of a `core_fact` is dropped silently rather than enqueued for user decision).
- **Provenance metadata on every fact** — `source: <channel-id>:<event_id>`, `extracted_by: <agent-id>`, `confidence: 0.x`.
- **Hash-anchored provenance** — every committed fact and procedure carries `content_hash: sha256(timestamp + source_event_id + content)`. Indexer verifies on read; mismatch surfaces as `IndexDriftDetected` (already in our watcher catalog). Cheap MemoryGraft defense — full cryptographic signing (Cryptographic Provenance Attestation) is overkill for local single-user; hashes are sufficient against in-band attackers.
- **Truth Maintenance gate at consolidation** (SP-020) — NLI check against `core_facts` before commit; contradictions enter `disputed` state and require user decision. Defends against consolidation-hallucination class of failures.
- **Consolidation pass acts as a second-look** — sees the full context of where the candidate fact came from; can reject suspicious ones.
- **User-visible audit log** — `spren memory why <claim>` shows the supersession chain back to source events. Lets the user spot poisoned facts and `forget` them.
- **Experience-poisoning defenses for v0.4+** (SP-022) — when sub-instances start producing `BranchSummary.procedural_lessons`, the procedure-retrieval path runs a safety reranker; failed-procedure tracking via `quarantined: true` frontmatter; trajectory-level review at consolidation against `<data-dir>/sandbox/shared/security/poisoning_patterns.yaml`. v0.3 ships the foundational defense (hash-anchored provenance) and reserves the patterns file (empty in v0.3, seeded in v0.4).

**SP-015** (Untrusted-channel writes never touch memory live) and **SP-020** (Core-fact contradictions never auto-commit) are the load-bearing memory-security principles. **SP-022** reserves the v0.4+ procedure-retrieval defense.

## 8. Multi-agent memory — bubbling up, with explicit routing

When sub-agents (working instances or team managers' subordinates) do work in parallel, their memory must NOT pollute the global KB directly. Pattern:

- Each sub-agent gets a **read-only snapshot** of relevant Tier 3 facts at spawn (filtered by task scope), plus their own Tier 1 working memory and a Tier 2-equivalent local log.
- They do their work in their scratchpad.
- On completion they emit `BranchSummary` (see § 2.C) including `new_facts_proposed`, `procedural_lessons`, `outstanding_todos`, `open_questions`.
- The parent agent reads the summary, reasons about it. Facts are routed through queue-then-consolidate (SP-015 holds — sub-agent outputs are not user-direct).

**Avoid mesh-style shared memory** between sub-agents. The [multi-agent memory survey from a computer architecture perspective](https://arxiv.org/abs/2603.10062) is explicit: "multi-agent memory consistency has not been formally defined." Until the field has a coherence model, our pattern is the safe default. A later release may add cross-team-manager direct queries if patterns emerge.

### 8.1 Role-aware retrieval (LEGOMem allocation)

[LEGOMem (AAMAS 2026)](https://arxiv.org/abs/2510.04851) showed that hierarchical agent systems gain 12-13pp on task benchmarks when memory is allocated *by role* rather than treated as a flat blob: orchestrator gets full-task memories, workers get subtask memories. Our main-agent / team-manager / working-instance hierarchy is exactly the shape LEGOMem optimizes. Default scopes:

- **Main agent's `recall` scope:** `shared/memory/` (RW) + `procedures/main/` (full-task procedural patterns).
- **Working instance's `recall` scope:** `shared/memory/` (R only) + own scratchpad (RW) + `procedures/<role>/` (subtask procedural patterns) — where `<role>` is the spawning skill's declared role.
- **Team manager's `recall` scope:** team memory (RW) + `procedures/<team>/` + `shared/memory/` (R).

The role-aware scope is an *attribute of the retrieval call*, not a hard partition — main agent CAN read a working-instance's scratchpad when investigating a failure (`09-meta-agent.md` § Sub-instance return contract grants this), but the *default* scope is role-narrow.

### 8.2 Bubble-up routing rules

When a working instance produces a `BranchSummary`, the consolidation pass routes facts by predicate-class to file-path via a small static lookup table. **No LLM in the routing decision.** This is the single biggest reduction in consolidation drift across runs.

| Predicate class | Routes to |
|---|---|
| User identity / demographics | `profile/identity.md` |
| Hard rails (medical, legal, financial, accessibility) | `profile/constraints.md` (sensitive: true) |
| Ethics / stances / values | `profile/values.md` |
| Operational preferences (tooling, comm, decision-style) | `profile/preferences.md` |
| Skills / expertise / capabilities | `profile/capabilities.md` |
| Daily / weekly patterns / locations | `profile/rhythm.md` |
| Goals at any horizon | `profile/goals.md` |
| Person facts | `people/p-<slug>.md` |
| Organization facts | `people/org-<slug>.md` |
| Project facts | `projects/proj-<slug>.md` |
| Procedural lessons (v0.4+) | `procedures/<slug>/SKILL.md` |
| Decisions made together | `decisions/dec-ADR-<n>-<slug>.md` |
| Episodic events | `journal/YYYY-MM-DD.md` |

Unknown predicate-class falls through to `journal/YYYY-MM-DD.md` (the safe default — episodic prose that the consolidation pass can re-categorize later).

### 8.3 Stated-vs-observed contradictions

When observed behavior contradicts a user-direct preference, we surface rather than auto-resolve. Mechanism: stated preferences land in `profile/preferences.md` with `source: user_direct`. Observed contradictions accumulate in `pending_facts` with `source: observed, contradicts: pref-id-X`. After N=3 contradicting observations within the freshness horizon, the consolidation pass surfaces it as a note in `active_context.md`: "Observed: user has rescheduled morning meetings 4 times in the last 2 weeks despite stated preference for them." The agent then either raises it conversationally during the next heartbeat ("Hey — I noticed you keep rescheduling morning meetings. Want me to switch the default?") OR adds a per-fact override ("user states they prefer morning meetings *but in practice* schedules them later"). The user reconciles. We do NOT silently flip the user's stated preference based on observed behavior — Spren's design prefers conversation over silent behavioral inference (SP-011 + SP-012 + the broader "the agent doesn't act unilaterally" stance in `09-meta-agent.md`).

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
- **Format-validated at the write boundary.** The indexer rejects malformed fact-blocks and YAML frontmatter; rejects log to `<data-dir>/sandbox/logs/consolidation_errors.md` with the raw input for diagnosis. Without this, weaker backbones (which ship with v0.4+ when sub-instances run on cheap models) silently corrupt the KB at 1.2-30.4% rates per [Anatomy of Agentic Memory](https://arxiv.org/html/2602.19320). The format validator runs *before* commit; malformed fact-blocks never land in markdown.
- **Hash-anchored on commit, hash-verified on read.** Every fact carries `content_hash: sha256(...)`; the indexer recomputes on every reindex and surfaces mismatches as `IndexDriftDetected`. Detects in-band tampering (a process that wrote to the markdown without going through the consolidation pass).

### Embedding model versioning

Every embedded chunk in the discovery index (Tier 4) carries `embedding_model_id`. v0.3 ships with one model; tagging is free. When v0.4 or v0.5 rotates the embedding model, `spren memory rebuild-embeddings` re-embeds chunks tagged with the old model_id without a schema change. This is the SSGM "reversible reconciliation" pattern applied at the embedding layer.

### Drift detection

`spren memory verify-index` command walks every fact in SQLite and confirms its source file still contains a matching block AND that the `content_hash` recomputes correctly. Any drift is reported (not auto-fixed; user reviews and decides). Defends against parser bugs, edge cases, and in-band tampering.

### What does NOT use the indexer

- The Tier 4 vector index for semantic discovery is built from prose body text, not from fact blocks. That path uses embedding API calls. Cheap, infrequent, separate from the structured-fact indexer.
- The session log (Tier 2) is written directly by the runtime; the indexer doesn't touch it.

### Why no LLM in this path

The user must be able to predict what their markdown edits will produce in the index. An LLM in the loop would mean an edit might or might not be indexed depending on the model's mood. Pure parsing makes the contract explicit: this YAML/fenced syntax → this SQL row. Nothing else.

LLMs are involved upstream — when proposing facts during extraction, when consolidating, when interpreting prose for the discovery index. But the markdown→SQL pipeline is bedrock: deterministic, debuggable, testable.

## 12. Procedural memory — markdown today, SKILL.md from v0.4

v0.3 ships procedural memory as a markdown subdirectory: `procedures/proc-<slug>/SKILL.md`. v0.4 starts producing curated procedural patterns from sub-agent `BranchSummary.procedural_lessons`. The format choice — Anthropic's [Agent Skills SKILL.md spec](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) (Oct 2025, OpenAI-adopted Dec 2025) — is committed now, before skills ship, so v0.4 implementers don't redesign the format and so future curated / community skills work without translation.

### 12.1 The format

```
procedures/
└── proc-<slug>/
    ├── SKILL.md              # entry point, YAML frontmatter + markdown body
    ├── scripts/              # optional executable scripts the skill invokes
    └── references/           # optional reference files
```

`SKILL.md` follows the Anthropic spec — required `name` and `description` in YAML frontmatter, markdown body. We extend the frontmatter with Spren-specific fields:

```yaml
---
name: workflow-inspection
description: Diagnose a failing workflow run; identify failure-class; suggest a fix. Use when a run has failed, a run is stuck, or a recurring failure pattern is suspected.
required_tools: [read_run, read_trace, grep]
recommended_models: [anthropic/claude-opus-4-7, openai/gpt-4o]
volatility: slow
validated_on_tasks: ["debug-failed-run", "investigate-stalled-run"]
cross_model_tested: true
failure_count: 0
last_failure_at: null
quarantined: false             # SP-022 — true if a v0.4 retrieval led to a hard-rail violation
schema_version: 1
---

# Workflow Inspection

(skill body in markdown — instructions, references, scripts)
```

`description` doubles as the trigger criterion for retrieval ("use when X"). Progressive disclosure: metadata always loaded (~100 tokens / skill), body loaded on activation (under 5K), bundled files loaded as needed.

### 12.2 What ships when

| Version | Procedural memory state |
|---|---|
| v0.3 | Format committed (SKILL.md). Zero curated procedures shipped. Procedures can be authored manually if the user wants. No retrieval-time safety reranker yet. |
| v0.4 | Curated catalog ships (Anthropic + Spren-curated). Sub-instances start producing `BranchSummary.procedural_lessons`; consolidation creates `SKILL.md` files from the lessons (commit threshold N≥2 successful trajectories). Retrieval-time safety reranker (SP-022) ships. `quarantined: true` lifecycle on hard-rail-violating procedures. |
| v0.5 | User-authored skills (their own install, their own risk; sandboxed via the OS-level outer envelope). Consolidation runs hierarchical Trace2Skill-style consolidation across procedural lessons. Failure-derived counterfactuals à la ReasoningBank. |
| v0.6+ | Community-skill installation (gated by SP-021 — four-tier permission model from the [Agent Skills survey](https://arxiv.org/abs/2602.12430)). |

### 12.3 Why SKILL.md and not a bespoke YAML

- **Cross-vendor.** A skill written for Spren works in Claude Code, the Anthropic API, OpenAI surfaces. Portability the user cares about.
- **Trace2Skill / Memp / LEGOMem all serialize naturally into this format.** No translation layer needed when consuming research-derived skill structures.
- **Open standard.** OpenAI adopted in Dec 2025; large changes are unlikely; small changes are fine because we own our skills.
- **The cost of choosing wrong scales with usage.** Trivial cost now (zero shipped procedures); real cost once users author skills.

We extend the frontmatter beyond Anthropic's required `name` + `description`. The Anthropic spec allows this; the extensions are Spren-specific operational metadata the open spec doesn't address.

## 13. Open questions for next round

- **Embedding model drift over years.** v0.3 ships `embedding_model_id` tagging; v0.4 ships `spren memory rebuild-embeddings`. Document the migration cadence (annual?).
- **Concrete trust-score computation per channel.** Manual config for v0.3 (settings UI per channel). v0.4 may add heuristic learning ("how often do facts from this channel get user-corrected"). Learned over time vs heuristic-based — open.
- **Cross-device sync** isn't addressed. If Spren ever runs on a phone too, the markdown layer makes sync trivial (git/Syncthing); the SQLite layer needs CRDT or single-writer discipline. v0.4+ territory.
- **The `pending_facts` queue size** — what's the threshold at which we trigger consolidation early (independent of the 24h+5-session rule)? Probably token-count-based: if pending_facts would exceed half the consolidation model's context, trigger now.
- **What goes in the discovery index from the session log** — full text of every event would balloon. Probably: every user message + every BranchSummary + every consolidated journal entry. Other event types (heartbeat ticks, tool calls) stay in the session log only.
- **Volatility threshold instrumentation.** The default Weibull-style table values (in §4 and the predicate-metadata starter table) are engineering judgment. Once Spren has 90+ days of dogfood data, re-tune from observed re-verification outcomes.
- **Cardinality table coverage.** The starter `functional | multi_valued` table covers ~30 common predicates. Expand organically. Decide whether to introduce richer cardinality (`set`, `ordered_list`, `time_series`, `count`) when patterns emerge — or stay binary.
- **Live (non-consolidation) NLI adjudication path.** Currently the TMS gate only runs at consolidation. v0.4+ may add live adjudication for select high-stakes write paths (escape valve already runs TMS; live extraction does not). Worth a re-evaluation pass after v0.3 ships.

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

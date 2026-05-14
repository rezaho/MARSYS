# Spren Session 08 — Meta-Agent Capabilities + Memory Write Paths

> Session plan. The implementer reads this as the primary source of truth for what Session 08 ships, how the read + write tool catalog + supersession algorithm + TMS gate + CUPMem 4-state adjudication + consolidation pass + persona-evolution mechanism + bond CLI + suggest-with-confirm flow + hard rails + budget caps fit together. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / MODIFY, the user journeys that close out Bundle 03's demo gate, the locked decisions, polish items, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./08-meta-agent-capabilities/acceptance.md`](./08-meta-agent-capabilities/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Architectural anchors (read before coding):
- [`../../../../architecture/spren/10-memory-architecture.md`](../../../../architecture/spren/10-memory-architecture.md) — full memory architecture with the May-2026 deltas applied. §2 write paths (incl. §2.D escape valve), §3 read tools, §4 predicate metadata, §5 supersession algorithm with cardinality dispatch, §6 6-stage consolidation pipeline (TMS gate + CUPMem 4-state output), §7 poisoning defenses, §8 multi-agent memory + routing rules, §11 indexer, §12 procedural memory.
- [`../../../../architecture/spren/09-meta-agent.md`](../../../../architecture/spren/09-meta-agent.md) — §"Authority tiers", §"Cost ceiling enforcement at the runtime level", §"Persona-evolution mechanism", §"Bond-violation handling".
- [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-012 (hard rails always confirm), SP-013 (cost ceiling load-bearing), SP-015 (untrusted-channel writes never live-touch memory), SP-016 (markdown source of truth), SP-017 (forget tombstones never deletes), SP-020 (TMS gate), SP-022 (experience-poisoning defenses for v0.4+), SP-023 (generic-runtime vs Spren-specific boundary).
- [`../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md`](../../../../../tmp/spren/research/06-memory-foundations/00-synthesis.md) — M1-M18 must-do deltas + S1-S5 should-do.
- [`../../../../../tmp/spren/research/06-memory-foundations/01-memory-architecture-deltas.md`](../../../../../tmp/spren/research/06-memory-foundations/01-memory-architecture-deltas.md) — May-2026 deltas with citations.
- [`../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md`](../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md) — bond mechanic, persona-evolution mechanism design.
- [`../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md`](../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md) — per-archetype voice + bond-violation specifics.
- [`./06-memory-foundation.md`](./06-memory-foundation.md), [`./07-meta-agent-core.md`](./07-meta-agent-core.md) — the foundations Session 08 builds on.

---

## 1. Bundle position + tier

- **Bundle**: 03 — Meta-agent becomes alive (Sessions 06 + 07 + 08). Session 08 closes the bundle. After Session 08 ships, the meta-agent reads workflows + traces + memory, suggests-with-confirm on writes, dispatches workflows, respects hard rails, runs the consolidation pass that rewrites the markdown KB on its own cadence, evolves its persona through the bond mechanism. The orb breathes; the bond starts to tune.
- **Session 08 scope**: the agent's *capabilities*. Read tool catalog (read_file, grep, lookup_facts, recall, verify_fact, confirm_with_user). Write tool catalog (commit_fact_now, set_active_context, add_run_note, create_workflow, update_workflow, archive_workflow, run_workflow). Hard-rail tool list. Suggest-with-confirm flow. Per-day budget cap + per-think token cap. Supersession algorithm with cardinality dispatch + confidence floor + trust-level checks. CUPMem 4-state adjudication at consolidation. TMS gate against core_facts (SP-020). Bubble-up routing rules. Hash-anchored provenance. Format validator at write boundary. Stated-vs-observed surfacing. Consolidation pipeline (6 stages) with PersonaReflection stage. Persona-evolution CLI (spren persona log/why/diff/approve/reject/revert). Bond reset CLI (spren bond reset). Voice-drift consumption.
- **Tier**: CRITICAL. The supersession algorithm is forward-mode (a bad commit can't be unwritten); the TMS gate + CUPMem adjudication are the load-bearing memory-correctness defenses; the persona-evolution mechanism touches the bond integrity. Full meta-process pipeline.
- **Approval gate**: peer Stage 0 conversation already complete (this brief). Researcher + Designer + Validator/Critic + Fact-checker + Synthesis. Multi-checkpoint user review per CRITICAL tier.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 06 (memory foundation) | ships first | Sandbox + session log + KB scaffold + indexer + pending_facts queue + memory CLI. Session 08 hard-depends. |
| Spren Session 07 (meta-agent core + persona) | ships before 08 | Daemon + inbox + scheduler + watchers + heartbeat + main agent loop + 6-axis system prompt + 5-archetype onboarding + execute_shell + voice-drift detector. Session 08 hard-depends. |
| Spren Session 02 (workflow CRUD) | shipped | The `create_workflow`, `update_workflow`, `archive_workflow` write tools wrap Session 02's REST endpoints. |
| Spren Session 04 (run execution) | shipped | The `run_workflow` write tool wraps `POST /v1/runs`. The `read_run`, `read_trace` read tools wrap Session 04 + Session 05 endpoints. |
| Spren Session 05 (run inspection) | shipped | `GET /v1/runs/{id}/trace` is consumed by `read_trace`. |
| `marsys.coordination.event_bus.EventBus` | live | The `run_workflow` tool subscribes to EventBus during dispatched runs to track success / failure. |

Session 08 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). Adds two further subpackages: `spren/runtime/memory/` (generic supersession + adjudication primitives — yes, supersession has generic shape) and `spren/agent/capabilities/` (Spren-specific tool catalog + consolidation pipeline + persona-evolution mechanism).

## 3. SP-023 boundary — what goes where

**Generic always-on runtime additions — `spren/runtime/memory/`:**

> **Pipeline shape locked at user redirect (2026-05-14)**: every adjudication step is a cheap LLM call. No specialized ML rerankers, no local NLI cross-encoders. Embedding model (FastEmbed BGE-small) is permitted *only for retrieval*, not for relevance scoring. Concrete file list below revised; the precise rerank+adjudicate pipeline is the subject of [`tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md`](../../../../../tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md) which the implementer reads as primary source.

- `runtime/memory/supersession.py` — `SupersessionEngine` ABC + the deterministic algorithm. Takes a fact + a fact store + predicate metadata; returns `[Mutation]`. Generic shape — the algorithm in `10-memory-architecture.md` §5 doesn't depend on Spren-specific event types.
- `runtime/memory/tms_gate.py` — `TMSGate.check(candidate_fact, core_facts) -> Decision`. **Implementation: cheap-LLM call** (Haiku 4.5 / GPT-5.4-mini class) per (candidate, core_fact) pair OR a single batched call over all core_facts; the research output picks. Returns DISPUTE-with-kind=tms when the LLM classifies a contradiction against any core fact above threshold.
- `runtime/memory/retrieval/__init__.py` — `MemoryRetriever` interface (the four-query hybrid). Generic shape; concrete Spren implementation in `agent/capabilities/`.
- `runtime/memory/retrieval/rrf.py` — Reciprocal Rank Fusion (Cormack 2009). Pure code; ~30 lines. `rrf_fuse(ranked_lists, k=60) -> ranked_list`.
- `runtime/memory/retrieval/hyde.py` — `HyDEExpander.expand(candidate, llm) -> list[str]`. Generic; one LLM call returns 3 hypothetical contradiction sentences for embedding. The prompt template is parameterized.
- `runtime/memory/retrieval/llm_rerank.py` — `LLMReranker.rerank(candidate, retrieved_facts, llm) -> ranked_list`. **Replaces the prior NLI cross-encoder rerank.** Listwise rerank prompt: one LLM call ranks N retrieved facts by relevance + relation-type to the candidate. Implementation pattern (listwise vs pairwise vs pointwise) decided by the research output.
- `runtime/memory/budget.py` — `BudgetEnforcer`. Generic; consumes `CostMeter` from Session 07 and the agent's tool registry.
- `runtime/memory/agent.py` — `MemoryUpdateAgent` ABC. The per-candidate agentic loop (extract → TMS → retrieve → llm-rerank → adjudicate → apply). Generic shape; subclassed in Spren-specific code with concrete prompts + retrieval implementation.
- `runtime/memory/consolidation.py` — `ConsolidationPipeline` ABC. The 6-stage pipeline shape (inventory → extract → memory-update-agent-loop → routing → commit → reindex) is generic; concrete Spren stages subclass.

**Files explicitly removed from earlier draft (2026-05-14 redirect): `runtime/memory/nli.py` (DeBERTa wrapper), `runtime/memory/cove.py` (CoVe primitive depended on NLI-disagreement trigger which no longer exists). The fully-LLM pipeline replaces both.**

**Spren-specific layer — `spren/agent/capabilities/`:**
- `capabilities/__init__.py` — module init.
- `capabilities/tools/__init__.py` — Spren's full tool registry. Imports the read tools, write tools, hard-rail wrappers.
- `capabilities/tools/read_file.py` through `recall.py` — six read tools, one file each. Wrap memory's `Sandbox.read_file`, indexer's `lookup_facts`, etc.
- `capabilities/tools/verify_fact.py`, `confirm_with_user.py` — the freshness + re-verification tools.
- `capabilities/tools/commit_fact_now.py` — escape-valve write tool. Hard-rail-confirmed; runs through Session 06's write path with TMS gate + supersession.
- `capabilities/tools/set_active_context.py`, `add_run_note.py` — low-risk write tools. NOT hard-rail (under standing-approvals umbrella).
- `capabilities/tools/create_workflow.py`, `update_workflow.py`, `archive_workflow.py` — workflow write tools. Each wraps Session 02's REST endpoints. `archive_workflow` is NOT hard-rail (archive is reversible); `delete_workflow` is hard-rail (different tool).
- `capabilities/tools/run_workflow.py` — the bridge to Marsys's `Orchestra.run()`. Spawns a workflow run; subscribes to EventBus for in-flight observability; returns `OrchestraResult` to the agent's next turn.
- `capabilities/tools/hard_rails.py` — the hard-rail tool list + `IsHardRail` predicate. Tools: `delete_workflow`, `revoke_channel`, `modify_settings`, `forget_fact` (different from `tombstone_fact`; the latter is the per-architecture default), `execute_shell`, `commit_fact_now`, anything that ends a run, anything that touches secrets, any spend over per-action budget cap.
- `capabilities/supersession.py` — Spren's concrete `SupersessionEngine`. Consumes generic engine + `predicate_metadata.py` (Session 06) + memory storage (Session 06).
- `capabilities/consolidation/__init__.py` — Spren consolidation pipeline.
- `capabilities/consolidation/pipeline.py` — Spren's concrete `ConsolidationPipeline` (subclass of generic). Wires the stages.
- `capabilities/consolidation/stages/inventory.py` — Stage 1: hash files, load `pending_facts`, load `BranchSummary.procedural_lessons`, load `PersonaFeedback`.
- `capabilities/consolidation/stages/extract.py` — Stage 2: per-turn-pair extraction LLM call (Mem0-style `FACT_RETRIEVAL_PROMPT` adapted). Returns list of `CandidateFact`.
- `capabilities/consolidation/stages/agentic_loop.py` — Stage 3: the per-candidate Memory Update Agent loop. For each candidate: TMS gate (cheap-LLM) → retrieve (4-query hybrid + RRF + cheap-LLM rerank) → adjudicate (1 LLM call with per-fact classification + aggregate operation) → supersession.
- `capabilities/consolidation/stages/routing.py` — Stage 4: bubble-up routing rules table application.
- `capabilities/consolidation/stages/commit.py` — Stage 5: git commit + content_hash + format-validate + `ConsolidationCompleted` event emission.
- `capabilities/consolidation/stages/reindex.py` — Stage 6: incremental MarkdownIndexer + Tier 4 update.
- `capabilities/memory_agent.py` — Spren-specific `SprenMemoryUpdateAgent`. Concrete subclass of `runtime/memory/agent.MemoryUpdateAgent`. Wires Spren's retrieval (entity-aware Q1, FTS5 Q2, sqlite-vec Q3, HyDE Q4), the Memory Update Agent prompt, and the supersession integration.
- `capabilities/retrieval.py` — Spren's concrete four-query hybrid retrieval. Composes `runtime/memory/retrieval/*` primitives with Spren-specific schemas (entities, facts, predicate clusters from `predicate_metadata.py`).
- `capabilities/data/predicate_clusters.yaml` — predicate-cluster lookup table for Q1 expansion. ~10 starter clusters (e.g., `lives_in` clusters with `current_city`, `time_zone`, `current_employer`; `birthday` clusters with `age`, `birth_year`). User-editable.
- `capabilities/data/extraction_prompt.txt` — Mem0-style `FACT_RETRIEVAL_PROMPT` adapted for our schema (per-fact entity_id + predicate + value + asserted_at).
- `capabilities/data/memory_update_agent_prompt.txt` — the adjudicator prompt with per-fact classification + aggregate operation output.
- `capabilities/data/hyde_prompt.txt` — the HyDE expander prompt (3 hypothetical contradiction sentences per candidate).
- `capabilities/data/llm_rerank_prompt.txt` — the cheap-LLM listwise rerank prompt.
- `capabilities/data/tms_gate_prompt.txt` — the cheap-LLM TMS-gate prompt (structured output: `{contradicts, contradicting_fact_id, confidence}`).
- `capabilities/data/persona_judge_prompt.txt` — the LLM-as-judge bond-integrity rubric for persona-evolution proposals.
- `capabilities/consolidation/persona_reflection.py` — the PersonaReflection stage that runs alongside the consolidation pipeline. Two-prompt reflection (salient questions → minimal-diff proposals). Outputs go to `pending_persona_changes/`.
- `capabilities/consolidation/scheduler.py` — wires consolidation triggering: `(24h ∧ ≥5 sessions) OR (≥1 PersonaFeedback queued ∧ ≥1h delay)`. Subscribes to inbox events to track session count + feedback queue.
- `capabilities/cli/persona.py` — `spren persona` CLI (log, why, diff, approve, reject, revert).
- `capabilities/cli/bond.py` — `spren bond reset` CLI.
- `capabilities/suggest_with_confirm.py` — the suggest-with-confirm UX flow. Wraps a tool execution: agent emits a draft + reasoning; surfaces as a `NeedsDecision` event; user approves via API → tool executes; standing approvals checked first.
- `capabilities/budget.py` — Spren's per-day budget cap + per-think token cap. Composes generic `BudgetEnforcer` + Spren's settings.
- `capabilities/api.py` — additional FastAPI routes: `POST /v1/agent/persona/proposals/{id}/approve`, `POST /v1/agent/persona/proposals/{id}/reject`, `POST /v1/agent/bond/reset` (with hard-rail confirmation), `GET /v1/agent/bond/status`.

SP-023 enforcement test extends to cover all new files: `tests/sp023_boundary.py` from Sessions 06 + 07.

## 4. What ships in Session 08

### 4.1 Read tool catalog (six tools)

Each tool is a Pydantic-typed function that the agent's tool registry exposes. Implementations consume Session 06's storage + indexer.

```python
# capabilities/tools/read_file.py
class ReadFileArgs(BaseModel):
    path: str  # path within sandbox

class ReadFileResult(BaseModel):
    content: str  # tail-truncated to 16K chars (configurable)
    truncated: bool
    last_modified: float

# Composes Session 06's Sandbox; respects per-agent permission tiers.
```

```python
# capabilities/tools/grep.py
class GrepArgs(BaseModel):
    pattern: str
    scope: Literal["all", "memory", "session_log", "runs", "current_project"] = "all"
    max_results: int = 50

class GrepResult(BaseModel):
    matches: list[GrepMatch]  # {path, line, col, context}
    truncated: bool

# Wraps `ripgrep` if available, falls back to Python `re` over the relevant scope.
```

```python
# capabilities/tools/lookup_facts.py
class LookupFactsArgs(BaseModel):
    entity_id: str
    predicate: str | None = None  # None = all predicates for this entity
    time: datetime | None = None  # historical state at this timestamp

class FactRecord(BaseModel):
    id: str
    entity_id: str
    predicate: str
    value: str
    asserted_at: datetime
    last_verified_at: datetime | None
    volatility: Literal["stable", "slow", "volatile"]
    cardinality: Literal["functional", "multi_valued"]
    confidence: float
    status: Literal["active", "superseded", "disputed", "tombstoned", "orphaned"]
    content_hash: str
    source: str
    freshness_weight: float  # computed; 1.0 = fresh, 0.0 = ancient

class LookupFactsResult(BaseModel):
    facts: list[FactRecord]
```

```python
# capabilities/tools/recall.py
class RecallArgs(BaseModel):
    query: str
    k: int = 5
    time_window: tuple[datetime, datetime] | None = None

class RecallSnippet(BaseModel):
    path: str
    snippet: str
    asserted_at: datetime
    volatility: str
    citation: str
    freshness_weight: float

class RecallResult(BaseModel):
    snippets: list[RecallSnippet]

# v0.3 implementation: BM25 over FTS5 (Session 06's NoopDiscoveryIndex);
# v0.4 swaps to hybrid BM25+vector+rerank.
```

```python
# capabilities/tools/verify_fact.py
class VerifyFactArgs(BaseModel):
    fact_id: str
    current_context: str  # the context the agent is currently reasoning over

class VerifyFactResult(BaseModel):
    verdict: Literal["confirmed", "contradicted", "neutral"]
    reasoning: str

# Calls a small LLM (cheap model) to NLI-check the fact against current context.
# On confirmed: refresh last_verified_at. On contradicted: mark possibly_stale.
```

```python
# capabilities/tools/confirm_with_user.py
class ConfirmWithUserArgs(BaseModel):
    fact_id: str
    draft_phrasing: str  # the message Spren sends the user

class ConfirmWithUserResult(BaseModel):
    confirmation_id: str  # the message ID the user can respond to

# Routes through Session 07's NeedsDecision flow. The user's response becomes the next supersession-eligible event.
```

All six are registered in the tool registry on daemon startup and exposed via OpenAI-style tool schemas to the LLM.

### 4.2 Write tool catalog

```python
# capabilities/tools/commit_fact_now.py
# Hard-rail. Gated by user confirmation. Source MUST be user_direct.
class CommitFactNowArgs(BaseModel):
    fact: FactDraft  # {entity_id, predicate, value, volatility, source}
    justification: str  # why this can't wait until consolidation

# Pre-flight: TMS gate (SP-020). If contradicts core_facts → reject with 409.
# Confirmation flow: surfaces fact + justification to user; user approves.
# On approval: writes to markdown via the same indexer/parser path as consolidation;
# logs as live_commit_event in session log; supersession algorithm runs.
```

```python
# capabilities/tools/set_active_context.py
# Not hard-rail; covered by standing approval on default policy.
class SetActiveContextArgs(BaseModel):
    operation: Literal["replace", "append", "compress"]
    content: str  # the new active_context.md content (replace) or fragment (append)

# Edits <sandbox>/shared/memory/active_context.md atomically.
# Supports the M18 "active-session adjustments" pattern — agent can record
# "user said be more terse this session" without it touching personas/main.yaml.
```

```python
# capabilities/tools/add_run_note.py
# Not hard-rail.
class AddRunNoteArgs(BaseModel):
    run_id: str
    note: str  # markdown

# Adds a user/agent annotation to a run record. Stored in <data-dir>/data/runs/{run_id}/notes.md.
```

```python
# capabilities/tools/create_workflow.py
# Hard-rail. Wraps POST /v1/workflows. provenance=meta_agent.
class CreateWorkflowArgs(BaseModel):
    name: str
    description: str | None
    definition: WorkflowDefinition  # from Session 02's Pydantic types

class CreateWorkflowResult(BaseModel):
    workflow_id: str
```

```python
# capabilities/tools/update_workflow.py
# Hard-rail. Wraps PUT /v1/workflows/{id}.
class UpdateWorkflowArgs(BaseModel):
    workflow_id: str
    definition: WorkflowDefinition  # full replacement
```

```python
# capabilities/tools/archive_workflow.py
# Hard-rail (archive is reversible but should still confirm — better safe).
class ArchiveWorkflowArgs(BaseModel):
    workflow_id: str
```

```python
# capabilities/tools/run_workflow.py
# Hard-rail (spends money; potentially long-running).
class RunWorkflowArgs(BaseModel):
    workflow_id: str
    task_input: TaskInput  # {text, attachments}

class RunWorkflowResult(BaseModel):
    run_id: str
    status: Literal["queued"]

# Wraps POST /v1/runs. Returns the run_id; the agent can subscribe to EventBus
# for in-flight observability or check status via read_run later.
```

### 4.3 Suggest-with-confirm flow

The agent never executes a hard-rail tool directly. Instead:

1. Agent emits a tool_call for the hard-rail tool.
2. `MainAgent._process_response` checks `tool.requires_confirmation` → True.
3. Standing approval check: `policy.is_approved(tool, args)` — the standing approvals are scoped per-tool per-arg-prefix.
4. If approved → tool executes immediately.
5. If not approved:
   - Emit `NeedsDecision` event into the inbox at P1 with `{tool_name, args, draft_message, suggestion_id}`.
   - The agent's response to the user (current turn) includes the draft message: "I want to \[do thing\]. Confirm to proceed."
   - User approves via `POST /v1/agent/confirm/<suggestion-id>` (web UI or CLI).
   - The deferred tool call executes; result feeds back into the agent's next turn.
   - User can grant standing approval at this confirmation point (a checkbox: "Don't ask again for this tool with these args").

Standing approvals live in `<sandbox>/shared/personas/main.policy.yaml`:

```yaml
standing_approvals:
  - tool: execute_shell
    cwd_prefix: /home/rezaho/research_projects/marsys-spren-work
    cmd_prefix: ["git"]
    granted_at: 2026-05-14T10:32:00Z
  - tool: add_run_note
    granted_at: 2026-05-14T10:32:00Z
  - tool: set_active_context
    granted_at: 2026-05-14T10:32:00Z
hard_rails:
  - delete_workflow
  - modify_settings
  - revoke_channel
  - forget_fact
  - run_workflow
  - update_workflow
  - create_workflow
  - archive_workflow
  - execute_shell
  - commit_fact_now
```

The agent can NEVER bypass hard rails by editing this file directly — the hard rails list is hardcoded in `capabilities/tools/hard_rails.py` and the policy reader merges it with the user-editable section. Standing approvals can be revoked at any time by editing `main.policy.yaml`.

### 4.4 Budget enforcement + Cost Center (SP-013)

**Direction (2026-05-14 user redirect)**: drop the per-action hard cap. The right primitive is a transparent **Cost Center** — a single page where users see every cost-incurring artifact (past sessions / live agents / scheduled jobs / their estimates) and can act on them. Daily budget remains the only hard limit. Removing the per-action cap removes the failure mode where users hit cryptic "blocked by per-action cap" errors and have to dig through settings to understand which knob to turn.

The agent never blocks itself with a per-action cap — only the daily budget acts as a guardrail. Within-day, every spend is auditable in the Cost Center. The user can see what's running, what's queued, what's already cost them money — and cancel anything from one place.

**Cost Center data model + endpoints (Session 08 ships):**

```python
# capabilities/cost_center.py — Spren-specific cost-center aggregator.
@dataclass
class CostEntry:
    id: str                      # ULID
    kind: Literal["agent_session", "workflow_run", "scheduled_job", "consolidation_pass", "channel_event"]
    label: str                   # human-readable: "Quill main session — May 14, morning", "research-pipeline (run-abc123)", etc.
    status: Literal["completed", "running", "scheduled", "deferred"]
    cost_usd: float | None       # actual for completed/running; estimate for scheduled
    started_at: datetime | None
    ended_at: datetime | None
    next_run_at: datetime | None # for scheduled
    source_id: str | None        # FK to source artifact (run_id, schedule_job_id, etc.)
    source_route: str | None     # the page where this artifact is edited (e.g., "/scheduled-jobs/{id}")

class CostCenter:
    """Aggregates costs across all sources for a unified view.
    Read-only aggregation; mutations go through source pages (SP-019)."""

    def list_entries(
        self,
        status: list[CostEntryStatus] | None = None,
        kind: list[CostEntryKind] | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[CostEntry]: ...

    def daily_summary(self, date: date) -> DailySummary:
        """Returns: total spent, total estimated for remaining today, by-kind breakdown."""

    def cancel(self, entry_id: str) -> None:
        """Cancel a running or scheduled entry. Routes to the source's cancel mechanism;
        the cost-center surface only orchestrates."""
```

**API routes:**
- `GET /v1/cost/entries` — list with filters (status, kind, time range)
- `GET /v1/cost/summary/today` — today's spend + estimate + by-kind breakdown
- `POST /v1/cost/entries/{id}/cancel` — cancel a running/scheduled entry (delegates to source)
- `GET /v1/cost/limits` — daily budget cap

Note: edits to schedules, workflows, etc. happen on their respective pages (e.g., `/scheduled-jobs/{id}/edit`). The cost center page provides the `source_route` link for each entry; clicking takes the user there. This keeps single-source-of-truth: each artifact has one canonical edit surface (SP-019).

**The agent-session aggregation question** (one practical issue to resolve in implementation): the main agent runs continuously. Defining "a session" for cost-center display: the implementer batches agent activity into "sessions" by inactivity gap (default: 30-minute idle = session boundary). Each session is one Cost Center entry with its tokens-in/out + cost. Heartbeat-only intervals don't create new sessions; user-direct events do.

**Budget enforcement (simplified):**

```python
# capabilities/budget.py
class SprenBudgetEnforcer:
    """Daily budget hard limit only. No per-action cap. Per-think token cap remains."""

    def __init__(self, settings: SettingsStore, cost_meter: CostMeter):
        self._settings = settings
        self._cost_meter = cost_meter

    def check_pre_turn(self, event: Event) -> BudgetDecision:
        """Called before each agent turn. Returns DEFER, ALLOW."""
        daily_cap = self._settings.get("cost.daily_budget_usd", 10.0)
        spent = self._cost_meter.daily_total_usd()
        if spent >= daily_cap:
            if event.priority == Priority.P0:
                return BudgetDecision.ALLOW  # P0 user-direct + critical workflow failures still process
            return BudgetDecision.DEFER  # schedule re-attempt at next budget window
        return BudgetDecision.ALLOW

    # No check_pre_tool_call — per-action cap is removed. Tool costs aggregate into daily total
    # and surface in the Cost Center; user retains visibility, no surprise blocks.
```

When the daily budget is exhausted, an inbox notification fires: "Daily budget exhausted ($10.00 / $10.00). Critical events (user-direct, workflow failures) still go through. Other work resumes tomorrow at <next budget window>. View what spent the budget at the Cost Center."

### 4.5 Supersession algorithm (memory write path)

The full algorithm from `10-memory-architecture.md` §5:

```python
# capabilities/supersession.py
class SprenSupersessionEngine(SupersessionEngine):
    def commit_fact(self, f_new: Fact, store: FactStore) -> list[Mutation]:
        # 1. Multi-valued predicate? Always additive.
        if predicate_metadata(f_new.predicate).cardinality == "multi_valued":
            return [InsertActive(f_new)]

        # 2. Functional predicate. Look up active fact.
        f_old = store.find_active(f_new.entity_id, f_new.predicate)

        # 3. No prior fact: insert.
        if f_old is None:
            return [InsertActive(f_new)]

        # 4. Same value: update last_seen.
        if f_old.value == f_new.value:
            return [UpdateLastSeen(f_old.id, now())]

        # 5. Confidence floor.
        if f_new.confidence < 0.6 or f_new.confidence < f_old.confidence:
            return [InsertDisputed(f_new, kind="confidence", paired=f_old.id),
                    UpdateDisputed(f_old.id, kind="confidence", paired=f_new.id),
                    EnqueueUserDecision(f_old, f_new, kind="confidence")]

        # 6. Stable + non-user source.
        if f_new.volatility == "stable" and f_new.source != "user_direct":
            return [InsertDisputed(f_new, kind="stable", paired=f_old.id),
                    UpdateDisputed(f_old.id, kind="stable", paired=f_new.id),
                    EnqueueUserDecision(f_old, f_new, kind="stable")]

        # 7. Trust-level check.
        if trust_level(f_new.source) < trust_level(f_old.source):
            return [InsertDisputed(f_new, kind="trust", paired=f_old.id),
                    UpdateDisputed(f_old.id, kind="trust", paired=f_new.id),
                    EnqueueUserDecision(f_old, f_new, kind="trust")]

        # 8. Standard supersession.
        return [InsertActive(f_new, supersedes=f_old.id),
                UpdateSuperseded(f_old.id, superseded_by=f_new.id)]
```

Mutations apply atomically against the markdown file (the indexer ingests the new state).

### 4.6 Consolidation pipeline — Memory Update Agent inside per-candidate agentic loop

> **Source-of-truth note (2026-05-14)**: the canonical pipeline spec lives at [`tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md`](../../../../../tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md) (the fully-LLM-driven design replacing the prior DeBERTa-rerank version). The summary below captures the shape; the research doc carries the prompt templates, exact retrieval pool sizes, and cost-stage breakdown. The earlier doc at `06-agentic-memory-update.md` is **superseded** as of 2026-05-14 — its rerank stage uses DeBERTa which the user redirected away from.

The consolidation pass is a 6-stage outer pipeline; Stage 3 is the per-candidate Memory Update Agent loop. Every stage is either deterministic code or a cheap LLM call.

**Stage 1 — Inventory** (`stages/inventory.py`):
- List all current markdown files; hash them.
- Load `pending_facts` since last run.
- Load sub-agent `BranchSummary.procedural_lessons` (empty in v0.3 — sub-instances ship in v0.4).
- Load `pending_persona_changes` queue (the PersonaFeedback events from previous turns).
- Load voice_drift events from session log since last run.

**Stage 2 — Extract** (`stages/extract.py`):
- Per-turn-pair extraction LLM call. Mem0-style [`FACT_RETRIEVAL_PROMPT`](https://github.com/mem0ai/mem0/blob/main/mem0/configs/prompts.py) adapted for our schema: input is the turn pair + recent conversation summary; output is a JSON list of `CandidateFact` objects with `{entity_id, predicate, value, asserted_at, source, confidence}`.
- Dedupe across turns within the pass.
- Cost: ~$0.003 per 10 turns at GPT-4o-mini-class pricing.

**Stage 3 — Memory Update Agent (the agentic loop, runs per candidate)** (`stages/agentic_loop.py`):

> Pipeline detail is locked in research doc 07 (post-redirect). Summary below; implementer reads research 07 as primary source.

For each candidate fact from Stage 2, the loop is fully LLM-driven (no DeBERTa, no specialized ML rerankers):

**3a — TMS gate** (cheap-LLM, fast, early-out): one LLM call (Haiku 4.5 / GPT-5.4-mini class) classifies the candidate against the `core_facts` set (facts in `profile/` with `volatility=stable` AND `source=user_direct`, ~50 facts at typical scale). The LLM is shown the candidate + the core_facts list and asked: "does this candidate contradict any core fact?" Returns a structured verdict: `{contradicts: bool, contradicting_fact_id: str | null, confidence: float}`. If contradicts AND confidence > 0.7, mark candidate as `disputed` with `kind=tms`; enqueue user decision; skip the rest of the loop. Cost: ~$0.0005 per candidate. SP-020.

**3b — Retrieve (four-query hybrid + RRF + LLM rerank)**: build the candidate's retrieval pool from four parallel queries:

- **Q1 — Structured by entity + predicate-cluster.** SQL `SELECT * FROM facts WHERE entity_id = ? AND status = 'active' AND (predicate = ? OR predicate IN <cluster>)`. Uses `predicate_clusters.yaml` for cluster expansion. Cost: <1ms; $0.
- **Q2 — BM25 / FTS5.** Lexical search over fact-text + entity-name strings. Catches alternate phrasings, name variants, vocabulary mismatches. k=20. Cost: ~5ms; $0.
- **Q3 — Dense vector.** sqlite-vec single-vector search using a local embedding model (FastEmbed / BGE-small, ~50MB local) over fact-text. Catches paraphrases, semantic similarity. k=20. Cost: ~1ms local; $0.
- **Q4 — HyDE expansion.** One small-LLM call generates 3 hypothetical contradiction sentences (e.g., for `lives_in: Paris` candidate → "User lives in Berlin", "User's current city is Hamburg"). Each is embedded and dense-retrieved. **Key to finding contradictions that don't share vocabulary** (the Berlin/Paris case). Cost: ~$0.0001 per candidate.

**Reciprocal Rank Fusion** (Cormack 2009, k=60) merges the four ranked lists. Top N (research-locked) → LLM rerank.

**LLM rerank**: one cheap-LLM listwise call ranks the fused candidates by relevance + relation-type to the candidate. Replaces the NLI cross-encoder rerank from earlier draft. Pattern (listwise vs pointwise vs pairwise) and prompt template locked in research 07. Cost: ~$0.001 per candidate.

**3c — Adjudicate** (one LLM call per candidate): the Memory Update Agent prompt. For each retrieved fact, the LLM classifies the relation: **SAME / RELEVANT_DIFFERENT / CONTRADICTS / UNRELATED**. Then outputs an aggregate operation recommendation: **ADD / UPDATE / DELETE / NOOP / DISPUTE**. Cost: ~$0.001 per candidate.

**3d — Apply via supersession algorithm** (deterministic): the LLM's `operation` field is **a recommendation**, not authoritative. Run `commit_fact(candidate, recommended_op, classifications)` from `10-memory-architecture.md` §5: cardinality dispatch (`has_dog: multi_valued` → ADD even if LLM says UPDATE — fixes Buddy/Scout); confidence floor (low-confidence candidates can't auto-supersede); stable-source check (non-user_direct can't auto-supersede stable facts); trust-level check. Mutation applied to markdown.

**Stage 3d (CoVe) is removed.** The earlier draft used CoVe as a tiebreaker on LLM-NLI disagreement; with no NLI signal, the trigger condition no longer exists. If the rerank or adjudicate calls are uncertain (low confidence), the supersession algorithm routes to DISPUTE — that's the existing safety net.

**Stage 4 — Routing** (`stages/routing.py`):
- For each fact about to be applied, the routing table from `10-memory-architecture.md` §8.2 determines the target file. Lookup table; no LLM.

**Stage 5 — Commit** (`stages/commit.py`):
- Apply all mutations atomically to markdown files.
- Compute `content_hash: sha256(asserted_at | source | value)` for every committed fact (M5; SP-022 foundational defense).
- Run format validator at the indexer boundary (M4; rejects malformed, logs to `consolidation_errors.md`).
- Single `git commit` on `<sandbox>/shared/memory/`.
- Emit `ConsolidationCompleted` event with `{tokens_in, tokens_out, cost_usd, facts_committed, facts_disputed, duration_ms}` (M17).

**Stage 6 — Reindex** (`stages/reindex.py`):
- MarkdownIndexer reads the diff and updates SQL projection (entities + facts tables).
- Tier 4 vector + FTS5 incrementally updated.

**The PersonaReflection stage** (`consolidation/persona_reflection.py`) runs after Stage 3 in the same pipeline pass:
- Reads delta + observed signals (deterministic, computed from session log) + current persona + `PersonaFeedback` events + voice_drift events.
- Step 1 prompt: "What are the 3 most salient questions about whether the persona is still the right shape for this user?" Returns 0-3 questions.
- For each question that survives the deterministic filter (doctrine boundary, identity boundary, confidence floor 0.6, bond-age ≥ 7 days):
- Step 2 prompt: "For this salient question, propose at most one minimal persona diff."
- Output: YAML files in `<sandbox>/shared/personas/pending_persona_changes/<prop-id>.yaml` + inbox items at P2.

### 4.6.1 Cost + latency budget

Per consolidation pass at ~10 candidates (final numbers locked in research 07; estimates below):

| Stage | LLM calls | $ approx | Latency |
|---|---|---|---|
| Stage 2 — Extract | 5 | $0.003 | ~15s |
| Stage 3a — TMS gate | 10 | $0.005 | ~10s |
| Stage 3b — Retrieve (HyDE) | 10 | $0.001 | ~5s |
| Stage 3b — LLM rerank | 10 | $0.010 | ~10s |
| Stage 3c — Adjudicate | 10 | $0.010 | ~10s |
| Stage 3d — Apply | 0 | $0 | <1s |
| Stages 4-6 | 0 | $0 | <2s |
| **Total** | ~45 | **~$0.030** | **~50s** |

The fully-LLM design lands at ~$0.03/pass — slightly more than the earlier hybrid ($0.01) but well under any practical ceiling ($1+/pass would be a problem; $0.03 is fine). The latency is faster (no GPU/CPU NLI inference). Headroom remains for telemetry, retries, and cadence increases. Final numbers locked when research 07 ships.

### 4.6.2 Naming convention

Per the memory-update research (`06-agentic-memory-update.md` Part 1), the system uses **"Memory Update Agent"** as the term-of-art for the per-candidate component. The previous shorthand "LLM-as-judge" is reserved in 2026 production literature for *evaluation* (Mem0 uses it for scoring answer quality on LoCoMo), not for memory writes. Spren's docs and code use:

- **Memory Update Agent** — the per-candidate agentic loop (Stage 3 of the consolidation pass).
- **Consolidation Pass** — the cadence-bound batch wrapper (24h+5sessions trigger).
- **Memory operations** — ADD / UPDATE / DELETE / NOOP / DISPUTE (Mem0's term + DISPUTE for our SP-020 case).
- **Write-side adjudication** — the per-fact classification step inside the agent (Memory Update Agent → Adjudicator sub-component).
- **TMS gate** — the SP-020 protective check against `core_facts` (named distinctly because its scope is narrower than full adjudication).

### 4.7 Persona-evolution CLI

```bash
spren persona log              # git log -p personas/main.yaml
spren persona why <axis>       # finds most recent commit touching axis; shows proposal + evidence
spren persona diff <since>     # diff between then and now
spren persona approve <prop-id>  # apply the diff; git commit; move proposal to journal
spren persona reject <prop-id> [--reason "..."]  # move to journal with reject_reason; feed back to next consolidation
spren persona revert <commit-sha>  # git revert; recommended for "I shouldn't have approved that" cases
spren persona list             # show pending proposals
spren persona show <prop-id>   # show details of one proposal
```

Backend routes:
- `POST /v1/agent/persona/proposals/{id}/approve`
- `POST /v1/agent/persona/proposals/{id}/reject`
- `GET /v1/agent/persona/proposals` (list)
- `GET /v1/agent/persona/proposals/{id}` (detail)
- `GET /v1/agent/persona/log` (recent commits on `personas/`)

### 4.8 Bond reset CLI

```bash
spren bond reset
```

Hard-rail-confirmed CLI command (no UI surface). Mechanics from `09-meta-agent.md` §"Bond-violation handling — Tier 3":

```
This will:
  - rename personas/main.yaml to personas/journal/<bond-id>.yaml
  - record a "bond-ended" event in the session log
  - return to the archetype picker; new identity.bonded_at on selection
  - keep all memory (it's still your memory)
  - keep all workflows (they're still your workflows)
  - your past Spren's identity is preserved in journal/ — not deleted

Are you sure? (y / N)
```

After confirmation:
- `personas/main.yaml` → `personas/journal/<bond-id>.yaml`. The bond-id is `bond-<bonded_at-isoformat>`.
- `bond_ended` event written to session log.
- Daemon transitions back to onboarding mode (Session 07's state machine reverses).
- Frontend re-routes to `/onboarding`.

`spren bond status` (read-only command):
- Bond age (`now() - identity.bonded_at`).
- Current archetype.
- Number of evolved axes.
- Number of approved persona proposals.
- Number of rejected persona proposals.
- Number of voice_drift events in the last week.

### 4.9 Voice-drift consumption

Session 07 ships the *detector* (regex post-pass on every agent message; logs `voice_drift` events). Session 08 ships the *consumer* — the consolidation pass reads voice_drift events; the PersonaReflection stage uses drift count + drift pattern as input to the salient-questions prompt:

> Voice-drift signals from the last consolidation window:
> - "Great question" used 3 times (3 different turns)
> - "Happy to help" used 1 time
>
> Given the user's current archetype is Vesper (which forbids these phrases), and given the agent's voice has drifted toward saccharine register, is the persona still the right shape? Or does the voice axis need recalibration?

If drift is sustained → persona-evolution may propose tightening the voice.style_tells.avoid list, or surfacing the issue to the user.

### 4.10 Stated-vs-observed surfacing

When observed behavior contradicts a user-direct preference (e.g., user states `prefers_morning_meetings: true` but reschedules morning meetings 4+ times in 2 weeks), the consolidation pass detects the pattern via deterministic signal computation (no LLM) and surfaces it as an `active_context.md` note:

> Observed: user has rescheduled morning meetings 4 times in the last 2 weeks despite stated preference for them.

The agent's next heartbeat may raise this conversationally ("Hey — I noticed you keep rescheduling morning meetings. Want me to switch the default?") OR add a per-fact override. The agent does NOT silently flip the user's stated preference. SP-011 + SP-012 + the bond integrity hold.

### 4.11 New deps

`pyproject.toml`:

```toml
[project]
dependencies = [
    # ... existing ...
    "fastembed ~= 0.4",        # lightweight ONNX embedding (BGE-small, ~50MB) for retrieval Q3 + Q4
    "sqlite-vec ~= 0.1.9",     # local vector index in SQLite (already in our Tier 4 plan)
]
```

No torch / transformers / sentence-transformers dependency. The 2026-05-14 redirect drops the local NLI cross-encoder; rerank is a cheap-LLM call via the existing provider client. The only local model is the embedding model (FastEmbed BGE-small at ~50MB) used purely for retrieval (Q3 dense + Q4 HyDE), not relevance scoring.

Optional system tooling: `ripgrep` for the `grep` tool's fast path; falls back to Python `re` if unavailable.

### 4.12 Tests

- **Pytest unit:** Each tool's args-schema validation; supersession algorithm's full decision matrix (functional / multi_valued / no-prior / same-value / low-confidence / stable+non-user / low-trust / standard-supersession); CUPMem adjudicator against fixture candidate-vs-existing pairs (KEEP/STALE/REPLACE/UNKNOWN); TMS gate against fixture core_facts; consolidation routing rules lookup; PersonaReflection deterministic filter (doctrine boundary, identity boundary, confidence floor, bond-age check); budget enforcer pre-turn + pre-tool-call decisions; standing-approvals matching.
- **Pytest integration:** End-to-end consolidation pass against a fixture session-log + fixture markdown KB → expected git commit + ConsolidationCompleted event; persona-evolution proposal generation against a fixture observed-signals state; persona approve/reject CLI roundtrip; bond reset CLI roundtrip + state-machine reverse; suggest-with-confirm flow end-to-end (agent emits tool_call → NeedsDecision event → user approves → tool runs); budget exhaustion behavior.
- **Per-archetype consolidation test:** for each of the 5 archetypes, run a fixture consolidation pass and verify the PersonaReflection stage's proposals are coherent with the archetype (e.g., a Vesper drift toward over-pushing should propose lowering initiative; a Flint drift toward saccharine should propose tightening voice).
- **Manual-verify checklist:** Run a real consolidation pass against the user's accumulated dogfood data; review the git commit; approve a proposal via CLI; verify persona update + git history.

## 5. What is OUT of scope

| Out of scope in Session 08 | Lands in |
|---|---|
| UI for persona-evolution proposals (settings → "About our bond" page) | Session 09 (or v0.4 — synthesis says v0.4 for the rich UI; Session 09 may ship a thin version alongside the four-surface command center) |
| UI for the suggest-with-confirm flow beyond the chat surface | Session 09 |
| The four-surface command center home page (Now / Since / Activity / Chat) | Session 09 |
| Sub-instance spawning + skills + skill consolidation | v0.4 |
| Team managers + team-scoped sandboxes | v0.4 |
| Channel-sourced fact extraction + the staleness watcher | v0.4 |
| Procedural memory consolidation from BranchSummary (SKILL.md format adopted but no curated procedures shipped) | v0.4 |
| Pause/resume Spren-side surface | v0.4 |
| Vector embeddings + hybrid retrieval (BM25 + vector + reranker) | v0.4 |
| Live (non-consolidation) NLI adjudication path | v0.4 |
| Custom tool authoring | v0.4 (Phase AA) |
| `vault/` encryption + `vault_lookup` tool | v0.4 |
| `poisoning_patterns.yaml` seeding + experience-poisoning defenses (SP-022) | v0.4 |
| CUPMem propagation-aware search (Type II implicit conflict) | v0.5+ (note for later) |
| RL-trained memory policies | v0.5+ (note for later) |

## 6. Files to CREATE / MODIFY in Session 08

### To CREATE — `spren/runtime/memory/` (generic, SP-023)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/runtime/memory/__init__.py` | Re-exports public runtime/memory API. |
| `packages/spren/src/spren/runtime/memory/supersession.py` | `SupersessionEngine` ABC + Mutation types. |
| `packages/spren/src/spren/runtime/memory/tms_gate.py` | TMS gate via cheap-LLM call (no specialized ML model). |
| `packages/spren/src/spren/runtime/memory/retrieval/__init__.py` | `MemoryRetriever` interface + four-query design. |
| `packages/spren/src/spren/runtime/memory/retrieval/rrf.py` | Reciprocal Rank Fusion (Cormack 2009). |
| `packages/spren/src/spren/runtime/memory/retrieval/hyde.py` | `HyDEExpander` — generic primitive; one LLM call returns 3 hypotheticals. |
| `packages/spren/src/spren/runtime/memory/retrieval/llm_rerank.py` | `LLMReranker` — listwise/pointwise/pairwise (locked in research 07). |
| `packages/spren/src/spren/runtime/memory/agent.py` | `MemoryUpdateAgent` ABC — the per-candidate agentic loop. |
| `packages/spren/src/spren/runtime/memory/budget.py` | Generic `BudgetEnforcer`. |
| `packages/spren/src/spren/runtime/memory/consolidation.py` | `ConsolidationPipeline` ABC + stage interfaces. |
| `packages/spren/src/spren/runtime/memory/cost_center.py` | Generic `CostCenter` aggregator. |

### To CREATE — `spren/agent/capabilities/` (Spren-specific)

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/agent/capabilities/__init__.py` | Module init. |
| `packages/spren/src/spren/agent/capabilities/tools/__init__.py` | Tool registry registration. |
| `packages/spren/src/spren/agent/capabilities/tools/read_file.py` | (one file each) read tools. |
| `packages/spren/src/spren/agent/capabilities/tools/grep.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/lookup_facts.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/recall.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/verify_fact.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/confirm_with_user.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/commit_fact_now.py` | escape-valve write tool. |
| `packages/spren/src/spren/agent/capabilities/tools/set_active_context.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/add_run_note.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/create_workflow.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/update_workflow.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/archive_workflow.py` | |
| `packages/spren/src/spren/agent/capabilities/tools/run_workflow.py` | bridges to `Orchestra.run()`. |
| `packages/spren/src/spren/agent/capabilities/tools/hard_rails.py` | hard-rail tool list + `IsHardRail` predicate. |
| `packages/spren/src/spren/agent/capabilities/supersession.py` | Spren concrete `SprenSupersessionEngine`. |
| `packages/spren/src/spren/agent/capabilities/budget.py` | Spren concrete `SprenBudgetEnforcer`. |
| `packages/spren/src/spren/agent/capabilities/suggest_with_confirm.py` | flow implementation. |
| `packages/spren/src/spren/agent/capabilities/policy.py` | Standing approvals + hard rails policy reader. |
| `packages/spren/src/spren/agent/capabilities/consolidation/__init__.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/pipeline.py` | Concrete 6-stage pipeline. |
| `packages/spren/src/spren/agent/capabilities/consolidation/scheduler.py` | Trigger logic. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/inventory.py` | Stage 1. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/extract.py` | Stage 2 — Mem0-style FACT_RETRIEVAL_PROMPT. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/agentic_loop.py` | Stage 3 — per-candidate Memory Update Agent loop (TMS → retrieve → LLM-rerank → adjudicate → apply). |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/routing.py` | Stage 4 — bubble-up routing rules. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/commit.py` | Stage 5 — git commit + content_hash + format-validate + ConsolidationCompleted event. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/reindex.py` | Stage 6 — incremental index update. |
| `packages/spren/src/spren/agent/capabilities/consolidation/persona_reflection.py` | PersonaReflection stage runs alongside Stage 3. |
| `packages/spren/src/spren/agent/capabilities/memory_agent.py` | `SprenMemoryUpdateAgent` — concrete subclass of `runtime/memory/agent.MemoryUpdateAgent`. Wires Spren's retrieval + Memory Update Agent prompt + supersession integration. |
| `packages/spren/src/spren/agent/capabilities/retrieval.py` | Spren four-query hybrid retrieval. Composes runtime/memory/retrieval/* primitives with Spren schemas. |
| `packages/spren/src/spren/agent/capabilities/data/predicate_clusters.yaml` | Predicate-cluster lookup table for Q1 expansion (~10 clusters). |
| `packages/spren/src/spren/agent/capabilities/data/extraction_prompt.txt` | Mem0-style FACT_RETRIEVAL_PROMPT adapted. |
| `packages/spren/src/spren/agent/capabilities/data/memory_update_agent_prompt.txt` | Adjudicator prompt with per-fact classification + aggregate operation. |
| `packages/spren/src/spren/agent/capabilities/data/hyde_prompt.txt` | HyDE expander prompt (3 hypothetical contradiction sentences). |
| `packages/spren/src/spren/agent/capabilities/data/llm_rerank_prompt.txt` | Listwise rerank prompt. |
| `packages/spren/src/spren/agent/capabilities/data/tms_gate_prompt.txt` | TMS gate cheap-LLM prompt. |
| `packages/spren/src/spren/agent/capabilities/data/persona_judge_prompt.txt` | LLM-as-judge final-approval gate for persona proposals. |
| `packages/spren/src/spren/agent/capabilities/cost_center.py` | Spren-specific Cost Center (aggregates all cost-incurring artifacts; routes edits to source pages). |
| `packages/spren/src/spren/agent/capabilities/cli/persona.py` | spren persona CLI. |
| `packages/spren/src/spren/agent/capabilities/cli/bond.py` | spren bond reset + status. |
| `packages/spren/src/spren/agent/capabilities/api.py` | new FastAPI routes for persona-evolution + bond. |
| `packages/spren/src/spren/agent/capabilities/data/routing_rules.yaml` | predicate-class → file-path lookup table. |
| `packages/spren/src/spren/agent/capabilities/data/main_policy_default.yaml` | default `main.policy.yaml` (hard rails list + empty standing approvals). |

### To CREATE — tests

| Path | Purpose |
|---|---|
| `packages/spren/tests/runtime/memory/test_supersession.py` | Generic engine decision matrix. |
| `packages/spren/tests/runtime/memory/test_tms_gate.py` | TMS gate against fixture core_facts via mocked-LLM; verdict structure + confidence threshold. |
| `packages/spren/tests/runtime/memory/retrieval/test_rrf.py` | Reciprocal Rank Fusion algorithm against Cormack 2009 expected outputs. |
| `packages/spren/tests/runtime/memory/retrieval/test_hyde.py` | HyDE expander returns 3 sentences; structured output mode. |
| `packages/spren/tests/runtime/memory/retrieval/test_llm_rerank.py` | LLM listwise rerank against fixture candidate + retrieved-facts pool. |
| `packages/spren/tests/runtime/memory/test_agent.py` | MemoryUpdateAgent ABC + per-candidate loop wiring. |
| `packages/spren/tests/runtime/memory/test_budget.py` | Budget enforcer. |
| `packages/spren/tests/runtime/memory/test_cost_center.py` | Cost Center aggregator: list, summary, cancel routing. |
| `packages/spren/tests/runtime/memory/test_consolidation_pipeline.py` | ABC pipeline + stage interfaces. |
| `packages/spren/tests/agent/capabilities/test_tools_<each>.py` | One per tool. |
| `packages/spren/tests/agent/capabilities/test_hard_rails.py` | Hard-rail enforcement matrix. |
| `packages/spren/tests/agent/capabilities/test_suggest_with_confirm.py` | Flow end-to-end. |
| `packages/spren/tests/agent/capabilities/test_supersession.py` | Spren concrete supersession integration. |
| `packages/spren/tests/agent/capabilities/test_consolidation_e2e.py` | Full 6-stage consolidation pass against fixture session log + fixture KB → expected git commit. |
| `packages/spren/tests/agent/capabilities/test_memory_agent_e2e.py` | Memory Update Agent end-to-end: candidate fact → TMS gate → retrieve (Q1-Q4 + RRF + LLM-rerank) → adjudicate → apply. Berlin/Paris semantic-contradiction case. Buddy/Scout cardinality-dispatch case. |
| `packages/spren/tests/agent/capabilities/test_persona_judge.py` | LLM-as-judge auto-approval gate: synthetic proposals, mocked judge responses; verify approve→diff applies, reject→journal, manual mode skips judge. |
| `packages/spren/tests/agent/capabilities/test_cost_center.py` | Spren CostCenter aggregator: list filtering, daily summary, cancel-routing to source. |
| `packages/spren/tests/agent/capabilities/test_retrieval.py` | Four-query hybrid retrieval against fixture KB with seeded entities + facts. |
| `packages/spren/tests/agent/capabilities/test_persona_reflection.py` | PersonaReflection deterministic filter + proposal generation. |
| `packages/spren/tests/agent/capabilities/cli/test_persona_cli.py` | Click CLI for persona. |
| `packages/spren/tests/agent/capabilities/cli/test_bond_cli.py` | Click CLI for bond reset. |
| `packages/spren/tests/integration/test_meta_agent_e2e.py` | Bundle 03 demo gate test: archetype pick → workflow run via run_workflow tool → trace inspect via read_trace → memory facts via lookup_facts → consolidation pass → persona evolution proposal → user approve → persona update. |
| `packages/spren/tests/agent/capabilities/test_stated_vs_observed.py` | The morning-meetings case. |

### To MODIFY

| Path | Edit |
|---|---|
| `packages/spren/src/spren/server.py` | Register the new tools at startup; register the consolidation scheduler; register the new API routes. |
| `packages/spren/src/spren/agent/main_agent.py` | Wire the budget enforcer into pre-turn check; wire the suggest-with-confirm flow into _process_response. |
| `packages/spren/src/spren/agent/system_prompt.py` | Axis #4 (Capabilities) now includes the full tool catalog. |
| `Justfile` | Add `just persona-cli`, `just bond-cli`, `just consolidate-now` recipes. |
| `docs/architecture/spren/10-memory-architecture.md` | Mark §5 / §6 / §7 / §8 / §11 / §12 sections as "shipping in Session 08". |
| `docs/architecture/spren/09-meta-agent.md` | Mark §"Persona-evolution mechanism" + §"Bond-violation handling" + §"Authority tiers" as shipping in Session 08. |

### To DELETE

None at the file level. Session 08 is purely additive on top of Session 07's foundations.

## 7. User journeys (anchor for Bundle 03 demo gate — closing)

Bundle 03 demo gate: meta-agent reads workflows + traces + memory; suggests-with-confirm on writes; dispatches workflows; respects hard rails; ticks heartbeat; tracks budget; persona evolves through the bond mechanism.

### J-7 — End-to-end: chat → workflow run → trace inspect → consolidation → persona evolution

State: fresh data dir; user picked Quill at onboarding; workflow `research-pipeline` exists from Bundle 01.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User opens home page (Quill orb breathing). Types: "run research-pipeline on the topic 'agent memory architectures'". | Home page chat | `POST /v1/meta/messages`. |
| 2 | Main agent dequeues; reasons; emits tool_call for `run_workflow` with `{workflow_id: <research-pipeline>, task_input: {text: "agent memory architectures"}}`. | (server-side) | — |
| 3 | Hard-rail confirmation: `run_workflow` is hard-rail. Suggest-with-confirm fires. Spren responds: "I want to run `research-pipeline` on 'agent memory architectures'. Estimated cost: $0.20. Confirm?" | Home page chat | Confirmation UI surfaces. |
| 4 | User confirms. | Web UI | `POST /v1/agent/confirm/<id>`. |
| 5 | `run_workflow` executes — `POST /v1/runs` fires; run_id returned. Agent's response: "Started run \[link\]. Watching now." | Home page chat | Spren orb shifts to `thinking` then `speaking` as AG-UI events stream from the workflow. |
| 6 | Workflow completes ($0.18 spent). Agent surfaces summary: "research-pipeline finished. Three sources flagged. Want me to summarize?" | Home page chat | (Quill voice — curious-and-precise.) |
| 7 | User: "yes." | Home page chat | Agent calls `read_trace` to get details, then summarizes. |
| 8 | User clicks the run link to inspect the trace. | `/runs/{id}` (Session 05) | Full trace tree visible. |
| 9 | User says "remember that I prefer markdown summaries to bullet lists." | Home page chat | Agent extracts the fact → `commit_fact_now` (escape-valve, hard-rail-confirmed) OR adds to pending_facts (default). For an explicit "remember", the agent uses commit_fact_now. |
| 10 | Hard-rail confirmation surfaces: "I want to remember `prefers_summary_format: markdown` in your preferences. Confirm?" | Home page chat | User confirms. |
| 11 | Fact lands in `profile/preferences.md` via the indexer; supersession algorithm runs (no prior fact); content_hash computed. Live commit logged in session log. | (server-side) | — |
| 12 | (~24h later, after 5+ sessions accumulated) Consolidation pass triggers. | (server-side) | Stage 1 inventories; Stage 2 extracts + adjudicates; Stage 3 TMS-gates; Stage 4 routes; Stage 5 supersession; Stage 6 git-commits. ConsolidationCompleted event fires. |
| 13 | PersonaReflection stage runs alongside consolidation. Observed signals: "user accepted 18/20 retry-failed-run proposals on first ask in the last 5 sessions." Step 1 returns salient question: "Cautiousness on retry_failed_run is currently `principled`; should it lift to `bold`?" Step 2 produces minimal diff. Proposal lands in `pending_persona_changes/`. | (server-side) | Inbox item at P2: "I have a persona proposal. `spren persona show <id>` to review." |
| 14 | User runs `spren persona show <id>`. | CLI | Prints proposal details: axis, current value, proposed value, evidence, confidence, reversibility. |
| 15 | User runs `spren persona approve <id>`. | CLI | Diff applied to `personas/main.yaml`; git commit recorded with structured message; `evolved_axes` list updated. The bond shifts: Quill's cautiousness on retry_failed_run is now `bold`. Next time the agent runs into a retry decision, it acts directly without asking. |

### J-8 — Bond reset

State: from J-7, several persona evolutions have been approved.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User runs `spren bond status`. | CLI | Bond age: 12 days. Archetype: quill. Evolved axes: [cautiousness.by_action_class.retry_failed_run, voice.style_tells.lean_into]. Approved proposals: 2. Rejected: 1. |
| 2 | User runs `spren bond reset`. | CLI | Confirmation prompt with the full mechanics description. |
| 3 | User confirms. | CLI | `personas/main.yaml` → `personas/journal/bond-2026-05-14T10:32:00Z.yaml`. `bond_ended` event in session log. Daemon transitions to onboarding mode. |
| 4 | User refreshes the web app. | Frontend | Re-routes to `/onboarding` (since `meta_agent.archetype` is null again). User picks a different archetype — say Vesper. New bond starts. Memory + workflows preserved. |

### J-9 — Cost Center + budget exhaustion

State: daily budget cap = $10; current spend = $9.95; user opens the Cost Center to see what spent the budget.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User opens Cost Center page. | `/cost-center` (Session 09 surface; Session 08 ships the data + endpoints) | Sees today's entries: 3 past agent sessions ($1.50 each), 1 running workflow ($2.20), 4 scheduled jobs (estimated $0.80 each), 1 consolidation pass ($0.03). Total spent: $9.95. Total estimated remaining: $3.20. |
| 2 | User clicks one of the scheduled jobs to edit cadence. | `source_route` link → `/scheduled-jobs/{id}` | Routed to the scheduled-jobs page (the canonical edit surface). Cancels job there. |
| 3 | User returns to Cost Center; spend is still $9.95 (cancelled job's estimate is reflected immediately as removed). | Cost Center | List updates. |
| 4 | User: "run research-pipeline." | Home page chat | Agent's pre-turn budget check: $9.95 < $10.00 daily cap; allowed. The workflow runs with cost $0.20. New daily total: $10.15. |
| 5 | Daily total now exceeds cap. | Inbox | Notification fires: "Daily budget exhausted ($10.15 / $10.00). Critical events still go through. Open Cost Center to see what spent the budget. Other work resumes tomorrow." |
| 6 | User: "schedule another run for tomorrow." | Home page chat | Agent schedules via `schedule_event` for next budget window (00:00 next day); the new entry surfaces in Cost Center under "Scheduled". |

## 8. Decisions locked

1. **Tool catalog complete in Session 08.** Read tools (6) + write tools (8 listed) ship in this session; v0.4 adds `pause_run`, `resume_run`, `discard_paused_session` (the v0.4 pause/resume Spren surface) + sub-instance spawn tools.

2. **Hard-rail tool list is hardcoded; user-editable standing approvals merge with it.** The hardcoded list cannot be edited by the agent. The standing-approvals section of `main.policy.yaml` is user-editable; the agent can suggest standing approvals (during a confirmation flow with "don't ask again for this scope") but cannot grant them itself.

3. **Cost Center surface replaces per-action hard cap.** Per the 2026-05-14 user redirect: no per-action cost cap. The Cost Center page shows every cost-incurring artifact (past sessions / live agents / scheduled jobs / their estimates) with one-click cancel. Daily budget remains the only hard limit. Edits to underlying artifacts route to their source pages (SP-019: each artifact has one canonical edit surface).

4. **Daily budget enforcement at pre-turn.** Cost check happens once per turn before the LLM call. Within-turn LLM calls aggregate against the per-think token cap (default 50K). When daily budget exhausts: P0 events still process; P1-P3 defer with a Cost Center notification.

5. **Cheap vs expensive model selection.** Routine (heartbeat, low-stakes responses, classifier-style extractions): cheap. Complex (consolidation pipeline's extract+adjudicate stage, PersonaReflection, hard-rail decisions): expensive. The mapping in `agent/model_selection.py`.

6. **TMS gate runs only at consolidation time in v0.3.** Live writes (escape valve) get the supersession algorithm but NOT the full TMS gate (only the simple "stable+non-user_direct → dispute" check). v0.4 may add live TMS adjudication for select high-stakes paths.

7. **CUPMem 4-state output is a prompt-engineering rule, not a separate LLM stage.** The Stage 2 (extract+adjudicate) prompt requires the LLM to output a verdict per candidate fact. Cheaper than two prompts; same effect; one prompt change.

8. **The PersonaReflection stage runs once per consolidation pass.** It outputs 0-3 proposals (or none if no salient questions surface). Confidence floor 0.6; bond-age ≥ 7 days; doctrine boundary; identity boundary — all enforced by deterministic filter before proposals reach the auto-approval gate.

8a. **Auto-approval is the default for persona proposals; LLM-as-judge gates final approval.** Per 2026-05-14 user redirect: the PersonaReflection stage emits proposals → deterministic filter (confidence/bond-age/doctrine/identity) → **LLM-as-judge** (one expensive-model call evaluates each surviving proposal against the bond integrity rubric: "would this evolution preserve identity / strengthen the bond / honor the user's stated values?") → if judge approves, the diff applies automatically + user gets an inbox notification with the diff + approval reasoning + a one-click `revert` link. If judge rejects, the proposal moves to journal with the judge's reasoning. Manual mode is opt-in via `settings.meta_agent.persona_evolution_mode = "manual"`; default is `"auto"`. The CLI commands (`spren persona approve <id>`, `spren persona reject <id>`) remain — they're the user's explicit override path.

8b. **LLM-as-judge for personas is the right framing.** Per research 06's terminology: "LLM-as-judge" applies to evaluation tasks (Mem0 uses it for scoring answer quality on LoCoMo). Persona-proposal approval IS evaluation — the judge scores whether the proposal serves the bond. This is consistent with 2026 production literature; we use the term correctly here (as opposed to write-side memory adjudication, which research 06 + 07 call "Memory Update Agent" — different mechanism, different name).

9. **Standing approvals scoped per-tool + per-arg-prefix.** E.g., `execute_shell` with `cwd_prefix=<repo>` and `cmd_prefix=["git"]` allows `git status` but not `git push --force`. The matching algorithm: longest-prefix match wins; ambiguity → confirmation required.

10. **`commit_fact_now` requires `source: user_direct` per SP-015.** The agent cannot escape-valve facts from sub-agent observations or untrusted channels. Channels-sourced facts always queue; consolidation reviews.

11. **`forget_fact` is a hard rail (vs `tombstone_fact`).** Hard rails confirm; tombstones are auditable but reversible. `forget` produces an audit trail (the tombstone record); the user retains ability to `restore`. Hard-rail check ensures user-confirmed every time.

12. **Bond reset preserves memory + workflows.** Only `personas/main.yaml` is archived. The user's facts about themselves, projects, etc. all survive. The new Spren reads them on first turn.

13. **Memory Update Agent — per-candidate agentic loop, fully LLM-driven (no specialized ML rerankers).** Per 2026-05-14 user redirect, the prior DeBERTa-rerank path is replaced. For each candidate fact: TMS gate (cheap-LLM, early-out) → four-query hybrid retrieval (Q1 structured by entity + predicate cluster; Q2 BM25/FTS5; Q3 dense vector via local embedding; Q4 HyDE expansion) fused via Reciprocal Rank Fusion (k=60) → **cheap-LLM rerank (listwise / pointwise / pairwise — pattern locked in research 07)** → adjudicator LLM call (per-fact classification + aggregate operation) → deterministic supersession algorithm. Supersession owns the final mutation decision; the LLM's operation field is advisory. Cost: ~$0.03 per pass at 10 candidates. Canonical spec: [`tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md`](../../../../../tmp/spren/research/06-memory-foundations/07-agentic-memory-update-no-deberta.md). The earlier `06-agentic-memory-update.md` is **superseded** as of 2026-05-14.

14. **No local NLI cross-encoder, no DeBERTa.** Removed entirely from the design. The TMS gate is a cheap-LLM call with structured output (`{contradicts: bool, contradicting_fact_id: str | null, confidence: float}`). The rerank is a cheap-LLM call returning a relevance ranking with structured output. No torch / transformers / sentence-transformers dependency. No bundled-or-downloaded specialized ML model.

15. **Embedding model for Q3 dense + Q4 HyDE = BGE-small via FastEmbed.** 384-dim, ~50MB on disk, runs locally via FastEmbed (ONNX-only, no torch dependency). Used purely for retrieval (vector search), not for relevance scoring. Optional cloud fallback to `text-embedding-3-small` (OpenAI) behind a config flag. Every embedded chunk carries `embedding_model_id` for v0.4-v0.5 rotation (M16).

16. **HyDE generates 3 hypothetical contradiction sentences per candidate.** Fixed count (not variable). Trade-off: predictable cost; covers ≥3 phrasings of "what would a contradicting fact look like"; 3 is the cheapest credible diversity. Per the [HyDE paper](https://arxiv.org/abs/2212.10496), one hypothetical works for long documents; for short fact-text we use 3 to span the contradiction space.

17. **Retrieval pool sizing: 20 fused → N reranked → 8 to adjudicator.** N locked in research 07. Memory-R1 retrieves "top-K"; Mem0 uses 10; we use 20 fused to give the adjudicator more semantic context without inflating prompt cost.

18. **Decoupled relation classification from operation choice.** The LLM classifies relations (SAME / RELEVANT_DIFFERENT / CONTRADICTS / UNRELATED); the supersession algorithm decides operations (ADD / UPDATE / DELETE / NOOP / DISPUTE) based on the relations + cardinality + confidence + trust gates. This is the Buddy/Scout fix: the LLM correctly identifies "Scout is a different dog from Buddy" (RELEVANT_DIFFERENT); supersession sees `has_dog: multi_valued` and adds rather than supersedes — even if the LLM gets the operation field wrong, cardinality dispatch overrides.

19. **One adjudicator prompt for both classification AND operation.** Single LLM call returns per-fact classifications + aggregate operation. Cheaper (one call vs two); the operation field is advisory anyway (supersession owns the final decision). Mitigation against contamination: prompt instructs the LLM to do classifications first, operation second.

20. **Confidence thresholds (initial; tune from telemetry).** TMS gate fires at LLM `confidence > 0.7`. Adjudicator classification confidence below 0.7 on CONTRADICTS routes to DISPUTE rather than UPDATE. Final values locked when research 07 ships and tuned from real-world telemetry in v0.4.

21. **CoVe step removed.** The earlier draft used CoVe as a tiebreaker on LLM-NLI disagreement; with no NLI signal, the trigger condition no longer exists. If the rerank or adjudicate calls produce low confidence, the supersession algorithm routes to DISPUTE — that's the existing safety net. No specialized "self-verification" sub-stage.

22. **Cost Center as the cost-control surface.** Per 2026-05-14 user redirect, the per-action hard cap is removed. The Cost Center surface (Session 08 backend, Session 09 frontend) shows every cost-incurring artifact in one place: past agent sessions, running workflows, scheduled jobs, consolidation passes — each with their cost, status, and link to the source page where they can be edited. Daily budget remains the only hard limit. Single source of truth: edits route to source pages (scheduled-jobs page edits scheduled jobs; runs page edits runs); the Cost Center is read-only aggregation + cancel.

23. **Persona-evolution auto-approval default; LLM-as-judge gates.** Per 2026-05-14 user redirect, the PersonaReflection pipeline emits proposals → deterministic filter (confidence/bond-age/doctrine/identity) → LLM-as-judge (one expensive-model evaluation against bond integrity rubric) → if approved: diff applies automatically, user notified via inbox with one-click revert; if rejected: proposal goes to journal with judge reasoning. Manual mode (`spren persona approve/reject`) becomes opt-in via `settings.meta_agent.persona_evolution_mode = "manual"`. Default is `"auto"`. CLI commands remain as override paths in either mode.

## 9. Polish items to address inside Session 08

1. **Adjudicator prompt template stability.** The Stage 3c prompt asks the LLM to output structured JSON: `{candidate_id, verdict, classifications, recommended_operation, reasoning}`. Use OpenAI / Anthropic tool-calling JSON mode (or equivalent) for reliable parsing. Implementer benchmarks JSON-mode reliability across the cheap model lineup; falls back to regex parsing of structured-text output if JSON mode flakes.

2. **TMS gate cost batching.** TMS-gate runs once per candidate fact during consolidation as a cheap-LLM call. At ~50 candidates per pass × ~50 core_facts, naive prompt-per-pair would be ~2500 calls. The actual implementation batches: each TMS-gate call sees the candidate + the full core_facts list and asks the LLM to flag any contradictions (one call per candidate). Cost ~$0.0005 per candidate × 50 candidates = $0.025 per pass — same order as the rerank stage. Implementer verifies the per-candidate batched prompt classifies correctly across the cheap-model lineup.

3. **Consolidation pipeline concurrency.** The 6 stages run sequentially per candidate but candidates within a stage can parallelize (especially Stage 2 extract + Stage 3 LLM calls). Implementer benchmarks; if a pass takes >2 min wall-clock at 100 candidates, parallelize at the candidate level within each stage.

4. **Persona-evolution proposal expiration.** Proposals expire 14 days after creation if neither approved nor rejected. The expiry check happens at consolidation pass time (no separate timer). On expiry: move to journal with `expired_at`; counts as soft no for re-proposal.

5. **`spren persona log` performance.** `git log -p personas/main.yaml` may be slow at year+ scale. Implementer benchmarks at 100, 500, 1000 commits; if slow, adds a `--limit` flag (default 20).

6. **Suggest-with-confirm message rendering.** The agent's draft message ("I want to do X") is rendered in the chat surface with the proposed args clearly visible (e.g., a code block for shell commands; a definition list for fact commits). The implementer designs the render templates per tool kind.

7. **Standing approval grant UX.** When the user confirms a hard-rail action, they see a small checkbox: "Don't ask again for this scope" (with a dropdown showing the scope: `tool: execute_shell, cwd: <project>, cmd: git*`). Default unchecked. If checked → standing approval added.

8. **Bond reset: orphan check.** Before performing the reset, verify no in-flight workflow runs are using the current persona (since the run system prompt is built per-turn using the active persona). Refuse with "Cannot reset bond while runs are in flight; wait or cancel them first."

9. **`spren persona why` for axes that have never been touched.** If the axis has no commit history (default value still active from archetype), output: `axis: voice.register | original archetype default (Vesper, 2026-05-14) | unchanged since`. No empty-result confusion.

10. **Consolidation cost telemetry.** ConsolidationCompleted event is logged per run; surface the rolling-7-day average cost-per-consolidation as a settings/diagnostics metric. If average exceeds 5× the expected ($0.05), surface a warning to the user — the consolidation pass might be regressing or the model has changed.

## 10. Success criteria

- **G-32** (read tool roundtrip): each of the 6 read tools works against fixture state.
- **G-33** (write tool hard rail): `commit_fact_now` triggers confirmation; user approves; fact lands.
- **G-34** (workflow dispatch): `run_workflow` triggers confirmation; user approves; run starts; agent receives lifecycle events.
- **G-35** (suggest-with-confirm standing approval): user grants standing approval on `execute_shell git in <repo>`; subsequent `git status` invocations don't re-prompt.
- **G-36** (consolidation end-to-end): fixture session log + 5+ pending facts → consolidation pass produces git commit + ConsolidationCompleted event with correct counts.
- **G-37** (TMS gate triggers): a candidate fact contradicts a core fact → cheap-LLM TMS-gate call returns `{contradicts: true, contradicting_fact_id: <id>, confidence: > 0.7}` → disputed state with kind=tms; user-decision item enqueued; rest of agentic loop skipped (early-out). Test with mocked-LLM fixture responses.
- **G-38** (Memory Update Agent — semantic contradiction case): the Berlin/Paris case end-to-end. Existing `(user, lives_in, Berlin)` fact in memory; candidate `(user, lives_in, Paris)` from session-log. HyDE Q4 generates "User lives in Berlin" hypothetical → dense-retrieves the existing Berlin fact (which Q3 vector search alone misses because Paris-token is far from Berlin-token). LLM rerank flags it as relevant + contradicting. LLM adjudicator classifies CONTRADICTS. Supersession algorithm: stable+user_direct → DISPUTE state; user-decision item enqueued. Verifies the four-query hybrid catches contradictions that don't share vocabulary.
- **G-38a** (RRF fusion): synthetic four-query result lists with overlapping documents → RRF ranks correctly per Cormack 2009 formula with k=60.
- **G-38b** (HyDE expansion produces 3 sentences): for each fixture candidate, HyDE returns a JSON list of exactly 3 hypothetical contradiction sentences.
- **G-38c** (LLM rerank): top-20 fused → LLM listwise rerank returns ordered top-N (N from research 07) with relevance + relation-type per fact. Verify against fixture candidate + retrieved-facts pool with mocked-LLM responses.
- **G-38d** (Memory Update Agent — Buddy/Scout case): existing `(user, has_dog, Buddy)` fact, candidate `(user, has_dog, Scout)`. LLM classifies RELEVANT_DIFFERENT. Even if LLM's operation field says UPDATE, supersession sees `has_dog: multi_valued` per `predicate_metadata.py` and emits ADD. Both Buddy and Scout end up `active`. Verifies cardinality dispatch overrides LLM operation field.
- **G-39** (supersession decision matrix): all 8 decision branches in the algorithm produce expected outputs against fixture inputs (cardinality, confidence floor, stable+non-user, trust-level, plus standard supersession).
- **G-40** (PersonaReflection): synthetic observed-signals state generates a proposal; deterministic filter blocks low-confidence and bond-age-too-young proposals; proposal reaches the LLM-as-judge gate.
- **G-40a** (LLM-as-judge auto-approval): proposal passes deterministic filter → LLM-as-judge approves → diff auto-applies; commit lands; user receives inbox notification with the diff + reasoning + revert link. Verify with mocked judge response.
- **G-40b** (LLM-as-judge rejection): proposal passes deterministic filter → LLM-as-judge rejects → proposal moves to journal with reasoning; no diff applied; user notified.
- **G-40c** (manual mode override): set `settings.meta_agent.persona_evolution_mode = "manual"` → judge step is skipped; proposal lands in `pending_persona_changes/` for CLI approve/reject as before.
- **G-41** (persona CLI override roundtrip): in either auto or manual mode, CLI approve / reject is the explicit override path → diff applied / proposal journaled → git commit → `evolved_axes` updated.
- **G-42** (bond reset): CLI reset → persona archived to journal → daemon onboarding-mode → frontend routes to /onboarding → re-pick → new bond.
- **G-43** (budget exhaustion): synthetic spend ≥ daily cap → P0 events still process; P1-P3 events deferred; user notified via Cost Center notification.
- **G-43a** (Cost Center aggregation): seed fixture with 3 past sessions, 1 running workflow, 4 scheduled jobs, 1 consolidation pass → `GET /v1/cost/entries` returns all with correct status + cost + source_route; `GET /v1/cost/summary/today` returns spent + estimated + by-kind breakdown.
- **G-43b** (Cost Center cancel routing): cancel a running entry → `POST /v1/cost/entries/{id}/cancel` delegates to source's cancel mechanism (e.g., `POST /v1/runs/{id}/cancel` for runs); aggregation reflects status update.
- **G-44** (stated-vs-observed surfacing): synthetic observed-state showing 4+ rescheduled morning meetings against stated preference → consolidation surfaces an `active_context.md` note → agent's heartbeat raises it.
- **C-11** (no SP-023 violation): SP-023 boundary test extended for new files.
- **C-12** (per-archetype tool selection coherence): for each archetype, run a fixture conversation; verify the tools the agent picks match the archetype's profile (e.g., Flint reaches for `execute_shell` more readily; Vesper waits longer; Quill calls `verify_fact` more often).
- **U-16** (manual smoke): J-7 + J-8 + J-9 from the user journeys against a real installation.

## 11. Open research items the implementer resolves in-flight

- **Adjudicator prompt template wording**. Starting point: research 07 §Recommended config for Spren. Implementer iterates against fixture cases until classification accuracy is >90% on a hand-labeled validation set of 50 candidate-vs-existing pairs across the 8 supported providers' cheap models.
- **TMS-gate prompt template wording**. Same iteration approach. Validation: 30 hand-labeled candidate-vs-core-facts pairs spanning contradiction / no-contradiction / ambiguous; target >95% recall on contradictions (false negatives are dangerous; false positives are recoverable via DISPUTE).
- **LLM rerank prompt pattern**. Research 07 picks listwise vs pointwise vs pairwise; implementer verifies the chosen pattern's output is reliably parseable across cheap-model lineup.
- **LLM-as-judge for personas — rubric prompt template**. Implementer drafts the bond-integrity rubric; iterates against fixture proposal cases (some clearly-good, some clearly-bad, some ambiguous) until human-judge agreement is >90%.
- **HyDE prompt phrasing**. The 3-sentence generation prompt from research 07. Implementer verifies the LLM consistently produces 3 distinct hypothetical contradictions across cheap models. Failure mode: the LLM sometimes returns 1 long sentence instead of 3. Mitigation: structured JSON output mode where supported.
- **predicate_clusters.yaml starter table**. ~10 clusters seeded based on the user's expected dogfood patterns. Implementer reviews + refines; v0.4 may add more.
- **ripgrep availability detection**. The `grep` tool prefers `rg` (faster); falls back to Python `re` if not installed. Document the install hint in the manual smoke section.
- **Standing approvals scope-matching algorithm details**. The longest-prefix match seems right; verify against edge cases like nested cwd_prefixes.
- **Bond reset's orphan-check semantics**. Are there any in-flight uses of the persona that would be affected? Sub-agents (v0.4) inherit persona at spawn — for v0.3 only the main agent uses it; the orphan check is essentially "any in-flight workflow runs that the main agent dispatched and is still observing."
- **`spren persona why` performance at year+ scale**. Benchmark; add pagination if needed.
- **Cost Center entry batching policy for agent sessions**. Default: 30-minute idle gap defines a session boundary; heartbeat-only intervals don't create new sessions; user-direct events do. Implementer benchmarks against real dogfood data; refines.

## 12. Status

- [ ] Tier confirmed (CRITICAL).
- [ ] Scope boundaries confirmed.
- [ ] SP-023 boundary confirmed.
- [ ] Files-to-CREATE list approved.
- [ ] Three user journeys approved.
- [ ] Decisions locked.
- [ ] Polish items captured.
- [ ] Success criteria affirmed.
- [ ] Acceptance criteria frozen at [`./08-meta-agent-capabilities/acceptance.md`](./08-meta-agent-capabilities/acceptance.md).
- [ ] Architect peer review at Stages 1, 3, 4 (CRITICAL multi-checkpoint).
- [ ] Sessions 06 + 07 implementation complete (Session 08 hard-depends).
- [ ] Bundle 03 testing scenarios + agent brief drafted at [`../testing/test-scenarios.md`](../testing/test-scenarios.md), [`../testing/test-session.md`](../testing/test-session.md).
- [ ] Session implementation complete.

## 13. Open questions — resolved

All Session 08 questions are resolved per the 2026-05-14 user redirect.

1. **Per-action cost cap.** **Locked: removed.** Replaced with the Cost Center surface (decision §22). Daily budget remains the only hard limit; users see and control all cost-incurring artifacts in one place. Per-action caps were creating "blocked job" surprises that forced users to dig through settings to understand which knob to turn — the wrong UX direction.

2. **NLI model bundled vs download.** **Locked: irrelevant — DeBERTa removed entirely.** No specialized ML rerankers in the design. TMS gate + rerank are cheap-LLM calls. Embedding model (BGE-small via FastEmbed, ~50MB) ships bundled because it's small and used purely for retrieval, not relevance scoring.

3. **`spren persona reject` reason field.** **Locked: optional but encouraged.** Reason field is fed back to the next consolidation's salient-questions prompt as evidence when provided. Note: in default auto mode, the LLM-as-judge handles approve/reject; CLI is the explicit override path (decision §23) where the reason field is most useful for the rare manual override.

4. **CoVe self-verification trigger.** **Locked: removed.** CoVe was a tiebreaker on LLM-NLI disagreement; with no NLI signal, the trigger condition no longer exists. Low-confidence LLM rerank or adjudicate routes to DISPUTE — that's the existing safety net.

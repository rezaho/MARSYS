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
- `runtime/memory/supersession.py` — `SupersessionEngine` ABC + the deterministic algorithm. Takes a fact + a fact store + predicate metadata; returns `[Mutation]`. Generic shape — the algorithm in `10-memory-architecture.md` §5 doesn't depend on Spren-specific event types. Spren's concrete implementation passes Spren's `Fact` type (which subclasses a generic `Fact` base).
- `runtime/memory/adjudication.py` — CUPMem 4-state adjudication primitive. `Adjudicator(llm_client, prompt_template).adjudicate(candidate, related_facts) -> Verdict`. Generic — the prompt is parameterized.
- `runtime/memory/tms_gate.py` — `TMSGate.check(candidate_fact, core_facts) -> Decision`. Generic NLI-classifier wrapper; takes any `Fact` with text content. Concrete NLI model is parameterized.
- `runtime/memory/budget.py` — `BudgetEnforcer`. Generic; consumes `CostMeter` from Session 07 and the agent's tool registry to enforce budget at tool-call time.
- `runtime/memory/consolidation.py` — `ConsolidationPipeline` ABC. The 6-stage pipeline shape (inventory → extract+adjudicate → TMS → routing → conflict resolution → apply+log) is generic; each stage takes a strategy. Concrete stages (Spren-specific routing table, Spren persona reflection) are subclassed in the Spren layer.

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
- `capabilities/consolidation/pipeline.py` — Spren's concrete `ConsolidationPipeline` (subclass of generic). Wires the 6 stages.
- `capabilities/consolidation/stages/inventory.py` — Stage 1.
- `capabilities/consolidation/stages/extract_adjudicate.py` — Stage 2 (CUPMem 4-state output).
- `capabilities/consolidation/stages/tms.py` — Stage 3 (TMS gate against core_facts).
- `capabilities/consolidation/stages/routing.py` — Stage 4 (bubble-up routing rules table).
- `capabilities/consolidation/stages/conflict_resolution.py` — Stage 5 (supersession dispatch).
- `capabilities/consolidation/stages/apply.py` — Stage 6 (git commit + content_hash + ConsolidationCompleted event).
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

### 4.4 Budget enforcement (SP-013)

```python
# capabilities/budget.py
class SprenBudgetEnforcer:
    """Per-day budget cap + per-think token cap. Composes runtime/memory/budget."""

    def __init__(self, settings: SettingsStore, cost_meter: CostMeter):
        self._settings = settings
        self._cost_meter = cost_meter

    def check_pre_turn(self, event: Event) -> BudgetDecision:
        """Called before each agent turn. Returns DEFER, ALLOW, or REFUSE."""
        daily_cap = self._settings.get("cost.daily_budget_usd", 10.0)
        spent = self._cost_meter.daily_total_usd()
        if spent >= daily_cap:
            if event.priority == Priority.P0:
                return BudgetDecision.ALLOW  # P0 user-direct + critical workflow failures still process
            return BudgetDecision.DEFER  # schedule re-attempt at next budget window

        per_action_cap = self._settings.get("cost.per_action_budget_usd", 0.50)
        # Per-think token cap is enforced at LLM call time.
        return BudgetDecision.ALLOW

    def check_pre_tool_call(self, tool_name: str, estimated_cost: float) -> BudgetDecision:
        """Called before a tool call that has a known cost (e.g., run_workflow).
        Cost over per-action cap → upgrades to hard rail (additional confirmation)."""
        if estimated_cost > self._settings.get("cost.per_action_budget_usd", 0.50):
            return BudgetDecision.ELEVATE_TO_HARDRAIL
        return BudgetDecision.ALLOW
```

When budget is exhausted, an inbox notification fires: "Daily budget exhausted ($10.00 / $10.00). I'll resume normal processing tomorrow at <date>. Critical events (user-direct, workflow failures) still go through. To increase the cap, edit `cost.daily_budget_usd` in settings."

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

### 4.6 Consolidation pipeline (6 stages)

`capabilities/consolidation/pipeline.py` runs the full pipeline from `10-memory-architecture.md` §6. Each stage is its own file:

**Stage 1 — Inventory** (`stages/inventory.py`):
- List all current markdown files; hash them.
- Load `pending_facts` since last run.
- Load sub-agent `BranchSummary.procedural_lessons` (empty in v0.3 — sub-instances ship in v0.4).
- Load `pending_persona_changes` queue (the PersonaFeedback events from previous turns).
- Load voice_drift events from session log since last run.

**Stage 2 — Extract & adjudicate** (`stages/extract_adjudicate.py`):
- LLM pass over the day's session-log delta + pending_facts.
- For each candidate fact, the LLM explicitly outputs CUPMem 4-state verdict: `KEEP existing | STALE existing | REPLACE existing | UNKNOWN`.
- Output: list of `(candidate_fact, verdict, related_existing_fact_ids)`.

**Stage 3 — Truth Maintenance gate** (`stages/tms.py`):
- For each candidate fact, run NLI classifier against `core_facts` (facts in `profile/` with volatility=stable AND source=user_direct).
- On contradiction → mark candidate `disputed` with `kind=tms`; enqueue user decision; do NOT auto-commit.
- Output: filtered candidate list (TMS-passing) + disputed-tms count for the ConsolidationCompleted event.

**Stage 4 — Routing** (`stages/routing.py`):
- For each candidate fact, apply the bubble-up routing rules table from `10-memory-architecture.md` §8.2 to determine the target file.
- Lookup table; no LLM.
- Output: list of `(candidate_fact, target_file_path)`.

**Stage 5 — Conflict resolution** (`stages/conflict_resolution.py`):
- For each candidate, run the supersession algorithm (cardinality dispatch + confidence floor + stable-source check + trust-level check + standard supersession).
- Output: list of `Mutation`s + the disputed-other-kinds count.

**Stage 6 — Apply & log** (`stages/apply.py`):
- Apply all `Mutation`s atomically to markdown files.
- Compute `content_hash: sha256(...)` for every committed fact.
- Run format validator at the indexer boundary (rejects malformed, logs to `consolidation_errors.md`).
- Single `git commit` on `<sandbox>/shared/memory/`.
- Emit `ConsolidationCompleted` event with `{tokens_in, tokens_out, cost_usd, facts_committed, facts_disputed, duration_ms}`.

**The PersonaReflection stage** (`consolidation/persona_reflection.py`) runs after Stage 5 in the same pipeline pass:
- Reads delta + observed signals (deterministic, computed from session log) + current persona + `PersonaFeedback` events + voice_drift events.
- Step 1 prompt: "What are the 3 most salient questions about whether the persona is still the right shape for this user?" Returns 0-3 questions.
- For each question that survives the deterministic filter (doctrine boundary, identity boundary, confidence floor 0.6, bond-age ≥ 7 days):
- Step 2 prompt: "For this salient question, propose at most one minimal persona diff."
- Output: YAML files in `<sandbox>/shared/personas/pending_persona_changes/<prop-id>.yaml` + inbox items at P2.

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
    # No new deps strictly required — Session 06+07 already pulled in everything needed.
    # Optional: ripgrep system tool for the grep tool's fast path; falls back to Python re if unavailable.
]
```

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
| `packages/spren/src/spren/runtime/memory/adjudication.py` | CUPMem 4-state primitive. |
| `packages/spren/src/spren/runtime/memory/tms_gate.py` | NLI classifier wrapper. |
| `packages/spren/src/spren/runtime/memory/budget.py` | Generic `BudgetEnforcer`. |
| `packages/spren/src/spren/runtime/memory/consolidation.py` | `ConsolidationPipeline` ABC + stage interfaces. |

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
| `packages/spren/src/spren/agent/capabilities/consolidation/pipeline.py` | concrete pipeline. |
| `packages/spren/src/spren/agent/capabilities/consolidation/scheduler.py` | trigger logic. |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/inventory.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/extract_adjudicate.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/tms.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/routing.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/conflict_resolution.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/stages/apply.py` | |
| `packages/spren/src/spren/agent/capabilities/consolidation/persona_reflection.py` | PersonaReflection stage. |
| `packages/spren/src/spren/agent/capabilities/cli/persona.py` | spren persona CLI. |
| `packages/spren/src/spren/agent/capabilities/cli/bond.py` | spren bond reset + status. |
| `packages/spren/src/spren/agent/capabilities/api.py` | new FastAPI routes for persona-evolution + bond. |
| `packages/spren/src/spren/agent/capabilities/data/routing_rules.yaml` | predicate-class → file-path lookup table. |
| `packages/spren/src/spren/agent/capabilities/data/main_policy_default.yaml` | default `main.policy.yaml` (hard rails list + empty standing approvals). |

### To CREATE — tests

| Path | Purpose |
|---|---|
| `packages/spren/tests/runtime/memory/test_supersession.py` | Generic engine decision matrix. |
| `packages/spren/tests/runtime/memory/test_adjudication.py` | CUPMem against fixture inputs. |
| `packages/spren/tests/runtime/memory/test_tms_gate.py` | NLI gate. |
| `packages/spren/tests/runtime/memory/test_budget.py` | Budget enforcer. |
| `packages/spren/tests/runtime/memory/test_consolidation_pipeline.py` | ABC pipeline + stage interfaces. |
| `packages/spren/tests/agent/capabilities/test_tools_<each>.py` | One per tool. |
| `packages/spren/tests/agent/capabilities/test_hard_rails.py` | Hard-rail enforcement matrix. |
| `packages/spren/tests/agent/capabilities/test_suggest_with_confirm.py` | Flow end-to-end. |
| `packages/spren/tests/agent/capabilities/test_supersession.py` | Spren concrete supersession integration. |
| `packages/spren/tests/agent/capabilities/test_consolidation_e2e.py` | Full consolidation pass against fixture session log + fixture KB → expected git commit. |
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

### J-9 — Budget exhaustion

State: daily budget cap = $10; current spend = $9.95.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User: "run research-pipeline." | Home page chat | Agent's pre-turn budget check: $9.95 + estimated $0.20 = $10.15 > $10. |
| 2 | Pre-tool-call check fires: cost over per-action cap → ELEVATE_TO_HARDRAIL. Confirmation prompt elevates: "This will push you over today's budget cap ($10.15 vs $10.00). Confirm to proceed anyway?" | Home page chat | (Voice register stays archetype-correct.) |
| 3 | User declines. | Web UI | `POST /v1/agent/confirm/<id>` with `accept: false`. Tool call cancelled. Agent: "Got it. Want me to schedule it for tomorrow?" |
| 4 | User: "yes." | Home page chat | Agent schedules the run via `schedule_event` for next budget window (00:00 next day). |

## 8. Decisions locked

1. **Tool catalog complete in Session 08.** Read tools (6) + write tools (8 listed) ship in this session; v0.4 adds `pause_run`, `resume_run`, `discard_paused_session` (the v0.4 pause/resume Spren surface) + sub-instance spawn tools.

2. **Hard-rail tool list is hardcoded; user-editable standing approvals merge with it.** The hardcoded list cannot be edited by the agent. The standing-approvals section of `main.policy.yaml` is user-editable; the agent can suggest standing approvals (during a confirmation flow with "don't ask again for this scope") but cannot grant them itself.

3. **Per-action cost cap.** Default $0.50 per tool call. Tools whose estimated cost exceeds this are elevated to hard rails for that invocation, regardless of the underlying tool's hard-rail status. So a `read_file` (~$0.0001) doesn't elevate; a `run_workflow` with estimated cost above $0.50 does, even if the user has standing approval for run_workflow on that workflow_id.

4. **Daily budget enforcement at pre-turn.** Cost check happens once per turn before the LLM call; per-tool checks happen during tool dispatch. Within-turn LLM calls aggregate against the per-think token cap (default 50K).

5. **Cheap vs expensive model selection.** Routine (heartbeat, low-stakes responses, classifier-style extractions): cheap. Complex (consolidation pipeline's extract+adjudicate stage, PersonaReflection, hard-rail decisions): expensive. The mapping in `agent/model_selection.py`.

6. **TMS gate runs only at consolidation time in v0.3.** Live writes (escape valve) get the supersession algorithm but NOT the full TMS gate (only the simple "stable+non-user_direct → dispute" check). v0.4 may add live TMS adjudication for select high-stakes paths.

7. **CUPMem 4-state output is a prompt-engineering rule, not a separate LLM stage.** The Stage 2 (extract+adjudicate) prompt requires the LLM to output a verdict per candidate fact. Cheaper than two prompts; same effect; one prompt change.

8. **The PersonaReflection stage runs once per consolidation pass.** It outputs 0-3 proposals (or none if no salient questions surface). Confidence floor 0.6; bond-age ≥ 7 days; doctrine boundary; identity boundary — all enforced by deterministic filter before proposals reach the user.

9. **Standing approvals scoped per-tool + per-arg-prefix.** E.g., `execute_shell` with `cwd_prefix=<repo>` and `cmd_prefix=["git"]` allows `git status` but not `git push --force`. The matching algorithm: longest-prefix match wins; ambiguity → confirmation required.

10. **`commit_fact_now` requires `source: user_direct` per SP-015.** The agent cannot escape-valve facts from sub-agent observations or untrusted channels. Channels-sourced facts always queue; consolidation reviews.

11. **`forget_fact` is a hard rail (vs `tombstone_fact`).** Hard rails confirm; tombstones are auditable but reversible. `forget` produces an audit trail (the tombstone record); the user retains ability to `restore`. Hard-rail check ensures user-confirmed every time.

12. **Bond reset preserves memory + workflows.** Only `personas/main.yaml` is archived. The user's facts about themselves, projects, etc. all survive. The new Spren reads them on first turn.

## 9. Polish items to address inside Session 08

1. **CUPMem prompt template stability.** The Stage 2 prompt asks the LLM to output structured JSON: `{candidate_id, verdict, related_existing_fact_ids, reasoning}`. Use OpenAI / Anthropic tool-calling JSON mode (or equivalent) for reliable parsing. Implementer benchmarks JSON-mode reliability across the cheap model lineup; falls back to regex parsing of structured-text output if JSON mode flakes.

2. **TMS gate latency.** NLI classifier runs once per candidate fact during consolidation. At ~50 candidates per pass and ~50 core_facts, that's ~2500 NLI checks per pass. Use a small dedicated NLI model (DeBERTa-v3-large or equivalent local model) for the classifier — fast, cheap, deterministic. If a local model is too heavy for shipping, fall back to LLM-as-judge with the cheap model + structured JSON output.

3. **Consolidation pipeline concurrency.** The 6 stages run sequentially per candidate but the candidates within a stage can parallelize (especially Stage 2 LLM calls and Stage 3 NLI checks). Implementer benchmarks; if a pass takes >2 min wall-clock at 100 candidates, parallelize at the candidate level within each stage.

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
- **G-37** (TMS gate triggers): a candidate fact contradicts a core fact → disputed state with kind=tms; user-decision item enqueued.
- **G-38** (CUPMem 4-state output): Stage 2 LLM verdicts parse correctly across cheap model providers (OpenAI gpt-4o-mini, Anthropic claude-haiku, etc. — implementer picks).
- **G-39** (supersession decision matrix): all 8 decision branches in the algorithm produce expected outputs against fixture inputs.
- **G-40** (PersonaReflection): synthetic observed-signals state generates a proposal; deterministic filter blocks low-confidence and bond-age-too-young proposals; proposal lands in `pending_persona_changes/`.
- **G-41** (persona approve roundtrip): CLI approve → diff applied to personas/main.yaml → git commit → `evolved_axes` updated.
- **G-42** (bond reset): CLI reset → persona archived to journal → daemon onboarding-mode → frontend routes to /onboarding → re-pick → new bond.
- **G-43** (budget exhaustion): synthetic spend = $10 → P0 events still process; P3 events deferred; user notified.
- **G-44** (stated-vs-observed surfacing): synthetic observed-state showing 4+ rescheduled morning meetings against stated preference → consolidation surfaces an `active_context.md` note → agent's heartbeat raises it.
- **C-11** (no SP-023 violation): SP-023 boundary test extended for new files.
- **C-12** (per-archetype tool selection coherence): for each archetype, run a fixture conversation; verify the tools the agent picks match the archetype's profile (e.g., Flint reaches for `execute_shell` more readily; Vesper waits longer; Quill calls `verify_fact` more often).
- **U-16** (manual smoke): J-7 + J-8 + J-9 from the user journeys against a real installation.

## 11. Open research items the implementer resolves in-flight

- NLI model choice for TMS gate. DeBERTa-v3-large fine-tuned on MNLI/ANLI is the strongest local option; latency 100-300ms per check. LLM-as-judge with cheap model is more flexible but slower (1-3s) and less deterministic. Implementer benchmarks both; defaults to whichever fits the consolidation budget at typical scale.
- CUPMem prompt template wording. The exact phrasing matters; the implementer iterates against fixture cases until verdict accuracy is >90% on a hand-labeled validation set of 50 candidate-vs-existing pairs.
- Cheap-model provider for consolidation. `gpt-4o-mini` vs `claude-haiku-4` vs others — benchmark cost + reliability + JSON-mode support; pick the v0.3 default; expose in settings.
- ripgrep availability detection. The `grep` tool prefers `rg` (faster); falls back to Python `re` if not installed. Document the install hint in the manual smoke section.
- Standing approvals scope-matching algorithm details. The longest-prefix match seems right; verify against edge cases like nested cwd_prefixes.
- Bond reset's orphan-check semantics. Are there *any* in-flight uses of the persona that would be affected? Sub-agents (v0.4) inherit persona at spawn — for v0.3 only the main agent uses it; the orphan check is essentially "any in-flight workflow runs that the main agent dispatched and is still observing."
- `spren persona why` performance at year+ scale. Benchmark; add pagination if needed.

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

## 13. Open questions for user input

1. **Per-action cost cap default.** $0.50 feels right for a single tool call (a typical workflow run is $0.05-$0.30). Should the default be $0.50, $1.00, $0.25? My pick: $0.50. Want your call.

2. **NLI model: ship a local DeBERTa or use LLM-as-judge?** Local model: deterministic + fast + adds ~500MB to install size. LLM-as-judge: less deterministic but no install bloat + uses the existing model fleet. My pick: LLM-as-judge in v0.3 (simpler shipping); revisit at v0.4 if accuracy is insufficient. Want your call.

3. **`spren persona reject` reason field.** Required or optional? If required, the user has to think about why; useful for the next consolidation's "don't re-propose" logic. If optional, easier UX. My pick: optional, but if provided, it's fed back as evidence to the next consolidation's salient-questions prompt. Want your call.

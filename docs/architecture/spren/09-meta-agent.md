# 09 — The Meta-Agent (Spren's Continuous Personal Assistant)

> The meta-agent is the central interface between the user and everything Spren does. It is not a chat surface. It is a continuously-active personal AI agent that lives as a daemon, handles events from many channels, has persistent memory, can spawn sub-agents and team managers, supervises their work, and acts on the user's behalf with appropriate confirmation flows.
>
> **It is the agent that runs your other agents.** Treat its design with the gravity that implies — Spren's product success rests heavily on this component being good.

This doc covers the meta-agent's execution model, agent hierarchy, event ingress, persona, authority, scope, and the sandbox + filesystem permission model. The memory architecture has its own doc: [`10-memory-architecture.md`](./10-memory-architecture.md).

---

## Why this is its own execution model (not a Marsys workflow)

The Marsys framework's `Orchestra.run()` is a request/response model — finite execution, bounded by `convergence_timeout`, returning an `OrchestraResult`. The meta-agent has fundamentally different requirements:

| Requirement | Marsys framework today | Gap |
|---|---|---|
| Long-running daemon (no end-of-execution) | `Orchestra.run()` is finite; convergence_timeout, branch_timeout, step_timeout assume bounded run | Big — would need new "always-on" execution mode |
| Event-driven inbox from many channels | Only `task` input at run start | Big — no event bus open to external triggers |
| Persistent state across events | Stateless agents (DP-001); per-branch memory isolation (DP-004) | Big — would either violate DP-001 or require parallel mechanism |
| Sub-instance spawning at runtime | The `Orchestrator` (per DP-003 unified-barrier) spawns parallel branches only at declared topology divergence points, not at arbitrary "spawn now" | Medium — close pattern, mismatched semantics |
| Cross-instance live introspection | Branch tracing is post-hoc; the `EventBus` (`coordination/event_bus.py`) emits `BranchCreatedEvent` / `BranchCompletedEvent` only at lifecycle boundaries | Medium — would need a query API on in-flight branches |
| Tools that mutate the system itself | Workflow tools don't include "edit a workflow" | Medium — needs new tool surface and permission model |

These gaps span the framework's TRUNK-CRITICAL internals (Orchestra, Orchestrator, RealRuntime, ValidationProcessor, TopologyGraph). Reshaping them risks the framework's contract with its other users. The decision: **build a separate execution model in Spren that uses Marsys agents as building blocks but does not go through `Orchestra.run()`.**

The Spren runtime *consumes* Marsys: when the meta-agent decides "let's run workflow X to investigate," it calls `Orchestra.run()` as a tool invocation. Marsys returns an `OrchestraResult`; Spren handles it. Healthy boundary. For in-flight observability of these tool-invoked runs, Spren subscribes to Marsys's `EventBus`.

If, after months of operation, we discover patterns from this runtime that belong upstream in Marsys, we propose them as a framework feature with an ADR. For now: keep separate.

## The agent hierarchy

Three agent kinds, from most to least durable. Two orthogonal concepts also in play: **Teams** (user-organizational units) and **Skills** (capability bundles). They are NOT the same thing.

### Main Agent (singleton)
- **Always loaded** — one per Spren install. Owns the master inbox.
- **Persona = "yours"** — name, voice, defaults set by the user.
- **Elevated tool surface** — every tool the system has. Can configure team managers, can read all sandboxes, can trigger any workflow, can write to shared memory.
- **Knows the org chart** — which teams exist, who manages each, which working instances are running where.
- Decision authority: where an event goes (handle inline, delegate to a team manager, spawn a working instance, ask the user).

### Teams and Team Managers (user-defined organizational units)

**Teams are user concepts, not system concepts.** The user creates teams to mirror their actual work organization — not to map onto Spren's internal subsystems. Examples:

- **Research Team** — agents specialized in domain research, citation gathering, literature review
- **Admin Team** — agents specialized in scheduling, status reporting, paperwork
- **Personal Life Team** — agents handling personal-affair tasks, reminders, errands

Each team has:
- **A team manager** — an agent with that team's persona, scoped tools, and authority over the team's domain. Lives at `~/.spren/sandbox/teams/<team-slug>/persona.yaml`.
- **A team-scoped sandbox** at `~/.spren/sandbox/teams/<team-slug>/` with its own memory directory and working-instance scratchpads.
- **Persistent loading** — once activated, the team manager stays loaded (default eviction at 24h idle, configurable).
- **Domain-scoped tool surface** — only what the team's role legitimately needs.
- **Authority to spawn working instances** within the team's scope, optionally loading skills into them.

The main agent creates teams (via tool, with user confirmation), routes events to the appropriate team manager when the event matches a domain, and configures team personas/permissions. Users explicitly create and configure teams in the settings UI as well.

### Skills (capability bundles, separate from teams)

**Skills are NOT teams.** A skill is a capability bundle — a system-prompt fragment + tool selections + memory-access patterns + recommended models — that ANY agent can load into a working instance for a specific task. Examples:

- `workflow-inspection` — debugging failed runs, trace analysis, suggesting fixes
- `communication` — multi-channel messaging, formatting outputs for each channel's conventions
- `web-research-with-citations` — broad web research with proper citation
- `debugging` — error analysis on artifacts and logs
- `planning` — breaking a goal into a structured plan
- `code-review` — reading code diffs and reasoning about correctness

Skills are stored at `<data-dir>/sandbox/shared/skills/<slug>.yaml`. Spren ships a curated catalog from v0.4 (when sub-instances arrive); users can author new skills via configuration in v0.4 and via code generation (with user approval) in v0.5.

**Typical pattern:** the main agent (or a team manager) receives an event → spawns a working instance → loads one or more skills into the instance → the instance does the work with the loaded capability → reports back. A working instance can have multiple skills loaded if its task spans capabilities.

A team manager can specify a default skill set for instances spawned within its team (e.g., the Research Team's manager might load `web-research-with-citations` + `planning` by default).

### Working Instances (ephemeral)
- **Spawned per task** by the main agent or by a team manager.
- **Inherits parent's persona** by default; can be overridden by the loaded skill(s).
- **Scoped tools** — base set + tools brought by loaded skills.
- **Has its own inbox** for receiving messages from its parent and from sub-spawned instances.
- **Periodic checkpointing** — writes state to `instances/<id>/scratchpad/state.json` atomically at every meaningful state transition (per SP-014). Crash recovery depends on this; main agent can investigate the last checkpoint of a crashed instance.
- **Terminate on completion** — report back to spawner with a structured `BranchSummary` (see "Sub-instance return contract" below).
- **Can be inspected mid-execution** by their parent via `inspect_instance(instance_id)` — returns current focus, recent thoughts, recent tool calls.

**Concurrency limit:** bounded number of concurrent working instances (default 10, configurable). Beyond that, new spawn requests queue. Prevents runaway spawning during a flood of events.

### Sub-instance return contract (`BranchSummary`)

When a working instance completes, it emits a structured summary to its parent's inbox. The instance writes the summary itself as its final action — it knows what it did better than any external observer.

```yaml
status: success | failed | timeout | cancelled
summary: <text written by the instance>
result: <structured data>
artifacts: [<paths under sandbox>]      # files produced
cost:
  tokens_in: int
  tokens_out: int
  reasoning_tokens: int
  by_provider:
    anthropic: usd
    openai: usd
duration_ms: int
new_facts_proposed:                     # queued in pending_facts; NOT auto-committed
  - {predicate, value, confidence, source}
outstanding_todos:                      # added to active_todos.md
  - {description, priority, estimated_cost?}
open_questions: [<text>]
conversation_log_ref: <path>             # full log NOT inlined; main agent can grep/recall

# Failure path (only when status != success):
error_category: timeout | tool_error | rate_limit | reasoning_failed | budget_exhausted | ...
error_summary: <text>
traceback_excerpt: <text>
state_at_failure: <serialized state>
diagnostic_notes: <text>                # what the instance noticed before dying
```

The parent agent reads the summary and decides next steps. **Special main-agent ability:** read access to ANY instance's full conversation log + scratchpad, queryable via `read_file`/`grep`/`recall` tools. So when a sub-instance fails or returns a confusing summary, the main agent can investigate the full context.

## Event ingress and the inbox model

Every agent has its own inbox — main agent, team managers, and even working instances. All inbound events flow through a single `EventIngress` and are routed to the appropriate agent's inbox. Inboxes are **always** queued — events are never delivered directly to an in-flight thinking agent.

### Sources

| Source | Event types | Default routing | Default priority |
|---|---|---|---|
| User direct (web UI / CLI) | `UserMessage`, `UserCommand`, `InterruptNow` | Main agent inbox | **P0** |
| Critical workflow failures | `WorkflowFailed` (high-criticality flag) | Main agent inbox | **P0** |
| Channel adapters (Slack / Telegram / Discord / web) — allowlisted users | `ChannelMessage` | Main agent inbox (unless team-routed) | **P1** |
| Working instance completion | `InstanceCompleted`, `InstanceFailed` | Spawning agent's inbox | **P1** |
| Workflow lifecycle | `WorkflowStarted`, `WorkflowFinished`, `WorkflowSuspended` | Main agent inbox | **P2** |
| Schedule trigger (v0.3) | `ScheduledRun` | Main agent inbox | **P2** |
| Heartbeat | `HeartbeatTick` | Main agent inbox | **P3** |
| Webhook receiver (v0.3) | `WebhookFired` | Main agent inbox | **P3** |

### Priority queue, non-preemptive

Each inbox is a priority queue with four levels (P0 highest, P3 lowest). The agent's main loop:

```
loop:
    event = inbox.pop_highest_priority()       # blocks if empty
    state = load_relevant_context(event)       # see "System prompt structure" below
    process_turn(event, state)                 # one or more tool calls; can be long
    update_activity_log(...)
```

**"Available"** = the agent's current `process_turn` has just completed (a tool call returned, a final response was emitted, OR a checkpoint was hit). Between turns, the agent picks the highest-priority pending event next. **Never preempts a turn in flight.**

### Chunked thinking (responsiveness without preemption)

Long thinks are CHUNKED at tool-call boundaries. A complex task makes many tool calls; between each call, the agent's loop checks if a P0 event has arrived. If yes:

1. Save current turn's working state to `scratchpad/state.json` (atomic write, per SP-014)
2. Process the P0 event in a fresh turn
3. Resume the long turn from saved state

This gives near-real-time responsiveness for user-direct messages and critical alerts without true preemption (which would risk inconsistent state).

P0 events that interrupt: only user-direct messages, `InterruptNow` tool calls, and critical workflow failures. P1+ events wait their turn.

### Routing and cross-agent messaging

Events default to the main agent's inbox. The main agent decides downstream routing:
- **Forward to a team manager** by enqueuing on the manager's inbox (P1 by default)
- **Spawn a working instance** which has its own inbox; subsequent messages to that instance flow to its inbox
- **Sub-instances reply to spawning agent** by enqueuing the `BranchSummary` event on the spawning agent's inbox (P1)

In v0.4 (when teams ship), team managers do NOT directly cross-talk; all routing goes through main. A later release may add direct manager-to-manager messaging if patterns emerge.

### Idle cost = zero

No event in any inbox = no LLM calls anywhere. The "alive" feeling comes from UX (memory + activity stream + notifications) and the heartbeat sweep, NOT from continuous thinking.

**SP-011 (carries from `08-design-principles.md`):** Bounded triggers. The agent wakes only via its inbox, and the inbox has a curated set of producers (see the "Supervision surface" section below). Implementers MUST NOT add polling loops finer than a watcher's defined cadence, implicit cadences invented at runtime, or runtime-spawned watchers; the watcher catalog is curated per version.

## Decision flow on each event

When the main agent dequeues an event, the standard flow:

```
Event dequeued
    ↓
Load relevant memory (read active_context.md, lookup_facts(entity), search_memory if needed)
    ↓
Reason: what should I do?
    ↓
┌─────────────────────────────────────────────────────────┐
│ Branch on decision:                                      │
│  (a) Handle inline — emit response, update state        │
│  (b) Delegate to team manager — enqueue on their inbox  │
│  (c) Spawn working instance — for parallel sub-task     │
│  (d) Ask user — enqueue a "needs decision" inbox item   │
│  (e) Defer — enqueue for next heartbeat                 │
└─────────────────────────────────────────────────────────┘
    ↓
Update activity log + (potentially) propose new facts to pending_facts
    ↓
Return to listening
```

**Pushback on the user's framing**: "the main agent doesn't implement himself or do the detailed work, it can create a new instance of itself." I'd refine this — the main agent doesn't ALWAYS delegate. Trivial responses ("status of run 47?") should be handled inline. Spawning a sub-instance has a cost (model call to spawn, memory to track, eventual completion event). Decision rule for spawning:

- **Spawn if:** (a) the task takes substantial reasoning, (b) we don't want to block the main inbox, (c) the work needs different tools/persona than the main agent has, (d) parallel work with another in-flight item.
- **Otherwise:** handle inline.

## Authority tiers

Default authority for v0.3: **suggest-with-one-click + hard rails.** Standing approvals ship in v0.4.

- Read tools (`list_workflows`, `read_run`, `read_trace`, etc.) → `requires_confirmation=False`
- Low-risk write tools (`add_run_note`, `send_status_to_user`, etc.) → `requires_confirmation=False`
- Write tools that mutate workflows or trigger runs (`create_workflow`, `update_workflow`, `trigger_run`, `connect_channel`, etc.) → **`requires_confirmation=True`** by default; agent proposes, user clicks "do it" in UI or replies "yes" in channel
- **Standing approvals** (settings UI): user can convert a confirmation into auto-approve for a scope ("you can retry failed runs without asking", "you can post to #my-channel without asking")
- **Hard rails** (always require confirmation regardless of standing approvals): `delete_workflow`, `revoke_channel`, `modify_settings`, anything that ends a run, anything that touches secrets, any spend over the per-action budget cap

Authority tiers are a runtime policy in `<data-dir>/sandbox/shared/personas/main.policy.yaml`. Each tool declares its default `requires_confirmation`; the policy file can override per-tool or per-scope. The user can edit this file or the settings UI.

**SP-012 (new principle):** Hard rails always confirm. Even with full standing approvals enabled, destructive or spend-significant actions go through user confirmation. Implementers MUST NOT add code paths that bypass hard rails.

## Scope of the meta-agent

The meta-agent is **THE communication channel between the user and all sub-services** of Spren. It can:

- **Manage workflows:** create, read, update, archive (delete is hard-railed)
- **Manage sub-agents:** configure team managers, spawn working instances, terminate misbehaving ones
- **Manage channels:** connect/disconnect Slack/Telegram/Discord/web; configure per-channel allowlists and trust
- **Manage triggers:** schedules (v0.4), webhooks (v0.4)
- **Manage personas:** edit team manager personas (its own persona is more locked-down — see "self-modification" below)
- **Trigger Marsys workflows** as a tool (calls `Orchestra.run()`)
- **Read all observability data** — runs, traces, costs

**Tool / skill creation tiered by version:**

- **v0.3: configuration only.** The agent picks from curated built-in tools / workflow-templates and assigns them. It can assemble new agent personas, tool *bindings*, and workflow definitions using JSON / YAML config — no code generation. "apt-get install" model.
- **v0.4: define new bindings.** The agent can assemble new "skills" (= bundles of prompts + tool selections + memory instructions) without writing code. Stored as YAML.
- **v0.5: code generation.** The agent can write new Python tool functions. Each requires explicit user approval (review the code, click approve), runs in an isolated execution context, can be reverted.

**Self-modification limits:** the meta-agent CAN edit team manager personas and configurations. It CANNOT edit its own persona, its own tool list, or its own permissions without explicit user confirmation. Reason: the persona/policy is the user's safety boundary against the agent self-modifying into a less-aligned state.

## System prompt structure — six axes of context

The agent's system prompt at each turn is assembled from six categories of context. Persona is one of them — not the only one. Each category lives in a specific source file/runtime store.

| # | Axis | Source | What it contains | When loaded |
|---|---|---|---|---|
| 1 | **Persona** | `~/.spren/sandbox/shared/personas/main.yaml` | Identity, voice, style — the soul (8 sub-axes; see below) | Every turn |
| 2 | **Doctrine** | `~/.spren/sandbox/shared/rules.md` | Operational dos/don'ts, hard rails, escalation patterns | Every turn |
| 3 | **Situational awareness** | `~/.spren/sandbox/shared/active_context.md` + runtime state | Time, timezone, DND flags, current focus, connected channels, system health, "weather of user's life" | Every turn |
| 4 | **Capabilities** | runtime + `shared/skills/` + `main.policy.yaml` | Available tools, loadable skills, current authority level, budget remaining, org chart (active teams + instances) | Every turn |
| 5 | **Active context** | `~/.spren/sandbox/shared/memory/active_todos.md` + recent activity log | Long-term TODO/planning list, current focus, in-flight tasks, recent N actions | Every turn |
| 6 | **Memory** | KB tools (`read_file` / `grep` / `recall`) | Long-term facts and learnings; retrieved on demand via tool calls | NOT eager-loaded; agent calls retrieval tools when needed |

**Five of the six categories are eager-loaded into every system prompt.** Only Memory is lazy (retrieved on demand). This keeps the prompt size predictable and gives the agent explicit control over what it surfaces from long-term memory.

Team managers and working instances have the same six-axis structure with their own scoped files (their own persona, their team's doctrine subset, their team's TODOs, etc.). Skills loaded into a working instance contribute additional Doctrine and Capabilities content for the duration of that instance's life.

### Persona — the 8 sub-axes (within axis #1)

| Sub-axis | Examples | Default |
|---|---|---|
| Voice / tone | curt / warm / professional / witty | "competent ops engineer" — calm, succinct |
| Verbosity | terse / standard / detailed | standard |
| Cautiousness | conservative / balanced / bold | balanced |
| Initiative | passive / responsive / proactive | responsive |
| Risk default | when in doubt: ask / proceed / refuse | ask |
| Communication frequency | rare / normal / chatty | normal |
| Areas of focus | what to prioritize watching | all workflows + cost + errors |
| Identity | name + optional avatar | "Spren" |

Persona lives at `~/.spren/sandbox/shared/personas/main.yaml`. Editable in settings UI. Re-read on persona change AND on every heartbeat. Team managers have their own persona files with the same 8-sub-axis shape.

### Why this structure is non-arbitrary

- Persona is *who the agent is* — slow-changing, set by the user
- Doctrine is *what it must/must not do* — slow-changing, set by user + system
- Situational awareness is *what's true right now* — fast-changing, updated by every event
- Capabilities is *what it can do today* — slow-changing, but budget remaining changes per turn
- Active context is *what's currently in progress* — fast-changing, the working state
- Memory is *what it has learned over time* — slow-growing, lazy-retrieved

Conflating any two of these reliably produces bad agent behavior. They are kept distinct in source AND in system prompt assembly.

## Long-term TODO / planning list

The main agent maintains a persistent list of in-flight tasks, deferred items, and standing follow-ups. Lives at `~/.spren/sandbox/shared/memory/active_todos.md` (markdown, source of truth) with a derived SQLite mirror for queries (same pattern as facts; see [`10-memory-architecture.md`](./10-memory-architecture.md) §11).

Format (one item per fact-block):

```yaml
::: todo id=t-2026-04-28-research-memory-arch
description: Synthesize memory research from the two agents and write 10-memory.md
status: in_progress | done | blocked | abandoned
priority: P0 | P1 | P2 | P3
created_at: 2026-04-28
deadline: 2026-04-29        # optional
estimated_cost_usd: 0.30    # optional
assigned_to: main | <team-slug> | <instance-id>
parent_event: <event-id>    # what triggered this TODO
:::
```

Read paths:
- Eagerly loaded into the agent's system prompt (axis #5: Active context) so the agent always knows what's in flight
- Queryable via the SQLite mirror for filters ("show me P0 items deadlined this week")

Write paths:
- The agent itself adds TODOs (during a turn, when it identifies a follow-up)
- Sub-instances bubble up `outstanding_todos` in their `BranchSummary`, which the parent commits
- The user adds TODOs via `spren todo add` CLI or UI

This is the **planning surface** of the agent — without it, long-running task management depends on memory recall, which is fragile.

## Sandbox and filesystem permissions

All agents — main, team managers, working instances, AND user-defined workflow agents — share a sandboxed filesystem under `~/.spren/sandbox/`. Permission tiers are enforced **at the application layer** (a `SandboxFilesystem` wrapper that all FS access goes through), not at the OS layer. This is concrete, auditable, cross-platform, and cheap to evolve.

```
~/.spren/sandbox/
├── shared/                        # cross-cutting, owned by main agent
│   ├── memory/                    # main: RW (via consolidation); user: RW (via $EDITOR); team managers + instances: R
│   │   ├── profile.md
│   │   ├── projects/
│   │   ├── relationships.md
│   │   ├── procedures/
│   │   ├── decisions/
│   │   ├── journal/YYYY-MM-DD.md
│   │   └── active_todos.md        # long-term TODO / planning list (markdown source of truth)
│   ├── workflows/                 # snapshot store; main: RW; sub-instances: R only
│   ├── personas/                  # main agent persona (team personas live under teams/<slug>/persona.yaml)
│   │   └── main.yaml              # main agent: RW (with hard rails); user: RW via UI
│   ├── skills/                    # capability bundles loadable into instances; populated from v0.4
│   │   ├── workflow-inspection.yaml
│   │   ├── communication.yaml
│   │   ├── web-research-with-citations.yaml
│   │   └── ...
│   ├── rules.md                   # operational rules / doctrine; main + user: RW
│   └── active_context.md          # situational awareness; main: RW; others: R
├── teams/                         # one dir per user-defined team (Research, Admin, etc.)
│   └── <team-slug>/               # e.g. research, admin, personal
│       ├── persona.yaml           # team manager's persona
│       ├── memory/                # team manager: RW; team's working instances: R
│       └── instances/<instance_id>/
│           └── scratchpad/        # the working instance: RW; team manager: R
├── runs/                          # one dir per Marsys workflow run
│   └── {run_id}/
│       ├── trace.ndjson
│       ├── workflow.json
│       └── artifacts/             # workflow agents: RW within run; meta-agents: R
├── archive/                       # closed runs, archived journals; main: RW; others: R
├── inbox/                         # event queue durable storage (for crash recovery)
│   └── pending/
└── logs/
    └── audit.log                  # every cross-tier write logged
```

Permission rules:

- **Main agent**: full RW everywhere except its own persona file (which user-confirms changes via hard rail) and the audit log (append-only)
- **Team managers**: RW within their `teams/<team-slug>/` tree; R on `shared/`; can request main agent to write `shared/`
- **Working instances**: RW within their `instances/<id>/scratchpad/`; R on `shared/` and on their team's `memory/`
- **User-defined workflow agents**: RW within their run's `artifacts/`; R on `shared/`
- **User**: RW on everything via CLI (`spren memory edit ...`) and UI

**On the word "sandbox":** Spren ships across multiple distribution channels (native installer, brew / winget / apt, npm, pipx, Docker) — the sandbox path resolves via `platformdirs` for native channels and is `<container>/data/sandbox/` in Docker mode. Permissions are enforced by the `SandboxFilesystem` wrapper at the application layer regardless of channel.

All agents run in the same Spren process — no per-agent or per-workflow container isolation in v0.3 / v0.4. A later release (likely v0.5+) may introduce stronger isolation (e.g., per-tool-execution sandboxing) when the meta-agent gains shell-execute powers via code-generated tools. The single-process logical sandbox is sufficient for the cost / scope of v0.3 / v0.4.

## Cost ceiling enforcement at the runtime level

Always-on with frontier models is unbounded spend. The "always-on" feeling comes from UX (memory + notifications + visible activity) — NOT from continuous thinking. Hard rules:

- **Per-day budget cap** (settings, default $10/day for v0.3)
- **Per-think token cap** (settings, default 50K tokens per agent turn; large enough for substantive reasoning)
- **Cheap model for routine work; expensive model for hard reasoning** — the agent picks via a `model_for_thinking_about(complexity_estimate)` helper
- **Refuse-to-act when over budget** — agent surfaces this to the user as an inbox item; doesn't just keep spending

**SP-013 (new principle):** Cost ceiling is load-bearing. Implementers MUST NOT add code paths that bypass the per-day budget cap. When the cap is hit, the agent stops processing all events except the most critical (workflow failures, user-direct messages); user is notified.

## The supervision surface

The agent has many things to keep an eye on, not just sub-agents. Real-world examples:

- **Time-based commitments:** "remind me at 5 PM about the dentist", recurring "morning briefing every weekday at 7 AM", a deadline mentioned in passing yesterday
- **External integration health:** Telegram polling connection drops, cloudflared tunnel goes down, Anthropic returns 429 rate-limit errors, an LLM model gets deprecated
- **System health:** disk filling with trace files, database growing past threshold, memory pressure
- **Workflow run oversight (NOT just sub-agents):** a scheduled workflow has failed 3× this week with the same error, a workflow's average cost has drifted upward, a long-running BrowserAgent is in step 50
- **Pending decisions / follow-ups:** suggestion sent to user 6 hours ago with no response — escalate or remind?
- **Memory hygiene:** pending-facts queue grew 50 items in an hour (consolidate early); a stable fact's volatility threshold passed; markdown KB drifted from SQL index
- **Cost / budget:** spend rate is 8× normal this hour; approaching daily cap
- **Sub-agent oversight (v0.3+):** timeout, freeze, runaway loop, off-track reasoning

These all need attention, but they need *different* mechanisms. Designing them all as "the agent polls everything every minute" produces unbounded cost. Designing them all as "wait for an external event" misses silent failures.

The supervision surface is **five producer categories, all feeding the same inbox.** The agent only ever wakes on inbox events. Different producers handle different concerns at different cost profiles.

### 1. External event sources

User messages (web/CLI), channel messages (Slack/Telegram/Discord — v0.3+), webhook fires (v0.3+), workflow lifecycle events (`WorkflowStarted`/`Finished`/`Failed`/`Suspended`).

Cost: zero when nothing inbound; proportional to inbound volume otherwise.

### 2. Time scheduler

One-off and recurring scheduled events. Backed by APScheduler in the daemon. Producers:
- **Agent-scheduled** via `schedule_event(at, payload)` and `schedule_recurring(cron_expr, payload)` tools — for reminders, deadlines, agent-driven follow-ups
- **User-scheduled** via UI / CLI — for cron-style workflow runs (v0.3) and recurring routines like the morning briefing
- **System-scheduled** — daily archival job, periodic full-scan of markdown KB, etc.

Cost: zero when nothing is due; one event-fire = one inbox event.

### 3. System watchers

A small, **curated catalog** of background tasks that monitor one condition each and emit typed events when their condition trips. Watchers run as Python coroutines in the daemon. They are **mechanical** — no LLM calls — and emit events only when something is worth waking the agent for.

**v0.3 watcher catalog** (ships in Session 07):

| Watcher | Cadence | Emits | Trigger |
|---|---|---|---|
| `BudgetWatcher` | every 1 min | `BudgetThresholdCrossed` | spend reaches 75% / 90% / 100% of daily cap |
| `DiskWatcher` | every 5 min | `DiskLow` | free space < 5 GB on `~/.spren/` partition |
| `ProviderHealthWatcher` | rolling window over recent LLM API calls | `ProviderRateLimited`, `ProviderUnavailable` | N consecutive 429s; sustained 5xx |
| `PendingFactsWatcher` | every 5 min | `ConsolidationDue` | queue grows past threshold (independent of the 24h+5-session time trigger) |
| `IndexDriftWatcher` | every 15 min | `IndexDriftDetected` | full-scan finds SQL ↔ markdown disagreement |
| `WorkflowRunWatcher` | per active run, every 30 sec | `WorkflowRunStalled`, `WorkflowRunOverBudget`, `WorkflowRunOvertime` | run idle > timeout, cost > per-run cap, wall-clock > limit |

**v0.3 watcher catalog** (added when their parent subsystems ship):

| Watcher | Cadence | Emits | Triggers |
|---|---|---|---|
| `SubInstanceLivenessWatcher` | per active sub-instance, ping every 30 sec | `SubInstanceUnresponsive`, `SubInstanceTimeout`, `SubInstanceBudgetExhausted`, `SubInstanceRunawayLoop` | missed pings; hard limits hit |
| `ChannelHealthWatcher` | per channel, every 1 min | `ChannelDisconnected`, `ChannelLagging` | connection state, polling lag |
| `TunnelWatcher` | every 1 min while a tunnel is up | `TunnelDown`, `TunnelURLChanged` | `cloudflared` exit, URL rotation |
| `VolatilityReverifyWatcher` | every 30 min | `FactReverificationDue` | scans facts whose volatility threshold has passed |

**Watcher properties:**
- Catalog is **fixed in code per version**. Agents do NOT spin up new watchers at runtime in v0.3/v0.4 (that's a v0.5+ consideration). All watchers are reviewable in source.
- Each watcher has its own cadence; the cadence is configurable in settings, not by the agent.
- A watcher's "trip" produces an inbox event with priority appropriate to severity (e.g., `BudgetThresholdCrossed(100%)` is P0; `IndexDriftDetected` is P3).
- Watchers are **mechanical**: no LLM calls in the watcher itself. The cost of supervision is bounded.
- Watcher state (last-checked, last-fired, current measurement) is queryable by the agent and visible in the UI.

Cost: ~zero (background tasks doing cheap I/O / SQL queries). LLM cost only when a watcher fires AND the agent processes the resulting inbox event.

### 4. Heartbeat — proactive review cadence

The agent's scheduled spontaneity. Default 30 min. Open-ended in *content* (the agent reasons about anything that needs reasoning), bounded in *cadence*. Detailed in the "Heartbeat — the proactive thinking cycle" section immediately below.

The heartbeat handles what watchers can't catch — **patterns and drift that require reasoning across history**:
- Workflow X failing repeatedly → the watcher catches each individual stall, but only the heartbeat notices the pattern
- Cost drifted upward over weeks → the budget watcher catches today's anomaly; the heartbeat catches the slow drift
- Awareness state has drifted (user said OOO but is messaging) → no watcher can detect this; the heartbeat reasons about it
- Pending suggestion to user has been unanswered for hours → no watcher; heartbeat decides whether to follow up

### 5. Sub-instance completions (v0.3+)

`InstanceCompleted` and `InstanceFailed` events from sub-instances reporting back. Already covered in the Working Instances section.

### How the agent uses these mechanisms

- For a **specific time** (reminder, deadline, supervisory check on a sub-instance) → use the time scheduler
- For a **standing condition** (budget, disk, integration health) → it already has a watcher; just react to the events
- For **pattern-detection across multiple subsystems** → handled in the heartbeat
- For **work the agent decides to delegate** → spawn a sub-instance (v0.3+); supervised by the sub-instance liveness watcher + optional agent-scheduled cognitive checks

The agent does NOT invent its own polling cadences. It uses the existing producers.

## Heartbeat — the proactive thinking cycle

The heartbeat is the agent's scheduled spontaneity: a configurable cadence (default 30 min) at which the agent wakes for proactive review, even when no event has arrived.

The heartbeat is NOT bookkeeping. It IS substantive thinking. On each tick, the agent runs a "review turn" with a system prompt extension that asks it to:

1. **Catch up:** read since-last-tick activity (workflow events, channel messages already processed, sub-instance completions). Use the session log via `grep` over the `events` table for the time window.
2. **Inspect running work:** for each active workflow run and (in v0.3+) each working instance, ask "is this still on track? am I learning anything from it?"
3. **Review the TODO list:** scan `active_todos.md` for items due, items aging, items blocked. Decide whether to act, ask the user, or defer.
4. **Look for patterns:** repeated failures of the same workflow? Cost drift? An assumption that's been invalidated?
5. **Consider proactive outreach:** is there something the user should know? "Run #47 has failed 3 times today — want me to investigate?" "You haven't checked the workflow you scheduled this morning — it ran successfully." Use the **suggest-with-confirm** flow (see Authority tiers) — surface the suggestion in the inbox, don't just message the user.
6. **Update situational awareness:** if `active_context.md` is stale relative to today's reality (e.g., user mentioned a new project yesterday), draft an update via the consolidation/extraction path.
7. **Update activity log:** emit a "since-last-tick" entry to the activity stream so the user can see what the agent did during the heartbeat.
8. **Reconcile crashes:** if any working instances are marked `interrupted` from a daemon crash, decide whether to retry, abandon, or surface to the user.

Heartbeat budget: the agent's per-day budget cap (SP-013) limits cumulative tick cost. If the budget is exhausted, heartbeats still fire for bookkeeping (no LLM call) but skip the substantive review until the next budget window.

The user controls heartbeat cadence via settings (`heartbeat.interval_minutes`). Common patterns:
- **30 min default** — for everyday use, gives the agent a feel of being alive without burning cost
- **5 min** — when actively monitoring something time-sensitive (the user can tighten temporarily)
- **6 hours** — when the user is away or wants to minimize spend

### Agent-scheduled events (the agent's own deferred wake-ups)

The agent has a `schedule_event(at: datetime, payload: dict)` tool. Calling it enqueues a future event into the inbox. When `at` arrives, the event fires as a regular `ScheduledRun` or `ScheduledReminder` inbox item. This is the bounded, visible way for the agent to say "remind me later" — it does NOT invent timer loops or implicit cadences (per SP-011).

Example: during a heartbeat, the agent decides "the user said the report deadline is Friday at 5 PM; let me check on Thursday morning whether we're on track." It calls `schedule_event(at=Thursday 9 AM, payload={kind: 'reminder', re: 'report-deadline'})`. On Thursday at 9 AM, that event lands in the agent's inbox just like any other.

## Lifecycle and crash recovery

- **Spren daemon starts** → loads main agent's persona + memory + active_context → starts EventIngress → ready
- **Daemon shutdown (graceful)** → drain inbox (or persist remaining events to `inbox/pending/`) → finish active turns → flush memory state → exit
- **Daemon crash** → on next start, durable inbox replayed; in-flight working instances marked `interrupted`; main agent reads `instances/<id>/scratchpad/` and decides whether to retry or close
- **Heartbeat** → fires per the configured cadence; agent runs a review turn (see Heartbeat section above)

**SP-014 (new principle):** Working instances must persist their scratchpad atomically. Every meaningful state transition writes to `instances/<id>/scratchpad/state.json` durably (atomic rename pattern). Crash recovery depends on this.

## Open questions for next round

- **Heartbeat interaction with budget caps:** if the daily budget is exhausted, do heartbeats still fire (with no LLM call, just bookkeeping)? I'd say yes — the user should still see "agent is paused, $X over budget, will resume tomorrow" status.
- **Team manager eviction policy:** when does an idle team manager get unloaded from memory? Currently I'm thinking after 24h of no activity for that team. Settable per team.
- **Per-channel trust scoring:** the security doc covers this in principle, but the concrete trust-score computation model needs design (manual config? learned from usage? heuristic-based?).
- **Skill versioning:** when a skill's YAML changes, do already-running instances reload? Lean: no — instances complete with the skill version they started with; new spawns get the new version.
- **Sub-instance permission inheritance:** does a working instance inherit its spawning agent's permissions, or get its own scoped set? Lean: scoped set, declared by the loading skill — but the skill can request to "inherit parent" if needed. Resolve before implementing.

# 08 — Design Principles

Cross-cutting invariants for Spren. Violating one of these = ASK FIRST. These are the spren-specific equivalents of the framework's DP-001..DP-007.

| ID | Principle | Applies to | Invariant |
|----|-----------|------------|-----------|
| SP-001 | **Framework purity preserved** | Any change to `packages/framework/src/marsys/...` | Spren must NOT modify TRUNK-CRITICAL framework files. New Spren code lives in `packages/spren/src/spren/`, `apps/web/src/`, `apps/desktop/`, or other non-framework paths. See `/CLAUDE.md`. |
| SP-002 | **Localhost-only by default** | `src/spren/server.py` | The HTTP server binds to `127.0.0.1`, never `0.0.0.0`. Per-launch auth token required on every request. The only exception is when the user explicitly opts into `marsys expose` (v0.3) for tunnel-based external access. |
| SP-003 | **One transport per concern** | `03-api-design.md` | REST for CRUD; SSE for server→client event streams; one POST for mid-run user interaction. NO WebSocket. NO gRPC. Don't add a transport unless an existing one cannot meet the requirement. |
| SP-004 | **AG-UI as the wire schema for run events** | EventBus → SSE translator | All server→client events sent over SSE conform to the AG-UI protocol event taxonomy. Don't invent a parallel schema. If AG-UI lacks an event we need, propose it upstream OR use AG-UI's `Custom` event type — don't add ad-hoc shapes. |
| SP-005 | **Pydantic is the source of truth for types** | All HTTP boundaries and persisted shapes | Types are defined in Pydantic; TS types are generated from JSON Schema at build time via `datamodel-code-generator`. Hand-written TS types for shared shapes are FORBIDDEN — they will drift. |
| SP-006 | **Single-version code** (no backward-compatibility shims) | All spren code | When we change the shape of a feature, we change it everywhere. No `if version_v1 do X else do Y` paths. Stored data is migrated via one-shot migration scripts in `src/spren/migrations/`. |
| SP-007 | **No mocked features in product code** | All spren code | Test fixtures at external boundaries (LLM responses, network) are fine and live in `tests/fixtures/`. In-codebase placeholder feature implementations are FORBIDDEN — they get treated as real by future agents and rot. Either the feature is built or it is not in scope. |
| SP-008 | **Distinct from MARSYS Studio** | All product decisions | When a feature is ambiguous between "local-single-user" and "hosted-team", default to the local-single-user interpretation. Multi-user, RBAC, sharing, hosted concerns are explicitly out of scope (they belong to Studio). See [`01-system-context.md`](./01-system-context.md). |
| SP-009 | **Run snapshots are immutable** | `src/spren/storage/...` | When a run starts, the workflow definition is snapshotted into the run record. Subsequent edits to the workflow do NOT alter past runs. This means traces are stable artifacts, not live views. |
| SP-010 | **Cost is tracked from token zero** | `06-observability.md` integration | Every LLM call's tokens are recorded in the trace AND aggregated into a per-run cost figure using a maintained provider rate table. Cost is a first-class signal in the UI from the first Spren release (v0.3), not a later retrofit. |
| SP-011 | **Bounded triggers** — the agent wakes only via the inbox, and the inbox has a finite, curated set of producers | Meta-agent runtime — `09-meta-agent.md` § "The supervision surface" | Allowed producers: (1) **external sources** — user messages, channel messages, webhook fires, workflow lifecycle events; (2) **time scheduler** — one-off and recurring scheduled events (created by agent, user, or system); (3) **system watchers** — mechanical background tasks emitting typed events when a condition trips (catalog fixed per version: budget, disk, provider, queue depth, drift detection, workflow run liveness in v0.3; sub-instance liveness, channel health, tunnel, volatility re-verification added in v0.4); (4) **heartbeat** — the agent's proactive review cadence (configurable); (5) **sub-instance completions** (v0.4+). Idle inbox + nothing pending = zero LLM calls. Forbidden: polling loops finer than a watcher's defined cadence, implicit cadences invented at runtime, agents creating new watchers at runtime (the watcher catalog is curated). The heartbeat IS spontaneous thinking — but on a defined cadence the user can see and control. |
| SP-012 | **Hard rails always confirm** | Authority tiers in `09-meta-agent.md` | Destructive or spend-significant actions (`delete_workflow`, `revoke_channel`, `modify_settings`, ending a run, touching secrets, any spend over the per-action budget cap) ALWAYS go through user confirmation, regardless of standing approvals. Implementers MUST NOT add code paths that bypass hard rails. |
| SP-013 | **Cost ceiling is load-bearing** | Meta-agent runtime + budget enforcement | Per-day budget cap is enforced at the runtime level, not advisory. When the cap is hit, the meta-agent stops processing all events except the most critical (workflow failures, user-direct messages); user is notified. Implementers MUST NOT add code paths that bypass the per-day budget cap. |
| SP-014 | **Working instances persist their scratchpad atomically** | `09-meta-agent.md` § Lifecycle | Every meaningful state transition in a working instance writes to `~/.spren/sandbox/teams/<m>/instances/<id>/scratchpad/state.json` durably (atomic rename pattern). Crash recovery depends on this. |
| SP-015 | **Untrusted-channel writes never touch memory live** | Memory write paths — `10-memory-architecture.md` | Facts extracted from non-user-direct sources (Slack, Discord, Telegram, webhooks, sub-agent outputs) MUST be queued in `pending_facts` and committed only by the async consolidation pass — never live-written to the markdown knowledge base. This is the primary defense against memory poisoning attacks (MINJA / MemoryGraft show >95% injection success without this defense). |
| SP-016 | **Markdown is the source of truth for memory** | `10-memory-architecture.md` | The user-editable markdown knowledge base is authoritative; the SQLite + vector index is rebuilt from it. The user can `vim` the markdown and the system MUST treat that edit as authoritative on next index rebuild. Implementers MUST NOT add a code path where the index can disagree with the markdown indefinitely. |
| SP-017 | **Forget tombstones, never deletes** | Memory user-control — `10-memory-architecture.md` | When the user calls `forget`, the fact entry is moved to `archive/forgotten/` with a tombstone in the live file. The consolidation pass MUST respect tombstones — it must not re-extract a tombstoned fact from the session log. True deletion would let the agent re-learn forgotten facts; we forbid it. |
| SP-018 | **Framework knows nothing of Spren** | Boundary between `packages/framework/` and `packages/spren/` (+ `apps/`) | The seam between framework and Spren is exactly three doors: (1) `Orchestra.run(topology, task) → OrchestraResult` for finite execution, (2) `EventBus.subscribe(event_type, listener)` for in-flight lifecycle events, (3) the generic `TelemetrySink` interface for run-event forwarding to external observability backends. The framework MUST NOT import any Spren type. The framework MUST NOT contain code paths that special-case Spren. New seams require explicit design discussion and ADR. Multi-consumer features (Cloud, Studio, third-party) prove out the boundary in practice. |
| SP-019 | **API is the single source of truth** | All clients (Tauri webview, browser tab, TUI, future) consuming `packages/spren/` | Every product surface — desktop GUI, browser GUI, TUI, scripted Python adapter — consumes the same FastAPI REST + SSE + POST surface. Zero UI logic in the backend. Zero duplicated computation in the clients. If a client needs derived state, the backend computes and exposes it; the client renders. Pydantic models are the contract; TypeScript is generated; Python types are imported directly. Hand-written client-side types are FORBIDDEN. |

## Anti-patterns to actively avoid

- **Mock services that "look real but return canned data"** for end-to-end testing. Use real services with recorded fixtures at the boundary. (SP-007)
- **Version-mode feature flags** that keep two implementations alive in production code. (SP-006)
- **Hand-rolling a TypeScript interface that mirrors a Pydantic model.** Generate it. (SP-005)
- **Hand-rolling a Python TUI client type that mirrors a Pydantic model.** Import it directly. (SP-019)
- **A new WebSocket endpoint "because real-time is faster".** SSE is real-time enough; reach for a new transport only when SSE truly cannot do the job. (SP-003)
- **"Reusable" abstractions added speculatively** before the second use case actually exists.
- **Importing Spren types from inside `packages/framework/`.** Forbidden. (SP-018)
- **Adding a Spren-specific code path inside the framework** ("if running under Spren, do X"). Forbidden. (SP-018)
- **A client that recomputes state the backend already exposes.** Move the computation to the backend. (SP-019)

## When to ASK FIRST

Always escalate when:
- Adding a new transport (WebSocket, gRPC, message bus)
- Modifying any TRUNK-CRITICAL framework file
- Changing the AG-UI event mapping (SP-004)
- Adding multi-user / RBAC / sharing concepts (SP-008)
- Adding a feature flag to keep two code paths alive (SP-006)
- Creating an in-codebase mock or stub of a real feature (SP-007)
- Adding a polling loop that wakes the meta-agent more often than the heartbeat (SP-011)
- Introducing a code path that could bypass hard rails or the per-day budget cap (SP-012, SP-013)
- Live-writing to memory from any non-user-direct source (SP-015)
- Storing memory anywhere other than the markdown KB as authoritative (SP-016)
- Hard-deleting a fact instead of tombstoning (SP-017)
- Adding a fourth seam between framework and Spren beyond `Orchestra.run()` / `EventBus.subscribe()` / `TelemetrySink` (SP-018)
- Putting UI logic in the backend OR letting a client recompute backend-derivable state (SP-019)

## Where these come from

- SP-001 enforces `/CLAUDE.md` § 4
- SP-002 from research: Tauri docs explicitly warn about localhost binding without auth
- SP-003 from research: AG-UI / Vercel AI SDK / Bedrock AgentCore all picked SSE; SSE outscales WS for our workload
- SP-004 from research: AG-UI is the emerging 2026 standard with major adopters (LangGraph, CrewAI, Pydantic AI, Bedrock)
- SP-005 from research: `datamodel-code-generator` is the robust 2026 path; hand-written TS drifts
- SP-006, SP-007 directly from user-stated rules during planning
- SP-008 to protect proprietary MARSYS Studio's product space
- SP-009 from observability best practices: traces must be stable retroactive records
- SP-010 from product reality: runaway LLM costs are the #1 user fear and need first-class visibility
- SP-011 from cost-ceiling math AND debuggability: "always thinking" with frontier models is unbounded spend; ad-hoc polling loops are also undebuggable (the user can't see why the agent woke up). The bounded-triggers rule keeps wake-up causes finite, visible, and user-controllable. The five producer categories (external sources, time scheduler, system watchers, heartbeat, sub-instance completions) cover the full supervision surface — sub-agent oversight, time-based commitments, integration health, cost monitoring, drift detection, pattern recognition. Each producer is curated per version; the agent doesn't invent new ones at runtime.
- SP-012, SP-013 from trust calibration: the meta-agent's central role means a bad action loses user trust permanently — hard rails are non-negotiable
- SP-014 from operational reality: long-running daemons crash; recovery requires durable scratchpad
- SP-015 from security research: MINJA (NeurIPS 2025) and MemoryGraft show >95% memory-injection success against production agents without the queue-then-consolidate defense
- SP-016 from the Letta filesystem benchmark: plain markdown + grep beat their own graph memory (74.0% vs 68.5% on LoCoMo); user-editability is the killer feature of file-based memory
- SP-017 from honest-forgetting: true deletion + retained event log = re-learning of forgotten facts; tombstones prevent this
- SP-018 from multi-consumer reality: the framework serves Spren, MARSYS Cloud, MARSYS Studio, and third-party Python users. A clean seam keeps each consumer independently shippable; framework features that special-case any one consumer break the contract with the others. The three doors (`Orchestra.run()`, `EventBus.subscribe()`, `TelemetrySink`) are sufficient for every consumer use case proven so far; new seams require evidence of multi-consumer demand.
- SP-019 from multi-client reality: the desktop GUI (Tauri webview), browser GUI (system browser tab), TUI (Textual), and Python adapter SDK all consume the same FastAPI surface. Each client renders the same domain in a different chrome. Putting UI logic in the backend creates per-client divergence; recomputing in clients creates inconsistency between clients. The API is the contract; renderers differ.

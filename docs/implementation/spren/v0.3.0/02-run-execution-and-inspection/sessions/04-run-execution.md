# Spren Session 04 — Run Execution + AG-UI Streaming + Cost

> Session plan. The implementer reads this as the primary source of truth for what Session 04 ships, how it integrates with framework Session 06 (AG-UI translator), and what's in vs out of scope. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / DELETE, the user journeys that extend Bundle A's Session 03 build flow into a runnable workflow, wireframes for the small Session 04 UI lift (Run button on canvas + minimal `/runs` route), data-model considerations, the locked decisions from the architect-stage Q&A, polish items the implementer addresses in-session, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./04-run-execution/acceptance.md`](./04-run-execution/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Design language anchor: [`../../01-visual-builder/sessions/03-visual-builder.md`](../../01-visual-builder/sessions/03-visual-builder.md) §9 Design System. Session 04 reuses Session 03's palette, typography, components, motion tokens, and Spren orb without redefining them. New surfaces added in Session 04 (Run button states, the minimal `/runs` route, the per-run progress indicator, completion toast) follow the same design language. Visual anchor for the Spren orb (whose reactive states change during a run): [`../../01-visual-builder/assets/spren-inspiration.png`](../../01-visual-builder/assets/spren-inspiration.png).

---

## 1. Bundle position + tier

- **Bundle**: B — Run Execution + Inspection (Sessions 04 + 05). Session 04 ships the run execution backend + the canvas Run button + a minimal `/runs` list view. Session 05 ships the full trace viewer UI, file upload + attachment endpoints, and run history filtering UI.
- **Bundle demo gate**: user builds a workflow in Bundle A's canvas, clicks Run, watches the workflow execute with the Spren orb shifting to `speaking` while each agent's turn is in flight (per-turn streaming — token-level streaming is a future framework session, see §15), sees the token count climb as turns complete and cost computed on completion, clicks through to see the run in `/runs` with status + total cost + duration. Trace viewer (nested spans, per-span cost) lands in Session 05; Session 04's `/runs/{id}` view is a thin row-from-table page with a link to the (Session 05) full trace viewer.
- **Tier**: HIGH-tier with implementer-side research. New surface area is moderate: AG-UI consumption pattern (wrapping the framework's `AGUIEventStream` async iterator over an SSE HTTP endpoint), SSE through Tauri webview reliability, cost-rate table format + provider lookup, `runs` table lifecycle + cancellation semantics with the new `cancelling` interim status, workflow-snapshot immutability at run start (SP-009), `WorkflowDefinition → Topology + AgentRegistry` materializer (load-bearing — see §5). Smaller new tech surface than Session 03 (no new client frameworks); larger backend surface.
- **Approval gate**: implementer-side; surfaces to user only if a decision in §15 needs the user's input or a polish item in §11 reveals a hidden architectural ambiguity.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 01 (foundation) | shipped | FastAPI sidecar, auth, capabilities. |
| Spren Session 02 (CRUD + types) | in flight | `workflows` table + Pydantic models. Session 04 reuses the type-generation pipeline + `GET /v1/tools` endpoint. |
| Spren Session 03 (visual builder) | shipped | Canvas at `apps/web/src/routes/workflows/$workflowId.tsx`, design system, Spren orb (`apps/web/src/components/Spren/`), `PresenceOrb` (hard-coded `state="idle"` today; Session 04 extends with optional `state?` prop), cmdk command palette (`stores/useCommands.ts`). Session 04 creates the canvas Run button fresh — Session 03's toolbar shipped Pattern modal + Save only, no Run button stub exists. |
| Framework Session 01 (NDJSON streaming tracing writer) | shipped | Backs the `trace.ndjson` per-run file used by the Session 05 trace viewer. Session 04 does NOT consume `trace.ndjson` for SSE replay (the file holds closed framework spans, NOT AG-UI events) — replay is via the in-memory buffer described in §3 below. |
| Framework Session 02 (`TelemetrySink` ABC + `SecretRedactor`) | shipped on `marsys-tracing-work`; not yet on `feature/spren-umbrella` | Not consumed by Session 04 directly (Spren v0.4 consumes via `SprenTelemetrySink`). The redactor runs at the framework's fan-out boundary; the AG-UI translator subscribes to `EventBus` directly (peer to `TraceCollector`). The framework code lands on this branch when `feature/tracing-streaming` merges to main. |
| **Framework Session 06 (AG-UI translator)** | **full brief locked at `marsys-tracing-work/.../06-aggui-translator.md`; implementation in flight in parallel** | Ships `marsys.coordination.aggui` exposing `AGUIEventStream(translator)` (constructor takes the translator directly, NOT `(orchestra, run_id)`), `aggui_event_to_sse(event)`, `AGGUIConfig(enabled: bool = False)`, `MarsysRunState`. Translator is built inside `Orchestra._wire_event_bus()` when `ExecutionConfig.aggui.enabled=True`; exposed as plain attribute `orchestra.aggui_translator`. `GenerationEvent` is mapped to `Custom("marsys.generation.metadata")` — Spren's cost calculator consumes that Custom event payload. Backpressure: drop-newest + `Custom("marsys.stream.lagged")`. Optional dep `marsys[aggui]` pulls `ag-ui-protocol==0.1.18` + `jsonpatch>=1.33`. Spren 04's AG-UI consumer code (~15% of session scope) integrates when Framework 06 merges; the rest of Spren 04 builds independently against locked seams. |
| **Framework Session 07 (cancel API)** | **scoped (new) — stub created alongside this Spren session** | Ships `Orchestra.cancel_session(session_id, force_after: float = 5.0) -> None`. Mirrors `Orchestra.pause_session()` shape: looks up the live `Orchestrator`, calls `await orchestrator.quiesce()` (already shipped by Session 03) inside `asyncio.wait_for(timeout=force_after)`; if the wait times out, calls `task.cancel()` on the run task and waits up to 2s more for terminal. Logs WARN if the task still hasn't honored cancel after both phases (rare; `convergence_timeout` / `branch_timeout` / `step_timeout` + the per-run budget cap (SP-013) are the eventual backstop). ~50 LOC additive function on `Orchestra`; no TRUNK-CRITICAL changes. Stub at `docs/implementation/framework/sessions/v0.3.0/07-cancel-session-api.md`; full brief authored by the framework architect. Spren 04's cancel endpoint integrates when Framework 07 merges. |
| `EventBus` + `Orchestra` (existing framework) | live | Spren constructs `Orchestra` per run; passes a frozen `Topology` + `AgentRegistry` materialized from `WorkflowDefinition`; passes `ExecutionConfig(aggui=AGGUIConfig(enabled=True), storage_backend=FileStorageBackend(<data-dir>/data/runs))`; the framework wires the AG-UI translator inside `_wire_event_bus()`. |
| Spren Session 02's `GET /v1/tools` | dep | Needed for cost-rate table cross-check (tool calls' cost is on the parent generation, not the tool — but the tool catalog is referenced when reading cost-by-tool in Session 05). Not blocking. |

Session 04 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). The only framework consumption is `Orchestra.run()` (with a materialized `Topology` + `AgentRegistry`), the `orchestra.aggui_translator` attribute (set by Framework 06 inside `_wire_event_bus()`), and `Orchestra.cancel_session()` (Framework 07). The Spren-side draft sweeper from Session 03 stays.

## 3. What ships in Session 04

Backend surfaces (the bulk of the session):

- **REST run execution endpoint**: `POST /v1/runs` accepting `{workflow_id, task_input: {text, attachments: []}}`. Race-free startup: (1) validate the workflow exists + is not archived; (2) insert `runs` row with `status=queued`; (3) freeze workflow `definition` as `<data-dir>/data/runs/{run_id}/workflow.json` (SP-009); (4) materialize `Topology + AgentRegistry + ExecutionConfig(aggui=AGGUIConfig(enabled=True))` via `runs/materialize.py`; (5) construct the `Orchestra` and call `_wire_event_bus()` synchronously so `orchestra.aggui_translator` is set before the response returns; (6) register `ActiveRun(orchestra, task, started: asyncio.Event, replay: deque(maxlen=1024))` in the module-level `_active_runs` dict; (7) schedule the lifecycle task; (8) return `{run_id, status: "queued"}`. SSE subscribers attaching after step 6 always find a wired translator; subscribers attaching during the brief sync wiring window do `get(run_id) or 404` then `await started.wait()` (2s timeout). Attachments handling deferred to Session 05 (Session 04 accepts the field but rejects non-empty arrays with 400 + `code: "ATTACHMENTS_NOT_YET_SUPPORTED"`).
- **REST run row read**: `GET /v1/runs/{id}` returns the `runs` table row.
- **REST run list**: `GET /v1/runs` with cursor pagination + `?workflow_id=` + `?status=` + `?since=`. Filter pattern reuses Session 02's cursor approach (`packages/spren/src/spren/storage/workflows.py:108-114`).
- **Aggregate run-events SSE stream**: `GET /v1/runs/events` — server-pushed SSE emitting `RunCreated` / `RunUpdated` / `RunFinished` / `RunCancelled` events for ALL runs (one stream for the whole list page). The lifecycle coordinator publishes to a `RunsBroker` singleton (`packages/spren/src/spren/runs/broker.py`, ~40 LOC, hand-rolled — does NOT reuse the framework's `EventBus` per SP-018); the endpoint subscribes per-client with a bounded `asyncio.Queue(maxsize=256)`, drop-oldest on overflow with a `STREAM_LAGGED` marker on the next put. Client `/runs` list mounts one stream on first paint and updates rows in place. Filter chips apply client-side. Replaces per-card SSE (which caps the list at ~3-5 simultaneous running cards). Each event payload carries `schema_version: int = 1`.
- **REST cancel endpoint**: `POST /v1/runs/{id}/cancel`. State transitions: `queued` → immediate `cancelled`; `running` → `cancelling` (interim) → calls `Orchestra.cancel_session(session_id, force_after=5.0)` (Framework 07; framework runs `quiesce()` for up to 5s, then `task.cancel()` for up to 2s more) → `cancelled` once the framework returns; `paused` → 409 (paused ships in v0.4); `cancelling` or terminal → 409. Server-side watchdog: if the framework call doesn't return within 10s total, log WARN, mark the row `cancelled` regardless, emit `Custom("marsys.cancel.timeout")` on the run's SSE stream — the leaked task is bounded by the per-run budget cap (SP-013) plus the framework's own timeouts. The handler returns `200` with the updated row immediately after the framework call returns; clients drive the user-visible 5-second countdown UX in §8.7. State transitions per [`../../../../../architecture/spren/03-api-design.md`](../../../../../architecture/spren/03-api-design.md) §Cancellation, updated to add the `cancelling` interim status.
- **SSE event stream**: `GET /v1/runs/{id}/events` opens an SSE connection that yields AG-UI events from `AGUIEventStream(orchestra.aggui_translator)`. Encodes each event via `aggui_event_to_sse(event)` from `marsys.coordination.aggui`. Reconnect-tolerant via an in-memory `collections.deque(maxlen=1024)` replay buffer per active run (held on the `ActiveRun` registration record): `Last-Event-ID` header → binary-search the deque by ULID `event_id` → replay matching tail → switch to live consumption. If `Last-Event-ID` is older than the deque's oldest entry, emit a single `Custom("marsys.stream.gap")` so the client knows to refresh its row state via REST and continue from now. Cold reads (run already terminated; no `ActiveRun`) return the `runs` row via the REST endpoint instead — the SSE endpoint returns `204 No Content` for terminal runs (Session 05's trace viewer is the reconstruction surface). Implemented at `packages/spren/src/spren/runs/sse.py`. Uses `sse-starlette`'s `EventSourceResponse` for keepalive + disconnect detection + `Last-Event-ID` header parsing.
- **Cost rate table + per-run aggregation**: `packages/spren/src/spren/cost.py` consumes `Custom("marsys.generation.metadata")` events from the AG-UI stream (per Framework 06's mapping for `GenerationEvent` — payload is `{model, provider, prompt_tokens, completion_tokens, reasoning_tokens, finish_reason}`). Looks up `(provider, model)` in `packages/spren/src/spren/cost_rates.py` (Python module, NOT YAML — type-checked, no parser dep, autocomplete; `RATES: dict[tuple[str, str], CostRate]` + `LAST_UPDATED: date` constants); computes `cost = (prompt × in_rate + completion × out_rate + reasoning × reasoning_rate) / 1_000_000`. Aggregates into `runs.total_cost_usd` + `runs.total_tokens_input` + `runs.total_tokens_output` after each Custom event and on terminal. Missing rate (unknown model) → log WARN naming the model + provider, emit zero. Initial population: anthropic + openai + openrouter + google + xai (the five providers in `marsys.models.ApiProvider` enum that have published per-token pricing; OAuth providers `openai-oauth` and `anthropic-oauth` are intentionally absent — they share rates with their non-OAuth counterparts and we cross-reference). Stale-rate warning: log WARN on daemon start if `LAST_UPDATED` is older than 90 days.
- **`runs` table lifecycle handler**: async coordinator in `packages/spren/src/spren/runs/lifecycle.py`. Owns `queued → running → cancelling? → succeeded|failed|cancelled` transitions; tracks `_active_runs: dict[str, ActiveRun]` (module-level) where each entry holds `{orchestra, task, started: asyncio.Event, replay: deque}`; awaits `Orchestra.run()`; persists `total_duration_ms` / `total_steps` / `final_response` / `error` on completion; deregisters from `_active_runs` on terminal. Mirrors `workers/draft_sweeper.py`'s module-pure idiom.
- **Spec → runtime materializer**: `packages/spren/src/spren/runs/materialize.py` (load-bearing, ~300-500 LOC, was invisible in the original plan). Converts `WorkflowDefinition` (Spren's Pydantic mirror) into the framework's runtime types: `TopologySpec → marsys.coordination.topology.core.Topology`, `dict[str, AgentSpec] + ModelConfigSpec → AgentRegistry` of materialized `BaseAgent` instances with API keys resolved from the secrets store, tool name strings → callables via `marsys.environment.tools.AVAILABLE_TOOLS`, assembled `ExecutionConfig(aggui=AGGUIConfig(enabled=True), tracing=TracingConfig(...), ...)`. Returns a `RuntimeBundle` dataclass consumed by `runs/lifecycle.py`. v0.4 `SprenTelemetrySink` and the meta-agent will reuse this materializer; if a second consumer materializes, it relocates to `packages/spren/src/spren/runtime/materialize.py`.
- **`trigger` field**: Session 04 only emits `manual` for the canvas Run button. Other values (`scheduled`, `webhook`, `messenger:*`) are accepted by the schema for forward-compatibility but rejected at the API surface in v0.3 (`code: "TRIGGER_NOT_YET_SUPPORTED"`).
- **`schema_version: int = 1`** on every cross-boundary Pydantic payload (`RunCreate`, `RunRead`, `RunListItem`, the aggregate SSE event union) per CLAUDE.md "Framework v0.4 patterns." Establishes the convention on Spren's side for the rest of v0.3.

UI surfaces (small, additive on top of Session 03):

- **Canvas Run button**: NEW component at `apps/web/src/components/RunButton.tsx`. Session 03's canvas toolbar ships Pattern modal + Save only; Session 04 adds the Run button. Slots into the toolbar at `apps/web/src/routes/workflows/$workflowId.tsx:467-499` after the Save button. States: `idle` (shows `Run`) → `submitting` (200ms spinner) → `running` (shows `Cancel · 0s · 0t`, increments per `Custom("marsys.generation.metadata")` Custom event from the SSE stream) → `cancelling` (shows `Cancelling… 5` countdown, see §8.7) → terminal (toast; button returns to `idle`). State driven by a jotai store at `apps/web/src/stores/run.ts` (`activeRunIdAtom`, `runStateAtom`, `tokenCountAtom`, `elapsedMsAtom`) — matches Session 03's idiom (`stores/canvas.ts`). The SSE consumer hook at `apps/web/src/hooks/useRunSse.ts` opens the stream on first paint after `Run` click, derives state, writes atoms.
- **Spren orb reactive state on canvas during a run**: extends `apps/web/src/components/TopBar/PresenceOrb.tsx` with optional `state?: SprenState` prop (defaults to `"idle"`; chat-sheet-on-click behavior unchanged). The canvas wraps `PresenceOrb` with the jotai-derived run state: `state="thinking"` between agent turns, `state="speaking"` during a turn's `TextMessage*` triple (per-turn, not per-token — framework ships per-turn streaming; see §15), returns to `state="idle"` on `RunFinished`.
- **Minimal `/runs` list page**: TanStack Router route at `/runs/index.tsx`. Reuses Session 03's workflow-card design language (card per run with status badge + workflow name + relative timestamp + cost). Filterable by workflow + status. Links to `/runs/{id}` (Session 05 ships the full inspector; Session 04 ships a thin placeholder at `/runs/$runId.tsx` that shows the `runs` row + a "Trace viewer coming in Session 05" empty state).
- **Status badge**: NEW component at `apps/web/src/components/StatusBadge/`. Near-clone of `apps/web/src/components/ProvenanceBadge/` (own CSS, own component file — two consumers don't justify a generic `<Chip>` abstraction). Six status values (`queued` / `running` / `cancelling` / `succeeded` / `failed` / `cancelled`). Same tag-markup typographic device as `ProvenanceBadge`. The `running` and `cancelling` variants display a pulsing leading dot via a shared `<PulseDot>` primitive that the Run button also reuses for its `running` state — single component, three consumers, one motion idiom (per §9). Colors per §10.
- **Completion toast**: NEW component at `apps/web/src/components/CompletionToast/`. Inline pattern (no toast library); on `RunFinished` (or `RunError` Custom event), slides in from bottom-right of the canvas showing `Completed in <duration> · $<cost>` (success) / `Failed: <short reason>` (failure) / `Cancelled after <duration> · $<cost>` (cancel). Auto-dismisses in 6s. Click → navigates to `/runs/{id}`. `aria-live="polite"`, manual dismiss button, respects `prefers-reduced-motion` (no slide-in animation; immediate appear). See §10 polish item 6.
- **Cancel confirmation**: clicking Cancel during a `running` state opens a small inline confirmation (not a modal) saying `Cancel run? Tool calls in flight will finish.` with `Cancel run` / `Keep running` buttons. Confirms → `POST /v1/runs/{id}/cancel` and the button transitions to `cancelling` state with the 5-second visible countdown. Copy is honest about the cooperative-then-forced semantics: Framework 07's `Orchestra.cancel_session(force_after=5.0)` runs `quiesce()` for up to 5s (during which in-flight tool calls finish naturally), then `task.cancel()` if not drained — matching the user-visible 5→0 countdown pace.

Tests:

- Pytest unit: cost calculation (`packages/spren/tests/test_cost.py` — every provider in `cost_rates.py` × every metric, edge case for missing rates emits zero + WARN); materializer happy-path (`WorkflowDefinition` → `Topology + AgentRegistry`); `RunsBroker` subscribe/publish/drop-oldest; lifecycle state machine; `02__create_runs.py` migration applies + rolls back cleanly.
- Pytest integration (`packages/spren/tests/integration/test_runs_lifecycle.py`, `test_runs_sse.py`): `POST /v1/runs` end-to-end → row inserted, workflow snapshot frozen, Orchestra constructed + wired before 201 returns (using a static-response model adapter — NOT a mock; SP-007), `started` event set, lifecycle task scheduled; SSE end-to-end against a static-response Orchestra → subscribe, observe AG-UI event sequence ending with `RunFinished`, verify replay buffer holds events; cancel mid-run via `Orchestra.cancel_session()` → row transitions `running → cancelling → cancelled`, watchdog timeout path covered; cost aggregation against a known-token-count fixture; `attachments` length > 0 → 400; archived workflow → 400; `cancelling` or terminal status → 409.
- Vitest unit (`apps/web/tests/unit/`): SSE consumer hook reducer logic (event sequence → atom state); `Last-Event-ID` resume against a fixture stream; cost calculator (cross-check with the Pytest unit); jotai store transitions.
- Playwright E2E (`apps/web/tests/e2e/run-execution.spec.ts`): user clicks Run on the canvas, observes orb `thinking → speaking → idle`, elapsed-time counter increments, token counter increments per generation, completion toast with cost. Cancel flow: click Cancel, confirm, watch the 5-second countdown, run ends `cancelled`. Reconnect: kill the SSE connection mid-run, reconnect, observe replay buffer picks up via Last-Event-ID. Failed run: a deliberately-broken topology produces a `RunError`; toast renders `Failed: <reason>`. Tauri-driver E2E: same flow in the Tauri webview (WSL2 inconclusive caveat per Session 01).
- Visual regression baselines via Playwright `toHaveScreenshot()` (NOT Argos — Bundle 01 AC-166 explicitly bans Argos; Playwright snapshots are the standard): canvas with Run button in each of `idle / submitting / running / cancel-confirm-open / cancelling`, completion toast (success / failure / cancel variants), `/runs` list with three rows (one of each terminal status), `/runs/{id}` placeholder page (succeeded + failed variants).
- Manual-verify checklist (implementer self-verification before claiming done), including the orb state transitions matching the spec in §10, the cancel inline-confirmation flow, and the replay buffer reconnect-with-Last-Event-ID path against a real SSE stream.

Tests that depend on Framework 06 + 07 merging are explicitly tagged in their docstrings — they ship with `pytest.mark.skipif` gates that read a runtime check (`importlib.util.find_spec("marsys.coordination.aggui") is not None` for 06; `hasattr(Orchestra, "cancel_session")` for 07) so the suite stays green during the parallel period and lights up automatically once the framework dependencies merge.

## 4. What is OUT of scope

| Out of scope in Session 04 | Lands in |
|---|---|
| File upload + attachment endpoints (`POST /v1/files`, file references in `task_input.attachments`) | Session 05 |
| Trace viewer UI (nested span tree, per-span cost chips, expand/collapse) | Session 05 |
| Run history filtering UI (date pickers, status multi-select beyond the basic chips) | Session 05 |
| `GET /v1/runs/{id}/trace` REST endpoint returning a hierarchical TraceTree from `trace.ndjson` | Session 05 |
| `GET /v1/runs/{id}/artifacts` + artifact download | Session 05 |
| `GET /v1/runs/{id}/workflow` (frozen snapshot read endpoint) | Session 05. Session 04 writes the snapshot; Session 05 ships the read endpoint + the inspector UI for it. |
| Mid-run user-interaction (`POST /v1/runs/{id}/respond` for paused user_interaction prompts) | Session 05 or 06 (TBD when the meta-agent's first user-interaction surface lands) |
| Pause + resume (framework primitives + Spren-side `POST /v1/runs/{id}/pause` / `resume`) | v0.4 (`v0.4-29`) |
| Cost charts (daily / weekly / per-workflow rollup visualizations) | Session 06 or v0.4 (budget UI deps on Session 08) |
| Budget cap enforcement (per-day + per-run cost caps; meta-agent refuses over-budget) | Session 08 (meta-agent capabilities owns the enforcement; Session 04 produces the cost numbers the cap reads) |
| Scheduled triggers (`trigger=scheduled`) + webhook triggers + messenger triggers | v0.4 (Phase W) |
| Live cost display on the canvas during a run | Polish item — see §11. Session 04 ships only final cost on completion toast. |
| Multi-tab same-run SSE coordination (two tabs watching the same run) | v0.4 if requested. Session 04 supports it implicitly (each tab opens its own SSE; the framework's iterator is per-subscriber) but does not optimize for it. |
| `SprenTelemetrySink` (the `python my_workflow.py → Spren UI` adapter) | v0.4 (`v0.4-27`) |

Anything labeled out-of-scope renders as "not available" or empty placeholder in Session 04's routes (capability-gated per SP-019 where the route exists).

## 5. Files to CREATE / DELETE in Session 04

### To CREATE

**Spren backend:**

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/routes/runs.py` | REST + SSE endpoint handlers (POST `/v1/runs`, GET `/v1/runs`, GET `/v1/runs/{id}`, POST `/v1/runs/{id}/cancel`, GET `/v1/runs/{id}/events`, GET `/v1/runs/events`). Router-level auth via `APIRouter(prefix="/v1/runs", dependencies=[Depends(require_auth)])` per Session 01's pattern. Uses `sse-starlette`'s `EventSourceResponse` for SSE. |
| `packages/spren/src/spren/runs/__init__.py` | Package marker. |
| `packages/spren/src/spren/runs/lifecycle.py` | Async run lifecycle coordinator. Module-level `_active_runs: dict[str, ActiveRun]` registry. `ActiveRun` dataclass: `{orchestra: Orchestra, task: asyncio.Task, started: asyncio.Event, replay: collections.deque}`. Functions: `register(run_id, active_run)`, `get(run_id) -> ActiveRun \| None`, `deregister(run_id)`, `run_lifecycle(run_id, ...)` (the task body — awaits `Orchestra.run()`, persists terminals, deregisters). Mirrors `workers/draft_sweeper.py`'s module-pure idiom. |
| `packages/spren/src/spren/runs/materialize.py` | Spec → runtime materializer. `materialize_run(definition: WorkflowDefinition, secrets: SecretsStore) -> RuntimeBundle`. Builds `marsys.coordination.topology.core.Topology` from `TopologySpec`, instantiates `BaseAgent` subclasses with API keys resolved from secrets, resolves tool-name strings to callables via `marsys.environment.tools.AVAILABLE_TOOLS`, assembles `ExecutionConfig(aggui=AGGUIConfig(enabled=True), tracing=TracingConfig(...), storage_backend=FileStorageBackend(<data-dir>/data/runs))`. Load-bearing — was invisible in the original plan. |
| `packages/spren/src/spren/runs/sse.py` | Per-run SSE wrapper. Wraps `AGUIEventStream(orchestra.aggui_translator)`. On connect: parse `Last-Event-ID` header, binary-search the `ActiveRun.replay` deque, replay matching tail, then attach to live iterator. Uses `aggui_event_to_sse(event)` helper from `marsys.coordination.aggui`. |
| `packages/spren/src/spren/runs/broker.py` | `RunsBroker` — hand-rolled (~40 LOC) async pub/sub for the aggregate `/v1/runs/events` stream. `async def subscribe() -> AsyncIterator[Event]` (bounded `asyncio.Queue(maxsize=256)`, drop-oldest with `STREAM_LAGGED` marker on next put), `def publish(event)` (non-blocking). Singleton constructed in `server.create_app()`. Does NOT reuse the framework's `EventBus` (would mix Spren-emitted events with framework events; SP-018). |
| `packages/spren/src/spren/cost.py` | Cost calculation. `calculate_cost(provider, model, prompt_tokens, completion_tokens, reasoning_tokens) -> Decimal` (lookup in `cost_rates.RATES`, missing → log WARN + return zero). `apply_to_run(run_id, cost, tokens_in, tokens_out)` updates the `runs` row aggregates. Consumes `Custom("marsys.generation.metadata")` events from the AG-UI stream — NOT raw `GenerationEvent`. Logs WARN at daemon start if `cost_rates.LAST_UPDATED` is older than 90 days. |
| `packages/spren/src/spren/cost_rates.py` | Python module rate table (NOT YAML). `RATES: dict[tuple[str, str], CostRate]` keyed by `(provider, model)`; `LAST_UPDATED: date`; `CostRate` is a frozen dataclass with `input_per_1m_usd`, `output_per_1m_usd`, `reasoning_per_1m_usd: Decimal \| None`. Initial population: anthropic + openai + openrouter + google + xai. |
| `packages/spren/src/spren/models/run.py` | Pydantic models: `RunCreate`, `RunRead`, `RunListItem`, `RunStatus` enum (six values: `queued / running / cancelling / succeeded / failed / cancelled`), `TaskInput`, aggregate SSE event union (`RunCreatedEvent / RunUpdatedEvent / RunFinishedEvent / RunCancelledEvent` discriminated by `type`). Every cross-boundary model carries `schema_version: int = 1`. |
| `packages/spren/src/spren/storage/runs.py` | DAL for the `runs` table. `insert_run(...)`, `fetch_run(id)`, `list_runs(cursor, limit, workflow_id, status, since)`, `update_run_status(id, status, **terminals)`, `apply_cost_delta(id, cost, tokens_in, tokens_out)`. Mirrors `storage/workflows.py`'s shape. |
| `packages/spren/src/spren/storage/migrations/02__create_runs.py` | Forward-only migration. Schema per `docs/architecture/spren/02-data-model.md:86-105` with the `RunStatus` extended to six values. Indexes on `(workflow_id, created_at)`, `(status)`, `(created_at)`. |
| `packages/spren/tests/test_cost.py` | Pytest unit (every provider × every metric, missing-rate edge case). |
| `packages/spren/tests/test_runs_materialize.py` | Pytest unit (materializer happy-path, secret resolution, tool resolution). |
| `packages/spren/tests/test_runs_broker.py` | Pytest unit (subscribe/publish/drop-oldest semantics). |
| `packages/spren/tests/test_runs_lifecycle.py` | Pytest unit (state machine transitions). |
| `packages/spren/tests/test_storage_runs.py` | Pytest unit (DAL + cursor pagination). |
| `packages/spren/tests/integration/test_runs_lifecycle_e2e.py` | Pytest integration (POST /v1/runs end-to-end with static-response model adapter). |
| `packages/spren/tests/integration/test_runs_sse.py` | Pytest integration (SSE end-to-end + replay buffer + Last-Event-ID resume). |

**Spren web:**

| Path | Purpose |
|---|---|
| `apps/web/src/routes/runs/index.tsx` | `/runs` list page. Mounts `useRunsListSse()` on first paint. |
| `apps/web/src/routes/runs/$runId.tsx` | `/runs/{id}` thin placeholder. |
| `apps/web/src/lib/run-sse.ts` | Per-run SSE consumer (uses `EventSource` with `Last-Event-ID`). Parses AG-UI events via `@ag-ui/core` types. |
| `apps/web/src/lib/runs-list-sse.ts` | Aggregate `/v1/runs/events` SSE hook. Returns a row-delta stream. |
| `apps/web/src/stores/run.ts` | Jotai atoms: `activeRunIdAtom`, `runStateAtom` (orb state), `tokenCountAtom`, `elapsedMsAtom`, `runStatusAtom`. |
| `apps/web/src/hooks/useRunSse.ts` | Opens the per-run SSE on `Run` click, derives state from events, writes the jotai atoms. |
| `apps/web/src/components/RunButton/RunButton.tsx` + `.css` | Run button (idle / submitting / running / cancelling / terminal states). Reads from `stores/run.ts`. |
| `apps/web/src/components/CompletionToast/CompletionToast.tsx` + `.css` | Completion toast (success / failure / cancel variants). `aria-live="polite"`, manual dismiss button, respects `prefers-reduced-motion`. |
| `apps/web/src/components/StatusBadge/StatusBadge.tsx` + `.css` | Run status chip (six status values). Near-clone of `ProvenanceBadge`. |
| `apps/web/src/components/PulseDot/PulseDot.tsx` + `.css` | Shared pulsing-dot primitive used by `RunButton` (`running`), `StatusBadge` (`running` / `cancelling`), and the canvas presence orb cadence. |
| `apps/web/tests/e2e/run-execution.spec.ts` | Playwright golden path + cancel + reconnect + failed-run. |
| `apps/web/tests/unit/run-sse.test.ts` | Vitest (SSE consumer reducer logic + `Last-Event-ID` resume against fixture stream). |

**Framework worktree (stub authored as part of this Spren session; full brief authored by the framework architect):**

| Path | Purpose |
|---|---|
| `docs/implementation/framework/sessions/v0.3.0/07-cancel-session-api.md` | Stub for Framework Session 07. Multi-consumer justification, scope (`Orchestra.cancel_session(session_id, force_after=5.0) -> None` + tests), open questions for the framework architect (force-after default, watchdog log levels, behavior when session_id unknown, whether `discard_paused_session` semantics merge). |

### To MODIFY

| Path | Change |
|---|---|
| `apps/web/src/components/TopBar/PresenceOrb.tsx` | Add optional `state?: SprenState` prop (default `"idle"`). Backwards-compatible; `onClick` chat-sheet behavior unchanged. |
| `apps/web/src/routes/workflows/$workflowId.tsx` (lines 467-499) | Add `<RunButton />` to the canvas toolbar after the `Save` button. |
| `apps/web/src/routes/workflows/$workflowId.tsx` (presence orb usage) | Pass `state={runStateAtom value}` from the jotai store to wire orb-during-run. |
| `apps/web/package.json` | Add `@ag-ui/core` and `@ag-ui/client` (pin exact pre-1.0 version at implementation time). |
| `packages/spren/pyproject.toml` | Add `sse-starlette>=2.1` runtime dep. Add `[project.optional-dependencies] aggui = ["marsys[aggui]"]` so Spren can pull the framework's optional AG-UI extras. |
| `packages/spren/src/spren/server.py` | Mount `make_runs_router(...)` in `create_app()`; construct the `RunsBroker` singleton; pass `broker` + `secrets_store` into the router. Lifespan handler: cancel any in-flight `_active_runs` tasks on shutdown. |
| `docs/architecture/spren/03-api-design.md` | Update §Cancellation to reflect the new `cancelling` interim status. Update §SSE event format to reflect the locked Framework 06 namespace (`marsys.coordination.aggui`) and the `AGUIEventStream(translator)` signature. |
| `docs/architecture/spren/02-data-model.md` | Update `runs.status` enum from five values to six (add `cancelling`). Update line 188 (currently mentions `marsys.transport.aggui`) → `marsys.coordination.aggui`. |
| `docs/implementation/framework/v0.3-spren-support.md` | Update Session 06 row from "scoped" to "in flight". Add Session 07 row. |
| `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/testing/test-scenarios.md` | Add Session 04's golden-path / edge-case / exploratory scenarios (G-08 through G-11, U-06, X-07, C-01, C-02 per §11). |

### To DELETE

None. Session 04 is purely additive (new routes, new tables, new UI surfaces). No Session 03 stub action exists for the Run button — Session 03's canvas toolbar shipped Pattern + Save only.

## 6. User journeys (anchor for Bundle B demo gate)

Bundle B's demo gate extends Bundle A's J-1 / J-2 / J-3 with the Run experience. Two new journeys for Session 04.

### J-1 — First run (user builds + runs a workflow)

State: workflow exists from Bundle A's J-1 (research-pipeline, 3 agents, `provenance=visual_builder`). User is on `/workflows/$id` canvas.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User clicks `Run` button on the canvas top toolbar. | Canvas | Button transitions to `submitting` state (subtle 200ms spinner). `POST /v1/runs` fires with `{workflow_id, task_input: {text: "", attachments: []}}`. (Task-input UX deferred — Session 04 ships an empty default; richer task input lands in Session 06 with meta-agent integration.) |
| 2 | Server creates the run: snapshots `definition` to `workflow.json`, inserts `runs` row with `status=queued`, spawns the run async, returns `{run_id, status: "queued"}` in ~50ms. | (server-side) | — |
| 3 | Client receives the response, opens SSE at `GET /v1/runs/{run_id}/events`. Button transitions to `running` state showing `Cancel · 0s · 0 tokens`. | Canvas | Spren presence orb (top-right) shifts to `thinking`. |
| 4 | Framework starts the orchestra, fires `RUN_STARTED` AG-UI event. Spren receives via SSE; updates `runs.status=running`, `runs.started_at`. | (server-side + canvas) | Orb stays `thinking`. |
| 5 | Framework's first agent (Researcher) generates a turn. AG-UI emits `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` / `TEXT_MESSAGE_END` (per agent turn — full content in a single delta; framework lacks token streaming, see §15). | Canvas | Orb shifts to `speaking` while the turn is in flight. Elapsed timer ticks. |
| 6 | Researcher calls `search_web` tool. AG-UI emits `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` / `TOOL_CALL_RESULT`. After the generation completes, AG-UI emits `Custom("marsys.generation.metadata")` carrying `{model, provider, prompt_tokens, completion_tokens, reasoning_tokens, finish_reason}`. Cost accumulates and the token counter on the Run button increments. | Canvas | Orb returns to `thinking` briefly between agent turns. |
| 7 | Writer agent's turn. Same flow. | Canvas | Continues. |
| 8 | Framework emits `RUN_FINISHED`. Spren updates `runs.status=succeeded`, `runs.finished_at`, `runs.final_response`, `runs.total_*` fields. | (server-side + canvas) | Orb returns to `idle`. Run button transitions to `terminal` state (briefly) then back to `idle`. Completion toast slides in: `Completed in 12.3s · $0.026`. |
| 9 | User clicks the toast. | Navigates to `/runs/{run_id}` | Thin placeholder page renders: status badge (`succeeded`), workflow name link back to canvas, total duration, total cost, total tokens. Below: `Trace viewer ships in Session 05` empty-state copy in the tag-markup typographic device (`<trace-viewer status="coming_in_session_05" />`). |
| 10 | User navigates to `/runs` via cmdk (`⌘K → "runs"`). | `/runs` list page | Renders cards: most recent run on top with status badge + workflow name + cost + duration. Filter chips: `All / Running / Cancelling / Succeeded / Failed / Cancelled`. |

### J-2 — Cancel mid-run

State: user is on the canvas with a workflow that takes ~60s to run (e.g., research-pipeline with browser-based tools).

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User clicks `Run`. | Canvas | Same as J-1 step 1. |
| 2 | Run starts; Researcher is mid-search after ~20s. | Canvas | Orb `speaking`. Cancel button visible on the Run button. |
| 3 | User clicks `Cancel`. | Inline confirmation appears below the Cancel button | Copy: `Cancel run? Tool calls in flight will finish.` + buttons `[Keep running] [Cancel run]`. |
| 4 | User clicks `Cancel run`. | `POST /v1/runs/{run_id}/cancel` | Server transitions row to `cancelling`, calls `Orchestra.cancel_session(run_id, force_after=5.0)` (Framework 07). Run button shows `Cancelling… 5` countdown (5 → 0). Status badge in any open `/runs` view shows `cancelling`. |
| 5 | Framework drains in-flight tool calls within 5s (`quiesce()` phase). If drain completes, status moves to `cancelled` immediately and the countdown shortcuts. Otherwise, framework runs `task.cancel()` at t=5s and waits up to 2s more for terminal. | (server-side + canvas) | Once framework returns terminal: orb returns to `idle`, Run button returns to `idle`, completion toast: `Cancelled after 22s · $0.008`. If watchdog (10s) fires first: row marked `cancelled`, toast notes "cleanup may continue in background"; per-run budget cap (SP-013) is the eventual backstop on a leaked task. |

### J-3 — Reconnect after network blip

State: user is watching a run, browser tab momentarily loses focus + network hiccups close the SSE connection.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | Run is in progress, SSE streaming. | Canvas | Tokens counting, orb `speaking`. |
| 2 | Network drops; SSE connection closes. Client detects `EventSource.readyState === CLOSED`. | (client-side) | Run button shows a thin `Reconnecting…` annotation in `--ink-faint` next to the elapsed timer. |
| 3 | Client retries with exponential backoff (1s, 2s, 4s, max 8s). On reconnect, sends `Last-Event-ID` header with the last AG-UI event ID it received. | (client-side) | — |
| 4 | Server binary-searches the in-memory replay buffer (`ActiveRun.replay`, `deque(maxlen=1024)`) by ULID `event_id`, replays the matching tail, then switches to live consumption. If `Last-Event-ID` is older than the deque's oldest entry (>1024 events ago), the server emits a single `Custom("marsys.stream.gap")` and the client refreshes its row state via REST and continues from now. | (server-side) | Token counter catches up to the live value within ~500ms. Orb state catches up. `Reconnecting…` annotation disappears. |

## 7. Skeleton wireframes (low-fi; ASCII)

Sessions 04 surfaces are small extensions of Session 03's design language. Refer to [`../../01-visual-builder/sessions/03-visual-builder.md`](../../01-visual-builder/sessions/03-visual-builder.md) §7 for the established surface chrome (top-bar with `spren.` wordmark + user avatar; presence orb top-right of non-home surfaces; no left nav rail; ⌘K navigation).

### W-A — Run button states (canvas top toolbar)

```
idle:                              submitting:
┌──────────────────────────┐       ┌──────────────────────────┐
│ [Lint ✓] [+ Pattern▾]    │       │ [Lint ✓] [+ Pattern▾]    │
│             [Run] [Save] │       │             [⟳]  [Save]  │
└──────────────────────────┘       └──────────────────────────┘

running:                           running with cancel-confirm:               cancelling (countdown 3):
┌─────────────────────────────────┐         ┌─────────────────────────────────┐
│ [Lint ✓] [+ Pattern▾]           │         │ [Lint ✓] [+ Pattern▾]           │
│  [Cancel · 12s · 487t]   [Save] │         │   [Cancelling… 3]        [Save] │
│  ┌──────────────────────────┐   │         │                                 │
│  │ Cancel run? Tool calls   │   │         │ (server called Orchestra.       │
│  │ in flight will finish.   │   │         │  cancel_session(force_after=5); │
│  │ [Keep running][Cancel run]│  │         │  countdown 5→0 mirrors the      │
│  └──────────────────────────┘   │         │  framework's quiesce window;    │
└─────────────────────────────────┘         │  if drain completes early, UI   │
                                            │  shortcuts to "Cancelled")      │
                                            └─────────────────────────────────┘
```

Run button uses the same shape language as Session 03's send button: `--ink` background default, `--magenta` on hover, 44px height, 16px border-radius, Geist 500/14px label. Cancel state uses `--magenta` background (signaling stop/active). Token counter + elapsed timer in Geist Mono 11px.

### W-B — Completion toast (bottom-right of canvas)

```
                                        ┌────────────────────────────────┐
                                        │ ◉ Completed in 12.3s · $0.026  │
                                        │   research-pipeline · view →   │
                                        └────────────────────────────────┘
```

Toast: white surface, 1px `--rule` border, 14px border-radius, 16px×20px padding. Status dot in `--magenta` (succeeded) / `--magenta-deep` (failed; see status palette below) / `--ink-soft` (cancelled). Workflow name + "view" link in `--ink-soft`. Auto-dismisses in 6s; click anywhere on toast → navigates to `/runs/{id}`. Failed state replaces the timing line with `Failed: <short reason>` (full error on `/runs/{id}`).

### W-C — `/runs` list page

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Runs                                                (R) │
├─────────────────────────────────────────────────────────────────────┤
│   Runs                                                  47 runs     │
│                                                                     │
│   [All] [Running] [Succeeded] [Failed] [Cancelled]                  │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ research-pipeline · [succeeded]                          │      │
│   │ 12.3s · $0.026 · 2 minutes ago                           │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ daily-summary · [● running]                              │      │
│   │ elapsed 38s · $0.011 so far · started just now           │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ pr-review-assist · [failed]                              │      │
│   │ 4.2s · $0.003 · yesterday · "Researcher returned…"       │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                ◉    │
└─────────────────────────────────────────────────────────────────────┘
```

Reuses Session 03's workflow-card design language. Status badge same shape as provenance badge (Geist Mono 10px, padded chip). Running rows live-update via SSE on the list endpoint (server-pushed UPDATE events) — see §15 question 4.

### W-D — `/runs/{id}` thin placeholder

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Runs  ›  research-pipeline · 2 minutes ago          (R) │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   [succeeded]   research-pipeline                                   │
│                                                                     │
│   Duration:    12.3s                                                │
│   Cost:        $0.026                                               │
│   Tokens:      8,432 in · 1,221 out                                 │
│   Started:     14:32:08 (2 minutes ago)                             │
│   Finished:    14:32:20                                             │
│                                                                     │
│   ─────────────────────────────────────                             │
│                                                                     │
│   <trace-viewer status="coming_in_session_05" />                    │
│                                                                ◉    │
│   The trace viewer lands in Session 05 alongside file uploads       │
│   and the run-inspection bundle.                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Tag-markup typographic device for the empty-state, consistent with Session 03's design system. Once Session 05 ships, the empty-state is replaced by the trace viewer; the metadata section above stays.

## 8. Decisions locked

These were the questions I considered before writing this draft. Each is resolved here unless §15 surfaces it as needing user input.

1. **`/runs` route in Session 04 or 05?** Session 04 ships a minimal `/runs` list page + a thin `/runs/{id}` placeholder. Session 05 fills in the trace viewer + file upload + filtering UI. Reasoning: Bundle B's demo gate ("user runs a workflow and sees it land") needs at least the list view; without it, the user is back on the canvas with no path to see history. The thin placeholder for `/runs/{id}` lets the completion toast's "view →" link land somewhere meaningful even before Session 05.
2. **Spren orb reacts to run state on canvas.** Session 03 shipped the presence orb (`apps/web/src/components/TopBar/PresenceOrb.tsx`) hard-coded to `state="idle"`. Session 04 extends `PresenceOrb` with optional `state?: SprenState` prop (default `"idle"`; chat-sheet-on-click behavior unchanged — backwards-compatible). The canvas wraps `PresenceOrb` with the jotai-derived run state from `apps/web/src/stores/run.ts`: `state="thinking"` between agent turns, `state="speaking"` during a turn's `TextMessage*` triple, returns to `state="idle"` on `RUN_FINISHED`.
3. **Live cost display on canvas.** Session 04 does NOT show live-accumulating cost during a run on the canvas. The token counter ticks; cost is final-only in the completion toast. Live cost is distracting in focus work + the cost-per-token rate table updates lag the model's pricing changes by days. Decision: ship final cost; add live cost in v0.4 if users ask.
4. **Workflow snapshot at run start (SP-009).** The full `definition` is frozen as `<data-dir>/data/runs/{run_id}/workflow.json` at the moment `POST /v1/runs` fires. The run executes against the snapshot; editing the workflow during a run does NOT affect the run. The `runs.workflow_id` foreign key references the live workflow row (for "what workflow was this a run of?") but the snapshot is the source of truth for what actually ran.
5. **Attachments in `task_input`.** Field accepted by the schema (Session 02 already includes `attachments: ["file_id_1", …]` in the data model), but a non-empty array returns `400 ATTACHMENTS_NOT_YET_SUPPORTED` in Session 04. Session 05 implements upload + reference resolution.
6. **Default task input on canvas Run.** Canvas's Run button submits `task_input: {text: "", attachments: []}` by default in Session 04 (no task-input modal yet). Workflows that depend on a task message will see an empty user message — that's an acceptable v0.3 limitation. Richer task input (a small dialog asking "What should the workflow work on?") lands in Session 06 when meta-agent integration arrives (the meta-agent owns the task-input experience).
7. **Cancel button placement + UX + cleanup timing.** Inline confirmation below the Cancel button (not a modal — modals are first-thought-laziness per design laws). Initial confirmation copy: `Cancel run? Tool calls in flight will finish.` + `[Keep running] [Cancel run]`. On confirm, the server calls `Orchestra.cancel_session(run_id, force_after=5.0)` (Framework 07) which (a) runs `quiesce()` for up to 5s — graceful drain phase where in-flight tool calls finish naturally, then (b) calls `task.cancel()` if not drained — forced phase, up to 2s more for terminal. The UI mirrors this with a visible 5-second countdown (5 → 0) on the Run button: if `quiesce()` returns early, the countdown shortcuts to `Cancelled`; if the countdown completes, the framework moved to phase (b). UI displays `Cancelled` once the framework returns terminal; toast confirms. Server-side watchdog: 10s total budget on the framework call; if exceeded, log WARN, mark row `cancelled` regardless, emit `Custom("marsys.cancel.timeout")` on the run's SSE stream — leaked task is bounded by per-run budget cap (SP-013) plus framework's own timeouts (`convergence_timeout` / `branch_timeout` / `step_timeout`). Two-phase semantics are real (5s graceful + 2s forced) — backed by a real framework primitive (`Orchestra.cancel_session`), not an in-Spren overlay.
8. **Reconnect retry backoff.** Exponential: 1s → 2s → 4s → max 8s, infinite retries until the user navigates away or the run terminates. No upper retry count — users on flaky networks should never see "permanently disconnected" unless they intend to.
9. **Live updates on `/runs` list page.** Aggregate `GET /v1/runs/events` SSE stream — one connection per client subscribing to ALL run-state-changed events for the whole list. Client mounts on first paint, updates rows in place as events arrive. Filter chips apply client-side. Per-card SSE (each running card opening its own stream) is explicitly rejected — it caps the list's scale at ~3-5 simultaneous running cards. The aggregate stream pattern scales to hundreds of runs at negligible cost (server keeps one event bus + one subscriber queue per client).
10. **Status badge palette.** Six status values, each with a single token from Session 03's palette:
   - `queued` → `--ink-soft` chip background, `--ink` text
   - `running` → `--peach` chip background, `--ink` text, leading `●` pulse animation via `<PulseDot>`
   - `cancelling` → `--peach` chip background, `--ink` text, leading `●` pulse animation (same as `running` — the user perceives the run as still active during the 5-second drain), label is `cancelling`
   - `succeeded` → `--magenta` chip background, white text
   - `failed` → `--magenta-deep` chip background, white text
   - `cancelled` → `--rule` chip background, `--ink-soft` text
   These reuse the brand palette without adding new colors. The `--peach` for running + cancelling ties the canvas orb's `speaking` color to the list's active-state indicator.

11. **Bundle 02 structure.** Sessions 04 + 05 form one demo-able feature slice — run a workflow live then inspect what happened. Layout on disk: `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/{04,05}-*.md` with sibling `testing/test-scenarios.md` (user-facing scenario list) + `testing/test-session.md` (Claude Code testing-agent brief), matching Bundle 01's pattern at `docs/implementation/spren/v0.3.0/01-visual-builder/`. Session 04's contribution to `test-scenarios.md` lands when Session 04 ships; Session 05's contribution lands when Session 05 ships; the bundle-end test runs only after both sessions ship + are individually green. Bundle 02's `test-scenarios.md` is fleshed out alongside this brief (Session 04 covered now; Session 05 placeholders that get expanded when Session 05 ships) so Session 04's implementer has a target structure to write the per-session manual-verify checklist against.

12. **Cost source: AG-UI Custom event only, no trace.ndjson fallback.** Session 04's cost calculator consumes `Custom("marsys.generation.metadata")` events from the AG-UI stream — the framework's mapping for `GenerationEvent` per Framework 06's brief. The trace.ndjson `generation` spans are NOT a parallel input path in Session 04; they're for Session 05's trace viewer (cold reads of terminal runs). One source of truth, one tested path. If a daemon crash leaves a run terminal-but-uncosted, Session 05's inspector can recompute cost from `trace.ndjson` as a robustness feature; that path is out of scope here.

13. **AG-UI events persistence: in-memory replay buffer, no on-disk file.** The reconnect-with-`Last-Event-ID` path uses an in-memory `collections.deque(maxlen=1024)` per active run, held on the `ActiveRun` registration record. Survives a network blip; does NOT survive daemon restart (terminal runs lose live-stream replay; Session 05's trace viewer is the reconstruction surface). ~50 LOC vs an NDJSON writer + reader pair. If the user's `Last-Event-ID` is older than the deque's oldest entry (>1024 events ago), server emits `Custom("marsys.stream.gap")`, client refreshes via REST and continues from now. Per-run memory bound: ~200KB at full deque (well under v0.3's single-user-local budget at typical 3-5 concurrent runs).

14. **`cancelling` interim status as enum value, not derived.** The `RunStatus` enum extends from five to six values: `queued / running / cancelling / succeeded / failed / cancelled`. Wire-visible enum value (clients render directly). Migration `02__create_runs.py` stores the enum as TEXT — no schema change needed when the enum extends. The alternative of `cancelling_at` timestamp + UI-side derivation was rejected: enum extension is simpler for clients (TUI, future Python adapter), unambiguous on the wire, no per-client derivation logic.

15. **Spec → runtime materializer location.** Lives at `packages/spren/src/spren/runs/materialize.py` for v0.3 (run-scoped consumer is the only consumer). When v0.4 ships `SprenTelemetrySink` and the meta-agent (both will materialize workflows for separate purposes), the materializer relocates to `packages/spren/src/spren/runtime/materialize.py`. Don't pre-build the `runtime/` directory — wait for the second consumer.

## 9. Design system additions

Session 04 adds NO new design-system tokens, fonts, or layout primitives. It reuses Session 03's complete system. New components shipped (`RunButton`, `CompletionToast`, `StatusBadge`, `RunCard`) all instantiate the existing tokens.

One pattern Session 04 establishes (formalizes for later sessions): **live-state UI elements** (the running run button, the running row on `/runs`) share a common motion idiom — a leading `●` glyph that pulses at the same cadence as the Spren orb's `speaking` state (`cubic-bezier(0.45, 0, 0.55, 1)`, ~2s period, peaks at 1.0 opacity, dips to 0.4). This binds the visual identity of "something is happening" across the orb + the run UI. Implement once in a `<PulseDot color={...} />` component; reuse across the run button, status badges, and any future live indicators.

## 10. Polish items to address inside Session 04

These are gaps the architect-stage draft surfaced that the implementer addresses in-session, not as nice-to-haves.

1. **SSE reliability through Tauri webview.** The browser's `EventSource` works fine in browsers; the Tauri webview's `EventSource` had historic quirks (auto-reconnect behavior, header passthrough). Implementer benchmarks SSE through the Tauri shell on macOS / Windows / Linux against the same flow in browser. If any platform requires a fallback (e.g., long-poll), the fallback is wired here, not punted.
2. **Cost rate freshness.** `cost_rates.py` ships with `LAST_UPDATED: date` constant + WARN log at daemon start if older than 90 days. Models get repriced quarterly-ish. Defer auto-update (calling provider pricing APIs) to v0.4.
3. **Run button keyboard shortcut.** ⌘R or similar should trigger Run on the canvas. Implementer picks the shortcut + documents in the cmdk overlay.
4. **Completion toast accessibility.** Toast must use `aria-live="polite"` (not steal focus), have a close button (some users want manual dismiss), and respect `prefers-reduced-motion` (skip the slide-in animation).
5. **Reduced-motion fallback for the running orb.** When `prefers-reduced-motion` is on, the canvas presence orb's `thinking` + `speaking` states must NOT animate — they degrade to static color shifts (peach for speaking, slightly warmer for thinking). The Spren orb base spec (Session 03 §9.3) already covers this; Session 04 verifies the `RunButton`'s `PulseDot` also respects it.
6. **Empty `/runs` list state.** Tag-markup typographic device: `<runs status="empty" />` with body copy `No runs yet. Build a workflow and click Run.` + a button linking to `/workflows`.
7. **Race-free SSE handshake.** First `EventSource` open after `POST /v1/runs` returns may arrive before the lifecycle task has set `started: asyncio.Event`. The SSE endpoint does `get(run_id) or 404` then `await started.wait()` with a 2s timeout (returns `503` with retry-after if exceeded). Implementer verifies this handshake works against a fast-startup test.

**Polish items deferred to Session 05 (out of scope for Session 04):**
- `/runs` list pagination performance at 1000+ rows (Session 05's filter UI is the natural place to address).
- Reconnect annotation copy variants (`Reconnecting in 4s…` / `Server unreachable — retrying…`); Session 04 ships one generic string.
- `/runs/{id}` failed-run error formatting (full stack trace vs short message + expand) — Session 05's inspector territory.
- Cold-reader replay from `trace.ndjson` (replaced by in-memory replay buffer per §8.13; the trace-viewer cold path is Session 05).

## 11. Success criteria

Extending Bundle A's `test-scenarios.md` (and feeding into Bundle B's `test-scenarios.md` once that file is created):

- **G-08** (run execution golden path): user opens a saved workflow, clicks Run, watches the orb state transitions (idle → thinking → speaking → idle), sees the Run button transition through states, receives the completion toast with cost, clicks through to `/runs/{id}` and sees the row metadata.
- **G-09** (cancel mid-run): user clicks Run, then Cancel mid-run, confirms in the inline prompt, observes the 5-second countdown on the Run button (or earlier shortcut if `quiesce()` returns early), watches the run terminate with `cancelling → cancelled` status transitions.
- **G-10** (reconnect): user starts a long-running workflow, network blip drops the SSE connection, client retries with `Last-Event-ID` from the in-memory replay buffer; UI catches up to current state within 500ms of reconnect with no events lost (within the 1024-event window).
- **G-11** (run list): user navigates to `/runs`, sees a list of recent runs with correct statuses + costs + durations; filter chips work (`All / Running / Cancelling / Succeeded / Failed / Cancelled`); clicking a run navigates to `/runs/{id}`; aggregate SSE updates running rows live.
- **U-06** (manual smoke): from a fresh data dir, build a workflow → run it → see cost computed correctly → see run on `/runs` → cancel a running workflow → verify it terminates.
- **Visual regression baselines via Playwright `toHaveScreenshot()`** (Bundle B extension): Run button states (idle / submitting / running / cancel-confirm-open / cancelling), completion toast (success / failure / cancel variants), `/runs` list (with rows of each status), `/runs/{id}` placeholder (succeeded + failed variants).
- **X-07** (offline behavior): browser loses network mid-run; UI reflects reconnecting state; on reconnect, full state catches up; no events lost within the replay buffer window. If `Last-Event-ID` is older than the buffer (1024 events ago), client receives `Custom("marsys.stream.gap")` and refreshes via REST.
- **C-01** (cost calculation accuracy): a 3-agent run with known token counts (using a static-response model adapter — NOT a mock; SP-007) produces a cost within 0.01¢ of hand-calculated.
- **C-02** (cost calculation unknown model): a workflow using a model NOT in the rate table completes successfully with `total_cost_usd=0.0` and a WARN-level log entry naming the missing `(provider, model)`.

## 12. Open research items the implementer resolves in-flight

Empirical version + behavior verification done during implementation, not via a separate pre-implementation research stage.

- AG-UI Python package version pin (`ag-ui-protocol==0.1.18` per Framework 06's pin; verify still current at implementation time via `pip index versions ag-ui-protocol`). Ships via `marsys[aggui]` optional extras.
- AG-UI npm package pins for the JS side: `@ag-ui/core` and `@ag-ui/client` exact versions (AG-UI is pre-1.0; pin tightly).
- SSE behavior in the Tauri 2 webview (per polish item 1) — empirical benchmark across macOS / Windows / Linux.
- `EventSource` `Last-Event-ID` header behavior across Chromium (browser) vs WebKit2GTK (Tauri Linux) vs WKWebView (Tauri macOS).
- `Custom("marsys.generation.metadata")` payload shape: verify against `marsys.coordination.aggui.mapping.map_generation_event` once Framework 06 lands (current locked contract: `{model, provider, prompt_tokens, completion_tokens, reasoning_tokens, finish_reason}`; if the framework adds fields, Spren's cost calculator gracefully ignores unknowns).
- Token-streaming nuance: framework currently ships per-turn (single `TextMessageContent` per assistant turn carrying full content). v0.4 may add token-level streaming via provider adapter changes; Session 04's UI must work with both — the orb shifts to `speaking` on `TextMessageStart` and back to `thinking` on `TextMessageEnd` regardless of how many `Content` events are between them.
- **Token-counter source: confirmed `Custom("marsys.generation.metadata")`** (not raw `GenerationEvent` from `EventBus` — that doesn't cross the AG-UI seam; not `TOOL_CALL_END` — tools don't carry token counts).

If any of these surface a conflict with the locked decisions in §8, the implementer flags it and asks before deviating.

## 13. Status

- [x] Tier confirmed (HIGH).
- [x] Scope boundaries confirmed (sections 3 + 4).
- [x] Files-to-CREATE + files-to-MODIFY list approved (section 5).
- [x] Three user journeys approved (section 6).
- [x] Skeleton wireframes approved (section 7).
- [x] Decisions locked (section 8).
- [x] Plan reconciled with locked Framework 06 contract: namespace `marsys.coordination.aggui`, iterator `AGUIEventStream(translator)`, gating via `ExecutionConfig.aggui = AGGUIConfig(enabled=True)`, cost source `Custom("marsys.generation.metadata")`, per-turn (not per-token) streaming. Architecture doc updates (02-data-model.md, 03-api-design.md, 06-observability.md) tracked as separate doc tasks.
- [x] Plan reconciled with cooperative-cancel reality: `cancelling` interim status, `Orchestra.cancel_session(force_after=5.0)` from Framework 07 (NEW), in-memory replay buffer (no `trace.ndjson` cold-replay), Playwright `toHaveScreenshot()` baselines (NOT Argos).
- [x] Polish items captured for in-session work (section 10); 4 items deferred to Session 05.
- [x] Success criteria affirmed (section 11).
- [ ] Framework Session 07 (`Orchestra.cancel_session`) stub authored at [`../../../../framework/sessions/v0.3.0/07-cancel-session-api.md`](../../../../framework/sessions/v0.3.0/07-cancel-session-api.md) (drafted in this session). Full brief authored by the framework architect; implementation in flight in `marsys-tracing-work` worktree.
- [ ] Acceptance criteria frozen at [`./04-run-execution/acceptance.md`](./04-run-execution/acceptance.md) — extracted by the `acceptance-criteria-extractor` agent before any code is written.
- [ ] Framework Session 06 (AG-UI translator) merged. Spren Session 04 builds the ~85% of scope that doesn't directly consume the AG-UI iterator now; the AG-UI consumer files (`runs/sse.py`, `lib/run-sse.ts`, `hooks/useRunSse.ts`) are gated behind `pytest.importorskip` / runtime-check skips and light up automatically once Framework 06 lands. Brief in [`../../../../framework/sessions/v0.3.0/06-aggui-translator.md`](../../../../framework/sessions/v0.3.0/06-aggui-translator.md).
- [ ] Framework Session 07 (`cancel_session`) merged. Spren Session 04's cancel endpoint is gated behind a `hasattr(Orchestra, "cancel_session")` runtime check; falls back to a no-op `200` with `cancelled` status until 07 lands.
- [ ] Session implementation complete (all acceptance criteria pass; polish items addressed; tests green; manual verify done).

**Next step:** apply the Framework Session 07 stub. Then `acceptance-criteria-extractor` freezes `./04-run-execution/acceptance.md`. Implementation can begin against the locked Framework 06 + 07 contracts; ~85% of scope is implementable now, the remainder integrates as the framework dependencies merge. Polish items in §10 are explicit acceptance criteria — scoped into the session, not nice-to-haves.

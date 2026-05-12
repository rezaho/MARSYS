# Spren Session 04 — Run Execution + AG-UI Streaming + Cost

> Session plan. The implementer reads this as the primary source of truth for what Session 04 ships, how it integrates with framework Session 06 (AG-UI translator), and what's in vs out of scope. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / DELETE, the user journeys that extend Bundle A's Session 03 build flow into a runnable workflow, wireframes for the small Session 04 UI lift (Run button on canvas + minimal `/runs` route), data-model considerations, the locked decisions from the architect-stage Q&A, polish items the implementer addresses in-session, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./04-run-execution/acceptance.md`](./04-run-execution/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Design language anchor: [`../../01-visual-builder/sessions/03-visual-builder.md`](../../01-visual-builder/sessions/03-visual-builder.md) §9 Design System. Session 04 reuses Session 03's palette, typography, components, motion tokens, and Spren orb without redefining them. New surfaces added in Session 04 (Run button states, the minimal `/runs` route, the per-run progress indicator, completion toast) follow the same design language. Visual anchor for the Spren orb (whose reactive states change during a run): [`../../01-visual-builder/assets/spren-inspiration.png`](../../01-visual-builder/assets/spren-inspiration.png).

---

## 1. Bundle position + tier

- **Bundle**: B — Run Execution + Inspection (Sessions 04 + 05). Session 04 ships the run execution backend + the canvas Run button + a minimal `/runs` list view. Session 05 ships the full trace viewer UI, file upload + attachment endpoints, and run history filtering UI.
- **Bundle demo gate**: user builds a workflow in Bundle A's canvas, clicks Run, watches the workflow execute live with token-by-token streaming visible in the Spren orb's "speaking" state, sees cost computed and displayed on completion, clicks through to see the run in `/runs` with status + total cost + duration. Trace viewer (nested spans, per-span cost) lands in Session 05; Session 04's `/runs/{id}` view is a thin row-from-table page with a link to the (Session 05) full trace viewer.
- **Tier**: HIGH-tier with implementer-side research. New surface area is moderate: AG-UI consumption pattern (consuming the framework's `AGUIEventStream` async iterator over an SSE HTTP endpoint), SSE through Tauri webview reliability, cost-rate table format + provider lookup, `runs` table lifecycle + cancellation semantics, workflow-snapshot immutability at run start (SP-009). Smaller new tech surface than Session 03 (no new client frameworks); larger backend surface.
- **Approval gate**: implementer-side; surfaces to user only if a decision in §15 needs the user's input or a polish item in §11 reveals a hidden architectural ambiguity.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 01 (foundation) | shipped | FastAPI sidecar, auth, capabilities. |
| Spren Session 02 (CRUD + types) | in flight | `workflows` table + Pydantic models. Session 04 reuses the type-generation pipeline + `GET /v1/tools` endpoint. |
| Spren Session 03 (visual builder) | locked design; ready for implementation | Provides the canvas, the Spren orb component, the `<Run>` button shell (designed but wired to a stub action in 03), the design system tokens, and the conversation-state layout polish item. Session 04 implements the actual Run wiring. |
| Framework Session 01 (NDJSON streaming tracing writer) | shipped | Backs the `trace.ndjson` per-run file Spren reads for cold-start replay of SSE streams. |
| Framework Session 02 (`TelemetrySink` ABC + `SecretRedactor`) | shipped | Not consumed by Session 04 directly (Spren v0.4 consumes via `SprenTelemetrySink`). The redactor runs at the framework's fan-out boundary and applies to the AG-UI stream too (filtered before AG-UI translation). |
| **Framework Session 06 (AG-UI translator)** | **scoped — required before Spren Session 04 starts** | Ships `AGUIEventStream(orchestra, run_id) -> AsyncIterator[AGUIEvent]` in the framework. Spren Session 04 wraps this iterator in an SSE HTTP endpoint at `GET /v1/runs/{id}/events`. See [`../../../../framework/sessions/v0.3.0/06-aggui-translator.md`](../../../../framework/sessions/v0.3.0/06-aggui-translator.md). |
| `EventBus` + `Orchestra` (existing framework) | live | Spren constructs `Orchestra` per run; passes the workflow's frozen `definition` snapshot; subscribes via the framework's `AGUIEventStream` adapter. |
| Spren Session 02's `GET /v1/tools` | dep | Needed for cost-rate table cross-check (tool calls' cost is on the parent generation, not the tool — but the tool catalog is referenced when reading cost-by-tool in Session 05). Not blocking. |

Session 04 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). The only framework consumption is `Orchestra.run()` (with a frozen workflow definition) + the `AGUIEventStream` adapter from framework Session 06. The Spren-side draft sweeper from Session 03 stays.

## 3. What ships in Session 04

Backend surfaces (the bulk of the session):

- **REST run execution endpoint**: `POST /v1/runs` accepting `{workflow_id, task_input: {text, attachments: []}}`. Validates the workflow exists + is not archived; freezes the workflow's current `definition` as `<data-dir>/data/runs/{run_id}/workflow.json` (SP-009 immutability); inserts a `runs` row with `status=queued`; spawns the run asynchronously; returns `{run_id, status: "queued"}` immediately. Attachments handling deferred to Session 05 (Session 04 accepts the field but ignores values; a non-empty `attachments` array returns 400 with `code: "ATTACHMENTS_NOT_YET_SUPPORTED"`).
- **REST run row read**: `GET /v1/runs/{id}` returns the `runs` table row.
- **REST run list**: `GET /v1/runs` with cursor pagination + `?workflow_id=` + `?status=` + `?since=`. Includes drafts-hidden filter consistent with Session 03's draft-workflow pattern.
- **Aggregate run-events SSE stream**: `GET /v1/runs/events` — a server-pushed SSE stream emitting `RunCreated` / `RunUpdated` / `RunFinished` / `RunCancelled` events for ALL runs (not per-run; one stream for the whole list page). Lifecycle coordinator emits to an in-process pub/sub; the endpoint subscribes per-client and emits per-row deltas. Client `/runs` list mounts one stream on first paint and updates rows in place. Filter chips apply client-side (cheap; small row count). Replaces the simpler per-card SSE approach to avoid scale ceilings on the list view.
- **REST cancel endpoint**: `POST /v1/runs/{id}/cancel`. State transitions per [`../../../../../architecture/spren/03-api-design.md`](../../../../../architecture/spren/03-api-design.md) §Cancellation: `queued` → immediate `cancelled`; `running` → signal `Orchestra` to abort, await cleanup, then `cancelled`; `paused` → 409 (paused only ships in v0.4); terminal → 409.
- **SSE event stream**: `GET /v1/runs/{id}/events` opens an SSE connection that yields AG-UI events from the framework's `AGUIEventStream(orchestra, run_id)` async iterator. Reconnect-tolerant: cold reader replays from the NDJSON trace file at `<data-dir>/data/runs/{run_id}/trace.ndjson` (filtering out the non-span line types per [`../../../../../architecture/spren/06-observability.md`](../../../../../architecture/spren/06-observability.md) §Diagnostic and terminal lines), then switches to live consumption. `Last-Event-ID` resume supported via the framework's ULID `event_id` ordering.
- **Cost rate table + per-run aggregation**: `packages/spren/src/spren/cost.py`. Loads a YAML file (`packages/spren/src/spren/cost_rates.yaml`) mapping `(provider, model) → {input_per_1m_usd, output_per_1m_usd, reasoning_per_1m_usd}`. On every `GenerationEvent` (consumed from the AG-UI stream OR from the trace.ndjson), computes `cost = (prompt × in_rate + completion × out_rate + reasoning × reasoning_rate) / 1_000_000`. Aggregates into `runs.total_cost_usd` + `runs.total_tokens_input` + `runs.total_tokens_output` on every span close and on `RunFinished`. Missing rate (unknown model) → warn-and-emit-zero, not crash.
- **`runs` table lifecycle handler**: a small async coordinator in `packages/spren/src/spren/runs/lifecycle.py` that owns the `queued → running → succeeded|failed|cancelled` transitions, snapshots the workflow at run start, fires the framework `Orchestra.run()`, persists `total_duration_ms` / `total_steps` / `final_response` / `error` on completion.
- **`trigger` field**: Session 04 only emits `manual` for the canvas Run button. Other values (`scheduled`, `webhook`, `messenger:*`) are accepted by the schema for forward-compatibility but rejected at the API surface in v0.3 (`code: "TRIGGER_NOT_YET_SUPPORTED"`).

UI surfaces (small, additive on top of Session 03):

- **Canvas Run button live**: Session 03 shipped the Run button as a UI element wired to a stub action. Session 04 wires it to `POST /v1/runs` and opens the SSE stream. Button states: `idle` (default, shows `Run`) → `submitting` (200ms spinner) → `running` (shows `Cancel` instead, plus a token count + elapsed timer) → terminal (toast: "Completed in 12s · $0.03" / "Failed" / "Cancelled"; button returns to `idle`).
- **Spren orb reactive state on canvas during a run**: the canvas's small presence orb (Session 03 polish item 3) reflects the run state per Session 03's `<Spren state="…" />` API:
  - `state="thinking"` while waiting for the first token from any agent
  - `state="speaking"` during token streaming
  - returns to `state="idle"` on `RunFinished`
- **Minimal `/runs` list page**: TanStack Router route at `/runs`. Reuses Session 03's workflow-card design language (card per run with status badge + workflow name + relative timestamp + cost). Filterable by workflow + status. Links to `/runs/{id}` (Session 05 ships the full inspector; Session 04 ships a thin placeholder that shows the `runs` row + a "Trace viewer coming in Session 05" empty state).
- **Status badge palette extension**: extends Session 03's provenance-badge pattern to status badges (`queued` / `running` / `succeeded` / `failed` / `cancelled`). Colors per §10. Same Geist Mono font + padded chip shape as provenance badges.
- **Completion toast**: on `RunFinished`, a toast slides in from the bottom-right of the canvas showing `Completed in <duration> · $<cost>` (or `Failed: <short reason>`). Auto-dismisses in 6s. Click toast → navigates to `/runs/{id}`.
- **Cancel confirmation**: clicking Cancel during a `running` state opens a small inline confirmation (not a modal — inline below the Cancel button) saying "Cancel run? Tool calls in flight will finish." with `Cancel run` / `Keep running` buttons. Confirms → `POST /v1/runs/{id}/cancel`.

Tests:

- Vitest unit: cost calculation (every provider × every metric, edge cases for missing rates), AG-UI event parsing (against fixture events), SSE reconnect logic (Last-Event-ID resume from a partial trace.ndjson), runs lifecycle state machine.
- Pytest integration: `POST /v1/runs` → workflow snapshot frozen + row inserted + Orchestra spawned (mock the LLM call with a static-response provider); SSE end-to-end (subscribe, see events stream, see RunFinished); cancel mid-run; cost aggregation against a known token-count run; `attachments` array of length > 0 returns 400; archived workflow rejected.
- Playwright E2E (browser + Tauri): user clicks Run on the canvas, watches the Spren orb shift to `thinking` then `speaking`, sees the elapsed-time counter increment, sees the completion toast with cost. Cancel flow: click Cancel mid-run, confirm, run ends in `cancelled` state. Reconnect: kill the SSE connection mid-run, reconnect, observe Last-Event-ID resume picks up where it left off.
- Tauri-driver E2E: same flow in the Tauri webview (with WSL2 inconclusive caveat per Session 01 lessons).
- Visual regression baselines (Argos, Bundle B extension of Bundle A's baselines): canvas with Run button in `running` state, completion toast, `/runs` list with three rows (one of each terminal status), `/runs/{id}` placeholder page.
- Manual-verify checklist (implementer self-verification before claiming done), including the orb state transitions matching the spec in §10 and the cancel inline-confirmation flow.

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

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/routes/runs.py` | REST endpoint handlers (POST/GET/list/cancel + SSE). Inherits route-level auth + CORS regex per Session 01's pattern. |
| `packages/spren/src/spren/runs/lifecycle.py` | Async run lifecycle coordinator (`queued → running → terminal`). Owns workflow snapshot at run start + Orchestra spawn + result persistence. |
| `packages/spren/src/spren/runs/sse.py` | SSE wrapper around the framework's `AGUIEventStream`. Handles cold-reader replay from `trace.ndjson` + live consumption + reconnect via Last-Event-ID. |
| `packages/spren/src/spren/cost.py` | Cost calculation. Loads `cost_rates.yaml`, computes per-GenerationEvent + per-run aggregation. |
| `packages/spren/src/spren/cost_rates.yaml` | Provider/model rate table. Initial population: anthropic + openai + openrouter + google (the providers Spren v0.3 lists in `SUPPORTED_MODELS`). |
| `packages/spren/src/spren/models/run.py` | Pydantic models: `RunCreate`, `RunRead`, `RunListItem`, `RunStatus` enum, `TaskInput`. |
| `packages/spren/src/spren/storage/migrations/<N>__create_runs_table.py` | Forward-only migration. Indexes on `(workflow_id, created_at)` + `(status)` + `(created_at)` for the list-page queries. |
| `apps/web/src/routes/runs/index.tsx` | `/runs` list page. |
| `apps/web/src/routes/runs/$runId.tsx` | `/runs/{id}` thin placeholder page. |
| `apps/web/src/lib/run-sse.ts` | Client-side per-run SSE consumer hook. Uses `EventSource` with `Last-Event-ID` resume. Returns parsed AG-UI events with TypeScript types from `@ag-ui/core`. |
| `apps/web/src/lib/runs-list-sse.ts` | Client-side aggregate `/v1/runs/events` SSE hook. Returns a row-delta stream the `/runs` list page consumes to update in place. |
| `packages/spren/src/spren/runs/events_pubsub.py` | In-process pub/sub the run lifecycle coordinator emits to; the aggregate SSE endpoint subscribes per-client. Bounded async queue per subscriber (default 256 events, drop-oldest with `STREAM_LAGGED` notice). |
| `apps/web/src/components/RunButton.tsx` | The Run button component (idle / submitting / running / terminal states). Wires to the SSE hook. |
| `apps/web/src/components/CompletionToast.tsx` | The completion toast (shadcn `Sonner` or `Toast` underneath, themed per Session 03's design system). |
| `apps/web/src/components/StatusBadge.tsx` | Run status chip (5 status values). Reuses provenance-badge styling. |
| `apps/web/tests/e2e/run-execution.spec.ts` | Playwright golden-path. |
| `packages/spren/tests/integration/test_runs_lifecycle.py` | Pytest integration. |
| `packages/spren/tests/test_cost.py` | Pytest unit for cost.py. |
| `packages/spren/tests/test_runs_sse.py` | Pytest integration for the SSE reconnect path. |

### To DELETE

None. Session 04 is purely additive (new routes, new tables, new UI surfaces). The only DELETION would be the stub action Session 03 wired to the Run button — that stub is replaced in place when `RunButton.tsx` lands; no separate DELETE step.

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
| 5 | Framework's first agent (Researcher) starts generating tokens. AG-UI emits `TEXT_MESSAGE_START` / `TEXT_MESSAGE_CONTENT` (per token). | Canvas | Orb shifts to `speaking`. Token counter on the Run button increments. Elapsed timer ticks. |
| 6 | Researcher calls `search_web` tool. AG-UI emits `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END`. Cost accumulates from prior `GenerationEvent`. | Canvas | Orb returns to `thinking` briefly between agent turns. Token counter holds (tools don't add tokens directly; only generations do). |
| 7 | Writer agent's turn. Same flow. | Canvas | Continues. |
| 8 | Framework emits `RUN_FINISHED`. Spren updates `runs.status=succeeded`, `runs.finished_at`, `runs.final_response`, `runs.total_*` fields. | (server-side + canvas) | Orb returns to `idle`. Run button transitions to `terminal` state (briefly) then back to `idle`. Completion toast slides in: `Completed in 12.3s · $0.026`. |
| 9 | User clicks the toast. | Navigates to `/runs/{run_id}` | Thin placeholder page renders: status badge (`succeeded`), workflow name link back to canvas, total duration, total cost, total tokens. Below: `Trace viewer ships in Session 05` empty-state copy in the tag-markup typographic device (`<trace-viewer status="coming_in_session_05" />`). |
| 10 | User navigates to `/runs` via cmdk (`⌘K → "runs"`). | `/runs` list page | Renders cards: most recent run on top with status badge + workflow name + cost + duration. Filter chips: `All / Running / Succeeded / Failed / Cancelled`. |

### J-2 — Cancel mid-run

State: user is on the canvas with a workflow that takes ~60s to run (e.g., research-pipeline with browser-based tools).

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User clicks `Run`. | Canvas | Same as J-1 step 1. |
| 2 | Run starts; Researcher is mid-search after ~20s. | Canvas | Orb `speaking`. Cancel button visible on the Run button. |
| 3 | User clicks `Cancel`. | Inline confirmation appears below the Cancel button | Copy: `Cancel run? Tool calls in flight will finish.` + buttons `[Keep running] [Cancel run]`. |
| 4 | User clicks `Cancel run`. | `POST /v1/runs/{run_id}/cancel` | Server signals Orchestra to abort. Spinner replaces Cancel button briefly. |
| 5 | Orchestra cleanup completes within timeout (configurable; default 10s). Framework emits `RUN_FINISHED` with `interrupt: "cancelled"` (or AG-UI's equivalent). | (server-side + canvas) | Orb returns to `idle`. Run button returns to `idle`. Completion toast: `Cancelled after 22s · $0.008`. |

### J-3 — Reconnect after network blip

State: user is watching a run, browser tab momentarily loses focus + network hiccups close the SSE connection.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | Run is in progress, SSE streaming. | Canvas | Tokens counting, orb `speaking`. |
| 2 | Network drops; SSE connection closes. Client detects `EventSource.readyState === CLOSED`. | (client-side) | Run button shows a thin `Reconnecting…` annotation in `--ink-faint` next to the elapsed timer. |
| 3 | Client retries with exponential backoff (1s, 2s, 4s, max 8s). On reconnect, sends `Last-Event-ID` header with the last AG-UI event ID it received. | (client-side) | — |
| 4 | Server reads `trace.ndjson` from start to the last ID's offset, then switches to live consumption. Stream resumes seamlessly. | (server-side) | Token counter catches up to the live value within ~500ms. Orb state catches up. `Reconnecting…` annotation disappears. |

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
│  │ Cancel run? Tool calls   │   │         │ (countdown ticks 5→4→3→2→1→0;   │
│  │ in flight will finish.   │   │         │  force-abort fires at "1";      │
│  │ [Keep running][Cancel run]│  │         │  UI shows "Cancelled" at 0)     │
│  └──────────────────────────┘   │         │                                 │
└─────────────────────────────────┘         └─────────────────────────────────┘
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
2. **Spren orb reacts to run state on canvas.** Session 03 polish item 3 ships the presence orb (48-72px) in the canvas top-right; polish item 1 ships the four reactive states. Session 04 wires the canvas presence orb to the run's AG-UI events (`thinking` between turns, `speaking` during token streaming, `idle` after `RUN_FINISHED`).
3. **Live cost display on canvas.** Session 04 does NOT show live-accumulating cost during a run on the canvas. The token counter ticks; cost is final-only in the completion toast. Live cost is distracting in focus work + the cost-per-token rate table updates lag the model's pricing changes by days. Decision: ship final cost; add live cost in v0.4 if users ask.
4. **Workflow snapshot at run start (SP-009).** The full `definition` is frozen as `<data-dir>/data/runs/{run_id}/workflow.json` at the moment `POST /v1/runs` fires. The run executes against the snapshot; editing the workflow during a run does NOT affect the run. The `runs.workflow_id` foreign key references the live workflow row (for "what workflow was this a run of?") but the snapshot is the source of truth for what actually ran.
5. **Attachments in `task_input`.** Field accepted by the schema (Session 02 already includes `attachments: ["file_id_1", …]` in the data model), but a non-empty array returns `400 ATTACHMENTS_NOT_YET_SUPPORTED` in Session 04. Session 05 implements upload + reference resolution.
6. **Default task input on canvas Run.** Canvas's Run button submits `task_input: {text: "", attachments: []}` by default in Session 04 (no task-input modal yet). Workflows that depend on a task message will see an empty user message — that's an acceptable v0.3 limitation. Richer task input (a small dialog asking "What should the workflow work on?") lands in Session 06 when meta-agent integration arrives (the meta-agent owns the task-input experience).
7. **Cancel button placement + UX + cleanup timing.** Inline confirmation below the Cancel button (not a modal — modals are first-thought-laziness per design laws). Initial confirmation copy: `Cancel run? Tool calls in flight will finish.` + `[Keep running] [Cancel run]`. On confirm, the UI replaces the buttons with `Cancelling…` + a visible **5-second countdown** (5 → 4 → 3 → 2 → 1 → 0). At t=4s (when countdown displays 1) Spren sends a force-abort signal to the framework, ahead of the user-visible 0 — this overlaps the framework's force-abort latency with the last second of the countdown, so when the countdown reaches 0 the run is already terminated. UI displays `Cancelled` immediately at 0; toast confirms. Two-phase semantics: graceful (0-4s, framework drains tool calls) then forced (4-5s, framework hard-aborts).
8. **Reconnect retry backoff.** Exponential: 1s → 2s → 4s → max 8s, infinite retries until the user navigates away or the run terminates. No upper retry count — users on flaky networks should never see "permanently disconnected" unless they intend to.
9. **Live updates on `/runs` list page.** Aggregate `GET /v1/runs/events` SSE stream — one connection per client subscribing to ALL run-state-changed events for the whole list. Client mounts on first paint, updates rows in place as events arrive. Filter chips apply client-side. Per-card SSE (each running card opening its own stream) is explicitly rejected — it caps the list's scale at ~3-5 simultaneous running cards. The aggregate stream pattern scales to hundreds of runs at negligible cost (server keeps one event bus + one subscriber queue per client).
10. **Status badge palette.** Five status values, each with a single token from Session 03's palette:
   - `queued` → `--ink-soft` chip background, `--ink` text
   - `running` → `--peach` chip background, `--ink` text, leading `●` pulse animation
   - `succeeded` → `--magenta` chip background, white text
   - `failed` → `--magenta-deep` chip background, white text
   - `cancelled` → `--rule` chip background, `--ink-soft` text
   These reuse the brand palette without adding new colors. The `--peach` for running ties the canvas orb's `speaking` color to the list's running indicator.

11. **Bundle 02 structure.** Sessions 04 + 05 form one demo-able feature slice — run a workflow live then inspect what happened. Layout on disk: `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/{04,05}-*.md` with sibling `testing/test-scenarios.md` (user-facing scenario list) + `testing/test-session.md` (Claude Code testing-agent brief), matching Bundle 01's pattern at `docs/implementation/spren/v0.3.0/01-visual-builder/`. Session 04's contribution to `test-scenarios.md` lands when Session 04 ships; Session 05's contribution lands when Session 05 ships; the bundle-end test runs only after both sessions ship + are individually green. Bundle 02's `test-scenarios.md` is fleshed out alongside this brief (Session 04 covered now; Session 05 placeholders that get expanded when Session 05 ships) so Session 04's implementer has a target structure to write the per-session manual-verify checklist against.

## 9. Design system additions

Session 04 adds NO new design-system tokens, fonts, or layout primitives. It reuses Session 03's complete system. New components shipped (`RunButton`, `CompletionToast`, `StatusBadge`, `RunCard`) all instantiate the existing tokens.

One pattern Session 04 establishes (formalizes for later sessions): **live-state UI elements** (the running run button, the running row on `/runs`) share a common motion idiom — a leading `●` glyph that pulses at the same cadence as the Spren orb's `speaking` state (`cubic-bezier(0.45, 0, 0.55, 1)`, ~2s period, peaks at 1.0 opacity, dips to 0.4). This binds the visual identity of "something is happening" across the orb + the run UI. Implement once in a `<PulseDot color={...} />` component; reuse across the run button, status badges, and any future live indicators.

## 10. Polish items to address inside Session 04

These are gaps the architect-stage draft surfaced that the implementer addresses in-session, not as nice-to-haves.

1. **SSE reliability through Tauri webview.** The browser's `EventSource` works fine in browsers; the Tauri webview's `EventSource` had historic quirks (auto-reconnect behavior, header passthrough). Implementer benchmarks SSE through the Tauri shell on macOS / Windows / Linux against the same flow in browser. If any platform requires a fallback (e.g., long-poll), the fallback is wired here, not punted.
2. **Cold-reader replay performance.** Reading `trace.ndjson` from start on every reconnect is wasteful for long runs. Implementer benchmarks: at what trace size does replay take longer than 500ms? If that threshold is hit for typical v0.3 workloads (3-agent, 30-second runs), implementer adds an offset index (`trace.ndjson.offset` — a small file mapping event_id → byte offset) so reconnects can `seek()` rather than scan.
3. **Cost rate freshness.** The YAML file ships with model pricing as of the release date. Models get repriced quarterly-ish. Add a `last_updated: YYYY-MM-DD` field to the YAML + log a warning on daemon start if older than 90 days. Defer auto-update (calling provider pricing APIs) to v0.4.
4. **Run button keyboard shortcut.** ⌘R or similar should trigger Run on the canvas. Implementer picks the shortcut + documents in the cmdk overlay.
5. **`/runs` list pagination performance at scale.** Cursor-based pagination per architecture; the implementer measures list rendering at 1000 / 10000 rows in the DB and confirms acceptable. If not acceptable, server-side virtual scrolling via the cursor pattern.
6. **Completion toast accessibility.** Toast must use `aria-live="polite"` (not steal focus), have a close button (some users want manual dismiss), and respect `prefers-reduced-motion` (skip the slide-in animation).
7. **Reduced-motion fallback for the running orb.** When `prefers-reduced-motion` is on, the canvas presence orb's `thinking` + `speaking` states must NOT animate — they degrade to static color shifts (peach for speaking, slightly warmer for thinking). The Spren orb base spec (Session 03 §9.3) already covers this; Session 04 verifies the `RunButton`'s `PulseDot` also respects it.
8. **Reconnect annotation copy.** `Reconnecting…` is generic. Implementer picks specific copy per state (e.g., `Reconnecting in 4s…` showing the next retry, or `Server unreachable — retrying…` after multiple failures). Decide during implementation; small UX win.
9. **Empty `/runs` list state.** Tag-markup typographic device: `<runs status="empty" />` with body copy `No runs yet. Build a workflow and click Run.` + a button linking to `/workflows`.
10. **Empty `/runs/{id}` failed-run state.** When a run fails, the placeholder page should show the error message above the trace-viewer-empty-state. Not just "trace viewer coming in 05" — also show what failed. Implementer decides the error formatting (full stack trace? short message + "expand for details"? — pick during implementation).

## 11. Success criteria

Extending Bundle A's `test-scenarios.md` (and feeding into Bundle B's `test-scenarios.md` once that file is created):

- **G-08** (run execution golden path): user opens a saved workflow, clicks Run, watches the orb state transitions (idle → thinking → speaking → idle), sees the Run button transition through states, receives the completion toast with cost, clicks through to `/runs/{id}` and sees the row metadata.
- **G-09** (cancel mid-run): user clicks Run, then Cancel mid-run, confirms in the inline prompt, watches the run terminate cleanly with `cancelled` status.
- **G-10** (reconnect): user starts a long-running workflow, closes the browser tab mid-run, reopens it; SSE re-establishes from `Last-Event-ID` and the UI catches up to current state within 500ms of reconnect.
- **G-11** (run list): user navigates to `/runs`, sees a list of recent runs with correct statuses + costs + durations; filter chips work; clicking a run navigates to `/runs/{id}`.
- **U-06** (manual smoke): from a fresh data dir, build a workflow → run it → see cost computed correctly → see run on `/runs` → cancel a running workflow → verify it terminates.
- **Visual regression baselines (Bundle B extension)**: Run button states (idle / submitting / running / cancel-confirm-open), completion toast, `/runs` list (with rows of each status), `/runs/{id}` placeholder.
- **X-07** (offline behavior): browser loses network mid-run; UI reflects reconnecting state; on reconnect, full state catches up; no events lost.
- **C-01** (cost calculation accuracy): a 3-agent run with known token counts (mocked LLM) produces a cost within 0.01¢ of hand-calculated.
- **C-02** (cost calculation unknown model): a workflow using a model NOT in the rate table completes successfully with `total_cost_usd=0.0` and a warn-level log entry naming the missing model.

## 12. Open research items the implementer resolves in-flight

Empirical version + behavior verification done during implementation, not via a separate pre-implementation research stage.

- AG-UI Python package version pin (`ag-ui-protocol` if it exists at implementation time; else import directly from framework Session 06's surface).
- AG-UI npm package pins for the JS side: `@ag-ui/core` and `@ag-ui/client` exact versions.
- SSE behavior in the Tauri 2 webview (per polish item 1) — empirical benchmark across macOS / Windows / Linux.
- `EventSource` `Last-Event-ID` header behavior across Chromium (browser) vs WebKit2GTK (Tauri Linux) vs WKWebView (Tauri macOS).
- Cost calculation: any `GenerationEvent` shape changes from the framework since the architecture doc was written (verify against current `coordination/status/events.py`).
- Cancel cleanup two-phase timing: Session 04 commits to a 5-second user-visible countdown with internal force-abort fired at t=4s (see §8.7). Verify the framework's `Orchestra` API supports a force-abort signal distinct from regular cancel. If not, the framework architect coordinates a new framework signal — surface to the framework architect agent before Spren Session 04 implementation; do NOT hack a Spren-side "I'm just considering it cancelled" overlay if the framework keeps running cleanup in the background (that would leak resources).
- Token-counter source: AG-UI `TOOL_CALL_END` vs raw `GenerationEvent` from `EventBus` — which gives reliable token counts in time for the live counter on the Run button? (Answer is probably: accumulate per `GenerationEvent` consumed from the AG-UI stream; verify.)

If any of these surface a conflict with the locked decisions in §8, the implementer flags it and asks before deviating.

## 13. Status

- [x] Tier confirmed (HIGH).
- [x] Scope boundaries confirmed (sections 3 + 4).
- [x] Files-to-CREATE list approved (section 5).
- [x] Three user journeys approved (section 6).
- [x] Skeleton wireframes approved (section 7).
- [x] Decisions locked (section 8).
- [x] Architecture doc updates landed (02-data-model.md, 03-api-design.md, 06-observability.md — AG-UI translator relocated from Spren-side to framework-side per Framework Session 06).
- [x] Polish items captured for in-session work (section 10).
- [x] Success criteria affirmed (section 11).
- [ ] Acceptance criteria frozen at [`./04-run-execution/acceptance.md`](./04-run-execution/acceptance.md) — extracted by the `acceptance-criteria-extractor` agent on the first implementation turn, before any code is written.
- [ ] Framework Session 06 (AG-UI translator) merged and released before Spren Session 04 implementation begins. Brief in [`../../../../framework/sessions/v0.3.0/06-aggui-translator.md`](../../../../framework/sessions/v0.3.0/06-aggui-translator.md).
- [ ] Session implementation complete (all acceptance criteria pass; polish items addressed; tests green; manual verify done).

**Next step:** acceptance-criteria-extractor freezes `./04-run-execution/acceptance.md` on the first implementation turn. Implementer begins after framework Session 06 lands. Polish items in §10 are explicit acceptance criteria — scoped into the session, not nice-to-haves.

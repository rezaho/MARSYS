# Acceptance criteria — Spren Session 04 (Run Execution + AG-UI Streaming + Cost)

Frozen at 2026-05-13T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Criteria tagged `[blocked-on: framework-06]` depend on Framework Session 06 (AG-UI translator at `marsys.coordination.aggui`) merging.
Criteria tagged `[blocked-on: framework-07]` depend on Framework Session 07 (`Orchestra.cancel_session`) merging.
During the parallel-implementation period, those criteria's tests skip-pass via runtime presence checks (`importlib.util.find_spec("marsys.coordination.aggui") is not None` for 06; `hasattr(Orchestra, "cancel_session")` for 07); they light up automatically when the framework dependencies merge.

## Functional — Backend `runs` table + migration

- AC-1: A forward-only migration (e.g., at `packages/spren/src/spren/storage/migrations/02__create_runs.py`) creates a `runs` table when applied to a fresh data dir.
- AC-2: The `runs` table has columns including at minimum `id` (TEXT primary key), `workflow_id` (TEXT foreign key to `workflows.id`), `status` (TEXT), `created_at`, `started_at`, `finished_at`, `total_duration_ms`, `total_cost_usd`, `total_tokens_input`, `total_tokens_output`, `total_steps`, `final_response`, `error`, and `trigger`.
- AC-3: The `runs` table has indexes on `(workflow_id, created_at)`, `(status)`, and `(created_at)`.
- AC-4: The migration applies cleanly twice in a row without raising (idempotent or guarded by version table).
- AC-5: The migration's `status` column accepts the string values `queued`, `running`, `cancelling`, `succeeded`, `failed`, and `cancelled` (no schema change required when extending the enum because the column is TEXT).

## Functional — Backend Pydantic models

- AC-6: A `RunStatus` enum exists with exactly the six values `queued`, `running`, `cancelling`, `succeeded`, `failed`, `cancelled`.
- AC-7: A `RunCreate` Pydantic model exists accepting `workflow_id: str` and `task_input: TaskInput`.
- AC-8: A `TaskInput` Pydantic model exists with `text: str` (default empty string) and `attachments: list[str]` (default empty list).
- AC-9: A `RunRead` Pydantic model exists exposing every persisted field of a `runs` row (id, workflow_id, status, created_at, started_at, finished_at, total_duration_ms, total_cost_usd, total_tokens_input, total_tokens_output, total_steps, final_response, error, trigger).
- AC-10: A `RunListItem` Pydantic model exists exposing the fields needed for the `/runs` list view (id, workflow_id, status, created_at, finished_at, total_duration_ms, total_cost_usd).
- AC-11: An aggregate SSE event union type exists with discriminated members `RunCreatedEvent`, `RunUpdatedEvent`, `RunFinishedEvent`, `RunCancelledEvent` keyed by a `type` discriminator field.
- AC-12: Every cross-boundary Pydantic payload (`RunCreate`, `RunRead`, `RunListItem`, each member of the SSE event union) carries a field `schema_version: int = 1`.
- AC-13: `RunCreate.task_input.attachments` defaulting to `[]` is accepted; the API surface rejection of non-empty `attachments` is enforced at the handler layer (AC-19), not the Pydantic schema.

## Functional — `POST /v1/runs`

- AC-14: A `POST /v1/runs` route is mounted on the FastAPI app and is reachable.
- AC-15: `POST /v1/runs` without a valid auth token returns HTTP 401.
- AC-16: `POST /v1/runs` for an unknown `workflow_id` returns HTTP 404.
- AC-17: `POST /v1/runs` for an archived workflow returns HTTP 400 with response body containing `code: "WORKFLOW_ARCHIVED"`.
- AC-18: `POST /v1/runs` for a valid request returns HTTP 201 with a JSON body containing `run_id` (string) and `status` (string equal to `"queued"`).
- AC-19: `POST /v1/runs` with a non-empty `task_input.attachments` array returns HTTP 400 with response body containing `code: "ATTACHMENTS_NOT_YET_SUPPORTED"`.
- AC-20: `POST /v1/runs` with a `trigger` field set to any value other than `manual` (e.g., `scheduled`, `webhook`, `messenger:*`) returns HTTP 400 with response body containing `code: "TRIGGER_NOT_YET_SUPPORTED"`.
- AC-21: A successful `POST /v1/runs` inserts a row into the `runs` table with the returned `run_id`, `status="queued"`, and the supplied `workflow_id`.
- AC-22: A successful `POST /v1/runs` writes a frozen workflow snapshot file at `<data-dir>/data/runs/{run_id}/workflow.json` whose contents equal the workflow's `definition` at the moment of the POST.
- AC-23: After a successful `POST /v1/runs` returns, the SSE endpoint at `GET /v1/runs/{id}/events` finds a wired AG-UI translator for that run (subscribers attaching immediately do not 404 due to a wiring race). [blocked-on: framework-06]
- AC-24: `POST /v1/runs` registers an active-run record in the in-process registry containing the orchestra reference, lifecycle task handle, a `started: asyncio.Event`, and an event-replay deque.
- AC-25: Editing the source workflow (`PUT /v1/workflows/{id}`) after a `POST /v1/runs` does NOT mutate the snapshot file at `<data-dir>/data/runs/{run_id}/workflow.json`.

## Functional — `GET /v1/runs/{id}`

- AC-26: A `GET /v1/runs/{id}` route is mounted on the FastAPI app and is reachable.
- AC-27: `GET /v1/runs/{id}` without a valid auth token returns HTTP 401.
- AC-28: `GET /v1/runs/{id}` for an unknown id returns HTTP 404.
- AC-29: `GET /v1/runs/{id}` for a known id returns HTTP 200 with a body matching the `RunRead` shape and `schema_version: 1`.

## Functional — `GET /v1/runs` (list + filters + cursor)

- AC-30: A `GET /v1/runs` route is mounted on the FastAPI app and is reachable.
- AC-31: `GET /v1/runs` without a valid auth token returns HTTP 401.
- AC-32: `GET /v1/runs` returns HTTP 200 with a JSON body shaped `{"items": [...], "next_cursor": <string or null>}`.
- AC-33: Each element of `items` matches the `RunListItem` shape (carrying `schema_version: 1`).
- AC-34: `GET /v1/runs?workflow_id=<id>` returns only rows whose `workflow_id` equals the supplied value.
- AC-35: `GET /v1/runs?status=running` returns only rows whose `status` equals `running`; the same applies to every value in `RunStatus`.
- AC-36: `GET /v1/runs?since=<ISO8601 timestamp>` returns only rows whose `created_at` is at or after the supplied timestamp.
- AC-37: `GET /v1/runs` results are ordered by `created_at` descending (most recent first).
- AC-38: When the result set exceeds the page size, `next_cursor` is non-null and supplying it as `?cursor=<value>` returns the next page without overlap.
- AC-39: When the result set is fully returned in one page, `next_cursor` is `null`.

## Functional — `POST /v1/runs/{id}/cancel`

- AC-40: A `POST /v1/runs/{id}/cancel` route is mounted on the FastAPI app and is reachable.
- AC-41: `POST /v1/runs/{id}/cancel` without a valid auth token returns HTTP 401.
- AC-42: `POST /v1/runs/{id}/cancel` for an unknown id returns HTTP 404.
- AC-43: `POST /v1/runs/{id}/cancel` on a `queued` run transitions the row immediately to `cancelled` and returns HTTP 200.
- AC-44: `POST /v1/runs/{id}/cancel` on a `running` run transitions the row to `cancelling`, then calls `Orchestra.cancel_session(session_id, force_after=5.0)`, then transitions the row to `cancelled` once the framework call returns; the handler returns HTTP 200 with the updated row. [blocked-on: framework-07]
- AC-45: `POST /v1/runs/{id}/cancel` on a run already in `cancelling` returns HTTP 409.
- AC-46: `POST /v1/runs/{id}/cancel` on a terminal run (`succeeded`, `failed`, `cancelled`) returns HTTP 409.
- AC-47: `POST /v1/runs/{id}/cancel` on a `paused` run returns HTTP 409 (paused state ships in v0.4 and is not cancellable in v0.3).
- AC-48: When `Orchestra.cancel_session` does not return within a 10s server-side watchdog window, the row is marked `cancelled` regardless and the handler returns HTTP 200; a WARN log entry is emitted naming the run id. [blocked-on: framework-07]
- AC-49: When the watchdog fires, a `Custom("marsys.cancel.timeout")` AG-UI event is emitted on the run's SSE stream. [blocked-on: framework-06] [blocked-on: framework-07]

## Functional — `GET /v1/runs/{id}/events` (per-run SSE)

- AC-50: A `GET /v1/runs/{id}/events` route is mounted on the FastAPI app and is reachable.
- AC-51: `GET /v1/runs/{id}/events` without a valid auth token returns HTTP 401.
- AC-52: `GET /v1/runs/{id}/events` for an unknown id returns HTTP 404.
- AC-53: `GET /v1/runs/{id}/events` for a terminal run (no live `ActiveRun`) returns HTTP 204 No Content.
- AC-54: `GET /v1/runs/{id}/events` for a live run returns an SSE response (`Content-Type: text/event-stream`) and the body streams AG-UI events serialized via `aggui_event_to_sse(event)`. [blocked-on: framework-06]
- AC-55: When the per-run SSE handshake races the lifecycle task startup, the endpoint waits up to 2s on the run's `started: asyncio.Event` before streaming; if exceeded, returns HTTP 503 with a `Retry-After` header.
- AC-56: When a client sends `Last-Event-ID: <ulid>` on connect, the endpoint replays events from the in-memory deque whose `event_id` is greater than the supplied value, then switches to live consumption. [blocked-on: framework-06]
- AC-57: When `Last-Event-ID` is older than the deque's oldest entry (deque holds 1024 events; older ids fall outside the window), the endpoint emits exactly one `Custom("marsys.stream.gap")` event before switching to live consumption. [blocked-on: framework-06]
- AC-58: Each AG-UI event emitted on the SSE stream carries an `event_id` that is a ULID (lexicographically sortable). [blocked-on: framework-06]

## Functional — `GET /v1/runs/events` (aggregate SSE) + `RunsBroker`

- AC-59: A `GET /v1/runs/events` route is mounted on the FastAPI app and is reachable.
- AC-60: `GET /v1/runs/events` without a valid auth token returns HTTP 401.
- AC-61: `GET /v1/runs/events` returns an SSE response (`Content-Type: text/event-stream`).
- AC-62: When a new run is created (via `POST /v1/runs`), all subscribed clients receive a `RunCreatedEvent` on the aggregate stream with `schema_version: 1`.
- AC-63: When a run's status changes (e.g., `queued → running`, `running → cancelling`), all subscribed clients receive a `RunUpdatedEvent` with the new `status` and `schema_version: 1`.
- AC-64: When a run reaches a non-cancelled terminal status (`succeeded`, `failed`), all subscribed clients receive a `RunFinishedEvent` with `schema_version: 1`.
- AC-65: When a run reaches `cancelled`, all subscribed clients receive a `RunCancelledEvent` with `schema_version: 1`.
- AC-66: A `RunsBroker` singleton (e.g., at `packages/spren/src/spren/runs/broker.py`) supports `subscribe()` returning a per-client async iterator and `publish(event)` non-blocking.
- AC-67: Each subscriber's queue has a bounded capacity of 256; on overflow, oldest events are dropped and the next yielded event is preceded by a `STREAM_LAGGED` marker.
- AC-68: `RunsBroker` does NOT subscribe to or republish events from the framework's `EventBus` (Spren-emitted aggregate events are produced exclusively by Spren-side lifecycle code).
- AC-69: When a subscriber disconnects mid-stream, the broker deregisters its queue and does not leak an orphan task.

## Functional — `runs/lifecycle.py` state machine + `_active_runs` registry

- AC-70: A module-level `_active_runs: dict[str, ActiveRun]` registry exists where `ActiveRun` carries the orchestra, the lifecycle task, a `started: asyncio.Event`, and an event-replay deque.
- AC-71: The registry exposes `register(run_id, active_run)`, `get(run_id) -> ActiveRun | None`, and `deregister(run_id)` operations.
- AC-72: A run's lifecycle task transitions `queued → running` on `Orchestra.run()` start (sets `runs.started_at`).
- AC-73: A run's lifecycle task transitions `running → succeeded` on a clean `Orchestra.run()` completion, persisting `runs.finished_at`, `runs.total_duration_ms`, `runs.total_steps`, and `runs.final_response`.
- AC-74: A run's lifecycle task transitions `running → failed` on an `Orchestra.run()` exception, persisting `runs.finished_at`, `runs.total_duration_ms`, and `runs.error` (the error message).
- AC-75: The lifecycle task deregisters its run id from `_active_runs` on terminal status (succeeded / failed / cancelled).
- AC-76: The replay deque held on the `ActiveRun` record is bounded at 1024 entries (FIFO drop-oldest beyond that). [blocked-on: framework-06]
- AC-77: On FastAPI lifespan shutdown, every in-flight task in `_active_runs` is cancelled.

## Functional — `runs/materialize.py` (spec → runtime)

- AC-78: A function (e.g., `materialize_run(definition, secrets) -> RuntimeBundle`) exists that converts a `WorkflowDefinition` Pydantic model into a runtime bundle containing a `marsys.coordination.topology.core.Topology`, an `AgentRegistry`, and an `ExecutionConfig`.
- AC-79: The materializer instantiates each `AgentSpec` as a `BaseAgent` subclass with the API key resolved from the secrets store (per-provider, not per-agent).
- AC-80: The materializer resolves tool name strings on each agent to callables via `marsys.environment.tools.AVAILABLE_TOOLS`.
- AC-81: The materializer assembles `ExecutionConfig` with `aggui = AGGUIConfig(enabled=True)`. [blocked-on: framework-06]
- AC-82: The materializer assembles `ExecutionConfig` with `storage_backend = FileStorageBackend(<data-dir>/data/runs)`.
- AC-83: An unknown tool name in the workflow (not in `AVAILABLE_TOOLS`) raises a clear materialization error before `Orchestra` is constructed.
- AC-84: An agent referencing a model without a corresponding API key in the secrets store raises a clear materialization error before `Orchestra` is constructed.

## Functional — `cost.py` + `cost_rates.py`

- AC-85: A `RATES: dict[tuple[str, str], CostRate]` constant exists keyed by `(provider, model)` tuples (Python module, not YAML).
- AC-86: `RATES` includes entries for the providers `anthropic`, `openai`, `openrouter`, `google`, and `xai`.
- AC-87: A `LAST_UPDATED: date` constant exists in the cost-rates module.
- AC-88: `CostRate` is a frozen dataclass exposing `input_per_1m_usd`, `output_per_1m_usd`, and `reasoning_per_1m_usd: Decimal | None`.
- AC-89: A `calculate_cost(provider, model, prompt_tokens, completion_tokens, reasoning_tokens) -> Decimal` function returns `(prompt × in_rate + completion × out_rate + reasoning × reasoning_rate) / 1_000_000` for a known `(provider, model)` pair.
- AC-90: `calculate_cost` for an unknown `(provider, model)` returns `Decimal("0")` and emits a WARN log entry naming the missing provider + model.
- AC-91: When a run consumes a `Custom("marsys.generation.metadata")` event with a known `(provider, model)`, `runs.total_cost_usd`, `runs.total_tokens_input`, and `runs.total_tokens_output` are incremented accordingly. [blocked-on: framework-06]
- AC-92: Daemon startup logs a WARN entry when `LAST_UPDATED` is older than 90 days from the current date.
- AC-93: For a 3-agent run with a static-response model adapter producing a known token count, `runs.total_cost_usd` matches the hand-calculated value within `Decimal("0.0001")` (0.01¢). [blocked-on: framework-06]
- AC-94: For a workflow whose model is not in `RATES`, the run completes successfully with `runs.total_cost_usd = 0` and a WARN log naming the missing `(provider, model)`. [blocked-on: framework-06]

## Functional — Auth + CORS

- AC-95: Every new endpoint introduced in Session 04 (`POST /v1/runs`, `GET /v1/runs`, `GET /v1/runs/{id}`, `POST /v1/runs/{id}/cancel`, `GET /v1/runs/{id}/events`, `GET /v1/runs/events`) returns HTTP 401 without a valid auth token.
- AC-96: The CORS regex inherited from `server.create_app()` (`^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$`) applies to every new endpoint without per-route reconfiguration.

## Functional — Schema versioning on cross-boundary payloads

- AC-97: Every JSON response body for the run REST endpoints includes a `schema_version: 1` field at the top-level model boundary.
- AC-98: Every event emitted on `GET /v1/runs/events` includes `schema_version: 1`.

## Functional — Dependency declarations

- AC-99: `packages/spren/pyproject.toml` declares `sse-starlette>=2.1` as a runtime dependency.
- AC-100: `packages/spren/pyproject.toml` declares an `[project.optional-dependencies] aggui` extras block referencing `marsys[aggui]`.
- AC-101: `apps/web/package.json` declares `@ag-ui/core` and `@ag-ui/client` (pinned exact versions).

## Functional — Frontend `/runs` list page

- AC-102: TanStack Router defines route `/runs` rendering a list page (e.g., at `apps/web/src/routes/runs/index.tsx`).
- AC-103: The `/runs` list page issues `GET /v1/runs` on first paint and renders one card per returned item.
- AC-104: Each row card renders the workflow name, a status badge, total cost (USD), total duration (ms), and a relative timestamp.
- AC-105: Clicking a row navigates to `/runs/{run_id}`.
- AC-106: The `/runs` list page renders filter chips labelled at minimum `All`, `Running`, `Cancelling`, `Succeeded`, `Failed`, `Cancelled`.
- AC-107: Selecting a filter chip restricts visible rows to those matching the chip's status (or shows all rows for `All`).
- AC-108: The `/runs` list page mounts `GET /v1/runs/events` on first paint and applies `RunCreated` / `RunUpdated` / `RunFinished` / `RunCancelled` events to the visible rows in place (no full refetch).
- AC-109: When the `/runs` list is empty, the page renders an empty state with copy directing the user to `/workflows` to build one.

## Functional — Frontend `/runs/{id}` placeholder

- AC-110: TanStack Router defines route `/runs/{id}` (e.g., at `apps/web/src/routes/runs/$runId.tsx`).
- AC-111: The `/runs/{id}` page issues `GET /v1/runs/{id}` and renders the row's status badge, workflow name (linking to `/workflows/{workflow_id}`), total duration, total cost, total tokens (input + output), `started_at` (with relative anchor), and `finished_at`.
- AC-112: The `/runs/{id}` page renders an empty-state placeholder for the trace viewer using the tag-markup typographic device (e.g., `<trace-viewer status="coming_in_session_05" />`) plus body copy noting Session 05.

## Functional — `RunButton` component + state machine

- AC-113: A `RunButton` component is rendered in the canvas top toolbar at `/workflows/{workflowId}` after the `Save` button.
- AC-114: When no run is active, `RunButton` shows the label `Run`.
- AC-115: Clicking `Run` transitions the button to a `submitting` state with a spinner visible for at least 200ms while `POST /v1/runs` is in flight.
- AC-116: On a successful `POST /v1/runs` response, `RunButton` transitions to a `running` state showing `Cancel · <elapsed seconds>s · <token count>t`.
- AC-117: While in `running` state, the elapsed-seconds counter increments at least once per second.
- AC-118: While in `running` state, the token counter increments on each `Custom("marsys.generation.metadata")` AG-UI event by the sum of `prompt_tokens + completion_tokens + reasoning_tokens` from that event. [blocked-on: framework-06]
- AC-119: Clicking the button while in `running` state opens an inline confirmation prompt (NOT a modal) with the copy `Cancel run? Tool calls in flight will finish.` and two buttons `Keep running` and `Cancel run`.
- AC-120: Clicking `Cancel run` issues `POST /v1/runs/{id}/cancel` and transitions the button to a `cancelling` state showing `Cancelling… <countdown>` with the countdown decrementing from 5 to 0 over 5 seconds.
- AC-121: When the framework returns terminal before the countdown finishes, the button shortcuts the countdown and proceeds to the terminal phase.
- AC-122: On run terminal (`succeeded` / `failed` / `cancelled`), `RunButton` returns to the `idle` state.
- AC-123: `RunButton` reads its state from a jotai store (e.g., `apps/web/src/stores/run.ts`) exposing at minimum `activeRunIdAtom`, `runStateAtom`, `tokenCountAtom`, `elapsedMsAtom`, and `runStatusAtom`.

## Functional — `CompletionToast` component

- AC-124: A `CompletionToast` component is rendered (e.g., at `apps/web/src/components/CompletionToast/`) and appears when a run reaches terminal.
- AC-125: For a `succeeded` run, the toast renders `Completed in <duration> · $<cost>` plus the workflow name.
- AC-126: For a `failed` run, the toast renders `Failed: <short reason>` (the full error is on `/runs/{id}`).
- AC-127: For a `cancelled` run, the toast renders `Cancelled after <duration> · $<cost>`.
- AC-128: The toast auto-dismisses 6 seconds after appearing.
- AC-129: Clicking the toast navigates to `/runs/{run_id}`.
- AC-130: The toast root element carries `aria-live="polite"`.
- AC-131: The toast renders a manual dismiss (close) button.
- AC-132: Under `prefers-reduced-motion: reduce`, the toast appears immediately without a slide-in animation.

## Functional — `StatusBadge` component

- AC-133: A `StatusBadge` component is rendered (e.g., at `apps/web/src/components/StatusBadge/`) accepting a `status` prop.
- AC-134: `StatusBadge` accepts and renders all six status values: `queued`, `running`, `cancelling`, `succeeded`, `failed`, `cancelled`.
- AC-135: The `queued` variant uses `--ink-soft` chip background with `--ink` text.
- AC-136: The `running` variant uses `--peach` chip background with `--ink` text and a leading pulsing `●` glyph.
- AC-137: The `cancelling` variant uses `--peach` chip background with `--ink` text and a leading pulsing `●` glyph (same active cadence as `running`).
- AC-138: The `succeeded` variant uses `--magenta` chip background with white text.
- AC-139: The `failed` variant uses `--magenta-deep` chip background with white text.
- AC-140: The `cancelled` variant uses `--rule` chip background with `--ink-soft` text.

## Functional — `PulseDot` shared primitive

- AC-141: A `PulseDot` component is rendered (e.g., at `apps/web/src/components/PulseDot/`) accepting a `color` prop.
- AC-142: `PulseDot` runs an opacity animation that peaks at `1.0` and dips to `0.4` on a ~2s period.
- AC-143: `PulseDot` uses the timing function `cubic-bezier(0.45, 0, 0.55, 1)`.
- AC-144: Under `prefers-reduced-motion: reduce`, `PulseDot`'s pulsing animation is disabled (the dot remains visible without pulsing).
- AC-145: `PulseDot` is consumed by both `RunButton` (in `running` state) and `StatusBadge` (in `running` and `cancelling` states).

## Functional — `PresenceOrb` extension

- AC-146: `PresenceOrb` accepts an optional `state?: SprenState` prop (defaulting to `"idle"` when omitted) without breaking any existing callers.
- AC-147: `PresenceOrb`'s existing `onClick` chat-sheet behavior remains unchanged regardless of the `state` prop value.
- AC-148: When `state="thinking"` is passed, the rendered orb is visually distinct from `state="idle"` (e.g., distinguishable by `data-state="thinking"`).
- AC-149: When `state="speaking"` is passed, the rendered orb is visually distinct from `state="thinking"` (e.g., distinguishable by `data-state="speaking"`).

## Functional — Canvas integration

- AC-150: The canvas at `/workflows/{workflowId}` renders the new `RunButton` after the existing `Save` button in the top toolbar.
- AC-151: The canvas wraps `PresenceOrb` and passes the jotai-derived run state into `state`.
- AC-152: When a run is in flight, the canvas presence orb's `data-state` reflects `thinking` between agent turns and `speaking` during a turn (driven by `TextMessageStart` / `TextMessageEnd` AG-UI events). [blocked-on: framework-06]
- AC-153: When a run reaches terminal, the canvas presence orb's `data-state` returns to `idle`.

## Functional — Jotai store + SSE hook

- AC-154: A jotai store exists (e.g., `apps/web/src/stores/run.ts`) exposing `activeRunIdAtom`, `runStateAtom`, `tokenCountAtom`, `elapsedMsAtom`, and `runStatusAtom`.
- AC-155: A hook (e.g., `useRunSse`) opens an SSE connection to `GET /v1/runs/{id}/events` after a successful `POST /v1/runs` and writes derived state into the jotai atoms. [blocked-on: framework-06]
- AC-156: When the SSE connection closes (`EventSource.readyState === CLOSED`), the hook attempts reconnect with exponential backoff at 1s, 2s, 4s, max 8s, retrying indefinitely until the run terminates or the page unmounts.
- AC-157: On reconnect, the hook sends the most-recently-received AG-UI `event_id` as a `Last-Event-ID` HTTP header. [blocked-on: framework-06]
- AC-158: While the SSE connection is reconnecting, the canvas surfaces a `Reconnecting…` annotation in `--ink-faint` near the elapsed timer.
- AC-159: When the SSE stream emits `Custom("marsys.stream.gap")`, the hook refetches the run row via `GET /v1/runs/{id}` and resumes live consumption from that point. [blocked-on: framework-06]

## Functional — Generated TypeScript types

- AC-160: The generated TypeScript file (e.g., `apps/web/src/lib/api-types.generated.ts`) exposes types corresponding to the Pydantic models in `models/run.py` (`RunCreate`, `RunRead`, `RunListItem`, `RunStatus`, `TaskInput`, the SSE event union).
- AC-161: No hand-written interface declarations mirroring the run Pydantic shapes exist anywhere in `apps/web/src/` (client types come from the generated file).

## Functional — cmdk command

- AC-162: The cmdk command palette registers a `Go to runs` command in the `Navigate` section.
- AC-163: Selecting `Go to runs` from cmdk navigates the router to `/runs` and dismisses the overlay.

## Functional — User journey checkpoints (Bundle B demo gate)

These are end-to-end demonstration checks; each maps to an observable outcome.

### J-1 — First run (canvas Run → completion)

- AC-164: J-1 step 1: clicking `Run` on the canvas of a saved workflow fires exactly one `POST /v1/runs` with `{workflow_id, task_input: {text: "", attachments: []}}`.
- AC-165: J-1 step 3: after `POST /v1/runs` returns, the client opens an SSE connection at `GET /v1/runs/{run_id}/events` and `RunButton` transitions to `running` state. [blocked-on: framework-06]
- AC-166: J-1 step 4: the canvas presence orb's `data-state` becomes `thinking` while the orchestra is starting. [blocked-on: framework-06]
- AC-167: J-1 step 5: the orb's `data-state` becomes `speaking` during each agent turn (between `TextMessageStart` and `TextMessageEnd`). [blocked-on: framework-06]
- AC-168: J-1 step 8: on `RUN_FINISHED`, `runs.status` becomes `succeeded`, `runs.finished_at` is set, `runs.final_response` is set, and the `total_*` aggregates are persisted. [blocked-on: framework-06]
- AC-169: J-1 step 8: the canvas presence orb returns to `data-state="idle"`, `RunButton` returns to `idle`, and `CompletionToast` slides in showing `Completed in <duration> · $<cost>`.
- AC-170: J-1 step 9: clicking the toast navigates to `/runs/{run_id}` showing the row metadata.
- AC-171: J-1 step 10: navigating to `/runs` via cmdk renders the new run as the most recent row with the correct status badge, cost, and duration.

### J-2 — Cancel mid-run

- AC-172: J-2 step 3: clicking `Cancel` while in `running` state opens the inline confirmation with the documented copy.
- AC-173: J-2 step 4: clicking `Cancel run` fires exactly one `POST /v1/runs/{run_id}/cancel`, the row transitions to `cancelling`, and the `RunButton` shows `Cancelling… 5` with countdown decrementing from 5 to 0. [blocked-on: framework-07]
- AC-174: J-2 step 5: when the framework's quiesce drains in flight before 5s, the row transitions to `cancelled` early and the button countdown shortcuts. [blocked-on: framework-07]
- AC-175: J-2 step 5: when the framework returns terminal, `CompletionToast` shows `Cancelled after <duration> · $<cost>`. [blocked-on: framework-07]

### J-3 — Reconnect after network blip

- AC-176: J-3 step 2: when the SSE connection closes mid-run, the canvas surfaces the `Reconnecting…` annotation. [blocked-on: framework-06]
- AC-177: J-3 step 3: the client retries the SSE connection with exponential backoff (1s → 2s → 4s → max 8s), supplying `Last-Event-ID`. [blocked-on: framework-06]
- AC-178: J-3 step 4: on reconnect, the in-memory replay buffer replays events newer than the supplied `Last-Event-ID`, the token counter catches up to the live value, and the orb state catches up — all within ~500ms. [blocked-on: framework-06]
- AC-179: J-3 step 4: when `Last-Event-ID` is older than the deque window, the SSE stream emits `Custom("marsys.stream.gap")` and the client refreshes its row via REST. [blocked-on: framework-06]

## Non-functional

### Performance

- AC-180: `RunsBroker.publish` is non-blocking (returns within microseconds regardless of subscriber queue state).
- AC-181: SSE reconnect backoff caps at 8s and never exceeds it.
- AC-182: The per-run replay deque caps memory at ~200KB at full capacity (1024 events).

### Accessibility

- AC-183: Every interactive element added in Session 04 (`RunButton`, `CompletionToast` close, filter chips on `/runs`, status badges if interactive) is reachable and operable via keyboard.
- AC-184: `CompletionToast` does not steal focus on appearance (uses `aria-live="polite"`, not `assertive`).
- AC-185: `prefers-reduced-motion: reduce` disables the slide-in animation on `CompletionToast` and the pulse animation on `PulseDot`; the orb's `thinking` / `speaking` states still convey via static color shifts (no looping animation).

### Security

- AC-186: All Session 04 backend endpoints require the per-launch auth token; tests verify a 401 response without it on every endpoint.

### Observability / error handling

- AC-187: A run that fails inside `Orchestra.run()` produces a row with `status="failed"`, `error` populated with the exception message, and a `RunFinishedEvent` (or equivalent failed-finish event) on the aggregate stream.
- AC-188: A `POST /v1/runs/{id}/cancel` watchdog timeout produces a WARN log entry naming the run id. [blocked-on: framework-07]
- AC-189: An unknown `(provider, model)` in cost calculation produces a WARN log entry naming both fields.

## Functional — Pytest coverage

- AC-190: Pytest unit tests cover cost calculation for every provider in `cost_rates.RATES` × every metric (input, output, reasoning).
- AC-191: A Pytest unit test covers the missing-rate edge case (unknown provider+model returns zero + emits WARN).
- AC-192: Pytest unit tests cover the materializer happy path (`WorkflowDefinition` → `RuntimeBundle`).
- AC-193: Pytest unit tests cover `RunsBroker` subscribe / publish / drop-oldest semantics.
- AC-194: Pytest unit tests cover the lifecycle state machine transitions for every status pair (queued→running, running→succeeded, running→failed, queued→cancelled, running→cancelling→cancelled).
- AC-195: A Pytest unit test covers the migration's apply + idempotent reapply.
- AC-196: A Pytest integration test covers `POST /v1/runs` end-to-end with a static-response model adapter (NOT a mock — adheres to SP-007), asserting row insertion, snapshot file write, orchestra construction + wiring before the 201 response, `started` event set, and lifecycle task scheduled.
- AC-197: A Pytest integration test covers SSE end-to-end against a static-response orchestra (subscribe, observe AG-UI event sequence ending with `RunFinished`, verify replay buffer holds events). [blocked-on: framework-06]
- AC-198: A Pytest integration test covers cancel mid-run (`running → cancelling → cancelled` transitions) using `Orchestra.cancel_session()`. [blocked-on: framework-07]
- AC-199: A Pytest integration test covers the cancel watchdog timeout path (framework call exceeds 10s; row marked cancelled regardless; WARN logged). [blocked-on: framework-07]
- AC-200: A Pytest integration test covers cost aggregation against a known-token-count fixture. [blocked-on: framework-06]
- AC-201: A Pytest integration test covers `POST /v1/runs` with `attachments=[<non-empty>]` returning HTTP 400 + `code: "ATTACHMENTS_NOT_YET_SUPPORTED"`.
- AC-202: A Pytest integration test covers `POST /v1/runs` for an archived workflow returning HTTP 400.
- AC-203: A Pytest integration test covers `POST /v1/runs/{id}/cancel` against a `cancelling` or terminal run returning HTTP 409.
- AC-204: Tests gated by `[blocked-on: framework-06]` skip-pass via a runtime check `importlib.util.find_spec("marsys.coordination.aggui") is not None` and light up automatically when the framework dependency merges.
- AC-205: Tests gated by `[blocked-on: framework-07]` skip-pass via a runtime check `hasattr(Orchestra, "cancel_session")` and light up automatically when the framework dependency merges.

## Functional — Vitest coverage

- AC-206: Vitest unit tests cover the SSE consumer hook reducer logic (event sequence in → atom state out). [blocked-on: framework-06]
- AC-207: A Vitest unit test covers the `Last-Event-ID` resume path against a fixture stream. [blocked-on: framework-06]
- AC-208: Vitest unit tests cover the cost calculator (cross-checked against the Pytest unit values).
- AC-209: Vitest unit tests cover the jotai store transitions (activeRunIdAtom, runStateAtom, tokenCountAtom, elapsedMsAtom updates).

## Functional — Playwright E2E + visual regression

- AC-210: A Playwright E2E test exists (e.g., `apps/web/tests/e2e/run-execution.spec.ts`) covering the J-1 golden path: click Run → orb transitions thinking → speaking → idle → completion toast renders with cost. [blocked-on: framework-06]
- AC-211: A Playwright E2E test covers the cancel flow (J-2): click Cancel → confirm → 5-second countdown → run ends `cancelled`. [blocked-on: framework-07]
- AC-212: A Playwright E2E test covers the reconnect flow (J-3): kill SSE mid-run → reconnect → replay catches up via Last-Event-ID. [blocked-on: framework-06]
- AC-213: A Playwright E2E test covers a deliberately failing run producing a `RunError` and the toast rendering `Failed: <reason>`. [blocked-on: framework-06]
- AC-214: A tauri-driver E2E test exists running the J-1 golden path inside the desktop shell.
- AC-215: Visual regression baselines are captured via Playwright `toHaveScreenshot()` (NOT Argos) for: canvas with `RunButton` in `idle`, `submitting`, `running`, `cancel-confirm-open`, and `cancelling` states.
- AC-216: Visual regression baselines are captured for `CompletionToast` in `success`, `failure`, and `cancel` variants.
- AC-217: Visual regression baselines are captured for the `/runs` list page with three rows (one of each terminal status).
- AC-218: Visual regression baselines are captured for the `/runs/{id}` placeholder page in `succeeded` and `failed` variants.
- AC-219: The visual regression suite uses Playwright's `toHaveScreenshot()` API exclusively (no `@argos-ci/*` dependency is added).

## Out of scope

The following are explicitly excluded from Session 04; tests asserting their presence would be wrong for this session:

- File upload + attachment endpoints (`POST /v1/files`, file references in `task_input.attachments`) — Session 05.
- Trace viewer UI (nested span tree, per-span cost chips, expand/collapse) — Session 05.
- Run history filtering UI beyond basic chips (date pickers, status multi-select) — Session 05.
- `GET /v1/runs/{id}/trace` REST endpoint returning a hierarchical trace tree from `trace.ndjson` — Session 05.
- `GET /v1/runs/{id}/artifacts` + artifact download — Session 05.
- `GET /v1/runs/{id}/workflow` (frozen snapshot read endpoint) — Session 05.
- Mid-run user-interaction (`POST /v1/runs/{id}/respond` for paused user_interaction prompts) — Session 05 or 06.
- Pause + resume (framework primitives + Spren-side `POST /v1/runs/{id}/pause` / `resume`) — v0.4.
- Cost charts (daily / weekly / per-workflow rollup visualizations) — Session 06 or v0.4.
- Budget cap enforcement (per-day + per-run cost caps; meta-agent refuses over-budget) — Session 08.
- Scheduled / webhook / messenger triggers — v0.4.
- Live cost display on the canvas during a run (Session 04 ships only final cost on completion toast).
- Multi-tab same-run SSE coordination — v0.4 if requested.
- `SprenTelemetrySink` (the `python my_workflow.py → Spren UI` adapter) — v0.4.
- Token-level streaming inside an agent turn (framework currently emits a single `TextMessageContent` per turn).
- Cold-reader replay from `trace.ndjson` for terminal runs (replaced by in-memory replay buffer; trace-viewer cold path is Session 05).
- Run keyboard shortcut details (the implementer picks ⌘R or similar; the criterion is documented presence of the shortcut, not the specific key).

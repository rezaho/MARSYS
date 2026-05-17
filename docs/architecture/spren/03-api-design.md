# 03 — API Design

## Transport rules

Per SP-003, SP-004, SP-019:

- **REST over HTTP** (FastAPI) for all CRUD
- **Server-Sent Events (SSE)** for server→client event streams during a run
- **One POST endpoint** for mid-run user interaction (correlated by question_id, no separate WS)
- **No WebSocket. No gRPC.** Adding either requires explicit design discussion.

All endpoints under `/v1/` (single API version per SP-006).

The API surface is the contract for every Spren client — the desktop GUI (Tauri webview), the browser GUI (system browser tab), the TUI (Textual), and the framework adapter SDK. There is no client-specific endpoint and no UI logic in the backend (SP-019). The API computes; clients render.

## Authentication

- Server binds to `127.0.0.1` only (SP-002)
- A per-launch auth token is generated when the server starts; printed to stdout AND injected into the frontend's bootstrap payload
- Every request must carry `Authorization: Bearer <token>`. Missing or wrong token → 401

The auth token is regenerated on every server restart. There is no login flow.

## Endpoint surface

### Workflows

```
GET    /v1/workflows                       # list, paginated, filterable by provenance / archived / etc.
POST   /v1/workflows                       # create (provenance defaults to `api`; visual builder sets `visual_builder`)
GET    /v1/workflows/{id}                  # read
PUT    /v1/workflows/{id}                  # replace
PATCH  /v1/workflows/{id}                  # partial update (e.g. archive)
DELETE /v1/workflows/{id}                  # hard delete (only if no runs reference it; otherwise PATCH archive)
POST   /v1/workflows/{id}/lint             # run topology compile + semantic linter; returns issues
POST   /v1/workflows/{id}/duplicate        # create a copy
POST   /v1/workflows/import-python         # multipart upload of a .py file using the marsys framework; parses agent + topology + config; returns the materialized workflow with provenance=`code_import` and provenance_metadata containing source filename + sha256
```

### Runs

```
POST   /v1/runs                            # body: {workflow_id, task_input: {text, attachments}}; returns {run_id, status: "queued"}
GET    /v1/runs                            # list, paginated, filterable by workflow_id/status/date
GET    /v1/runs/{id}                       # row from `runs` table
GET    /v1/runs/{id}/events                # SSE stream of AG-UI events for this run
POST   /v1/runs/{id}/respond               # body: {question_id, answer}; resolves a paused user_interaction
POST   /v1/runs/{id}/cancel                # cancel a running execution (one-way; no resume)
POST   /v1/runs/{id}/pause                 # (v0.4) pause a running execution; status → paused; state preserved on disk
POST   /v1/runs/{id}/resume                # (v0.4) resume a paused execution; status → running; continues from snapshot
GET    /v1/runs/{id}/trace                 # full trace tree (reads trace.ndjson, returns hierarchical JSON)
GET    /v1/runs/{id}/workflow              # the frozen workflow.json snapshot
GET    /v1/runs/{id}/artifacts             # list artifacts produced by the run
GET    /v1/runs/{id}/artifacts/{name}      # download an artifact
```

### Files

```
POST   /v1/files                           # multipart upload; returns {file_id, mime_type, size_bytes, sha256}
GET    /v1/files/{id}                      # metadata
GET    /v1/files/{id}/download             # raw bytes
DELETE /v1/files/{id}                      # delete (only if not referenced by any run)
```

### Settings + secrets

```
GET    /v1/settings                        # all settings
PUT    /v1/settings/{key}                  # update one
GET    /v1/secrets                         # list key names (NEVER values)
PUT    /v1/secrets/{key_name}              # body: {value}; stored per 07-security.md
DELETE /v1/secrets/{key_name}              # delete
```

### Meta-agent (v0.3: read-only chat)

```
POST   /v1/meta/messages                   # body: {message, conversation_id?}; returns {message_id, conversation_id}
GET    /v1/meta/conversations/{id}/events  # SSE stream of meta-agent's AG-UI events for this turn
GET    /v1/meta/conversations              # list past conversations
GET    /v1/meta/conversations/{id}         # one conversation history
```

### Health and bootstrap

```
GET    /healthz                            # 200 OK + {version, started_at}; NO auth required (used by Tauri shell + Docker healthchecks)
GET    /v1/bootstrap                       # auth required. Returns the capability response that drives client conditional rendering:
                                           #   { framework: {version}, spren: {active, version} | null,
                                           #     surfaces: ["gui", "tui"], capabilities: {...feature flags...},
                                           #     endpoints: {workflows: "/v1/workflows", runs: "/v1/runs", ...} }
                                           # Every client (GUI in Tauri, GUI in browser, TUI) calls this
                                           # on connect and gates feature-conditional rendering.
```

## SSE event format

All `/events` SSE streams emit lines in this format:

```
event: <ag-ui-event-type>
data: {<ag-ui-event-payload-json>}
id: <event-id>

```

Event types map 1:1 to AG-UI:
- `RunStarted` / `RunFinished` (with `interrupt` reason if mid-run pause)
- `StepStarted` / `StepFinished`
- `TextMessageStart` / `TextMessageContent` / `TextMessageEnd` (token streaming)
- `ToolCallStart` / `ToolCallArgs` / `ToolCallEnd`
- `ReasoningStart` / `ReasoningContent` / `ReasoningEnd` (for thinking-capable models)
- `StateSnapshot` / `StateDelta`
- `Custom` (for events that don't fit standard types — used sparingly, see SP-004)

The translation layer ships in the framework at `marsys.transport.aggui`. It subscribes to MARSYS `EventBus` and emits AG-UI events as `AsyncIterator[AGUIEvent]` from `AGUIEventStream(orchestra, run_id)`. Spren wraps the iterator in an SSE HTTP endpoint at `/v1/runs/{id}/events`.

## OpenAPI, TypeScript, and Python type generation

FastAPI emits `/openapi.json` automatically. The same Pydantic models are reused across every client (SP-019):

1. `apps/web` build script generates `/openapi.json` via a transient sidecar process at build time (NOT committed to git) and runs `openapi-typescript` to generate `apps/web/src/lib/api-types.generated.ts`. Every Pydantic model the frontend consumes reaches OpenAPI through a route handler — there is no second emitter. Models that don't surface via REST are not part of the frontend's contract.
2. The Spren API client (`apps/web/src/lib/api.ts`) imports those types.
3. The TUI (`apps/tui/`) imports the same Pydantic models directly from `packages/spren/src/spren/models/`. No code generation needed — same Python.
4. The framework adapter (`spren.telemetry`, in `packages/spren/src/spren/telemetry/`) imports the same Pydantic models for its API client.

For shapes shared with the trace stream (which AG-UI provides), use the `@ag-ui/core` and `@ag-ui/client` npm packages on the JS side and `ag-ui-protocol` on the Python side directly — no need to regenerate.

> **AG-UI version caveat:** the AG-UI packages remain pre-1.0. Pin tightly (`==` for Python, exact version for npm) and treat as experimental. Expect breaking changes between minor releases. We adopt AG-UI for its event taxonomy, not for stability — if it churns mid-release, fall back to a hand-rolled SSE schema modeled after AG-UI's events.

## Error response format

All non-2xx responses follow:

```json
{
  "error": {
    "code": "WORKFLOW_NOT_FOUND",
    "message": "No workflow with id 01J...",
    "details": {...}
  }
}
```

Codes are string enums defined in Pydantic (`spren.models.errors.ErrorCode`); the generated TypeScript types in `apps/web/src/lib/api-types.generated.ts` mirror them. Clients render error UI against the generated string union.

## Pagination

Cursor-based, not page-based. List responses include:

```json
{
  "items": [...],
  "next_cursor": "opaque-string-or-null",
  "has_more": true
}
```

Listing accepts `?cursor=...&limit=N` (max limit 100). The cursor is the opaque string-form ULID of the last returned row; the server uses it as `WHERE id > :cursor ORDER BY id` to fetch the next page. ULIDs are k-sortable monotonic, so this orders correctly and survives daemon restart.

## Idempotency

Mutating endpoints accept an optional `Idempotency-Key` header. If the same key is replayed within 24 hours, the response is replayed from a cache stored in the `_idempotency` table inside `<data-dir>/data/spren.db`. The cache key is `(method, path, idempotency_key)`; mismatched method/path with the same key is treated as a fresh request. Expired rows are swept on startup. Required when posting from the meta-agent (so a retry doesn't double-create).

## Cancellation semantics

`POST /v1/runs/{id}/cancel`:
- If `queued`: marks `cancelled` immediately
- If `running`: signals the marsys orchestra to abort; orchestra runs cleanup; status moves to `cancelled` once cleanup completes
- If `paused`: calls `Orchestra.discard_paused_session(run_id)` immediately, deleting the snapshot from the storage backend; status moves to `cancelled`. The `runs` row stays as run-history for inspection
- If terminal: returns 409

## Pause / resume semantics (v0.4)

`POST /v1/runs/{id}/pause`:
- If `running`: signals `Orchestra.pause_session(run_id)`; orchestra cleanly halts after in-flight tool calls complete (with timeout); the framework writes the state snapshot atomically (write-temp + fsync + rename + parent-dir fsync) to `<data-dir>/data/runs/{run_id}/snapshot.json` via the configured `FileStorageBackend(root=<data-dir>/data/runs)`; status moves to `paused`
- If `queued`: marks `paused` without spawning the run; resumes at the front of the queue when resumed
- If `paused` or terminal: returns 409

`POST /v1/runs/{id}/resume`:
- If `paused`: reads the snapshot via `Orchestra.resume_session(run_id)`; status moves to `running`; events resume on the existing `/v1/runs/{id}/events` SSE stream from where they left off (subscribers re-fetch + tail-follow per the standard reconnection pattern)
- If `paused` AND the snapshot's `framework_version` does not match the running framework: the framework raises `IncompatibleSnapshotError`; the endpoint catches it and returns 409 with `{"error": {"code": "INCOMPATIBLE_SNAPSHOT", "message": "snapshot was created on framework v{X}; running v{Y}", "details": {"snapshot_version": "X", "running_version": "Y"}}}`. The inspector renders the explicit error and offers the discard affordance
- If any other state: returns 409

`GET /v1/runs?status=paused`:
- Hydrated at daemon startup from `Orchestra.list_paused_sessions() -> list[PausedSessionMetadata]` so paused runs survive across `spren launch` cycles. Metadata-only enumeration; full snapshots are not loaded eagerly. The list endpoint stays the canonical surface — no separate `GET /v1/runs/paused`.

**At-least-once tool-call contract on resume**: pause awaits the in-flight branch tick, but a tool call that was about to start when pause arrived re-runs on resume. Spren's user-visible help documents this; the workflow editor surfaces a warning at definition time when a workflow uses tools whose framework registry entry is non-idempotent.

**Missed schedules during pause**: barrier timeouts and other scheduled events whose `fire_at` was crossed during the paused interval are skip-and-logged by the framework on resume; the run inspector renders these as "missed during pause" entries in the trace view. The run continues without spurious timeout-fail cascades.

**Snapshot retention**: the framework's periodic sweeper (invoked in `Orchestra.__init__`) deletes snapshots older than 30 days; Spren's `runs` row stays as run-history for inspection. Users can discard a paused run early via `POST /v1/runs/{id}/cancel` (which calls `discard_paused_session` immediately).

Paused runs survive daemon restart. On daemon startup, runs in `paused` status remain paused until the user or meta-agent explicitly calls `resume`. The framework primitive (`Orchestra.pause_session()`, `Orchestra.resume_session()`, `Orchestra.list_paused_sessions()`, `Orchestra.discard_paused_session()`) lands in framework v0.4 support — see [`docs/implementation/framework/sessions/v0.3.0/03-pause-resume-completion.md`](../../implementation/framework/sessions/v0.3.0/03-pause-resume-completion.md).

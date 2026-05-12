# Spren Session 05 — Run Inspection (Trace Viewer + File Uploads + Run History UI)

> Session plan. The implementer reads this as the primary source of truth for what Session 05 ships, how the trace viewer + file uploads + run history filters integrate with Session 04's runs surface, and what's in vs out of scope. Captures bundle position, scope boundaries, dependency check, files-to-CREATE / MODIFY / DELETE, the user journeys that close out Bundle 02's demo gate, wireframes for the new surfaces, the locked decisions from the architect-stage Q&A, polish items the implementer addresses in-session, success criteria, and open research items the implementer resolves in-flight.
>
> Status: **draft — subject to user redirect**. Acceptance criteria are frozen separately at [`./05-run-inspection/acceptance.md`](./05-run-inspection/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Design language anchor: [`../../01-visual-builder/sessions/03-visual-builder.md`](../../01-visual-builder/sessions/03-visual-builder.md) §9 Design System. Session 05 reuses Session 03's palette, typography, components, motion tokens, and Spren orb without redefining them. New surfaces added in Session 05 (the full trace viewer at `/runs/{id}`, the right-drawer span detail panel, the canvas-side `📎` attach affordance, the date-range + multi-status filter rail on `/runs`) follow the same design language. Bundle 02 closes its demo gate with this session.

---

## 1. Bundle position + tier

- **Bundle**: 02 — Run Execution + Inspection (Sessions 04 + 05). Session 05 closes the bundle by shipping the full `/runs/{id}` inspector + file upload (backend + canvas attach affordance) + run history filtering UI on `/runs`. Session 04 ships the live run experience; Session 05 ships what the user does *after* a run finishes (or while it's running, viewing the same trace tree live).
- **Bundle demo gate (closed by this session)**: user has a workflow saved from Bundle 01. From the canvas, the user clicks `📎`, attaches a PDF, clicks `Run`, watches the Spren orb stream tokens, sees the completion toast, clicks through to `/runs/{id}` and sees the full nested-span trace tree with per-span timing + cost + status chips. Clicks any span → side drawer shows full attributes (model, tokens, prompt finish reason, tool args, etc.). Views the attached PDF inline-listed under the run's metadata. Navigates back to `/runs`, filters to last 7 days + status=failed, inspects a failed run's error context.
- **Tier**: HIGH-tier with implementer-side research. Two new surface areas: (a) the trace tree component (recursive hierarchical render, virtualized at depth + breadth, click-to-detail) and (b) the file upload pipeline (multipart upload with progress, server-side size cap, sha256 + storage path resolution, deletion guard). Larger UI surface than Session 04; smaller backend surface than Session 04 (most endpoints are read-only).
- **Approval gate**: implementer-side; surfaces to user only if a decision in §13 needs the user's input or a polish item in §10 reveals a hidden architectural ambiguity.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 01 (foundation) | shipped | FastAPI sidecar, auth, capabilities. |
| Spren Session 02 (CRUD + types) | shipped | Pydantic types + `/v1/workflows` REST + the TS type generation pipeline Session 05 reuses for the new `RunTrace` + `FileMetadata` shapes. |
| Spren Session 03 (visual builder) | shipped | Provides the canvas + Run button (Session 04 wired live). Session 05 adds the `📎` attach affordance next to the Run button. |
| Spren Session 04 (run execution + AG-UI streaming + cost) | shipped before this starts | Provides `runs` table + `<data-dir>/data/runs/{run_id}/` directory + `trace.ndjson` reads + per-run aggregates (cost, tokens, duration) + the `/runs` list page Session 05 enhances with filters + the `/runs/{id}` thin placeholder Session 05 replaces. The `attachments` array on `POST /v1/runs` was schema-accepted but value-rejected in Session 04 (`code: "ATTACHMENTS_NOT_YET_SUPPORTED"`); Session 05 enables non-empty values. |
| Framework Session 01 (NDJSON streaming tracing writer) | shipped | `<data-dir>/data/runs/{run_id}/trace.ndjson` is the on-disk source of truth Session 05's trace endpoint reads. The line format + the two non-span diagnostic line types (`stream_event`, `stream_completed`) + the crashed-run indicator (missing terminal marker) are framework contract; Spren reads them as-is. |
| Framework Session 02 (`TelemetrySink` + `SecretRedactor`) | shipped | The redactor runs at the framework's fan-out boundary, so spans in `trace.ndjson` already have secrets scrubbed. Session 05's trace viewer renders the redacted spans without re-redacting. |
| Framework Session 06 (AG-UI translator) | shipped before Session 04 starts | Spren Session 05's `/runs/{id}` view also subscribes to Session 04's SSE consumer for *live runs* (the trace tree builds incrementally as spans close during a still-running run). Same `AGUIEventStream` wrapping; nothing new from the framework side. |

Session 05 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). All work lands inside `apps/web/` (most of the surface) and `packages/spren/` server-side routes. The `files` table is a Spren-only addition; no framework awareness of it.

## 3. What ships in Session 05

### Backend surfaces

- **`GET /v1/runs/{id}/trace`** — at `packages/spren/src/spren/routes/runs.py` (extending Session 04's file). Reads `<data-dir>/data/runs/{run_id}/trace.ndjson` line by line, filters out the two non-span diagnostic line types (`stream_event` for queue-overflow notices, `stream_completed` for the terminal marker), parses each closed span, builds a hierarchical tree by `parent_span_id`, returns `RunTrace` JSON. Streamable but ships as a single JSON document in v0.3 (traces are ≤10MB per data-model.md retention spec; bound JSON is fine). Includes `completion_status: "completed" | "crashed"` derived from terminal-marker presence + `total_spans` + `disk_error_count` from the marker's attributes. For runs still `running`, returns the tree-up-to-now (no `stream_completed` marker yet → `completion_status: "in_progress"`); the client is expected to subscribe to Session 04's `/v1/runs/{id}/events` SSE for live updates as new spans close.
- **`GET /v1/runs/{id}/workflow`** — returns the contents of `<data-dir>/data/runs/{run_id}/workflow.json` as a `WorkflowDefinition` (Pydantic). This is the frozen-at-run-start snapshot per SP-009; differs from `/v1/workflows/{workflow_id}` which returns the *current* (possibly-edited-since) definition. The trace viewer panel uses this to show "the workflow that actually ran" alongside the live edit.
- **`GET /v1/runs/{id}/artifacts`** — lists files in `<data-dir>/data/runs/{run_id}/artifacts/` if that directory exists; returns `{"items": list[ArtifactInfo]}` where `ArtifactInfo = {"name": str, "size_bytes": int, "mime_type": str, "created_at": str}` (ISO 8601). Returns `{"items": []}` for runs without artifacts (the common case in v0.3 — no framework tool writes artifacts yet; structured-artifact support lands in v0.4).
- **`GET /v1/runs/{id}/artifacts/{name}`** — token-protected file read. Path is constrained to `<data-dir>/data/runs/{run_id}/artifacts/{name}` with no path traversal (filename validated as `[^/\\]+$`). Returns the bytes with `Content-Type: <detected mime>` + `Content-Disposition: attachment; filename="<name>"`.
- **`POST /v1/files`** — at `packages/spren/src/spren/routes/files.py` (new file). Multipart upload (FastAPI `UploadFile`). Streams body to `<data-dir>/data/files/{file_id}/<original_filename>` (write-temp + atomic rename) while computing sha256. Enforces per-file size cap (default 100MB; setting `files.max_per_file_mb`) and aggregate cap (default 5GB total; setting `files.max_total_gb`). On success, inserts a `files` row + returns `FileUploadResponse = {file_id, original_name, mime_type, size_bytes, sha256}`. On cap exceeded: 413 with `code: "FILE_TOO_LARGE"` or `code: "STORAGE_CAP_EXCEEDED"`.
- **`GET /v1/files/{id}`** — metadata only (the row, minus the on-disk path). 404 if not found.
- **`GET /v1/files/{id}/download`** — raw bytes with `Content-Disposition: attachment; filename="<original_name>"`. Same path-confinement check as artifacts.
- **`DELETE /v1/files/{id}`** — soft-rejects if the file_id appears in any `runs.task_input.attachments` array (409 with `code: "FILE_REFERENCED_BY_RUNS", details: {run_ids: [...]}`); else hard-deletes the file row + the on-disk bytes. (Future workflow that "archives the run" would discard the run row + free the file reference; out of scope for v0.3.)
- **`POST /v1/runs` attachments resolution** — Session 04 schema-accepted `task_input.attachments: list[str]` but rejected non-empty values. Session 05 enables: each `file_id` in the array is resolved to its `(original_name, path, mime_type, size_bytes)`; the run lifecycle coordinator (Session 04's `packages/spren/src/spren/runs/lifecycle.py`) appends a system-context block to the `task` string before passing to `Orchestra.run()`:

  ```
  <task_text>

  Files attached to this run:
  - report.pdf (/Users/.../data/files/{id}/report.pdf, application/pdf, 1.2 MB)
  - data.csv (/Users/.../data/files/{id}/data.csv, text/csv, 87 KB)

  Use the `read_file` tool to access them.
  ```

  Unknown `file_id` → 400 with `code: "ATTACHMENT_NOT_FOUND", details: {file_id}`. The run row stores the attachments verbatim (the array of file_ids); no archived state required.
- **`GET /v1/runs` filter extensions** — Session 04 shipped `workflow_id`, `status` (single), and `since` (relative shorthand). Session 05 extends:
  - `?status=running,succeeded` — comma-separated multi-value (Session 04 was single-value; Session 05 enables OR matching on the canonical status enum).
  - `?since=<iso8601>` — absolute timestamp variant alongside the relative shorthand (`?since=24h` from Session 04 still works; `?since=2026-05-01T00:00:00Z` is the new addition).
  - `?until=<iso8601>` — paired upper bound for date-range filters.
  - `?workflow_id=<id1>,<id2>` — multi-value variant of Session 04's single-workflow filter, for users filtering by a small set of frequently-run workflows.

  All existing single-value semantics from Session 04 stay backward-compatible — adding a comma doesn't break Session 04 callers.

### Frontend surfaces

- **`/runs/{id}` full inspector (replaces Session 04's thin placeholder)** at `apps/web/src/routes/runs/$runId.tsx`. The placeholder block (the `<trace-viewer status="coming_in_session_05" />` empty state + the "trace viewer ships in Session 05" copy) is deleted in this session and replaced by the full inspector. The metadata header (status badge, duration, cost, token counts, start/finish times) stays — Session 04 already shipped that part. Below it: the trace tree + the right-drawer span detail panel + an attached-files inline section + a workflow snapshot accordion + an artifacts list accordion.
- **Trace tree component** at `apps/web/src/components/TraceTree/TraceTree.tsx`. Recursive render of `RunTrace.spans` (hierarchical). Each span row: caret (▶/▼) for expand/collapse, kind chip (`generation`, `tool`, `step`, `branch`, `execution`), span name, right-aligned timing chip (Geist Mono 11px in `--ink-soft`), right-aligned cost chip (`$0.012` in Geist Mono 11px), status indicator (`✓` for ok, `✗` in `--magenta-deep` for error). 16px indentation per depth level with a vertical line connecting parent→child. Click row → opens the span detail drawer.
- **Span detail panel** at `apps/web/src/components/TraceTree/SpanDetailPanel.tsx`. Right-drawer (320px wide, slides in from the right of the trace section, NOT a full-page modal). Renders the clicked span's kind, name, timing, status, full attributes (kind-specific layout per [`../../../../../architecture/spren/06-observability.md`](../../../../../architecture/spren/06-observability.md) §Attributes by kind), full `events` array (validation decisions, user-interaction prompts), and `links`. For `generation` spans: model, provider, prompt/completion/reasoning tokens, response_time_ms, finish_reason, has_thinking + has_tool_calls flags. For `tool` spans: tool name, agent name, arguments (`[REDACTED]` if redactor scrubbed), result_summary. Long fields collapse to "Show full" expand buttons. Closes on Esc or click-outside.
- **Canvas attach affordance** at `apps/web/src/components/FileAttachInput.tsx`. A small `📎` icon button placed in the canvas top toolbar to the left of the Run button (Session 04 ships Run; Session 05 inserts the paperclip before it). The icon shows a small count badge when files are attached (`📎(2)`). Click → native file picker (multi-select). Drag-and-drop on the canvas anywhere also picks files up as attachments. Each file uploads to `POST /v1/files` with progress; on success, the file_id is appended to a Jotai canvas-attachments atom (`apps/web/src/stores/canvasAttachments.ts`). The Run button's `POST /v1/runs` request reads this atom for `task_input.attachments`. On run completion (success or cancel), the atom resets. A small attachments list pops out below the `📎` icon on click — shows filename + size + an `×` to remove. Drag-and-drop UX: a translucent overlay appears across the canvas when a file is being dragged over; drop = upload.
- **Attachment list (read-only on `/runs/{id}`)** at `apps/web/src/components/AttachmentList.tsx`. Renders the run's attached files: filename, size, a `Download` link (calls `GET /v1/files/{id}/download`). Empty state when no attachments: `<attachments status="none" />` in the tag-markup typographic device.
- **Run history filter rail on `/runs`** at `apps/web/src/components/RunHistoryFilters.tsx`. Replaces Session 04's single-select chip row. Three filters combined:
  - **Date range filter** — relative pill bar (`Today` / `Yesterday` / `Last 7 days` / `Last 30 days` / `Custom...`); `Custom` opens a small two-input dialog (start date, end date) using Geist Mono date inputs (no third-party date picker library — native `<input type="date">` with shadcn styling). Selection updates URL search params (`?since=...&until=...`).
  - **Status multi-select** — five pills (`Running` / `Queued` / `Succeeded` / `Failed` / `Cancelled`), each independently toggleable. Multi-select fires a comma-separated `?status=` param. Default: all selected (no filter).
  - **Workflow filter** — dropdown listing all non-archived workflows by name + provenance badge. Selection fires `?workflow_id=<id>`. Multi-select via a small inline-multi-select pattern (shadcn `Command`-style with checkboxes).
- **`/runs/{id}` workflow snapshot accordion** — a collapsed section ("Workflow as run") that, when expanded, renders a small read-only canvas (xyflow with `interactionsDisabled`) using the frozen `WorkflowDefinition` from `GET /v1/runs/{id}/workflow`. This lets the user inspect what topology actually ran, separate from the workflow's current (possibly-edited-since) state. Side button "Open in canvas" navigates to `/workflows/{workflow_id}` showing the live workflow.
- **Re-run affordance** — a `Re-run` button next to the metadata header on `/runs/{id}`. Click → fires `POST /v1/runs` with the same `workflow_id` + `task_input` (text + attachments copied verbatim from the original run); navigates to the new `/runs/{new_id}` which is now-running. The same file_ids are referenced; no re-upload needed.

### Tests

- **Vitest unit**: trace tree builder (parent-by-parent_span_id wiring + orphan handling + crashed-run terminal-marker absence detection); span detail panel renders correct attributes per `kind`; file upload state machine (idle → uploading → uploaded → run-submitted → reset); date-range filter param serialization; status multi-select combine logic.
- **Pytest integration**: `GET /v1/runs/{id}/trace` against a synthetic trace.ndjson fixture (10 spans, 3 levels deep, one with `kind="stream_event"` diagnostic + one terminal `stream_completed`) — verify tree shape, filtered diagnostics, `completion_status` correctness. Same fixture with the terminal marker missing → `completion_status: "crashed"`. `POST /v1/files` round-trip (upload + metadata read + download + sha256 round-trip + cap rejection at 101MB). `DELETE /v1/files/{id}` reference check (attaches file to a run, then attempts delete → 409 with run_ids). `POST /v1/runs` with `task_input.attachments=[id]` → run lifecycle stores attachments; task string passed to `Orchestra.run()` contains the file path + filename + mime type + size. Run-history filter combinations (date range + multi-status + workflow filter — verify SQL composes correctly + result set matches).
- **Playwright E2E (browser + Tauri)**: user clicks `📎` on the canvas, picks a PDF from disk, sees the count update to `📎(1)`. Clicks Run. Watches the Spren orb stream. Run completes. Clicks the completion toast → lands on `/runs/{id}`. Sees the trace tree with at least 5 spans (3 generations, 1 tool, 1 step). Clicks the first generation span → drawer opens with model name + token counts. Closes drawer. Expands the "Workflow as run" accordion → sees the frozen topology rendered. Navigates to `/runs`. Sets date range to "Last 7 days", status to "Failed". Sees only failed runs from the last week. Clicks one → trace viewer shows the failed span with `✗` in `--magenta-deep`.
- **Tauri-driver E2E**: same flow in the Tauri webview, with file-picker invocation through the OS dialog (per platform; macOS/Linux/Windows runs separately; WSL2 inconclusive).
- **Visual regression baselines (Bundle 02 extension of Bundle 01's baselines)**: `/runs/{id}` full inspector with 3 spans, with 30 spans (testing virtualization), the span detail drawer open, the attachments inline section with 2 attached files, the workflow snapshot accordion expanded, the failed-run inspector, the `/runs` filter rail (default state, all-status-deselected, custom-date-range dialog open), the canvas with `📎(3)` count + the attachment list popout, the canvas with the drag-and-drop overlay visible.
- **Manual-verify checklist** (implementer self-verification before claiming done), covering the trace tree's keyboard navigation + the reduced-motion fallback + the file upload progress UX + the `Re-run` from a completed run → new run lands with same attachments.

## 4. What is OUT of scope

| Out of scope in Session 05 | Lands in |
|---|---|
| Task-input dialog (asking "What should the workflow work on?" before Run, with a text field + an attach affordance bound to the dialog instead of the canvas top toolbar) | Session 06 (meta-agent integration owns the task-input experience; the dialog absorbs / coexists with the Session 05 `📎` icon, depending on Session 06's design call). The empty-text default from Session 04 §8.6 still applies as the canvas-Run shortcut. |
| Trace search (find spans by name/kind across a single run) | v0.4. The trace tree is small enough in v0.3 (≤10MB / few hundred spans) that visual scanning + expand/collapse is sufficient. |
| Cross-run trace search (find runs that touched span X) | v0.4. SQLite FTS5 backs it; out of scope for v0.3. |
| Cost charts (rollup graphs: daily / weekly / per-workflow stacked bars) | Session 06 or v0.4. Session 05 ships the data; the chart layer is its own surface. |
| Mid-run user-interaction (`POST /v1/runs/{id}/respond` for paused user_interaction prompts) | Session 06 (the meta-agent's first user-interaction surface). |
| Pause / resume (framework primitives + Spren `POST /v1/runs/{id}/pause` / `resume`) | v0.4 (`v0.4-29`). Session 05 does not block on this. |
| Live-tail of an in-progress run's trace tree as new spans close (so the tree rebuilds incrementally as `RUN_FINISHED` is approached) | Session 05 ships a basic version: the tree polls `GET /v1/runs/{id}/trace` every 2 s while `status='running'`, replacing the tree wholesale on each poll. The richer streaming-tree approach (subscribe to Session 04's SSE consumer, apply per-event-delta to the tree) is deferred to v0.4 polish — the v0.3 polling approach is acceptable for a single user / handful of concurrent runs. |
| Inline preview of attached files (PDF inline viewer, image thumbnails, text-file syntax highlighting) | v0.4. v0.3 ships download links + metadata only. |
| Inline preview of artifact outputs | v0.4 (depends on the v0.4 artifact tool taxonomy). |
| File library / "all my uploaded files" page (independent of any specific run) | v0.4 if requested. v0.3 files are per-run-attached. The `DELETE /v1/files/{id}` endpoint is exposed for v0.3 but no UI surface lists global files. |
| File deduplication on upload (sha256-based "you've uploaded this before; reusing") | v0.4. v0.3 every upload creates a new row. |
| Bulk operations on runs (delete multiple, archive multiple) | v0.4. |
| Run notes / annotations (user-added comments on a run) | Session 08's meta-agent `add_run_note` write tool ships this. |
| OpenTelemetry / external tracing export (export trace.ndjson to LangSmith / Phoenix / etc.) | v0.4. `SprenTelemetrySink` lands in v0.4-27 for the `python my_workflow.py → Spren UI` direction. External-export of Spren-dispatched runs follows the `TelemetrySink` Protocol if a user wants to wire one in. |

Anything labeled out-of-scope renders as "not available" or empty placeholder in Session 05's routes (capability-gated per SP-019 where the route exists).

## 5. Files to CREATE / MODIFY / DELETE in Session 05

### To CREATE

| Path | Purpose |
|---|---|
| `packages/spren/src/spren/routes/files.py` | REST endpoint handlers (POST/GET/list/download/delete). Route-level auth + CORS regex per Session 01's pattern. |
| `packages/spren/src/spren/files/__init__.py` | Module init. |
| `packages/spren/src/spren/files/upload.py` | Multipart upload handler. Streams body to `<data-dir>/data/files/{file_id}/<original_name>` (write-temp + atomic rename + fsync). Computes sha256 during stream. Enforces per-file + aggregate caps. Returns the row. |
| `packages/spren/src/spren/files/lookup.py` | Resolves `file_id → (row, on-disk path)`. Used by `GET /v1/files/{id}/download`, by the run lifecycle coordinator's attachments resolution, and by the `DELETE` reference check. |
| `packages/spren/src/spren/runs/trace.py` | Trace.ndjson parser + hierarchical tree builder. Reads line-by-line, filters `kind in {"stream_event","stream_completed"}` for span-tree purposes but uses them for `completion_status` + `total_spans` derivation; builds tree by `parent_span_id`; returns `RunTrace` Pydantic. |
| `packages/spren/src/spren/runs/artifacts.py` | Lists files in `<data-dir>/data/runs/{run_id}/artifacts/`. Same path-confinement logic as files.py for the per-artifact download endpoint. |
| `packages/spren/src/spren/models/file.py` | Pydantic: `FileMetadata`, `FileUploadResponse`. |
| `packages/spren/src/spren/models/trace.py` | Pydantic: `RunTrace`, `SpanNode` (recursive via `children: list["SpanNode"]`), `RunTraceCompletionStatus` enum. Mirrors the framework's `Span` shape on the wire, adapted to a tree (parent_span_id replaced by children-of nesting). |
| `packages/spren/src/spren/models/artifact.py` | Pydantic: `ArtifactInfo`. |
| `packages/spren/src/spren/storage/migrations/<N>__create_files_table.py` | Forward-only migration. Schema per `02-data-model.md` §files. Indexes on `(created_at)` for sweepers and `(sha256)` for future dedup. |
| `apps/web/src/components/TraceTree/TraceTree.tsx` | Recursive tree component. Renders `RunTrace.spans` array. Keyboard nav (j/k/Enter/Esc). Virtualization-on-need (see §10). |
| `apps/web/src/components/TraceTree/SpanRow.tsx` | Single-row primitive. Used by TraceTree. Renders kind chip, name, timing chip, cost chip, status indicator. |
| `apps/web/src/components/TraceTree/SpanDetailPanel.tsx` | Right-drawer panel for clicked span. Kind-specific attribute layouts. "Show full" expansions. |
| `apps/web/src/components/FileAttachInput.tsx` | The `📎` icon button + popout attachments list + the canvas-wide drag-and-drop overlay logic. |
| `apps/web/src/components/AttachmentList.tsx` | Read-only file list (used on `/runs/{id}`). |
| `apps/web/src/components/RunHistoryFilters.tsx` | Combined filter rail (date range + multi-status + workflow filter). |
| `apps/web/src/components/DateRangeFilter.tsx` | Relative-pill + custom dialog. |
| `apps/web/src/components/StatusMultiSelect.tsx` | Multi-select pill row. |
| `apps/web/src/components/WorkflowFilter.tsx` | Inline-multi-select workflow dropdown. |
| `apps/web/src/components/WorkflowSnapshotAccordion.tsx` | Collapsed-by-default section that, when expanded, renders the frozen `WorkflowDefinition` as a read-only xyflow canvas. |
| `apps/web/src/components/ArtifactsList.tsx` | Artifacts list (with download links; empty-state for v0.3 common case). |
| `apps/web/src/lib/run-trace.ts` | Fetch + parse trace JSON + the 2-second poll loop for live runs. |
| `apps/web/src/lib/files.ts` | File upload client (multipart with progress event support); `DELETE` client. |
| `apps/web/src/stores/canvasAttachments.ts` | Jotai atom for the canvas-side attached files state. Reset on run completion. |
| `apps/web/src/lib/run-rerun.ts` | Helper that copies a run's `task_input` and fires `POST /v1/runs`; navigates. |
| `apps/web/tests/e2e/run-inspection.spec.ts` | Playwright golden-path (J-1 + J-2). |
| `packages/spren/tests/integration/test_files_routes.py` | Pytest integration for files endpoints + cap enforcement + reference-check delete. |
| `packages/spren/tests/integration/test_run_trace.py` | Pytest integration for trace endpoint + crashed-run detection. |
| `packages/spren/tests/integration/test_runs_filters.py` | Pytest integration for the extended `GET /v1/runs` query params. |
| `packages/spren/tests/fixtures/traces/synthetic_3agent_trace.ndjson` | Hand-crafted 10-span fixture (3 generations, 1 tool, 1 step, 1 branch, 1 execution + the terminal marker). |
| `packages/spren/tests/fixtures/traces/crashed_trace.ndjson` | Same shape minus the terminal marker. |

### To MODIFY (in-place edits to Session 04's files)

| Path | Edit |
|---|---|
| `apps/web/src/routes/runs/$runId.tsx` | Replace the `<trace-viewer status="coming_in_session_05" />` placeholder block with the full inspector (TraceTree + SpanDetailPanel + AttachmentList + WorkflowSnapshotAccordion + ArtifactsList + Re-run button). The metadata header (status badge + duration + cost + tokens + timestamps) stays unchanged. |
| `apps/web/src/routes/runs/index.tsx` | Replace Session 04's single-select chip row with `<RunHistoryFilters />` (the new combined filter rail). Update URL search-param handling to read `since`, `until`, `status` (multi), `workflow_id` (multi). |
| `packages/spren/src/spren/routes/runs.py` | Extend the `POST /v1/runs` handler to resolve non-empty `task_input.attachments` (was 400-rejected in Session 04). Extend the `GET /v1/runs` handler to accept the new query params. Add the three new endpoints (`/trace`, `/workflow`, `/artifacts`, `/artifacts/{name}`). |
| `packages/spren/src/spren/runs/lifecycle.py` | Add the attachments-to-task system-context block (per §3 above). |
| `apps/web/src/components/RunButton.tsx` (Session 04) | Add a sibling `<FileAttachInput />` to the left of the Run button. (Single-line addition; the Run button itself doesn't change.) |
| `packages/spren/src/spren/server.py` | Mount the new files router. |

### To DELETE

None at the file level. Session 05 is purely additive on the backend (new routes + new tables + new modules) and replaces UI content in-place on the two pre-existing routes. No file rename or deletion.

The placeholder block inside `apps/web/src/routes/runs/$runId.tsx` (the `<trace-viewer status="coming_in_session_05" />` tag-markup empty state from Session 04 §7 W-D) is deleted in-place when the full inspector lands.

## 6. User journeys (anchor for Bundle 02 demo gate)

Three journeys for Session 05. J-4 + J-5 + J-6 extend Bundle 02's demo gate from where Session 04's J-1/J-2/J-3 ended.

### J-4 — Inspect a completed run

State: user has a workflow from Bundle 01 that ran successfully via Session 04. The completion toast appeared; user clicked through to `/runs/{id}`.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User lands on `/runs/{id}` after clicking the completion toast. | `/runs/{id}` | Metadata header renders (status badge `succeeded`, duration `12.3s`, cost `$0.026`, tokens `8,432 in / 1,221 out`, timestamps). Below it, the trace tree renders fully expanded by default. |
| 2 | User sees the trace: `execution > branch > step #1 Researcher > generation`, etc. Click on the `generation` span under Researcher. | Trace tree | Row highlights. Span detail drawer slides in from the right. |
| 3 | Drawer renders the generation's attributes: model `anthropic/claude-opus-4-5`, tokens `3,201 in / 542 out`, reasoning tokens `120`, response time `4.5s`, finish reason `stop`, has_thinking `yes`, has_tool_calls `no`. | Span detail drawer | All fields visible. "Show full prompt" + "Show full response" buttons collapsed (full content not stored by default per `TracingConfig.include_message_content=false`). |
| 4 | User presses Esc. | Span detail drawer | Closes. Tree row loses highlight. |
| 5 | User clicks the `tool` span under Researcher (`search_web`). | Trace tree | Row highlights. Drawer slides in. |
| 6 | Drawer renders the tool span's attributes: tool name `search_web`, agent name `Researcher`, arguments `{"query": "[REDACTED]"}` (the redactor stripped a token from the args), result_summary `5 results returned`. | Span detail drawer | All fields visible. The `[REDACTED]` value has a small `?` next to it explaining the SecretRedactor ran. |
| 7 | User scrolls down on the inspector and expands the "Workflow as run" accordion. | Workflow snapshot accordion | A read-only canvas renders the 3-agent topology that was frozen at run start. Side button "Open in canvas" navigates to `/workflows/{id}`. |
| 8 | User clicks the `Re-run` button at the top right. | Re-run | `POST /v1/runs` fires with the same `workflow_id` + `task_input.text` + `task_input.attachments=[]`. Navigates to `/runs/{new_id}`. |
| 9 | The new `/runs/{new_id}` page shows status=`running`; the trace tree starts empty + polls every 2s; the orb (top-right presence orb) shifts to `thinking` then `speaking` (via Session 04's SSE wiring on the page). | `/runs/{new_id}` | Tree builds incrementally as spans close on the server. |

### J-5 — Attach a file and run a workflow

State: user has a workflow saved from Bundle 01. User is on the canvas.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User drags a PDF (`report.pdf`, 1.2MB) from the desktop onto the canvas. | Canvas | A translucent overlay covers the canvas: `Drop to attach (report.pdf)`. |
| 2 | User releases the drop. | Canvas | Overlay disappears. The `📎` icon in the top toolbar updates to `📎(1)`. Below the icon, a small popout appears for 2s: `report.pdf · 1.2 MB attached`. |
| 3 | User clicks `📎(1)`. | Attachment popout | Permanent attachment popout: list of one file with filename, size, `×` to remove. |
| 4 | User adds a second file via the popout's `+ Add file` button. | Popout | Native file picker; user selects `data.csv` (87 KB). On upload completion: `📎(2)`. |
| 5 | User clicks Run. | Run button | `POST /v1/runs` with `task_input: {text: "", attachments: [<id1>, <id2>]}`. Run starts as in Session 04's J-1. Orb shifts to `thinking`. |
| 6 | Backend resolves attachments + appends file paths to the system context. The Researcher agent reads the files via `read_file`. The Writer agent summarizes. | (server-side) | Run proceeds. AG-UI events stream. |
| 7 | Run completes. Completion toast: `Completed in 18.4s · $0.034`. | Canvas | Orb returns to `idle`. The canvas attachment atom resets to empty (`📎` icon no longer shows count). |
| 8 | User clicks the toast → lands on `/runs/{id}`. | `/runs/{id}` | Inspector renders. Below the metadata: an "Attachments" section showing `report.pdf` and `data.csv` with `Download` links. Trace tree includes the two `read_file` tool spans alongside the generations. |
| 9 | User clicks `Download` on `report.pdf`. | Browser file download | The PDF downloads. |

### J-6 — Filter run history

State: user has accumulated runs over the past several weeks (some succeeded, some failed, some cancelled, across 4 different workflows).

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User navigates to `/runs` via ⌘K → "runs". | `/runs` | Filter rail at top: date range pills (`Today / Yesterday / Last 7 days / Last 30 days / Custom`), status multi-select (all 5 selected by default), workflow dropdown (defaults to All workflows). |
| 2 | User clicks `Last 7 days`. | URL updates to `?since=<-7d>` | List re-fetches; renders runs from the past 7 days only. |
| 3 | User clicks the status filter, deselects `Running` + `Queued` + `Succeeded`, leaving only `Failed` + `Cancelled` selected. | URL updates with `status=failed,cancelled` | List re-fetches. Three runs visible (all failed or cancelled in the last week). |
| 4 | User clicks the workflow dropdown, types "research" in the search field, selects `research-pipeline`. | URL updates with `workflow_id=<id>` | List re-fetches. One run visible (the only failed research-pipeline run in the last week). |
| 5 | User clicks the run card. | Navigates to `/runs/{id}` | Inspector renders. Status badge `failed`. Below metadata: error message in `--magenta-deep` (per the run's `runs.error` column from Session 04). Trace tree shows the failed span with `✗` indicator. |
| 6 | User clicks the failed span. | Span detail drawer | Renders the span's `status="error"` + the error attributes (e.g., `error: ValidationError`, traceback in a collapsed-by-default expansion). |
| 7 | User navigates back via the breadcrumb. | `/runs` | Filters persist (date range + multi-status + workflow). URL still has the search params. |
| 8 | User clicks `Custom` in the date range row. | Custom date dialog | Two inputs: start date + end date. User picks a 30-day range from a month ago. Apply. List re-fetches with `?since=...&until=...`. |

## 7. Skeleton wireframes (low-fi; ASCII)

Sessions 05 surfaces are larger extensions of Session 03's design language than Session 04 was. Refer to [`../../01-visual-builder/sessions/03-visual-builder.md`](../../01-visual-builder/sessions/03-visual-builder.md) §7 for the established surface chrome (top-bar with `spren.` wordmark + user avatar; presence orb top-right of non-home surfaces; no left nav rail; ⌘K navigation).

### W-A — `/runs/{id}` full inspector

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Runs  ›  research-pipeline · 2 minutes ago                      (R) │
├─────────────────────────────────────────────────────────────────────────────────┤
│   [succeeded]   research-pipeline                                  [Re-run]      │
│   Duration:    12.3s   ·   Cost: $0.026   ·   Tokens: 8,432 in / 1,221 out       │
│   Started:     14:32:08   ·   Finished: 14:32:20                                 │
│                                                                                  │
│   ──────────────────────────────────────────────────────────────────────         │
│                                                                                  │
│   Trace                                                                          │
│   ┌─────────────────────────────────────────────────┐  ┌──────────────────────┐  │
│   │ ▼ execution · research-pipeline   12.3s  $0.026 │  │ Span: generation     │  │
│   │   ▼ branch · linear                12.3s $0.026 │  │ claude-opus-4-5      │  │
│   │     ▼ step #1 · Researcher          4.5s $0.012 │  │                      │  │
│   │       └ generation                  4.5s $0.012 │  │ Model:    anthropic/ │  │
│   │       └ tool · search_web           0.8s        │  │           claude-... │  │
│   │     ▼ step #2 · Writer              3.2s $0.014 │  │ Tokens:   3,201 / 542│  │
│   │       └ generation                  3.2s $0.014 │  │ Reasoning: 120       │  │
│   │     ▼ step #3 · Editor              4.6s $...   │  │ Time:     4.5s       │  │
│   │       └ generation                  4.6s $...   │  │ Finish:   stop       │  │
│   └─────────────────────────────────────────────────┘  │ Thinking: yes        │  │
│                                                        │ Tool calls: no       │  │
│                                                        │                      │  │
│                                                        │ [Show full prompt ▾] │  │
│   ▶ Attachments (2)                                    │ [Show full response] │  │
│   ▶ Workflow as run                                    └──────────────────────┘  │
│   ▶ Artifacts (0)                                                          ◉     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- Metadata header above a horizontal rule (`--rule` 1px line).
- Trace tree on the left; span detail drawer on the right (slides in when a span is clicked). Default state: drawer hidden, trace tree full-width.
- Indentation: 16px per depth level. Vertical line in `--rule` connecting parent → first/last child.
- Span row: caret + kind chip + name + timing chip (right-aligned) + cost chip (right-aligned).
- `✓` for ok, `✗` for error (in `--magenta-deep`). No icon for in-progress.
- Drawer width: 320px on desktop, full-width on narrow viewports.
- Below the trace section: three accordion-style sections (Attachments, Workflow as run, Artifacts). Collapsed by default; counts in parentheses give a quick read.

### W-B — Span detail panel (expanded view)

```
┌────────────────────────────────┐
│ generation                  ✕  │
│ claude-opus-4-5                │
├────────────────────────────────┤
│                                │
│ Model:    anthropic/claude-... │
│ Provider: anthropic            │
│                                │
│ Tokens:                        │
│   Prompt:     3,201            │
│   Completion: 542              │
│   Reasoning:  120              │
│                                │
│ Time:     4,532ms              │
│ Finish:   stop                 │
│ Thinking: yes                  │
│ Tool calls: no                 │
│                                │
│ Cost:     $0.012               │
│                                │
│ ──────────────                 │
│                                │
│ Events: (none)                 │
│ Links:  (none)                 │
│                                │
│ ──────────────                 │
│                                │
│ [Show full prompt ▾]           │
│ [Show full response ▾]         │
│                                │
└────────────────────────────────┘
```

- Header: kind chip (`generation`) + close button (×). Title in Geist 500/14.
- Sections separated by horizontal rules.
- Geist Mono 12px for numeric values; Geist Sans 13px for labels.
- "Show full prompt/response" buttons disabled if `TracingConfig.include_message_content=false` was set (the default in v0.3); tooltip explains why.
- Different `kind`s show different attribute layouts (per the architecture doc's per-kind attribute table); the drawer is one component with kind-specific renderers inside.

### W-C — Canvas with `📎` attach affordance + attachment popout

```
idle (no files):                                  with attached files:
┌──────────────────────────────┐                  ┌──────────────────────────────┐
│ [Lint ✓] [+ Pattern▾]        │                  │ [Lint ✓] [+ Pattern▾]        │
│             [📎] [Run] [Save]│                  │            [📎(2)] [Run] [Save]│
└──────────────────────────────┘                  │  ┌────────────────────────┐  │
                                                  │  │ report.pdf · 1.2 MB  × │  │
during drag-and-drop (file over canvas):          │  │ data.csv · 87 KB     × │  │
┌────────────────────────────────────┐            │  │ [+ Add file]           │  │
│ [Lint ✓] [+ Pattern▾]              │            │  └────────────────────────┘  │
│             [📎] [Run] [Save]      │            └──────────────────────────────┘
│  ┌────────────────────────────┐    │
│  │                            │    │            during upload:
│  │   Drop to attach           │    │            ┌──────────────────────────────┐
│  │   (report.pdf)             │    │            │ [Lint ✓] [+ Pattern▾]        │
│  │                            │    │            │            [📎(2)] [Run] [Save]│
│  └────────────────────────────┘    │            │  ┌────────────────────────┐  │
└────────────────────────────────────┘            │  │ report.pdf · 1.2 MB  × │  │
                                                  │  │ data.csv · uploading…  │  │
                                                  │  └────────────────────────┘  │
                                                  └──────────────────────────────┘
```

- `📎` icon: 32×32px tap target, `--ink-soft` default color, `--magenta` on hover. Count badge appears top-right of the icon when files are attached (Geist Mono 9px, `--magenta` background, white text, 12px circle).
- Popout positioned below the `📎` icon (right-aligned with the canvas toolbar). White surface, 1px `--rule` border, 16px padding, max-width 320px.
- Each row: filename + size, with `×` button right-aligned to remove.
- `+ Add file` button at the bottom: opens native file picker; multi-select allowed.
- Drag-and-drop overlay: `--surface-elevated` with 50% opacity over the canvas; the message in the center, Geist 500/16px.
- During upload: per-file progress as `uploading…` text with a fading dot animation; on success, the size value replaces it. On failure: `× upload failed; retry` in `--magenta-deep`.

### W-D — `/runs` with filter rail

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Runs                                                            (R) │
├─────────────────────────────────────────────────────────────────────────────────┤
│   Runs                                                              12 of 47    │
│                                                                                  │
│   Date:    [Today] [Yesterday] [Last 7 days] [Last 30 days] [Custom...]          │
│   Status:  [● Running] [● Queued] [● Succeeded] [✗ Failed] [○ Cancelled]         │
│   Workflow: [All workflows ▾]                                                    │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────┐               │
│   │ research-pipeline · [failed]                                 │               │
│   │ 4.2s · $0.003 · 3 hours ago · "Researcher returned…"         │               │
│   └──────────────────────────────────────────────────────────────┘               │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────┐               │
│   │ daily-summary · [cancelled]                                  │               │
│   │ 22s · $0.008 · yesterday                                     │               │
│   └──────────────────────────────────────────────────────────────┘               │
│                                                                          ◉       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- Three rows of filters. Date range = relative pills (single-select; `Custom` opens dialog); Status = multi-select pills (each toggleable independently); Workflow = dropdown (single-select for now; multi-select inline in v0.4 if needed).
- Selected pills get `--magenta` background. Unselected get `--surface-elevated` background.
- Workflow dropdown menu shows workflows with provenance badges next to names.
- The card list below filters is unchanged from Session 04 W-C (just narrows under filter pressure). Total count `(12 of 47)` in the top-right.

### W-E — Failed-run inspector

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Runs  ›  research-pipeline · 3h ago                             (R) │
├─────────────────────────────────────────────────────────────────────────────────┤
│   [failed]   research-pipeline                                      [Re-run]    │
│   Duration:  4.2s   ·   Cost: $0.003   ·   Tokens: 821 in / 12 out               │
│                                                                                  │
│   ┌──── Error ───────────────────────────────────────────────────────┐           │
│   │ Researcher returned an invalid JSON response.                    │           │
│   │ Expected: {...result_fields...}                                  │           │
│   │ Got: "I'm having trouble with that request — could you rephrase?"│           │
│   │ [Expand traceback ▾]                                             │           │
│   └──────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
│   Trace                                                                          │
│   ▼ execution · research-pipeline    4.2s   $0.003                               │
│     ▼ branch · linear                 4.2s   $0.003                              │
│       ▼ step #1 · Researcher          4.2s   $0.003                              │
│         └ generation                  3.5s   $0.003                              │
│         └ validation                   ✗    [click to view]                      │
│                                                                                  │
│   ▶ Attachments (0)                                                              │
│   ▶ Workflow as run                                                              │
│   ▶ Artifacts (0)                                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

- Error block in `--surface-elevated` with `--magenta-deep` left-edge accent (4px). Body text in Geist 400/14, error preview in Geist Mono 12.
- Failed span row shows `✗` in `--magenta-deep`. Clicking it opens the drawer with the full error attributes + the validation event details.

### W-F — Attachments inline section on `/runs/{id}`

```
collapsed:                              expanded:
▶ Attachments (2)                       ▼ Attachments (2)
                                          ┌────────────────────────────────┐
                                          │ 📄 report.pdf                  │
                                          │    1.2 MB · application/pdf    │
                                          │    [Download]                  │
                                          │ ────────────                   │
                                          │ 📊 data.csv                    │
                                          │    87 KB · text/csv            │
                                          │    [Download]                  │
                                          └────────────────────────────────┘
```

- Filename in Geist 500/14; metadata row in Geist Mono 11px `--ink-soft`. Download button in `--ink-soft` background, `--magenta` on hover.
- Icon per mime-type: `📄` for documents, `📊` for spreadsheets, `🖼` for images, `📁` for other. Generated client-side from `mime_type`.

## 8. Decisions locked

These were considered before writing this draft. Each is resolved here unless the architect surfaces it again in §13 as needing user input.

1. **`/runs/{id}` is one route; the trace tree + detail drawer + accordions all coexist on it.** No tabbed layout (`Trace | Workflow | Artifacts | Attachments`) — tabs hide context. The accordions below the tree fold/unfold without changing route, so the trace stays visible while the user toggles other panels.
2. **Span detail = right drawer, not modal.** A modal would obscure the trace tree; the drawer keeps both visible. Drawer width 320px on desktop, full-width on narrow viewports.
3. **Live-trace approach for `running` runs.** v0.3 ships the polling approach (`GET /v1/runs/{id}/trace` every 2s while status=`running`, replacing the tree wholesale on each poll). The richer SSE-derived per-event-delta approach is deferred to v0.4 polish. Justification: the simpler approach is sufficient for typical 30-second-runs with ≤100 spans; the user perceives the tree filling in over a few seconds; the deeper streaming approach isn't load-bearing for the demo.
4. **Trace tree default state: fully expanded.** When the trace lands, all spans are expanded so the user sees the full structure at a glance. Click caret to collapse a subtree. User-set collapse state does NOT persist across navigations (kept in component state, not URL or localStorage); the default reverts to fully-expanded on each page load. v0.4 may add localStorage persistence if user behavior data warrants it.
5. **Span cost on parent rows is summed from descendants.** The leaf `generation` spans have direct cost; parent `step` / `branch` / `execution` spans display the sum of descendants. The sum is computed client-side (the API returns per-span cost only at the leaf level).
6. **Crashed-run rendering.** When the terminal `stream_completed` marker is missing, the inspector renders a `Crashed during run · trace may be incomplete` banner in `--magenta-deep` above the trace tree. The tree still renders what's available; the user is forewarned that the data is partial.
7. **File upload size cap = 100MB per file, 5GB aggregate by default.** Both user-configurable via `settings.files.max_per_file_mb` and `settings.files.max_total_gb`. The setting check happens at the start of the multipart stream (reading content-length header) AND continuously during the stream (hard-stop with 413 if the stream exceeds the cap mid-upload — a misreporting client). Storage cap is enforced by querying `SELECT SUM(size_bytes) FROM files` before accepting a new upload; if accepting would exceed the cap, return 413 with `code: "STORAGE_CAP_EXCEEDED"` and require the user to delete files first. No silent overwrite.
8. **File upload path on disk.** `<data-dir>/data/files/{file_id}/<original_name>`. The `original_name` preserves the user's filename in the path (useful when manually inspecting via Finder/Explorer); the `file_id` prefix dedupes against name collisions. Filename is sanitized for filesystem safety (`[A-Za-z0-9._-]` allowlist; other chars replaced with `_`). On download, the `Content-Disposition` header restores the original (un-sanitized) name.
9. **File deletion guard.** `DELETE /v1/files/{id}` checks whether the `file_id` appears in any row's `runs.task_input` (SQL: `WHERE json_extract(task_input, '$.attachments') LIKE '%' || :file_id || '%'`) — naive substring match because the array is small (typically 0-3 file_ids) and the JSON is short. If matched: 409 with `code: "FILE_REFERENCED_BY_RUNS", details: {run_ids: [list]}`. The UI offers to navigate to one of the referenced runs.
10. **Filter rail filters compose with AND.** Date range AND status multi-select AND workflow filter. Within status: OR (any of the selected statuses). Within workflow filter (when multi-value enabled v0.4): OR (any of the selected workflows). The filter rail's URL params are the source of truth; the React state derives from URL on every navigation so back/forward + share-URL behaviors work.
11. **Date range filter default = no filter** (not "Today"). The user lands on `/runs` and sees ALL runs by default. Choosing a date pill narrows. This avoids the "where did my old runs go?" confusion from a hidden-by-default narrow window.
12. **`Re-run` copies attachments verbatim.** The new run references the same `file_id`s; no re-upload. If a referenced file was deleted between the original run and re-run, the new run's `POST /v1/runs` fails with `ATTACHMENT_NOT_FOUND`; the inspector surfaces this and offers to retry with a different attachment.
13. **Trace tree virtualization threshold = 200 visible rows.** Below 200 visible rows (the typical v0.3 case), render naively (one DOM node per row). At 200+, lazy virtualization kicks in via `@tanstack/react-virtual`. The implementer benchmarks (polish item 1) and decides whether to ship the virtualization in this session or defer to v0.4. Defaults to "skip virtualization in v0.3" if benchmarks show no degradation at typical scale.
14. **Read-only canvas in "Workflow as run".** The accordion's xyflow canvas is `interactionsDisabled` (no drag, no edge creation, no node config form). The user can pan + zoom for inspection. Click any node → small read-only tooltip with the agent's model + tools (no edit affordance).
15. **Attachment mime icons are inferred from `mime_type`, not extension.** The server detects mime via `python-magic` or `mimetypes` stdlib on upload; the response carries it. The frontend's icon map keys off the detected type, not the filename's extension (so a `data.txt` that's actually a CSV gets the CSV icon based on content sniffing).

## 9. Design system additions

Session 05 adds NO new design-system tokens, fonts, or layout primitives. It reuses Session 03's complete system (palette, typography, motion tokens, the Spren orb).

New components Session 05 establishes (none with new design tokens; all reuse existing primitives):

- `TraceTree` + `SpanRow` — new layout pattern (recursive indented row), but typography and color tokens are all Session 03's.
- `SpanDetailPanel` — right drawer using Session 03's shadcn `Drawer` primitive themed with Session 03's tokens.
- `FileAttachInput` — uses Session 03's button + popover primitives + dropzone idiom.
- `RunHistoryFilters` — composed entirely of Session 03's button/select/checkbox primitives.

One pattern Session 05 formalizes for later sessions: **attribute lists** (key-value tables in span detail drawers, run metadata header, attachment metadata). Same idiom every time: 12px label in Geist Sans `--ink-soft`, value in Geist Mono 12px `--ink`, 12px row gap, label/value separated by a colon + 8px gap. Implementer ships a `<AttributeList />` component once and reuses it across all four surfaces (span detail, run metadata, attachment metadata, the workflow snapshot's per-node tooltip).

## 10. Polish items to address inside Session 05

These are gaps the architect-stage draft surfaced that the implementer addresses in-session, not as nice-to-haves.

1. **Trace tree perf at 1000+ spans.** Implementer benchmarks rendering at 100, 500, 1000, 5000 spans (synthetic fixture). If rendering >200ms or scrolling is janky at typical v0.3 sizes (≤200 spans), virtualization is needed; if benchmarks pass cleanly, defer to v0.4. The decision lands in this session's manual-verify checklist.
2. **Span detail full-content expansion.** "Show full prompt" / "Show full response" expansions are no-ops by default because `TracingConfig.include_message_content=false` is the framework default (full content not stored). The drawer shows a small explainer tooltip when the user hovers the disabled button: "Full prompt content not stored. Enable in settings → tracing → include_message_content."
3. **File upload progress.** Implementer wires `XMLHttpRequest` progress events (fetch + multipart doesn't expose upload progress reliably). Progress shown as `uploading… 42%` text + a 2px progress bar below the row in the popout. On completion: text replaced with the size value.
4. **File upload failure handling.** Network errors during upload show `× upload failed; retry` in `--magenta-deep`. Click retry → re-attempts the same file from byte 0 (no resumable uploads in v0.3 — keep simple). Size-cap rejection (413 from server) shows the specific reason inline.
5. **Drag-and-drop UX.** When a file is dragged over the canvas, the overlay's preview shows the filename being dragged (read from `DataTransfer.items`). Multi-file drag-and-drop: overlay shows `Drop to attach (3 files)`. Drag a non-file (e.g., a node from the palette) → overlay does NOT appear (filter via `DataTransfer.types`).
6. **Reduced-motion fallback for tree expand/collapse.** When `prefers-reduced-motion` is set, the caret toggles open/closed instantly (no 150ms slide animation on the children container). The drawer slide-in also degrades to instant.
7. **Keyboard navigation for the trace tree.** `j`/`k` (or arrow keys) move the focused row down/up. `Enter` opens the span detail drawer. `Esc` closes the drawer + returns focus to the previously-focused row. `→` expands a collapsed subtree; `←` collapses an expanded one.
8. **Date range filter custom-dialog edge cases.** `end < start` rejects with inline error. `end > today` defaults to today. Date picker uses the user's locale + the OS's preferred date format; the URL param is always ISO 8601 regardless of display format.
9. **Multi-status pill state.** When user deselects all 5 status pills, the filter rail interprets this as "no status filter" (equivalent to all selected), NOT as "show nothing." A small inline hint surfaces: `Showing all statuses (deselect leaves the filter inactive)`.
10. **Empty `/runs/{id}` states.**
    - Run still `running` and no spans closed yet: `<trace status="waiting_for_first_span" />` empty-state copy: `The run is starting. The trace tree will populate as spans close.`
    - Run `failed` before any span closed (rare; e.g., the Orchestra crashed during init): the trace section shows the error block from W-E but with `<trace status="no_spans_emitted" />` instead of a tree.
    - Run completed but `attachments=[]`: the Attachments accordion is hidden entirely (not "Attachments (0)" — just absent). Same for Artifacts (which is empty for most v0.3 runs).
11. **Re-run with stale attachments.** If a referenced file was deleted between the original run and the re-run click, the new run's `POST /v1/runs` returns `ATTACHMENT_NOT_FOUND`. The inspector surfaces an inline toast: `One attached file is no longer available. Re-run with remaining attachments? · [Confirm] · [Cancel]`. Confirm fires `POST /v1/runs` with the surviving file_ids.
12. **Trace JSON response size cap.** If `trace.ndjson` exceeds 50MB (well above typical retention), the endpoint truncates to the first 50MB worth of spans and adds `truncated: true` + `truncation_reason: "trace_size_cap"` to the response. The inspector surfaces a `Trace truncated for size · raw file at <data-dir>/data/runs/{id}/trace.ndjson` banner. The user can copy the raw file path for offline analysis.
13. **Artifact path traversal hardening.** The `GET /v1/runs/{id}/artifacts/{name}` endpoint resolves the filename against `<data-dir>/data/runs/{run_id}/artifacts/` and verifies (via `pathlib.Path.resolve()` + parent-dir comparison) that the resolved path is contained inside the artifacts directory. Any `..` or absolute path in `name` → 400. Per implementation, never serve files outside that directory regardless of what the URL says.

## 11. Success criteria

Extending Bundle 02's `test-scenarios.md` (which lands alongside this brief):

- **G-12** (inspect completed run): user clicks a run from `/runs`, lands on `/runs/{id}`, sees the trace tree with all spans expanded; clicks a generation span; drawer renders the model + token data correctly; closes; clicks a tool span; drawer renders the tool args (with redaction if applied); closes; expands "Workflow as run"; sees the frozen topology.
- **G-13** (attach file → run → inspect attachments): user drags a 1MB PDF onto the canvas, sees `📎(1)`, clicks Run, run completes, clicks through to `/runs/{id}`, expands Attachments, downloads the file; the downloaded file's sha256 matches the upload's sha256.
- **G-14** (filter run history end-to-end): user navigates to `/runs` with 50 mixed-status runs across 4 workflows; sets `Last 7 days` + status=`Failed,Cancelled` + workflow=`research-pipeline`; sees the filtered list correctly; clicks one; inspector renders the run; uses browser back; filters are still applied; URL contains the filter params.
- **G-15** (re-run with attachments): user re-runs a completed run with 2 attachments; new run starts; trace builds incrementally via polling; new run's attachments list shows the same 2 files (same `file_id`s).
- **G-16** (crashed-run rendering): user inspects a run where the framework crashed mid-execution (terminal marker missing); banner shows `Crashed during run · trace may be incomplete`; tree renders what's available.
- **U-07** (manual smoke): from a fresh data dir, build a workflow → attach a 1MB file → run it → inspect the trace → filter `/runs` to today → delete an unreferenced file via the API (since v0.3 has no UI surface for that).
- **C-03** (sha256 round-trip): a 5MB file uploaded then downloaded preserves bytes exactly; sha256 matches.
- **C-04** (file deletion guard): attempting to delete a file that's attached to a completed run returns 409 with `code: "FILE_REFERENCED_BY_RUNS"` + the run_id; the file is NOT deleted from disk.
- **C-05** (storage cap enforcement): uploading enough files to exceed the 5GB cap returns 413 with `code: "STORAGE_CAP_EXCEEDED"`; partial upload is cleaned up (no orphan bytes on disk).
- **X-08** (trace endpoint at scale): a synthetic 10MB / 1000-span `trace.ndjson` returns the full tree within 500ms; rendering at 1000 spans is acceptable per polish item 1's benchmark.
- **X-09** (path traversal): `GET /v1/runs/{id}/artifacts/..%2Fworkflow.json` returns 400, NOT a file outside the artifacts dir.
- **X-10** (file upload offline behavior): browser loses network mid-multipart-upload; client detects, surfaces `× upload failed; retry`; retry succeeds.
- **Visual regression baselines (Bundle 02 closing snapshots)**: `/runs/{id}` full inspector (succeeded), `/runs/{id}` full inspector (failed with error banner), span detail drawer open (generation, tool, step), canvas with `📎(2)` count + attachment popout open, canvas with drag-and-drop overlay, `/runs` with filter rail at default and at filtered-down states, the failed-run inspector with the drawer open on the validation error span.

## 12. Open research items the implementer resolves in-flight

Empirical version + behavior verification done during implementation, not via a separate pre-implementation research stage.

- **Trace JSON streaming.** Whether to ship `GET /v1/runs/{id}/trace` as a single bound JSON response or a chunked streaming response. v0.3 sizes (≤10MB typical, 50MB cap) are fine bound. Implementer verifies at typical sizes; if encoding latency >200ms server-side, switch to a streaming-line-by-line response (NDJSON over HTTP). Decision in the manual-verify checklist.
- **`@tanstack/react-virtual` integration with recursive trees.** The virtualization library expects a flat list. The implementer either (a) flattens the tree for virtualization and re-derives parent/child for rendering, or (b) virtualizes only the outermost level and renders children inline. Decide during implementation if virtualization is needed at v0.3 scale (polish item 1).
- **`python-magic` Linux dependency.** `python-magic` for content-sniffing requires `libmagic` on Linux. Verify the dependency is acceptable; fall back to `mimetypes.guess_type()` (stdlib, extension-based) if `libmagic` install friction outweighs the benefit. The frontend icon map keys off whatever the server returns; either backend works.
- **`UploadFile.read()` streaming behavior in FastAPI.** Verify FastAPI/Starlette streams the multipart body without buffering the whole file in memory. If buffering happens (memory leak risk at 100MB uploads), switch to manual `request.stream()` iteration.
- **SQLite `json_extract` performance on attachments substring match.** For `WHERE json_extract(task_input, '$.attachments') LIKE '%file_id%'` queries: at ~1000 runs in the DB, this is fine (full scan, microseconds). At 100k runs (v0.5 scale), an FTS5 index on the attachments array would be needed. Verify the v0.3 scale assumption holds.
- **xyflow read-only mode.** Confirm `nodesDraggable=false`, `nodesConnectable=false`, `edgesUpdatable=false`, `panOnDrag=true`, `zoomOnScroll=true` reads cleanly with no flashes-of-interactivity on initial render. The "interactionsDisabled" idiom is `@xyflow/react`'s standard but verify it doesn't leak hover states from Session 03's editor mode.
- **`Re-run` with stale workflow.** If the original run's `workflow_id` was archived between the original run and re-run, what happens? Decision: re-run fires anyway; the run executes against the *frozen* workflow.json (SP-009), not the archived workflow row. Verify the lifecycle coordinator reads the frozen snapshot, not `workflows.definition`. If it currently reads from `workflows.definition`, this needs a fix in this session — the snapshot is the source of truth post-Session 04.

If any of these surface a conflict with the locked decisions in §8, the implementer flags it and asks before deviating.

## 13. Status

- [ ] Tier confirmed (HIGH).
- [ ] Scope boundaries confirmed (sections 3 + 4).
- [ ] Files-to-CREATE list approved (section 5).
- [ ] Three user journeys approved (section 6).
- [ ] Skeleton wireframes approved (section 7).
- [ ] Decisions locked (section 8).
- [ ] Polish items captured for in-session work (section 10).
- [ ] Success criteria affirmed (section 11).
- [ ] Acceptance criteria frozen at [`./05-run-inspection/acceptance.md`](./05-run-inspection/acceptance.md) — extracted by the `acceptance-criteria-extractor` agent on the first implementation turn, before any code is written.
- [ ] Session 04 (run execution + AG-UI streaming + cost) merged before Session 05 implementation begins.
- [ ] Session implementation complete (all acceptance criteria pass; polish items addressed; tests green; manual verify done).
- [ ] Bundle 02 testing/test-scenarios.md fleshed out with the full Session 04 + Session 05 scenario set (filled alongside this brief — see [`../testing/test-scenarios.md`](../testing/test-scenarios.md)). Bundle 02 end-to-end testing runs only after both Session 04 + Session 05 ship.

**Next step:** user reviews; once locked, `acceptance-criteria-extractor` freezes `./05-run-inspection/acceptance.md` on the first implementation turn. Implementer begins after Session 04 lands. Polish items in §10 are explicit acceptance criteria — they're scoped into the session, not nice-to-haves.

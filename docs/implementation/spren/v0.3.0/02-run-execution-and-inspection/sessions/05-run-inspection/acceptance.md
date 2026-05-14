# Acceptance criteria — Spren Session 05 (Run Inspection: Trace Viewer + File Uploads + Run History UI)

Frozen at 2026-05-14T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

Numbering continues from Session 04's AC-219; this session begins at AC-220.

## Functional — Trace endpoint (`GET /v1/runs/{id}/trace`)

- AC-220: `GET /v1/runs/{id}/trace` returns 200 with a `RunTrace` JSON document for a run whose per-run NDJSON trace file exists on disk.
- AC-221: The endpoint locates the trace file by globbing `*.ndjson` inside `<data-dir>/data/runs/{run_id}/` and parses the single match.
- AC-222: When two or more `*.ndjson` files exist in the per-run directory (an invariant violation the per-run output_dir is meant to prevent), the endpoint surfaces an error rather than silently picking one.
- AC-223: The response body is a JSON object whose top-level keys include `spans`, `completion_status`, and `truncated_line_count`.
- AC-224: `spans` is a hierarchical (tree) list whose elements have a recursive `children` array; the wire shape replaces `parent_span_id` with children-of nesting.
- AC-225: Each `SpanNode` includes at minimum `span_id`, `kind`, `name`, timing fields, status, attributes, events, and `children`.
- AC-226: `completion_status` is one of exactly three string values: `complete`, `truncated`, `crashed` (framework spelling — `complete`, NOT `completed`).
- AC-227: There is no `"in_progress"` value in the `completion_status` enum returned by the endpoint.
- AC-228: `truncated_line_count` is an integer reflecting the count of partial / unparseable trailing lines from the reader.
- AC-229: When the synthetic 10-span fixture (3 generations, 1 tool, 1 step, 1 branch, 1 execution + the terminal marker) is used, the parsed tree shape exactly matches the fixture's parent/child wiring.
- AC-230: When the fixture contains a `kind="stream_event"` diagnostic line, that line is filtered out of the returned `spans` tree.
- AC-231: When the fixture contains a `stream_completed` terminal marker, `completion_status` is `complete`.
- AC-232: When the terminal marker is absent (crashed-trace fixture), `completion_status` is `crashed`.
- AC-233: When the run row's `status` is `running`, the endpoint returns the tree-up-to-now via the same parser path (no separate code path).
- AC-234: When `trace.ndjson` exceeds the 50MB cap, the endpoint truncates spans to the first 50MB and the response body sets `truncated: true` with `truncation_reason: "trace_size_cap"`.
- AC-235: The endpoint requires authentication; a request without the per-launch token returns 401.
- AC-236: The endpoint responds with 404 for a `run_id` that does not exist.
- AC-237: Orphan spans (with a `parent_span_id` that does not correspond to a known span_id in the file) are handled and surfaced rather than dropped silently.
- AC-238: The trace endpoint adapter (`runs/trace.py`) does NOT re-implement NDJSON parsing — it adapts the framework-shipped reader's output. (implied — verifiable via integration that exercises the framework reader's behavior; cannot be tested by reading source per the audit rule, so verified by the trace endpoint exhibiting parser-identical output across edge fixtures.)

## Functional — Workflow snapshot endpoint (`GET /v1/runs/{id}/workflow`)

- AC-239: `GET /v1/runs/{id}/workflow` returns 200 with a `WorkflowDefinition` JSON document.
- AC-240: The body returned is the contents of `<data-dir>/data/runs/{run_id}/workflow.json` (the frozen-at-run-start snapshot).
- AC-241: When the live `/v1/workflows/{workflow_id}` definition has been edited since the run started, the snapshot endpoint still returns the unmodified frozen version.
- AC-242: The endpoint requires authentication; unauth'd request returns 401.
- AC-243: The endpoint returns 404 when no snapshot file exists for the run.

## Functional — Artifacts list + download (`GET /v1/runs/{id}/artifacts*`)

- AC-244: `GET /v1/runs/{id}/artifacts` returns `{"items": []}` for a run with no artifacts directory or an empty one.
- AC-245: `GET /v1/runs/{id}/artifacts` returns `{"items": [...]}` listing files under `<data-dir>/data/runs/{run_id}/artifacts/` when present.
- AC-246: Each `ArtifactInfo` item includes `name`, `size_bytes`, `mime_type`, and `created_at` as ISO 8601.
- AC-247: `GET /v1/runs/{id}/artifacts/{name}` returns the file bytes for a valid artifact name.
- AC-248: The artifact download response sets `Content-Type` to the detected mime type.
- AC-249: The artifact download response sets `Content-Disposition: attachment; filename="<name>"`.
- AC-250: `GET /v1/runs/{id}/artifacts/{name}` rejects path traversal attempts: a `name` containing `..` returns 400 (no file outside the artifacts dir is ever served).
- AC-251: `GET /v1/runs/{id}/artifacts/{name}` rejects absolute paths in `name` with 400.
- AC-252: A URL-encoded traversal (e.g., `..%2Fworkflow.json`) is rejected with 400 — never serves a file outside the artifacts directory.
- AC-253: The artifact download enforces path containment via resolved-path comparison against the artifacts directory.
- AC-254: Artifact endpoints require authentication; unauth'd requests return 401.

## Functional — File upload (`POST /v1/files`)

- AC-255: `POST /v1/files` accepts a multipart upload and returns 201 with a `FileUploadResponse` JSON body on success.
- AC-256: `FileUploadResponse` includes at minimum `file_id`, `original_name`, `mime_type`, `size_bytes`, and `sha256`.
- AC-257: The uploaded bytes are written to `<data-dir>/data/files/{file_id}/<original_name>` on disk.
- AC-258: The on-disk write is atomic — write-temp + rename — so a partially-written file is never visible at the final path.
- AC-259: The original filename in the on-disk path is sanitized to the `[A-Za-z0-9._-]` allowlist (other chars replaced with `_`).
- AC-260: The `original_name` field returned in the response and stored in the metadata row preserves the user-supplied (un-sanitized) name; download `Content-Disposition` restores it.
- AC-261: A `files` row is inserted on successful upload.
- AC-262: The server computes sha256 during the upload stream and the value is stored in the `files` row.
- AC-263: A `sha256` round-trip succeeds: a 5MB file uploaded then downloaded preserves bytes exactly and the sha256 matches.
- AC-264: A file exceeding the per-file size cap (default 100MB, settable via `files.max_per_file_mb`) is rejected with HTTP 413 and `code: "FILE_TOO_LARGE"`.
- AC-265: The per-file cap is enforced both via Content-Length pre-check and continuously during the stream — a misreporting client whose actual bytes exceed the cap mid-stream is hard-stopped with 413.
- AC-266: Uploading a file that would push aggregate storage over the cap (default 5GB, settable via `files.max_total_gb`) is rejected with HTTP 413 and `code: "STORAGE_CAP_EXCEEDED"`.
- AC-267: Aggregate cap enforcement queries `SUM(size_bytes)` across the `files` table before accepting a new upload.
- AC-268: When an upload is rejected (cap or aggregate), partial bytes are cleaned up from disk — no orphan files remain.
- AC-269: `POST /v1/files` requires authentication; unauth'd requests return 401.
- AC-270: The upload handler streams the request body without buffering the entire file in memory (verifiable by uploading at the per-file cap without OOM).

## Functional — File metadata, download, and delete

- AC-271: `GET /v1/files/{id}` returns the file metadata row (without the on-disk path) for a known `file_id`.
- AC-272: `GET /v1/files/{id}` returns 404 for an unknown `file_id`.
- AC-273: `GET /v1/files/{id}/download` returns the raw bytes for a known `file_id`.
- AC-274: The download response sets `Content-Disposition: attachment; filename="<original_name>"` using the un-sanitized original name.
- AC-275: The download endpoint enforces the same path-confinement check as artifacts.
- AC-276: `DELETE /v1/files/{id}` returns 200 (or 204) and hard-deletes the row + on-disk bytes when the file is not referenced by any run.
- AC-277: `DELETE /v1/files/{id}` returns 409 with `code: "FILE_REFERENCED_BY_RUNS"` when the `file_id` appears in any `runs.task_input.attachments` array.
- AC-278: The 409 response body's `details` includes a `run_ids` list of all runs referencing the file.
- AC-279: When 409 is returned, the file row and on-disk bytes are NOT deleted.
- AC-280: The reference check uses element-equality via SQLite `json_each` (not substring matching) — immune to ULID-substring collisions.
- AC-281: All file metadata/download/delete endpoints require authentication; unauth'd requests return 401.

## Functional — `POST /v1/runs` attachment validation + lifecycle

- AC-282: `POST /v1/runs` with `task_input.attachments=[]` continues to succeed (Session 04 behaviour preserved).
- AC-283: `POST /v1/runs` with a non-empty `task_input.attachments` list of valid `file_id`s succeeds with 201 (Session 04's `ATTACHMENTS_NOT_YET_SUPPORTED` rejection is gone).
- AC-284: `POST /v1/runs` validates each `file_id` synchronously — before returning 201 — by looking it up in the `files` table.
- AC-285: An unknown `file_id` in `task_input.attachments` causes the POST to return 400 with `code: "ATTACHMENT_NOT_FOUND"` and `details: {file_id: <missing_id>}`.
- AC-286: When validation fails, the run is NOT created (no `runs` row inserted).
- AC-287: The `runs` row stores the validated `task_input.attachments` array verbatim (the array of file_ids).
- AC-288: The run lifecycle coordinator re-resolves the validated `file_id`s to file paths + metadata at run start.
- AC-289: The lifecycle coordinator appends a system-context block to `task_input.text` before passing the resulting `task` string to `Orchestra.execute()`.
- AC-290: The system-context block lists each attached file with at minimum its filename, absolute on-disk path, mime type, and human-readable size.
- AC-291: The system-context block instructs the agent to access files via the `read_file` tool (the prompt includes "Use the `read_file` tool to access them." or equivalent literal language verifiable from the integration test).
- AC-292: An integration test fixture asserts that `Orchestra.execute()` receives a `task` string that contains the file path, filename, mime type, and size for each attachment.
- AC-293: When `task_input.text` is empty (canvas-Run shortcut), the system-context block is still appended.

## Functional — `GET /v1/runs` filter extensions

- AC-294: `GET /v1/runs?status=running,succeeded` returns runs whose status is in the comma-separated set (OR matching).
- AC-295: Single-value `?status=running` calls (Session 04 callers) continue to work unchanged (backward compatible).
- AC-296: `GET /v1/runs?since=2026-05-01T00:00:00Z` (absolute ISO 8601) returns runs created at or after that timestamp.
- AC-297: `GET /v1/runs?since=24h` (relative shorthand from Session 04) continues to work unchanged.
- AC-298: `GET /v1/runs?until=<iso8601>` returns runs created at or before that timestamp.
- AC-299: `?since` and `?until` compose as a date range (AND).
- AC-300: `GET /v1/runs?workflow_id=<id1>,<id2>` returns runs whose `workflow_id` is in the set (OR matching).
- AC-301: Single-value `?workflow_id=<id>` calls (Session 04 callers) continue to work unchanged.
- AC-302: All filters compose with AND across families: date range AND status set AND workflow set.
- AC-303: An invalid ISO 8601 in `since` or `until` returns 400.
- AC-304: An invalid `status` value (not in the canonical enum) returns 400.

## Functional — Tracing wired in `materialize.py` [blocked-on: tracing-on-by-default]

- AC-305: `_build_execution_config` constructs an `ExecutionConfig` whose `tracing` field has `enabled=True`. [blocked-on: tracing-on-by-default]
- AC-306: The constructed `TracingConfig.output_dir` is `<data-dir>/data/runs/{run_id}` for the run being materialized. [blocked-on: tracing-on-by-default]
- AC-307: The constructed `TracingConfig.include_message_content` is `True` (Spren preserves the framework default; full content captured). [blocked-on: tracing-on-by-default]
- AC-308: `materialize_run(...)` accepts `data_dir` and `run_id` kwargs and threads them through to the per-run `output_dir` value. [blocked-on: tracing-on-by-default]
- AC-309: After a real run completes, exactly one `*.ndjson` file exists at `<data-dir>/data/runs/{run_id}/`. [blocked-on: tracing-on-by-default]
- AC-310: The lifecycle coordinator passes `data_dir` + `run_id` through to `materialize_run(...)`.

## Functional — Trace tree component (`apps/web/src/components/TraceTree`)

- AC-311: The trace tree renders `RunTrace.spans` recursively, with parent spans containing nested children.
- AC-312: Each span row displays a caret (▶ collapsed / ▼ expanded), kind chip, span name, right-aligned timing chip, right-aligned cost chip, and status indicator.
- AC-313: The status indicator is `✓` for ok and `✗` for error spans.
- AC-314: Error-status spans render the `✗` indicator in the magenta-deep palette colour.
- AC-315: Indentation is 16px per depth level.
- AC-316: A vertical line connects parent → child rows visually.
- AC-317: Clicking a span row opens the span detail drawer.
- AC-318: Clicking the caret toggles expand/collapse for that subtree without opening the drawer.
- AC-319: Default state on initial render: all spans expanded.
- AC-320: User-toggled collapse state does NOT persist across navigations (kept in component state, not URL or localStorage).
- AC-321: Reverting to a `/runs/{id}` route reverts to the all-expanded default.
- AC-322: Parent rows display a cost equal to the sum of all descendants' costs (computed client-side; the API only carries leaf costs).
- AC-323: Tree builder correctly wires children using each span's `parent_span_id` (or `children` if pre-nested by the API).
- AC-324: Orphan spans (parent_span_id pointing to no known span) are surfaced rather than silently dropped.

### Crashed-run rendering

- AC-325: When the API response carries `completion_status: "crashed"`, the inspector renders a banner above the trace tree reading `Crashed during run · trace may be incomplete` (or equivalent literal copy verifiable from the test).
- AC-326: The banner is rendered in the magenta-deep palette colour.
- AC-327: Even when crashed, the tree still renders the spans that were captured.

### Live-run polling

- AC-328: While `runs.status === "running"`, the trace fetcher polls `GET /v1/runs/{id}/trace` every 2 seconds.
- AC-329: Each poll replaces the tree wholesale (no incremental delta application).
- AC-330: When `runs.status` transitions away from `running`, polling stops.

### Empty + waiting states

- AC-331: A `running` run with zero spans yet rendered shows a `<trace status="waiting_for_first_span" />` empty state with literal copy `The run is starting. The trace tree will populate as spans close.` (or copy verifiable from the test).
- AC-332: A `failed` run with zero spans (Orchestra crashed during init) shows the error block in place of a tree, with `<trace status="no_spans_emitted" />` instead of an empty tree placeholder.

### Virtualization

- AC-333: With ≤200 visible rows the tree renders naively (one DOM node per row).
- AC-334: At ≥200 visible rows, virtualization is applied (or the implementer explicitly defers per polish item 1's benchmark — verified by manual-verify checklist).
- AC-335: A 1000-span synthetic fixture is verified to render and remain interactive.

### Keyboard navigation

- AC-336: `j` / down-arrow moves focus to the next visible row.
- AC-337: `k` / up-arrow moves focus to the previous visible row.
- AC-338: `Enter` opens the span detail drawer for the focused row.
- AC-339: `Esc` closes the span detail drawer and returns focus to the previously-focused row.
- AC-340: `→` expands a collapsed subtree.
- AC-341: `←` collapses an expanded subtree.

### Reduced motion

- AC-342: When `prefers-reduced-motion` is set, the caret toggles open/closed instantly with no slide animation on the children container.
- AC-343: When `prefers-reduced-motion` is set, the drawer slide-in degrades to instant.

## Functional — Span detail panel (`SpanDetailPanel`)

- AC-344: The span detail panel is built atop the existing `SlideOver` primitive (`apps/web/src/components/ui/SlideOver/SlideOver.tsx`).
- AC-345: The panel renders on the right side of the viewport.
- AC-346: The panel width is 320px on desktop viewports.
- AC-347: The panel is full-width on narrow viewports.
- AC-348: The panel passes `dismissOnBackdrop={false}` so accidental backdrop clicks while reading attributes do NOT dismiss the drawer.
- AC-349: `Esc` dismisses the drawer.
- AC-350: A close button in the drawer header dismisses the drawer.
- AC-351: The drawer renders the span's kind, name, timing, and status.
- AC-352: For `generation` spans, the drawer renders model, provider, prompt tokens, completion tokens, reasoning tokens, response_time_ms, finish_reason, has_thinking, has_tool_calls, and cost.
- AC-353: For `tool` spans, the drawer renders tool name, agent name, arguments, and result_summary.
- AC-354: When the redactor scrubbed an argument value, the drawer renders `[REDACTED]` for that value.
- AC-355: A redacted value gets a small affordance (e.g., `?` indicator) explaining the SecretRedactor ran.
- AC-356: The drawer renders the full `events` array (validation decisions, user-interaction prompts) when present.
- AC-357: The drawer renders `links` when present.
- AC-358: A "Show full prompt" button appears on `generation` span drawers when `prompt_content` is captured.
- AC-359: Clicking "Show full prompt" toggles an inline collapsed-by-default expansion.
- AC-360: The expanded prompt content renders in a Geist Mono 12px monospace block with `--surface-elevated` background.
- AC-361: A second click on "Show full prompt" collapses the expansion.
- AC-362: A "Show full response" button has the same toggle behaviour.
- AC-363: Long content (>2000 chars) inside an expansion has a "Copy to clipboard" affordance.
- AC-364: When a `generation` span lacks `prompt_content` / `response_content` attributes (e.g., older traces or `include_message_content=False`), the Show-full buttons render disabled with the tooltip `Content not captured for this span` (or copy verifiable from the test).

## Functional — Canvas FileAttachInput

- AC-365: A `📎` icon button is rendered in the canvas top toolbar to the immediate left of the Run button.
- AC-366: With no files attached, the icon shows just `📎`.
- AC-367: When N files are attached, the icon shows a count badge `📎(N)`.
- AC-368: Clicking the icon opens the attachments popout listing each file with filename, size, and an `×` to remove.
- AC-369: The popout includes a `+ Add file` button that opens a native multi-select file picker.
- AC-370: Files chosen via the picker upload to `POST /v1/files` with progress events.
- AC-371: Drag-and-drop a file onto anywhere on the canvas triggers the upload flow.
- AC-372: When a file is being dragged over the canvas, a translucent overlay covers the canvas with the message `Drop to attach (<filename>)` (or copy verifiable from the test).
- AC-373: For multi-file drag-and-drop, the overlay reads `Drop to attach (N files)` (or equivalent verifiable copy).
- AC-374: When a non-file (e.g., a node from the palette) is dragged, the drag-and-drop overlay does NOT appear (filter via `DataTransfer.types`).
- AC-375: On successful upload, the new `file_id` is appended to a Jotai canvas-attachments atom (`apps/web/src/stores/canvasAttachments.ts`).
- AC-376: On run completion (success or cancel), the canvas-attachments atom resets to empty.
- AC-377: The `Run` button's `POST /v1/runs` request body's `task_input.attachments` is read from the Jotai atom.
- AC-378: Each upload row shows progress as `uploading… N%` text plus a 2px progress bar below the row while uploading.
- AC-379: On upload completion, the progress text is replaced with the file's size value.
- AC-380: On upload failure (network or other error), the row shows `× upload failed; retry` in the magenta-deep colour.
- AC-381: Clicking retry on a failed upload re-attempts the upload from byte 0.
- AC-382: Size-cap rejection (413 from server) shows the specific reason inline on the row.
- AC-383: Removing a file via `×` removes the file_id from the Jotai atom.
- AC-384: Drag-and-drop UX: the overlay shows the dragged filename (read from `DataTransfer.items`).

## Functional — `/runs/{id}` full inspector

- AC-385: `/runs/{id}` renders the metadata header (status badge, duration, cost, token counts, start/finish timestamps) — this part is preserved from Session 04.
- AC-386: The Session 04 placeholder (`<trace-viewer status="coming_in_session_05" />`) is removed.
- AC-387: Below the metadata header, the inspector renders (in order): the trace tree section, the Attachments accordion, the Workflow-as-run accordion, the Artifacts accordion.
- AC-388: A `Re-run` button is rendered near the metadata header.
- AC-389: A horizontal rule separates the metadata header from the trace section.

### AttachmentList

- AC-390: When the run has zero attachments, the Attachments accordion is hidden entirely (NOT shown as `Attachments (0)`).
- AC-391: When the run has attachments, the Attachments accordion shows the count: `Attachments (N)`.
- AC-392: Expanding the Attachments accordion renders one row per file, each showing the filename, size, mime type, and a `Download` link.
- AC-393: Clicking `Download` triggers a file download via `GET /v1/files/{id}/download` for the chosen file.
- AC-394: The downloaded file's bytes match the originally-uploaded file's bytes.
- AC-395: The downloaded file's sha256 matches the upload's sha256.
- AC-396: A mime-type-aware icon is rendered for each attachment row.
- AC-397: The mime icon mapping keys off the server-detected `mime_type` (not the filename extension).

### Workflow snapshot accordion

- AC-398: A "Workflow as run" accordion section is rendered, collapsed by default.
- AC-399: Expanding the accordion fetches and renders a read-only xyflow canvas using the frozen `WorkflowDefinition` from `GET /v1/runs/{id}/workflow`.
- AC-400: The read-only xyflow canvas has `nodesDraggable={false}`, `nodesConnectable={false}`, `edgesUpdatable={false}`, `elementsSelectable={false}`.
- AC-401: The read-only xyflow canvas has `panOnDrag={true}` and `zoomOnScroll={true}`.
- AC-402: A side button "Open in canvas" navigates to `/workflows/{workflow_id}` (the live, possibly-edited workflow).
- AC-403: Clicking a node in the read-only canvas shows a small read-only tooltip with the agent's model + tools (no edit affordance).
- AC-404: The read-only canvas reuses Session 03's existing `workflowToReactFlow` conversion utility.

### Artifacts accordion

- AC-405: When the run has zero artifacts, the Artifacts accordion is hidden entirely.
- AC-406: When artifacts exist, the accordion shows the count `Artifacts (N)`.
- AC-407: Expanding the accordion lists each artifact with name, size, and a download link via `GET /v1/runs/{id}/artifacts/{name}`.

### Re-run

- AC-408: Clicking `Re-run` fires `POST /v1/runs` with the same `workflow_id` as the original run.
- AC-409: The re-run request copies `task_input.text` verbatim from the original run.
- AC-410: The re-run request copies `task_input.attachments` (the array of file_ids) verbatim — no re-upload occurs.
- AC-411: On successful re-run create, the user is navigated to `/runs/{new_id}`.
- AC-412: The new run's trace tree starts empty and builds incrementally via the 2s polling loop.
- AC-413: When a referenced file was deleted between original run and re-run, the re-run `POST /v1/runs` returns `ATTACHMENT_NOT_FOUND` and the inspector surfaces an inline toast `One attached file is no longer available. Re-run with remaining attachments? · [Confirm] · [Cancel]` (or copy verifiable from the test).
- AC-414: Confirming the toast fires `POST /v1/runs` with only the surviving file_ids.
- AC-415: When the original run's `workflow_id` was archived, the re-run still executes against the frozen `workflow.json` snapshot (NOT the live archived row).

## Functional — Run history filter rail (`RunHistoryFilters`)

- AC-416: All three sub-pieces (DateRangeFilter, StatusMultiSelect, WorkflowFilter) ship inline within the same `RunHistoryFilters.tsx` file (no premature extraction per CLAUDE.md anti-pattern #4).
- AC-417: The default state shows all runs (no date filter, all 5 statuses selected, all workflows).
- AC-418: The date row renders relative pills: `Today`, `Yesterday`, `Last 7 days`, `Last 30 days`, `Custom...`.
- AC-419: The date row is single-select (only one date pill active at a time).
- AC-420: Selecting `Last 7 days` updates the URL with `?since=<-7d>` (or equivalent ISO 8601).
- AC-421: Clicking `Custom...` opens a two-input dialog (start date, end date).
- AC-422: The custom date dialog uses native `<input type="date">` with shadcn styling (no third-party date picker library).
- AC-423: The custom date dialog uses the user's locale and OS preferred date display format.
- AC-424: The URL params `since` / `until` are always serialized as ISO 8601 regardless of display format.
- AC-425: Custom date dialog: `end < start` rejects with an inline error.
- AC-426: Custom date dialog: `end > today` defaults to today.
- AC-427: Status row renders five pills: `Running`, `Queued`, `Succeeded`, `Failed`, `Cancelled`.
- AC-428: Each status pill is independently toggleable.
- AC-429: Selecting a subset of pills serializes a comma-separated `?status=` URL param.
- AC-430: Selected pills render with magenta background.
- AC-431: Unselected pills render with `--surface-elevated` background.
- AC-432: Deselecting all 5 status pills is interpreted as "no status filter" (equivalent to all selected), NOT as "show nothing."
- AC-433: When all 5 status pills are deselected, an inline hint surfaces: `Showing all statuses (deselect leaves the filter inactive)` (or copy verifiable from the test).
- AC-434: The workflow filter is a dropdown listing non-archived workflows by name with provenance badges.
- AC-435: The dropdown supports search-as-you-type filtering of the workflow list.
- AC-436: Selecting a workflow updates the URL with `?workflow_id=<id>`.
- AC-437: All filters compose with AND across families.
- AC-438: Within the status family, selected pills compose with OR.
- AC-439: The URL search params are the source of truth for filter state.
- AC-440: React state for filters derives from URL on every navigation (back/forward and shared URLs work correctly).
- AC-441: Browser back navigates to the previous filter set; the displayed filters reflect that state.
- AC-442: A run count `(N of M)` is rendered in the top-right showing the filtered count vs total.

## Functional — File metadata + storage

- AC-443: A forward-only migration creates a `files` table per the data model spec.
- AC-444: The `files` table has an index on `(created_at)` (for sweepers).
- AC-445: The `files` table has an index on `(sha256)` (for future dedup).
- AC-446: Every uploaded file results in a new row regardless of sha256 (no v0.3 dedup; v0.4 work).

## Non-functional — Security

- AC-447: All Spren routes added in this session except `/healthz` (which is not added by this session) require the per-launch token; missing or wrong token returns 401.
- AC-448: All routes added in this session honour the existing CORS regex from `server.create_app()`.
- AC-449: `GET /v1/runs/{id}/artifacts/{name}` is hardened against URL-encoded path traversal (e.g., `..%2F`, `%2e%2e/`).
- AC-450: `POST /v1/files` and `GET /v1/files/{id}/download` confine file access to the configured `<data-dir>/data/files/` root.
- AC-451: Argument-redaction performed by the framework's `SecretRedactor` is preserved in the trace endpoint response (Spren does NOT re-render unredacted arguments from another source).

## Non-functional — Performance

- AC-452: A synthetic 10MB / 1000-span `trace.ndjson` returns the full tree from `GET /v1/runs/{id}/trace` within 500ms.
- AC-453: Trace tree rendering at 1000 spans is acceptable per the polish-item-1 benchmark and the rendering performance is captured in the manual-verify checklist.
- AC-454: Trace tree benchmarks at 100, 500, 1000, and 5000 spans are captured during implementation.

## Non-functional — Error handling + observability

- AC-455: When `trace.ndjson` exceeds the 50MB cap, the inspector surfaces a banner: `Trace truncated for size · raw file at <data-dir>/data/runs/{id}/trace.ndjson` (or copy verifiable from the test) with the raw file path made copy-able.
- AC-456: Network errors during multipart upload surface `× upload failed; retry` in the magenta-deep colour without crashing the canvas.
- AC-457: A browser losing network mid-multipart upload triggers the failure UI; clicking retry succeeds when network returns.
- AC-458: Mid-stream cap exceedance does NOT leave orphan partial bytes on disk.

## Non-functional — Accessibility

- AC-459: The trace tree is keyboard-operable end-to-end (per AC-336–AC-341 plus standard tab semantics).
- AC-460: The span detail drawer respects focus trap (provided by `SlideOver`) when open.
- AC-461: When the drawer closes, focus returns to the previously-focused tree row.
- AC-462: The drag-and-drop overlay has reduced-motion fallbacks consistent with the rest of the inspector.

## Out of scope

- Task-input dialog (lands in Session 06)
- Trace search across spans within a run (v0.4)
- Cross-run trace search (v0.4)
- Cost charts / rollup graphs (Session 06 or v0.4)
- Mid-run user-interaction (`POST /v1/runs/{id}/respond`) (Session 06)
- Pause / resume primitives + `POST /v1/runs/{id}/pause` / `resume` (v0.4)
- SSE-derived per-event-delta tree updates (v0.4 polish; v0.3 ships polling-only)
- Inline preview of attached files (PDF viewer, image thumbnails, syntax-highlighted text) (v0.4)
- Inline preview of artifact outputs (v0.4)
- File library / "all my uploaded files" page (v0.4)
- File deduplication on upload (v0.4)
- Bulk operations on runs (v0.4)
- Run notes / annotations (Session 08)
- OpenTelemetry / external tracing export (v0.4 — `SprenTelemetrySink`)
- Multi-select on workflow filter (v0.4 if requested; Session 05 ships single-select)
- localStorage persistence of trace-tree expand/collapse state (v0.4)
- Resumable file uploads / chunked uploads (v0.3 keeps simple upload)
- Hand-rolled NDJSON parser in Spren (re-uses framework's `TraceTree.from_ndjson` per SP-001 + decision §16)

## Resolutions of flagged criteria (2026-05-14)

- **AC-451 — trust framework redaction (no re-redaction)**: Architecture confirms `SecretRedactor` runs at `packages/framework/src/marsys/coordination/tracing/collector.py:632` BEFORE the NDJSON writer (and any other sink). Spans on disk are already scrubbed. Spren MUST NOT re-redact — it would either be a no-op (waste) or, worse, would silently mutate values the framework intentionally left alone. Plan §3 SpanDetailPanel line "renders the redacted spans without re-redacting" stands. AC-451 reflects this.
- **AC-422 — shadcn styling = Tailwind tokens**: Session 03 shipped Tailwind + the `apps/web/src/components/ui/` primitive set (Button, Card, Dialog, SlideOver, Kbd, TagMarkup). "shadcn styling" here means design tokens from Session 03 applied to a native `<input type="date">`; no third-party date picker library. AC-422 stands as written.

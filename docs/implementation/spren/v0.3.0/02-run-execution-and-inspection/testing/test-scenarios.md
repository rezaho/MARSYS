# Bundle 02 — Run Execution and Inspection — Test Scenarios

**Bundle**: 02 — Run execution and inspection
**Sessions**: 04 (Run execution + AG-UI streaming + cost), 05 (Run inspection: trace viewer + file uploads + run history UI)
**Demo-able outcome**: From a workflow saved in Bundle 01, user attaches a PDF, clicks Run, watches the workflow execute live with token-by-token streaming visible in the Spren orb's `speaking` state, sees cost computed and displayed on completion, navigates to `/runs/{id}` to see the full nested-span trace viewer with per-span timing + cost + status chips, clicks any span to view its full attributes in a side drawer, downloads the attached PDF inline, navigates back to `/runs`, filters to last 7 days + status=failed, inspects a failed run's error context. Cancel mid-run cleanly. Reconnect after a network blip without losing event continuity.

## Overview

This file is the source of truth for Bundle 02's bundle-end real-world test. Four testing layers consume it at the bundle boundary, matching Bundle 01's pattern:

1. **Cross-session scripted integration** — Playwright (browser GUI) + tauri-driver/WebdriverIO (Tauri shell) + httpx (API-only flows). Runs in CI; deterministic.
2. **Agent-driven exploratory** — Claude Code with `mcp__claude-in-chrome__*` reads this file and runs through golden-path + edge-case + exploratory scenarios, plus self-directed variations. Produces `tmp/spren/v0.3.0/02-run-execution-and-inspection/testing/exploratory-report.md`.
3. **Visual regression** — Playwright screenshot snapshots (the local Bundle 01 idiom; not Argos) extending Bundle 01's baseline set. Diffs flagged.
4. **User manual smoke** — 5-min checklist on a fresh data dir + fresh install. Bundle ships only on 100% pass.

The cross-session seams are the load-bearing focus. Per-session tests cover their own session in isolation; Bundle 02's test exercises the joins between Sessions 04 + 05 + their interaction with Bundle 01:

- Session 04's `runs` table + `<data-dir>/data/runs/{id}/trace.ndjson` → Session 05's `GET /v1/runs/{id}/trace` reading the same file
- Session 04's `POST /v1/runs` attachments rejection (`code: "ATTACHMENTS_NOT_YET_SUPPORTED"`) → Session 05's full attachments resolution producing a populated `task_input.attachments` field round-trip
- Session 04's SSE stream `/v1/runs/{id}/events` → Session 05's `/runs/{id}` page subscribing for live-tree polling (or the v0.4 SSE-delta path if shipped early)
- Bundle 01's saved workflow → Session 04's frozen-at-run-start `workflow.json` → Session 05's "Workflow as run" accordion rendering that exact snapshot
- Bundle 01's canvas Run button shell → Session 04's live wiring + Session 05's adjacent `📎` attach affordance
- Session 04's `/runs` list page with simple chips → Session 05's filter rail replacing them in place; existing routes still work
- Session 04's cost computation on every `GenerationEvent` → Session 05's per-span cost chip readable on the inspector

Bundle 02 also exercises a framework-side dependency: **Framework Session 06 (AG-UI translator) must be merged + released before Bundle 02 implementation starts.** Bundle 02's bundle-end test runs only after Sessions 04 + 05 + Framework Session 06 are all green.

## Setup steps

Every scenario assumes a clean baseline unless stated otherwise. The "fresh-state" preamble extends Bundle 01's:

```bash
# 1. Start from a clean checkout of feature/spren-umbrella (or the v0.3 release tag once cut)
cd /path/to/marsys-spren-work
git status                                      # clean tree

# 2. Wipe per-user data dir for spren (preserves your real data; bundle tests run against an empty store)
DATA_DIR="$(uv run --package spren python -c 'import platformdirs; print(platformdirs.user_data_dir(\"spren\"))')"
rm -rf "$DATA_DIR"

# 3. Confirm dev ports are free
ss -ltn | grep -E ':(8765|5173)\b' && echo "PORT IN USE — kill before continuing" && exit 1

# 4. Provide an LLM key for live runs (Bundle 02's FIRST bundle to need this)
export OPENROUTER_API_KEY="<key>"               # required for G-08..G-16 + most scenarios

# 5. Install + build
just install                                    # uv sync + pnpm install + cargo fetch + cargo install tauri-cli
just build                                      # vite build + dist guard + copy to packages/spren/src/spren/_webui/
just test                                       # baseline must be green: framework 841/764, spren ≥21, tui ≥2, web ≥1, e2e ≥2, cargo ≥7
```

If `just test` is not green at the baseline counts, **stop**. Bundle-end testing assumes per-session tests pass.

For the Tauri-shell scenarios, native macOS / Windows / Linux is required — WSLg's webkit2gtk integration is broken (Session 01 lessons). On WSL2, run only the browser-GUI + httpx layers; mark Tauri scenarios `inconclusive — environment` rather than `pass`.

For deterministic cost calculation in scenarios, use the canonical model `openai/gpt-4.1-mini` (or whichever model the v0.3 cost-rates.yaml has stable known-good rates for). For trace-shape scenarios, the synthetic trace fixtures live at `packages/spren/tests/fixtures/traces/*.ndjson`; tests load these directly via the `runs/trace` endpoint after seeding a fake `<data-dir>/data/runs/{id}/` directory.

The scripted integration layer is invoked via a `just bundle-02` recipe (`apps/web/tests/bundle/` for Playwright + `packages/spren/tests/bundle/` for httpx). The recipe lands when the suite is first authored.

## Golden-path scenarios

Must-pass on every bundle-end run. These cover the demo-able outcome end-to-end.

### G-08 — Run execution golden path (Session 04 — live)

**Setup**: `just dev-desktop` running, fresh data dir, one workflow created from Bundle 01 (3-agent research-pipeline, `provenance=visual_builder`).

**Steps**:
1. User opens the workflow on the canvas.
2. Clicks `Run`. Button transitions to `submitting` (~200ms), then `running` showing `Cancel · 0s · 0 tokens`. `POST /v1/runs` fires with `task_input: {text:"", attachments:[]}`.
3. SSE stream `/v1/runs/{id}/events` opens. `RUN_STARTED` arrives.
4. Researcher's first generation streams: orb shifts from `thinking` to `speaking`; Run button's token counter increments; elapsed timer ticks.
5. Researcher → Writer → Editor (each: thinking → speaking → tool calls).
6. `RUN_FINISHED` arrives. Orb returns to `idle`. Run button returns to `idle`. Completion toast slides in: `Completed in 12.3s · $0.026`.
7. User clicks the toast → navigates to `/runs/{id}`.
8. Inspector renders the run metadata (status `succeeded`, duration `12.3s`, cost `$0.026`, tokens `~8,432 in / 1,221 out`, timestamps).

**Expected**: Orb state transitions visible at every phase. Token counter monotonically increases. Final cost matches hand-calculation within 0.01¢ for the known model rates. No `🛑 STREAM_LAGGED` warning in console.

**Edge variations to vary**: dark/light mode toggle mid-run (Session 03's tokens; should not flash); window resize during run.

### G-09 — Cancel mid-run with 5s countdown (Session 04 §8.7)

**Setup**: Same as G-08; use a workflow tuned to run ~60s (e.g., research-pipeline with browser-based tools).

**Steps**:
1. User clicks Run; run begins; ~20s in, Researcher is mid-search.
2. User clicks `Cancel` on the Run button.
3. Inline confirmation appears below: `Cancel run? Tool calls in flight will finish.` + `[Keep running] [Cancel run]`.
4. User clicks `Cancel run`. UI shows `Cancelling… 5` countdown.
5. Countdown ticks 5 → 4 → 3 → 2 → 1 → 0.
6. At the moment countdown displays `1` (t=4s of countdown), Spren sends force-abort to the framework.
7. At countdown=0, UI displays `Cancelled`. Toast: `Cancelled after 22s · $0.008`.
8. Run row: `status=cancelled`, `finished_at` set.

**Expected**: User-visible countdown smooth (no jitter). Framework's `Orchestra.abort()` (or equivalent — verified by polish item §12 in Session 04) signaled at the right moment. No orphaned tool-call processes after cancel.

### G-10 — Reconnect after network blip (Session 04 §J-3)

**Setup**: `just dev` running, browser tab open on a running workflow's `/workflows/{id}` page.

**Steps**:
1. Run is in progress; SSE streaming tokens.
2. In Chrome DevTools → Network → Throttle to "Offline". Wait 3s.
3. Restore "Online".
4. Within 8s (backoff 1s → 2s → 4s → max 8s), client re-establishes SSE with `Last-Event-ID: <last-received-id>`.
5. Server: reads `trace.ndjson` from start up to last-received-id offset, then switches to live consumption. Stream resumes.
6. UI: `Reconnecting…` annotation disappears within 500ms of reconnect; token counter catches up.

**Expected**: No duplicate events on reconnect. No events lost.

### G-11 — Run list page (Session 04 W-C)

**Setup**: Fresh data dir; create 5 runs across 3 workflows (2 succeeded, 1 failed, 1 cancelled, 1 currently running). Aggregate `/v1/runs/events` SSE active.

**Steps**:
1. Navigate to `/runs` via ⌘K → "runs".
2. 5 cards visible, most recent at top.
3. Each card: workflow name link + status badge + cost + duration + relative timestamp.
4. Aggregate SSE emits a `RunUpdated` event for the running run → its card updates in place (elapsed time + token count ticks).
5. Click `Failed` filter chip → only the failed run visible.
6. Click `All` chip → all 5 visible again.
7. Click any card → navigates to `/runs/{id}`.

**Expected**: Aggregate SSE is one connection (verify via Chrome DevTools network). Filter changes are instant (no re-fetch — client-side filter on a small list). Status badges use the palette per Session 04 §8.10.

### G-12 — Inspect completed run with full trace (Session 05 J-4)

**Setup**: One succeeded run from G-08. User lands on `/runs/{id}`.

**Steps**:
1. Inspector renders. Below metadata: `Trace` section with the tree fully expanded by default.
2. Tree shape: `execution > branch > step #1 Researcher > generation`, etc.
3. Click on a `generation` span. Drawer slides in from the right.
4. Drawer shows: model name, provider, prompt/completion/reasoning tokens, response_time_ms, finish_reason, has_thinking, has_tool_calls, cost.
5. User clicks `Show full prompt` — tooltip explains why the button is disabled (`TracingConfig.include_message_content=false` is the default).
6. User presses Esc → drawer closes; row loses highlight.
7. User clicks a `tool` span. Drawer renders: tool name, agent name, arguments (with `[REDACTED]` substitution where the SecretRedactor ran), result_summary.
8. User expands the "Workflow as run" accordion → read-only canvas renders the 3-agent topology snapshotted at run start.
9. User clicks `Re-run`. New run starts. UI navigates to `/runs/{new_id}`; tree builds incrementally as Session 05's polling kicks in.

**Expected**: Trace tree renders without delay (<200ms for typical 30-span trace). Drawer slides cleanly; no layout jank. Per-span cost = generation-span cost; parent cost = sum-of-descendants. `Re-run` creates a new run with the same `task_input.text`.

### G-13 — Attach file → run → inspect attachments (Session 05 J-5)

**Setup**: Fresh data dir; one workflow ready; a 1.2MB PDF on disk (e.g., `packages/spren/tests/fixtures/files/sample_report.pdf`).

**Steps**:
1. User opens canvas. Drags the PDF from the file system onto the canvas (anywhere).
2. Translucent overlay appears: `Drop to attach (sample_report.pdf)`.
3. User releases drop. Overlay disappears. `📎(1)` appears next to the Run button. Brief popout: `sample_report.pdf · 1.2 MB attached`.
4. User clicks `📎(1)` → permanent popout with the file + `+ Add file` button.
5. Adds a second file (87KB CSV) via the picker. Upload progresses. After completion: `📎(2)`.
6. Clicks Run. `POST /v1/runs` includes `task_input.attachments=[file_id_1, file_id_2]`.
7. Backend resolves attachments → appends file paths + names + mime types + sizes to the system context block before passing to `Orchestra.run()`.
8. Run executes. The Researcher agent uses `read_file` (visible in the trace).
9. Run completes; user clicks toast → `/runs/{id}`.
10. Below metadata: Attachments accordion shows 2 files: `sample_report.pdf 1.2 MB application/pdf [Download]` and `data.csv 87 KB text/csv [Download]`.
11. User clicks `Download` on the PDF. Browser downloads; sha256 of downloaded bytes matches the upload's sha256.

**Expected**: Attachments survive the run row → inspector → download path. SecretRedactor doesn't strip file paths (they're not secret data; only the file contents going into `read_file` could be).

### G-14 — Filter run history end-to-end (Session 05 J-6)

**Setup**: Fresh data dir; create 50 runs across 4 workflows over a 3-week window (use a fixture or manual seed). Mix of succeeded, failed, cancelled.

**Steps**:
1. User navigates to `/runs`. Filter rail at top.
2. Default state: all date ranges off (showing all 50), all 5 status pills selected, workflow dropdown `All workflows`.
3. User clicks `Last 7 days` pill → URL updates to `?since=<-7d>`. List re-fetches; ~20 runs visible.
4. User deselects `Running` + `Queued` + `Succeeded` pills, leaving `Failed` + `Cancelled` → URL updates with `status=failed,cancelled`. List re-fetches; ~6 visible.
5. User clicks workflow dropdown → types "research" in search → selects `research-pipeline` → URL adds `workflow_id=<id>`. List re-fetches; 2 visible.
6. User clicks first run card → `/runs/{id}` shows the failed run with the error block + the trace tree's failed span.
7. User clicks browser back. Filters persist. URL still has the search params.
8. User clicks `Custom` in the date row. Two-input dialog opens. Picks a 30-day range from a month ago. Apply. List re-fetches.

**Expected**: Filter composition is AND across rails (date AND status AND workflow); within a multi-select rail, OR (any-of). URL is the source of truth. Back/forward + share-URL work cleanly.

### G-15 — Re-run with attachments (Session 05 §8.12)

**Setup**: One completed run with 2 attachments from G-13.

**Steps**:
1. User clicks `Re-run` on `/runs/{id}`.
2. `POST /v1/runs` fires with same `workflow_id` + same `task_input.text` + same `task_input.attachments` (same file_ids).
3. New run starts; navigates to `/runs/{new_id}`.
4. Trace tree builds incrementally (Session 05 polls every 2s while status=`running`).
5. On completion, Attachments accordion shows the same 2 files (same file_ids; no re-upload occurred).

**Expected**: No new `files` rows inserted (verify via SQLite `SELECT COUNT(*) FROM files`). File paths on disk unchanged.

### G-16 — Crashed-run rendering (Session 05 §8.6)

**Setup**: Inject a "crashed" run state by writing a `trace.ndjson` fixture that omits the terminal `stream_completed` marker. Insert the corresponding `runs` row with `status=failed`, `error="framework crash simulation"`.

**Steps**:
1. Navigate to `/runs/{crashed_id}`.
2. Banner renders above the trace: `Crashed during run · trace may be incomplete` in `--magenta-deep`.
3. Trace tree renders what's available (the spans that closed before the crash).
4. Clicking a span opens the drawer normally.

**Expected**: No silent failure; the user is forewarned.

## Edge-case scenarios

Must-pass; failure modes per-session checklists exercise + cross-session seams.

### E-13 — Cost calculation accuracy (Session 04 C-01)

For a known-token-count run (mocked LLM, deterministic token counts in 3 generations), compute hand-cost from the rate table. Live cost in the run row matches within 0.01¢.

### E-14 — Cost with unknown model (Session 04 C-02)

A workflow using a model NOT in `cost_rates.yaml` completes successfully. `runs.total_cost_usd=0.0`. WARN-level log entry: `cost.rate.missing: provider=foo model=bar`.

### E-15 — Run history filter with no matches

Filter `/runs` to `?status=running` when no runs are running. Empty state: `<runs status="empty_after_filter" />` with reset-filters button.

### E-16 — Workflow archived between run and inspect

1. Create a workflow; run it; archive the workflow.
2. View `/runs/{id}` — inspector still renders. "Open in canvas" link from the workflow snapshot accordion still works (canvas shows archived state).
3. `Re-run` button shows a confirmation: `The original workflow is archived. Re-run against the snapshot anyway?`

**Why this is a Bundle 02 scenario**: SP-009 immutability — the snapshot is the run's source of truth, decoupled from the live workflow. Bundle 02 exercises this seam end-to-end.

### E-17 — Re-run with stale attachment (Session 05 §10.11)

1. Run a workflow with 2 attachments. Delete one of the files via `DELETE /v1/files/{id}` (forcibly: temporarily disable the reference check OR ensure the file is no longer referenced).
2. From `/runs/{id}`, click `Re-run`.
3. Toast: `One attached file is no longer available. Re-run with remaining attachments? · [Confirm] · [Cancel]`.
4. Confirm → new run fires with only the surviving file_ids.

### E-18 — File upload size cap (Session 05 §8.7)

Upload a file = 100MB + 1 byte (or whatever exceeds the configured cap). Endpoint responds 413 with `code: "FILE_TOO_LARGE"`. No partial file on disk (verify via `ls <data-dir>/data/files/`).

### E-19 — File storage cap (Session 05 §8.7)

With 4.99GB of files already uploaded and the cap at 5GB, upload a 50MB file. Endpoint responds 413 with `code: "STORAGE_CAP_EXCEEDED"`. Then upload a 5MB file (within remaining headroom). Succeeds.

### E-20 — File deletion reference guard (Session 05 §8.9 + C-04)

1. Upload a file, attach to a run, run completes.
2. `DELETE /v1/files/{file_id}` → 409 with `code: "FILE_REFERENCED_BY_RUNS", details: {run_ids: [...]}`.
3. File still exists on disk + in DB.
4. Archive the run row (use `DELETE /v1/runs/{id}` once that endpoint ships in a future session; or manually update DB).
5. `DELETE /v1/files/{file_id}` again → 204; file deleted.

### E-21 — Trace endpoint at scale (Session 05 X-08)

A synthetic 10MB / 1000-span `trace.ndjson`. `GET /v1/runs/{id}/trace` returns the full tree within 500ms server-side; client renders within 1s for the initial paint (with virtualization if needed).

### E-22 — Attachment path traversal (Session 05 X-09)

`GET /v1/runs/{id}/artifacts/..%2F..%2Fworkflow.json` → 400. The endpoint's `pathlib.Path.resolve()` check rejects any resolved path outside the per-run artifacts directory.

### E-23 — Concurrent runs (basic stress)

Start 5 runs in parallel via `POST /v1/runs`. All succeed with distinct `run_id`s. SSE streams to 5 different clients work in parallel without cross-contamination.

### E-24 — Aggregate `/v1/runs/events` lagged subscriber (Session 04 §3)

Subscribe to `/v1/runs/events`; rapidly fire 1000 `RunUpdated` events. Subscriber's bounded queue (256 events, drop-oldest) eventually drops events with a `STREAM_LAGGED` notice in the stream. Client receives the notice, knows to re-fetch the list page wholesale to catch up.

### E-25 — Cancel-during-cleanup race

1. Click Run; click Cancel mid-run; while the 5s countdown ticks; click Cancel again on the cancel-confirm dialog (already cancelled).
2. UI is idempotent: countdown continues; second click is a no-op (or "Already cancelling" inline note).

### E-26 — SSE reconnect with stale Last-Event-ID

1. Subscribe to `/v1/runs/{id}/events`. Receive event A (id=`01J...A`).
2. Disconnect.
3. Reconnect with `Last-Event-ID: 01J...DOES_NOT_EXIST`.
4. Server falls back to cold-start replay from `trace.ndjson` beginning. No 500.

## Exploratory scenarios

Variations the agent-driven exploratory layer should try. Not exhaustive.

- **X-11 — Browser back/forward mid-run**: start a run on `/workflows/{id}`, navigate to `/runs` mid-run, hit back. SSE re-subscribes; UI catches up.
- **X-12 — Refresh during in-flight run**: start a run, refresh the page mid-run. The page re-mounts; SSE re-subscribes via Last-Event-ID; tree on `/runs/{id}` shows the spans accumulated so far.
- **X-13 — Tauri + browser tab side-by-side**: open Tauri shell AND a browser tab against the same sidecar. Click Run in one; verify the other reflects the new run on refresh (no live sync expected in v0.3 between two clients on the same run unless they both subscribe).
- **X-14 — Network throttling during file upload**: throttle to "Slow 3G"; upload a 5MB file; verify upload progress UX is smooth + retry on failure works.
- **X-15 — Dark/light theme during inspector**: toggle OS-level theme while on `/runs/{id}`. Trace tree + drawer re-renders cleanly.
- **X-16 — Resize Tauri window to extremes**: 800x600 minimum; full-screen. Inspector layout adapts; drawer becomes full-width on narrow viewports.
- **X-17 — Drag-and-drop multi-file on canvas**: drag 5 files at once; overlay shows `Drop to attach (5 files)`; all 5 upload.
- **X-18 — Trace with a very-deep span tree**: a workflow with nested branches 10 levels deep. Indentation visible; horizontal scroll if needed; keyboard nav still works.
- **X-19 — Run a workflow with a model the user hasn't configured a key for**: `POST /v1/runs` fires; run fails at the first generation with a provider auth error. Inspector renders the error; user is directed to settings.
- **X-20 — Filter rail URL share**: copy the URL from a filtered `/runs` view; paste into a new tab; same filter state.
- **X-21 — Search of attached file by partial sha256 via the API** (using the unexposed `/v1/files/{id}` route): verify the file metadata is correctly stored.
- **X-22 — Re-run a run whose workflow was edited (not archived) since**: re-run executes against the *snapshotted* workflow (per §8.14 / SP-009), not the edited live one. Verify the snapshot accordion's render matches the workflow.json — NOT the live workflow.

The agent SHOULD try things outside this list — the point is surfacing gaps the implementer didn't anticipate. Pass/fail/inconclusive per scenario, with screenshots/logs into `tmp/spren/v0.3.0/02-run-execution-and-inspection/testing/exploratory-report.md`.

## Per-layer guidance

### Cross-session scripted (Playwright + tauri-driver + httpx)

**Owns**: G-08..G-16, E-13..E-26 (where automatable), exploratory ones with deterministic input.

**Tooling split**:
- httpx (Python) — E-13, E-14, E-18, E-19, E-20, E-21, E-22, E-23, E-24, E-26 (API-only flows; no UI)
- Playwright (chromium) — G-08..G-16 (most), E-15, E-17, X-11..X-22 (UI)
- tauri-driver / WebdriverIO — Tauri-shell scenarios (G-08, G-13, G-14 in Tauri-shell variant); skipped on WSL2

**Run as**: `apps/web/tests/bundle/02-*.spec.ts` (Playwright) + `packages/spren/tests/bundle/test_02_*.py` (httpx), invoked via `just bundle-02`. Layer 1's full pass + green is a precondition to invoking Layer 2.

**LLM-touching tests**: G-08, G-09, G-13, G-15 + most exploratory. Use `OPENROUTER_API_KEY` for live LLM calls OR a known-output fake-provider fixture (cassette-recorded; deterministic on replay).

### Agent-driven exploratory

**Owns**: All G + E scenarios (re-runs them; cross-checks against scripted), all X scenarios, plus its own variations.

**Specific scripted variations the agent should try first** (before improvising):
- Run a workflow with empty `task_input.text` AND empty `attachments` (the Session 04 default). Verify the agents handle empty input gracefully (Researcher might ask `What should I research?` via user_interaction — out of scope for v0.3 so the run might just complete with a "no input" response).
- Upload a 0-byte file (empty file). Verify the file lands on disk + the row stores `size_bytes=0`.
- Upload a file with a Unicode filename (e.g., `エッセイ.txt`). Verify the sanitization preserves the user-facing original_name on download.
- Run two workflows in quick succession; observe whether the orb's reactive state correctly reflects the *most-recent* run (no cross-contamination from the earlier still-running one).
- Stress: open 10 browser tabs all to `/runs`; observe aggregate SSE handles the fan-out.

**Tooling**: `mcp__claude-in-chrome__*` for the browser layer; `mcp__claude-in-chrome__tabs_context_mcp` for state inspection. For Tauri, agent invokes the binary via subprocess + drives via tauri-driver.

**Output**: `tmp/spren/v0.3.0/02-run-execution-and-inspection/testing/exploratory-report.md` with per-scenario pass/fail/inconclusive + screenshots/logs/traces + at least 5 implementer-not-anticipated paths with results.

### Visual regression

**Owns**: Bundle 02's snapshot set (extends Bundle 01's baseline).

**Key screens to snapshot**:
1. Canvas with Run button `idle` (Bundle 01 baseline, re-confirmed)
2. Canvas with Run button `submitting` (mid-200ms-spinner)
3. Canvas with Run button `running` (showing `Cancel · Ns · Nt`)
4. Canvas with cancel-confirm inline dialog open
5. Canvas with `Cancelling… 3` countdown active
6. Canvas with `📎(0)` icon (empty state)
7. Canvas with `📎(2)` icon + attachment popout open
8. Canvas with drag-and-drop overlay active
9. Completion toast (succeeded, failed, cancelled variants)
10. `/runs` filter rail at default
11. `/runs` filter rail at `Last 7 days + Failed,Cancelled + research-pipeline`
12. `/runs/{id}` succeeded inspector (full tree + drawer closed)
13. `/runs/{id}` succeeded inspector (drawer open on a generation span)
14. `/runs/{id}` succeeded inspector (drawer open on a tool span with `[REDACTED]` args)
15. `/runs/{id}` failed inspector with error banner
16. `/runs/{id}` crashed-run inspector with "Crashed during run" banner
17. `/runs/{id}` workflow-as-run accordion expanded
18. `/runs/{id}` attachments accordion expanded (2 files)
19. `/runs` empty-state after filter narrows to no matches

Snapshots stored in `apps/web/tests/e2e/__screenshots__/bundle-02/`. Per Bundle 01's Playwright-native idiom (not Argos), diffs are local; an explicit "accept as new baseline" step is required to update.

### User manual smoke (≤6 items, ≤5 minutes)

The user runs through this personally on a fresh data dir + fresh `just install && just build`. Bundle 02 ships only on 100% pass.

- [ ] **U-08** — Open canvas; click Run on a 3-agent workflow; orb shifts to `thinking` then `speaking`; completion toast appears with cost; click through to `/runs/{id}` and see the full trace tree with at least 5 spans + correct token counts
- [ ] **U-09** — Click Cancel mid-run; confirm in the inline prompt; watch the 5-second countdown; run terminates as `cancelled`; toast confirms
- [ ] **U-10** — Drag a PDF onto the canvas; `📎(1)` appears; click Run; on completion, open `/runs/{id}` and see the file in the attachments accordion; download it; file bytes match
- [ ] **U-11** — Navigate to `/runs`; set filter to "Last 7 days + Failed"; verify only failed runs from the last week are visible; click one; see the error block + failed span on the inspector
- [ ] **U-12** — Click `Re-run` on a completed run; new run starts; trace tree builds incrementally as spans close; new run completes successfully with the same attachments referenced (verify file IDs match via SQL or by re-opening the new run's attachments accordion)
- [ ] **U-13** — Refresh the browser tab during an in-progress run; SSE re-subscribes; tree + token counter catch up to current state within a few seconds; no events appear lost

If any item fails: bundle does not ship. Investigate, fix, re-run.

## Cross-bundle dependencies

- Bundle 01 (visual-builder) ships the workflows Bundle 02 runs + the design system Bundle 02 reuses + the canvas Run button shell Session 04 wires + the canvas `+ Pattern` modal Bundle 02 leaves alone.
- Framework Session 06 (AG-UI translator) lands before Bundle 02 implementation begins.
- Framework Session 01 (NDJSON streaming tracing) is shipped — the trace.ndjson file backs Session 05's `GET /v1/runs/{id}/trace` + the SSE reconnect-replay path.
- Framework Session 02 (`TelemetrySink` + `SecretRedactor`) is shipped — the redactor runs at the framework's fan-out boundary; Session 05 renders pre-redacted spans without re-redacting.
- Framework Session 03 (pause/resume) is shipped — but pause/resume itself is v0.4 in Spren (`v0.4-29`); Bundle 02 doesn't expose pause/resume UX.
- Framework Session 04 (workflow serializer) is in-flight on `marsys-tracing-work`; Spren Session 02's 24-hour follow-up cleanup PR (drop the local mirrors) ships when Framework 04 merges; not blocking Bundle 02.

# Bundle A — Visual Workflow Builder — Test Scenarios

**Bundle**: A — Visual workflow builder
**Sessions**: 01 (Foundation), 02 (Workflow CRUD + types + Python import), 03 (Visual builder)
**Demo-able outcome**: User installs the umbrella in dev mode, opens the Tauri shell, builds a 3-agent workflow visually with `@xyflow/react`, saves it, sees it in the list with `provenance=visual_builder`. User also imports a workflow from a `.py` file. Build → save → reload → workflow round-trips intact. Lint catches missing tool/agent refs inline.

## Overview

This file is the source of truth for Bundle A's bundle-end real-world test. Four testing layers consume it at the bundle boundary:

1. **Cross-session scripted integration** — Playwright (browser GUI) + tauri-driver/WebdriverIO (Tauri shell) + httpx (API-only flows). Runs in CI; deterministic. Covers golden-path + edge-case scenarios that benefit from automation.
2. **Agent-driven exploratory** — Claude Code with `mcp__claude-in-chrome__*` reads this file and runs through golden-path + edge-case + exploratory scenarios, plus variations of its own. Produces `tmp/spren/v0.3.0/01-visual-builder/testing/exploratory-report.md`.
3. **Visual regression** — Argos snapshots key screens. Bundle A captures the first baseline; Bundles B onward diff against it.
4. **User manual smoke** — 5-min checklist on a fresh data dir + fresh install. Bundle ships only on 100% pass.

Coverage approach is layered: golden path first (must-pass; demo-able outcome), then edge cases (failure modes the per-session manual-verify checklists already cover, plus the cross-session seams those checklists don't), then exploratory variations (the things implementers don't think to test).

The cross-session seams are the load-bearing focus. Per-session tests cover their own session in isolation; Bundle A's test exercises the joins:

- The auth token from Session 01's bootstrap flows correctly through Session 02's `/v1/workflows` endpoints
- Session 01's `BootstrapResponse.endpoints` map carries the workflow URL Session 02 added; the frontend reads it without hard-coding
- Session 01's `_webui/` build trick still serves the Session 02 + 03 placeholder/real UI when run via `just dev-desktop` after `just build`
- The Tauri stdin-shutdown protocol (Session 01 fixup) cleanly drains in-flight `/v1/workflows` requests from Session 02 / 03
- The Pydantic-derived TS types from Session 02 compile in the Session 03 React tree
- Session 03's canvas writes a workflow whose stored `definition` round-trips through Session 02's GET endpoint without lossy serialization

## Setup steps

Every scenario assumes a clean baseline unless stated otherwise. The "fresh-state" preamble:

```bash
# 1. Start from a clean checkout of feature/spren-umbrella (or the v0.3 release tag once cut)
cd /path/to/marsys-spren-work
git status                                      # clean tree

# 2. Wipe per-user data dir for spren (keeps your real data safe; bundle tests run against an empty store)
DATA_DIR="$(uv run --package spren python -c 'import platformdirs; print(platformdirs.user_data_dir(\"spren\"))')"
rm -rf "$DATA_DIR"

# 3. Confirm dev ports are free
ss -ltn | grep -E ':(8765|5173)\b' && echo "PORT IN USE — kill before continuing" && exit 1

# 4. Provide an LLM key for any LLM-touching tests (Bundle A does not exercise live runs;
#    Session 02's Python-import endpoint reads no LLMs. Bundle B is the first bundle that needs this.)
export OPENROUTER_API_KEY="<key-or-skip>"        # not required for Bundle A scenarios

# 5. Install + build
just install                                    # uv sync + pnpm install + cargo fetch + cargo install tauri-cli
just build                                      # vite build + dist guard + copy to packages/spren/src/spren/_webui/
just test                                       # baseline must be green: framework 841/764, spren 21+, tui 2, web 1, e2e 2, cargo 7
```

If `just test` is not green at the baseline counts, **stop**. Bundle-end testing assumes the per-session tests pass.

For the Tauri-shell scenarios, native macOS / Windows / Linux is required — WSLg's webkit2gtk integration is broken (Session 01 lessons). On WSL2, run only the browser-GUI + httpx layers; mark Tauri scenarios `inconclusive — environment` rather than `pass`.

The scripted integration layer is invoked via a `just bundle-a` recipe that ships alongside the integration suite (`apps/web/tests/bundle/` for Playwright + `packages/spren/tests/bundle/` for httpx). The recipe lands when the suite is first authored; it is not part of Session 01 or 02's per-session deliverables.

## Golden-path scenarios

Must-pass on every bundle-end run. These cover the demo-able outcome end-to-end.

### G-01 — Tauri shell launches and renders bootstrap (Session 01 seam)

**Setup**: Fresh data dir per preamble.

**Steps**:
1. `just dev-desktop`
2. Tauri window opens, sized 1280x800
3. Sidecar visible in `ps -ef | grep 'python -m spren'`
4. DevTools network panel shows `GET /healthz` → 200 (no auth) and `GET /v1/bootstrap` → 200 with the BootstrapResponse JSON
5. The placeholder UI renders the bootstrap content (framework version, spren version, surfaces, started_at, data_dir)

**Expected**: No CORS error in console. No auth failure. Window paints within 5 s of `just dev-desktop` start.

**Edge variations to vary**: dark/light mode toggle (OS-level); resize window to minimum + maximum.

### G-02 — Browser GUI loads with token-fragment auth (Session 01 seam)

**Setup**: `just dev` running.

**Steps**:
1. Read the sidecar's stdout — record `spren-ready: port=<N> token=<T>`
2. Navigate browser to `http://localhost:5173/#token=<T>`
3. Page renders the bootstrap content
4. URL fragment is stripped after first read (`window.location.href` no longer contains `#token=`)
5. Refresh the page → "auth required" message (token was consumed)

**Expected**: First visit succeeds; refresh produces explicit auth-required state, NOT a silent failure.

### G-03 — Workflow CRUD round-trip via REST (Session 02)

**Setup**: `just dev` running, token captured.

**Steps**:
1. `POST /v1/workflows` with a 2-agent definition → 201 with generated ULID, `created_at`, `updated_at`, `provenance="api"`
2. `GET /v1/workflows` → 200 with the new workflow in `items`; `next_cursor=null`
3. `GET /v1/workflows/{id}` → 200; matches POST response
4. `PUT /v1/workflows/{id}` with new definition → 200; `updated_at` is later than `created_at`; `created_at` unchanged
5. `PATCH /v1/workflows/{id}` with `{"is_archived": true}` → 200
6. `GET /v1/workflows` → empty `items`; `GET /v1/workflows?archived=true` → 200 with the workflow
7. `DELETE /v1/workflows/{id}` → 204
8. `GET /v1/workflows/{id}` → 404 with `{"error": {"code": "WORKFLOW_NOT_FOUND", ...}}`

**Expected**: All operations idempotent on retry with `Idempotency-Key` header; cursor is null with one-item list.

### G-04 — Python file import round-trip (Session 02)

**Setup**: `just dev` running, token captured. Use `packages/spren/tests/fixtures/python_workflows/valid_minimal.py` as the source file.

**Steps**:
1. `curl -X POST -F "file=@valid_minimal.py" -H "Authorization: Bearer $TOKEN" http://localhost:8765/v1/workflows/import-python` → 201
2. Response carries `provenance="code_import"` and `provenance_metadata={"source_filename": "valid_minimal.py", "sha256": "<hex>"}`
3. `GET /v1/workflows/{id}` → returns the same `definition` shape
4. SQLite query on `<data-dir>/data/spren.db`: `SELECT id, name, provenance FROM workflows WHERE id=...` → row exists with `provenance='code_import'`

**Expected**: Topology + agents + execution config from the `.py` file appear in the stored workflow definition.

### G-05 — Cross-session bootstrap-to-workflows wiring (Session 01 + 02 seam)

**Setup**: `just dev` running, token captured.

**Steps**:
1. `GET /v1/bootstrap` → response includes `endpoints.workflows == "/v1/workflows"`
2. Frontend reads this endpoint URL from the bootstrap response (NOT a hardcoded string in `lib/api.ts`)
3. Frontend lists workflows by calling that URL
4. Verify via `grep -rn '"/v1/workflows"' apps/web/src/` — references appear only as type literals or in tests, not as hardcoded fetch targets

**Expected**: API contract for the workflow endpoint is discoverable from `/v1/bootstrap`; clients don't hardcode paths.

### G-06 — `_webui/` production build flow (Session 01 + 02 seam)

**Setup**: `just clean && just install && just build`.

**Steps**:
1. After `just build`, `apps/web/dist/index.html` exists
2. `packages/spren/src/spren/_webui/index.html` exists (copied from dist by the build recipe)
3. `uv run --package spren python -m spren --port 8765` (no Vite dev server)
4. `curl -I http://127.0.0.1:8765/` → 200 (StaticFiles mount serves the built bundle)
5. `curl -L http://127.0.0.1:8765/#token=$TOKEN` → HTML containing the placeholder workflow UI's content (Session 02's UI; Session 03 replaces with the real builder)
6. The bundled UI renders workflows from `/v1/workflows` (proves the production-mode wiring works without the Vite dev server)

**Expected**: One-process production mode (sidecar serves both API and UI) functions identically to the two-process dev mode.

### G-07 — Visual builder round-trip (Session 03 — placeholder, fill at Session 03 ship time)

**Setup**: Tauri shell open via `just dev-desktop`, fresh data dir.

**Steps** (TBD when Session 03 ships):
1. Open the canvas route
2. Drag agent nodes onto the canvas, connect them with edges, configure agent properties via the agent form
3. Apply a pattern preset (e.g., supervisor-with-team)
4. Save → workflow appears in the list with `provenance="visual_builder"`
5. Reload the page; open the saved workflow; canvas renders the same topology + agent config

**Expected**: Workflow definition round-trips through SQLite + the canvas without losing layout, edge metadata, or agent fields.

### G-08 — Tauri shutdown drains sidecar cleanly (Session 01 fixup seam, Session 02 in-flight)

**Setup**: `just dev-desktop` running; `apps/web` page renders the Session 02/03 workflow list.

**Steps**:
1. Kick off a `POST /v1/workflows` from the UI (or a deliberately slow `POST /v1/workflows/import-python` with a 900 KB `.py` fixture)
2. While the request is in flight, close the Tauri window
3. Tauri sends `shutdown\n` to sidecar stdin; sidecar finishes the lifespan-shutdown handler
4. `ps` shows no orphaned `python -m spren` after 2 s
5. Re-open via `just dev-desktop`; `GET /v1/workflows` shows the workflow created in step 1 (the in-flight request completed before shutdown)

**Expected**: Graceful drain; no SIGKILL fallback fires; no data loss.

## Edge-case scenarios

Must-pass; failure modes the per-session checklists already exercise, plus cross-session seams.

### E-01 — Auth missing or wrong on every Session 02 endpoint

For each of `/v1/workflows` (GET, POST), `/v1/workflows/{id}` (GET, PUT, PATCH, DELETE), `/v1/workflows/{id}/lint`, `/v1/workflows/import-python`:

- No `Authorization` header → 401 (NOT 422)
- Wrong token → 401
- Correct token → 200/201/204 depending on verb

**Why this is a Bundle A scenario, not Session 02**: confirms Session 01's `make_auth_dependency` factory is correctly applied at router level (per the established pattern, NOT in handler signatures). A regression where Session 02 used `Annotated[None, Depends(require_auth)]` would surface as 422 not 401 and pass per-session tests if they only assert "non-2xx".

### E-02 — CORS seam from each origin

Preflight `OPTIONS /v1/workflows` from each origin:
- `http://127.0.0.1:5173` → 200, allow-origin echoes
- `http://localhost:5173` → 200
- `tauri://localhost` → 200
- `http://malicious.example` → no `Access-Control-Allow-Origin` header (browser blocks)

**Why this is a Bundle A scenario**: catches a regression where Session 02 added per-route CORS overrides instead of relying on Session 01's app-level regex.

### E-03 — Sidecar restart preserves persisted workflows

1. Create 5 workflows via `POST /v1/workflows`
2. Kill the sidecar (Ctrl-C in `just dev`)
3. Restart `just dev`
4. New token issued; new sidecar instance
5. `GET /v1/workflows` with the new token → 200 with all 5 workflows

**Expected**: Per-launch token is regenerated. Workflows persist in `<data-dir>/data/spren.db`. No migration re-run (verify via `_migrations` table; no duplicate row).

### E-04 — Migrations idempotent across daemon restarts

1. Fresh data dir; start sidecar; sidecar runs first migration; `_migrations` table has one row
2. Restart sidecar; `_migrations` table still has one row (NOT two)
3. Inspect `workflows` table schema via `PRAGMA table_info(workflows)`; columns include `provenance` and `provenance_metadata`

### E-05 — Python import rejection paths (Session 02)

For each forbidden construct, the importer returns 422 with `error.code="PYTHON_IMPORT_REJECTED"` and a clear `details.reason`:
- `invalid_dict_dsl.py` (uses `{"Start -> Researcher": ...}`) → reason `dict_dsl_unsupported`
- `invalid_exec.py` (contains `exec(...)`) → reason `exec_forbidden`
- `invalid_dynamic_topology.py` (builds edges via comprehension) → reason `comprehension_forbidden`
- `invalid_too_large.py` (1.5 MB) → reason `file_too_large`
- A non-UTF-8 file → reason `encoding_unsupported`
- A binary `.bin` renamed to `.py` → reason `decoding_failed`

**Expected**: No partial workflow row inserted on rejection (verify with `SELECT COUNT(*) FROM workflows`).

### E-06 — Idempotency-Key replay window

1. `POST /v1/workflows -H "Idempotency-Key: abc"` body X → 201 with workflow A
2. Same `POST` with same key + body within 24 h → 201 with the cached response (workflow A; NOT a new workflow B)
3. SQLite confirms only one workflow exists
4. Same key + DIFFERENT body → 409 with `IDEMPOTENCY_KEY_MISMATCH` (or per spec)

### E-07 — Cursor pagination across boundaries

1. Create 25 workflows
2. `GET /v1/workflows?limit=10` → 10 items, non-null `next_cursor`, `has_more=true`
3. `GET /v1/workflows?cursor=<next>&limit=10` → next 10 items
4. `GET /v1/workflows?cursor=<next>&limit=10` → final 5 items, `next_cursor=null`, `has_more=false`
5. Tampered cursor (alter one base64 char) → 400 with `CURSOR_INVALID`

### E-08 — Filter by provenance + archived

1. Create workflows with mixed provenance: 2× `api`, 1× `code_import` (via Python import endpoint), 1× `visual_builder` (via Session 03 UI; n/a until Session 03 ships — until then mark via direct API set)
2. `GET /v1/workflows?provenance=code_import` → only the code_import row
3. Archive 1 workflow; `GET /v1/workflows` excludes it; `GET /v1/workflows?archived=true` includes it
4. Combine: `GET /v1/workflows?provenance=api&archived=true` → only archived api workflows

### E-09 — TS type generation pipeline (Session 02 seam)

1. After `just build`, `apps/web/src/lib/api-types.generated.ts` and `apps/web/src/lib/types.generated.ts` exist + non-empty
2. `pnpm --filter @marsys/spren-web typecheck` passes (the generated types compile under TS 6)
3. `grep -rn 'interface Workflow\|type Workflow' apps/web/src/lib/` shows the generated types as the source — NO hand-written mirror in `api.ts` after Session 02's final cleanup commit

### E-10 — Playwright collision guard (Session 01 fixup)

1. `just dev` running on port 5173
2. In another terminal: `pnpm --filter @marsys/spren-web exec playwright test`
3. Preflight script aborts with the documented message: `Vite dev server is already running on 127.0.0.1:5173 ...`
4. After stopping `just dev`, the same Playwright command succeeds normally

### E-11 — Windows shell parity (Session 01 fixup)

On a Windows 11 host with PowerShell:
1. `just install` completes
2. `just dev` starts both processes (PowerShell-based recipe variant)
3. `just test` runs all suites green
4. `just build` produces `_webui/` with the Windows-shell `Test-Path` guard

**Expected**: All recipes function identically to the Unix variants.

### E-12 — Concurrent CRUD requests (basic stress)

1. Spawn 20 concurrent `POST /v1/workflows` requests
2. All succeed with distinct ULIDs
3. SQLite contains 20 rows
4. `GET /v1/workflows` returns 20 items consistently across re-fetches

**Expected**: SQLite WAL mode handles concurrent writes; ULIDs collision-free.

## Exploratory scenarios

Variations the agent-driven exploratory layer should try. Not exhaustive; agent should add its own.

- **X-01 — Browser back/forward mid-CRUD**: create a workflow, navigate away, hit back. Workflow visible? Form state lost gracefully?
- **X-02 — Refresh during in-flight POST**: start a workflow create, refresh the page mid-request. Did the workflow get created? Is the UI consistent on reload?
- **X-03 — Tauri + browser tab side-by-side**: open the Tauri shell AND a browser tab against the same sidecar. Create a workflow in one; does the other reflect it on refresh? (No live sync expected in v0.3 — verify behavior is deterministic.)
- **X-04 — Network throttling**: Chromium DevTools throttle to "Slow 3G"; CRUD operations should succeed (eventually) without UI desync; loading states should be visible.
- **X-05 — Dark/light/system theme**: toggle OS-level theme mid-session. UI re-renders without breaking. (Argos visual regression ALSO catches this.)
- **X-06 — Resize Tauri window to extremes**: 800x600 minimum and full-screen maximum. Layout doesn't break; canvas (Session 03) remains usable.
- **X-07 — Clipboard paste of malformed JSON in the placeholder definition textarea (Session 02)**: paste `{"foo":` (truncated) → form submission produces a clear validation error, not a 500.
- **X-08 — Upload a .py file with a BOM**: Python import endpoint should accept (or reject with a clear reason if BOM is forbidden); no 500.
- **X-09 — Slow disk on data dir**: simulate via `chmod 444` or a fuse FS with latency. Migrations + workflow writes either succeed or fail with a clear error; no silent corruption.
- **X-10 — Token rotation mid-session**: kill sidecar, restart, copy new token into the existing browser's URL fragment, refresh. UI reconnects cleanly.

The agent-driven layer SHOULD also try things outside this list — the point is to surface gaps the implementer didn't anticipate. Pass/fail/inconclusive per scenario, with screenshots/logs into `tmp/spren/v0.3.0/01-visual-builder/testing/exploratory-report.md`.

## Per-layer guidance

### Cross-session scripted (Playwright + tauri-driver + httpx)

**Owns**: G-01..G-08 (where automatable; G-07 lands when Session 03 ships), E-01..E-04, E-06..E-09, E-12.

**Tooling split**:
- httpx (Python) — E-01, E-03, E-04, E-06, E-07, E-08, E-12 (API-only flows; no UI)
- Playwright (chromium) — G-02, G-05, G-06, parts of G-03/G-04 via the placeholder UI
- tauri-driver / WebdriverIO — G-01, G-08 (native-shell + lifecycle scenarios; skipped on WSL2)

**Run as**: a new `apps/web/tests/bundle/` (Playwright) + `packages/spren/tests/bundle/` (httpx) suite, invoked via a new `just bundle-a` recipe. Layer 1's full pass + green is a precondition to invoking Layer 2.

**LLM-touching tests**: none in Bundle A (no runs yet — runs come in Bundle B).

### Agent-driven exploratory

**Owns**: All G + E scenarios (re-runs them; cross-checks against scripted), all X scenarios, plus its own variations.

**Specific scripted variations the agent should try first** (before improvising):
- Submit empty workflow definition `{}` via POST and via the placeholder UI form
- Upload a `.py` file with a syntax error
- Send `Authorization: Bearer ` (empty token) on every endpoint
- Open the Tauri window, navigate to an unknown route (`/foo`); does TanStack Router's not-found render correctly?
- Disable JavaScript in the browser tab; the auth-required message should still render (or at least the page should not 500)

**Tooling**: `mcp__claude-in-chrome__*` for the browser layer; `mcp__claude-in-chrome__tabs_context_mcp` for state inspection. For Tauri, the agent invokes the binary via subprocess + drives via tauri-driver.

**Output**: `tmp/spren/v0.3.0/01-visual-builder/testing/exploratory-report.md` with per-scenario pass/fail/inconclusive + screenshots/logs/traces + at least 5 implementer-not-anticipated paths attempted with results.

### Visual regression (Argos)

**Owns**: First-baseline capture for Bundle A. Diffs flagged from Bundle B onward.

**Key screens to snapshot**:
1. Bootstrap rendered (browser GUI, light theme, default viewport 1280x800)
2. Bootstrap rendered (browser GUI, dark theme)
3. Tauri window rendered (light theme; if WSL2 host, this snapshot is captured on the macOS/Windows/Linux native runner only)
4. Workflow list — empty state (Session 02 placeholder UI)
5. Workflow list — 3 workflows with mixed provenance badges
6. Auth-required message
7. (Session 03) Canvas — empty
8. (Session 03) Canvas — 3-agent topology rendered
9. (Session 03) Agent config form — open
10. (Session 03) Lint issues panel — with at least one issue rendered

Snapshots 1–6 are captured at Bundle A end with Sessions 01 + 02 shipped; 7–10 land when Session 03 ships and the Bundle A baseline is updated before Bundle B starts.

### User manual smoke (≤6 items, ≤5 minutes)

The user runs through this personally on a fresh data dir + fresh `just install && just build` run. Bundle A ships only on 100% pass.

- [ ] **U-01** — `just dev-desktop` opens a Tauri window within 10 seconds; bootstrap content visible; no console errors
- [ ] **U-02** — Create a workflow via the Session 02 placeholder UI; refresh page; workflow still listed with `provenance=api` badge
- [ ] **U-03** — Upload a known-good `.py` file from `packages/spren/tests/fixtures/python_workflows/valid_minimal.py`; workflow appears with `provenance=code_import` badge
- [ ] **U-04** — Upload `invalid_exec.py`; clear error message in UI; no row inserted
- [ ] **U-05** — (Session 03 — fill at ship) Build a 3-agent workflow on the canvas; save; reload; canvas renders same topology
- [ ] **U-06** — Close the Tauri window; sidecar exits within 2 seconds (verify via `ps` if comfortable; otherwise just confirm window closes cleanly without hang)

If any item fails: bundle does not ship. Investigate, fix, re-run.

## Session 03 placeholder

**Section currently covers Sessions 01 + 02 only. Session 03 visual builder scenarios will be filled in when Session 03 ships.**

The skeleton above includes Session 03 placeholders at G-07 (canvas round-trip), U-05 (manual smoke build-and-reload), and visual-regression snapshots 7–10. When Session 03's brief is finalized, the following scenario categories will land:

- **Canvas drag-drop**: dragging an agent palette item onto the canvas; multi-select drag; auto-layout
- **Edge connection**: connecting two nodes; rejecting an invalid connection (cycle, type mismatch); deleting an edge
- **Agent config form**: opening the form on a node click; field validation (model picker, tools list, instruction); save flows back to the canvas
- **Pattern preset insertion**: inserting a supervisor-with-team or research-pipeline preset into an empty canvas; into a non-empty canvas
- **Semantic linter UI**: workflow with a missing tool reference shows an inline lint marker; clicking the marker opens the issue detail; fixing the issue clears the marker
- **Provenance badges in the list**: visual_builder, code_import, api badges render distinctly
- **Theme + design system**: dark/light renders cleanly; cmdk command palette opens via keyboard shortcut; Geist font loaded; radix primitives behave correctly

Each category becomes 2–4 concrete scenarios when Session 03 lands. The overall structure (golden-path + edge-case + exploratory + per-layer ownership) carries over.

Until Session 03 ships, **Bundle A's bundle-end test does NOT gate the bundle's progress** — Bundle A is by definition complete only when Session 03 ships. Sessions 01 + 02 by themselves do not ship a bundle; they ship the foundation Session 03 builds on.

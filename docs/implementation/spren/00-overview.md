# 00 — Implementation Overview

## Version map

| Version | Outcome | Doc |
|---------|---------|-----|
| **v0.3 — MVP** | Single user installs Spren via the native installer (front door) or any secondary channel (brew / winget / apt / npm / pipx / Docker). Opens the desktop GUI (Tauri webview) or the same UI in a system browser tab. Talks to a continuously-active local meta-agent on the home page (NOT a chat box — a four-surface command center). Builds 3-agent workflows visually OR imports them from a marsys `.py` file. Runs workflows with file uploads, watches live traces, reviews run history, edits memory in `vim`. Meta-agent has read-only + low-risk-write authority with hard rails on destructive actions. NO sub-instance spawning yet, NO teams, NO channels, NO vector recall, NO daily consolidation, NO TUI, NO Tauri-polish features (tray / autostart / OS notifications) — those are v0.4. | [`v0.3-mvp.md`](./v0.3-mvp.md) |
| **v0.4 — TUI, Tauri Polish, Always-on Org & Channels** | Textual TUI client (`spren tui`). Tauri tray + native menus + autostart + OS notifications + deep links + file associations. Sub-instance spawning + team managers + full skills catalog. Tier 4 hybrid index + daily consolidation pipeline + volatility-based re-verification. Cron, webhooks, `cloudflared` tunnel. Telegram polling + Discord gateway. Workflow templates + import/export. `TelemetrySink` framework PR + `SprenTelemetrySink` adapter (in `packages/spren/src/spren/telemetry/`). Standing approvals. Advanced semantic linter. | [`v0.4-extensions.md`](./v0.4-extensions.md) |
| **v0.5 — Tunnels, Cloud, Multi-user** | Slack + WhatsApp (tunnel-required messengers). Documented cloud deployment. Code-generated tools with user approval. Optional multi-user mode if demand justifies. | [`v0.5-future.md`](./v0.5-future.md) |

## v0.3 sessions — 10 substantial sessions

Each session ships a coherent capability that stands alone. Session N+1 may depend on Session N being done.

| # | Title | Outcome | Phase coverage |
|---|-------|---------|----------------|
| 01 | **Foundation** ✅ shipped | Restructure umbrella repo into `packages/framework/` (existing marsys code relocated) + `packages/spren/` (new, FastAPI server) + `apps/web/` (Vite + React + TanStack Router skeleton, web pkg `@marsys/spren-web`) + `apps/desktop/` (Tauri 2 shell that spawns the Python sidecar via `std::process::Command`) + `apps/tui/` (placeholder); uv workspace (explicit member list) + pnpm workspace + Cargo workspace + `Justfile`; minimal FastAPI server with `/healthz` + `/v1/bootstrap` + per-launch auth (`make_auth_dependency` factory + route-level `dependencies=[Depends(...)]`) + CORS regex (localhost-any-port + `tauri://localhost`); `just dev` runs sidecar + Vite; `just dev-desktop` runs Vite + Tauri (Tauri spawns sidecar internally); framework regression tests pass with zero changes after relocation. **Tailwind / `@xyflow/react` / cmdk / radix-ui / shadcn / MSW DEFERRED to Session 03.** | A1 + A2 |
| 02 | **Workflow CRUD + types + Python import** ✅ shipped | SQLite `workflows` table + forward-only migrations runner (with `provenance` field); Pydantic schemas for Workflow / WorkflowDefinition mirroring marsys topology types; full REST CRUD; `POST /v1/workflows/import-python` parses a marsys-framework `.py` file upload and materializes a workflow record (provenance: `code_import`); OpenAPI → `openapi-typescript` → TS types consumed by `apps/web`; integration tests against real SQLite | A3 |
| 03 | **Visual builder** ✅ shipped | Vite + React + TanStack Router with the "Sunrise" design system (Geist Sans/Mono, cmdk shell); workflow list/edit UI with provenance badges; `@xyflow/react` canvas with custom node types per `NodeType`; right-rail agent config form; pattern preset insertion (HUB_AND_SPOKE, PIPELINE, HIERARCHICAL, MESH); topology compile + semantic linter with inline issue display | B1+B2+B3 |
| 04 | **Run execution + tracing** | `EventBus → AG-UI translator` consumer (translator itself ships framework-side); `POST /v1/runs` background-task execution calling `Orchestra.run()`; `GET /v1/runs/{id}/events` SSE stream; cost rate table YAML + per-run aggregation | C1+C2 |
| 05 | **Run inspection** | `POST /v1/files` multipart upload + `<data-dir>/data/files/{file_id}/` storage + `file_id` referencing in run task input; run history backend (cursor-paginated REST list + filters) + run history list UI; trace viewer UI (Langfuse-idiom nested spans, token/latency/cost chips) | C3+D1+D2 |
| 06 | **Memory subsystem** | `SandboxFilesystem` permission wrapper (application-layer, role-tiered); session log Tier 2 (SQLite `events` + JSONL daily files + FTS5); markdown KB scaffold (`<data-dir>/sandbox/shared/memory/` with default `personas/main.yaml`, `rules.md`, `active_context.md`, `active_todos.md`); deterministic markdown → SQL indexer (file watcher + parser + UPSERT, no LLM); pending facts queue + supersession algorithm (5-step deterministic flow with `disputed` state); `spren memory` CLI | E1+E2+E3+E4 |
| 07 | **Meta-agent core** | Spren daemon + EventIngress + per-agent priority inbox (P0–P3) + chunked thinking at tool-call boundaries; APScheduler-backed time scheduler for one-off + recurring events; v0.3 system watcher catalog (Budget / Disk / ProviderHealth / PendingFacts / IndexDrift / WorkflowRun); heartbeat (default 30 min); graceful shutdown drains inbox; durable inbox replay on crash; main agent loop + six-axis system prompt assembly | F1+F2 |
| 08 | **Meta-agent capabilities** | Read tools (`list_workflows`, `read_workflow`, `read_run`, `read_trace`, `read_file`, `grep`, `lookup_facts`); write tools (`create_workflow`, `update_workflow`, `archive_workflow`, `add_run_note`, `set_active_context`); suggest-with-confirm flow + hard-rail tool list; per-day budget cap + per-think token cap + cheap/expensive model selection; `run_workflow` tool + manual trigger via meta-agent | F3+F4 |
| 09 | **Home page UI** | Home route (`/`) with four-surface command center: Now / Since-you-were-away / Activity / Chat input; suggestion confirm flow; activity stream; chat input → meta-agent SSE roundtrip; cost meter + budget status in header | G1+G2 |
| 10 | **Polish + native distribution + release** | API key management (OS keychain primary, encrypted SQLite fallback); first-run onboarding (prompt-to-scaffold drafts a workflow from user description); light + dark theme polish; PyInstaller sidecar build matrix; Tauri 2 installer build (.dmg / .msi / .deb / .AppImage); `install.sh` script + manifest server scaffolding; secondary-channel packaging (brew tap, winget manifest, npm wrapper, pipx wheel for server-only mode, Docker image); code-signing & notarization in CI; basic Tauri auto-updater; E2E test suite (Playwright + tauri-driver) | H1+H2+H3+I1 |

## Why 10 sessions instead of 25 or 40

Each plan-mode session has overhead: implementer reads context, asks questions, plans, implements, tests, reviews. Sessions narrower than ~1000 lines of code/tests don't justify that overhead. 10 substantial sessions with clear logical scope minimizes that overhead while keeping each session digestible (each is roughly one focused work-block, not a multi-day epic).

## Dependency graph

```
01 (foundation)
   ↓
02 (workflow CRUD + types)
   ↓
03 (visual builder) ─── independent of 04+05 thereafter
   ↓
04 (run execution + tracing)
   ↓
05 (run inspection: files + history + trace viewer)
   ↓
06 (memory subsystem) ─── could parallelize with 04+05 in principle, but easier to sequence
   ↓
07 (meta-agent core) ─── depends on 06 for sandbox, session log, KB
   ↓
08 (meta-agent capabilities) ─── depends on 02 for write-tool targets
   ↓
09 (home page UI) ─── depends on 07+08 for the chat-SSE backend
   ↓
10 (polish + distribution + release)
```

Sessions 03 and 04+05 are independent of each other; in principle a multi-implementer team could parallelize them. For a single-implementer rollout, sequence 01→02→03→04→05→06→07→08→09→10.

## Open strategic questions before v0.3 ships

- **Meta-agent default model** — recommend Claude Sonnet 4.6 as default for cost-effectiveness; user override
- **Volatility predicate mapping** — which predicates default to `stable` / `slow` / `volatile` (initial table can be small; expand from real-world usage)
- **Heartbeat default cadence** — 30 min vs 60 min (settle in Session 07)
- **Per-day budget default** — $10/day proposed; reasonable for solo use, adjustable in settings
- **Skills catalog for v0.3** — none (skills system fully ships in v0.4 with sub-instances). v0.3's main agent is single-agent only; skills require sub-instances to be useful
- **Code-signing certificate procurement** — Apple Developer Program enrollment ($99/yr), Windows EV cert, GPG keypair: must complete before Session 10 begins
- **`spren.dev` (or chosen) domain registration + manifest server hosting plan** — needed for install.sh distribution and the Tauri auto-updater
- **OSS-only vs freemium funnel** — undecided; affects landing-page copy + release-channel signage. Settle before Session 10's release prep.

## v0.4 sessions (preview)

See [`v0.4-extensions.md`](./v0.4-extensions.md). Roughly 30 sessions covering: TUI client (Textual), Tauri polish (tray / autostart / OS notifications / deep links / file associations), sub-instance spawning + lifecycle, teams + team managers, full skills catalog, Tier 4 hybrid index + recall tool, daily consolidation pipeline, standing approvals, channel adapters (Telegram polling + Discord gateway), cron scheduler, webhook + cloudflared, workflow templates + import/export, `SprenTelemetrySink` adapter, advanced semantic linter consumption, pause/resume Spren surface, user + meta-agent custom tools.

## v0.5 sessions (preview)

See [`v0.5-future.md`](./v0.5-future.md). Looser. Items: Slack + WhatsApp messengers, hosted/cloud deployment, code-generated tools with user approval, multi-user mode (only if demand).

## Test discipline

Per [`sessions/README.md`](./sessions/README.md), every session produces unit + integration + E2E tests where applicable. CI runs all three on every PR. v0.3 acceptance gate: full Playwright + tauri-driver golden-path passes on macOS, Linux, and Windows in CI.

## Framework features Spren depends on

Spren-driven framework features are tracked in [`../framework/`](../framework/) — a parallel directory mirroring this Spren one. Per-version plans and per-session framework PR briefs. v0.3 blocks on two framework features (NDJSON streaming tracing writer + AG-UI event-stream translator); v0.4 builds on four more (`TelemetrySink`, pause/resume completion, workflow serializer, advanced semantic linter).

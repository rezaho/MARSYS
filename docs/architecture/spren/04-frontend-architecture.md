# 04 — Frontend Architecture

The Spren product surfaces are three clients of one FastAPI backend (SP-019): a **desktop GUI** (Tauri webview), the **same GUI in a system browser tab** (for users who prefer it), and a **TUI** (terminal user interface, Textual). All three consume the same REST + SSE + POST surface.

This document covers the GUI stack (Tauri + Vite + React + TanStack Router) shared between the desktop and browser surfaces. The TUI is documented in [`09-meta-agent.md`](./09-meta-agent.md) and the implementation plan; it consumes the same FastAPI surface from Python (Textual rendering only) and shares no frontend code with the GUI.

## GUI stack

| Layer | Choice | Why |
|-------|--------|-----|
| Desktop shell | **Tauri 2** | Rust shell + system webview. Native installer per platform. Built-in auto-updater. Sidecar pattern manages the Python backend. Smaller binaries than Electron, lower RAM, native feel via system webview. |
| Build tool | **Vite** | Purpose-built for SPAs. Fast dev server (HMR < 50ms). Tauri's official default. Spren is a local-first SPA: there is no SSR / RSC requirement, so Next.js is not used. |
| UI framework | **React 19** | Ecosystem alignment with shadcn / Radix / `@xyflow/react` / Tremor / cmdk. |
| Language | **TypeScript ^6** | Pinned in `apps/web/package.json`; re-verify before each release. |
| Routing | **TanStack Router** | Best-in-class type inference; pairs with Pydantic-generated TS for end-to-end type safety (SP-005, SP-019). File-based route generation. |
| Server-state cache | **TanStack Query** | Caching, mutations, SSE invalidation. |
| Client state (global) | **Zustand** | Small, native `useSyncExternalStore`. |
| Client state (canvas-hot) | **Jotai** | Fine-grained reactivity for high-frequency canvas updates (selection, drag). |
| Forms | **React Hook Form + Zod** | Schema validation on the client; backend Pydantic remains authoritative. |
| Component primitives | **shadcn/ui + `radix-ui`** | Heavy theme customization away from generic-shadcn defaults. |
| Charts | **shadcn/ui charts** (Recharts under the hood) | Lighter than Tremor; revisit if insufficient. |
| Canvas | **`@xyflow/react`** | The current React Flow package; legacy `reactflow` is not used. |
| Motion | **`motion`** (single package, `motion/react`) | Successor to Framer Motion + Motion One. |
| Command palette | **cmdk** | Linear / Vercel primitive. Primary navigation surface; `:` palette in the TUI mirrors it. |
| Type | **`geist`** (Sans / Mono via subpaths) | Single package; works inside Vite via standard CSS imports. Mono is used heavily for agent names, tool IDs, span IDs. |
| Styling | **Tailwind v4 + CSS variables** | Tokens defined as `@theme inline` over the design-token CSS variables; shadcn primitives consume the same variables. |
| Static asset delivery | Vite bundle | Tauri embeds it as app resources; the FastAPI sidecar serves the same files from `_webui/` for browser-tab mode. |

Baseline deps (TypeScript, React 19, Vite, TanStack Router, TanStack Query, the `geist` type package, and MSW for test-only network stubbing) underpin every surface. Tailwind, shadcn, Radix, cmdk, `@xyflow/react`, Zustand, Jotai, React Hook Form, Zod, and motion land alongside the design system. Pin exact major.minor in `package.json`; re-verify before each release.

## Distribution shape — single bundle, three surfaces

```
                          ┌──────────────────────────────────┐
                          │  Spren backend (Python sidecar)  │
                          │  FastAPI: REST + SSE + POST      │
                          │  Serves apps/web Vite bundle on  │
                          │  /  (production / browser-tab)   │
                          └──────────────┬───────────────────┘
                                         │ same API contract
            ┌───────────────────────────┬┴────────────────────────┐
            │                           │                         │
  ┌─────────▼─────────────┐  ┌──────────▼─────────┐  ┌────────────▼──────────┐
  │ Tauri desktop         │  │ Browser tab        │  │ TUI                   │
  │ Rust shell + system   │  │ System browser     │  │ Textual (Python)      │
  │ webview loads same    │  │ loads same Vite    │  │ httpx + SSE against   │
  │ Vite bundle; injects  │  │ bundle from        │  │ same FastAPI surface; │
  │ __SPREN_AUTH__ +      │  │ FastAPI; reads     │  │ no shared frontend    │
  │ __SPREN_PORT__        │  │ #token=… fragment  │  │ code with the GUI     │
  └───────────────────────┘  └────────────────────┘  └───────────────────────┘
```

The Vite bundle (`apps/web/`, package `@marsys/spren-web`) is built once per release. `pnpm --filter @marsys/spren-web build` writes `apps/web/dist/`; the Justfile `build` recipe copies it to `packages/spren/src/spren/_webui/`. setuptools picks it up via `[tool.setuptools.package-data] spren = ["_webui/**/*"]`; the FastAPI app conditionally mounts `StaticFiles` only when `_webui/index.html` is present, so dev mode (where `_webui/` is empty) does not shadow the Vite dev server.

## Routing layout (TanStack Router, file-based)

```
apps/web/src/routes/
├── __root.tsx                        # root: providers (capabilities, query), command palette
├── index.tsx                         # home — Spren orb + greeting + input + "coming soon"
│                                     #   (the four-surface command center lands in v0.4+ — Sessions 07–09)
├── workflows/
│   ├── index.tsx                     # list (filterable by provenance)
│   ├── new.tsx                       # POST-on-mount + redirect to /workflows/$id
│   └── $workflowId.tsx               # canvas editor (xyflow + right-rail agent form + lint chip)
├── runs/                             # (v0.4+) global run history + trace inspector
├── triggers/                         # (v0.4+) schedules, webhooks, channels
├── memory/                           # (v0.4+) KB browser + facet view
├── settings/                         # (v0.4+) general / secrets / budgets / meta-agent
└── about.tsx
```

Routes are generated into `routeTree.gen.ts` by the TanStack Router Vite plugin. Tauri's webview and the browser tab use the same bundle and the same routes.

### Home in v0.3 (orb-only)

In v0.3 the home (`/`) is a single Spren orb (animated SVG, four reactive states — idle / typing / thinking / speaking), a greeting line, a single input bar, and a footer command-bar hint. There is **no persistent left rail**. Navigation is ⌘K (cmdk palette) plus the clickable `spren.` wordmark in the top bar; non-home surfaces also carry a small breathing presence orb (56 px) at top-right that opens a chat sheet overlay.

The four-surface command center described in earlier drafts of this document (**Now** / **Since you were away** / **Activity** / **Chat input**) **lands in Sessions 07–09 when the meta-agent runtime is wired live** — shipping it as dead placeholders in v0.3 would violate SP-007 (no mocked features in product code). Until then, the chat-input slot is wired to a stub-response delay so the orb-state-transition discipline is exercised on the surface 07–09 will later subscribe to.

## Bootstrap and capability detection

The product runs in modes determined at runtime, not at build time. A single endpoint reports what's there.

```python
# packages/spren/src/spren/server.py
class BootstrapResponse(BaseModel):
    framework: FrameworkInfo
    spren: SprenInfo
    surfaces: list[str]                    # ["gui", "tui"] — what this binary supports
    capabilities: Capabilities             # feature flags (channels enabled, cost ceiling, ...)
    endpoints: dict[str, str]              # named URL prefixes for clients
    started_at: str                        # ISO 8601 UTC
    data_dir: str                          # platformdirs.user_data_dir("spren")
```

A `CapabilitiesProvider` mounted at the root (`__root.tsx`) calls `GET /v1/bootstrap` on mount, stashes the response in a `CapabilitiesContext`, and exposes it via `useCapabilities()` in `apps/web/src/providers/capabilities.tsx`. Feature-conditional UI gates on the `capabilities` and `surfaces` fields and renders a clear "not available" notice when a capability is absent rather than crashing or hiding silently. Every Spren client (Tauri webview, browser tab, TUI) calls this endpoint on connect — the API is the single source of truth (SP-019).

The frontend resolves the API base URL via `resolveBaseUrl()` in `apps/web/src/lib/api.ts`: it prefers `window.__SPREN_PORT__` (Tauri injection), falls back to `import.meta.env.VITE_SPREN_API_URL` (browser dev), then same-origin (production, where the FastAPI sidecar serves the Vite bundle on `/`). The frontend never hardcodes a port.

Auth tokens (per-launch, regenerated on every server restart, SP-002) reach the frontend two ways:

- **Tauri**: the Rust shell spawns the Python sidecar, parses the stdout ready signal `spren-ready: port=<N> token=<T> data-dir=<P>`, and injects both `window.__SPREN_AUTH__` and `window.__SPREN_PORT__` via `init_script` before the bundle scripts run.
- **Browser tab**: the user opens a URL with the token in a fragment (`#token=...`). The frontend reads the token, then immediately strips the fragment via `history.replaceState` so it does not sit in the URL bar (and never appears in server logs — the fragment is client-side only).

Every API request carries `Authorization: Bearer <token>`. Missing or wrong token → 401.

## Type generation (Pydantic → TypeScript)

Per SP-005, Pydantic is the source of truth for types. The frontend never hand-writes TypeScript that mirrors a Pydantic model. The build script in `apps/web/` produces ONE generated file:

- `apps/web/src/lib/api-types.generated.ts` — every shape the frontend consumes, derived from `/openapi.json` via `openapi-typescript`. Models that don't surface via REST aren't part of the frontend's contract; if a future need surfaces a model that doesn't have a route, it gets added as a route response, not via a parallel generator.

`generate-types.mjs` (Node native ESM) runs only as `prebuild` so production builds are always fresh. It spawns a transient sidecar, parses the `spren-ready: port=N token=T` line from stdout, fetches `/openapi.json` with the auth token, writes the snapshot to `apps/web/openapi-snapshot.json` (gitignored), runs `openapi-typescript`, and shuts the sidecar down via stdin. The Spren API client (`apps/web/src/lib/api.ts`) imports from the generated file via `components["schemas"][...]`; placeholder interfaces declared inline during early sessions are replaced once their generated counterparts exist.

For shapes shared with the run event stream, the JS side imports `@ag-ui/core` and `@ag-ui/client` directly — no regeneration needed; the same Pydantic models are re-imported on the Python side (TUI, future framework adapter) directly from `packages/spren/src/spren/models/`.

## Layout: top bar + ⌘K + per-route chrome

```
┌──────────────────────────────────────────────────────────────────────┐
│  spren.  ›  breadcrumb (non-home only)                       avatar  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HOME (`/`):                                                         │
│    [ Spren orb — animated, breathing ]                               │
│    "Welcome / Welcome back, {name}."                                 │
│    [ input bar — stub send for v0.3, live meta-agent in 07–09 ]      │
│    coming-soon framing                                               │
│    ⌘K ·  hint to open the palette                                    │
│                                                                      │
│  CANVAS (`/workflows/$id`):                                          │
│    [ lint chip ] [ + Pattern ] [ Save ]                              │
│    ┌──────────────────────────────────────┬─────────────────────┐    │
│    │  xyflow canvas (uni/bi edges)        │  agent config form  │    │
│    │  palette: Agent / User / System /Tool│  (when node sel'd)  │    │
│    └──────────────────────────────────────┴─────────────────────┘    │
│                                              presence orb (corner)   │
│                                                                      │
│  WORKFLOWS LIST (`/workflows`):                                      │
│    [ + New ] [ + Import .py ] · filter chips · provenance badges     │
│                                                                      │
│  ⌘K palette overlay (when invoked) — primary navigation surface      │
└──────────────────────────────────────────────────────────────────────┘
```

No persistent left rail. The cmdk command palette is the primary navigation surface — pages register and unregister commands as they mount via a Zustand slice (`apps/web/src/stores/commands.ts`). The canvas's right rail is context-sensitive: it appears only when a canvas node is selected and disappears otherwise.

## Canvas (`@xyflow/react`) specifics

- One custom node type per marsys `NodeType`: `AgentNode`, `UserNode`, `SystemNode`, `ToolNode` (lowercase enum values per the data model — `agent`, `user`, `system`, `tool`)
- Custom edge component renders the edge type with appropriate styling: `invoke` solid, `notify` dashed, `query` with arrow on both sides for `alternating` pattern, `stream` with token-flow indicator
- Selection panel on right rail shows the selected node's full agent config (`agent_model`, goal, instruction, tools, memory, etc.) — editable inline with React Hook Form + Zod. The agent's model-config field is named `agent_model` because Pydantic v2 reserves `model_config` on `BaseModel`
- Pattern preset insertion: button "+ Pattern" → modal → pick `HUB_AND_SPOKE` / `PIPELINE` / `HIERARCHICAL` / `MESH` → fill params → graph injected
- Lint state shown inline on nodes (warning/error icons) and aggregated in a top toolbar; `POST /v1/workflows/{id}/lint` is the source
- Provenance badge on each workflow (visual builder / meta-agent / code-imported / template / API), driven by the `provenance` column on the `workflows` row

## State management split

| Concern | Where | Why |
|---------|-------|-----|
| Bootstrap / capabilities | React context (`CapabilitiesProvider`) | Read once on mount; gates feature-conditional UI |
| Server data (workflows, runs, settings) | TanStack Query | Caching, refetching, mutations, SSE invalidation |
| AG-UI event stream subscriptions | Custom hook over `EventSource` API | One subscription per `run_id`; AG-UI events buffered into a ring buffer, exposed as a Zustand slice |
| Canvas selection / hover / drag state | Jotai atoms | Fine-grained reactivity for high-frequency updates |
| Active conversation with meta-agent | Zustand slice | Persisted to `localStorage` for refresh recovery (browser tab) or to disk via Tauri (desktop) |
| Theme + UI prefs | Zustand slice | Persisted as above |
| Cmdk command set | Zustand slice | Pages register/unregister commands as they mount |

## Live updates strategy

Per SP-003 / SP-004: REST for CRUD, SSE for server→client streams, one `POST /v1/runs/{id}/respond` for mid-run user interaction. No WebSocket. No gRPC. AG-UI is the wire schema for run events; the JS side imports `@ag-ui/core` + `@ag-ui/client` directly.

- Run page subscribes to `/v1/runs/{id}/events` via `EventSource`
- Reducer applies AG-UI events to a `RunState` object: span tree, current status, tokens-streaming-target, last-error
- Trace timeline component is a memoized render over `RunState.spans`
- When tab is hidden, `EventSource` is paused (browser native); missed events replay on resume by re-fetching the trace and updating to current
- Meta-agent home subscribes to `GET /v1/meta/conversations/{id}/events` for the activity stream and Since-you-were-away update events; same reducer pattern

## Error handling

- Error boundaries at: route-level, panel-level (so canvas error doesn't kill the whole app)
- API errors: toast with `error.code`-based localized message; full `error` object surfaced in dev mode
- SSE disconnects: auto-reconnect with exponential backoff, capped at 30s; UI shows "reconnecting" indicator
- Network down: cached state remains usable; mutations queued and replayed on reconnect (TanStack Query handles this)
- Tauri-specific: shell-level error overlay if the Python sidecar dies; "restart sidecar" action surfaced

## Theming + design tokens

The v0.3 aesthetic anchors on the Spren orb (asymmetric egg, peach → coral → magenta → magenta-deep gradient with layered blur giving it a grainy life) over a warm-white ground. Light mode is first-class and the v0.3 default; the dark-mode warm-console palette earlier drafts described is deferred to v0.4+ (and can later live behind a `[data-theme="dark"]` opt-in).

The locked v0.3 tokens live in `apps/web/src/styles/tokens.css`:

```css
:root {
  --ground: #fcfaf6;            /* page background, warm off-white */
  --surface: #ffffff;           /* cards, modals, input bars */

  --peach: #ffceaa;             /* orb 0% — peach highlight */
  --coral: #ff876c;             /* orb ~42% — load-bearing middle */
  --magenta: #e82182;           /* orb ~82% + the accent color */
  --magenta-deep: #c9146c;      /* orb 100% + meta_agent badge */

  --ink: #1a1410;               /* primary text, send-button default */
  --ink-soft: #6b5f58;          /* secondary text, breadcrumbs */
  --ink-faint: #9b8f88;         /* tertiary — ONLY at 18px+ */
  --rule: #ede6da;              /* hairline borders */
  --rule-soft: #f4efe5;         /* kbd backgrounds, hover surfaces */
}
```

The `geist` npm package is **not** consumed (it peers `next>=13.2.0`); instead the design system imports `@fontsource/geist-sans` + `@fontsource/geist-mono` and Tailwind v4 maps them via an `@theme inline` block in `apps/web/src/styles/globals.css`. Empty states use the tag-markup typographic device (`<agent name="" model="" tools={...} />`).

## Tauri webview integration

The Tauri shell (`apps/desktop/`, Rust crate at `apps/desktop/src/main.rs` — no `src-tauri/` subdir, since the frontend already lives in the sibling `apps/web/`) is a thin Rust binary that:

1. Spawns the Python sidecar (`uv run --package spren python -m spren --port 0`) as a managed child process. The sidecar binds to `127.0.0.1` only, picks a free port, generates the per-launch auth token itself, and writes it to `<data-dir>/runtime/auth-token` (mode 0600).
2. Reads stdout for the ready signal `spren-ready: port=<N> token=<T> data-dir=<P>`, captures port and token, then opens the webview. In dev the webview points at the Vite dev server (`http://localhost:5173`); in production it loads the bundled Vite assets.
3. Injects both `window.__SPREN_AUTH__` and `window.__SPREN_PORT__` via `init_script` before the bundle scripts run, so the frontend reads the token from a window variable and resolves the API base URL via `__SPREN_PORT__` rather than parsing a URL fragment.
4. Handles app lifecycle: tray icon (later versions), autostart, OS notifications, deep links.
5. On quit: sends `shutdown\n` to the sidecar's stdin, closes the pipe, polls for clean exit up to a short timeout (2s), and falls back to force-kill only if the timeout elapses. The sidecar's stdin watcher only engages when stdin is a pipe (not a TTY), so `python -m spren` in a normal shell still works.

CORS on the sidecar is locked to `r"^(http://(127\.0\.0\.1|localhost)(:\d+)?|tauri://localhost)$"`, covering both dev (Vite at any port) and the Tauri webview origin (`tauri://localhost`).

The Vite bundle is identical between Tauri and browser-tab mode. The only differences are how the frontend obtains the auth token (window variable in Tauri, URL fragment in browser tab) and how it resolves the API base URL (`__SPREN_PORT__` in Tauri, `VITE_SPREN_API_URL` in browser dev, same-origin in production).

> **WSLg caveat:** on WSL2 / WSLg hosts, `wry::Builder::run()` returns `Ok(_)` without rendering a window — a known gap between `webkit2gtk-4.1` and the WSLg X server. Logic verification (sidecar spawn, ready-line parsing, auth injection) works through cargo unit tests + Playwright E2E. Visible-window verification needs native macOS, Windows, or Linux.

## Testing

- **Unit:** Vitest + Testing Library (React) — pure components, hooks, reducers. Zustand and Jotai stores tested via their store-level APIs without React.
- **Component visuals:** Storybook (optional in early Spren releases; required as the design system stabilizes).
- **E2E (browser):** Playwright. Full FastAPI lifecycle in tests; Vite dev server managed by `webServer` config.
- **E2E (Tauri):** Tauri's WebDriver integration (tauri-driver) for desktop-shell behaviors that don't apply in browser mode.
- **API mocking:** MSW for unit/integration tests; real FastAPI dev server for integration mode (preferred where feasible).
- **Type tests:** `tsc --noEmit` in CI; generated types tested with sample round-trips.

No mock services in product code (SP-007). Pin exact major.minor; re-verify before each release.

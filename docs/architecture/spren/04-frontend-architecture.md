# 04 — Frontend Architecture

The Spren product surfaces are three clients of one FastAPI backend (SP-019): a **desktop GUI** (Tauri webview), the **same GUI in a system browser tab** (for users who prefer it), and a **TUI** (terminal user interface, Textual). All three consume the same REST + SSE + POST surface.

This document covers the GUI stack (Tauri + Vite + React + TanStack Router) shared between the desktop and browser surfaces. The TUI is documented in [`09-meta-agent.md`](./09-meta-agent.md) and the implementation plan; it is API-only, no shared frontend code with the GUI.

## GUI stack

| Layer | Choice | Why |
|-------|--------|-----|
| Desktop shell | **Tauri 2** | Rust shell + system webview. Native installer per platform. Built-in auto-updater. Sidecar pattern manages the Python backend. Smaller binaries than Electron, lower RAM, native feel via system webview. |
| Build tool | **Vite** | Purpose-built for SPAs. Fast dev server (HMR < 50ms). Tauri's official default. No SSR / static-export corners to navigate. |
| UI framework | **React 19** | Ecosystem alignment with shadcn / Radix / `@xyflow/react` / Tremor / cmdk. |
| Language | **TypeScript** | Pinned to a stable line; reverify before each release. |
| Routing | **TanStack Router** | Best-in-class type inference; pairs with Pydantic-generated TS for end-to-end type safety (SP-005, SP-019). File-based route generation. |
| Server-state cache | **TanStack Query** | Caching, mutations, SSE invalidation. |
| Client state (global) | **Zustand** | Small, native `useSyncExternalStore`. |
| Client state (canvas-hot) | **Jotai** | Fine-grained reactivity for high-frequency canvas updates (selection, drag). |
| Forms | **React Hook Form + Zod** | Schema validation on the client; backend validation remains authoritative. |
| Component primitives | **shadcn/ui + `radix-ui`** | Heavy theme customization away from generic-shadcn defaults. |
| Charts | **shadcn/ui charts** (Recharts under the hood) | Lighter than Tremor; revisit if insufficient. |
| Canvas | **`@xyflow/react`** | The current React Flow package; legacy `reactflow` is not used. |
| Motion | **`motion`** (single package, `motion/react`) | Successor to Framer Motion + Motion One. |
| Command palette | **cmdk** | Linear / Vercel primitive. Primary navigation surface. |
| Type | **`geist`** (Sans / Mono / Pixel via subpaths) | Single package; works inside Vite via standard CSS imports. |
| Static text bundle | Inline in the Vite bundle | Static export concept does not apply; Tauri ships the bundle as files inside the app resources, browser-tab mode is served by FastAPI. |

Pin exact major.minor in `package.json`; re-verify before each release.

## Distribution shape — single bundle, three surfaces

```
                                    ┌──────────────────────────────────┐
                                    │  Spren backend (Python sidecar)  │
                                    │  FastAPI: REST + SSE + POST      │
                                    │  Serves the static apps/web      │
                                    │  bundle on /  (browser-tab mode) │
                                    └──────────────┬───────────────────┘
                                                   │ same API contract
       ┌───────────────────────────┬───────────────┼───────────────────────────┐
       │                           │               │                           │
  ┌────▼────────────────┐  ┌───────▼─────────┐  ┌──▼───────────────┐  ┌───────▼─────────────┐
  │ Tauri desktop       │  │ Browser tab     │  │ TUI              │  │ Framework adapter   │
  │ Rust shell + system │  │ System browser  │  │ Textual (Python) │  │ (Python SDK)        │
  │ webview loads same  │  │ loads same Vite │  │ Same FastAPI API │  │ Pushes events to    │
  │ Vite bundle         │  │ bundle from     │  │                  │  │ FastAPI when env    │
  │                     │  │ FastAPI         │  │                  │  │ var is set          │
  └─────────────────────┘  └─────────────────┘  └──────────────────┘  └─────────────────────┘
```

The Vite bundle is built once per release; Tauri embeds it as app resources, FastAPI serves the same files for the browser-tab mode.

## Routing layout (TanStack Router, file-based)

```
apps/web/src/routes/
├── __root.tsx                        # root: theme, fonts, providers (capabilities, query, zustand)
├── index.tsx                         # home — four-surface command center (Now / Inbox / Activity / Chat)
├── workflows/
│   ├── index.tsx                     # list (filterable by provenance)
│   ├── new.tsx                       # blank canvas (or scaffold from prompt)
│   └── $workflowId/
│       ├── index.tsx                 # canvas editor
│       └── runs.tsx                  # runs filtered to this workflow
├── runs/
│   ├── index.tsx                     # global run history
│   └── $runId.tsx                    # trace inspector for one run
├── triggers/                         # schedules, webhooks, channels (later versions)
├── memory/
│   ├── index.tsx                     # KB browser
│   └── $facetSlug.tsx                # facet view (profile, projects, ...)
├── settings/
│   ├── index.tsx                     # general
│   ├── secrets.tsx                   # API keys
│   ├── budgets.tsx                   # cost caps
│   └── meta-agent.tsx                # meta-agent model + tools
└── about.tsx
```

Routes are generated into `routeTree.gen.ts` by the TanStack Router Vite plugin. Tauri's webview and the browser tab use the same bundle and the same routes.

## Bootstrap and capability detection

The product runs in modes determined at runtime, not at build time. A single endpoint reports what's there.

```python
# packages/spren/src/spren/routes/bootstrap.py
class BootstrapResponse(BaseModel):
    framework: FrameworkInfo
    spren: SprenInfo
    surfaces: list[str]                    # ["gui", "tui"] — what this binary supports
    capabilities: Capabilities             # feature flags (channels enabled, cost ceiling, ...)
    endpoints: dict[str, str]              # named URL prefixes for clients
```

The frontend root provider (`__root.tsx`) calls `/v1/bootstrap` once on mount, stashes the response in a `CapabilitiesContext`, and gates feature-conditional UI on it. Clients that depend on a capability render a clear "not available" notice when the capability is absent rather than crashing or hiding silently.

For a Tauri-managed launch: the Rust shell starts the Python sidecar, waits for the readiness signal on stdout, captures the per-launch auth token, and injects it into the webview as a window-level variable. The frontend reads the variable instead of parsing it from a URL fragment when running under Tauri. Browser-tab mode keeps the URL-fragment flow.

## Layout: three rails + cmdk

```
┌──────────────────────────────────────────────────────────────────────┐
│  ⌘K palette overlay (when invoked)                                   │
├─────────┬──────────────────────────────────────────┬─────────────────┤
│  LEFT   │  MAIN                                    │  RIGHT          │
│  RAIL   │  - home (four-surface command center)    │  RAIL           │
│  - logo │  - workflow canvas (when editing)        │  - active       │
│  - nav  │  - run inspector (when viewing)          │    artifact     │
│  - WFs  │  - settings                              │    (workflow,   │
│  - runs │  - memory browser                        │     run, etc.)  │
│  - sched│                                          │                 │
│  - chans│                                          │  - streaming    │
│  - sett │                                          │    tokens       │
└─────────┴──────────────────────────────────────────┴─────────────────┘
```

Left rail collapsible. Right rail context-sensitive (collapses when nothing active). Main always present; primary surface based on route.

## Canvas (`@xyflow/react`) specifics

- One custom node type per marsys `NodeType`: `AgentNode`, `UserNode`, `SystemNode`, `ToolNode`
- Custom edge component renders the edge type with appropriate styling (INVOKE solid, NOTIFY dashed, QUERY with arrow on both sides for ALTERNATING, etc.)
- Selection panel on right rail shows the selected node's full agent config (model, goal, instruction, tools, memory, etc.) — editable inline with React Hook Form + Zod
- Pattern preset insertion: button "+ Pattern" → modal → pick HUB_AND_SPOKE / PIPELINE / HIERARCHICAL / MESH → fill params → graph injected
- Lint state shown inline on nodes (warning/error icons) and aggregated in a top toolbar
- Provenance badge on each workflow node (visual builder / meta-agent / imported / Python code)

## State management split

| Concern | Where | Why |
|---------|-------|-----|
| Server data (workflows, runs, settings) | TanStack Query | Caching, refetching, mutations, SSE invalidation |
| AG-UI event stream subscriptions | Custom hook over `EventSource` API | One subscription per `run_id`; events buffered into a ring buffer, exposed as a Zustand slice |
| Canvas selection / hover / drag state | Jotai atoms | Fine-grained reactivity for high-frequency updates |
| Active conversation with meta-agent | Zustand slice | Persisted to `localStorage` for refresh recovery (browser tab) or to disk via Tauri (desktop) |
| Theme + UI prefs | Zustand slice | Persisted as above |
| Cmdk command set | Zustand slice | Pages register/unregister commands as they mount |

## Live updates strategy

- Run page subscribes to `/v1/runs/{id}/events` via `EventSource`
- Reducer applies AG-UI events to a `RunState` object: span tree, current status, tokens-streaming-target, last-error
- Trace timeline component is a memoized render over `RunState.spans`
- When tab is hidden, `EventSource` is paused (browser native); we replay missed events on resume by re-fetching the trace and updating to current
- Meta-agent home subscribes to `/v1/meta/events` for the activity stream and inbox-update events; same reducer pattern

## Error handling

- Error boundaries at: route-level, panel-level (so canvas error doesn't kill the whole app)
- API errors: toast with `error.code`-based localized message; full `error` object surfaced in dev mode
- SSE disconnects: auto-reconnect with exponential backoff, capped at 30s; UI shows "reconnecting" indicator
- Network down: cached state remains usable; mutations queued and replayed on reconnect (TanStack Query handles this)
- Tauri-specific: shell-level error overlay if the Python sidecar dies; "restart sidecar" action surfaced

## Theming + design tokens

CSS variables in `apps/web/src/styles/tokens.css`:

```css
:root {
  --bg-base: #0c0a08;
  --bg-elevated: #14120f;
  --accent-warm: #e8b26a;
  --accent-cool: #7df2d0;
  --text-primary: #f6f1e8;
  --text-muted: #a29b8e;
  --error: #e07856;
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --font-sans: 'Geist Sans', system-ui;
  --font-mono: 'Geist Mono', ui-monospace;
}
[data-theme="light"] { /* parallel set */ }
```

Tailwind reads these via `@theme inline`. shadcn components use them through their CSS-variable theming. Hex values are approximate; refine in design iteration.

## Tauri webview integration

The Tauri shell (`apps/desktop/`) is a thin Rust binary that:

1. Starts the Python sidecar (`packages/spren/`) as a managed child process. Passes `--port <random>` and `--auth-token <random>` on the command line.
2. Reads stdout for the ready signal (`spren-ready: <port>`), then opens the webview pointing at `http://127.0.0.1:<port>/`.
3. Injects the auth token into the webview as `window.__SPREN_AUTH__` before the bundle scripts run, so the frontend doesn't need URL-fragment parsing.
4. Handles app lifecycle: tray icon (later versions), autostart, OS notifications, deep links.
5. On quit: sends shutdown command to sidecar via stdin, waits for graceful exit, then terminates if needed.

The Vite bundle is identical between Tauri and browser-tab mode. The only difference is how the frontend obtains the auth token (window variable in Tauri, URL fragment in browser).

## Testing

- **Unit:** Vitest + Testing Library (React) — pure components, hooks, reducers. Zustand and Jotai stores tested via their store-level APIs without React.
- **Component visuals:** Storybook (optional in early Spren releases; required as the design system stabilizes).
- **E2E (browser):** Playwright. Full FastAPI lifecycle in tests; Vite dev server managed by `webServer` config.
- **E2E (Tauri):** Tauri's WebDriver integration (tauri-driver) for desktop-shell behaviors that don't apply in browser mode.
- **API mocking:** MSW for unit/integration tests; real FastAPI dev server for integration mode (preferred where feasible).
- **Type tests:** `tsc --noEmit` in CI; generated types tested with sample round-trips.

No mock services in product code (SP-007). Pin exact major.minor; re-verify before each release.

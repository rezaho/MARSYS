# Spren Session 03 — Visual Builder

> Session plan. The implementer reads this as the primary source of truth for what Session 03 ships, how it looks, and what's in vs out of scope. Captures bundle position, scope boundaries, files-to-DELETE candidates, the three user journeys that anchor Bundle A's demo gate, skeleton wireframes for the new surfaces, the locked design system (palette + typography + Spren orb spec + UI components + CSS recipes), the decisions locked from prior architect-stage exploration, the polish items the implementer addresses in-session, and success criteria.
>
> Status: **ready for implementation**. Acceptance criteria are frozen separately at [`./03-visual-builder/acceptance.md`](./03-visual-builder/acceptance.md) before coding starts (extracted by `acceptance-criteria-extractor` agent on the first implementation turn).

Visual anchor: [`./03-visual-builder/assets/spren-inspiration.png`](./03-visual-builder/assets/spren-inspiration.png) — the soft grainy egg-shape that is Spren's living presence. The entire UI is built around this. Reference HTML preview of the design system in context lives at [`./03-visual-builder/palette-preview.html`](./03-visual-builder/palette-preview.html) (open in a browser to see the orb animation + typography + spacing — not source of truth for implementation, but a sanity check).

---

## 1. Bundle position + tier

- **Bundle**: A — Visual Workflow Builder (Sessions 01 + 02 + 03).
- **Bundle demo gate**: user installs umbrella, opens Tauri shell, lands on the Spren orb home, opens the workflow surface (via ⌘K or asking Spren), builds a 3-agent workflow visually with `@xyflow/react`, saves it, sees it in the list with `provenance=visual_builder`. Also imports a workflow from a `.py` file. Build → save → reload round-trips the topology. Lint catches missing tool/agent refs inline.
- **Tier**: HIGH-tier with Researcher. New tech surface area is broad: `@xyflow/react@^12.10`, cmdk@^1.1, shadcn + Radix, Tailwind v4, `@fontsource/geist-sans` + `@fontsource/geist-mono` (NOT the `geist` npm package — that one peers `next>=13.2.0` and exports return `NextFontWithVariable` which isn't usable in Vite), Jotai, Zustand, React Hook Form + Zod, motion@^12 (`motion/react` subpath import). 2026 idiom research is load-bearing.
- **Approval gate**: Stage 4 (synthesis) only. Pipeline runs autonomously through stages 1–3.5.

## 2. Dependency check

| Dependency | State | Notes |
|---|---|---|
| Spren Session 01 (foundation) | shipped | TanStack Router + Vite + auth flow + bootstrap endpoint + `_webui/` build trick. |
| Spren Session 02 (CRUD + types + Python import) | shipped | Pydantic types + `/v1/workflows` REST + `/v1/workflows/import-python`. The TS type generation pipeline is wired here. Session 03 consumes the generated types — never hand-writes mirrors (SP-005). |
| `GET /v1/tools` (locked Q9) | **NOT shipped — Session 03 adds it** | Verified absent in `packages/spren/src/spren/routes/`. The brief's earlier claim that Session 02 ships it was wrong. Locked shape: `{"items":[{"name":str,"source":"framework","description":str\|null}]}`. Source = framework's `AVAILABLE_TOOLS` dict at `packages/framework/src/marsys/environment/tools.py`; `description` is `__doc__` first line (or `null`). |
| `POST /v1/workflows/{id}/lint` (locked Q4) | **NOT shipped — Session 03 adds it** | Verified absent. Locked shape: `{"findings":[LintFinding]}` where `LintFinding = {severity:"error"\|"warning", code:str, node_name:str\|null, edge:[str,str]\|null, message:str, suggestion:str\|null}`. Implementation wraps `marsys.coordination.topology.TopologyGraph.validate_workflow()` (which raises a single multi-line `TopologyError`) — the Spren adapter parses the message into per-finding rows and adds Spren-only findings (`unknown_tool`, `missing_agent_ref`). See §3 backend surfaces for the adapter spec. |
| Framework Session 01 (NDJSON tracing) | shipped | Not consumed by Session 03. |
| Framework Session 02 (`TelemetrySink`) | shipped | Not consumed by Session 03. |
| Framework Session 05 (advanced linter) | held pending Session 03 | The canvas's lint surface needs are inputs to that brief. Session 03's lint adapter is the regex-bridge interim until Framework Session 05 ships structured findings. |
| `TopologyGraph.validate_workflow()` (existing framework) | live at `packages/framework/src/marsys/coordination/topology/graph.py:1395` | Raises `TopologyError` with a newline-concatenated multi-line message. Spren's lint adapter parses this. |

Session 03 does NOT touch any TRUNK-CRITICAL framework file (SP-001, SP-018). All work lands inside `apps/web/` and `packages/spren/` server-side surfaces.

### 2.1 Design pivot from `docs/architecture/spren/04-frontend-architecture.md`

`04-frontend-architecture.md` lines 62–83 + 122–138 + 182–201 describe the home page as a **four-surface command center** (Now / Since you were away / Activity / Chat input) with a persistent left rail and dark-mode tokens. **Session 03 pivots this**: the home in v0.3 is a single Spren orb + greeting + input + footer hint, in light mode, with NO persistent left rail. The four-surface command center is deferred to Sessions 07–09 when the meta-agent is wired live (the surfaces would otherwise be dead placeholders contradicting SP-007).

Session 03 updates the architecture doc in the same PR:
- §1 surfaces table — note `/` is orb-only in v0.3
- §3 (Layout) — replace the three-rail diagram with the orb-home + cmdk + presence-orb layout
- §11 Theming — switch the canonical examples to light-mode tokens (the dark-mode set remains documented as the eventual second theme but not the default)

## 3. What ships in Session 03

UI surfaces:

- TanStack Router routes: `/`, `/workflows`, `/workflows/new`, `/workflows/$workflowId`. (Existing `/runs` etc. remain placeholders — Session 05+.)
- **Home (`/`): Spren orb interface (v0.3 "coming soon" framing).** The orb (animated SVG, per §9 spec) is the entire home. Greeting, input bar, command-bar hint, and a small subline noting that the four-surface command center (Now / Since you were away / Activity) arrives with the meta-agent in Sessions 07–09. No persistent left nav rail. Navigation is ⌘K or conversational ("show me my workflows"). The send button on the input bar fires a stub-response (see §10 polish item 10) — the live meta-agent wiring lands in 07–09. See W-A. **This replaces the four-surface command-center home described in `docs/architecture/spren/04-frontend-architecture.md` lines 62–83 — see §2.1 Design pivot above.**
- **Top-bar chrome (every surface):** `spren.` wordmark (clickable, links home) on the left, surface breadcrumb in the center, user avatar on the right. No left nav rail anywhere.
- **Presence orb on non-home surfaces:** a small (48-72px) breathing version of the orb in the top-right corner of `/workflows`, canvas, etc. Clicking opens a chat sheet overlay. (Polish item 3.)
- Workflow list page: cards with provenance badges, filters (provenance / archived), `+ New Workflow` and `+ Import from Python` buttons.
- Canvas page: `@xyflow/react` editor with custom node components per `NodeType`, custom edge components for uni- and bi-directional only (Q2 lock — alternating + symmetric patterns and the 4 EdgeType values are NOT exposed in v0.3), top toolbar (workflow name editor + Lint chip + `+ Pattern` + Save + Run-placeholder), left palette (Agent / User / System / Tool draggables), right rail with agent/edge configuration form (React Hook Form + Zod), `+ Pattern` modal with the four presets.
- Importer warning UX: if a `.py` import contains `<~>` (alternating) or `<|>` (symmetric) patterns, the canvas auto-converts them to plain bidirectional edges and surfaces a non-blocking warning toast + per-converted-edge inline marker (Q2 lock).
- Lint surface: top-toolbar aggregated chip (debounced 300ms POST `/v1/workflows/{id}/lint`) + inline node/edge markers; clicking a marker scrolls the issue list and highlights the node.
- Empty states + tag-markup typographic devices (`<agent name="" model="" />` etc.) for canvas, list, and right rail.
- Design system foundation: `apps/web/src/styles/tokens.css` with the locked palette (§9), Tailwind v4 wired with `@theme inline` over the tokens, Geist Sans + Mono via the `geist` package, light mode first-class. Cmdk command palette mounted at `__root.tsx`. shadcn primitive install (button / card / dialog / dropdown / input / select / switch / textarea / toast / tooltip) with theme overrides.
- **Spren orb component:** `apps/web/src/components/Spren.tsx` rendering the SVG orb with reactive states (idle / typing / thinking / speaking) per §9.
- **Workflow create flow (Q7 lock):** `POST /v1/workflows` fires from inside the `/workflows/new` route component's `useEffect` on first mount (NOT from a TanStack Router `loader` — that would cause hover-prefetch ghosts), returns a ULID, the route redirects to `/workflows/{id}` (via `router.navigate({ to: "/workflows/$id", params: { workflowId: id }, replace: true })`). Empty drafts are detected by the predicate `provenance='visual_builder' AND topology.nodes=[]` — no schema migration; first-explicit-save advances `updated_at` and the row leaves draft state. The list endpoint filters drafts via `include_drafts=false` (default). The sweeper deletes empty drafts older than 24 hours.
- **Auto-layout on first load (Q8 lock):** when the stored definition has no node positions (most commonly: imported workflows), the canvas runs Dagre or ELK (Researcher to recommend) on first load. User-dragged positions persist on next save.

Backend surfaces (added in this session — Session 02 did not ship these):

- **`GET /v1/tools`** at `packages/spren/src/spren/routes/tools.py`. Returns `ToolListResponse = {"items": list[ToolInfo]}` where `ToolInfo = {"name": str, "source": Literal["framework"], "description": str | None}`. Source data: framework's `AVAILABLE_TOOLS` dict at `packages/framework/src/marsys/environment/tools.py:544`. `description` is the callable's `__doc__` first line (whitespace-trimmed); `None` if the callable has no docstring. Auth-gated; cached for the FastAPI process lifetime (the framework's tool registry is import-time-static).
- **`POST /v1/workflows/{id}/lint`** at `packages/spren/src/spren/routes/lint.py`. Returns `LintResponse = {"findings": list[LintFinding]}` where `LintFinding = {"severity": Literal["error","warning"], "code": Literal["unreachable","cycle_no_escape","missing_agent_ref","unknown_tool","dangling_edge","missing_required_field"], "node_name": str | None, "edge": tuple[str, str] | None, "message": str, "suggestion": str | None}`. Implementation:
  1. Load the workflow definition from SQLite by id.
  2. Run Spren-side cross-ref checks first (unknown tool names against the cached tool list, missing `agent_ref` keys, dangling edge endpoints). These never raise — they collect findings directly.
  3. Convert `WorkflowDefinition` → `marsys.coordination.topology.TopologyGraph` via a Spren-side adapter in `packages/spren/src/spren/lint/topology_adapter.py` (creates a `TopologyGraph` and replays each `NodeSpec` / `EdgeSpec` as `add_node` / `add_edge` calls).
  4. Call `TopologyGraph.validate_workflow()` inside a `try`; on `TopologyError`, parse the multi-line message via regex (`r"node '(?P<name>[^']+)'"`, `r"nodes (?P<scc>\{[^}]+\})"`) into structured findings. Unparseable lines become a single `severity=error code=unreachable message=<raw>` finding (defensive fallback).
  5. Return all findings combined.
  Lint is non-blocking — the endpoint always returns 200 with findings (even if `findings=[]`); only auth and 404 (workflow not found) produce non-200.
- **Draft sweeper** at `packages/spren/src/spren/workers/draft_sweeper.py`. Runs on a 4-hour interval from the FastAPI lifespan handler (a background `asyncio.create_task` that sleeps + sweeps in a loop). Predicate: rows where `provenance='visual_builder' AND json_extract(definition, '$.topology.nodes') = '[]' AND updated_at < (now - 24h)` — empty drafts older than 24 hours. The predicate avoids a schema migration (no `is_draft` column needed); first-explicit-save advances `updated_at` so the row leaves draft state automatically.
- **List filter** — `GET /v1/workflows?include_drafts=false` (default `false`) drops the empty-draft predicate above from the result set. The caller passes `include_drafts=true` only for admin/sweeper-debug surfaces.

Tests:

- Vitest unit: Zustand + Jotai store behavior, reducers, agent-form Zod schemas, orb state machine.
- Playwright E2E: golden-path build (drag two agents, connect, configure, save, reload, verify); pattern-preset insertion; lint surface; provenance badge rendering; orb states (idle/typing/thinking/speaking observable via DOM attributes); ⌘K navigation.
- Tauri-driver E2E: same golden-path but invoked from the desktop shell.
- Visual regression baselines via Playwright's built-in `toHaveScreenshot()` (NOT Argos — Argos is a paid SaaS dep that contradicts SP-008's local-single-user posture; Playwright stores PNG baselines in `apps/web/tests/e2e/__screenshots__/` and diffs locally): orb home (idle), orb home (typing — input focused), canvas empty, canvas with topology, agent config form, lint issues panel.
- Manual-verify checklist (implementer self-verification before claiming done), including orb behavior across all four states + responsive breakpoints.

## 4. What is OUT of scope

| Out of scope in Session 03 | Lands in |
|---|---|
| Run execution (Run button live, AG-UI event stream subscription, live token streaming) | Session 04 |
| Trace viewer (nested span tree, cost chips per span) | Session 05 |
| Run history list + filtering | Session 05 |
| File upload / attachments | Session 05 |
| Full meta-agent surfaces (chat input → agent response on home, inbox, activity stream) | Sessions 07–09. The orb's "thinking" + "speaking" states are visually implemented in Session 03 but wired to the real meta-agent in 07–09. In Session 03 they fire on stub inputs (e.g., the input bar's "send" simulates a delay then displays a placeholder reply). |
| Memory browser + facet view | Session 06 |
| Settings / secrets / budgets / meta-agent config | Session 10 |
| Custom user-authored tools (in-app code editor) | v0.4 (`v0.4-30`) |
| Meta-agent-generated tools (with user approval) | v0.4 (`v0.4-31`) |
| TUI surface | v0.4 |
| Tauri polish (tray, autostart, OS notifications, deep links) | v0.4 |
| Triggers / schedules / channels routes | v0.4+ |
| Workflow gallery (curated multi-file templates with metadata) | v0.4. Canvas load-path is template-agnostic — same path that loads `code_import` workflows accepts a template. |
| The 4 framework `EdgeType` semantics (`NOTIFY` / `QUERY` / `STREAM` distinct behavior) | Future framework session; they're currently inert. Session 03 exposes only direction (uni/bi). |
| Alternating + symmetric edge patterns in canvas | Same framework session. Importer auto-converts these to plain bidirectional with a warning. |

Anything labeled out-of-scope renders as "not available" or empty placeholder in Session 03's routes (capability-gated per SP-019).

## 5. Files to DELETE in Session 03

| Path | Why | Replaced by |
|---|---|---|
| `apps/web/src/routes/index.tsx` (Session 02 placeholder UI) | The placeholder was a temporary bridge for Bundle A's foundation gate; Session 03 ships the orb home. | The orb-home implementation. |
| Any inline-styled placeholder cards / forms left over from Session 01–02 | Session 03 introduces Tailwind + shadcn; legacy inline styles must go (no `_v2` shims, no toggles). | Themed shadcn primitives. |
| Hand-written interface declarations in `apps/web/src/lib/api.ts` that mirror Session 02's Pydantic shapes | Session 02 wires the generated types; any local mirrors are anti-pattern (SP-005). | Generated types from `api-types.generated.ts`. |
| Any left-nav-rail components from Session 01–02 placeholder | Session 03 removes the persistent left rail. Navigation = ⌘K + clickable wordmark + presence orb. | None — the chrome simplifies. |

The implementer enumerates the exact files to delete in Stage 2 Designer; that list is the legacy-retention check at brief-writing time and at session-end audit.

## 6. User journeys (anchor for Bundle A demo gate)

These are the end-to-end flows Session 03 must deliver. Each journey step generates at least one screen-state we wireframe in §7. Bundle A's `test-scenarios.md` G-07 + U-05 + visual regression snapshots trace back to these.

### J-1 — First-time user (no workflows)

State: fresh data dir, Tauri shell launches, sidecar boots, bootstrap fetched, no workflows exist.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | Tauri window paints. | Home (`/`) | Orb breathes. Greeting: "Welcome." (no name on first launch — no profile yet). Input bar placeholder: "What's on your mind?". Footer command-bar hint. |
| 2 | User presses ⌘K. | Cmdk overlay | Command list visible. |
| 3 | User types "new workflow" and selects `Create new workflow`. | Cmdk → navigates to `/workflows/new` | `POST /v1/workflows` fires immediately (Q7 lock); ULID returned; browser redirects to `/workflows/{id}` with an empty canvas, draft tag, name "Untitled workflow" (editable). Right rail collapsed. Hint text on canvas: drag from palette, pick a pattern, or open ⌘K. Presence orb (48px) appears top-right. |
| 4 | Open ⌘K, type "agent". | Cmdk overlay | Command list filtered to `Add Agent node` and adjacent commands. |
| 5 | Click `Add Agent node`. | Canvas | Empty `Agent` node appears at viewport center with placeholder name `Agent 1`. Auto-selects → right rail opens with agent config form (defaults: name=Agent 1, model=blank required, instructions=blank required, tools=empty, memory_retention=session). |
| 6 | Fill the agent form: name `Researcher`, model `anthropic/claude-opus-4.7`, instructions, tools `[search_web, browse_url]`. | Right rail | Form validation: name required, model required. Apply commits to Jotai canvas state + React Hook Form internal state. The node label updates. |
| 7 | Add a second agent (`Writer`, `anthropic/claude-sonnet-4.6`, instructions, tools `[]`). Same flow. | Canvas + right rail | Two agent nodes visible. |
| 8 | Drag a `User` node from the palette onto the canvas. | Canvas | User node appears. |
| 9 | Connect User → Researcher (drag from User's output handle to Researcher's input handle). Edge default unidirectional (Q2 lock — only uni / bi exposed). | Canvas | Edge renders as `→` solid line. |
| 10 | Connect Researcher → Writer. Unidirectional. | Canvas | Edge renders solid. |
| 11 | Lint runs (debounced 300ms after last edit, POST `/v1/workflows/{id}/lint`). | Top toolbar lint chip | Chip turns green `✓ OK` if topology validates; otherwise yellow with N warnings; clicking opens the issues panel. |
| 12 | Click `Save`. | Top toolbar | `PUT /v1/workflows/{id}` (row was created on route mount per Q7); `draft` tag drops; `provenance` stays `visual_builder`; toast confirms. |
| 13 | Click the `spren.` wordmark. | Returns to `/` (orb home) | Orb breathes. |
| 14 | Press ⌘K → "workflows" → enter. | Navigates to `/workflows` | The workflow appears with provenance badge `<provenance:visual_builder/>`. |

Bundle B (Session 04) extends this journey with the `Run` button. Session 03 ends at step 14.

### J-2 — Returning user (existing workflows)

State: workflows exist via various provenances (some `api`, some `code_import`, some `visual_builder`).

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | Tauri window paints. | Home (`/`) | Orb breathes. Greeting: "Welcome back, [name]." with name in `--magenta`. Subline: "Tell me what you're thinking about." Input bar focused-ready. |
| 2 | User presses ⌘K and types "workflows". | Cmdk → `/workflows` | Workflow list shows all workflows with provenance badges. Filter chips: All / Visual / Imported / Meta-agent. Empty drafts hidden by default. |
| 3 | Click a workflow card (e.g., a `code_import` workflow). | Navigates to `/workflows/$id` | Canvas paints with the workflow's stored topology. Auto-layout runs on first load if positions missing (Q8); positions persisted on save. Presence orb (48px) in top-right. |
| 4 | User adds a `Tool` node from the palette and connects it to `Researcher`. | Canvas + right rail | Tool node form prompts for tool reference (typed dropdown from `GET /v1/tools` — Q9 lock). |
| 5 | Lint surfaces a warning: tool reference `not_a_tool` is unknown. | Lint chip + inline node marker | User sees warning with did-you-mean suggestion. |
| 6 | User fixes the tool reference. Lint clears. | Lint chip | Chip back to `✓ OK`. |
| 7 | Click `Save`. | `PUT /v1/workflows/{id}` | Toast confirms. Provenance stays `code_import` (Q3 lock — creation-attribution does not flip on edit). `updated_at` advances. |

### J-3 — Python file import

State: user has a `.py` file using the marsys framework. `POST /v1/workflows/import-python` exists from Session 02.

| # | Action | Surface | Feedback |
|---|---|---|---|
| 1 | User talks to Spren ("import this Python file") or uses ⌘K → `+ Import from Python`. | Native file picker opens | `.py` extension filter. |
| 2 | Select `my_workflow.py`. | Multipart upload to `POST /v1/workflows/import-python`. | Spinner during upload + parse. |
| 3 | Backend parses + creates workflow with `provenance=code_import`, `provenance_metadata={source_filename, sha256}`. Any alternating (`<~>`) or symmetric (`<|>`) edges are auto-converted to plain bidirectional, surfaced in the response as `warnings: [{edge_id, kind, original_pattern}]`. | Toast confirms; workflow appears in list with `<provenance:code_import/>` badge. | If warnings present, toast offers "Review warnings" → opens the canvas with the affected edges highlighted. |
| 4 | Click the workflow. | Navigates to `/workflows/$id` | Canvas paints. Auto-layout runs (importer doesn't compute positions — Q8). Converted edges (if any) marked yellow with hover tooltip explaining the conversion. |
| 5 | User edits (adds a node, configures an existing agent). | Canvas + right rail | Same flows as J-2. |
| 6 | Save. | `PUT` | Provenance stays `code_import` (Q3). |

## 7. Skeleton wireframes (low-fi; ASCII)

These are spatial sketches anchored on the journey steps. Detail (component spacing, exact icon set, tag-markup empty-state catalog, hover states, focus rings) is the Designer's job in Stage 2.

### W-A — Home (orb-centered)

The home page is the Spren orb. See §9 Design System and [`./03-visual-builder/assets/spren-inspiration.png`](./03-visual-builder/assets/spren-inspiration.png) for the visual reference.

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.                                                         (R) │
│                                                                     │
│                                                                     │
│                                                                     │
│                       [ Spren orb — animated ]                      │
│                                                                     │
│                                                                     │
│             Welcome back, Reza.                                     │
│             Tell me what you're thinking about.                     │
│                                                                     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  What's on your mind?                                  [→]  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│                                                                     │
│              [⌘] [K]   ·   workflows · runs · memory · settings     │
└─────────────────────────────────────────────────────────────────────┘
```

- Wordmark `spren.` at top-left (clickable, links home). Period in `--magenta`.
- User avatar at top-right (coral-to-magenta gradient circle).
- Spren orb in upper-center, animated per §9 (idle by default).
- Greeting (Geist 400/44-52px) with name in `--magenta`.
- Subline (Geist 400/15px, `--ink-soft`).
- Input bar (white surface, 1px `--rule` border, 24px border-radius, 520-560px wide). Send button is a 44px circle, `--ink` default, `--magenta` on hover.
- Footer command-bar hint (Geist Mono 10.5px, `--ink-faint`, uppercase, letter-spaced).

No persistent left nav. Navigation = ⌘K or asking Spren.

> Note (W-B / W-C / W-D below): the left nav rail shown in earlier wireframe drafts is **removed**. Top bar on every surface: `spren.` wordmark (links home) + breadcrumb in `--ink-soft` mono + user avatar. A small (48-72px) breathing presence orb sits in the top-right corner of every non-home surface; clicking opens a chat sheet overlaying the surface.

### W-B — Workflow list (`/workflows`, populated)

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.   ›  Workflows                                          (R) │
├─────────────────────────────────────────────────────────────────────┤
│   Workflows                                  [+ New]  [+ .py]       │
│                                                                     │
│   [All]  [Visual]  [Imported]  [Meta-agent]                         │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ research-pipeline                       [visual_builder] │      │
│   │ 3 agents · last run 2h ago                               │      │
│   │ User → Researcher → Writer                               │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ daily-summary                             [code_import]  │      │
│   │ 2 agents · yesterday · ● running                         │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────┐      │
│   │ pr-review-assist                            [meta_agent] │      │
│   │ 4 agents · created today                                 │      │
│   └──────────────────────────────────────────────────────────┘      │
│                                                                ◉    │
│                                                          ↑ presence │
└─────────────────────────────────────────────────────────────────────┘
```

Filter chips left-aligned. Cards stacked vertically (no grid). Presence orb at bottom-right or top-right (Designer pins position in Stage 2).

### W-C — Canvas (`/workflows/new` redirected to `/workflows/$id`, empty)

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.  ›  Untitled workflow [✎]   [Lint ·] [+ Pattern▾]  [Save]   │
├─────────────────────────────────────────────────────────────────────┤
│ ╔════════════════════════════════════════════════════════════════╗  │
│ ║                                                                ║  │
│ ║                                                                ║  │
│ ║                <agent name=""                                  ║  │
│ ║                 model=""                                       ║  │
│ ║                 tools={...} />                                 ║  │
│ ║                                                                ║  │
│ ║         Drag from the palette, pick a pattern, or open ⌘K.    ║  │
│ ║                                                                ║  │
│ ║                                                                ║  │
│ ╚════════════════════════════════════════════════════════════════╝  │
│  Palette: [Agent] [User] [System] [Tool]                       ◉    │
└─────────────────────────────────────────────────────────────────────┘
```

Canvas background: subtle dot-grid at 24px spacing using `--rule` at 30% opacity.

### W-D — Canvas (`/workflows/$id`, with topology + node selected)

```
┌─────────────────────────────────────────────────────────────────────┐
│  spren.  ›  research-pipeline (●)   [Lint ✓] [+ Pattern▾]  [Save]   │
├──────────────────────────────────────────────────────┬──────────────┤
│ ╔══════════════════════════════════════════════════╗ │ Researcher   │
│ ║   ┌─────┐                                        ║ │ ──────────── │
│ ║   │User │ ──→  ┌──────────┐                      ║ │ Name:        │
│ ║   └─────┘      │Researcher│                      ║ │ [Researcher] │
│ ║                │  ⚙ ●     │                      ║ │              │
│ ║                └────┬─────┘                      ║ │ Model:       │
│ ║                     │                            ║ │ [opus-4.7 ▾] │
│ ║                     ▼                            ║ │              │
│ ║                ┌──────────┐                      ║ │ Instructions │
│ ║                │  Writer  │                      ║ │ [Find auth…] │
│ ║                └──────────┘                      ║ │              │
│ ╚══════════════════════════════════════════════════╝ │ Tools:       │
│  Palette: [Agent] [User] [System] [Tool]        ◉    │ [search_web, │
│                                                      │  browse_url] │
└──────────────────────────────────────────────────────┴──────────────┘
```

Edge `→`: unidirectional (default on drag). Bidirectional renders as `↔` dashed. Per Q2 lock — only these two variants. Selected node: 1px `--magenta` border + box-shadow halo.

### W-E — Right rail (agent config detail, scrolls)

```
┌────────────────────────────────────────────┐
│ <agent name="Researcher"                   │
│   model="opus-4.7"                         │
│   tools={web,scholar} />                   │
│                                            │
│ Identity                                   │
│ ─────────────────────────                  │
│ Name *           [Researcher_____________] │
│ Goal             [Find authoritative …___] │
│                                            │
│ Model                                      │
│ ─────────────────────────                  │
│ Provider/Model * [anthropic/opus-4.7 ▾]    │
│ Temperature      [0.7]                     │
│ Max tokens       [4096]                    │
│                                            │
│ Instructions *                             │
│ ┌────────────────────────────────────────┐ │
│ │ You are a research agent. Find sources │ │
│ │ on the user's topic. Cite each one.    │ │
│ └────────────────────────────────────────┘ │
│                                            │
│ Tools                                      │
│ ─────────────────────────                  │
│ [✓] search_web                             │
│ [✓] browse_url                             │
│ [ ] read_pdf                               │
│ [ ] write_file                             │
│                                            │
│ Memory retention   [session ▾]             │
│ Allowed peers      [Writer]                │
│                                            │
│ [Delete node]                    [Apply]   │
└────────────────────────────────────────────┘
```

Tool list comes from `GET /v1/tools` (Q9 lock).

### W-F — Cmdk overlay (default surface; persistent across routes)

```
┌─────────────────────────────────────────────────────┐
│ Type a command or search...                         │
├─────────────────────────────────────────────────────┤
│  Canvas                                             │
│ ▶ Add Agent node                                    │
│   Add User node                                     │
│   Add System node                                   │
│   Add Tool node                                     │
│   Insert pattern: HUB_AND_SPOKE / PIPELINE / …      │
│   Run lint                                          │
│   Save workflow                                     │
│  Navigate                                           │
│   Go to Workflows                                   │
│   Go to Runs                                        │
│   Go to Memory                                      │
│   Go to Settings                                    │
│   Go home (Spren)                                   │
│  Workflows                                          │
│   Open: research-pipeline                           │
│   Open: daily-summary                               │
│  Create                                             │
│   Create new workflow                               │
│   Import from Python                                │
└─────────────────────────────────────────────────────┘
```

Per-surface command registration: each route's `onMount` registers its commands to a Zustand slice; `onUnmount` deregisters. Cmdk filter is fuzzy (cmdk built-in).

### W-G — `+ Pattern` modal

```
┌─────────────────────────────────────────────────────┐
│  Insert pattern                                ⓧ    │
├─────────────────────────────────────────────────────┤
│   ⦿ HUB_AND_SPOKE                                   │
│       One supervisor distributes work to N peers.   │
│       For: parallel research, fan-out.              │
│                                                     │
│   ○ PIPELINE                                        │
│       Linear chain: A → B → C.                      │
│       For: stage-based processing.                  │
│                                                     │
│   ○ HIERARCHICAL                                    │
│       Multi-level supervisor tree.                  │
│       For: large teams.                             │
│                                                     │
│   ○ MESH                                            │
│       All-to-all peer communication.                │
│       For: deliberation, synthesis.                 │
│                                                     │
│   Number of agents: [3 ▾]   (where applicable)      │
│   Insert at:        [empty canvas / replace / merge]│
│                                                     │
│   [Cancel]                            [Insert]      │
└─────────────────────────────────────────────────────┘
```

Pattern insertion is a Jotai canvas-state op; agents are inserted with placeholder names + the user fills via the right rail.

### W-H — Lint surface (top + node + panel)

```
Top-toolbar lint chip:
   ┌─ green ─┐   ┌─ yellow ──┐   ┌─ red ──────┐
   │ Lint ✓  │   │ Lint  2 ⚠ │   │ Lint  1 ✕  │
   └─────────┘   └───────────┘   └────────────┘

Click chip → bottom slide-up panel:
┌─────────────────────────────────────────────────────────────────────┐
│  Lint issues                                                  ⓧ     │
├─────────────────────────────────────────────────────────────────────┤
│ ⚠ Researcher: tool 'browse_url' not in registered tools.            │
│   Did you mean 'fetch_url'?                                         │
│   [Go to node]                                                      │
│                                                                     │
│ ⚠ Writer: no edges connect to Writer.                               │
│   Add an edge from Researcher.                                      │
│   [Go to node]                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 8. Decisions locked

These were the open questions at the start of Stage 0. Each is resolved.

1. **First-touch canvas:** blank with hint text + cmdk shortcut. The `+ Pattern` modal covers "give me a starter shape." Full template gallery (curated multi-file) deferred to v0.4; the canvas load-path is template-agnostic.
2. **Edge variants:** only unidirectional (`→`) and bidirectional (`↔`) are exposed in the canvas. The framework's 4 `EdgeType` values (`INVOKE` / `NOTIFY` / `QUERY` / `STREAM`) are inert in execution today and would mislead users; deferred to a future framework session. The alternating + symmetric edge patterns are similarly untested in execution; the importer auto-converts them to plain bidirectional with a non-blocking warning + per-edge inline marker.
3. **Provenance on edit:** stays. Provenance is creation-attribution; `updated_at` + `definition_version` track edits. Saving a `code_import` workflow after canvas edits does NOT flip it to `visual_builder`.
4. **Lint surface in v0.3:** debounced POST `/v1/workflows/{id}/lint` (300ms). SSE deferred to v0.4 discussion (decide when the v0.4 advanced linter brief is written).
5. **Palette:** the strict 6-color system in §9 — warm-white ground + the Spren orb's internal peach/pink/magenta gradient. No cyan, no purple, no third hue outside the orb.
6. **Empty-state copy + tag-markup catalog:** pinned in Session 03 as part of the surface design.
7. **Workflow create flow:** create-on-route. `POST /v1/workflows` fires on `/workflows/new` mount; row redirects to `/workflows/{id}` with `draft=true` until first explicit save. Sweeper deletes empty drafts older than 24h. Drafts hidden from default list view.
8. **Importer node positions:** importer doesn't compute positions; canvas runs Dagre/ELK auto-layout on first load when positions are missing; positions persist on first save. Researcher recommends between Dagre and ELK based on bundle size + render perf.
9. **Tool-picker source:** `GET /v1/tools` endpoint (added in Session 02; if not yet wired there, Session 03 adds it). Returns framework's `AVAILABLE_TOOLS` registry + Spren-side tools. The picker reads this; the tool field accepts only registered names. Custom user-authored tools deferred to v0.4 (`v0.4-30`); meta-agent-generated tools deferred to v0.4 (`v0.4-31`).

## 9. Design System (locked)

Visual anchor: [`./03-visual-builder/assets/spren-inspiration.png`](./03-visual-builder/assets/spren-inspiration.png). The entire system is built around the soft grainy egg-form: warm white surface, the orb is the only color in the room.

### 9.1 Color palette

Strict coral-dominant system anchored on the inspiration image. The orb is the only place where multiple colors converge — coral occupies the top ~50% of the orb's gradient, magenta the bottom, with no soft-pink middle stop (the v4 reference + the inspiration both show coral → magenta directly, not coral → pink → magenta).

| Token | Hex | Role |
|---|---|---|
| `--ground` | `#FCFAF6` | Page background. Warm off-white. |
| `--surface` | `#FFFFFF` | Cards, modals, input bars, raised surfaces. |
| `--peach` | `#FFCEAA` | Spren orb top stop (lightest peach). |
| `--coral` | `#FF876C` | Spren orb upper-mid stop. Coral takes the larger share of the gradient. |
| `--magenta` | `#E82182` | Spren orb mid-lower stop + accent color (wordmark dot, user's name in welcome, send-button-hover background, input focus ring, selected canvas-node border). |
| `--magenta-deep` | `#C9146C` | Spren orb deepest stop + provenance badge for `meta_agent`. |
| `--ink` | `#1A1410` | Primary text. Send-button default background. |
| `--ink-soft` | `#6B5F58` | Secondary text. Breadcrumbs. Subline copy. |
| `--ink-faint` | `#9B8F88` | Tertiary text — used ONLY at 18px+ (polish item 8 — fails WCAG AA at smaller sizes). |
| `--rule` | `#EDE6DA` | Hairline borders. |
| `--rule-soft` | `#F4EFE5` | Border-quiet surfaces (kbd backgrounds, hover surfaces). |

The previous palette had a soft-pink (`#FF8FA8`) middle stop and `--peach-mid` (`#FFB088`) — both removed. Peach + coral + magenta + magenta-deep are now the four orb stops with coral the load-bearing middle. Magenta outside the orb appears ONLY for: wordmark dot, the user's name in the welcome, send-button hover state, input focus ring, selected canvas-node border. Peach and coral live only inside the orb.

### 9.2 Typography

Strict modern sans-serif. No serif, no italic-as-emphasis-device. Emphasis through color + weight, not typeface.

- **Display + UI:** Geist (300 / 400 / 500 / 600). Loaded via the `geist` npm package or Google Fonts.
- **Mono:** Geist Mono (400 / 500). For command-bar hint, timestamps, agent identifiers, tag-markup empty-state device, code-style chrome in the canvas.
- **Fallback chain:** `'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif` and `'Geist Mono', ui-monospace, SFMono-Regular, monospace`.

Type scale (px, semantic name):

| Name | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| display-l | 48 | 400 | -0.025em | Home greeting. |
| display-m | 32 | 500 | -0.02em | List page titles ("Workflows"). |
| display-s | 24 | 400 | -0.01em | Wordmark `spren.` |
| body-l | 16 | 400 | 0 | Default body. |
| body | 15 | 400 | 0 | Input bar text, subline copy. |
| body-s | 13 | 400 | 0 | Workflow card metadata. |
| label | 12 | 500 | 0.01em | Form labels. |
| mono | 11 | 400 | 0.02em | Provenance badges, breadcrumbs, tag-markup. |
| mono-s | 10.5 | 500 | 0.16em / uppercase | Footer command-bar hint. |

Body line-height: 1.5–1.6. Max body width: 60–65ch.

### 9.3 Spren orb — specification

Lives at `apps/web/src/components/Spren.tsx`. SVG-based, multi-layer crossfade approach (adapted from the user-approved reference at [`./03-visual-builder/spren-orb-v4.html`](./03-visual-builder/spren-orb-v4.html)) — but with state-transition discipline the v4 prototype lacks.

**Why multi-layer crossfade, not single-path-morph:** The v4 reference uses four separate `<svg>` layers (idle / typing / thinking / speaking), each with its own gradient + blur stack + path-morph timeline. Layers crossfade by toggling an `.active` class (opacity + scale transition over 700ms). This gives us:
- Per-state visuals tuned independently (typing has the three-dot cinematic split; thinking has the high-frequency pulse; etc.) without a single timeline trying to be all four.
- The "always morph back through the same converged-orb pose" behavior the user wants — every layer's "rest" pose IS the same egg-shape gradient orb, so crossfading between any two layers naturally passes through that pose at 50% opacity each.
- No `feTurbulence` per-frame redraw — the grain comes from layered `feGaussianBlur` stacks at different scales (deep / mid / core), which is what the inspiration image actually shows on close inspection.

**State machine** (driven by a single `state` prop):

```tsx
<Spren state="idle"     />  // default — slow breath, single converged orb
<Spren state="typing"   />  // input focused — three-dot vortex orbit cycle (12s)
<Spren state="thinking" />  // request in flight — fast pulse + edge-mask drift (2s)
<Spren state="speaking" />  // response streaming — amplitude pump + slow float (1.8s + 8s)
```

The component owns a small internal state machine that:
1. Tracks the *target* state from the prop and the *displayed* state.
2. On state-change, runs a **700ms crossfade** between the outgoing layer and the incoming layer.
3. **On entering `typing`**: re-keys the typing layer so the keyframe animations restart from `0%` (force-reflow via key-prop change + `<animation>` `begin="indefinite"`'s `beginElement()`). The v4 prototype's typing animation continues from wherever it was — this is fixed here.
4. **On leaving any state**: the outgoing layer fades opacity 1 → 0 over 700ms while scale transitions to 0.92, which visually "settles" the layer to the converged-orb pose regardless of where in its own timeline it was. The incoming layer fades 0 → 1 with scale 0.92 → 1.0. Result: every transition feels like a unified morph through the same calm orb shape.
5. **Reduced-motion fallback**: if `prefers-reduced-motion: reduce`, render only the idle layer with all keyframe animations replaced by `animation: none`; the orb is a static gradient egg with a soft drop-shadow.

**Single canonical "egg" path** shared by every layer's base shape (asymmetric, leans subtly right at the top per the inspiration):
```
M 250, 75
C 340, 65   415, 140  420, 250
C 425, 360  355, 425  250, 425
C 145, 425  75, 360   80, 250
C 85, 140   160, 60   250, 75 Z
```

Each layer animates this path via `<animate attributeName="d" calcMode="spline">` (matching v4's morph timing) but lands back on the canonical path at the loop edges.

**Coral-dominant gradient** (radial, `cx=50%` `cy=42%` `r=75%`):

```
0%:   #FFCEAA  100% opaque   (peach highlight)
42%:  #FF876C  100% opaque   (coral — the load-bearing middle)
82%:  #E82182  100% opaque   (magenta)
100%: #C9146C  100% opaque   (deep magenta)
```

NO soft-pink intermediate stop; coral occupies the largest band (0–42% peach→coral; 42–82% coral→magenta; 82–100% magenta→deep). The radial `fx` / `fy` focus drifts per state (idle: `fx 35-45%`, `fy 30-45%`; speaking: same range but faster cycle; thinking: wider drift `fx 30-70%`).

**Layered blur stack** (replaces feTurbulence; matches the inspiration's "alive but soft" feel):

```
deep-blur:  <feGaussianBlur stdDeviation="55">  opacity 0.65   (outer glow)
mid-blur:   <feGaussianBlur stdDeviation="25">  opacity 0.85   (body)
core-blur:  <feGaussianBlur stdDeviation="3">   opacity 1.0    (defined edge, soft)
edge-mask:  radial mask animating cx/cy in 8s   (gives the slow "breathing" highlight shift)
```

Three `<use>` elements reference the same path with the three blur filters at different opacities, masked by the edge-mask for the highlight drift. This is what the inspiration image's "grainy soft" texture actually is — not noise, but layered blur of a single gradient.

**Per-state overrides** (each layer adds these on top of the base):

| State | Path morph cycle | Gradient cycle | Per-layer animation |
|---|---|---|---|
| `idle` | 8s, 4-keyframe asymmetric morph | 8s `fx`/`fy` drift | Container floats `translateY(0 → -6 → 0)` over 8s + microscopic scale 1.02/0.98 |
| `typing` | 6/3/2s for 3 split layers | 12s `cx` drift `25%→75%` | Three-dot vortex orbit (the v4 cinematic split — dot1 swoops left, dot2/dot3 cascade in, then merge back into a single orb at 85–100% of the 12s cycle); on re-entry the keyframes restart from 0% |
| `thinking` | 3s asymmetric pulse + 1s horizontal shake `±5px` | 1.5s `fx`/`fy` rapid orbit | Container pulses 0.92 ↔ 0.98 every 2s; ground glow intensifies (opacity 0.14 → 0.22) |
| `speaking` | 8s float-idle + 1.8s scale amplitude `0.96 ↔ 1.22` | Identical to idle but faster | Container amplitude-pumps per `speak-amplitude` keyframes; subtle `feDisplacementMap` displacement scale animates 0 → 90 → 0 to simulate vocal cord vibration; idle float still runs underneath |

**Sizing:**
- Home stage: 320 × 380px container, `viewBox="-50 -50 600 600"` (lets the deep-blur layer extend outside the visible orb).
- Non-home presence indicator: 56px (50% of typical avatar size, 40px on mobile per §10.7). Same SVG, same gradient, slowed: 24s idle float instead of 8s.
- The orb is rendered inside an absolutely-positioned wrapper `.spren-wrap` with `pointer-events: auto` so it's clickable (opens the chat sheet on non-home surfaces).

**Performance:** The blur-stack approach avoids `feTurbulence`'s per-frame redraw cost. Initial benchmark target: <2ms scripting + <4ms paint per frame on WebKit2GTK 4.1 (Tauri Linux). The implementer benchmarks via `performance.measure` around a 60-frame capture and provides a static fallback if either threshold is exceeded; the fallback is the idle layer with all `<animate>` elements removed (still feels alive via the container's CSS float keyframes).

### 9.4 UI components

**Top bar (every surface):**
- Height 76px (32px vertical + 32px content + 12px breathing).
- Left: `spren.` wordmark — Geist 400/24px, period in `--magenta`. Clickable, links home (no-op on home).
- Center (non-home only): breadcrumb in Geist Mono 12px `--ink-soft`. Format: `Workflows  ›  research-pipeline`.
- Right: user avatar circle — 34px, gradient `--peach → --magenta` at 140°, white initial in Geist 500/12px, subtle drop-shadow.
- Horizontal padding: 48px desktop, scales to 24px below 640px.

**Input bar (home, also the chat input wherever Spren conversation appears):**
- Surface: white, 1px `--rule` border, 24px border-radius, padding `6px 6px 6px 24px`.
- Width: 520-560px (polish item 4 — narrower than prototype's 640px).
- Input: Geist 400/15px, placeholder `--ink-faint`.
- Send button: 44×44px circle, `--ink` default background, white arrow icon, `--magenta` background on hover (240ms `cubic-bezier(0.2, 0, 0, 1)`). Disabled state (input empty): opacity 0.4, cursor disabled, no hover.
- Focus state: border becomes `--pink`, box-shadow `0 0 0 4px rgba(232, 90, 160, 0.06), 0 12px 40px -16px rgba(232, 90, 160, 0.25)`.
- Keyboard: Enter sends, Shift+Enter newline, Esc dismisses focus.

**Workflow card (`/workflows`):**
- White surface, 1px `--rule` border, 14px border-radius, padding 14px×16px.
- Title: Geist 600/14px `--ink`.
- Provenance badge: Geist Mono 10px, 2px×7px padding, 3px border-radius, colored by type:
  - `visual_builder` → `--magenta` background, white text
  - `code_import` → `--peach` background, `--ink` text
  - `meta_agent` → `--magenta-deep` background, white text (visible in 03; wired live in 07-09)
- Metadata row: Geist 400/12px `--ink-faint`, items separated by `·` glyph.
- Optional flow preview (Geist Mono 11px `--ink-soft`): `User → Researcher → Writer`.
- Hover: border darkens (`--ink-faint` at 30%), subtle 1.005 scale, cursor pointer.

**Canvas:**
- Background: `--ground` with subtle dot-grid pattern (`radial-gradient(circle, var(--rule) 1px, transparent 1px)` at 24px spacing, 30% opacity).
- Nodes: white surface, 1px `--rule` border, 10px border-radius. Agent nodes show tag-markup label (Geist Mono 11px) above the agent name (Geist 600/13px). Selected: 1px `--magenta` border + box-shadow `0 0 0 3px rgba(232, 90, 160, 0.12)`.
- Edges: 1.5px `--ink-soft` stroke. Unidirectional: solid `→`. Bidirectional: dashed `↔`. No other variants in v0.3.
- Toolbar buttons: same shape language as the send button — `--ink` default, `--magenta` hover, 240ms transition.

**Empty states + tag-markup typographic device:**
- `<agent name="" model="" tools={...} />` in Geist Mono 12-14px, color `--ink-faint`, line-height 1.7.
- Used on: empty canvas, empty config rail (no node selected), empty workflow list.

**Command-bar hint (home footer):**
- Geist Mono 10.5px `--ink-faint`, `letter-spacing: 0.16em`, uppercase.
- `kbd` elements: `--rule-soft` background, 1px `--rule` border, 4px radius, 1px×5px padding.
- Items separated by `·` glyph in `--rule` color (lighter than the labels).

### 9.5 Spacing

4px base scale: `4 / 8 / 12 / 14 / 16 / 20 / 24 / 28 / 32 / 40 / 48 / 56 / 64 / 80`.

Home stage: 48px vertical gap orb → greeting; 48px greeting → input. Workflow cards: 12px vertical gap. Canvas nodes: 24px gap on auto-layout.

### 9.6 Motion tokens

- `--ease-out: cubic-bezier(0.2, 0, 0, 1)` — standard micro-interactions (240ms for hover/focus, 100ms for active).
- `--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1)` — gentle overshoot for hover lift (200ms).
- `--ease-organic: cubic-bezier(0.45, 0, 0.55, 1)` — orb breath / morph (6-14s loops).

All animations respect `prefers-reduced-motion`. Orb has a documented static fallback (§9.3).

### 9.7 Accessibility commitments

- All touch targets ≥44×44px (send button is exactly 44).
- `:focus-visible` distinct from `:focus` and `:focus-within` (polish item 6).
- Color contrast: `--ink` on `--ground` is ~17:1 (AAA). `--ink-soft` on `--ground` is ~5.5:1 (AA). `--ink-faint` on `--ground` is ~3.1:1 — used only at 18px+ (large-text threshold). The Designer in Stage 2 audits every use of `--ink-faint` and bumps to `--ink-soft` if size < 18px (polish item 8).
- Orb is `role="button"` `aria-label="Talk to Spren"` — clickable, focusable, opens the chat sheet.
- All canvas nodes are keyboard-navigable with Tab; node selection with Enter; node deletion with Backspace.

### 9.8 Key CSS recipes

**Multi-layer crossfade container:**
```css
.spren-wrap {
  position: relative;
  width: 320px;
  height: 380px;
}
.spren-layer {
  position: absolute;
  inset: 0;
  opacity: 0;
  pointer-events: none;
  transform: scale(0.92);
  transition: opacity 700ms cubic-bezier(0.4, 0, 0.2, 1),
              transform 700ms cubic-bezier(0.4, 0, 0.2, 1);
}
.spren-layer[data-active="true"] {
  opacity: 1;
  transform: scale(1);
}
.spren-svg {
  width: 100%;
  height: 100%;
  transform-origin: 50% 50%;
  overflow: visible;
}
@media (prefers-reduced-motion: reduce) {
  .spren-svg, .spren-svg * { animation: none !important; }
  .spren-wrap {
    filter: drop-shadow(0 12px 48px rgba(232, 33, 130, 0.25));
  }
}
```

**Per-layer keyframes** (idle shown; typing/thinking/speaking each get their own set):
```css
.spren-layer[data-state="idle"] .spren-svg {
  animation: float-idle 8s ease-in-out infinite;
}
@keyframes float-idle {
  0%, 100% { transform: scale(1)    translate(0px, 0px)   rotate(0deg);   }
  33%      { transform: scale(1.02) translate(8px, -6px)  rotate(2deg);   }
  66%      { transform: scale(0.98) translate(-6px, 8px)  rotate(-1.5deg);}
}
```

**Layered blur stack** (replaces the deprecated grain filter — see §9.3 rationale):
```html
<filter id="orb-deep-blur" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="55"/>
</filter>
<filter id="orb-mid-blur" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="25"/>
</filter>
<filter id="orb-core-blur" x="-50%" y="-50%" width="200%" height="200%">
  <feGaussianBlur stdDeviation="3"/>
</filter>
<!-- Same path is <use>'d three times at opacities 0.65 / 0.85 / 1.0 -->
```

**Input focus:**
```css
.input-bar {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 24px;
  transition: border-color 240ms cubic-bezier(0.2, 0, 0, 1),
              box-shadow 240ms cubic-bezier(0.2, 0, 0, 1);
}
.input-bar:focus-within {
  border-color: var(--pink);
  box-shadow: 0 0 0 4px rgba(232, 90, 160, 0.06),
              0 12px 40px -16px rgba(232, 90, 160, 0.25);
}
```

**Send button (ink default, magenta hover, white text):**
```css
.send {
  background: var(--ink);
  color: white;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  transition: background 240ms cubic-bezier(0.2, 0, 0, 1),
              transform 100ms ease;
}
.send:hover:not(:disabled) { background: var(--magenta); }
.send:active:not(:disabled) { transform: scale(0.96); }
.send:disabled { opacity: 0.4; cursor: not-allowed; }
```

## 10. Polish items to address inside Session 03

These are the design gaps the prototype surfaced; they're scoped INTO Session 03's implementation rather than punted to a later session. Each becomes an explicit acceptance criterion the implementer must check off.

1. **Dynamic orb states.** Implement all four (idle / typing / thinking / speaking) per §9.3. Class-based, swappable from `<Spren state="…" />`. Demo each in the manual-verify checklist. Visual regression baselines cover idle + typing minimum.
2. **Asymmetric orb path.** Tune the SVG `<path>` to lean per the inspiration image's slightly-wider-lower-right form. Currently the path in §9.3 is mirror-symmetric. Acceptance: side-by-side comparison with [`./03-visual-builder/assets/spren-inspiration.png`](./03-visual-builder/assets/spren-inspiration.png) shows visual match.
3. **Spren on non-home surfaces.** Implement the presence orb (48-72px) at top-right of `/workflows`, canvas, agent config, etc. Clicking opens a chat sheet overlaying the surface; the sheet contains the same input bar + send button. The sheet dismisses with Esc or click-outside. Designer in Stage 2 commits to exact sheet behavior (full-overlay vs side-anchored).
4. **Welcome screen tightening.** Greeting headline at 48px (Geist 400, `-0.025em` tracking). Subline cut to one sentence. Input bar width 520-560px. Add a small temporal anchor under or beside the wordmark (e.g., `Tuesday · 9:14` in Geist Mono 11px `--ink-faint`) — returning users feel Spren remembers context.
5. **Design-system docs out of the home.** Embedded design notes in the prototype broke the meditation of the home page. Ship documentation as a separate `apps/web/src/design-system/` directory with MDX, or via Storybook (Researcher to recommend approach in Stage 1).
6. **Missing interactions and states.**
   - `:focus-visible` distinct from `:focus-within` so keyboard users get clear focus indication.
   - Disabled send button when input empty (opacity 0.4, `not-allowed` cursor).
   - Keyboard: Enter sends, Shift+Enter newline, Esc dismisses focus.
   - Touch-active states on mobile (the `:active` rule applies on tap too).
   - `aria-label` on the orb (it's clickable, opens chat).
7. **Mobile responsive.** Currently the prototype assumes desktop sizes. Define breakpoints:
   - `< 640px`: orb scales to 240×280, header padding tightens to 24px, input bar full-width with 24px gutters, presence orb shrinks to 40px.
   - `safe-area-inset-*` handling for notched devices.
   - Touch targets stay ≥44px at all breakpoints.
8. **Accessibility contrast audit.** `--ink-faint` on `--ground` is ~3.1:1 — passes for large text (18px+) but fails AA for body sizes. Designer audits every `--ink-faint` use and substitutes `--ink-soft` (5.5:1) where size < 18px. Document the rule in the design-system MDX.
9. **Spring physics on orb motion.** Replace linear ease-in-out on the float/morph loops with `cubic-bezier(0.45, 0, 0.55, 1)` (defined as `--ease-organic`). Visual test against the inspiration image's organic feel.
10. **Conversation state surfaces.** Decide what happens after the user sends a message:
    - Where Spren's reply renders (below orb / next to orb / in a panel / replacing the greeting).
    - Multi-turn conversation shape (scrolling chat / last-message-only).
    - Whether the orb shrinks/moves when conversation grows.
    - This is the dynamic-states question (item 1) extended into layout. **In Session 03 the meta-agent is not yet wired live (Sessions 07-09 do that), so the "send" action fires a delay + stub response.** But the visual layout is committed in Session 03 so 07-09 can wire the live event stream without re-designing the surface.

## 11. Success criteria

From `docs/implementation/spren/bundles/A-visual-builder/test-scenarios.md`:

- **G-07** (canvas round-trip golden path): user opens canvas (after `+ New Workflow` from ⌘K), drags agent nodes, connects with edges, configures via agent form, applies a pattern preset, saves; workflow appears with `provenance=visual_builder`; reload page → canvas renders the same topology + agent config without losing layout, edge metadata, or fields.
- **U-05** (manual smoke): build a 3-agent workflow on the canvas; save; reload; canvas renders the same topology.
- **Visual regression baselines** (via Playwright's built-in `toHaveScreenshot()`): orb home (idle), orb home (typing — input focused), canvas empty, canvas with 3-agent topology, agent config form, lint issues panel.
- **E-08** (provenance filter): the `visual_builder` filter on `/workflows` includes Session 03–authored workflows and excludes others.
- **X-06** (resize Tauri window): canvas remains usable at 800×600 minimum and full-screen; orb home reflows correctly.
- **Orb states**: each of the four reactive states (idle/typing/thinking/speaking) is demonstrable via a debug toggle and visually distinct. The state-transition discipline is testable: re-entering `typing` restarts the keyframe animation from `0%` (assertable via `getComputedStyle` on a marker layer + animation iteration count); transitioning away from any state crossfades through the converged-orb pose.
- **Importer warning E2E**: importing a `.py` file containing `<~>` or `<|>` edges surfaces the warning toast + per-edge inline markers.
- **Empty draft sweeper**: a workflow created via `+ New Workflow` but never saved disappears from the database 24h later via the predicate-based sweeper.
- **GET /v1/tools surface**: the tool picker in the right rail's agent form lists every name from `marsys.environment.tools.AVAILABLE_TOOLS`; the lint chip turns yellow when an agent references a name not in that list.
- **POST /v1/workflows/{id}/lint surface**: the chip aggregates findings; clicking opens the panel; per-finding "Go to node" focuses the canvas node.

Session 03 contributes concrete Playwright tests for G-07 + the canvas-specific extensions of E-01..E-12 + X-01..X-10 + the orb state coverage + the two new endpoints.

## 12. Open research items the implementer resolves in-flight

These are the empirical-verification questions the implementer answers during implementation. The heavy-research stages are short-circuited; the implementer's only research is version + API verification against React 19 + TS 6.

- `@xyflow/react@^12.10` API (custom nodes / custom edges / `connectionLineComponent` prop / `nodeOrigin` prop / `panOnDrag` prop) + peer-dep matrix vs React 19. (Brief previously claimed "v15" — corrected to `^12.10`.)
- cmdk@^1.1 patterns (per-surface command registration via Zustand slice, fuzzy filter, accessibility) + React 19 peer-dep verified.
- shadcn primitives + Tailwind v4 idioms (`@theme inline` over CSS variables; theme override approach).
- Geist via `@fontsource/geist-sans` + `@fontsource/geist-mono` (NOT the `geist` npm package — it peers `next>=13.2.0`).
- Auto-layout: Dagre wins for v0.3 (~30 KB vs ELK's ~400 KB; nothing >15 nodes ships). Implementer still benchmarks on a 15-node imported workflow to confirm.
- Spring-physics easing curves for organic motion (`cubic-bezier(0.45, 0, 0.55, 1)` is the default in §9.6 — verify it lands organically against the v4 reference).
- Storybook vs MDX-in-`src/design-system/` for shipping the design-system docs (polish item 5).
- Provenance UX patterns from adjacent visual editors (n8n, Zapier, Linear, Retool) — what to copy for the workflow-list cards, what to avoid.

Removed from this list (locked in this revision):
- SVG `feTurbulence` perf — the orb no longer uses `feTurbulence`; it uses a layered blur stack (§9.3).
- `geist` npm package — explicitly dropped in favor of `@fontsource/*` (§2 tier line).
- Argos vs Playwright snapshot — locked Playwright (`toHaveScreenshot()`) for SP-008 (no SaaS dep).

If any of these surface a conflict with the locked design system in §9, the implementer flags it and asks before deviating. The design system is the source of truth; research informs implementation, doesn't override the brief.

## 13. Status

- [x] Tier confirmed (HIGH).
- [x] Scope boundaries confirmed (sections 3 + 4).
- [x] Files-to-DELETE list approved (section 5).
- [x] Three user journeys approved (section 6).
- [x] Skeleton wireframes approved (section 7).
- [x] Decisions locked (section 8). All 9 open questions resolved.
- [x] Design system locked (section 9). Anchored on [`./03-visual-builder/assets/spren-inspiration.png`](./03-visual-builder/assets/spren-inspiration.png).
- [x] Polish items captured for in-session work (section 10).
- [x] Success criteria affirmed (section 11).
- [ ] Acceptance criteria frozen at [`./03-visual-builder/acceptance.md`](./03-visual-builder/acceptance.md) — extracted by the `acceptance-criteria-extractor` agent on the first implementation turn, before any code is written.
- [ ] Session implementation complete (all acceptance criteria pass; polish items addressed; tests green; manual verify done).

**Next step:** the implementer reads this file end-to-end, runs the `acceptance-criteria-extractor` to freeze `./03-visual-builder/acceptance.md`, then implements. The polish items in §10 are explicit acceptance criteria — they're scoped into the session, not nice-to-haves.

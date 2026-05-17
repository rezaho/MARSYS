# Acceptance criteria — Spren Session 03 (Visual Builder)

Frozen at 2026-05-12T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

## Functional — Backend endpoints

### `GET /v1/tools`

- AC-1: A `GET /v1/tools` route is mounted on the FastAPI app (e.g., at `packages/spren/src/spren/routes/tools.py`) and is reachable.
- AC-2: `GET /v1/tools` without a valid auth token returns HTTP 401.
- AC-3: `GET /v1/tools` with a valid auth token returns HTTP 200 with a JSON body shaped `{"items": [...]}` (top-level `items` key holding a list).
- AC-4: Each element of `items` is an object with exactly the keys `name` (string), `source` (string), and `description` (string or null).
- AC-5: For every item returned, `source` equals the string `"framework"`.
- AC-6: For every framework tool present in `marsys.environment.tools.AVAILABLE_TOOLS`, there is exactly one item in `items` whose `name` equals the registry key.
- AC-7: `description` for each item equals the first non-empty line of the tool callable's `__doc__` (whitespace-trimmed); `description` is `null` when the callable has no docstring.
- AC-8: Repeated calls to `GET /v1/tools` within one server process return the same list (the registry is cached for the FastAPI process lifetime).

### `POST /v1/workflows/{id}/lint`

- AC-9: A `POST /v1/workflows/{id}/lint` route is mounted on the FastAPI app and is reachable.
- AC-10: `POST /v1/workflows/{id}/lint` without a valid auth token returns HTTP 401.
- AC-11: `POST /v1/workflows/{id}/lint` for an unknown workflow id returns HTTP 404.
- AC-12: `POST /v1/workflows/{id}/lint` for a known workflow returns HTTP 200 with a JSON body shaped `{"findings": [...]}` (top-level `findings` key holding a list, possibly empty).
- AC-13: Each finding is an object with the keys `severity`, `code`, `node_name`, `edge`, `message`, and `suggestion`.
- AC-14: `severity` is one of the strings `"error"` or `"warning"`.
- AC-15: `code` is one of the strings `"unreachable"`, `"cycle_no_escape"`, `"missing_agent_ref"`, `"unknown_tool"`, `"dangling_edge"`, `"missing_required_field"`.
- AC-16: `node_name` is either a string or `null`; `edge` is either a 2-element array of strings `[source, target]` or `null`; `message` is a string; `suggestion` is either a string or `null`.
- AC-17: A workflow that references a tool name not present in `GET /v1/tools` produces at least one finding with `code="unknown_tool"` and the offending node identified via `node_name`.
- AC-18: A workflow that references an agent identifier with no matching agent definition produces at least one finding with `code="missing_agent_ref"`.
- ~~AC-19: A workflow whose `TopologyGraph.validate_workflow()` raises `TopologyError` produces at least one finding for each parseable line of the error message, with `severity="error"` or `severity="warning"` and the affected `node_name` populated when the original message identified a node.~~ **REVISED [2026-05-12]**: Spren v0.3 topologies do not register det-nodes (Start/End/User per `marsys.coordination.execution.det_nodes`), so `TopologyGraph.validate_workflow()` would short-circuit on the `if not self.det_nodes: return` guard. The Spren-side linter therefore computes the same *category* of findings (unreachable, cycle-no-escape, dangling-edge, etc.) directly against `WorkflowDefinition` instead of regex-parsing framework error messages. AC-19 is superseded by AC-17 + AC-18 + the cycle/unreachable rule coverage in `test_lint_workflow.py`. The framework-regex-bridge approach is reconsidered when Framework Session 05 ships structured findings on a Pydantic boundary.
- AC-20: A workflow that validates cleanly (no Spren-side cross-ref issues AND `TopologyGraph.validate_workflow()` does not raise) returns HTTP 200 with `{"findings": []}`.
- AC-21: The lint endpoint never returns a 5xx for a structurally-valid request (the only non-2xx outcomes are 401 unauthenticated and 404 unknown id).

### Workflow create flow and draft sweeper

- AC-22: `GET /v1/workflows` accepts an `include_drafts` query parameter; default is `false`.
- AC-23: With `include_drafts=false` (default), the response excludes workflows where `provenance="visual_builder"` AND `topology.nodes == []`.
- AC-24: With `include_drafts=true`, the response includes empty-`visual_builder` drafts.
- AC-25: A background sweeper runs on the FastAPI lifespan handler at a 4-hour interval.
- AC-26: The sweeper deletes rows where `provenance="visual_builder"` AND `topology.nodes == []` AND `updated_at` is older than 24 hours.
- AC-27: The sweeper does NOT delete rows whose `topology.nodes` is non-empty, regardless of age.
- AC-28: The sweeper does NOT delete rows whose `provenance` is not `"visual_builder"`.
- AC-29: A first explicit `PUT /v1/workflows/{id}` save advances `updated_at` such that the workflow no longer matches the empty-draft predicate (because nodes are non-empty after save).
- AC-30: No schema migration introducing an `is_draft` column is performed; draft detection is predicate-based at query time.

## Functional — Frontend routes

- AC-31: TanStack Router defines route `/` (home).
- AC-32: TanStack Router defines route `/workflows`.
- AC-33: TanStack Router defines route `/workflows/new` and entering it triggers exactly one `POST /v1/workflows` request, then redirects to `/workflows/{returned_id}` with `replace: true`.
- AC-34: The `POST /v1/workflows` for the create flow fires from a component effect on first mount (NOT from a TanStack Router `loader` — hover-prefetch must not create ghost rows).
- AC-35: TanStack Router defines route `/workflows/$workflowId` rendering the canvas.
- AC-36: Clicking the `spren.` wordmark in the top bar from any non-home route navigates to `/`.

## Functional — Home (`/`)

- AC-37: The home renders a Spren orb component (`apps/web/src/components/Spren.tsx`).
- AC-38: The home renders a greeting (e.g., "Welcome." for first-time users; "Welcome back, {name}." for returning users) where the user's name (when present) is colored `--magenta`.
- AC-39: The home renders an input bar with placeholder text and a circular send button.
- AC-40: The home renders a footer command-bar hint (Geist Mono, `--ink-faint`, uppercase, letter-spaced).
- AC-41: The home renders a small temporal anchor (e.g., `Tuesday · 9:14`) near the wordmark in Geist Mono.
- AC-42: The home does NOT render a persistent left navigation rail.
- AC-43: The home does NOT render the four-surface command center (Now / Since you were away / Activity / Chat input).
- AC-44: Pressing Enter inside the input bar (with non-empty content) fires the send action.
- AC-45: Pressing Shift+Enter in the input bar inserts a newline rather than firing send.
- AC-46: Pressing Esc inside a focused input bar dismisses focus.
- AC-47: The send button is disabled (opacity ~0.4, `cursor: not-allowed`, no hover transition) when the input is empty.
- AC-48: The send action delivers a stub response after a delay (the live meta-agent is not wired in Session 03); the response is observable in the DOM.

## Functional — Top-bar chrome (every surface)

- AC-49: Every route renders the `spren.` wordmark on the left of the top bar; the period (`.`) is rendered in `--magenta`.
- AC-50: Non-home routes render a breadcrumb in the center of the top bar in Geist Mono, color `--ink-soft`.
- AC-51: Every route renders a user-avatar circle on the right, sized 34px, gradient `--peach → --magenta`.
- AC-52: Every non-home route renders a presence orb in the top-right area, 48-72px (40px on mobile breakpoints), reusing the same Spren SVG component, breathing/idle animation.
- AC-53: Clicking the presence orb opens a chat sheet overlay that contains an input bar and send button matching the home input.
- AC-54: The chat sheet dismisses on `Esc` keypress.
- AC-55: The chat sheet dismisses on click-outside the sheet.

## Functional — Spren orb component

- AC-56: `<Spren state="idle" />`, `<Spren state="typing" />`, `<Spren state="thinking" />`, `<Spren state="speaking" />` are all valid prop values and each renders a visually distinct DOM (e.g., distinguishable by `data-state` attribute on a layer element or equivalent).
- AC-57: The current state is observable from the DOM via an attribute or class (e.g., `data-active="true"` on the active layer with `data-state="<state>"`), so external tests can read which layer is currently visible.
- AC-58: Switching the `state` prop from any value X to any value Y triggers a crossfade lasting approximately 700ms during which both the outgoing and incoming layers are present in the DOM.
- AC-59: During the crossfade, the outgoing layer transitions from `opacity: 1` + `scale: 1` to `opacity: 0` + `scale: 0.92`.
- AC-60: During the crossfade, the incoming layer transitions from `opacity: 0` + `scale: 0.92` to `opacity: 1` + `scale: 1`.
- AC-61: Re-entering the `typing` state (transition X → typing where the previous state was already typing earlier in the session) restarts the typing layer's keyframe animation from `0%` (assertable via a key-prop change, `beginElement()`, or animation iteration-count reset visible on the element).
- AC-62: Every layer's base SVG `<path>` uses the single canonical egg-shape path (the same `d` attribute value across idle/typing/thinking/speaking layers).
- AC-63: The orb's gradient uses exactly four stops at offsets approximately `0%`, `42%`, `82%`, `100%`, with colors `#FFCEAA`, `#FF876C`, `#E82182`, `#C9146C` respectively.
- AC-64: No stop in the orb gradient uses the previously-considered soft-pink color `#FF8FA8` or any color resolved from a `--pink` CSS token.
- AC-65: The orb uses a layered Gaussian-blur stack (multiple `<feGaussianBlur>` filters at different `stdDeviation` values applied to the same path) rather than `<feTurbulence>` for grain.
- AC-66: The orb on the home stage is rendered inside a wrapper sized approximately 320×380px.
- AC-67: The presence orb on non-home surfaces is rendered at approximately 48-72px (40px at mobile breakpoint < 640px).
- AC-68: The orb element exposes `role="button"` and `aria-label="Talk to Spren"` (or equivalent accessible name) and is keyboard-focusable.
- AC-69: When `prefers-reduced-motion: reduce` is set, the orb renders only the idle layer with all keyframe animations disabled (e.g., `animation: none`).

## Functional — Cmdk command palette

- AC-70: A cmdk command palette overlay is mounted at the router root and opens on `⌘K` / `Ctrl+K`.
- AC-71: The cmdk overlay supports fuzzy filtering of commands via cmdk's built-in filter.
- AC-72: Each route registers its surface-specific commands when mounted and deregisters them on unmount (commands are stored in a Zustand slice or equivalent shared store).
- AC-73: The cmdk overlay groups commands into sections including at minimum: Canvas (route-specific), Navigate, Workflows, and Create.
- AC-74: The Navigate section contains commands to go to `/workflows`, `/runs`, `/memory`, `/settings`, and `/` (home).
- AC-75: Selecting a cmdk navigation command navigates the router to the corresponding route and dismisses the overlay.
- AC-76: Selecting `Create new workflow` from cmdk navigates to `/workflows/new`.
- AC-77: Selecting `Import from Python` opens the native file picker filtered to `.py` files.

## Functional — Workflow list (`/workflows`)

- AC-78: The list page renders a card for each workflow returned by `GET /v1/workflows?include_drafts=false`.
- AC-79: Each card renders a provenance badge with text matching the workflow's `provenance` value.
- AC-80: The provenance badge for `visual_builder` has `--magenta` background and white text.
- AC-81: The provenance badge for `code_import` has `--peach` background and `--ink`-colored text.
- AC-82: The provenance badge for `meta_agent` has `--magenta-deep` background and white text.
- AC-83: The list page renders filter chips labelled `All`, `Visual`, `Imported`, `Meta-agent`.
- AC-84: Selecting the `Visual` chip filters the visible cards to those with `provenance="visual_builder"`.
- AC-85: Selecting the `Imported` chip filters the visible cards to those with `provenance="code_import"`.
- AC-86: Selecting the `Meta-agent` chip filters the visible cards to those with `provenance="meta_agent"`.
- AC-87: Empty-`visual_builder` drafts are not rendered on the list by default (because the backend filters them).
- AC-88: The list page renders a `+ New Workflow` button that, when clicked, navigates to `/workflows/new`.
- AC-89: The list page renders a `+ Import from Python` button that opens the file picker and uploads the selected `.py` to `POST /v1/workflows/import-python`.
- AC-90: Clicking a workflow card navigates to `/workflows/$workflowId`.

## Functional — Canvas (`/workflows/$workflowId`)

- AC-91: The canvas page renders `@xyflow/react`'s flow editor.
- AC-92: A left palette is rendered listing draggable items: `Agent`, `User`, `System`, `Tool`.
- AC-93: Dragging an item from the palette onto the canvas creates a node of the corresponding type at the drop position.
- AC-94: The canvas registers custom React components for each `NodeType` (Agent / User / System / Tool) rather than using xyflow defaults.
- AC-95: The canvas registers custom React edge components for unidirectional and bidirectional edges only.
- AC-96: Unidirectional edges render as a solid line with a single arrow head (`→`).
- AC-97: Bidirectional edges render as a dashed line with two arrow heads (`↔`).
- AC-98: The canvas does NOT expose UI controls for the framework `EdgeType` values (`INVOKE` / `NOTIFY` / `QUERY` / `STREAM`).
- AC-99: The canvas does NOT expose UI controls for alternating (`<~>`) or symmetric (`<|>`) edge patterns.
- AC-100: Connecting two nodes by dragging from one node's output handle to another's input handle creates a unidirectional edge by default.
- AC-101: A top toolbar is rendered with an editable workflow name field, a Lint chip, a `+ Pattern` button, and a `Save` button (a placeholder `Run` button may also appear but is non-functional in Session 03).
- AC-102: Selecting a node opens the right rail with a configuration form for that node type.
- AC-103: The agent configuration form is implemented with React Hook Form and uses a Zod schema for validation.
- AC-104: The agent form validates required fields (at minimum `name` and `model`) and surfaces errors when the user attempts to apply with a missing required value.
- AC-105: The agent form's tool picker is populated from `GET /v1/tools` and accepts only registered tool names.
- AC-106: Selecting an edge opens the right rail with a configuration form for edge direction (uni/bi).
- AC-107: Clicking `Save` issues a `PUT /v1/workflows/{id}` carrying the current canvas state and shows a success toast on 2xx.
- AC-108: After a successful save, reloading the route (`/workflows/$workflowId`) re-renders the same topology (nodes, edges, positions, agent config fields) — a build → save → reload round-trip preserves state.
- AC-109: When the loaded workflow definition has no node positions, the canvas runs an auto-layout (Dagre or ELK) on first load and renders positioned nodes.
- AC-110: After the user drags nodes and saves, the persisted positions are used on next load (auto-layout does not re-run).
- AC-111: Provenance is creation-attribution and does NOT change on save (saving an edit to a `code_import` workflow keeps `provenance="code_import"`).
- AC-112: The canvas background renders a subtle dot-grid pattern at 24px spacing using `--rule` at low opacity.
- AC-113: A selected canvas node renders with a 1px `--magenta` border and a magenta-tinted box-shadow halo.
- AC-114: Canvas nodes are keyboard-navigable with Tab; pressing Enter selects the focused node; pressing Backspace deletes the focused node.

## Functional — Lint surface

- AC-115: The canvas top toolbar renders a Lint chip whose state reflects findings.
- AC-116: The Lint chip renders in a "green / OK" state when findings is empty, a "yellow / warnings" state when only warnings are present, and a "red / errors" state when at least one error is present.
- AC-117: Edits to the canvas trigger a `POST /v1/workflows/{id}/lint` request debounced to 300ms after the last edit.
- AC-118: Clicking the Lint chip opens a panel listing each finding with its severity icon, message, and (when present) suggestion.
- AC-119: Each finding in the panel renders a `Go to node` action; clicking it focuses/selects the canvas node identified by `node_name` (when present).
- AC-120: Findings with `node_name` set also render an inline marker on the corresponding canvas node.

## Functional — Pattern modal

- AC-121: Clicking `+ Pattern` opens a modal listing four pattern presets: `HUB_AND_SPOKE`, `PIPELINE`, `HIERARCHICAL`, `MESH`.
- AC-122: Each preset entry shows a short description.
- AC-123: The modal includes a `Number of agents` selector (active for presets where applicable).
- AC-124: The modal includes an `Insert at` selector with options `empty canvas`, `replace`, `merge`.
- AC-125: Clicking `Insert` inserts the chosen pattern into the canvas state (the new nodes/edges appear on the canvas).
- AC-126: Clicking `Cancel` dismisses the modal without modifying canvas state.

## Functional — Python import warning UX

- AC-127: When `POST /v1/workflows/import-python` returns warnings (one or more entries indicating alternating/symmetric edges auto-converted to bidirectional), the frontend renders a non-blocking toast.
- AC-128: The toast offers a `Review warnings` action that opens the canvas with the affected edges visually marked.
- AC-129: Each auto-converted edge renders with an inline marker (e.g., yellow tint) and a hover tooltip explaining the conversion (which original pattern was converted to bidirectional).
- AC-130: After a Python import with auto-converted edges, the canvas does not render any alternating or symmetric edge variant — only bidirectional.

## Functional — Design system

- AC-131: `apps/web/src/styles/tokens.css` exists and defines CSS custom properties for `--ground`, `--surface`, `--peach`, `--coral`, `--magenta`, `--magenta-deep`, `--ink`, `--ink-soft`, `--ink-faint`, `--rule`, `--rule-soft`.
- AC-132: `tokens.css` does NOT define a `--pink` CSS custom property (the previous soft-pink token is removed).
- AC-133: Tailwind v4 is wired with `@theme inline` referencing the CSS variables from `tokens.css`.
- AC-134: Geist Sans and Geist Mono are loaded via `@fontsource/geist-sans` and `@fontsource/geist-mono` (NOT via the `geist` npm package).
- AC-135: A typography scale matching the documented sizes/weights (display-l, display-m, display-s, body-l, body, body-s, label, mono, mono-s) is exposed (e.g., as Tailwind utility classes or CSS classes).
- AC-136: Motion tokens `--ease-out`, `--ease-spring`, and `--ease-organic` are defined as CSS custom properties with the documented cubic-bezier values.
- AC-137: shadcn primitives are installed for at least: button, card, dialog, dropdown, input, select, switch, textarea, toast, tooltip.
- AC-138: `--ink-faint` is not applied to any rendered text smaller than 18px (the contrast audit substitutes `--ink-soft` for sub-18px copy).
- AC-139: All interactive touch targets (send button, palette items, toolbar buttons, etc.) are at least 44×44px.
- AC-140: `:focus-visible` styles are distinct from `:focus-within` styles (keyboard focus rings differ from container focus rings).
- AC-141: The send button is exactly 44×44px, circular, `--ink` background by default, transitions to `--magenta` background on hover.

## Functional — Architecture doc reconciliation

- AC-142: `docs/architecture/spren/04-frontend-architecture.md` is updated in the same PR to reflect the design pivot: the home is orb-only in v0.3 (NOT a four-surface command center), the persistent left rail is removed, the canonical theming examples use light-mode tokens.

## Functional — Files-to-DELETE

- AC-143: `apps/web/src/routes/index.tsx`'s Session-02-placeholder content is replaced wholesale by the new orb-home implementation (no `_v2` / `_new` / `_simple` variant file ships alongside it).
- AC-144: No inline-styled placeholder cards/forms from Session 01–02 remain (Tailwind + shadcn primitives replace them; no toggle / conditional rendering layer keeps the legacy version alive).
- AC-145: No hand-written interface declarations mirroring Pydantic shapes from Session 02 remain in `apps/web/src/lib/api.ts` or elsewhere under `apps/web/src/`; client types come from the generated `api-types.generated.ts`.
- AC-146: No left-navigation-rail component from Session 01–02 ships in this session (the chrome simplifies to wordmark + breadcrumb + avatar + presence orb).

## Functional — Workspace and dependencies

- AC-147: `apps/web/package.json` declares `@xyflow/react` at version `^12.10`.
- AC-148: `apps/web/package.json` declares `cmdk` at version `^1.1`.
- AC-149: `apps/web/package.json` declares `motion` at version `^12` (consumed via the `motion/react` subpath import).
- AC-150: `apps/web/package.json` declares `@fontsource/geist-sans` and `@fontsource/geist-mono`.
- AC-151: `apps/web/package.json` declares `jotai`, `zustand`, `react-hook-form`, and `zod` as dependencies.
- AC-152: `apps/web/package.json` declares Tailwind v4 (e.g., `tailwindcss@^4`) and shadcn-related Radix primitives matching the components listed in AC-137.
- AC-153: `apps/web/package.json` does NOT declare the `geist` npm package (the one that peers `next>=13.2.0`).
- AC-154: `apps/web/package.json` does NOT declare `@argos-ci/playwright` or any Argos SaaS dependency.

## Functional — Tests required

- AC-155: Vitest unit tests exist covering the Jotai canvas state operations, Zustand cmdk-command-registration slice behavior, agent-form Zod schema validation (including required-field failures), and the Spren orb internal state machine (target/displayed state, crossfade timing, typing re-entry restart).
- AC-156: A Playwright E2E test exists that executes the J-1 golden path: open home → ⌘K → create new workflow → add Agent nodes via cmdk → configure via form → connect with edges → save → see lint chip green → click wordmark → home → cmdk → workflows → workflow appears with `visual_builder` badge.
- AC-157: A Playwright E2E test exists that executes a 3-agent build → save → reload round-trip and asserts that the topology, agent config fields, and edge metadata match before and after reload (U-05).
- AC-158: A Playwright E2E test exists that asserts the `Visual` filter chip on `/workflows` includes Session 03–authored workflows and excludes others (E-08).
- AC-159: A Playwright E2E test exists that imports a `.py` file containing alternating (`<~>`) or symmetric (`<|>`) edges and asserts the warning toast appears, per-edge inline markers render, and the canvas only renders bidirectional edges for the affected pairs.
- AC-160: A Playwright E2E test exists asserting that each of the four orb states (idle/typing/thinking/speaking) is reachable via a debug toggle or natural trigger and is observable as visually-distinct via DOM attributes (and/or screenshot).
- AC-161: A Playwright E2E test exists asserting that re-entering `typing` restarts the typing layer's keyframe animation from `0%` (assertable via `getComputedStyle` on a marker element or animation iteration property).
- AC-162: A Playwright E2E test exists asserting that the tool picker in the agent config form is populated from `GET /v1/tools` and the lint chip turns yellow when an agent references a name not in that list.
- AC-163: A Playwright E2E test exists asserting that clicking a lint-panel `Go to node` action focuses/selects the corresponding canvas node.
- AC-164: A tauri-driver E2E test exists that executes the same golden-path (J-1) inside the desktop shell.
- AC-165: Visual regression baselines exist (stored under `apps/web/tests/e2e/__screenshots__/`) using Playwright's built-in `toHaveScreenshot()` for: orb home idle, orb home typing (input focused), canvas empty, canvas with topology, agent config form, lint issues panel.
- AC-166: The visual regression test suite uses Playwright's `toHaveScreenshot()` API (NOT Argos or any third-party SaaS comparator).

## Functional — User journey checkpoints

These are end-to-end demonstration checks; each maps to an observable outcome.

### J-1 — First-time user

- AC-167: J-1 step 1: on fresh data dir, opening the Tauri shell or web app lands on `/` showing the orb, greeting `Welcome.` (no name on first launch), placeholder `What's on your mind?`, and footer command-bar hint.
- AC-168: J-1 step 3: pressing ⌘K, typing "new workflow", and selecting `Create new workflow` triggers a single `POST /v1/workflows` then a redirect to `/workflows/{ulid}` with an empty canvas, draft state, editable name `Untitled workflow`.
- AC-169: J-1 step 5: selecting `Add Agent node` from cmdk creates an Agent node at the viewport center named `Agent 1` and auto-selects it (right rail opens with the agent form).
- AC-170: J-1 step 11: after edits the lint chip transitions from a non-green state to green within 300ms+ of debouncing once topology validates.
- AC-171: J-1 step 12: clicking `Save` issues a `PUT /v1/workflows/{id}` and shows a confirmation toast; the saved row drops the empty-draft predicate (non-empty `topology.nodes`).
- AC-172: J-1 step 14: navigating to `/workflows` shows the saved workflow with the `visual_builder` provenance badge.

### J-2 — Returning user

- AC-173: J-2 step 1: when a profile name is available, the home greeting renders `Welcome back, {name}.` with the name styled in `--magenta`.
- AC-174: J-2 step 3: clicking a `code_import` workflow card opens its canvas and paints the stored topology; positions are computed via auto-layout if missing.
- AC-175: J-2 step 5: adding a `Tool` node with a tool name not in `GET /v1/tools` surfaces a lint warning with a did-you-mean suggestion (when one can be computed) on the inline marker and in the panel.
- AC-176: J-2 step 7: saving the edited `code_import` workflow keeps `provenance="code_import"` and advances `updated_at`.

### J-3 — Python file import

- AC-177: J-3 step 1: opening cmdk → `+ Import from Python` opens a native file picker filtered to `.py`.
- AC-178: J-3 step 3: a Python import that includes `<~>` or `<|>` edges creates the workflow with `provenance="code_import"`, surfaces a toast, and the response includes `warnings` describing each converted edge.
- AC-179: J-3 step 4: opening the imported workflow's canvas runs auto-layout and renders converted edges with a yellow inline marker and a hover tooltip.

## Non-functional

### Performance

- AC-180: The orb animation hits the target of <2ms scripting + <4ms paint per frame on the implementer's benchmark setup; if either threshold is exceeded, a static fallback is rendered (idle layer with all `<animate>` elements removed).
- AC-181: Lint requests are debounced to 300ms (no more than one lint request fires per 300ms quiet window after the last edit).

### Accessibility

- AC-182: All animations respect `prefers-reduced-motion: reduce` (orb falls back to a static gradient; other CSS animations stop).
- AC-183: `--ink` on `--ground` and `--ink-soft` on `--ground` pass WCAG AA contrast at body sizes; `--ink-faint` is never used on body sizes < 18px.
- AC-184: Every interactive element (send button, palette items, cmdk commands, canvas nodes, edges, orb) is reachable and operable via keyboard.

### Responsive

- AC-185: Below 640px viewport width, the orb scales to approximately 240×280 on the home, header padding tightens to 24px, the input bar becomes full-width with 24px gutters, and the presence orb shrinks to ~40px.
- AC-186: Safe-area insets (`safe-area-inset-*`) are respected on notched device viewports.
- AC-187: The canvas remains usable at 800×600 minimum viewport size (X-06).

### Security

- AC-188: All Session 03 backend endpoints require the per-launch auth token; `GET /v1/tools` and `POST /v1/workflows/{id}/lint` return 401 without it.

### Observability / error handling

- AC-189: Save failures (`PUT /v1/workflows/{id}` non-2xx) surface a user-visible toast describing the error rather than silently failing.
- AC-190: Python import failures (`POST /v1/workflows/import-python` non-2xx) surface a user-visible toast.

## Out of scope

The following are explicitly excluded from Session 03; tests asserting their presence would be wrong for this session:

- Live Run execution (the Run button being functional, AG-UI event stream subscription, live token streaming) — Session 04.
- Trace viewer (nested span tree, cost chips per span) — Session 05.
- Run history list + filtering — Session 05.
- File upload / attachments outside the Python import flow — Session 05.
- Full meta-agent surfaces (chat input wired to a real reply, inbox, activity stream). The orb's `thinking` and `speaking` states render visually but are triggered by stub inputs — Sessions 07–09.
- Memory browser / facet view — Session 06.
- Settings / secrets / budgets / meta-agent config UI — Session 10.
- Custom user-authored tools (in-app code editor) — v0.4.
- Meta-agent-generated tools (with user approval) — v0.4.
- TUI surface — v0.4.
- Tauri polish (tray, autostart, OS notifications, deep links) — v0.4.
- Triggers / schedules / channels routes — v0.4+.
- Workflow gallery (curated multi-file templates with metadata) — v0.4.
- Distinct semantic behavior for the four framework `EdgeType` values (`INVOKE` / `NOTIFY` / `QUERY` / `STREAM`) — future framework session.
- Alternating + symmetric edge patterns as user-selectable variants in the canvas — future framework session.
- SSE-based lint streaming — deferred to v0.4 (Framework Session 05 advanced linter).

## Post-implementation additions [added 2026-05-12]

Added during visual review after the initial Session 03 ship. These criteria are mandatory for the v0.3.1 cut.

### Sidebar menu

- AC-191 [added 2026-05-12]: A hamburger trigger button is rendered at the far-left of the top bar on every route, with `aria-label` "Open menu" / "Close menu" and `aria-expanded` reflecting the open state.
- AC-192 [added 2026-05-12]: Clicking the trigger opens a 280 px (desktop) / `min(320 px, calc(100vw - 48px))` (mobile) slide-in sidebar panel from the left edge, with a dim backdrop behind it.
- AC-193 [added 2026-05-12]: The sidebar contains a "Surfaces" section listing Home + Workflows (both linkable) and a "Coming soon" section listing Runs, Memory, Settings (disabled, with a one-line hint each).
- AC-194 [added 2026-05-12]: The sidebar closes on Esc, on click outside (backdrop mousedown), or on clicking any link inside (the link navigates and closes implicitly).
- AC-195 [added 2026-05-12]: The first link in the sidebar receives keyboard focus when the sidebar opens.
- AC-196 [added 2026-05-12]: The sidebar's open state lives in a Zustand slice at `apps/web/src/stores/ui.ts` so any component can call `setSidebarOpen` / `toggleSidebar`.

### Orb micro-interactions — Tier 1

- AC-197 [added 2026-05-12]: On non-home routes the presence orb renders at 80 × 80 px (56 × 56 px below 640 px viewport width), anchored to the lower-right with a 24 px gutter (16 px on mobile). The previous 56 px top-right placement is removed.
- AC-198 [added 2026-05-12]: The presence orb wrapper runs an `18 s` idle-drift keyframe animation (transform translate ± 8 px, rotate ± 1.4°), offset from the 8 s SVG breath cycle.
- AC-199 [added 2026-05-12]: Hovering an interactive presence/tiny orb (clickable variant) increases scale to 1.04 and saturation to 1.15 over 280 ms via `cubic-bezier(0.34, 1.56, 0.64, 1)`.
- AC-200 [added 2026-05-12]: Clicking the orb runs a squash-and-bounce: `scale(0.92)` while `:active` (80 ms ease-out), back to 1.0 on release. CSS-only.
- AC-201 [added 2026-05-12]: Any focus event on an `<input>`, `<textarea>`, or `contenteditable` element anywhere in the document sets `data-focus-pulse="true"` on the Spren wrapper, which triggers a 700 ms saturation pulse keyframe. The attribute resets to `"false"` after 700 ms. Debounced 200 ms so rapid focus changes don't strobe.
- AC-202 [added 2026-05-12]: The Spren component exposes a `mood` prop accepting `"attentive" | "curious" | "unsettled"`. The mood reflects on the wrapper's `data-mood` attribute. CSS attribute selectors tint the orb without touching the SVG.
- AC-203 [added 2026-05-12]: The `unsettled` mood doubles the idle-drift rate (18 s → 9 s loop) for non-stage sizes via a CSS `animation-duration` override keyed on `[data-mood="unsettled"][data-size="presence" | "tiny"]`.
- AC-204 [added 2026-05-12]: Default `mood` is `"attentive"` when the prop is omitted.

### Performance kill switch

- AC-205 [added 2026-05-12]: `apps/web/src/lib/perf-monitor.ts` exports a `startPerfMonitor()` function that samples `requestAnimationFrame` deltas over rolling 60-frame windows. When the median exceeds 20 ms for two consecutive windows, it sets `document.documentElement.dataset.sprenDegraded = "true"`. When the next window's median drops back under 20 ms, the attribute is removed.
- AC-206 [added 2026-05-12]: `startPerfMonitor()` is invoked once from `apps/web/src/main.tsx` on app boot.
- AC-207 [added 2026-05-12]: The monitor resets its sample buffer when `document.hidden` becomes true and skips its sampling tick until visibility returns.
- AC-208 [added 2026-05-12]: When `data-spren-degraded="true"` is set on `<html>`, CSS in `Spren.css` freezes the wrapper's idle-drift animation on presence + tiny sizes via `animation: none`.

### Reduced-motion fallback

- AC-209 [added 2026-05-12]: Under `prefers-reduced-motion: reduce`, the state-transition crossfade duration is shortened from 700 ms to 200 ms (state transitions still carry information).
- AC-210 [added 2026-05-12]: Under `prefers-reduced-motion: reduce`, a 12 s opacity ripple (0.92 ↔ 1.0) runs on the wrapper as the "still present" signal — pure opacity, no transform.
- AC-211 [added 2026-05-12]: Under `prefers-reduced-motion: reduce`, drift / shake / vortex / hover-scale animations are disabled. The hover saturation pulse is preserved.

### `/workflows/new` reliability

- AC-212 [added 2026-05-12]: When `/workflows/new` mounts and the capabilities provider is still bootstrapping, the page renders "Connecting to Spren…" until capabilities resolve.
- AC-213 [added 2026-05-12]: When capabilities fail (no auth token or bootstrap exception), the page renders "Can't reach the Spren sidecar." plus the underlying error message and a link back to `/`.
- AC-214 [added 2026-05-12]: While the create mutation is pending less than 5 s, the page renders "Setting up a new canvas…" with the running orb in `thinking` state.
- AC-215 [added 2026-05-12]: When the create mutation has been pending for 5 s or longer, the page additionally surfaces the elapsed time and a Retry button + Cancel-to-`/workflows` link.
- AC-216 [added 2026-05-12]: When the create mutation errors, the page renders "Couldn't create workflow." plus the underlying error message and a Retry button + Cancel link. The error is `console.error`'d for dev visibility.
- AC-217 [added 2026-05-12]: When the mutation succeeds, the page renders "Opening canvas…" briefly while the router redirects to `/workflows/{id}` with `replace: true`.

### Test coverage additions

- AC-218 [added 2026-05-12]: Vitest unit tests cover: sidebar trigger toggles store state; sidebar panel mounts only when open; backdrop click closes; aria-expanded reflects open state (4+ tests in `tests/sidebar.test.tsx`).
- AC-219 [added 2026-05-12]: Vitest unit tests cover: mood prop reflects on data-mood, default is attentive, focus-pulse fires on input focus (5+ tests added to `tests/spren-orb.test.tsx`).

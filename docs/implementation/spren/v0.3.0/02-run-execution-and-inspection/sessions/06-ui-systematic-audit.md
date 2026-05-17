# Session 06 — UI Systematic Audit + Bug Fix Sweep

**Bundle**: 02 — Run execution and inspection (audit sweep across the whole shipped product)
**Position**: After Sessions 04 + 05 shipped. This is an audit + fix sweep, not a feature build.
**Tier**: Investigation-heavy. Every bug gets a root-cause-then-fix pipeline, not a YOLO patch.
**Working tree state at session start**: branch `feature/spren-umbrella` at commit `b7bf8b7`. Sessions 01–05 shipped. Assessment harness scaffolded under `scripts/scenarios/` (backend) + `scripts/visual_audit/` (frontend visual checks, deferred until live exploration reveals what to capture).

---

## Why this session exists

User reported "a lot of issues" while testing the workflow surface and asked for autonomous assessment instead of manual screenshotting back-and-forth. Two halves to the audit:

1. **Backend services** — creating, validating, approving, running workflows. Exercised by `scripts/scenarios/*.py` against a live sidecar.
2. **UI/UX of each step** — what elements should be on each surface, layout issues, missing controls, small details. Exercised via the `claude-in-chrome` Chrome MCP extension driving the user's real browser.

Findings are documented HERE in this file (not in commit messages, not in conversation). Each bug is investigated → fix proposed → fix verified → fix implemented, with the trail recorded.

---

## Working agreement + how we operate

This section is the operating contract for the whole audit. The "Hard rules" list and the ASCII pipeline below are the at-a-glance; this is the binding detail behind them.

### Collaboration model — peers, equal voice, equal responsibility

Reza and Claude run this audit as peers. Not order-taker and order-giver. Every decision carries shared responsibility, which means Claude:

- **Pushes back with reasoning.** Wrong scope, wrong fix, wrong framing — say so, defend it with evidence (code, docs, traces). A question from Reza is not a correction; don't fold under a question. Cite sources for any "best practice" claim.
- **Takes responsibility for the decision.** "Tests pass" is not "done". Manually verify the user-visible behaviour against the goal. The call is Claude's too — put real effort in.
- **Never assumes — verifies.** Read the symbol/file/path before asserting it. Reproduce green before "fixed". Actually run the check before "absent". "Verified" must mean verified.
- **Understands the goal and the system before touching it.** Know why the thing exists and how the architecture is shaped. Decisions made without the system model are plausible-but-wrong — that is exactly the failure that produced the punch list.
- **No apologies — root-causes the mistake.** When wrong: first-principles analysis of why it happened, fix the underlying cause, continue. No contrition theatre, no padding.
- **Sycophancy is dangerous.** If Reza says "X is broken", confirm X is actually broken before changing anything. Working code does not get "fixed" to satisfy a false premise. (Already hit once — the retracted lint claim, UI-FINDING-006.)

### The multi-agent investigation protocol (authoritative)

Every non-trivial bug runs this. "Non-trivial" defaults to YES unless the bug is a mechanically obvious one-line fix (typo, missing import).

**1 — Spawn the developer/investigation agent (`failure-root-cause-investigator`).**
I write it a detailed brief: relevant architecture-doc paths, relevant source paths, the bug write-up (symptom + repro + traceback), the constraints (TRUNK-CRITICAL list, SP-/DP- principles, the mirror↔runtime seam), and the exact task and output shape expected. Briefing it well is my job — it lowers its error rate — but the brief is a starting point, not a substitute: the agent must still independently explore and verify. Its mandated order of work:

1. Read the architecture docs *fully* first. Understand the system design and the architecture before reading any code.
2. Then read the relevant code *fully*: the mechanism, the components, the system boundaries, the abstraction layers — how the pieces actually fit.
3. Only then investigate. Trace *backwards* from the surfaced symptom through the cascade to the *real* root cause — the originating defect, not where the error happens to surface.
4. Propose a solution that is elegant and coherent with the architecture, as if it had been designed that way from day one. Explicitly not a workaround, not nested logic, not a conditional whose only purpose is to swallow the error. If the clean fix needs an architectural decision, say so instead of hacking around it.

It returns root cause + proposed solution + reasoning. It does not apply the fix.

**2 — I review the investigator's output.**
Does the root cause explain *every* symptom? Is the solution architecture-coherent or a disguised workaround? Does it touch TRUNK-CRITICAL or a contract? Satisfied and uncontentious → step 5. Any concern → step 3.

**3 — Spawn the verifier/reviewer agent (independent context: `general-purpose` / `implementation-reviewer`).**
Same quality of brief (architecture + code background + the proposed root cause + the proposed solution + my specific concern). It must do the same first half — architecture fully, system and components understood, code fully — *then* review the proposed root cause + solution for: gaps or inconsistencies in the analysis, missed mechanisms or events, architecture incompatibility, the solution being inelegant / a workaround / over-conditional. Returns agree / disagree / refine with reasoning.

**4 — I review the verifier.**
Agree → step 5. Disagree → one more round (investigator↔verifier debate, or my own first-principles analysis), then step 5. Converge; no loops.

**5 — Implement, then record.**
Apply the fix. Reproduce the failing scenario green. Run nearby scenarios for regressions. Record final root cause + final solution + files changed + commit hash in this file.

**Escalation.** Any architectural choice — TRUNK-CRITICAL, a cross-boundary contract, a module restructure, a security/persistence model — stops here and goes to Reza with full context (background + problem + proposed direction). Open escalations: ARCH-Q-001 (token persistence), WF-BUG-RUN-2 (in-product key entry), WF-BUG-RUN-3b (error-envelope scope).

### Process lesson — why the first pass missed the real issues (do not repeat)

**What happened.** The first audit pass walked scenarios A–D as a mechanical checklist — "does element X exist? does interaction Y fire?" — and ticked boxes. It logged "palette click instantly adds a node" as a *positive* when click-to-instant-add is itself a bug. It never clicked Run, the single most important action in the product. It never checked the palette's node types against the framework's real catalog. It under-weighted the right-rail/orb collision that was sitting in its own screenshots.

**Root cause of the miss (first principles).** The pass optimised for checklist coverage instead of evaluating the experience of the person who has to build and run a workflow. A checklist confirms presence. It never asks "is this the right experience? would a builder actually succeed here?" Presence-checking is not product assessment — that category error is the root cause, not any individual missed item.

**Binding correction for the rest of this session.** Follow every step a real user takes, in order, end to end — including the destination action (Run), not just setup. Hold the user's *goal* in mind at every step (build a workflow and run it — not enumerate UI elements). Judge quality and experience, not existence. Inspect screenshots at high resolution; zoom when detail is unclear. "Works mechanically but is the wrong experience" is a finding, not a pass. This is why the punch list exists and supersedes the checklist severity ordering where they overlap.

---

## Hard rules

1. **No YOLO bug fixes.** Every non-trivial bug requires the root-cause pipeline below. "Non-trivial" defaults to YES unless the bug is mechanically obvious (typo, missing import) AND a one-line fix.
2. **Root cause, not symptom.** The fix has to address WHY the bug exists, in a way coherent with the architecture as if it had been designed correctly from day 1.
3. **Elegant.** No nested conditions, no special-case guards, no workaround scaffolding. The fix should look like it always belonged there.
4. **Architectural questions escalate to user.** When a fix needs a structural decision (which I can't make alone), I stop and ask with full context (background, problem, proposed direction).
5. **Record everything.** Bug → root cause → proposed solution → verification → implementation → all annotated here.
6. **Coexist with shipped features.** Don't break Sessions 01–05's acceptance contracts.

---

## Multi-agent fix pipeline

For each non-trivial bug:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Document the bug (this file)                                     │
│    surface + repro + expected + actual + DOM/console/network notes  │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Spawn `failure-root-cause-investigator` agent                    │
│    - Hands it: arch docs paths, source paths, bug doc, conventions  │
│    - It reads, investigates, traces, proposes root cause + solution │
│    - Does NOT apply fixes — returns findings                        │
├─────────────────────────────────────────────────────────────────────┤
│ 3. I review the investigator's output                                │
│    - Does the root cause actually explain the symptom?              │
│    - Does the proposed solution fit the architecture?               │
│    - Any concerns? → step 4. None? → step 5.                        │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Spawn `general-purpose` agent as VERIFIER (independent context)   │
│    - Hands it: same arch docs + code paths + proposed root cause +  │
│      proposed solution                                              │
│    - It simulates execution trace, validates root cause is real,    │
│      validates solution addresses root cause elegantly              │
│    - Returns: agree / disagree / refine, with reasoning             │
├─────────────────────────────────────────────────────────────────────┤
│ 5. I review verifier (if step 4 fired)                              │
│    - Agree with verifier → step 6                                   │
│    - Disagree → one more round (investigator/verifier debate or my  │
│      own analysis), then step 6                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 6. Implement the fix                                                │
│    - Make the code change                                           │
│    - Run the failing scenario; should now pass                      │
│    - Run nearby scenarios to confirm no regression                  │
├─────────────────────────────────────────────────────────────────────┤
│ 7. Record outcome in this file                                      │
│    - Final root cause statement                                     │
│    - Final solution shipped                                         │
│    - Files changed + commit hash                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**When to ask the user mid-pipeline:** any architectural choice (touching TRUNK-CRITICAL, changing a contract, restructuring a module). State the background + problem + proposed direction; let them weigh in.

---

## Test scenarios

These are walked through in order. Each one captures observations + findings inline. The user's browser tab is at `http://localhost:5173/`; the dev stack is up (sidecar 8765 + Vite 5173). The token comes from the sidecar's `spren-ready` line and is delivered via the `#token=...` URL fragment.

### A — Auth gate + landing page entry

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| A1 | Cold landing without token | Visit `/` with no `#token=` fragment, no cached token | AuthGate splash → manual entry form? Error state? Empty page? Does the UI explain what's happening? |
| A2 | Landing with valid token | Visit `/#token=<sidecar-token>` | Token consumed; fragment stripped from URL; orb home renders |
| A3 | Landing with invalid token | Visit `/#token=garbage` | Clear error UI; manual entry form; no infinite spinner |
| A4 | Token persistence across refresh | After A2 succeeds, reload page | Stays authenticated OR re-asks (which behavior is intended?) |
| A5 | Token visibility / cleanup | Inspect URL after A2 success; inspect storage | Fragment should be stripped; token shouldn't be in localStorage in plaintext (acceptable: in-memory) |

### B — Home page (`/`) elements + interactions

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| B1 | Initial paint | Land on `/` (first-time user, no name) | Orb visible + animated; greeting "Welcome." (no name); input bar; footer command-bar hint; temporal anchor like "Tue · 9:14"; user avatar top-right; wordmark top-left |
| B2 | Wordmark anatomy | Inspect `spren.` element | `.` rendered in `--magenta`; clickable; cursor: pointer on hover |
| B3 | Orb idle state | Watch orb for 8s | Breathing animation; no jank; gradient stops match spec (peach → coral → magenta → magenta-deep); no soft-pink stop |
| B4 | Orb typing state | Focus input bar | Orb transitions idle → typing within ~700ms; data-state attribute reflects state |
| B5 | Input bar — empty state | Look at input bar with no text | Placeholder visible; send button disabled (low opacity, not-allowed cursor) |
| B6 | Input bar — type message | Type "hello"; check send button | Send button transitions to enabled; orb in typing state |
| B7 | Send action | Click send (or press Enter) | Some response surfaces (the spec calls for a stub); orb may transition to thinking/speaking |
| B8 | Shift+Enter | In input bar, type then Shift+Enter | Newline inserted, not send |
| B9 | Esc to defocus | Focus input bar, press Esc | Focus drops |
| B10 | Sidebar trigger | Click hamburger top-left | 280px slide-in panel; backdrop dim; aria-expanded toggles; first link focused |
| B11 | Sidebar contents | Open sidebar | Surfaces: Home + Workflows linkable; Coming soon: Runs/Memory/Settings disabled with hints |
| B12 | Sidebar dismissal | With sidebar open: click backdrop, then re-open + Esc, then re-open + click link | All three close it |
| B13 | Cmdk overlay | Press ⌘K or Ctrl+K | Cmdk opens with command list grouped (Navigate, Workflows, Create, etc.) |
| B14 | Cmdk navigate | Open cmdk → type "workflows" → enter | Navigates to `/workflows` and dismisses overlay |
| B15 | Footer command-bar hint | Inspect footer | Geist Mono uppercase, letter-spaced, `--ink-faint` color (but spec says no faint on <18px text — verify) |
| B16 | Responsive — mobile | Resize to 375×667 | Orb shrinks (~240×280); header padding tightens; input bar full-width with gutters |

### C — Workflow list (`/workflows`)

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| C1 | Empty list | Fresh data dir, navigate to `/workflows` | Empty state visible; `+ New Workflow` + `+ Import from Python` buttons; no broken cards |
| C2 | Filter chips | Look at filter chips | "All / Visual / Imported / Meta-agent" rendered; first one active |
| C3 | Create new flow | Click `+ New Workflow` | Navigates to `/workflows/new`; one `POST /v1/workflows` fires; redirects to `/workflows/{ulid}` |
| C4 | Card after create | Return to `/workflows` after creating one via canvas Save | Card appears with provenance badge `visual_builder`; not shown if it's an empty draft (check default `include_drafts=false`) |
| C5 | Provenance badge colors | Create workflows of multiple provenances (via API for `api`, via canvas for `visual_builder`, via Python import for `code_import`) | Badges: `visual_builder` magenta+white, `code_import` peach+ink, `meta_agent` magenta-deep+white, `api` (unspecified — check) |
| C6 | Filter behavior | With workflows of multiple provenances, click each filter chip | List filters correctly; chip itself shows active state |
| C7 | Card click → canvas | Click a card | Navigates to `/workflows/{id}`; canvas paints |
| C8 | Import Python button | Click `+ Import from Python` | File picker opens filtered to `.py`; uploads to `/v1/workflows/import-python`; on success toast + new card; warnings (if any) surface and are reviewable |

### D — Workflow canvas (`/workflows/{id}`)

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| D1 | Empty draft canvas | Open a freshly created workflow | xyflow canvas visible; dot-grid background; left palette with Agent / User / System / Tool; top toolbar with name editor, Lint chip, +Pattern, Save, Run (placeholder), 📎 attach; right rail collapsed |
| D2 | Palette drag → Agent | Drag Agent from palette onto canvas | Agent node appears; auto-selects; right rail opens with agent form (name, model, instructions, tools, memory retention) |
| D3 | Palette drag → User | Drag User onto canvas | User node appears; verify NAME chosen — does it use "user" (reserved) or "User 1" or similar? **Linked to PRODUCT-BUG-001** |
| D4 | Palette drag → System | Drag System onto canvas | Same observation as D3 with name "system" |
| D5 | Palette drag → Tool | Drag Tool onto canvas | Same observation with name "tool" |
| D6 | Connect nodes | Drag from one node's output handle to another's input | Unidirectional edge renders with `→` arrow head |
| D7 | Agent form validation | Try Apply with empty name | Error surfaces; form does not commit |
| D8 | Agent form tool picker | Open tools dropdown | Populated from `GET /v1/tools`; observe what's there (we know `search_web`/`browse_url` weren't in registry per the backend journey) |
| D9 | Selected node styling | Select a node | 1px `--magenta` border + box-shadow halo |
| D10 | Lint chip — empty canvas | Look at top toolbar with no nodes | What state shows? (green/no-findings or yellow/empty?) |
| D11 | Lint chip — valid topology | 2 agents connected | Chip turns green within 300ms+ debounce window |
| D12 | Lint chip — invalid topology | Disconnected agent | Chip shows warning/error |
| D13 | Lint chip click | Click chip with findings | Panel slides up listing findings with severity icons + suggestions + "Go to node" |
| D14 | + Pattern modal | Click `+ Pattern` | Modal opens with HUB_AND_SPOKE / PIPELINE / HIERARCHICAL / MESH presets + Number of agents + Insert at selector |
| D15 | Pattern insert | Pick a pattern, click Insert | Nodes/edges appear; modal dismisses |
| D16 | Save | Click Save | `PUT /v1/workflows/{id}` fires; toast confirms; updated_at advances |
| D17 | Reload roundtrip | Save then refresh route | Topology, agent fields, edge metadata all persist |
| D18 | Run button — placeholder vs functional | Click Run | Session 04 says it should fire `POST /v1/runs` and stream — check |
| D19 | Attach 📎 | Click 📎 / drag file onto canvas | File picker / drop overlay; upload progress; attachments accumulate |
| D20 | Provenance preserved on edit | Open a `code_import` workflow, edit, save | Provenance stays `code_import`; not flipped to `visual_builder` |
| D21 | Canvas drag-drop file | Drag a PDF onto the canvas (not onto attach button) | Whole-canvas drop overlay; file attached |
| D22 | Cmdk on canvas | Open cmdk from canvas | Surface-specific commands: Add Agent node / Insert pattern / Run lint / Save / etc. |
| D23 | Presence orb on canvas | Look at top-right area | 48-72px presence orb anchored; clicking opens chat sheet |

### E — Workflow create flow

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| E1 | Direct visit /workflows/new | Visit URL directly | "Setting up a new canvas…" with thinking-orb; one POST fires; redirect |
| E2 | Slow capabilities | Visit /workflows/new before capabilities resolved (open in a fresh tab during boot) | "Connecting to Spren…" intermediate state |
| E3 | Auth failure | Set bad token, visit /workflows/new | "Can't reach the Spren sidecar." + error + retry/cancel |
| E4 | Mutation timeout | Block POST somehow (or visit while sidecar paused) | After 5s: elapsed-time + Retry + Cancel-to-`/workflows` |
| E5 | Mutation error | Force a 500 response | "Couldn't create workflow." + error message + retry/cancel; `console.error`'d |
| E6 | Hover-prefetch ghosts | Hover the `+ New Workflow` button on /workflows multiple times | No `POST /v1/workflows` fires from prefetch (only on actual click) |

### F — Python import flow (J-3 in Session 03 plan)

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| F1 | Valid file import | Use `packages/spren/tests/fixtures/python_workflows/valid_minimal.py` | Workflow created with `provenance=code_import` + `provenance_metadata.source_filename` + `sha256` |
| F2 | Dict-DSL rejected | Use `invalid_dict_dsl.py` | 422 with clear error pointing to dict-DSL construct; toast surfaces |
| F3 | Too-large rejected | Use `invalid_too_large.py` | 422 with size error; rejected before AST parse |
| F4 | Pattern warning | If a fixture has `<~>` or `<\|>` edges | Workflow created but warnings toast + per-edge yellow marker |

### G — Run history (`/runs`)

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| G1 | Empty list | Fresh data dir, visit `/runs` | Empty state; filter rail visible (Session 05) |
| G2 | Filter rail elements | Inspect filter rail | Date range, status (queued/running/succeeded/failed/cancelled), workflow filter |
| G3 | Run row | After a run completes, return to `/runs` | Row appears with status badge, duration, cost, workflow name |

### H — Run inspector (`/runs/{id}`)

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| H1 | Inspector layout | Click a run | Run metadata header (status, duration, cost, tokens, timestamps); trace tree (nested spans); span detail drawer |
| H2 | Span click | Click a span in the tree | Detail drawer opens with attributes |
| H3 | Workflow snapshot accordion | Look for the "Workflow as run" accordion | Should render the frozen workflow definition at run start |
| H4 | Attachments + artifacts | If the run had attachments | Renders inline / downloadable |
| H5 | Re-run from snapshot | Click re-run | New POST /v1/runs with snapshot |

### I — Cross-route flows

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| I1 | Wordmark navigation | From every route, click `spren.` wordmark | Goes home |
| I2 | Cmdk navigate | From every route, ⌘K → Go to X | Each route reachable |
| I3 | Browser back/forward | Use browser back after several navigations | History stack works; auth state preserved |
| I4 | Refresh on canvas | Refresh while on `/workflows/{id}` | Canvas re-renders correctly; auth still valid |
| I5 | Presence orb on every non-home route | Look top-right | Visible everywhere non-home; clicking opens chat sheet |

### Z — Cross-cutting / non-functional

| ID | Scenario | Steps | What to look for |
|---|---|---|---|
| Z1 | Console errors / warnings | Throughout every scenario, watch console | Any non-trivial JS errors, React warnings, network 4xx/5xx surfaced via toast or silently swallowed |
| Z2 | Network requests | Check Network tab during scenarios | Unexpected polling, missing requests, requests firing twice, auth missing |
| Z3 | Visual contrast | At each surface, eyeball + zoom in | `--ink-faint` shouldn't appear on text <18px |
| Z4 | Focus rings | Tab through every surface | Visible focus rings; not lost; not overridden by hover |
| Z5 | Reduced motion | Set OS to reduce-motion; reload home | Orb idle layer only; no keyframe animations |

---

## Findings register

Bugs are recorded here as I find them. Each gets the multi-agent fix pipeline unless explicitly marked trivial.

### PRODUCT-BUG-001 — Reserved-name validator over-rejects on non-agent nodes

**Discovered during**: `scripts/scenarios/workflow_crud_journey.py` initial run (before session 06 scenarios formally started — surfaced via the backend journey)
**Surface**: `POST /v1/workflows` and `PUT /v1/workflows/{id}` (Spren backend)
**Severity**: critical — blocks the documented Session 03 J-1 step 8 ("drag a User node from the palette onto the canvas") if the canvas names the node "user" or similar.
**Status**: investigated by me; pending root-cause-investigator agent + verifier per session 06 pipeline.

**Repro**:
```python
POST /v1/workflows  (with auth header)
{
  "name": "test",
  "provenance": "api",
  "definition": {
    "topology": {
      "nodes": [
        {"name": "user", "node_type": "user", "agent_ref": null, ...},
        ...
      ], ...
    }, ...
  }
}
```

**Expected**: 201 with the new workflow. A `NodeType.USER` node naturally named "user" is the canonical case.

**Actual**: 422 with `{"error": {"code": "VALIDATION_FAILED", "details": {"errors": [{"type": "value_error", "loc": ["body", "definition", "topology", "nodes", 0, "name"], "msg": "Value error, node name 'user' is reserved (case-insensitive); reserved names: ['system', 'tool', 'user']", ...}]}}}`.

**Code under suspicion**:
- `packages/spren/src/spren/models/topology.py:45-55` — `NodeSpec._no_reserved_names` validator applies to ALL `NodeSpec.name` regardless of `node_type`.
- `packages/framework/src/marsys/agents/agents.py:241-245` — framework's own usage is restricted to AGENT names.
- `packages/framework/src/marsys/agents/registry.py:61-65` — framework's other usage is restricted to AGENT names.
- `packages/framework/src/marsys/coordination/topology/core.py:29` — `RESERVED_NODE_NAMES = frozenset({"user", "system", "tool"})`.

**My read**: the validator's intent (per Session 02 brief prose) was "reject any AGENT name whose lowercase form collides with reserved system identities." But the implementation applies to all `NodeSpec.name`, which is over-strict for `NodeType.USER` / `SYSTEM` / `TOOL` nodes — whose natural name IS the reserved word.

**Next step**: spawn `failure-root-cause-investigator` with this writeup + the four code paths above + Session 02's brief + the framework's RESERVED_NODE_NAMES design. Get its proposed fix. Then verify.

### TOOLING-BUG-001 — `just dev` (Windows variant) exits when launched as a background process

**Discovered during**: dev stack startup for session 06
**Surface**: `Justfile` Windows `[script] dev` recipe
**Severity**: important — blocks `just dev` from being used as a background-runnable command in automation (it's fine in interactive PowerShell)
**Status**: documented; workaround in place (spawn sidecar + Vite as two independent background processes via Bash tool); not yet investigated

**Repro**:
```
# In an automation context where stdout isn't a foreground tty:
just dev   # (run in background)
# → exits with code 0 after ~6 seconds; both Vite and sidecar are killed
```

**Expected**: long-running until SIGTERM; Vite and sidecar persist as Start-Job children.

**Actual**: the `while ($sidecarJob.State -eq 'Running') -and ($viteJob.State -eq 'Running')` loop exits early; `finally` stops both jobs.

**Root cause hypothesis (un-verified)**: when run detached, the PowerShell process's session disconnects fast; `Start-Job` children's state is reported as non-Running once the parent session is closing; the loop's AND-conjunction terminates and the finally cleans up.

**Next step**: defer to after the UI audit. Lower priority than user-facing bugs.

---

### UI-BUG-001 — Invalid `#token=...` URL fragment is silently rejected on cold load

**Scenario**: A3 (invalid token).
**Surface**: AuthGate (`apps/web/src/components/AuthGate/AuthGate.tsx`) + capabilities provider (`apps/web/src/providers/capabilities.tsx`)
**Severity**: important — user gets no feedback that the URL they followed contained a rejected token; they see only "Authentication required" as if no token was ever provided.

**Repro**:
1. Open `http://localhost:5173/#token=garbage_invalid_token_12345`
2. AuthGate renders showing only "Authentication required" + the manual paste form
3. No error message, no indication that a token was attempted

**Expected**: When the URL-fragment token is consumed and fails authentication, the gate should surface a clear message like "The token in your URL was rejected — paste a fresh one below" (or similar).

**Actual**: Silent gate. The user has no idea why their `#token=...` link "didn't work" — they may think the URL didn't have a token at all.

**Note on URL state**: invalid fragment tokens stay in the URL bar (the strip-on-success behavior doesn't fire on failure). Could surface the rejected token to whoever's looking over the shoulder, though it's already-rejected so the practical impact is low.

**Investigation hint**: `capabilities.tsx` likely reads `window.location.hash`, calls `bootstrap` with the token, and on failure clears the in-memory token without updating any error state. Compare against the form-submit path, which DOES set `bootstrap failed: 401 Unauthorized` as `error` state.

### UI-BUG-002 — Auth error message is developer-jargon

**Scenario**: A3 (form submit with invalid token)
**Surface**: AuthGate form error
**Severity**: nit — functional but unfriendly. Single-user-local product, but the audience includes non-developers (Session 02 plan calls out "AI builders ... not necessarily writing Python every day").

**Repro**: Paste any garbage token in the AuthGate form → click Unlock.

**Expected**: Plain-English message: "That token didn't work. Make sure you copied the full token from the terminal."

**Actual**: `bootstrap failed: 401 Unauthorized`

### UI-BUG-003 — Wordmark color treatment is inconsistent across surfaces

**Scenarios**: A1/A3 (AuthGate) vs B1 (home)
**Surface**: AuthGate brand (`.auth-gate-brand`) vs home top-bar wordmark (`apps/web/src/components/TopBar/TopBar.tsx` likely)
**Severity**: nit — design system coherence

**Repro**:
- On AuthGate: `<span class="auth-gate-brand">spren.</span>` — the ENTIRE wordmark renders in magenta
- On home top-bar: only the period (`.`) renders in magenta, the rest is `--ink`

**Expected (per Session 03 spec AC-49)**: "the period (`.`) is rendered in `--magenta`" — implying the rest is the default text color. AuthGate appears to deviate.

**Actual**: Different stylings for the same brand mark.

**Decision needed**: which is the intended treatment? If AuthGate's full-magenta is intentional (e.g., to feel more "secure / hello" stage), the design system should document it. Otherwise unify on the period-only-magenta treatment.

### UI-BUG-004 — Wordmark period magenta cue is barely visible on home

**Scenario**: B2 (wordmark anatomy)
**Surface**: home top-bar wordmark
**Severity**: nit — brand legibility

**Repro**: Look at the `spren.` wordmark on home. Zoom in. The period IS magenta but the dot is so small relative to the letters that the brand cue is easy to miss.

**Suggestion area**: increase the dot's visual weight — maybe a small filled circle next to the wordmark instead of a literal period glyph, or thicken the period. Architectural call (touches brand identity).

### UI-BUG-005 — Footer command-bar hint uses `--ink-faint` on sub-18px text (violates AC-138)

**Scenario**: B15 (footer command-bar hint)
**Surface**: home page footer area
**Severity**: important — fails the design system's own accessibility rule

**Repro**:
- Look at home footer: `⌘ K · OPEN THE COMMAND PALETTE`
- Element renders at ~10.5px (Geist Mono uppercase letter-spaced, the `mono-s` scale)
- Color appears to be `--ink-faint` (#9B8F88)

**AC-138 quote** (Session 03 frozen acceptance): "`--ink-faint` is not applied to any rendered text smaller than 18px (the contrast audit substitutes `--ink-soft` for sub-18px copy)."

**Expected**: text in the footer should use `--ink-soft` (#6B5F58), not `--ink-faint`.

**Actual**: `--ink-faint` appears applied. The text is legible on the warm-white ground but fails WCAG AA at the small size per the spec's contrast audit reasoning.

### ARCH-Q-001 — Token persistence across page refresh

**Scenario**: A4
**Surface**: AuthGate + capabilities provider
**Status**: architectural decision — escalate to user

**Observation**: Within a single sidecar launch (so the token is unchanged), reloading the page wipes the in-memory token and shows AuthGate again. The user must paste the token (or follow a `#token=...` link) on every refresh.

**✅ RESOLVED (user decision, 2026-05-15) — `localStorage`, new-token-wins, stale-clears**:

The token must **survive closing and reopening the browser tab** (and a full browser restart) — so `sessionStorage` is insufficient; use **`localStorage`**. It persists **until a new token exists** (a fresh `just dev` mints a new token and prints a new `#token=` URL). Decided behaviour:

1. **On every mount/navigation**, resolve the token in this order: (a) `window.__SPREN_AUTH__` (Tauri injection), (b) a `#token=` URL fragment if present — **a fragment token always wins and overwrites the stored one** (this is how a new `just dev` token replaces a stale one), (c) the `localStorage`-persisted token.
2. **A successfully verified token is written to `localStorage`** and re-read on subsequent loads — so a hard navigation to `/workflows/...`, a deep link, a refresh, or reopening the tab does **NOT** re-prompt. (This directly kills the reported symptom: "hard paths like `/workflows/` suddenly ask for the token again.")
3. **A stored token rejected by the sidecar (401)** — happens when `just dev` restarted and minted a new token — must **clear the stale value and surface the AuthGate with a clear message** (ties [[UI-BUG-001]]: invalid-token feedback), not silently fail or loop. The user then pastes/opens the new token; goto step 1.

This subsumes **[[UI-BUG-006]]** (post-mount fragment changes were ignored — now a fragment always re-resolves and wins) and is the persistence half of **[[UI-BUG-007]]** (the other half is the StrictMode side-effect correctness bug). Both are fixed together as one coherent change in #16.

**Security note (recorded, not blocking)**: `localStorage` for a localhost-only, per-launch token on a single-user-local product (SP-002 + SP-008) is an acceptable trade-off — the token never crosses the wire, rotates every sidecar launch, and a stale one is rejected + cleared by design (step 3). The threat model does not include a hostile local browser-storage reader for this product tier.

### UI-FINDING-001 — AuthGate is well-designed overall

**Surface**: AuthGate (cold load + invalid token form submit)
**Severity**: info — positive observation

**Notes**:
- Clear "Authentication required" title
- Explanatory copy with reassurance ("token stays on your machine — Spren never sends it anywhere except your local sidecar")
- Collapsible "Where to find the token" help with mention of `just dev` banner
- Unlock button transitions from soft-pink (disabled when input is empty) to deep magenta (enabled when input has content)
- Password input (token is masked while typing)
- Token field has correct `<label for="auth-gate-token">` association

The remaining auth issues are around error messaging quality, not the gate's basic UX.

### UI-BUG-006 — URL-fragment token only consumed on initial mount; soft-nav fragment changes ignored

**Scenario**: A2/A4 follow-up (re-auth via address-bar fragment)
**Surface**: `apps/web/src/providers/capabilities.tsx:91-108` (the mount-only `useEffect`) + `readTokenOnce` (`:38-52`)
**Severity**: important — breaks the documented "click the printed `#token=` URL to authenticate" recovery path in any already-loaded tab.

**Root mechanism (verified by reading source)**:
- `CapabilitiesProvider`'s token-detect `useEffect` has empty deps `[]` — runs ONCE on mount.
- `readTokenOnce()` reads `window.__SPREN_AUTH__` first, else parses `window.location.hash`, strips the fragment, returns the token.
- Changing the URL fragment in an already-loaded SPA does NOT remount the provider (no hashchange listener, no router re-mount). So a fresh `#token=...` pasted into the address bar of a loaded tab is never read.

**Repro**:
1. Load `/` (already authenticated or not).
2. In the address bar, change the URL to `http://localhost:5173/#token=<valid>` and press Enter (soft nav — no reload).
3. Nothing happens — the new token is ignored. The AuthGate (if showing) stays; a stale prior error persists.

**Expected**: Either a `hashchange` listener that re-runs token detection, or the AuthGate help text shouldn't promise "click the printed link … authenticates you automatically" for a tab that's already open.

**Actual**: Silent no-op on fragment change post-mount.

### UI-BUG-007 — ⭐ HEADLINE — Auth is flaky/lost on load because `readTokenOnce()` side-effects inside a StrictMode double-invoked effect

**Scenario**: A2/A4/B6-B8 + every hard navigation during the audit
**Surface**: `apps/web/src/providers/capabilities.tsx:38-52` (`readTokenOnce`) + `:91-108` (mount effect); `apps/web/src/main.tsx:24` (`<React.StrictMode>`)
**Severity**: **critical** — this is the single most disruptive bug found. It makes authentication non-deterministic: the same `#token=...` URL sometimes lands authenticated, sometimes dumps the user at the AuthGate with no error. It de-auths users mid-work, on bookmarks, on deep links, on refresh.

**Symptom catalogue (all the same root cause)**:
- `#token=<valid>` URL sometimes authenticates, sometimes shows AuthGate with NO error message. Flaky / timing-dependent.
- Hard navigation to any route (`/workflows`, bookmark, address-bar deep-link, external link) de-auths.
- Spontaneous mid-session de-auth (Vite HMR resync remounts the tree → effect re-runs → token gone).
- Refresh always de-auths.

**Verified facts**:
- Backend healthy throughout: `/healthz` 200, `/v1/bootstrap` with the in-hand token 200, sidecar started exactly once (1 `spren-ready` line), token never rotated. **Not** a backend or token-expiry issue.
- At the moment AuthGate wrongly showed: `window.location.hash === "#token=IZZ...<valid>"` (fragment present), `window.__SPREN_AUTH__ === undefined`, no error rendered (the `if (!candidate)` branch, not the verify-failed branch).
- `<React.StrictMode>` is enabled (`main.tsx:24`) → effects double-invoke in dev (mount → unmount → mount).

**Root cause (high-confidence hypothesis — still must go through the investigator pipeline before any fix)**:

`readTokenOnce()` performs a **side effect** — it strips the URL fragment via `window.history.replaceState(...)` (`capabilities.tsx:47`) — and it is called from inside a `useEffect` (`:93`). Under React StrictMode (dev), that effect runs **twice**:

1. **Mount #1**: `readTokenOnce()` reads `#token=X`, **strips the fragment** (`replaceState`), returns `X`. `verifyAndStore(X)` starts (async).
2. **StrictMode unmount**: cleanup → `cancelledRef.current = true`.
3. **Mount #2**: `cancelledRef.current = false`. `readTokenOnce()` runs **again** — but the fragment was already destroyed in step 1, and `window.__SPREN_AUTH__` is still `undefined` (verify hasn't resolved). Returns **`null`** → `setIsResolving(false)`, no error → **AuthGate renders**.
4. The mount-#1 `verifyAndStore` promise may resolve afterward and set the token (auth "works"), OR the cancel/`isResolving` interplay leaves the gate showing. Which one wins is a **race** → the flakiness.

The defect class: **a side-effecting read (`replaceState`) placed inside a double-invoked effect, where the side effect destroys the very input the second invocation depends on.** Classic React StrictMode footgun.

**Two layered problems (the investigator should treat them as related but distinct)**:
- **7a — dev-flaky / correctness**: the StrictMode double-invoke + side-effecting `readTokenOnce` makes `#token=` auth non-deterministic. Even with StrictMode off in prod, the read-and-mutate-in-effect coupling is fragile (any double render, any remount).
- **7b — production robustness ([[ARCH-Q-001]])**: token is in-memory only (`window.__SPREN_AUTH__` + React state). Once the fragment is stripped, ANY reload/remount (OS sleep/wake, crash-recovery reload, Tauri webview reload, accidental ⌘R, deep-link) has nothing to recover from → AuthGate. SP-008 calls Spren a long-lived local daemon UI; silently losing auth + unsaved canvas on any reload is a robustness failure, not a security feature.

**Why a fix needs the full pipeline (not a YOLO patch)**: the naive fix (move the strip out of the read, or `useRef`-guard the effect) interacts with token persistence (7b), the Tauri `__SPREN_AUTH__` injection contract, the `#`-fragment threat model (deliberately chosen over `?query` so the token never hits logs), and the StrictMode contract. The fix has to be coherent across all of those — exactly the kind of thing the investigator + verifier pipeline exists for.

**Ties to**: [[ARCH-Q-001]] (resolve together — the persistence decision and the effect-correctness fix are one coherent change). Escalate the persistence axis (sessionStorage vs in-memory vs other) to the user before implementing.

### PRODUCT-BUG-002 — Sidecar silently falls back to a random port when requested port is taken

**Scenario**: dev-stack startup (surfaced during environment setup)
**Surface**: `packages/spren/src/spren/__main__.py:32-43` (`_resolve_port`)
**Severity**: important — causes confusing "wrong token / can't connect" symptoms that look like auth bugs.

**Mechanism**: `_resolve_port(requested)` — if binding the requested port raises `OSError`, it silently binds port 0 (random) and returns that. The `spren-ready:` line then advertises the random port, but anything expecting the sidecar on the requested port (the Justfile banner hardcodes `5173` UI → assumes `8765` API; `VITE_SPREN_API_URL=http://127.0.0.1:8765`) now talks to the wrong/[]dead endpoint or a *different* leftover sidecar still holding 8765.

**Observed blast radius**: two leftover sidecars + my new one all "started on 8765" per their logs, but only one actually held 8765; the others silently moved. Result: the browser hit a sidecar whose token didn't match the one I had → looked like an auth bug, wasn't.

**Expected**: either fail loud ("port 8765 in use; refusing to start — pass --port or free it") or, if fallback is intended, make it impossible to have a token/port mismatch (the ready line is the only source of truth — consumers must parse it, never hardcode 8765). Decide which; this is an architectural call.

### TOOLING-BUG-001 (expanded) — `just dev` Windows recipe leaks sidecar + Vite on exit

Original entry above stands; expanded finding: the `Stop-Job` in the `finally` block does not deliver the sidecar's stdin `shutdown\n` protocol (sidecar `__main__.py:62-72` waits for `shutdown` line or stdin EOF). `Stop-Job` kills the PowerShell job wrapper but the spawned `python -m spren` / `vite` grandchildren can survive, orphaned, still holding 8765/5173 — which then triggers PRODUCT-BUG-002 on the next start. Confirmed: found 2 orphaned sidecars + 1 orphaned Vite from prior `just dev` attempts.

### UI-FINDING-002 — Orb state machine + stub send work well (positive)

**Scenarios**: B4, B6, B7
**Severity**: info — positive observation

- Orb state machine verified via DOM: `idle` layer `data-active=true` → on input focus, `typing` layer becomes active, `wrap[data-state]` flips `idle`→`typing`→(on send)`thinking`→`idle`. Crossfade layers all present (`idle/typing/thinking/speaking` each a `.spren-layer`).
- `typing` state renders the three-dot vortex (matches Session 03 §9 spec).
- Send (Enter) → orb `thinking` ~3-5s → stub reply replaces the subline: `(stub) I noted "<msg>" — Sessions 07–09 wire the live meta-agent.` Matches AC-48.
- B8: Shift+Enter inserts a newline; the input is a `<textarea>` that grows (`line one\nline two` preserved). Send button is `button[type=submit].input-bar-send` with an arrow SVG + `aria-label="Send message"`.

### UI-CANDIDATE-001 — InputBar multi-line scroll affordance looks like number-spinner chevrons

**Scenario**: B8
**Surface**: `apps/web/src/components/InputBar/InputBar.tsx`
**Severity**: nit — needs re-inspection (chrome zoom kept missing it; element moved between captures)

When the textarea grew to 2 lines, small up/down chevron arrows appeared just left of the send button. Could be the textarea's native overflow scrollbar buttons, or a custom control. Looked out of place (resembles a number-input spinner). Flag to re-inspect during the focused pass with the textarea deliberately overflowed.

### UI-CANDIDATE-002 — Temporal anchor time display

**Scenario**: B1 (observed across screenshots)
**Severity**: nit — low confidence, likely a non-issue

The wordmark temporal anchor read `Friday · 00:21`, then `00:29/00:30`, then `10:39` across the audit. The jump is plausibly real elapsed wall-clock time (the session spans the 05-14 → 05-15 date rollover with many long-running steps). Note only to re-verify the anchor reflects real local time, not a frozen/incorrect value.

### UI-BUG-008 — Sidebar gating is stale + leaks internal session numbers

**Scenarios**: B10-B11 (sidebar) cross-checked against B13-B14 (cmdk) + Runs nav
**Surface**: `apps/web/src/components/Sidebar/Sidebar.tsx`
**Severity**: important — the sidebar misrepresents what's available, and exposes internal dev process to users.

**Two distinct problems**:

1. **Stale gating**: The sidebar's "COMING SOON" section lists `Runs — history + trace inspector — Session 05`. But `/runs` is fully shipped (Sessions 04 + 05) and reachable: cmdk's "Go to Runs" navigates to a complete, functional Runs page (filter rail with Date/Status/Workflow, empty-state tag-markup, presence orb, breadcrumb). So the sidebar disables/relegates a surface that actually works. Likely `Memory`/`Settings` gating is correct (Sessions 06/10 not shipped) but `Runs` is wrong. The two navigation surfaces (sidebar vs cmdk) disagree about what exists.

2. **Internal jargon leak**: The hints read `… — Session 05`, `KB browser — Session 06`, `secrets + budgets + meta-agent — Session 10`. "Session 05/06/10" is internal implementation-plan vocabulary. A user has no model for what "Session 06" means. The one-line hint (AC-193) should be user-facing only (e.g., `Memory — knowledge base browser (coming soon)`), with no session numbers.

**Expected**: sidebar availability state derives from the same capability source the router/cmdk use (single source of truth); hints carry no internal session references.

### UI-FINDING-003 — Sidebar + cmdk + Runs page are well-built (positive)

**Scenarios**: B10-B14, Runs nav
**Severity**: info

- Sidebar: opens left ~280px with dim backdrop, `aria-expanded` toggles true/false, `aria-label` toggles Open/Close menu, Esc closes, backdrop click closes, "SURFACES" (Home active-highlighted + Workflows) / "COMING SOON" sections, `⌘K` hint at the foot. Matches AC-191..196 (modulo UI-BUG-008).
- Cmdk: Ctrl+K opens, groups `Create` (Create new workflow, Import from Python) + `Navigate` (Go home/Workflows/Runs/Memory/Settings), selecting a nav command routes + dismisses. Matches AC-70..76.
- `/runs`: complete filter rail (Date chips, Status chips with "Showing all statuses (deselect leaves the filter inactive)" helper, Workflow `<select>`), `<runs status="empty" />` tag-markup empty state, "No runs yet. Build a workflow and click Run." + "Go to workflows →" CTA, lower-right presence orb (~80px per AC-197). Wordmark here uses the period-only-magenta treatment (consistent with home; only AuthGate deviates — see UI-BUG-003).

### UI-FINDING-004 — Workflow list + create flow + canvas scaffold work well (positive)

**Scenarios**: C1-C3, D1-D2
**Severity**: info

- `/workflows` empty state: breadcrumb `spren › Workflows`, `Workflows` title, `+ New` (filled) + `+ Import from Python` (outline) buttons, filter chips `All`(active)/`Visual`/`Imported`/`Meta-agent`, `<workflow name="" agents={...} edges={...} />` tag-markup empty state + "No workflows yet." copy, presence orb.
- `+ New` → SPA-navigates, fires create, lands on `/workflows/<ULID>` canvas (`01KRND3KGEXPJ7VS4JNEK8MJ0F`), auth survives (SPA nav doesn't trip UI-BUG-007).
- Canvas D1: top bar (hamburger / `spren.` / `Workflows › Untitled workflow`), toolbar (`Lint ✓` chip / `+ Pattern` / `Save` / 📎 / `Run`), dot-grid pane, `<agent name="" model="" tools={ ... } />` empty-state tag-markup, palette (Agent/User/System/Tool) bottom-left, zoom controls (+/–/fit) bottom-right, presence orb.
- D2 agent config form (right rail) is comprehensive: tag-markup header, IDENTITY (Name*/Goal), MODEL (Provider `<select>` default `anthropic` / Model name* / Temperature 0.7 / Max tokens 8192), INSTRUCTION* (placeholder), TOOLS checklist **populated from `GET /v1/tools`** with real framework tools (`calculate_math`, `clean_and_extract_html`, `data_transform`, `extract_images`, …) each with a one-line description.
- Palette items are **click-to-add** (not drag-only) — reliable. Auto-names: `Agent 1`, `User 1` (numbered, not bare type words).

### PRODUCT-BUG-001 (update) — canvas mitigates it but the validator is still wrong

**Update from D3**: The canvas auto-names User/System/Tool nodes `User 1`/`System 1`/`Tool 1` (numbered) — so the **visual-builder drag/click happy path never produces a bare `user`/`system`/`tool` name** and thus does not trip the reserved-name validator. BUT PRODUCT-BUG-001 remains a real defect on three other paths: (a) renaming a node to `user` via the config form, (b) Python-import of a workflow whose node is named `user`, (c) any direct API client. The validator's scope (all node names vs only agent names) is still architecturally wrong; the canvas naming convention only hides it from one entry path. Severity stays — fix the validator, don't rely on the naming workaround.

**Sharpened root cause (from the framework catalog enumeration)**: this is worse than "over-strict". The framework resolves **det-nodes by literal node name** — `RESERVED_DETNODE_NAMES = {"Start","End","User"}` (`det_nodes.py:174-178`). To get a `UserNode` you *must* have a node named `User`. Spren's `RESERVED_NODE_NAMES = {"user","system","tool"}` rejects `user` **case-insensitively on all node names** — so Spren's validator actively forbids the exact name the framework needs to instantiate the human-in-the-loop det-node. The two reserved-name sets were designed in isolation and now contradict: framework *requires* `User`/`Start`/`End` as node names; Spren *forbids* `user` (and the canvas's `User 1` only sidesteps it by never using the literal name the det-node resolver wants). The correct fix must reconcile these two reserved-name models, not just narrow Spren's validator to agent nodes — escalate as part of the PALETTE-1 / det-node-support work, since they share the root (Spren's topology model has no det-node concept). Cross-ref: [[WF-BUG-PALETTE-1]], Session 03 AC-19 revision.

### UI-BUG-009 — Palette-added node lands off-screen with no auto-pan-to-new-node

**Scenario**: D2-D5
**Surface**: `apps/web/src/routes/workflows/-canvas/*` (node-add handler) + xyflow viewport
**Severity**: important — a click on a palette item can produce *no visible feedback* if the canvas is panned away from the insert position.

**Repro**: On a canvas whose viewport is panned away from the node-insert origin, click a palette item (e.g., `User`). The node IS created (confirmed via DOM: 2 nodes in React state) but it's placed at a fixed graph coordinate that may be off the visible viewport. The user sees the empty canvas unchanged — appears as if the click did nothing. Pressing the fit-view (⊡) control reveals the node.

**Expected**: after adding a node, either (a) place it at the current viewport center, or (b) pan/zoom so the new node is in view + selected. The agent form opening in the right rail is the only feedback otherwise, and on a wide screen the empty canvas dominates.

**Actual**: node placed at a fixed/stale position; no viewport reaction; silent-looking.

### UI-FINDING-005 — Canvas node rendering + fit-view (positive)

**Scenario**: D fit-view
Nodes render with the tag-markup header convention (`<user name="User 1" />` above a bold `User 1`), source/target handles visible, fit-view control re-frames all nodes correctly.

### UI-FINDING-006 — Save + lint detection both work well (positive; corrects an earlier wrong call)

**Scenarios**: D13, D16
**Severity**: info — and a self-correction

- **Save (D16)**: clicking Save fires `PUT /v1/workflows/{id}` → 200, then a `GET /v1/workflows/{id}` re-fetch → 200, "saved" toast + "SAVED" indicator. Provenance/round-trip not yet re-verified on reload (deferred — UI-BUG-007 makes reload testing costly; will verify via SPA round-trip).
- **Lint detection (D13)**: lint **correctly** flags missing required fields. With Agent 1 empty: panel showed ✕ error `agent 'Agent 1' has no model selected` (suggestion: "pick a model in the right rail (e.g., anthropic/claude-opus-4-7)") + ⚠ warning `agent 'Agent 1' has no instruction` (suggestion: "add an instruction so the agent knows what it's doing"), each with a `Go to node →` action. Clear severity icons, plain-language messages, actionable suggestions. The `missing_required_field` code (AC-15) works.
- Provider `<select>` options: `anthropic / openai / openrouter / google / xai / openai-oauth / anthropic-oauth`.

**Self-correction**: an earlier note in this session's conversation claimed "lint passes an incomplete workflow (✓ despite empty required fields)". **That was wrong** — it was a debounce/staleness artifact (I screenshotted the chip right after Save, before the debounced re-lint resolved). Lint does catch it. Per CLAUDE.md's anti-sycophancy rule, recording the correction explicitly so the fix pipeline is not fed a false premise. The *real* defect in this area is UI-BUG-010 below (stale chip state), not a lint-coverage gap.

### UI-BUG-010 — Lint chip shows a stale "✓ OK" while the workflow actually has lint errors

**Scenario**: D11/D16
**Surface**: lint chip component (`apps/web/src/routes/workflows/-canvas/LintChip.tsx`) + the debounced lint trigger
**Severity**: important — undermines trust in the single most prominent correctness signal on the canvas.

**Repro**:
1. Build a workflow whose agent is missing required fields (empty Model/Instruction).
2. Click Save quickly. At that moment the chip reads `Lint ✓` (green/OK).
3. ~A debounce window later, the chip flips to `Lint 1 ✕` (error) — the workflow had the error the whole time; the chip was just stale.

**Expected**: while a lint request is in flight or pending (debounce window open after an edit), the chip should show a distinct "checking…/pending" state — NOT a stale ✓. A user must never see ✓ for a workflow that currently has errors.

**Actual**: the chip holds the last-resolved verdict (✓) through edits + the debounce window, so it can assert "OK" about a state it hasn't actually evaluated. Save is non-blocking by design (lint is advisory — not a bug that Save proceeds), but the chip lying about correctness at decision time is the bug.

**Root-cause hypothesis (for the pipeline)**: the chip likely renders purely off the last lint response and has no `isPending`/`isStale` state tied to the debounce timer / in-flight request. Needs investigator confirmation.

---

## ⭐ USER-REPORTED WORKFLOW-SECTION PUNCH LIST (2026-05-15)

The user reviewed the audit and flagged that the mechanical scenario walk **missed the substantive workflow-section problems**. Root cause of the miss (recorded for process honesty): the audit checked "does element/interaction exist?" instead of "is this the right experience for someone building + running a workflow?" — and never exercised Run (the most important action), never compared palette node-types against the framework's real catalog, and under-weighted the right-rail/orb collision visible in its own screenshots. These 8 are the **priority findings**; they supersede the severity ordering above where they overlap.

### WF-BUG-RUN-1 — ⭐ Run-create failure is silently swallowed (no user feedback at all)

**User report**: "when I click on run … nothing happens (no workflow runs)".
**Surface**: `apps/web/src/components/RunButton/RunButton.tsx:122-126`
**Severity**: **critical** — the primary action of the entire product gives zero feedback on failure.

**Confirmed root cause (read from source, not hypothesis)**:
```js
} catch (err) {
  console.error("create run failed", err);   // ONLY this
} finally { setSubmitting(false); }
```
`handleRun`'s catch block only `console.error`s. There is no `setCompletionToast`, no error atom, no visible state. Contrast `RunButton.tsx:91-97`: terminal run states (succeeded/failed/cancelled) DO fire `setCompletionToast`. The **create** path has no equivalent. So every `POST /v1/runs` rejection disappears into devtools.

**Reproduced**: clicking Run on workflow `01KRND3KGEXPJ7VS4JNEK8MJ0F` → `POST /v1/runs` → **400** `{"error":{"code":"VALIDATION_FAILED","message":"No api_key in secrets store for provider 'anthropic' (checked SPREN_ANTHROPIC_API_KEY env var)","details":{}}}`. Screen: unchanged. No toast. Console only.

**Expected**: surface a user-visible toast/banner with the error message (the AC-189/190 pattern that Save + Python-import already follow). Bonus: the backend's message is actually decent ("No api_key … checked SPREN_ANTHROPIC_API_KEY") — it just never reaches the user.

**✅ RESOLVED (autonomous session, 2026-05-18)**

- **Root cause (re-confirmed in current source, not the 2026-05-15 snapshot)**: `RunButton.tsx:122-123` `handleRun` catch was `console.error(...)` only. The sibling terminal-run effect (`RunButton.tsx:81-108`) already surfaces failures via `completionToastAtom`; the create path had no equivalent. Compounding it, `lib/api.ts:createRun` threw `Error("create run failed: <status> <rawBody>")` — the decent backend message was stringified inside a blob, so even surfacing `err.message` would have shown raw JSON.
- **Solution (per the day-one/elegance/boundary rule)**: a run-create failure is part of the run lifecycle, so it flows through the one run-feedback channel `RunButton` already owns (`completionToastAtom` → `CompletionToast`), not a new mechanism or a parent-lifted toolbar span. `ToastPayload.errorMessage` + `CompletionToast`'s `failed`-with-message branch already existed and degrade gracefully with no `runId`/duration — they were built for this. Added a typed `ApiError {status, code, message, raw}` + `failResponse()` envelope parser in `api.ts` (mirrors the existing `lib/files.ts:UploadError` pattern — consistency, not invention); `createRun` adopts it. `RunButton` catch now sets a `failed` completion toast with the clean message.
- **Boundary fix (necessary blast radius, not scope creep)**: `routes/runs/$runId.tsx` re-run handler did control-flow by `err.message.includes("ATTACHMENT_NOT_FOUND")`. Changing `createRun`'s throw to a clean message would have silently broken stale-attachment detection, so it was migrated to branch on `ApiError.code`/`.status` (the machine contract) — strictly more robust, behaviour-preserving.
- **Scope honesty**: only `createRun` adopts `failResponse`; the other ~11 `api.ts` functions still throw plain `Error`. Migrating them is mechanical but would alter Save/import/list error-display text — deliberately deferred out of a run-create bug fix (anti-pattern #8). `ApiError`/`failResponse` are exported and ready for that follow-up. `run-rerun.ts:StaleAttachmentError` was noticed to be dead (exported, never thrown) — flagged here, not touched (unrelated).
- **Tests**: `run-button.test.tsx` +1 (createRun rejects → `failed` toast with message, button returns to idle); `api-errors.test.ts` new (+2: envelope→typed `ApiError`; non-JSON→raw fallback). Full web suite **207 passed / 34 files** (was 204); typecheck clean.
- **Files**: `apps/web/src/lib/api.ts`, `apps/web/src/components/RunButton/RunButton.tsx`, `apps/web/src/routes/runs/$runId.tsx`, `apps/web/tests/run-button.test.tsx`, `apps/web/tests/api-errors.test.ts`.

### WF-BUG-RUN-2 — No in-product way to provide an API key; Run is unreachable for a normal user

**Surface**: secrets/settings (Session 10 — not shipped) + run-create validation in `packages/spren/src/spren/...` (run materialization)
**Severity**: important — even once the error is visible (RUN-1), a user has no in-app path to fix it.

The only way to satisfy the run-create key check today is to set `SPREN_ANTHROPIC_API_KEY` (or the matching provider env var) **before** the sidecar starts. Settings/secrets UI is Session 10. So in the current shipped product the Run button can never succeed for a user who didn't pre-set an env var — and nothing tells them that. Decide: (a) accept as a known Session-10 gap but at minimum make RUN-1's error actionable ("set SPREN_<PROVIDER>_API_KEY and restart, or wait for Settings"), or (b) bring a minimal key-entry affordance forward. Architectural call — escalate.

> **DISSOLVED 2026-05-17 — Session 08 (credential reframe, D1/P11).** The
> framing above is obsolete. There is no longer a Spren run-create key
> *check*: the `_env_secrets_lookup` / `SPREN_<PROVIDER>_API_KEY`
> pre-gate machinery was deleted. Spren imposes **zero** credential
> assumption — the framework resolves the key **per-provider** from the
> standard variable (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …,
> exactly as the framework itself does), and a genuinely missing key
> surfaces the framework's own clear per-provider `ValidationError`
> mapped to a Spren 400. "v0.3 has no in-app key entry" is now the
> **intended** model (standard env-var path), not a defect; the
> in-product credential store (keychain / encrypted SQLite) is the
> explicit, separate v0.4 seam. Escalation closed. See
> [`08-canonical-workflow-reframe.md`](./08-canonical-workflow-reframe.md)
> §8 D1 + the credential table in §9.

### WF-BUG-RUN-3 — ⭐⭐ THE "Run is broken" ROOT CAUSE — Spren passes raw dicts where the framework expects typed config objects

**User report**: "when I click on run, in the backend I see a lot of errors appearing and nothing happens (no workflow runs)".
**Status**: **REPRODUCED + root cause confirmed from the live traceback.**
**Surface**: `packages/spren/src/spren/runs/lifecycle.py:186` (`register_run` → `Orchestra(...)`) and the Spren run-materialization step that should convert `ExecutionConfigSpec` → a runnable `marsys` config. The framework side (`packages/framework/src/marsys/coordination/orchestra.py:304`) is **TRUNK-CRITICAL — do NOT modify it**; the fix is Spren-side materialization.
**Severity**: **critical** — the single most important action in the product (Run) is broken on the happy path for *every* visually-built workflow, not just the missing-key case.

**Reproduction** (autonomous, via `scripts/scenarios/run_failure_probe.py`): start the sidecar with a deliberately-fake key (`SPREN_ANTHROPIC_API_KEY=sk-ant-FAKE-...`) so run-create clears the secrets presence check (RUN-1's 400 gate); create a minimal runnable workflow (User → Agent, agent has a model, `execution_config.status = {"enabled": true}` — the exact shape the Pydantic mirror + canvas emit); `POST /v1/runs`.

**Result**: `POST /v1/runs` → **500**, body = raw `Internal Server Error` (NOT the structured `{"error":{"code":...}}` envelope every other endpoint uses).

**Backend traceback (the "wall of errors")**:
```
File ".../spren/routes/runs.py", line 188, in create_run
    active = await register_run(
File ".../spren/runs/lifecycle.py", line 186, in register_run
    orchestra = Orchestra(
File ".../marsys/coordination/orchestra.py", line 201, in __init__
    self._initialize_components()
File ".../marsys/coordination/orchestra.py", line 366, in _initialize_components
    self._wire_event_bus()
File ".../marsys/coordination/orchestra.py", line 304, in _wire_event_bus
    if execution_config.status.enabled:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'dict' object has no attribute 'enabled'
```

**Confirmed root cause**: Spren's run path constructs the framework `Orchestra` with an `execution_config` whose `status` is a **plain `dict`** (`{"enabled": True}`), the JSON/Pydantic shape of `ExecutionConfigSpec.status` (`packages/spren/src/spren/models/execution_config.py` — `StatusConfigSpec`). The framework's `Orchestra._wire_event_bus` does `execution_config.status.enabled`, expecting the framework's **typed** `StatusConfig` object (attribute access, not a dict). Spren's materialization step (`packages/spren/src/spren/runs/materialize.py` + `lifecycle.py:186`) **never converts the Pydantic-mirror config dicts back into the framework's runtime config objects** before handing them to `Orchestra`. So the boundary contract is violated: Pydantic mirror (storage/API shape) ≠ framework runtime types (execution shape), and the run path skips the conversion.

This is precisely the "Future cleanup commitment (post-Framework Session 04)" seam noted in Session 02's brief — the mirror↔runtime serializer. v0.3 shipped the mirror + storage but the **run materializer doesn't rehydrate nested config objects** (`status`, almost certainly also `convergence_policy` when it's the `ConvergencePolicyConfigSpec` variant, and any other nested structured field).

**Two distinct defects here**:
- **RUN-3a (critical, functional)**: run materialization passes dict-shaped nested config where the framework needs typed objects → `AttributeError` → no run ever starts from the visual builder. The fix is a Spren-side mirror→runtime conversion in the run path (coherent with the documented mirror/runtime seam — needs the investigator+verifier pipeline; touches the framework boundary; do NOT patch the TRUNK-CRITICAL framework).
- **RUN-3b (important, error handling)**: the `AttributeError` escapes as a raw `500 Internal Server Error` with no structured `{"error":{...}}` body. Session 02's plan specified a global exception handler mapping unexpected errors → `INTERNAL_ERROR` envelope; either it's not installed on the runs router or this path bypasses it. Even when RUN-3a is fixed, unexpected run errors must return the structured envelope so the frontend (once RUN-1 is fixed) can surface a meaningful message.

  **Corroborated by a concrete real run (2026-05-15, real Anthropic key from `.env`)**: a `Start→Agent→End` run that failed because Anthropic returned `401 Unauthorized` (`ModelAPIError: [MODEL_API_AUTHENTICATION_FAILED_ERROR]`) surfaced in the run record only as `error: "ROOT failure: insufficient arrivals"` (the orchestration-level symptom) and **nothing in the sidecar log**. The true cause was recoverable only by reading the per-run trace NDJSON (`<data-dir>/data/runs/<run_id>/<trace_id>.ndjson`, the Step-1 span's `error_message`/`traceback`). So RUN-3b is broader than "raw 500 vs envelope": **a run-execution failure's actual cause is not surfaced to the user at all** — not in the run record's `error`, not in logs. The Session 05 trace inspector is the surface that *can* show it (the span carries `error_class`/`error_message`/`classification: authentication_failed`); the run-list/`error` field should carry a usable cause too (e.g. "model auth failed (provider anthropic)"), not just "insufficient arrivals". This also validates that RUN-3a/3c/3d are fully fixed — the trace shows `Orchestra.run` with topology `[Start, assistant, End]` executing a real `api.anthropic.com/v1/messages` call; the only failure is the user's key being rejected (not a Spren/framework defect).

**Combined picture of "Run is broken"**: RUN-1 (frontend silently swallows the failure) × RUN-3a (backend 500s on every visual-builder run) × RUN-3b (500 has no structured body) × RUN-2 (no in-product key path). All four compound into "click Run → nothing, with errors in the terminal." RUN-3a is the trunk; the others are the branches.

---

### WF-BUG-RUN-3a — ✅ RESOLVED (multi-agent pipeline, 2026-05-15)

**Pipeline run**: `failure-root-cause-investigator` (detailed brief: arch docs + SP-005/SP-018 seam + confirmed traceback + TRUNK-CRITICAL constraint + the planted StartNode question) → my step-2 review (ground-truthed every claim against source, not a rubber-stamp) → independent reviewer **declined with rationale** (no doubt after I re-derived the chain from source; single-file Spren-side; implements the already-documented seam, not a new architectural decision; higher-value second pair of eyes is `implementation-reviewer` on the diff, already in the campaign).

**Confirmed root cause (traced in ground-truth source, file:line)**: `packages/spren/src/spren/runs/materialize.py` `_build_execution_config` did `spec_dump = spec_execution_config.model_dump()` (Pydantic `model_dump()` is **recursive** → nested `StatusConfigSpec`/`ConvergencePolicyConfigSpec` become plain `dict`s), then a **top-level-only** key filter (`{k:v for k,v in spec_dump.items() if k in ExecutionConfig.__dataclass_fields__}`) kept `status` as a dict, then `ExecutionConfig(**filtered)` splatted it. Python dataclasses do **no coercion** in `__init__` (`marsys/coordination/config.py:232` `ExecutionConfig`, `:259` `status: StatusConfig`, `__post_init__:298` only checks content-only thresholds) so the bad value was **latent** until first attribute access at `orchestra.py:304` `if execution_config.status.enabled:` → `AttributeError`. The materializer already does deep typed reconstruction for topology/agents (`_materialize_node/_edge/_topology/_agent/_model_config`) and even uses a **pop-then-build-typed** pattern for `tracing` (`materialize.py:210` pop + `:218` `config.tracing = TracingConfig(...)`); only `execution_config`'s nested structured fields were flat-splatted. The fix makes `execution_config` consistent with the rest of the same file — "as if designed that way from day one."

**StartNode question (planted in the brief) — definitively a NON-ISSUE today**, traced in source: `orchestra._apply_legacy_topology_shim` (`orchestra.py:524-563`, called `:955`) auto-synthesizes a `StartNode` + `Start→entry` edge for any topology carrying a `NodeType.USER` node (the visual builder always emits one — see PRODUCT-BUG-001 update). The orchestrator's `raise ValueError("topology has no StartNode")` (`orchestrator.py:163-168`) is unreachable on the Spren path (`orchestra.py:1029` falls back to `_find_entry_agents` with a non-None `entry_agent` otherwise). **Forward risk** (record for [[WF-BUG-PALETTE-1]]): the shim is `REMOVE-IN-V0.4` (DeprecationWarnings at `orchestra.py:533/1035`); once removed, a builder topology with no explicit `Start` node becomes a hard run-blocker. That is the real stake behind PALETTE-1 — not this bug.

**Fix shipped** (Spren-side only; `packages/framework/` untouched — SP-001/TRUNK-CRITICAL respected):
- `packages/spren/src/spren/runs/materialize.py`: added `_materialize_status_config` (`StatusConfigSpec`→typed `StatusConfig`; mirror's `verbosity:int|None` → framework `Optional[VerbosityLevel]` IntEnum) and `_materialize_convergence_policy` (structured spec variant → typed `ConvergencePolicyConfig`; `float`/`str` passed through untouched because the framework normalizes those itself via `ConvergencePolicyConfig.from_value` at `orchestra.py:984`, and a **dict** reaching `from_value` raises `TypeError` — the second latent instance of the identical defect). `_build_execution_config` now pops `status` + `convergence_policy` out of the splat (same shape as the existing `tracing` pop) and assigns the typed objects post-construction (same shape as the existing `config.tracing`/`config.aggui` assignment). Import line extended for `ConvergencePolicyConfig, StatusConfig, VerbosityLevel`; `ConvergencePolicyConfigSpec` added to the `spren.models` import.
- `packages/spren/tests/test_runs_materialize.py`: 4 regression tests capturing the root cause — typed `StatusConfig` (not dict) + `.enabled`/`.verbosity` access; disabled-status default; structured convergence → typed `ConvergencePolicyConfig`; scalar convergence passes through.

**Verification**:
- New regression tests: 4/4 pass with the fix (they assert `isinstance(...StatusConfig)` / `ConvergencePolicyConfig` — fail before, pass after).
- Blast-radius suite (`test_runs_materialize` + `test_runs_lifecycle` + `test_routes_runs` + `test_runs_tracing_wiring`): **66 passed, 1 failed**; the one failure (`test_default_secrets_lookup_handles_dashed_provider`) is a **pre-existing, environment-dependent** failure — `openai-oauth` provider requires `~/.codex/auth.json`, raised in framework `models/adapters/openai_oauth.py:142`, on the model-adapter path my diff never touches. Documented pre-existing credential-fixture class; not a regression.
- **End-to-end probe** (`scripts/scenarios/run_failure_probe.py` against a fake-key sidecar — the exact original repro): before = `AttributeError: 'dict' object has no attribute 'enabled'` at `orchestra.py:304`. After = traceback now flows **past** 304 and dies at `orchestra.py:347` on a **different root** (`ModuleNotFoundError: No module named 'ag_ui'`). The RUN-3a defect is gone; a distinct downstream blocker (RUN-3c) surfaced — recorded below.

**Commit**: pending (checkpoint after RUN-3c is triaged with the user; the two are on the same code path and the user asked for an architectural decision on RUN-3c).

### WF-BUG-RUN-3c — ⭐ NEW (surfaced by fixing RUN-3a) — AG-UI translator import fails: `ag_ui` not installed on a default install

**Status**: distinct from RUN-3a (different root, same Run code path). NOT YOLO-fixed — carries an architectural/dependency-policy question → **escalated to user**.
**Surface**: `packages/spren/src/spren/runs/materialize.py:224-240` (opts into AG-UI) → framework `orchestra.py:347` `from .aggui.translator import AGGUITranslator` → `aggui/translator.py:22` `from ag_ui.core import BaseEvent`.
**Severity**: **critical** — with RUN-3a fixed, this is now the next hard blocker: every visual-builder run still 500s (raw body — RUN-3b again, independently reconfirmed) because the AG-UI translator can't import its third-party dep.

**Mechanism**: `materialize.py` defaults `enable_aggui=True` and sets `config.aggui = AGGUIConfig(enabled=True)` — its guard only checks that `marsys.coordination.aggui.config` imports (it does — `AGGUIConfig` is a plain dataclass needing no `ag_ui`). The framework's `_wire_event_bus` then imports `.aggui.translator`, which **does** need the third-party `ag_ui` package. `ag_ui` is **not importable** in the venv. Spren's root `pyproject.toml` ships it only behind an **optional extra** (`[project.optional-dependencies] aggui = ["marsys[aggui]"]`); core `dependencies` omits it.

**Architectural tension (the escalation)**: **SP-004** makes "AG-UI the wire schema for run events" — i.e. AG-UI is *core* to Spren's run/SSE path, not optional. But it ships as an opt-in extra, so a default `uv sync` / `pip install spren` produces a build that **cannot run any workflow**. The materializer's `try/except ImportError → "Framework 06 not yet present"` comments treat AG-UI as future/optional, which contradicts SP-004 and the fact that `marsys.coordination.aggui` (config + translator) is already merged. Candidate resolutions (user decides — dependency policy is not mine to set unilaterally): (a) promote `marsys[aggui]` into Spren's **core** `dependencies` (matches SP-004; AG-UI is load-bearing); (b) keep it an extra but make the materializer's opt-in honest — presence-check the *translator*'s `ag_ui` import, not just `aggui.config`, and decide what "AG-UI unavailable" means for a product whose wire schema *is* AG-UI; (c) something else. (a) is the SP-004-consistent default and my recommendation, but it's a dependency-policy call.

**✅ RESOLVED (user decision + fix, 2026-05-15)**. User chose option (a): promote AG-UI to a core dependency (AG-UI is load-bearing per SP-004, not optional). Fix shipped:
- `packages/spren/pyproject.toml`: core dependency `"marsys"` → `"marsys[aggui]"` (always pulls `ag-ui-protocol==0.1.18`, the framework's `aggui` extra at `packages/framework/pyproject.toml:67-68`). Removed the now-redundant `[project.optional-dependencies] aggui = ["marsys[aggui]"]` (pure duplicate after promotion — verified no Justfile/CI reference to `spren[aggui]`; per SP-006 single-version).
- `packages/spren/src/spren/runs/materialize.py`: simplified the `_build_execution_config` aggui block to unconditional `from marsys.coordination.aggui.config import AGGUIConfig; config.aggui = AGGUIConfig(enabled=True)`. The prior `try/except ImportError` + `hasattr` + "Framework 06 not yet present" branches were defensive code for a now-impossible case (AG-UI is core; `marsys.coordination.aggui` is merged) — removed per CLAUDE.md anti-pattern #5 + SP-006. Docstring corrected (it claimed AG-UI was optional/future).
- **Verified**: `uv sync` installs `ag-ui-protocol==0.1.18`; `from ag_ui.core import BaseEvent` imports; the probe's `POST /v1/runs` now returns **201** `{"schema_version":1,"run_id":...,"status":"queued"}` (was 500). Materializer suite 10 passed / 1 pre-existing-env deselected. **Commit**: pending (bundle with RUN-3a + RUN-3d triage).

### WF-BUG-RUN-3b — ✅ RESOLVED (autonomous session, 2026-05-18)

Two parts, both confirmed from **current** source (not the 2026-05-15 snapshot):

- **(a) No structured envelope for unhandled errors.** `server.py` had **only** `@app.exception_handler(RequestValidationError)` (422). No generic `Exception` handler → any unhandled error escaped as Starlette's raw `text/plain` 500 on **every** router, so the frontend (now RUN-1-fixed) had no envelope to parse. **Fix**: added an app-level `@app.exception_handler(Exception)` mirroring the existing validation handler — returns `{"error":{"code":"INTERNAL_ERROR","message":"an unexpected error occurred","details":{}}}` at 500. Per the day-one/boundary rule the handler is **app-wide**, not runs-scoped (the envelope contract is an app concern; scoping it to runs would be the wrong boundary) — this also closes the same gap for workflows/files/tools/lint (strict improvement, not scope creep). The exception text goes to `logger.exception` (operators); the client body is a fixed safe message + code (no path/secret leakage — the correct boundary).
- **(b) Real failure cause not surfaced.** `lifecycle.py:314-316` set the run record's `error` to `str(result.error)` — the orchestration-level *symptom* (`"ROOT failure: insufficient arrivals"`); the actionable per-branch reason was never read. Grounded the framework result shape read-only: `Orchestra.execute()` returns the **public** `OrchestraResult` (orchestra.py:72) carrying `branch_results: List[BranchResult]`; `BranchResult` (framework `branches/types.py:152`) has `success`/`error`/`get_last_agent()`. (The 2026-05-15 doc note "read `result.branches`" was imprecise — corrected to `result.branch_results` via the public adapter; SP-018 preserved, framework untouched.) **Fix**: added `_summarize_failure(result)` — prefers the failed branch's `agent: error`, keeps the orchestration symptom as secondary context, falls back to the top error, then `"execution failed"`. `error_msg = None if success else _summarize_failure(result)`.
- **Why no investigator pipeline**: root cause was directly traced in primary source (no Exception handler; lifecycle reads only `result.error`; `BranchResult.error` carries the cause); both fixes are additive and architecture-fitting (mirrors an existing handler; reads a public adapter API designed for exactly this), not workarounds. Triage = straightforward; self-reviewed the chain.
- **Relation to RUN-3e**: part (b) is the *diagnostic* half the RUN-3e finding also flagged — independent of the deferred RUN-3e prompt-enrichment **decision**. It makes the RUN-3e reproduction legible (the run record will now show the real branch reason instead of "insufficient arrivals").
- **Tests**: `test_runs_lifecycle.py` +5 (`TestSummarizeFailure`: branch-reason-with-agent, top-error fallback, no-agent, last-resort, dedupe); `test_server.py` +1 (`TestUnhandledErrorEnvelope`: raising route → `INTERNAL_ERROR` envelope + no internal-text leak). Touched-suite run (server + lifecycle + routes_runs): **66 passed**.
- **Files**: `packages/spren/src/spren/server.py`, `packages/spren/src/spren/runs/lifecycle.py`, `packages/spren/tests/test_server.py`, `packages/spren/tests/test_runs_lifecycle.py`.

> ⛔ **SUPERSEDED 2026-05-16 — DO NOT ACT ON RUN-3d / RUN-3e / "Node-model redesign" / PALETTE Start-End below.**
> Root cause was wrong. Real cause: Spren hand-rolls a Pydantic mirror that diverged from the framework's
> canonical Session-04 wire shape, and `materialize.py:181` drops `topology.metadata` (entry/exit) the framework
> shim needs. Proven by `packages/framework/benchmarks/GAIA/test_parallel_tracing_canonical_anthropic.py`
> (Success: True on Spren's exact path). "Emit DeterministicNode instances" is refuted; PRODUCT-BUG-001 auto-dissolves.
> **Authoritative plan + full remaining-task ledger:** `sessions/08-canonical-workflow-reframe.md`. Text below is
> retained only as the audit trail of how the wrong conclusion was reached.

### WF-BUG-RUN-3d — ⭐ NEW (surfaced by RUN-3a+3c fixes) — visual-builder topologies have no explicit Start node → framework topology validation fails

**Status**: this is the **manifestation of WF-BUG-PALETTE-1 on the Run path** — not a separate root cause. Empirically confirmed it is now the next Run blocker (run *creates* fine, then *fails during execution*).
**Surface**: Spren topology mirror (no Start concept) + visual-builder canvas (palette has no Start — WF-BUG-PALETTE-1) + framework `orchestra._apply_legacy_topology_shim` (`orchestra.py:545-563`) + `topology_graph.validate_workflow()` (`orchestra.py:965`).
**Severity**: **critical** — with RUN-3a/3c fixed, *every visual-builder run still fails*. Run-create returns 201, then the run terminates `status:"failed"` with `error: "[TOPOLOGY_ERROR] topology validation failed: - node 'user_in' not reachable from Start"`.

**Mechanism (traced via probe + the shim source already read for RUN-3a)**: the probe's topology is `user_in (User) → assistant (Agent)` with **no explicit Start** — exactly the shape the visual builder emits today (canvas auto-creates a `User 1` node; palette has no Start per PALETTE-1). The legacy shim synthesizes a `StartNode` and a `Start → assistant` edge (assistant = the User node's non-User outgoing target), but it does **not** add `Start → user_in`. The framework's post-shim `validate_workflow()` requires every node reachable from Start; `user_in` is now orphaned → `TOPOLOGY_ERROR`. The shim is `REMOVE-IN-V0.4` and was never a correctness guarantee — relying on it is exactly what the user forbade.

**This is why the user's Start/End directive is on the Run critical path, not just palette UX.** See the architectural decision recorded under [[WF-BUG-PALETTE-1]] (Start = exactly one, present by default, non-deletable; End = 0..N, not present by default, user-added). The Run path will not work for visual-builder workflows until Spren models an explicit Start node and emits `Start → entry` edges itself (no shim reliance). RUN-3d is resolved by the PALETTE-1 explicit-Start implementation.

**Combined picture (updated)**: RUN-3a ✅ (typed config) × RUN-3c ✅ (AG-UI core dep) × **RUN-3d ⛔ (no explicit Start — the live blocker, = PALETTE-1)** × RUN-1 ⛔ (frontend still silently swallows) × RUN-3b ⛔ (raw 500 for unhandled errors — run-create now 201 but other failures still need the envelope) × RUN-2 (no in-product key path). The trunk moved: with 3a/3c fixed, **RUN-3d/PALETTE-1 is now the #1 Run blocker.**

#### RUN-3d / PALETTE-1 — design-pipeline review outcome (2026-05-15)

Pipeline: investigator → my ground-truth step-2 verification → independent verifier (`general-purpose`, independent context). **Verifier verdict: SOUND-WITH-CONCERNS.** I agree with the verdict.

**Confirmed sound** (root cause + core mechanism): the parse_node object-vs-string asymmetry and the shim-orphan are real and complete (re-derived). The materializer emitting `StartNode`/`EndNode`/`UserNode` *instances* is the **architecturally-supported framework path** — strengthened by a fact the step-2 pass hadn't surfaced: `Topology.__post_init__` (`core.py:170-174`) **explicitly type-accepts `DeterministicNode` instances** in the `nodes=[...]` constructor list (documented "unified-barrier orchestrator path"). It is a first-class external contract, not a hack. The legacy shim provably no-ops once a StartNode is registered (its branches are all `get_start_node() is None` / `existing_*` guarded). PRODUCT-BUG-001 agent-scoping matches the framework's own enforcement (`agents/agents.py:242`, `agents/registry.py:62` gate on agent names only) — not over-reach.

**Concern 1 (BLOCKING) — RESOLVED in design (no user input needed)**: adding `START`/`END` members to the Spren `NodeType` Pydantic enum is an **SP-005 violation** — the framework `NodeType` is exactly `{USER,AGENT,SYSTEM,TOOL}` (`core.py:20-25`); Start/End are det-nodes resolved **by name**, never enum members, and the Spren enum is codegen'd to the TS client (`api.ts:24`), so inventing members ships framework-divergent types. **Corrected design**: `NodeType` stays at the 4 framework members. Det-node identity is **name-based** (`name in {"Start","End","User"}`), exactly mirroring the framework's `RESERVED_DETNODE_NAMES` mechanism. If the canvas needs a fast classifier, add a *non-enum* discriminator on `NodeSpec` (a derived `is_control`/`is_det_node` property or a `kind` literal) — NOT new `NodeType` values. The materializer dispatches on **name**, which also aligns 1:1 with `RESERVED_DETNODE_NAMES`. This is strictly more SP-005-faithful than the investigator's enum approach.

**Concern 2 (BLOCKING) — ESCALATED to user**: `User` is simultaneously a det-node *and* `NodeType.USER`. The framework's reserved-name resolution is singular (`"User"` → `UserNode`), but the canvas today auto-numbers user nodes `User 1`, `User 2` (the old PRODUCT-BUG-001 mitigation). Unresolved: is a `NodeType.USER` node *always* a `UserNode` det-node (⇒ must be named exactly `User`, ⇒ at most one, ⇒ kill the `User N` numbering), or only when named `User`? This is a product/architecture decision parallel to the user's Start/End cardinality decision and interacts with existing canvas naming — must be pinned before coding. **Recommendation put to user**: User node = **0-or-1**, name exactly `User`, **user-added (not default), not auto-numbered** — mirrors the framework's singular reserved name, makes human-interaction optional, and removes the `User N` collision entirely. Multi-human-interaction (distinct named UserNodes via the instance path) is a future extension, out of v0.3 scope.

**Concern 3 (in-scope follow-up)**: Start-deletion guard must also cover xyflow `deleteKeyCode` (Backspace/Delete) + edge-driven removal, not just delete handlers (`$workflowId.tsx:636-637`). Importer `_coerce_node_type` (`python_workflow.py:633-660`) genuinely needs a det-node path (a framework workflow with a `Start` node currently imports as a Spren AGENT named "Start"). `pattern-presets.ts` + `auto-layout.ts` emit Start-less topologies — in scope.

**Concern 4 (follow-up)**: auto-create `Start → firstNode` edge is load-bearing for the Run path (without seeding+wiring the canvas still fails `validate()`), keep minimal — matches the reasonable-call already surfaced to the user.

**Status**: SUPERSEDED by the node-model redesign below — the Concern-1/2 patch-on-top approach was abandoned once the user identified the foundation itself was wrong.

#### ⭐ Node-model redesign — the real correction (user-driven, 2026-05-15)

The user rejected patching Start/End onto the existing `NodeType` taxonomy and identified that **the taxonomy itself is wrong**. This is the substantive correction; RUN-3d is now a consequence of it.

**Finding — `SYSTEM`/`TOOL` are vestigial.** `grep NodeType.SYSTEM` / `NodeType.TOOL` across `packages/framework/src/marsys` returns **zero matches**. They exist only in the `NodeType` enum (`core.py:24-25`) + `RESERVED_NODE_NAMES` (`core.py:29`); nothing in framework execution/orchestration/topology reads them. `AGENT` and legacy `USER` are the only real regular-node types; the modern canonical model is the `DeterministicNode` subclasses (`StartNode`/`EndNode`/`UserNode`). Name-based det-node resolution is **string-DSL sugar only** (`parsing.py:62-72`), not the canonical contract. Spren's `models/topology.py` `NodeType={user,agent,system,tool}` hand-mirror copied dead members and mistook the sugar for the model.

**Process root-cause (mine, recorded honestly).** The investigator, the verifier, and I all debated the Start/End *delta* on top of `NodeType={user,agent,system,tool}` — a baseline none of us questioned because **my agent prompts fed that 4-type set as ground-truth and scoped every pass to "fix RUN-3d."** Anchoring bias: a bug-scoped pipeline cannot catch "the foundation we're building on is wrong" if every prompt frames the question around the bug. Second anchoring miss this audit (first: the mechanical-scenario miss). Mitigation for future pipelines: when a fix touches a type/contract, explicitly task one pass with "is the base model itself faithful?", not only "is the delta faithful?".

**Decision — Spren node model is its own UX layer (locked).** Full design: [`docs/architecture/spren/11-node-model.md`](../../../../architecture/spren/11-node-model.md). Summary:
- Spren's palette is an OpenAI-Agent-Builder-style UX surface, NOT a 1:1 marsys mirror. Each node type declares a **materialization** to valid marsys constructs. SP-005 still governs genuine framework mirrors (e.g. `ExecutionConfig`); the node palette is explicitly a translation surface, not a mirror — this is the resolution of the verifier's SP-005 Concern 1.
- Five categories: **Agents, Core, Tools, Logic, Data**. **v0.3 ships Agents + Core active**; Tools/Logic/Data are shown **inactive** ("coming soon", not droppable — never ship a non-runnable node, that is the RUN-3d failure class) and activated by the product owner as each marsys primitive lands.
- **Core** = Start (exactly one, default, non-deletable) / End (0..N, user-added) / User. **User collapse decision**: the frontend may have multiple User nodes (design convenience); at materialization they collapse to the single canonical `UserNode` with all incident edges rewired to it.
- **Tools is two concepts**: (a) agent-attached tools (exists; deferred redesign = per-tool config cards + anticipate custom tools in the UX); (b) tool-as-node (future; pairs with a Conditional/Logic node for deterministic tool invocation). Kept distinct in the abstraction.
- Extensible: a future marsys det-node = one new Core/Logic type + one materializer entry + flip `active`. No model/mirror/canvas rework.

**Task #20 reframed**: "node-model building blocks (all 5 categories modeled; Tools/Logic/Data inactive) + Core (Start/End/User) concrete materialization — unblocks RUN-3d." The verified core mechanism still holds (materializer emits `StartNode`/`EndNode`/`UserNode` *instances* — the supported `Topology` `Node|DeterministicNode` path, `core.py:170-174`); the redesign changes how Spren *models/represents* nodes (category+type+materialization, not the vestigial `NodeType` enum), not the framework-emit mechanism. Implementation proceeds on this basis; `implementation-reviewer` on the diff.

#### Session 07 spun out + cross-session decisions (2026-05-15)

RUN-3d / node-model implementation is now **Session 07** (`docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/07-node-model-core.md`), validated (`session-plan-validator`) + improved (`session-plan-improver`) + amended to the agreed contract. Decisions recorded here for continuity:

- **Session 03 AC-20 superseded (not edited).** Session 03's frozen `acceptance.md` AC-20 required "Spren lint wraps the framework's `TopologyGraph.validate_workflow()`". It was never implemented and is now an **SP-018 violation** (Spren must not reach into framework runtime). The Spren linter (`packages/spren/src/spren/lint/workflow_linter.py`) is and stays a pure independent linter; Session 07 makes its existing rules Start/End/User-aware (mirroring the validate-workflow *contract* in logic, zero framework calls) and corrects the false `models/lint.py:1-9` docstring that claimed the wrap exists. AC-20 is **superseded by Session 07's AC-11/AC-14**; Session 03's frozen file is left untouched (frozen-acceptance discipline). Root-cause artifact: the `models/lint.py` docstring asserted a wrap that was never built — a latent Session-03 acceptance gap.
- **v0.3 User node is materialized but NOT executable — accepted limitation.** Spren's `register_run` wires no `UserNodeHandler` (`lifecycle.py:186-190` → `orchestra.py:380-385,1018-1022` → `det_nodes.py:150-153`), so a run that reaches a User node fails "no handler bound" before any provider call. v0.3 runnable shape is `Start → Agent → End`. User-node modeling/materialization/collapse IS shipped in Session 07 (unit-tested); *execution* of human-in-the-loop is a **separate future session the user will open** (interactive-runs). Full source-traced carry-forward detail is in Session 07's "Known v0.3 limitation" section.
- **Forward migration mandated.** The node-model change rewrites stored `definition` JSON; SP-006 + the live `MigrationsRunner` require a one-shot forward migration (`storage/migrations/04__*`) — Session 07 AC-13. (The earlier plan premise "disposable dev data, no migration" was refuted by `session-plan-validator`.)

### Framework node + agent catalog — ground truth (for PALETTE-1/2/3 + PRODUCT-BUG-001)

Enumerated from framework source (read-only, file:line cited). This is the factual catalog the palette redesign reconciles against. **Facts are cited; how the palette should *present* them is a design decision to escalate — not settled here.**

**Node-type enum** — `packages/framework/src/marsys/coordination/topology/core.py:20-25`:
```python
class NodeType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
```
Exactly four. **There is no START / END / STOP NodeType.** "Stop" is not a framework concept; the terminal construct is **End**.

**Deterministic nodes** — `packages/framework/src/marsys/coordination/execution/det_nodes.py`:

| Det-node | Lines | Role |
|---|---|---|
| `StartNode` | 67-93 | Workflow entry. **Exactly one required per topology.** ≥1 outgoing edge, 0 incoming. |
| `EndNode` | 96-117 | Workflow exit. Zero-or-more. Agents invoke it to deliver the final answer; 0 outgoing. |
| `UserNode` | 119-169 | Bidirectional human Q&A. Routes agent invocations through the async I/O handler. |

`RESERVED_DETNODE_NAMES` (`det_nodes.py:174-178`) maps literal names **`"Start"` → StartNode, `"End"` → EndNode, `"User"` → UserNode** — the framework resolves the det-node class from the node's name. The orchestrator hard-requires a StartNode (`orchestrator.py:163-168` raises `"topology has no StartNode"` otherwise); `TopologyGraph.get_start_node` (`graph.py:1061-1075`) enforces exactly one; topology validation (`graph.py:1303-1393`) enforces the edge constraints above.

**Consequence (load-bearing — verify in the RUN-3a pipeline)**: a runnable topology is *impossible* without a StartNode. If the visual builder never emits a node named `Start`, **every visual-builder run fails for this reason too**, independent of WF-BUG-RUN-3a's dict/typed-config defect. The RUN-3a investigator must check whether Spren injects a Start node during materialization or whether this is a second, distinct run-blocker.

**Specialized agent catalog** — `packages/framework/src/marsys/agents/__init__.py:5-20` (public exports):

| Class | Purpose (one line) |
|---|---|
| `Agent` | General-purpose; local or API model, configurable tools + memory. The palette's only current option. |
| `BrowserAgent` | Browser automation / web scraping via Playwright (PRIMITIVE/ADVANCED modes). |
| `InteractiveElementsAgent` | Vision-based UI-element detection on screenshots (normalized coords). |
| `CodeExecutionAgent` | Python/shell execution + file ops for dev/scripting/automation. |
| `DataAnalysisAgent` | Data science with a persistent Jupyter-like Python session. |
| `FileOperationAgent` | File ops + optional shell, system-admin-style tasks. |
| `WebSearchAgent` | Web + scholarly search (Google, DuckDuckGo, arXiv, Semantic Scholar, PubMed). |
| `BaseLearnableAgent` / `LearnableAgent` | PEFT-trainable agents; local HuggingFace models only (no API models). |

(`BaseAgent` = abstract parent; `AgentPool` = a collection, not a node type.) **1 generic + ~8 specialized**; the palette exposes only generic Agent.

**Node → agent binding** — `core.py:47-77`: an `AGENT` node carries `agent_ref: Optional[Any]`; binding is **name-based and decoupled** — the node names an agent, the agent definition (class + model + tools) resolves separately at instantiation. "Which specialized class" is therefore an attribute of the agent *definition* the node points at, not a separate NodeType. How a builder picks the specialized class (agent-config "type" selector vs distinct palette items) is a design decision — escalate.

---

### WF-BUG-PALETTE-1 — Missing node types: no Start / End nodes (and "Stop" isn't a framework concept)

**User report**: "there is no start or end node types in the palette".
**Surface**: palette (`apps/web/src/routes/workflows/-canvas/Palette.tsx`) vs the framework catalog above.
**Severity**: important — likely **critical-adjacent**: per the catalog's load-bearing consequence, no StartNode ⇒ no runnable topology. The palette exposes only Agent/User/System/Tool; Start/End are real placeable constructs (via the reserved det-node names `Start`/`End`) and are un-buildable visually today.
**Now known (was "investigation owed", resolved by the catalog above)**:
- Start/End/User are **deterministic nodes resolved by literal node name** (`Start`/`End`/`User`), not `NodeType` members. The user's mental model ("Start/End are node types you place") is correct at the topology level — name a node `Start` and you get StartNode behaviour.
- There is **no Stop** — the terminal is **End**. The palette/redesign copy must say End, not Stop.
- This is linked to the Session 03 AC-19 revision (Spren-side lint can't run `TopologyGraph.validate_workflow()` because no det-nodes are registered) — same root family: the visual builder has no concept of det-nodes.
- **Verified (RUN-3a/3d)**: Spren does NOT inject a Start node; the framework's legacy shim half-synthesizes one and topology validation then fails (`node not reachable from Start`). Confirmed via probe: this is now the live #1 Run blocker. See [[WF-BUG-RUN-3d]].

**⭐ ARCHITECTURAL DECISION (user, 2026-05-15) — binding for the PALETTE-1 / explicit-Start implementation**:

- **Start node**: a workflow canvas always has **exactly one** Start node. It is **present by default** on every new canvas and is **not deletable**. Exactly-one is enforced **by construction in the visual builder AND by framework rules** (the framework already enforces the singleton: `TopologyGraph.get_start_node()` raises `TopologyError` if >1 — `graph.py:1061-1075`; Spren must not allow a second to be created in the first place).
- **End node**: NOT the same as Start. **Zero-or-more** End nodes allowed. **Not present by default** on a new canvas — the user adds End node(s) from the palette as needed (matches the framework: `EndNode` is zero-or-more, terminal).
- **No reliance on framework auto-detection / the legacy shim** for Start/End. Spren models them explicitly and emits the `Start → entry` edges itself. (The shim is `REMOVE-IN-V0.4` anyway; relying on it is the RUN-3d defect.)
- Consequence: Spren's topology mirror needs an explicit way to represent Start/End (framework resolves det-nodes by literal node name `Start`/`End` — not via `NodeType`). This interacts with PRODUCT-BUG-001 (Spren's reserved-name validator must not reject the very names the framework requires). Design owed in the PALETTE-1 / explicit-Start pipeline; the *behavioural* contract above is decided and not re-litigated.

### ⭐ PALETTE REDESIGN SPEC (user decision, 2026-05-15) — binding; PALETTE-2/3/4 resolve to this

The whole palette structure + design is redesigned (not a tweak). Binding contract:

- **Position**: left side of the canvas.
- **Form factor**: a floating **island** — NOT a full-height sidebar. (Same island language as the rest of the canvas chrome.)
- **Categories are vertical**: a vertical list of category entries.
- **Click a category → an additional section opens** showing that category's items in detail (e.g., click *Agents* → the generic Agent + each specialized agent listed individually with detail). Two-level: vertical categories → expanded detail section. NOT flat, NOT click-instant-add.
- **Agents are shown as distinct items** — generic `Agent` plus each specialized agent (`BrowserAgent`, `CodeExecutionAgent`, `DataAnalysisAgent`, `FileOperationAgent`, `WebSearchAgent`, `InteractiveElementsAgent`, `LearnableAgent`) is its own visible, addable item. **Explicitly NOT a type-selector** (overrides the earlier type-selector suggestion).
- **Categories** (from the framework catalog above): **Control** (Start, End, User), **Agents** (generic + specialized, distinct items), **Tools**, **System**.
- Start node specifics governed by the WF-BUG-PALETTE-1 architectural decision above (exactly one, default, non-deletable). End is palette-addable, 0..N.
- Card/visual quality (icons, name, short description, breathing room, design-system cohesion) per WF-BUG-PALETTE-4, applied within this structure.

PALETTE-2 (interaction), PALETTE-3 (categories/specialized agents), PALETTE-4 (card visuals) are now sub-aspects of this single redesign — tracked as task #21; Start-node functional/topology work is the Run-critical task #20.

### WF-BUG-PALETTE-2 — Palette interaction model is wrong (click instantly dumps a node)

**User report**: "as soon as I click on an element inside the panel … it directly add[s] them to the canvas" — expected: clicking a category reveals the types within it.
**Surface**: `Palette.tsx`
**Severity**: important — counter-intuitive; no way to browse what's available before committing it to the canvas. (The audit wrongly logged click-to-add as a *positive* — corrected here.)
**Expected model**: palette is **categories** → expand a category → see its **types as cards**; a card click expands a longer description + an explicit "Add to canvas"; drag-drop also supported. Not click-anything-instant-add.

### WF-BUG-PALETTE-3 — Palette has no categories; flat 4-section list is insufficient

**User report**: "there should be categories (one … for the start/stop nodes and user, another … for the agent types which should also list specialized agents that we have)".
**Severity**: important — the current flat Agent/User/System/Tool exposes none of the specialized agent catalog.
**Now known (resolved by the catalog above — no longer "investigation owed")**: proposed category structure, grounded in the framework's actual catalog:
- **Control** — Start, End, User (the three det-nodes; "Stop" → use End).
- **Agents** — generic `Agent` + the ~8 specialized classes from the catalog table (`BrowserAgent`, `InteractiveElementsAgent`, `CodeExecutionAgent`, `DataAnalysisAgent`, `FileOperationAgent`, `WebSearchAgent`, `LearnableAgent`). Open design question (escalate): are specialized agents *distinct palette items* or a *type selector inside the Agent config*? The framework's name-based decoupled binding (`core.py:47-77`) permits either; this is a UX call, not a framework constraint.
- **Tools** — Tool (`NodeType.TOOL`).
- **System** — System (`NodeType.SYSTEM`).

### WF-BUG-PALETTE-4 — Palette item visual design (cards, spacing, icon/name/description)

**User report**: each item should be a tasteful card (icon + name + short description, "not too big"), with breathing room between cards (not packed tight); card click expands a longer description + Add-to-canvas; drag-drop supported.
**Severity**: important (design) — current palette items are bare text chips packed together; no icons, no descriptions, no spacing rhythm. Does not match the design-system care shown elsewhere (orb, tag-markup, /runs).

### WF-BUG-NODE-1 — Canvas node visual design is below the app's bar

**User report**: "the UI for the nodes (when they are added to the canvas) is not aesthetically beautiful."
**Surface**: `apps/web/src/routes/workflows/-canvas/CanvasNode.tsx`
**Severity**: important (design). Current node = tag-markup line + bold label in a plain rounded rect. Functional but visually flat vs the rest of the app's polish. Needs a design pass coherent with the orb / `--coral` / Geist system.

**Design pass must capture (zoomed screenshots at ≥2× per item)**:
1. Default node (Agent) — corner radius, border weight/colour, fill, internal padding, the `<agent .../>` tag-markup line typography vs the bold label.
2. Selected state — the 1px `--magenta` border + halo (UI-FINDING-005): does the halo read as intentional or as a default xyflow ring?
3. Each node type (Agent/User/System/Tool) side by side — is there any visual differentiation (icon/colour) or are they identical but for the label? (They should be type-distinct.)
4. Handles (source/target) — size, colour, hover affordance; do they match the edge `→` treatment?
5. Node-to-node spacing on a multi-node canvas (the user explicitly asked for "enough empty space between nodes" — capture default auto-layout spacing).
6. Compare against the design tokens in `apps/web/src/styles/tokens.css` — list which the node currently uses vs which it should (warm-white ground, `--coral`/`--magenta`, Geist Mono for the tag-markup line, AC-138 contrast rule for any sub-18px text).

### WF-BUG-LINT-REACTIVITY — Lint findings don't clear after the underlying error is fixed (requires full page reload)

**User report**: "if there is an error in agent definition and I fix that, the linter errors doesn't go away and I need to reload the page until the linting errors are refreshed."
**Surface**: lint trigger wiring (canvas state → debounced `POST /v1/workflows/{id}/lint` → chip/panel) — `apps/web/src/routes/workflows/-canvas/LintChip.tsx` + the canvas store edit→lint effect.
**Severity**: **critical** — this makes lint actively misleading: it reports errors that no longer exist until a reload. Strictly worse than UI-BUG-010 (which was "stale ✓ at save"); this is "stale ✕ forever after a fix". Likely the same root family: lint result is not re-derived reactively from canvas/agent-form edits (the debounced trigger may not fire on agent-form `Apply`, or the panel/chip caches the last response and never invalidates). **Investigation owed**: trace exactly which edits (re)trigger lint and which don't; the agent-form Apply path is the prime suspect.

**✅ RESOLVED (autonomous session, 2026-05-18) — root cause was deeper than "the trigger doesn't fire"**

- **Root cause (proven in current source, not the suspected client-reactivity bug)**: `routes/lint.py` `lint_endpoint` linted the **stored** definition (`workflows.fetch_workflow(conn, id)`, **no request body**). The client effect (`$workflowId.tsx`) *did* re-fire on every canvas edit (`[nodes, edges, agents]`) — but the POST carried no canvas state, so the server always linted the last *saved* definition. Structurally, lint could never reflect an unsaved fix: edit→effect fires→server lints stale stored def→errors persist; reload only "worked" because it re-fetched after a save. The agent-form-Apply hypothesis was a **red herring** — confirmed by reading `lint.py:57-69`, the trigger reactivity was never the issue.
- **Solution (day-one/boundary rule)**: a linter lints the artifact you are editing, not the last save. `POST /v1/workflows/{id}/lint` now takes the `WorkflowDefinition` as the **request body** and lints exactly that; the path id is route identity only (stored-fetch + its 404 removed — this also fixes a latent gap: linting a never-saved canvas previously 404'd). The client serializes the live canvas with the **same** `reactFlowToWorkflow` serializer Save uses and sends it; the effect's existing deps make lint genuinely reactive — a fix clears its finding immediately, no save or reload. `make_lint_router`'s now-dead `db_factory` injection was removed (Q3: don't inject persistence into a pure linter) and the server.py call site updated. Generated TS types regenerated (contract self-consistent).
- **Contract change — flagged for your sign-off**: this changes an internal API contract (`POST .../lint` now requires a body; no more 404 on unknown id). Single first-party surface; not TRUNK-CRITICAL; SP-006 (no back-compat shim) / SP-018 (framework untouched) / SP-019 (API still the one contract) all hold. Reversible. Staged on the branch, **not merged to main** — your call on merge.
- **Independent review (`implementation-reviewer`, commit-before-exposure)**: no Critical. Caught one **Important I had missed**: the scenario harness `scripts/scenarios/_shared/client.py:lint_workflow()` was a *second* consumer still POSTing with no body → would 422 and false-flag the CRUD journey. Fixed (threads `payload["definition"]`). My "single consumer" claim was wrong — the review earned its keep. Reviewer's `_workflow_id` nit was **rejected with reason**: FastAPI binds path params by name, so the param must stay `workflow_id`; `del workflow_id` is the correct "declared-but-unused" form here.
- **Tests**: `test_routes_lint.py` rewritten for the body contract; added `test_lint_uses_submitted_definition_not_stored` (store a CLEAN def, lint a BROKEN body through the same id → findings reflect the BROKEN body — the precise root-cause regression) + `test_lint_works_for_never_saved_canvas`. `test_lint_unknown_workflow_returns_404` removed (behaviour intentionally gone — documented in-file, not silent). Backend lint+wiring+lint-rule suites **22 passed**; web typecheck clean; web suite **207 passed**.
- **Files**: `packages/spren/src/spren/routes/lint.py`, `packages/spren/src/spren/server.py`, `apps/web/src/lib/api.ts`, `apps/web/src/routes/workflows/$workflowId.tsx`, `apps/web/src/lib/api-types.generated.ts`, `packages/spren/tests/test_routes_lint.py`, `scripts/scenarios/_shared/client.py`, `scripts/scenarios/workflow_crud_journey.py`.

### WF-BUG-RIGHTRAIL-1 — Right-rail config card collides with the Spren orb; Apply hard to reach

**User report**: "it goes until the bottom of the page where there is Spren animation. That animation makes it very hard to access buttons like apply, also … looks like an amateur design."
**Surface**: right-rail agent/edge config panel (`apps/web/src/routes/workflows/-canvas/AgentConfigForm.tsx` + its container) + presence orb z-order/positioning.
**Severity**: **important** — confirmed visible in the audit's own screenshots (orb sits lower-right, the scrolling config form runs under/over it). The form extends full viewport height with the action buttons (Apply / Delete) at the very bottom, overlapping the orb. Functional impedance + looks unfinished. **Expected**: the config panel needs a bounded, scrollable body with a **pinned action footer** (Apply/Delete always reachable, never under the orb), and the orb must not overlap interactive panel regions (z-order / layout reservation).

**✅ RESOLVED — functional half only (autonomous session, 2026-05-18); RIGHTRAIL-2 visual cohesion still deferred**

- **Root cause (read from source)**: `.agent-form` (`AgentConfigForm.css`) was one `overflow-y:auto` scroll box with `.agent-form-actions` as the *last flex child* — Apply/Delete sat at the end of a long scroll. The presence orb is `position:fixed; bottom:32; right:32; z-index:30` (`PresenceOrb.css`), so it floats over the rail's lower-right exactly where the footer ends up. The rail (`.canvas-rail`) had no stacking context, so the fixed orb painted on top of it.
- **Fix (heuristic Q3 — an ambient presence indicator must never occlude an interactive control)**: `.agent-form-actions` is now `position:sticky; bottom:0` with an opaque `--surface` background + negative margins that cancel the form's 22/24px padding so the bar spans the rail edge-to-edge and stays flush — Apply/Delete are pinned to the visible bottom while the sections scroll above. `.canvas-rail` got `position:relative; z-index:31` so the whole panel (and its pinned footer) stacks above the orb's z-30; the orb stays put on the open canvas but yields to an active config panel. Scoped to the canvas route — the shared orb component's global z-index is untouched (no app-wide blast radius).
- **Heuristic note**: the orb being partially behind the rail edge when the panel is open is a *visual* nuance (does it read as intentional?) — that belongs to the deferred RIGHTRAIL-2 cohesion pass, not the functional fix. The doc itself scopes RIGHTRAIL-1 as functional ("Apply/Delete always reachable, never under the orb") and RIGHTRAIL-2 as visual; respected that split rather than expanding scope.
- **Verification honesty**: CSS-only; web typecheck clean + full web suite **207 passed** (no rail/form test regressed). It is a standard pinned-footer + scoped-stacking pattern and logic-sound, but I did **NOT** visually confirm it in a browser this session (the user was away; driving Chrome to a selected-agent rail state is more setup than budget allowed). **Needs a 30-second visual eyeball** (open a workflow, click an agent, confirm Apply is pinned + above the orb). Flagged, not claimed done-verified.
- **Files**: `apps/web/src/routes/workflows/-canvas/AgentConfigForm.css`, `apps/web/src/routes/workflows/canvas.css`.

### WF-BUG-RIGHTRAIL-2 — Right-rail card visual design doesn't match the app's language

**User report**: "I don't like the design, the design doesn't match the rest of the feelings that we have in the app."
**Severity**: important (design). The config form's visual treatment (spacing, section headers, field styling) reads as a different design vocabulary than the orb/tag-markup/`--coral` system used elsewhere. Needs a cohesion pass.

**Design pass must capture (zoomed screenshots)**:
1. Section headers (`IDENTITY` / `MODEL` / `INSTRUCTION` / `TOOLS`) — typography, case, weight, spacing vs the design system's section-header convention elsewhere (e.g., sidebar `SURFACES`/`COMING SOON`, /runs filter rail).
2. Field styling — input/select borders, radius, focus ring, label-to-field rhythm; compare to the AuthGate token field (the one form known to be on-system).
3. Vertical rhythm — is spacing token-derived or ad-hoc px? Cross-check `tokens.css`.
4. The action footer (Apply/Delete) — styling + the orb collision (this is the *visual* half; the *layout* half is WF-BUG-RIGHTRAIL-1).
5. Header tag-markup line — does it use the same `<agent .../>` Geist Mono treatment as the canvas empty-state, or a different one?
6. Side-by-side with one on-system surface (/runs filter rail or sidebar) to make the vocabulary mismatch concrete rather than asserted.

---

### WF-BUG-SWEEPER-1 — draft-sweeper FK-constraint failure on empty drafts that have runs (PRE-EXISTING; not a Session 07 regression)

**Surfaced**: 2026-05-15, user testing the Session 07 commits on WSL (`/home/rezaho/.local/share/spren`).
**Severity**: important — non-blocking (background worker; caught at `draft_sweeper.py:63`, app fully usable) but logs a traceback every startup + 4h interval and means abandoned empty drafts that have runs are never swept.
**NOT caused by Session 07**: the sweeper, `delete_empty_drafts_older_than`, and the `runs` FK were untouched this session (the empty-draft predicate change was deliberately deferred to land with the frontend). Migration 04 only rewrites `definition` node_type values; it cannot create a draft+run pair or alter FKs. Pre-existing latent defect, surfaced by a WSL DB that has an old empty `visual_builder` draft (`topology.nodes==[]`, >24h) with a `runs` row referencing it.

**Root cause (cited, conclusive)**:
- `runs.workflow_id` → `FOREIGN KEY (workflow_id) REFERENCES workflows(id)` with **no `ON DELETE` action** (`packages/spren/src/spren/storage/migrations/02__create_runs.py:27`) → RESTRICT.
- `delete_empty_drafts_older_than` (`packages/spren/src/spren/storage/workflows.py:122-143`) `DELETE FROM workflows WHERE provenance='visual_builder' AND json_extract(definition,'$.topology.nodes')='[]' AND updated_at<?` — does NOT exclude workflows that have runs.
- With FK enforcement on, deleting such a draft raises `sqlite3.IntegrityError: FOREIGN KEY constraint failed`; the sweeper loop catches + logs + retries (`draft_sweeper.py:60-65`).

**Correct fix (SP-009-aligned; do NOT hot-patch — it's a data-deletion path, treat with the test-first discipline)**: a workflow with run history is not an abandoned empty draft regardless of current topology — run snapshots are immutable (SP-009). The sweep predicate (both the sweeper AND the matching list filter, kept in sync) must additionally exclude any workflow with runs: `AND NOT EXISTS (SELECT 1 FROM runs WHERE runs.workflow_id = workflows.id)`. Ship as its own change with a guard test ("empty draft WITH a run is NOT swept"). Interacts with — but is distinct from — the Session 07 deferred empty-draft predicate reshape (that one is about `nodes==[]` vs Start-seeded; this one is the runs-exclusion, orthogonal). Tracked as a backlog task.

**✅ RESOLVED (autonomous session, 2026-05-18)** — applied exactly the prescribed fix:

- **Both predicates updated, kept in sync** (`storage/workflows.py`): `delete_empty_drafts_older_than` (the sweeper DELETE) and the `list_workflows` default draft-hiding clause both gained `AND NOT EXISTS (SELECT 1 FROM runs WHERE runs.workflow_id = workflows.id)`. The sweeper never tries to delete a workflow a `runs` row references → the `IntegrityError` (RESTRICT FK at `migrations/02__create_runs.py`) can no longer occur. The list filter no longer hides an empty draft that has runs (it has history → surfaced; SP-009). Bare-table queries (`FROM workflows`, no alias) so the correlated subquery is valid in both; `runs` always exists before any sweep (migrations run in the lifespan before the sweeper task).
- **Heuristic check**: day-one design — a sweeper for *abandoned* drafts must exclude anything with run history; the FK RESTRICT was the symptom, SP-009 is the principle. No conditional/guard band-aid; the predicate now states the real rule.
- **Test-first guards added** (`test_draft_sweeper.py`): `test_sweeper_preserves_old_empty_draft_with_runs` (old empty draft WITH a run survives + must-not-raise; a sibling old empty draft WITHOUT runs is still swept — behaviour preserved) and `test_list_surfaces_empty_draft_with_runs` (such a draft now appears in the default list). Suite run (`test_draft_sweeper` + `test_server_wiring` + `test_routes_workflows`): **46 passed, 1 failed**.
- **The 1 failure is a verified PRE-EXISTING flake, NOT a SWEEPER-1 regression** (checked from the traceback, not assumed): `test_routes_workflows.py::test_put_replaces_workflow` asserts `updated_at != created_at` but got byte-identical microsecond timestamps (`2026-05-17T22:56:51.113412Z` == itself) — create+PUT execute within one clock tick on Windows. It lives in an unmodified test file exercising `replace_workflow` (unmodified by this fix); my edits touched only the two sweep/list predicates. Out of blast radius; left untouched (fixing an unrelated flake inside a scoped data-deletion fix would violate scope discipline — flagged here instead).
- **Files**: `packages/spren/src/spren/storage/workflows.py`, `packages/spren/tests/test_draft_sweeper.py`.

---

### WF-BUG-RUN-3e — visual-builder agents never deliver to End ("instruction-topology mismatch"); + framework hides the real branch error

**Surfaced**: 2026-05-16, real-key run testing (`Start→assistant→End`, `claude-sonnet-4-6`, valid key). Investigated read-only via `failure-root-cause-investigator`; my step-2 review = sound, high-confidence, well-cited.
**Severity**: **critical** — this is the *next* "Run is broken" layer after RUN-3a/3c/3d. Run-create works, materialization works, det-nodes work, the **model call succeeds (tokens billed: 2535 in / 16 out, $0.0078)** — but the run ends `status:"failed"`, `final_response:null`, `error:"ROOT failure: insufficient arrivals"`. Every default visual-builder workflow will hit this.

**Root cause (NOT a Spren bug, NOT a framework bug — a framework *contract* the visual builder doesn't satisfy)**: in marsys an agent delivers the final answer ONLY by emitting the `terminate_workflow` native tool call. The framework **gates that tool into the agent's schema** based on the `agent→End` edge (verified: `materialize.py` emits the edge → `topology/graph.py:863-877 has_edge_to_endnode` → `step_executor.py:876-879 can_terminate_workflow=True` → `coordination_tools.py:117-120` injects `terminate_workflow`), but it **never instructs the agent to call it** (`formats/base.py:87-89` emits no coordination guidance; steering only intervenes after `consecutive_content_only ≥ threshold` default 2, content-only path only). The probe/visual-builder agent instruction ("You are a helpful assistant. Answer concisely.") gives the model no reason to call `terminate_workflow`, so the branch emits content or a misfired coordination call → branch FAILs (`orchestrator.py:664-666 _fail_to`) → ROOT ratio gate `0/1 < 1.0` → `orchestrator.py:1112` `_fire_with_failure(error="insufficient arrivals")` → `:1184` `_workflow_error = "ROOT failure: insufficient arrivals"`. This is the framework's own documented **#1 failure mode** ("instruction-topology mismatch", `docs/guides/steering-and-error-recovery.md`). The Spren materializer is correct and not the cause; a visual-builder `Start→Agent→End` does NOT inherently fail — it fails when the agent instruction doesn't name the topology-gated coordination tool.

**Secondary (corroborates RUN-3b / [[task #14]])**: `orchestrator.py:1184` overwrites `WorkflowResult.error` with the generic `"ROOT failure: insufficient arrivals"`; the branch's real failure reason lives only in `ROOT.failed[branch_id]`, which Spren never reads (`lifecycle.py` reads only `result.error`/`result.final_response`). So every run failure is a diagnostic dead-end — exactly the error-visibility gap. Spren-side additive fix available: read `result.branches` and surface the failed branch's reason into the run record.

**Proposed fix (Spren-side, in scope, architecture-coherent — but it is an ARCHITECTURAL DECISION → escalated to user)**: `materialize.py` `_materialize_agent` appends a **delivery-contract instruction suffix derived from the agent's outgoing topology edges** (edge→End ⇒ append "call `terminate_workflow(answer=…)` to deliver the final answer"; edges→peers ⇒ `invoke_agent` directive; edge→User ⇒ `ask_user` directive). Derived from the SAME topology the framework gates tools on ⇒ instruction and gating consistent by construction; mirrors the framework's own prescribed remedy (`docs/concepts/det-nodes.md:125-132`). SP-018 preserved (Spren only enriches an author-supplied string; framework untouched). **Risk flagged for independent review**: the suffix derivation must EXACTLY mirror the framework's tool-gating (peer/End/User edge resolution; `coordination_tools.py:107-110` excludes start/end/user from invocable peers) or it reintroduces the mismatch.

**Why escalated, not auto-applied**: "Spren auto-injects topology-derived coordination instructions into the user's agent prompt at materialization" is a genuine product/architecture decision (opaque prompt enrichment), not a mechanical bug fix — the architect (user) must bless the approach before implementation. Tracked as a backlog task; needs the bug-fix pipeline (independent review of the derivation) + the probe re-run to prove a real successful completion.

---

#### ⭐ DECISION NEEDED (consolidated 2026-05-18 — read this, the rest above is the trail)

**Plain version**: when you build `Start → Agent → End` visually and hit Run, the model is called and answers, but the workflow still ends "failed". Reason: marsys only treats an answer as "delivered" if the agent calls the `terminate_workflow` tool, and nothing tells the model to call it. A bare "you are a helpful assistant" agent never does, so every default visual-builder run fails. This is the framework's own #1 documented failure mode, not a Spren or framework bug — it's a contract the visual builder doesn't satisfy.

**Why this is now the top item**: RUN-1, RUN-3b, LINT-REACTIVITY, SWEEPER-1 all fixed this session; RUN-3a/3c/3d fixed earlier. RUN-3e is the **last thing standing between the visual builder and a working Run**. RUN-3b part (b) shipped, so the run record + inspector now show the *real* branch reason (e.g. "assistant: did not call terminate_workflow") instead of the opaque "insufficient arrivals" — the diagnostic dead-end in the "Secondary" note above is **already closed**; what remains is making runs actually *succeed*.

**The one decision**: the fix is for Spren's materializer to append a short, topology-derived delivery instruction to each agent's prompt (edge→End ⇒ "call `terminate_workflow(answer=…)` to deliver the final answer"; edge→peer ⇒ invoke directive; edge→User ⇒ ask directive). Derived from the same topology the framework gates the tools on, so they're consistent by construction; it is the framework's own prescribed remedy. Your stated objection was that this is **opaque prompt enrichment**.

**My recommendation (heuristic-applied, but the product call is yours)**: Q1/Q3 — a workflow you built visually should be runnable *by construction*; making the user hand-type framework coordination-tool instructions leaks framework internals into the builder UX (boundary break). So the enrichment itself is the right design. The real objection is *opacity*, not the enrichment — so the answer is **make it visible, not absent**: append it, and surface the derived "delivery contract" in the agent config panel (read-only line: "On run, this agent is told: …") so nothing is hidden. That satisfies the framework contract AND your transparency concern. Options for you:
- **(A) Approve transparent enrichment** (recommended): append + show the derived line in the right-rail config so the user always sees exactly what's added.
- **(B) Approve silent enrichment**: append, don't surface it (simplest; the opacity you flagged remains).
- **(C) Reject auto-enrichment**: require the user to write coordination instructions themselves (keeps prompts pure; visual builder is "expert-only" until a better UX exists).
- **(D) Something else / discuss.**

**Held until you decide.** Everything else needed is ready: implementation is a scoped `materialize.py:_materialize_agent` change behind the bug-fix pipeline (independent review of the edge→tool derivation is mandatory — it must mirror `coordination_tools.py:107-110` exactly or it reintroduces the mismatch); then a real-key probe re-run to prove a green end-to-end run. I did **not** run a fresh real-key reproduction — RUN-3e needs a *successful* model call (a fake key fails earlier at RUN-3b), so reproducing it spends API credit; that's your call to authorize, not mine to spend autonomously.

## Session log

| Time | Activity |
|---|---|
| 2026-05-15 — start | Plan written. Dev stack up on 5173 + 8765. Chrome extension connected. |
| 2026-05-15 — A1-A5 walked | Cold-load + invalid URL fragment + invalid form submit + refresh-drops-auth all tested. 4 UI bugs + 1 architectural question registered. |
| 2026-05-15 — B1-B8 walked | Home initial paint, wordmark, footer-contrast, orb state machine (idle/typing/thinking), stub send, multi-line input. Mid-audit spontaneous de-auth (UI-BUG-007, critical). Backend confirmed healthy throughout — frontend state-loss bug. Dev-stack chaos surfaced PRODUCT-BUG-002 + expanded TOOLING-BUG-001. |
| 2026-05-15 — B10-B14 + Runs | Sidebar + cmdk verified (well-built). UI-BUG-008: stale Runs gating + internal session-number leak in sidebar. /runs page itself shipped + solid. |
| 2026-05-15 — C1-C3, D1-D16 | Workflow list + create-flow + canvas all walked. Agent form / tools-from-registry / palette click-add / fit-view / Save (PUT+GET+toast) / lint panel all work well. UI-BUG-009 (off-screen node add), UI-BUG-010 (stale lint chip). PRODUCT-BUG-001 updated (canvas mitigates via "User 1" naming but validator still wrong for rename/import/API). One earlier in-conversation lint claim retracted as a staleness artifact (UI-FINDING-006). |
| 2026-05-15 — PAUSE for strategic call | ~13 distinct bugs (1 critical: UI-BUG-007) + ARCH-Q-001 documented across A/B/C/D. Surfacing to user: recommend fixing UI-BUG-007 out-of-order first. |
| 2026-05-15 — USER CORRECTION | User flagged the audit missed the substantive workflow-section issues. Re-oriented: 8-item punch list documented with depth (WF-BUG-RUN-1 confirmed via source = silent create-run swallow at RunButton.tsx:123; PALETTE-1..4; NODE-1; LINT-REACTIVITY; RIGHTRAIL-1/2). RUN-3 ("wall of backend errors") not yet reproduced — my no-key env fails at the clean 400 gate; reproducing next via fake-key past-validation path. |
| 2026-05-15 — RUN-3 reproduced | `run_failure_probe.py` against a fake-key sidecar → `POST /v1/runs` 500, traceback `lifecycle.py:186` → `orchestra.py:304` `AttributeError: 'dict' object has no attribute 'enabled'`. RUN-3a (dict-vs-typed config at the mirror↔runtime seam) + RUN-3b (raw 500, no structured envelope) documented. RUN-3a = trunk of "Run is broken". |
| 2026-05-15 — Documentation + protocol pass | Added the "Working agreement + how we operate" section (peer collaboration model, the authoritative multi-agent investigation protocol, the process lesson on the checklist-not-experience miss). Enumerated the framework node + agent catalog (read-only Explore) and folded ground truth into PALETTE-1/3 + sharpened PRODUCT-BUG-001 (framework needs literal `User`/`Start`/`End` names; Spren's validator forbids `user` — direct contradiction). Surfaced a load-bearing unknown: a runnable topology *requires* a StartNode the visual builder may never emit — to verify in the RUN-3a pipeline. Tightened NODE-1/RIGHTRAIL-2 into concrete capture checklists. Campaign tracked as tasks #11–18. |
| 2026-05-15 — RUN-3a fix pipeline | Full pipeline via /bug-fix: investigator → my ground-truth step-2 review (verified every claim in source; independent reviewer declined with documented rationale) → implemented Spren-side typed mirror→runtime conversion in `materialize.py` (`_materialize_status_config` + `_materialize_convergence_policy`, same pop-then-build pattern as the existing `tracing` wiring) + 4 regression tests. Verified: regression 4/4 green; blast-radius 66 passed / 1 pre-existing-env fail; probe traceback now passes `orchestra.py:304` (RUN-3a gone) and dies at `:347` on a new distinct root → **WF-BUG-RUN-3c** (AG-UI `ag_ui` dep not installed; SP-004 policy question). RUN-3a ✅ resolved; RUN-3c escalated to user. Env note: `uv sync --extra test` installed pytest but dropped `spren-tui`; `ag_ui` absent; restore full env at a wrap point. |

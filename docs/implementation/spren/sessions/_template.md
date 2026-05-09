# Session NN: <Title>

> Copy this file to `NN-<short-title>.md` and fill out every section. Don't skip any. If a section truly doesn't apply, write "n/a — <one-line reason>" so the implementer knows it was considered.

---

## Working rules — how we collaborate (READ FIRST, every session)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of Spren. We work together — not above-and-below.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend your position with evidence. Don't change your mind just because someone asked a question — only when you're genuinely convinced.
- **Stay engaged.** This is collaborative work, not request-response. Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it. If you see something this session is doing that conflicts with the architecture, raise it.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure. If something needs to change to succeed, change it (within scope) or escalate (out of scope).
- **You own correctness.** Tests passing isn't enough — the work has to actually do what was promised. Manually verify acceptance criteria, not just unit tests.
- **You own follow-through.** Update this session file with what was actually built. Update the version's roadmap doc to reflect outcomes. Add a "Lessons / Surprises" entry if anything surprised you.

### Double-check before any decision

- **Read the code before changing it.** Don't assume a function does what its name suggests — read it. Don't assume a config has a certain value — open the file.
- **Verify file paths and symbols still exist** before referencing them. The brief may be days old; the code may have moved.
- **Re-confirm assumptions out loud** in tool-call comments or notes. If you're about to act on a belief like "this dependency exports X", state the belief and verify it.
- **Run tests after every meaningful change**, not just at the end. A failing test caught immediately is a 5-minute fix; caught at session end it's a 2-hour debugging session.
- **Use git commits as checkpoints.** After each acceptance criterion is met and tests pass, commit. If something later goes wrong, you can revert to a known-good state.

### Critically assess the plan itself

This brief was written by humans and other agents. It might be wrong, outdated, or naive about something. **Don't follow it blindly.**

- **Read the actual code** for any module the brief references. If the code contradicts the brief, surface it before writing.
- **Spawn an independent sub-agent for verification** when you have material doubt about a load-bearing claim. Use Agent / general-purpose with a prompt like: "I'm about to do X based on the assumption that Y. Read [files] and tell me whether Y holds. Disagree if you find evidence against Y." Two perspectives catch errors a single perspective misses.
- **Run online research** for any "best practice" or "we use library X because Y" claim that isn't obviously self-evident. If 2026 docs disagree with this brief, defer to current docs and push back.
- **Cite sources when you challenge the plan.** "I think this is wrong" is weak; "I think this is wrong because [docs/issue/blog post] says [X]" is strong.

### Ask the user when blocked on intent

Strategic, opinionated, or subjective decisions belong to the user. **Don't guess.** Use `AskUserQuestion` to surface them. Examples that warrant asking:

- **Strategic:** "Should this feature be deferred to v0.3 because it's growing larger than scoped?"
- **Product vision:** "The brief says X but the bigger picture in [overview] suggests Y might be better. Which interpretation is right?"
- **Opinionated/subjective:** "There are two reasonable shapes for this API; the brief is ambiguous. Which do you prefer?"
- **Something doesn't feel right:** "The plan has us doing X but I think it'll cause Y issue six months out. Worth discussing."
- **Out-of-scope expansion:** "Implementing X correctly requires also touching Z, which is out of scope. Should I expand scope or stop and split into two sessions?"

**Do NOT** use AskUserQuestion for purely technical implementation choices (e.g., "should this variable be named foo or bar"). Those you decide. Ask for *intent*, not for decisions you can make.

### Build the smallest thing that works first

- Don't try to land everything at once. Get the smallest version that satisfies acceptance criteria, then iterate.
- If you finish with time + budget left, iterate within scope OR (better) flag the next item for the next session — don't silently expand scope.

### Don't expand scope silently

If you discover a missing piece while implementing, surface it. Either:
- It's small enough to absorb (note it in "What was actually built")
- It's not, and you open a follow-up session

NEVER silently add a feature that wasn't in the brief.

### Foundational project rules (in addition to this session brief)

- [`/CLAUDE.md`](../../../../CLAUDE.md) — TRUNK criticality map, design principles DP-001..DP-007, env setup, git rules
- [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — Spren-specific principles SP-001..SP-019

---

## The big picture — what we're building and why

> Filled in identically across all sessions of a version. Read it every time so the session is grounded in the product, not just its own scope.

### What Spren is

Spren is a **continuously-active personal AI assistant** that lives as a local daemon on the user's machine. It has its own event loop (responds to messages, scheduled triggers, workflow lifecycle events, and heartbeat ticks), persistent memory across sessions and channels, the ability to spawn sub-agents and team managers, and the authority to act on the user's behalf with appropriate confirmation flows.

It is **the agent that runs your other agents.** It sits on top of the open-source [marsys Python framework](../../../../packages/framework/src/marsys/) and uses marsys as a tool to execute user-defined multi-agent workflows. The seam between framework and Spren is exactly three doors (SP-018): `Orchestra.run()`, `EventBus.subscribe()`, `TelemetrySink`.

It is **NOT** a chat surface. The home page is a four-surface command center: "Now" (what the agent is currently doing), "Since you were away" (items needing decision), "Activity" (chronological log), and a chat input. The chat is one of four surfaces, not the page.

Spren ships three product surfaces that all consume the same FastAPI backend (SP-019): the **desktop GUI** (Tauri webview loading the Vite + React + TanStack Router bundle), the **browser GUI** (same Vite bundle served by the Python sidecar at `http://127.0.0.1:<port>/`), and the **TUI** (Textual; ships in v0.4). Distribution is the native installer per platform with secondary channels (brew / winget / apt / npm / pipx / Docker).

It is **distinct from** two proprietary products in sibling repos: MARSYS Cloud (hosted control plane) and MARSYS Studio (hosted SaaS UI). Spren stays focused on the **single-user-local** case so it doesn't cannibalize the proprietary product line.

### Why this work matters

Today the only way to use the marsys framework is to write Python. That excludes the audience the framework's capabilities deserve — AI builders who pay for ChatGPT Plus and use Cursor or Claude Code, but don't write Python every day. A visual builder + always-on personal assistant lowers the barrier dramatically.

Beyond visual building, Spren's pitch is **observability + agency**: the user can see why their agents made the decisions they made (nested span traces, cost per span), AND can delegate the operational work of running and tuning their agents to the meta-agent. Spren is the tool you give your AI agents to so you don't have to micro-manage them.

### Your role as an implementer

You are building one slice of a careful product. The slice has been scoped, designed, and discussed at length. Your job:

1. **Honor the architecture** ([`docs/architecture/spren/`](../../../architecture/spren/)) — especially the design principles (SP-001..SP-019). Violating one = ASK FIRST.
2. **Honor the design principles**: no mocks of in-codebase features (SP-007), no backward-compatibility shims (SP-006), no TRUNK-CRITICAL framework changes (SP-001), framework knows nothing of Spren (SP-018), API is single source of truth (SP-019), bounded triggers (SP-011), hard rails always confirm (SP-012), cost ceiling is load-bearing (SP-013), markdown is source of truth for memory (SP-016), forget tombstones (SP-017), and the rest.
3. **Ship a working artifact at session end.** Sessions don't end mid-feature. If a feature is too big, split into two; don't half-finish.
4. **Write all required tests** (unit + integration + E2E where applicable) — non-negotiable.
5. **Push back when something is wrong.** You're a peer.

### Where to read deeper if you need it

- High-level: [`docs/architecture/spren/00-overview.md`](../../../architecture/spren/00-overview.md)
- System context (how Spren relates to marsys / Cloud / Studio): [`docs/architecture/spren/01-system-context.md`](../../../architecture/spren/01-system-context.md)
- Frontend architecture (Vite + React + TanStack Router + Tauri): [`docs/architecture/spren/04-frontend-architecture.md`](../../../architecture/spren/04-frontend-architecture.md)
- Packaging & distribution (native installer + secondary channels): [`docs/architecture/spren/05-packaging-distribution.md`](../../../architecture/spren/05-packaging-distribution.md)
- The meta-agent design (execution model, agent hierarchy, persona, authority, sandbox): [`docs/architecture/spren/09-meta-agent.md`](../../../architecture/spren/09-meta-agent.md)
- Memory architecture (4 tiers, supersession, consolidation, indexer): [`docs/architecture/spren/10-memory-architecture.md`](../../../architecture/spren/10-memory-architecture.md)
- The full v0.3 plan and session list: [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md), [`docs/implementation/spren/00-overview.md`](../00-overview.md)

---

## What came before this session

> Brief summary of the previous session(s) outcomes — enough that the implementer knows the starting state without reading every prior session file. Pointer to specific previous session briefs if deeper context is needed.

**Previous sessions:**
<list with one-line summary of what each previous session shipped, OR "None — this is the first session" for Session 01>

**State at start of this session:**
<2-4 sentences describing what already exists in the repo / what's been deployed / what's working. The implementer should be able to verify this state with a few git/grep commands before they start.>

**Relevant prior session briefs to consult if needed:**
- `sessions/<NN>-<title>.md` — <one-line description of when to read it>

---

## What this session ships

> One paragraph describing the outcome the user can verify after this session. Concrete, testable, observable.

<paragraph>

### Acceptance criteria

- [ ] <criterion 1; testable>
- [ ] <criterion 2>
- [ ] <criterion 3>
- [ ] All required tests written and passing (see Tests section)
- [ ] No mocks of in-codebase features (SP-007)
- [ ] No backward-compat code (SP-006)
- [ ] No TRUNK-CRITICAL framework changes (SP-001) — or explicit deviation approved in this brief

---

## Background reading (do this before writing code)

> Do not skim. Read these.

1. [`/CLAUDE.md`](../../../../CLAUDE.md) — project rules, TRUNK criticality
2. [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — Spren design principles
3. [`docs/architecture/spren/<relevant-area-doc>.md`](../../../architecture/spren/) — architecture for this session's area
4. [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) — current version plan
5. <source files in `packages/framework/src/marsys/...` or `packages/spren/src/spren/...` or `apps/web/src/...` or `apps/desktop/...` or `apps/tui/...`> — verify current state of any code this brief references
6. <other relevant external docs / blog posts / RFCs the brief cites>

**Verify before proceeding:**
- Run `git log --oneline -20 <relevant paths>` to see recent activity
- Run `grep -r '<key symbols>' <paths>` to confirm symbols still exist where the brief says they do
- Read referenced files end-to-end for any module you're integrating with
- If something has moved/renamed/changed, **update the brief or escalate** before writing code

---

## Detailed plan

> Files to create/modify with paths and short descriptions. Code shape where it's load-bearing (function signatures, key data shapes, key SQL). NOT line-by-line implementation. Leave room for the implementer's judgment on naming, internal structure, comments.

### Files to create

- `<path>` — <description>

### Files to modify

- `<path>` — <change>; <reason>

### Files NOT to touch

- `<path>` — <why it's off-limits this session>

### Load-bearing shapes

> If there are data shapes, function signatures, SQL schemas, or HTTP contracts that the rest of the system depends on, define them here. Other things implementer chooses.

```python
# example
class Foo(BaseModel):
    bar: int
    baz: str
```

---

## Hard rules (per session, every session)

### No mocks of in-codebase features (SP-007)

**Forbidden:**
- Implementing a fake version of feature X to "test the rest of the system" then forgetting it's fake
- Stub functions that return canned data so the build passes when the underlying feature isn't done
- Placeholder UI components that show static content "for now"

**Allowed:**
- Test fixtures that record/replay external LLM/HTTP responses (in `tests/fixtures/`, clearly named, only test code consumes)
- Boundary stubs at network seams in tests (FastAPI test client, MSW for frontend) where the seam IS the test contract
- Pre-recorded VCR-style cassettes for deterministic tests against real APIs (in `tests/cassettes/`)

If you'd need to stub a missing internal feature: **don't**. Either expand scope to include it OR split this session.

### No backward compatibility (SP-006)

**Forbidden:**
- Keeping the old shape alongside the new shape with conditional routing
- Deprecation shims, legacy fallbacks, "TODO remove after vN" code
- Feature flags that keep multiple implementations alive in production

**Allowed:**
- **Migrations** — one-shot transformations that read v_n stored data and write v_{n+1}. The post-migration code only knows the new shape.
- API versioning at the public HTTP boundary if/when we have third-party API consumers (not yet — use one version)

When changing a feature's shape: change it everywhere. Delete the old code. Stored data gets a migration.

### No TRUNK-CRITICAL framework changes (SP-001)

Per `/CLAUDE.md`, the TRUNK-CRITICAL framework files are `packages/framework/src/marsys/coordination/orchestra.py`, `packages/framework/src/marsys/coordination/execution/orchestrator.py`, `packages/framework/src/marsys/coordination/execution/real_runtime.py`, `packages/framework/src/marsys/coordination/validation/response_validator.py`, `packages/framework/src/marsys/coordination/topology/graph.py`. Don't modify without explicit approval. New Spren code lives in `packages/spren/src/spren/`, `apps/web/src/`, `apps/desktop/src-tauri/`, `apps/tui/src/`, or wherever the brief specifies for that session.

### Other Spren design principles (SP-001..SP-019)

See [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md). Especially:
- SP-002: localhost-only by default + per-launch token
- SP-003: one transport per concern (REST/SSE/POST; no WS)
- SP-004: AG-UI as wire schema
- SP-005: Pydantic is source of truth for types
- SP-011: bounded triggers — agent wakes only via the inbox; producers are external sources / time scheduler / system watchers / heartbeat / sub-instance completions; no ad-hoc polling, no runtime-spawned watchers
- SP-012: hard rails always confirm
- SP-013: cost ceiling load-bearing
- SP-014: working instances persist scratchpad atomically
- SP-015: untrusted-channel writes never live-touch memory
- SP-016: markdown is source of truth for memory
- SP-017: forget tombstones, never deletes
- SP-018: framework knows nothing of Spren — three doors only (`Orchestra.run()`, `EventBus.subscribe()`, `TelemetrySink`)
- SP-019: API is single source of truth — every client (Tauri webview, browser tab, TUI, Python adapter SDK) consumes the same FastAPI surface; zero UI logic in backend

### Clean code rules

- Smallest implementation that passes acceptance criteria
- Delete code aggressively when it stops being used; no commented-out blocks
- No descriptive comments for self-naming code — only WHY when not obvious
- One concern per file when reasonable; don't pre-factor
- Don't add error handling for scenarios that can't happen (trust internal types and framework guarantees; only validate at system boundaries)

---

## Tests (required for "done")

Tests are not optional, not deferred. Every session ships all three test types that apply to its work.

### Unit tests

- Small, fast, deterministic
- External boundaries (LLM, network, filesystem outside `tmp/`) stubbed via fixtures
- Cover the logic written this session
- Live in `packages/<package>/tests/<mirror of src layout>/test_*.py` (Python) or `apps/web/src/<...>.test.ts` (TS) or `apps/tui/tests/` (TUI Python)

### Integration tests

- Exercise the boundary between the changed module and its real collaborators
- Real SQLite DB, real FastAPI test client, real marsys framework code
- Mock ONLY the LLM provider call (via cassette or fixture)
- Live in `packages/<package>/tests/integration/` or `apps/web/tests/integration/`

### End-to-end tests

- At least one scenario exercising the system as a real user would, including UI where applicable
- Playwright for browser flows; tauri-driver for Tauri shell flows; full FastAPI server lifecycle in test setup/teardown
- Live in `apps/web/tests/e2e/` for browser; `apps/desktop/tests/e2e/` for Tauri; `apps/tui/tests/e2e/` for TUI

### Test framework conventions

- Python: pytest; fixtures in `conftest.py`
- TypeScript: Vitest for unit/integration; Playwright for browser E2E; tauri-driver for Tauri E2E
- TUI: pytest + Textual's pilot test harness
- Coverage NOT pursued as a metric; correctness pursued via the three-test discipline

### When a session legitimately can't produce one type

Explain why in this section. Examples:
- "Backend-only session has no UI to E2E" → unit + integration cover it
- "Pure refactor with no behavior change" → existing tests cover it; document any new edge cases

---

## Open questions for the user

> List anything the implementer should ask before starting. If empty, write "None — session is fully specified." If non-empty, the implementer must use AskUserQuestion at session start.

- <question 1>
- <question 2>

---

## Sign-off

On completion:

1. Update **What was actually built** below with the delta from the plan, if any
2. Update [`docs/implementation/spren/v0.3-mvp.md`](../v0.3-mvp.md) checkboxes to reflect this session's outcomes
3. Add a **Lessons / Surprises** entry below if anything surprising came up

### What was actually built (filled by implementer)

> _Implementer fills this in when the session is done. Plan vs reality delta. Anything added, removed, or done differently from the brief, with reasons._

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in. Anything that future sessions should know — gotchas, things that took longer than expected, things that turned out to be easier, decisions that ended up mattering._

# Framework Session NN: <Title>

> Copy this file to `NN-<short-title>.md` and fill out every section. If a section truly doesn't apply, write "n/a — <one-line reason>".

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of the marsys framework. We work together — not above-and-below.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if a "best practice" cited here is outdated, or if a structural choice will cause us pain later, say so. Defend your position with evidence.
- **Stay engaged.** Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it. If a Spren-side assumption embedded in this brief doesn't hold from the framework's perspective, push back loudly.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify acceptance criteria, not just unit tests.
- **You own follow-through.** Update this file's "What was actually built". Update the corresponding `v0.X-spren-support.md` checkbox. Add "Lessons / Surprises" if anything surprised you.

### Double-check before any decision

- **Read the framework code before changing it.** Don't assume; verify.
- **Verify file paths and symbols still exist** before referencing them.
- **Run framework tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief was written from the Spren-side perspective. **It might miss framework-internal constraints.**

- **Read the actual framework code** for any module the brief references.
- **Spawn an independent verification sub-agent** when material doubt exists.
- **Run online research** for any "best practice" claim.
- **Cite sources when you challenge the plan.**

### Ask the framework team when blocked on intent

Strategic, opinionated, or architectural decisions belong to the framework team (not Spren-side planning). Use `AskUserQuestion` for:

- **Strategic:** "This feature could be implemented two ways. Which fits the framework's direction?"
- **Architectural:** "This brief assumes the right place for this feature is module X. Is it actually module Y?"
- **Naming / API surface:** "The brief proposes name X for this protocol. Is there a framework convention I should follow?"
- **Scope expansion:** "Implementing this requires also touching Z. Should I expand or split?"

Do NOT ask for purely technical implementation choices you can decide.

### Multi-consumer justification (every feature, no exceptions)

This is what makes a feature framework-appropriate vs Spren-only. Before writing code, the implementer must be able to name at least one OTHER plausible consumer of this feature beyond Spren. If only Spren can use it, the design is wrong — escalate.

Examples of valid other consumers: LangSmith / Phoenix / OpenLLMetry / Langfuse adapters; MARSYS Cloud; MARSYS Studio; third-party Python users instrumenting their own marsys workflows; framework's own internal tooling.

### Build the smallest thing that works first

- Don't try to land everything at once. Get the smallest version that satisfies acceptance criteria, then iterate.
- If you finish with time + budget left, iterate within scope OR flag the next item for the next session.

### Don't expand scope silently

- Frame's PR review will catch silent scope expansion; don't waste reviewer cycles.

### Foundational project rules (in addition to this session brief)

- The framework worktree's `CLAUDE.md` — TRUNK criticality, framework design principles DP-001..DP-007
- Framework architecture docs (in the framework worktree's `docs/architecture/framework/`)
- This umbrella's [`/CLAUDE.md`](../../../../CLAUDE.md) — for context on Spren as a consumer
- Spren design principles (SP-018 framework purity, SP-019 API as truth) — for understanding what Spren expects of the framework: [`docs/architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md)

---

## The big picture — what this feature is and why

> Filled in identically across all framework sessions. Read it every time so the session is grounded in the bigger picture, not just its own scope.

### Why the framework has features that come from Spren-side planning

The marsys framework serves multiple consumers: Spren (this OSS umbrella's meta-agent), MARSYS Cloud (proprietary hosted control plane), MARSYS Studio (proprietary hosted UI), third-party Python users, and observability backends (LangSmith / Phoenix / OpenLLMetry / Langfuse / custom).

When one consumer (Spren) needs a framework capability that doesn't yet exist, the resulting framework feature must be **multi-consumer-justifiable** — usable by other consumers, not Spren-specific. The Spren team plans the feature, drafts the brief, and the framework team implements it on the framework's normal contribution flow.

This dir (`docs/implementation/framework/`) is where Spren-driven framework work is planned. Each session ships as one framework PR; Spren consumes the merged feature in a follow-up Spren-side session.

### What MARSYS Spren is (consumer context)

MARSYS Spren is the open-source umbrella product on top of the marsys framework. It contains a continuously-active personal AI assistant (the meta-agent), a visual workflow builder, a Tauri desktop shell, a Textual TUI, and a Python adapter SDK. It consumes the framework via three doors (per Spren's SP-018):

1. **`Orchestra.run(topology, task) → OrchestraResult`** — finite workflow execution
2. **`EventBus.subscribe(event_type, listener)`** — in-process workflow lifecycle events
3. **`TelemetrySink` protocol** — generic observability hook (added by Session 02)

This session adds (or extends) one of those doors. The implementation must respect the framework's own design principles (DP-001..DP-007) and avoid TRUNK-CRITICAL changes.

### Your role as a framework implementer

1. Honor the framework's architecture (in the framework worktree)
2. Honor the framework's design principles (DP-001..DP-007)
3. Honor multi-consumer justification (no Spren-only paths)
4. Ship a working PR — never half-finished
5. Write all required tests
6. Push back when something is wrong

### Where to read deeper if you need it

- Framework's own architecture docs (in the framework worktree): `docs/architecture/framework/`
- Framework's own `CLAUDE.md` (in the framework worktree)
- Spren's view of this feature: [`../00-overview.md`](../00-overview.md) and the corresponding `v0.X-spren-support.md`
- Affected framework modules in `packages/framework/src/marsys/...`

---

## What came before this session

> Brief summary of the framework state this session starts from. Pointers to relevant prior sessions if any.

**Previous framework PRs from this dir:**
<list of prior framework sessions and what they shipped, OR "None — first framework PR motivated by Spren">

**State at start of this session:**
<2-4 sentences describing what's in the framework currently. Implementer should be able to verify with grep / git log / reading.>

**Verify state with:**
```bash
cd /home/rezaho/research_projects/marsys-spren-work/packages/framework/
source ../../.venv/bin/activate
pytest tests/ -x --tb=short                # capture baseline test counts
git log --oneline -20 src/marsys/<relevant-module>/
grep -rn '<key symbols>' src/marsys/<relevant paths>/
```

---

## What this session ships

> One paragraph describing the merged-PR outcome. Concrete, testable, observable.

<paragraph>

### Acceptance criteria

- [ ] <criterion 1; testable in CI>
- [ ] <criterion 2>
- [ ] Framework regression suite green (zero new failures vs. baseline)
- [ ] New feature has its own test coverage (unit + integration where applicable)
- [ ] **Multi-consumer justification documented in PR description**: explicit list of consumers beyond Spren who can use this surface
- [ ] No TRUNK-CRITICAL changes (or explicit framework-team approval cited in PR)
- [ ] Framework architecture docs updated (where applicable)
- [ ] Framework's CHANGELOG.md entry added
- [ ] PR description references this session brief
- [ ] PR merges to framework's main; tagged framework release that includes it noted in this session file's "What was actually built"

---

## Background reading (do this before writing code)

> Don't skim. Read these.

1. The framework worktree's `CLAUDE.md`
2. The framework architecture docs in the framework worktree (especially the area this session touches)
3. [`../00-overview.md`](../00-overview.md) — Spren-driven framework backlog
4. [`../../architecture/spren/08-design-principles.md`](../../../architecture/spren/08-design-principles.md) — SP-018 (the seam Spren expects to consume), SP-019 (API as truth)
5. <source files in `packages/framework/src/marsys/...`> — verify current state of any code this brief references
6. <relevant external docs / RFCs / blog posts cited in the brief>

**Verify before proceeding:**
- Capture baseline test counts BEFORE any change
- `git log --oneline -20 <relevant paths>` to see recent activity
- Read referenced framework files end-to-end for any module you're integrating with
- If something has moved/renamed/changed, **update the brief or escalate** before writing code

---

## Detailed plan

> Files to create/modify. Code shape where it's load-bearing. NOT line-by-line implementation.

### Files to create

- `packages/framework/src/marsys/<path>` — <description>

### Files to modify

- `packages/framework/src/marsys/<path>` — <change>; <reason>

### Files NOT to touch

- TRUNK-CRITICAL: `packages/framework/src/marsys/coordination/orchestra.py`, `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`, `coordination/topology/graph.py`
- <other off-limits modules>

### Load-bearing shapes

> If there are protocols, classes, or function signatures that downstream consumers will rely on, define them here precisely. Other things implementer chooses.

```python
# example
class Foo(Protocol):
    async def bar(self, x: int) -> str: ...
```

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] List at least one consumer beyond Spren that can use this feature
- [ ] No Spren type imported in this PR (Spren is in `packages/spren/`, not `packages/framework/`)
- [ ] No "if running under Spren" code paths

### Framework design principles

Per the framework's `CLAUDE.md` § design principles:
- DP-001: pure agent logic
- DP-002: centralized validation
- DP-003: unified-barrier orchestration
- DP-004: branch isolation
- DP-005: topology-driven routing
- DP-006: adapter pattern
- DP-007: format pluggability

If this feature would force a violation of any of these, **escalate** before writing code.

### No TRUNK-CRITICAL changes (framework's own SP-001 equivalent)

Anything that requires a non-additive change to `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph` requires explicit framework-team approval AND an ADR.

### Clean code rules

- Smallest implementation that passes acceptance criteria
- Delete code aggressively when it stops being used
- No descriptive comments for self-naming code — only WHY when not obvious
- One concern per file when reasonable

---

## Tests (required for "done")

### Unit tests

- Small, fast, deterministic
- Live in `packages/framework/tests/<mirror of src layout>/test_*.py`
- Cover the new feature's logic

### Integration tests

- Exercise the new feature against real framework collaborators
- Live in `packages/framework/tests/integration/`
- Mock ONLY external boundaries (LLM provider) where applicable

### Framework regression test

- Entire framework test suite passes with the SAME counts as baseline (no new failures, no skips introduced silently)
- Document baseline + post-change counts in "What was actually built"

### Multi-consumer test

- At minimum, write a test that exercises the feature from a consumer-style call (not just from inside the framework's own internals). Demonstrates the feature is genuinely usable.

---

## Open questions for the framework team

> List anything the implementer should ask before starting. If empty, write "None — session is fully specified."

- <question 1>
- <question 2>

---

## Sign-off

On completion:

1. Update **What was actually built** below with the delta from the plan, if any
2. Update [`../v0.X-spren-support.md`](../v0.X-spren-support.md) checkbox for this feature
3. Note the framework release version that ships this feature (e.g., "shipped in framework v0.3.1")
4. Add a **Lessons / Surprises** entry below

### What was actually built (filled by implementer)

> _Implementer fills this in._
>
> Include: baseline test counts (before change), post-change test counts (must match for regression suite + new tests added), framework PR number + URL, framework release version that includes this feature, anything done differently from the plan with reasons.

### Lessons / Surprises (filled by implementer)

> _Implementer fills this in._

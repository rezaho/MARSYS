# Framework Implementation — Spren-Driven Backlog

Plans for marsys framework features that Spren needs in order to ship its versioned releases. The framework has its own release cadence and contributor base; this directory tracks the **subset of framework work motivated by Spren** so the two product lines can be planned and executed in parallel.

## Reading order

1. [`00-overview.md`](./00-overview.md) — principles, alignment with Spren versions, dependency-direction rules
2. [`v0.3-spren-support.md`](./v0.3-spren-support.md) — framework features required for Spren v0.3 to ship
3. [`v0.4-spren-support.md`](./v0.4-spren-support.md) — framework features required for Spren v0.4 to ship
4. [`v0.5-future.md`](./v0.5-future.md) — looser plan tied to Spren v0.5 and beyond
5. [`sessions/`](./sessions/) — per-session implementation briefs, each landing as a single framework PR

## What this directory is

- A **Spren-side view** of framework work. The framework team owns the sequencing, naming, and release-cadence of these features within the framework's own version line. This dir says "by Spren v0.X, framework feature Y must be merged and released."
- An **alignment artifact**. When Spren plans a session that depends on a framework feature, the corresponding framework session brief gets pulled forward.
- A **per-session framework PR template** so each item is small, testable, and reviewable as a single PR.

## What this directory is NOT

- The framework's own roadmap. The framework has independent direction (its current major refactor — unified-barrier orchestration — happened on its own; Spren consumes the result).
- A claim on framework version numbers. The framework picks its own release cadence; this dir notes "needed by Spren v0.X" without prescribing whether it lands in framework v0.3.x patches, a v0.4 major, etc.
- TRUNK-CRITICAL framework changes from the Spren branch. Every item here lands via the framework's normal contribution flow (feature branch on the framework's repo / branch, PR to main, framework review).

## How sessions work

Each session in `sessions/` is a self-contained framework PR brief:

- Scope: one coherent feature (e.g., one new module, one protocol, one writer)
- Acceptance: framework regression tests pass; the new feature has its own tests; documentation updated
- Out of scope: anything Spren-side. Spren consumes the merged framework feature in a follow-up Spren-side session.

## Cross-references

- Spren-side roadmap: [`../spren/`](../spren/)
- Spren-side architecture (consumer): [`../../architecture/spren/`](../../architecture/spren/) — especially `01-system-context.md` § "Framework contributions Spren motivates" and `08-design-principles.md` SP-018
- Framework architecture: [`../../architecture/framework/`](../../architecture/framework/) (canonical content lives in the framework worktree)
- Framework's own working rules: framework's `CLAUDE.md` (in the framework worktree)

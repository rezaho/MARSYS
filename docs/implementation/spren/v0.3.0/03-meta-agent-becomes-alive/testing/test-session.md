# Bundle 03 (Meta-Agent Becomes Alive) — Test Session

> Status: **stub** until Sessions 06 + 07 + 08 ship. Fleshed out fully on the first bundle-end testing pass. Target audience: a Claude Code agent with browser MCP + Playwright + tauri-driver + httpx access. Runs the scenarios in [`./test-scenarios.md`](./test-scenarios.md) across the four testing layers and produces a result report.

## Goal

Execute the four-layer test suite for Bundle 03 (Sessions 06 + 07 + 08 → "the meta-agent breathes"):

1. **Scripted integration** — Playwright + tauri-driver + httpx, deterministic. Runs in CI via `just bundle-03`. CLI tests via `click.testing.CliRunner` invoked through subprocess for end-to-end realism.
2. **Agent-driven exploratory** — Read [`./test-scenarios.md`](./test-scenarios.md), run through every G-* / E-* / X-* / U-* scenario plus self-directed variations. Per-archetype voice tests verify the felt-experience matches `05-five-archetypes.md`. Produce `./test-runs/<ISO-date>-exploratory.md`.
3. **Visual regression** — Playwright screenshot snapshots extending Bundle 01 + 02's baseline set. Bundle 03 adds: onboarding picker, each archetype's home page, suggest-with-confirm UI variants, bond status surface.
4. **User manual smoke** — short checklist (~8 items, ~8 minutes) on a fresh data dir + fresh install. Bundle ships only on 100% pass.

## What the testing agent does

(Filled in when this stub is promoted. Should include: harness setup, environment preconditions including LLM API keys for the meta-agent, scenario execution order, result-report shape, escalation rules for ambiguous failures, GIF-recording conventions for the per-archetype voice samples, cost-budget guard so the agent doesn't burn tokens looping on a flaky scenario, the consolidation-pass override `just consolidate-now` so the agent doesn't have to wait 24h.)

Special considerations for Bundle 03:

- **Per-archetype voice tests** are the most novel testing surface. The agent compares Spren's outputs against the canonical voice samples in `05-five-archetypes.md` for each archetype; uses an LLM-as-judge for register matching; flags drift.
- **Persona-evolution tests** require the agent to wait for (or trigger) consolidation passes; verify the resulting proposals; approve/reject via CLI; verify the persona file diff matches the approved proposal.
- **Bond reset** is destructive. Tests run on isolated data dirs; the bond reset doesn't pollute other test runs.

## What the user does after the agent finishes

(Filled in when this stub is promoted. Should include: how to read the agent's report, which findings warrant a fix vs. a follow-up session, how to record the manual smoke results, when Bundle 03 is considered shipped — the bond mechanism's longitudinal nature means "shipped" is a point-in-time check; the bond-evolution-quality is judged longitudinally over weeks of dogfood.)

## Reference

- Test scenarios (user-facing list of what to test): [`./test-scenarios.md`](./test-scenarios.md)
- Sessions in scope: [`../sessions/06-memory-foundation.md`](../sessions/06-memory-foundation.md), [`../sessions/07-meta-agent-core.md`](../sessions/07-meta-agent-core.md), [`../sessions/08-meta-agent-capabilities.md`](../sessions/08-meta-agent-capabilities.md)
- Per-session frozen acceptance criteria: `../sessions/<NN>-<slug>/acceptance.md`
- Bundle 01's testing pattern: [`../../01-visual-builder/testing/test-session.md`](../../01-visual-builder/testing/test-session.md)
- Bundle 02's testing pattern: [`../../02-run-execution-and-inspection/testing/test-session.md`](../../02-run-execution-and-inspection/testing/test-session.md)
- Cross-bundle dependency: Bundle 01 ships the workflows Bundle 03's meta-agent dispatches; Bundle 02 ships the run lifecycle + trace endpoints the meta-agent's read tools wrap.
- Framework dependency: Framework Session 06 (AG-UI translator) — already a Bundle 02 dependency; nothing new for Bundle 03.
- Architecture: [`../../../../architecture/spren/09-meta-agent.md`](../../../../architecture/spren/09-meta-agent.md), [`../../../../architecture/spren/10-memory-architecture.md`](../../../../architecture/spren/10-memory-architecture.md), [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md).
- Bond + voice spec: [`../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md`](../../../../../tmp/spren/research/06-memory-foundations/02-spren-soul-and-bond.md), [`../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md`](../../../../../tmp/spren/research/06-memory-foundations/05-five-archetypes.md).

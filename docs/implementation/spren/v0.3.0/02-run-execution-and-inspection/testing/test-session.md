# Bundle 02 (Run Execution and Inspection) — Test Session

> Status: **stub** until both Session 04 and Session 05 ship. Fleshed out fully on the first bundle-end testing pass. Target audience: a Claude Code agent with `mcp__claude-in-chrome__*` + Playwright + tauri-driver + httpx access. Runs the scenarios in [`./test-scenarios.md`](./test-scenarios.md) across the four testing layers and produces a result report.

## Goal

Execute the four-layer test suite for Bundle 02 (Sessions 04 + 05 → "user runs a workflow live + inspects the trace + filters run history + attaches files"):

1. **Scripted integration** — Playwright + tauri-driver + httpx, deterministic. Runs in CI via `just bundle-02`.
2. **Agent-driven exploratory** — Read [`./test-scenarios.md`](./test-scenarios.md), run through every G-* / E-* / X-* / U-* scenario plus self-directed variations. Produce `./test-runs/<ISO-date>-exploratory.md`.
3. **Visual regression** — Playwright screenshot snapshots extending Bundle 01's baseline set. Bundle 02 adds the run-execution + inspector screens.
4. **User manual smoke** — short checklist (U-08..U-13) on a fresh data dir + fresh install.

## What the testing agent does

(Filled in when this stub is promoted. Should include: harness setup, environment preconditions including `OPENROUTER_API_KEY` for live LLM calls, scenario execution order, result-report shape, escalation rules for ambiguous failures, GIF-recording conventions for run-execution evidence, cost-budget guard so the agent doesn't burn tokens looping on a flaky scenario.)

## What the user does after the agent finishes

(Filled in when this stub is promoted. Should include: how to read the agent's report, which findings warrant a fix vs. a follow-up session, how to record the manual smoke results, when Bundle 02 is considered shipped.)

## Reference

- Test scenarios (user-facing list of what to test): [`./test-scenarios.md`](./test-scenarios.md)
- Sessions in scope: [`../sessions/04-run-execution.md`](../sessions/04-run-execution.md), [`../sessions/05-run-inspection.md`](../sessions/05-run-inspection.md)
- Per-session frozen acceptance criteria: `../sessions/<NN>-<slug>/acceptance.md` (extracted by `acceptance-criteria-extractor` on each session's first implementation turn)
- Bundle 01's testing pattern (mirror this one's structure once promoted): [`../../01-visual-builder/testing/test-session.md`](../../01-visual-builder/testing/test-session.md)
- Cross-bundle dependency: Bundle 01 ships the workflows + design system + canvas Run button shell that Bundle 02 builds on
- Framework dependency: Framework Session 06 (AG-UI translator) must be merged + released before Bundle 02 implementation begins

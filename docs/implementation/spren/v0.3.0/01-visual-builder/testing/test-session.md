# Bundle 01 (Visual Builder) — Test Session

> Status: **stub**. Filled in fully when Bundle 01 testing begins. Target audience: a Claude Code agent with `mcp__claude-in-chrome__*` + Playwright + tauri-driver + httpx access. Runs the scenarios in [`./test-scenarios.md`](./test-scenarios.md) across the four testing layers and produces a result report.

## Goal

Execute the four-layer test suite for Bundle 01 (Sessions 01 + 02 + 03 → "user installs umbrella + builds workflow visually"):

1. **Scripted integration** — Playwright + tauri-driver + httpx, deterministic. Runs in CI.
2. **Agent-driven exploratory** — Read [`./test-scenarios.md`](./test-scenarios.md), run through every G-* / E-* / X-* / U-* scenario plus self-directed variations. Produce `./test-runs/<ISO-date>-exploratory.md`.
3. **Visual regression** — Argos snapshots of the key screens. Bundle 01 captures the first baseline.
4. **User manual smoke** — short checklist on a fresh data dir + fresh install.

## What the testing agent does

(Filled in when this stub is promoted. Should include: harness setup, environment preconditions, scenario execution order, result-report shape, escalation rules for ambiguous failures, GIF-recording conventions for visual evidence.)

## What the user does after the agent finishes

(Filled in when this stub is promoted. Should include: how to read the agent's report, which findings warrant a fix vs. a follow-up session, how to record the manual smoke results.)

## Reference

- Test scenarios (user-facing list of what to test): [`./test-scenarios.md`](./test-scenarios.md)
- Sessions in scope: [`../sessions/01-foundation.md`](../sessions/01-foundation.md), [`../sessions/02-workflow-crud-types.md`](../sessions/02-workflow-crud-types.md), [`../sessions/03-visual-builder.md`](../sessions/03-visual-builder.md)
- Bundle's frozen acceptance criteria per session: `../sessions/<NN>-<slug>/acceptance.md`

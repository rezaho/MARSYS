# Architecture Index

This directory holds architecture documentation for the **marsys Python framework** (`src/marsys/`).

| Subdir | What | Audience | Status |
|--------|------|----------|--------|
| [`framework/`](./framework/) | The marsys framework — execution flow, design principles, component registry, ADRs | Framework contributors and AI agents working on framework code | Stable; load-bearing for all agent work |

## What lives in `framework/`

- `overview.md` — execution flow, module boundaries, central concepts
- `design-principles.md` — DP-001 through DP-007 (cross-cutting invariants)
- `components/*.yaml` — component registry (one YAML per TRUNK-CRITICAL or TRUNK-STABLE component)
- `decisions/ADR-*.md` — architecture decision records

## Status of these docs in source control

These docs are currently **not committed** (`docs/architecture/` is gitignored — see `.gitignore`). They are local-only working docs for the project team and AI agents. Public user documentation lives at `docs/api/`, `docs/concepts/`, `docs/guides/`, etc., which are tracked.

## Pointers

- AI-agent workflow rules: [`/CLAUDE.md`](../../CLAUDE.md)
- Framework public docs (committed): [`docs/api/`](../api/), [`docs/concepts/`](../concepts/)

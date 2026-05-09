# Spren — Architecture

Architecture docs for the **open-source local-first personal AI assistant** being built on top of the marsys Python framework.

## Reading order

For new contributors and AI agents joining spren work, read in this order:

1. [`00-overview.md`](./00-overview.md) — what the app is, who it's for, why it exists
2. [`01-system-context.md`](./01-system-context.md) — how it relates to the marsys framework, MARSYS Cloud, MARSYS Studio
3. [`08-design-principles.md`](./08-design-principles.md) — cross-cutting invariants (load-bearing — read before any code change)
4. Then jump into the area you're working on:
   - Working on storage / schemas → [`02-data-model.md`](./02-data-model.md)
   - Working on the HTTP/SSE API → [`03-api-design.md`](./03-api-design.md)
   - Working on the GUI / TUI / Tauri shell → [`04-frontend-architecture.md`](./04-frontend-architecture.md)
   - Working on installer / packaging → [`05-packaging-distribution.md`](./05-packaging-distribution.md)
   - Working on traces / cost / metrics → [`06-observability.md`](./06-observability.md)
   - Working on auth / channels / secrets → [`07-security.md`](./07-security.md)

## What this is NOT

- This is **not** documentation for the marsys framework itself (that's [`../framework/`](../framework/))
- This is **not** documentation for proprietary MARSYS Studio or Cloud (different repos, different teams, different product)
- This is **not** user-facing docs (those will live under `docs/concepts/` or `docs/guides/` once we ship)

## Status of these docs in source control

These docs are **tracked in this repo's git history** (not gitignored). The Spren repo is private during early design; tracking them gives a fresh contributor or AI agent the full design state on clone. When we ship and add public-facing user docs, we may revisit this layout.

## Implementation roadmap

See [`docs/implementation/`](../../implementation/spren/) for the version-by-version plan and per-session briefs.

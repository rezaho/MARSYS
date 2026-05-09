# Spren — Implementation Roadmap

Plans for building the open-source Spren app in this repo.

## Reading order

1. [`00-overview.md`](./00-overview.md) — version map, dependency graph, milestone summary
2. [`v0.3-mvp.md`](./v0.3-mvp.md) — detailed v0.3 plan, session-by-session
3. [`v0.4-extensions.md`](./v0.4-extensions.md) — detailed v0.4 plan (TUI, Tauri polish, always-on org & channels)
4. [`v0.5-future.md`](./v0.5-future.md) — looser v0.5 outline
5. [`sessions/`](./sessions/) — per-session implementation briefs

## How sessions work

Implementation is sliced into **sessions**. Each session is a self-contained brief that an implementer agent (in a fresh session, with no memory of prior conversations) can execute end-to-end.

Each session must:
- Produce a working, testable artifact at the end (no half-finished features)
- Build on prior sessions in dependency order
- Follow the contract in [`sessions/README.md`](./sessions/README.md)
- Use the structure in [`sessions/_template.md`](./sessions/_template.md)

## Where this connects to other docs

- Architecture (what we're building): [`docs/architecture/`](../../architecture/spren/)
- Project rules (working agreement, criticality map, principles): [`/CLAUDE.md`](../../../CLAUDE.md)
- Spren design principles: [`docs/architecture/spren/08-design-principles.md`](../../architecture/spren/08-design-principles.md)
- Framework features Spren depends on (per-version plans + per-session framework PRs): [`docs/implementation/framework/`](../framework/)

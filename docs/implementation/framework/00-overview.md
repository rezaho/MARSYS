# 00 — Framework Implementation Overview (Spren-Driven Backlog)

## Principles

1. **Framework knows nothing of Spren** (SP-018). Every feature here is multi-consumer-justifiable. The framework gains a generic capability; Spren is one consumer; LangSmith / Phoenix / OpenLLMetry / MARSYS Cloud / MARSYS Studio / third-party Python users are equally valid consumers.
2. **No TRUNK-CRITICAL refactors driven by Spren.** Items here extend rather than replace existing framework code. Anything that would require touching `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, or `TopologyGraph` in a non-additive way is escalated for framework-team architectural review before any work begins.
3. **Each session = one PR.** Small, scoped, reviewable. Land on the framework's main via the framework's normal contribution flow.
4. **Spren consumes after merge.** Spren-side sessions that depend on a framework feature reference the merged framework version; they don't bundle framework changes into Spren-branch commits.
5. **The seam stays small** (SP-018). The framework's public surface to Spren is exactly: `Orchestra.run()`, `EventBus.subscribe()`, `TelemetrySink` (added in v0.4-support). Don't grow more doors casually.

## Spren ↔ framework version alignment

| Spren version | Framework features required (must be merged + released before Spren ships) |
|---|---|
| **v0.3 (foundation)** | NDJSON streaming tracing writer ([`v0.3-spren-support.md`](./v0.3-spren-support.md)) |
| **v0.4 (TUI + Tauri polish + always-on org)** | `TelemetrySink` protocol; pause/resume completion in `state_manager`; workflow definition serializer; advanced semantic linter ([`v0.4-spren-support.md`](./v0.4-spren-support.md)) |
| **v0.5+** | TBD; tied to Spren v0.5 scope ([`v0.5-future.md`](./v0.5-future.md)) |

The framework's own version sequence (v0.2.x, v0.3.x, etc.) is set by the framework team. This directory only specifies "must be live in some framework release before Spren v0.X ships" — exact framework version numbering is a coordination question.

## Sessions in this directory

| # | Title | Spren version requiring it | Status |
|---|-------|---------------------------|--------|
| 01 | [NDJSON streaming tracing writer](./sessions/v0.3.0/01-ndjson-streaming-tracing-writer.md) | v0.3 | Drafted |
| 02 | [`TelemetrySink` protocol + EventBus integration](./sessions/v0.3.0/02-telemetry-sink-protocol.md) | v0.4 | Drafted |
| 03 | [Pause/resume completion in `state_manager`](./sessions/v0.3.0/03-pause-resume-completion.md) | v0.4 | Drafted |
| 04 | [Workflow definition serializer](./sessions/v0.3.0/04-workflow-serializer.md) | v0.4 | Drafted |
| 05 | [Advanced semantic linter](./sessions/v0.3.0/05-semantic-linter.md) | v0.4 | Drafted |

Sessions are numbered in execution order, NOT in framework-internal-version order. Multiple sessions can land in the same framework release.

## Dependency graph (Spren-driven)

```
                            ┌────────────────────────────────────┐
                            │ Framework current main             │
                            │ (unified-barrier merged)           │
                            └─────────────┬──────────────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
                  ▼                       ▼                       ▼
          01 NDJSON writer         02 TelemetrySink        03 pause/resume
          (Spren v0.3 dep)         (Spren v0.4 dep)        (Spren v0.4 dep)
                  │                       │                       │
                  │                       │                       │
                  ▼                       ▼                       ▼
                                  04 workflow serializer  05 semantic linter
                                  (Spren v0.4 dep)        (Spren v0.4 dep)
```

Sessions 01–05 are independent of each other within the framework; they can land in any order.

## Working rules for framework sessions

Each session brief follows the same peer-collaboration norms as Spren sessions (see [`sessions/_template.md`](./sessions/_template.md)). The implementer:

1. Reads the framework's own working rules (the framework worktree's `CLAUDE.md`)
2. Reads the framework architecture docs (in the framework worktree)
3. Reads this session brief
4. Implements within scope
5. Writes framework-style tests (unit + integration; framework regression suite must pass)
6. Lands a single PR on the framework's main branch
7. Reports back so Spren-side planning can mark the dependency as satisfied

## When framework features should NOT come from this dir

Features that the framework needs for its own development (refactors, internal cleanup, new orchestration patterns) are owned by the framework team and tracked in framework-internal docs. This dir is exclusively for features that exist because Spren needs them — even if the implementation has multi-consumer value, the motivation is Spren-driven and the planning happens here.

## Cross-references

- Spren plan: [`../spren/00-overview.md`](../spren/00-overview.md)
- Spren architecture (consumer of framework features): [`../../architecture/spren/`](../../architecture/spren/)
- Spren design principles SP-018 (framework knows nothing of Spren) and SP-019 (API single source of truth): [`../../architecture/spren/08-design-principles.md`](../../architecture/spren/08-design-principles.md)

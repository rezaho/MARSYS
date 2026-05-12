# Framework Session 06: AG-UI Event Stream Translator

**Status: scoped (stub). Full brief written during the framework's own architect-session flow when this session is prioritized.**

Required by Spren v0.3 Session 04 (Run execution + tracing consumption). The Spren-side run inspector wraps the adapter shipped here in an SSE HTTP endpoint. Other consumers (MARSYS Cloud, MARSYS Studio, third-party hosted dashboards, custom UIs) consume the same adapter for live event streaming.

---

## Multi-consumer justification (mandatory)

Every framework user who needs to stream live agent events to a UI benefits from a generic AG-UI adapter. AG-UI is an open protocol (https://docs.ag-ui.com) developed by CopilotKit and others as a portable wire format for streaming agent activity. Spren is one consumer; the adapter is intentionally shaped to fit any AG-UI-speaking client one-to-one. Third parties can build a UI on the framework without re-implementing event translation.

**Forbidden:** any code path special-cased for Spren, or any AG-UI extension that only Spren consumes. The framework knows about the AG-UI standard; it knows nothing about specific consumers.

---

## The big picture — what we're building and why

### What this PR ships

A new framework adapter at `marsys.transport.aggui` (or equivalent namespace — first open question below) that:

1. Defines Pydantic models for each AG-UI event type (`RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR`, `TEXT_MESSAGE_START` / `CONTENT` / `END`, `TOOL_CALL_START` / `ARGS` / `END` / `RESULT`, `STATE_SNAPSHOT`, `STATE_DELTA`, `CUSTOM`, etc.) pinned to a specific AG-UI version.
2. Maps `EventBus` events 1:1 (or near-1:1) to AG-UI events. The mapping table is the load-bearing artifact of this session.
3. Exposes `AGUIEventStream(orchestra, run_id) -> AsyncIterator[AGUIEvent]` as the consumer-facing interface.
4. Ships `schema_version: int = 1` on every AG-UI event so consumers can detect protocol drift (matches the `TelemetrySink` pattern from Session 02).
5. Tests: unit (event mapping coverage) + integration (a real 3-agent workflow run → assert every emitted AG-UI event is well-formed and matches the AG-UI spec).

### Why this is framework, not Spren

1. **AG-UI is a standard, not a Spren invention.** Framework support enables any UI to consume framework runs without re-implementing translation.
2. **Translation requires deep framework knowledge.** Event types, branch IDs, span structures, tool-call lifecycle, token streams — all framework-internal. Better to centralize once than re-import into every consumer.
3. **Aligns with the `TelemetrySink` precedent (Session 02).** Framework defines the shape + ships a default adapter; ecosystem consumes. Spren becomes one consumer of a generic seam.
4. **MARSYS Cloud + Studio will consume too** (planned downstream products). If AG-UI lives in Spren, they re-implement or weirdly depend on Spren's translator package. If it lives in the framework, they consume the same hook.
5. **Doesn't violate SP-018** (framework knows nothing of Spren). The framework gains awareness of AG-UI (an external open standard), not of Spren — same as how the framework's `TelemetrySink` is aware of the "span-shaped sink" concept across LangSmith/Phoenix without knowing about Spren.

---

## Open questions for the framework architect

These need explicit answers before implementation. Surface in the full brief during architect-session.

1. **Package location.** `marsys.coordination.aggui` (matches `coordination/telemetry/` and `coordination/tracing/`) vs `marsys.adapters.aggui` (new top-level adapters namespace) vs `marsys.transport.aggui` (new transport namespace). The first matches existing structure but conflates "coordination" with "transport." The second/third introduce a new namespace but signal the abstraction layer more clearly.

2. **Subscribe to `EventBus` directly, or to `TelemetrySink` spans?** `EventBus` gives raw events at native granularity (every token, every tool call, every branch lifecycle); AG-UI is event-oriented at similar granularity, so direct subscription is a clean match. `TelemetrySink` gives closed spans only — coarser, post-hoc. AG-UI's `TEXT_MESSAGE_START` / `CONTENT` / `END` triplet specifically needs streaming granularity, so `EventBus` is the right level. Confirm with first-principles in the full brief.

3. **AG-UI version pin.** Target which version of the AG-UI spec? As of writing, AG-UI is at v0.x. Pick a specific tag (e.g., v0.5.0), document the supported feature set, and document the upgrade path for when AG-UI ships a breaking change. Include this version in `AGUIEvent.schema_version` or in a sibling field.

4. **Custom marsys events.** How does the translator handle marsys-specific events that don't map cleanly to AG-UI (e.g., barrier lifecycle, rendezvous detection, branch spawning)? Two options: (a) use AG-UI's `CUSTOM` event with a structured `type` discriminator (`marsys.barrier.opened`, etc.), (b) define a framework-extension namespace within AG-UI and ship a JSON Schema for those extensions. (a) is simpler; (b) is more rigorous if multiple framework users adopt it.

5. **Backpressure.** What happens if the consumer is slow? Buffer with bounded size + drop oldest? Fail fast? Block the producer? `TelemetrySink` already adopts async-batched at the sink; AG-UI is real-time so the semantics matter more. Recommend: bounded async queue per `AGUIEventStream` (default ~1000 events), drop-oldest on overflow with a `STREAM_LAGGED` event emitted to the consumer so it knows.

6. **SSE transport.** Does the framework provide a default SSE encoder (`AGUIEvent` → SSE wire format string), or is SSE encoding a consumer-side concern? Recommendation: framework provides a `aggui_event_to_sse(event) -> str` helper but does not ship an HTTP server. Consumers (Spren's FastAPI endpoint, Cloud's HTTP server, etc.) wire HTTP themselves. Keeps the framework HTTP-agnostic.

7. **`STATE_SNAPSHOT` / `STATE_DELTA` semantics.** AG-UI defines state events for clients that want to track workflow-level state (current selected agent, conversation history, etc.). What state does marsys emit at this level? Probably: the workflow's current branch set + per-branch active agent + step count. Define the state schema explicitly in the brief.

8. **Multi-run streams.** Does a single `AGUIEventStream` handle one `run_id` (point query) or can it filter across multiple runs (subscribe pattern)? v0.3 only needs the one-run-at-a-time pattern (Spren's run inspector watches one run). The brief should explicitly NOT design for multi-run streams; that's a later session if needed.

---

## High-level scope

In scope:
- Pydantic models for AG-UI event types (pinned version)
- `EventBus → AGUIEvent` mapping logic with a complete table documenting every mapping
- `AGUIEventStream(orchestra, run_id) -> AsyncIterator[AGUIEvent]` interface
- Backpressure handling (bounded queue + drop-oldest + `STREAM_LAGGED` notice)
- `aggui_event_to_sse(event) -> str` helper for consumers that want SSE
- Unit tests (every mapping covered) + integration test (real 3-agent run, every emitted event well-formed)
- AG-UI spec compliance test (validate against the official AG-UI JSON Schema if published)
- Framework regression suite green

Out of scope:
- HTTP server (consumer-side concern)
- AG-UI client (consumer-side concern)
- Multi-run streams (one-run-at-a-time is the only v0.3 need)
- Persistence of AG-UI events (the `TelemetrySink` adapter already handles span-shaped persistence)
- AG-UI v0.x → v1.x migration tooling (handle when AG-UI v1 ships)

---

## Acceptance

- [ ] Every AG-UI event type from the pinned version has a Pydantic model in `marsys.transport.aggui.events` (or equivalent path).
- [ ] `EventBus` event → AG-UI event mapping is exhaustive: the test suite enumerates every framework event type and asserts a corresponding AG-UI event is emitted (or that the framework event is documented as intentionally unmapped, e.g., internal-only events).
- [ ] `AGUIEventStream(orchestra, run_id)` yields events in correct order, with `RUN_STARTED` first and `RUN_FINISHED` / `RUN_ERROR` last.
- [ ] Backpressure: a slow consumer scenario (sleep 100ms per event) on a fast run does not deadlock; either the consumer keeps up or `STREAM_LAGGED` is emitted with the drop count.
- [ ] `schema_version: int = 1` on every event; consumers can detect mismatch.
- [ ] Framework regression test suite passes with zero new failures.
- [ ] Integration test: a 3-agent research workflow (User → Researcher → Writer) emits a complete, valid AG-UI event sequence consumable by an AG-UI-compatible client.

---

## Cross-references

- Consumer: Spren v0.3 Session 04 — `docs/implementation/spren/v0.3.0/02-run-execution-and-inspection/sessions/04-run-execution.md` (to be written)
- Related framework pattern (multi-consumer hooks): [`./02-telemetry-sink-protocol.md`](./02-telemetry-sink-protocol.md)
- Framework Spren-support summary: [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md)
- AG-UI specification: https://docs.ag-ui.com
- Spren architecture: [`../../../../architecture/spren/03-api-design.md`](../../../../architecture/spren/03-api-design.md), [`../../../../architecture/spren/06-observability.md`](../../../../architecture/spren/06-observability.md)
- Spren design principles: [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-004 (AG-UI as wire schema), SP-018 (framework purity)

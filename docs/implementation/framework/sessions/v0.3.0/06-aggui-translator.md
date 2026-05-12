# Framework Session 06: AG-UI Event Stream Translator

**Status: full brief, validated 2026-05-12.** Stub expanded 2026-05-12 during architect-session; revised post-validator + post-improver with implementer + user before implementation. Open questions resolved (see §"Open questions" — now §"Resolved decisions").

Ships a framework-internal adapter that subscribes to `EventBus` and emits AG-UI protocol events as an async iterator. Consumer: any UI that speaks AG-UI. Primary v0.3 consumer is Spren's `/v1/runs/{id}/events` SSE endpoint (Spren v0.3 Session 04 — plan not yet written; this brief is the contract that defines what Spren will consume).

---

## Working rules — how we collaborate (READ FIRST)

You are a peer on this project. You are NOT an order-taker. You share equal voice and equal responsibility for the success of the marsys framework.

### Be a peer with equal voice

- **Push back when you disagree.** If this brief is wrong, or if it cites an API that no longer exists, or if the chosen namespace will cause us pain later, say so. Defend with evidence.
- **Stay engaged.** Comment in this session file as you go; flag concerns before they become problems.
- **Be proactive.** If you see something this session is missing, raise it. If an assumption embedded in this brief doesn't hold against the actual EventBus event taxonomy, push back loudly.

### Take responsibility

- **Ownership is shared.** If something fails, it's our shared failure.
- **You own correctness.** Manually verify the mapping against a real run, not just unit tests.
- **You own follow-through.** Update this file's "What was actually built". Update [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) to add Session 06 to the table.

### Double-check before any decision

- **Read the framework code before writing it.** The event taxonomy in §"State at start of this session" was verified by an Explore agent at brief-write time, but commits land daily — re-grep before relying on a class name.
- **Verify the AG-UI SDK version.** Brief pins `ag-ui-protocol==0.1.18` (April 2026). If a newer pre-1.0 release exists when you implement, decide whether to bump and re-verify the event names; AG-UI is pre-1.0 and may have moved.
- **Run framework tests after every meaningful change**, not just at end.
- **Use git commits as checkpoints.**

### Critically assess the plan itself

This brief depends on three external surfaces that can drift:

1. The current framework event taxonomy (re-grep `class.*Event` in `coordination/status/events.py` and `coordination/tracing/events.py` before mapping).
2. The current `ag-ui-protocol` Python SDK shape (re-verify `EventType` enum members against [`ag_ui.core`](https://docs.ag-ui.com/sdk/python/core/overview) on PyPI).
3. The current `Orchestra.execute` finalize block — sinks are now in `TraceCollector.sinks` (Session 02 ship); the AGGUI translator is NOT a sink (it subscribes to `EventBus` directly) but `Orchestra` still owns the registration lifecycle.

If any cited symbol or event has drifted, **stop and update the brief or escalate** before coding.

### Multi-consumer justification (mandatory)

Every framework user who needs to stream live agent events to an AG-UI-speaking client benefits from this translator. AG-UI is an open protocol (https://docs.ag-ui.com, developed by CopilotKit and adopted by LangGraph / CrewAI / Pydantic AI / Bedrock AgentCore as of 2026) — a portable wire format for streaming agent activity. Spren is the first consumer. Subsequent consumers (MARSYS Cloud, MARSYS Studio, third-party hosted dashboards, custom UIs in Python or JS) consume the same translator with zero changes.

**Forbidden:** any code path special-cased for Spren, or any AG-UI extension that only Spren consumes. The framework knows about the AG-UI standard; it knows nothing about specific consumers.

### Foundational project rules

- Framework `CLAUDE.md` — TRUNK-CRITICAL list; design principles DP-001..DP-007
- Framework architecture overview at `docs/architecture/framework/overview.md`
- Spren design principles at `docs/architecture/spren/08-design-principles.md` — SP-004 (AG-UI as wire schema) and SP-018 (framework purity) are the load-bearing invariants for this session
- Spren API design at `docs/architecture/spren/03-api-design.md` — the SSE endpoint at `/v1/runs/{id}/events` Spren wraps around this translator; the AG-UI version caveat at line 130 (pre-1.0, may force fallback)
- Spren observability at `docs/architecture/spren/06-observability.md` — the trace pipeline diagram still says translator lives in Spren; **this brief moves it to the framework and the observability doc must be updated** (out of scope for this session; tracked as a separate doc-update task)

---

## The big picture — what we're building and why

### What this PR ships

A new framework module at `marsys.coordination.aggui/` (sibling to `coordination/tracing/`) that:

1. Subscribes to `EventBus` directly (parallel subscriber to `TraceCollector`). Receives every framework event at native granularity.
2. Translates each `EventBus` event to one or more AG-UI events using the `ag-ui-protocol` SDK's Pydantic models. The mapping is the load-bearing artifact of this session — see §"Event mapping table" below.
3. Exposes `AGUIEventStream(orchestra, run_id) -> AsyncIterator[BaseEvent]` as the consumer-facing interface. Consumers iterate; the stream yields AG-UI events until `RunFinished` or `RunError`.
4. Ships `aggui_event_to_sse(event) -> str` as a thin convenience wrapper around `ag_ui.encoder.EventEncoder` — saves Spren / Cloud / others from re-importing the encoder.
5. Adds one new framework `EventBus` event — `AssistantMessageEvent` — to surface the assistant's response content (the framework today emits `AgentMessagesPreparedEvent` for input messages but nothing for the model's output text). This unlocks `TEXT_MESSAGE_*` event emission. The new event is multi-consumer-justifiable (every observability consumer wants the output content; LangSmith / Phoenix / Langfuse adapters already infer it from span attributes, but the dedicated event is the clean surface).
6. Handles backpressure via bounded `asyncio.Queue` (default 10000, matches `NDJSONTraceWriter` precedent) with **drop-newest + `Custom("marsys.stream.lagged")`** notification. Drop-oldest would break AG-UI's `TextMessageStart/Content/End` ordering invariants; drop-newest preserves stream coherence for the prefix the consumer already saw.
7. Stream-level handshake: emitted as a leading `Custom("marsys.aggui.handshake")` event BEFORE `RunStarted`, carrying `{schema_version: 1, marsys_version, ag_ui_version}` so consumers can detect protocol drift. **Verified against `ag-ui-protocol==0.1.18`**: `RunStartedEvent.input` is a strongly-typed `RunAgentInput` (designed to echo client-side request shape; requires `thread_id`, `run_id`, `state`, `messages`, `tools`, `context`, `forwarded_props`), not a free-form dict. Hooking the handshake into `input` would require synthesizing those required fields — uglier than a leading Custom. AG-UI's `CustomEvent` is the documented escape hatch; consumers that strictly filter on lifecycle events can either inspect their stream's first event or rely on the schema-version metadata via an out-of-band channel.

### Why this is framework, not Spren

1. **AG-UI is a standard, not a Spren invention.** Framework support enables any UI to consume framework runs without re-implementing translation.
2. **Translation requires deep framework knowledge.** Event types, branch IDs, span structures, tool-call lifecycle, barrier convergence — all framework-internal. Better to centralize once than re-import into every consumer.
3. **Aligns with the `TelemetrySink` precedent (Session 02).** Framework defines the shape + ships the default; ecosystem consumes. Spren becomes one consumer of a generic seam.
4. **MARSYS Cloud + Studio will consume too** (planned downstream products). If AG-UI lives in Spren, they re-implement or weirdly depend on Spren's translator package.
5. **Doesn't violate SP-018** (framework knows nothing of Spren). The framework gains awareness of AG-UI (an external open standard, just like OpenAI tool-calling format), not of Spren. Same precedent as `TelemetrySink`: framework knows about "span-shaped sink" without knowing about LangSmith vs Phoenix vs Spren.

### What's NEW relative to the stub

The stub left eight open questions. This brief answers all of them. Net changes from stub:

| Question | Stub's lean | Brief's answer | Why |
|---|---|---|---|
| 1. Namespace | three options | `marsys.coordination.aggui` | Sibling to `coordination/tracing/`; same shelf as event-driven subsystems. `transport/` introduces a new top-level for one feature (overstructured). |
| 2. EventBus vs TelemetrySink | EventBus | EventBus, confirmed | TelemetrySink only sees closed spans. AG-UI needs per-event granularity. Confirmed against the actual `EventBus` event taxonomy. |
| 3. AG-UI version pin | "v0.5.0?" | `ag-ui-protocol==0.1.18` | Actual current PyPI version. Adopt SDK, don't hand-roll. Saves ~500 LOC of Pydantic model definitions. Stub was outdated — AG-UI Python SDK is at 0.1.x, not 0.5.x. |
| 4. Custom marsys events | (a) or (b) | (a) Custom with `name=marsys.<domain>.<event>` | Simpler; JSON Schema docs ship separately at `docs/architecture/framework/aggui-custom-events.md`. Defer extension namespace to v0.5 if multi-consumer demand surfaces. |
| 5. Backpressure | drop-oldest + STREAM_LAGGED | **drop-newest** + Custom("marsys.stream.lagged") | Drop-oldest breaks Start/Content/End protocol triples; drop-newest preserves prefix coherence. |
| 6. SSE transport | thin helper | thin helper using `ag_ui.encoder.EventEncoder` | Adopt SDK's encoder. Helper wraps it; no custom SSE serialization. |
| 7. STATE_SNAPSHOT schema | "probably branches + agents + steps" | `MarsysRunState` defined inline (see §State events) | Lock the schema; consumers depend on it. |
| 8. Multi-run streams | one-run-only | one-run-only, confirmed | Defer; not needed for v0.3. |

### Scope addition: `AssistantMessageEvent`

The framework currently emits `AgentMessagesPreparedEvent` (input to model) but has no event for the model's output content. The output flows back through `Agent._run()` as a `Message` and lands in branch memory — never crosses the EventBus.

Without this event, the translator can emit `TextMessageStart` and `TextMessageEnd` but `TextMessageContent` would be empty. That breaks the AG-UI consumer contract (clients render incremental content).

Adding `AssistantMessageEvent` is a 10-line emission site in `agents/agents.py` immediately after `model.arun()` returns (mirrors the `AgentMessagesPreparedEvent` pattern at the symmetric position). Multi-consumer benefit: LangSmith / Phoenix / Langfuse adapters can stop inferring assistant content from span attributes and use the dedicated event.

This is in scope for Session 06 because (a) it is the same code area, (b) without it the translator's mapping table has a load-bearing gap, (c) the event is generic — not AG-UI-specific. If the implementer disagrees and wants to defer to a follow-up session, escalate before coding — but the consequence is that v0.3 AG-UI streams ship without assistant content, which makes the translator nearly useless to Spren's run inspector.

### Out of scope (deliberate)

| Excluded | Reason | Where it lands |
|---|---|---|
| Token-level LLM streaming | Framework today calls `model.arun()` which is async-complete. Adding chunked-streaming requires changes to every provider adapter. Big session. | Future framework session (likely v0.4). Until then, `TextMessageContent` carries the full assistant text in a single delta — protocol-compliant but coarser than ideal. |
| HTTP server | Consumer concern. Spren wraps the iterator in a FastAPI endpoint; Cloud will wrap in its own HTTP server. | Consumer side. |
| AG-UI v0.x → v1.x migration tooling | AG-UI is pre-1.0; v1 not on the roadmap as of brief-write time. | Handle when AG-UI tags v1. |
| Multi-run streams | One stream per `run_id`. Spren's run inspector watches one run. | Defer. |
| Persistence of AG-UI events | The `TelemetrySink` sinks already handle span-shaped persistence (Session 02). AG-UI is a wire format, not a storage format. | `TelemetrySink` (existing). |
| AG-UI client | Consumer concern. | Consumer side. |
| `ValidationDecisionEvent` mapping | Event class is defined but has no emission site today (Phase 4 future work). | Map when emission lands. |

---

## What came before this session

Previous framework PRs from this dir:

- **Session 01** — NDJSON streaming tracing writer ([`./01-ndjson-streaming-tracing-writer.md`](./01-ndjson-streaming-tracing-writer.md)). Shipped: `TraceCollector._stream_span` (collector.py:403-417), ULID migration, `Orchestra` finalize with timeout-bounded close.
- **Session 02** — `TelemetrySink` ABC + `SecretRedactor` ([`./02-telemetry-sink-protocol.md`](./02-telemetry-sink-protocol.md)). Shipped: `coordination/tracing/sink.py` (TelemetrySink ABC), `coordination/tracing/redactor.py`, `NDJSONTraceWriter` reclassified to inherit from `TelemetrySink`. `TraceCollector.writers` renamed to `sinks`. This is the multi-consumer pattern Session 06 mirrors.

### State at start of this session (verified at brief-write time — re-verify before coding)

**EventBus and event taxonomy** (`coordination/event_bus.py`, `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`):

Line numbers verified 2026-05-12; nevertheless re-grep before relying on a specific line. Path prefix omitted when unambiguous — paths fully qualified: `coordination/execution/{step_executor,orchestrator}.py`, `coordination/communication/user_node_handler.py`, `agents/{agents,memory}.py`, `coordination/{orchestra,events}.py`, `coordination/status/events.py`, `coordination/tracing/events.py`.

| Event class | Source module | Emission site | Fires when |
|---|---|---|---|
| `ExecutionStartEvent` | tracing/events.py | `coordination/orchestra.py` (in `Orchestra.execute()`) | `Orchestra.execute()` begins |
| `FinalResponseEvent` | status/events.py | `coordination/orchestra.py:840, 1053` | Run completes (success or failure path) |
| `CriticalErrorEvent` | status/events.py | `coordination/execution/step_executor.py:1216, 1363` | Framework errors requiring user action |
| `ErrorEvent` | status/events.py | `agents/agents.py:3029`, `coordination/execution/step_executor.py:176` | Recoverable error caught |
| `ResourceLimitEvent` | status/events.py | `coordination/execution/step_executor.py:1298` | Pool exhausted |
| `AgentStartEvent` | status/events.py | `coordination/execution/step_executor.py:260` | Agent step begins |
| `AgentMessagesPreparedEvent` | status/events.py | `agents/agents.py:3112` | Just before `model.arun()` (input messages) |
| `AgentThinkingEvent` | status/events.py | `coordination/execution/step_executor.py:427` | Model response has thinking/reasoning |
| `AgentCompleteEvent` | status/events.py | `coordination/execution/step_executor.py:686, 838` | Agent step finishes |
| `GenerationEvent` | tracing/events.py | `coordination/execution/step_executor.py:665` | After LLM generation (token counts) |
| `ToolCallEvent` | status/events.py | `coordination/execution/tool_executor.py:202, 300, 349` | Tool lifecycle (started / completed / failed) |
| `BranchCreatedEvent` | coordination/events.py | `coordination/execution/orchestrator.py:511` (def), 623, 1162 (call) | New branch spawned |
| `BranchCompletedEvent` | coordination/events.py | `coordination/execution/orchestrator.py:529` (def), 806, 813, 853, 860, 1216 (call) | Branch completes |
| `BranchEvent` | status/events.py | **NO emission site yet** | Defined but never emitted; TraceCollector subscribes anyway. Disposition: `NOT_YET_EMITTED` bucket (see §Mapping). |
| `ParallelGroupEvent` | status/events.py | `coordination/execution/orchestrator.py:542` (def), 741, 1141 (call) | Parallel fork lifecycle |
| `ConvergenceEvent` | tracing/events.py | `coordination/execution/orchestrator.py:562` (def), 1135 (call) | Multi-arrival barrier fires |
| `UserInteractionEvent` | status/events.py | `coordination/communication/user_node_handler.py:125, 426` | UserNode request (starting) / response (completed). NOTE: line 426 was previously miscredited to `FinalResponseEvent` in an earlier brief draft — verified to be `UserInteractionEvent`. |
| `PlanCreatedEvent` / `Updated` / `ItemAdded` / `ItemRemoved` / `Cleared` | status/events.py | `agents/planning/state.py` | Planning state lifecycle |
| `CompactionEvent` | status/events.py | `agents/memory.py:748` (helper `_emit_compaction_event`) | Memory compaction |
| `MemoryResetEvent` | agents/memory.py | `agents/memory.py:759` | Memory reset. **NOTE: `MemoryResetEvent` does NOT inherit from `StatusEvent`** — fields are `agent_name`, `timestamp` only (no `session_id`/`event_id`/`branch_id`/`metadata`). Reflection-based event discovery must handle this explicitly. |
| `ValidationDecisionEvent` | tracing/events.py | **NO emission site yet** | Disposition: `NOT_YET_EMITTED` bucket. |

**To be added in this session**:

| Event class | Module | Emission site | Fires when |
|---|---|---|---|
| `AssistantMessageEvent` | status/events.py | `agents/agents.py` after `model.arun()` returns | Model returns assistant response (text + optional tool_calls metadata) |

**AG-UI Python SDK** (`ag-ui-protocol==0.1.18`):

- `ag_ui.core` — Pydantic event classes + `EventType` enum. 16+ event types. Field names are camelCase on wire (`messageId`, `delta`). Type discriminator is `type` field with ALL_CAPS values (`TEXT_MESSAGE_CONTENT`, `RUN_STARTED`, etc.).
- `ag_ui.encoder.EventEncoder` — `.encode(event) -> str` returns `data: {json}\n\n` SSE-ready. **Adopt this; do not hand-roll.**
- Caveat: pre-1.0; pin tightly with `==`. The Spren architecture warns at `docs/architecture/spren/03-api-design.md:130` that AG-UI may force a hand-rolled fallback if it churns. Brief doesn't pre-build the fallback; if AG-UI 0.2.x breaks our build, that's a separate session.

**Tracing layer state**:

- `TraceCollector` is at `coordination/tracing/collector.py`. Subscribes to **13 event types** via `_subscribe_to_events` at L94-118 (verified 2026-05-12): `ExecutionStartEvent, GenerationEvent, ValidationDecisionEvent, ConvergenceEvent, AgentStartEvent, AgentMessagesPreparedEvent, AgentCompleteEvent, ToolCallEvent, BranchCreatedEvent, BranchCompletedEvent, BranchEvent, ErrorEvent, FinalResponseEvent`. After Session 02, owns `sinks` (renamed from `writers`). NOT touched by Session 06 — the AGGUI translator is a peer subscriber to EventBus, not a sink consumer. Session 06 adds `AssistantMessageEvent` as the 14th subscription.
- `Orchestra._initialize_components` is at `orchestra.py:340` and calls `_wire_event_bus()` (`orchestra.py:286-338`). **`TraceCollector` construction is in `_wire_event_bus()`, not directly in `_initialize_components()`.** Session 06 adds AGGUI translator construction inside `_wire_event_bus()` in parallel to TraceCollector. This matters: `_wire_event_bus()` is also called from `resume_session` (Session 03's pause/resume), so a resumed run automatically re-wires the AG-UI translator on the fresh EventBus.

**Verify state with**:

```bash
cd /home/rezaho/research_projects/marsys-tracing-work/
source .venv/bin/activate

# Capture baseline before any change
cd packages/framework
pytest tests/ -x --tb=short 2>&1 | tail -3

# Re-verify event class inventory
grep -rn 'class.*Event' src/marsys/coordination/status/events.py src/marsys/coordination/tracing/events.py src/marsys/coordination/events.py

# Re-verify emission sites for events the translator subscribes to
grep -rn 'event_bus.emit\|await.*emit' src/marsys/agents/agents.py src/marsys/coordination/

# Verify AG-UI SDK is reachable + the version is still 0.1.18 (or note new version)
python -c "import importlib.metadata; print(importlib.metadata.version('ag-ui-protocol'))" 2>/dev/null || \
  echo "ag-ui-protocol not yet installed; will be added in this session"
```

If any cited line number has drifted or an event class has been renamed, **stop and update this brief or escalate** before coding.

---

## Event mapping table — the load-bearing artifact

Every framework `EventBus` event must have a documented disposition. "Internal-only" means the translator intentionally drops it (rationale stated). The mapping is exhaustive: tests assert every event class is either mapped or explicitly listed as internal.

### Lifecycle events

| MARSYS event | AG-UI event(s) | Field mapping | Notes |
|---|---|---|---|
| `ExecutionStartEvent` | `Custom("marsys.aggui.handshake")` + `RunStarted` + `StateSnapshot` | Custom: `value={"schema_version": 1, "marsys_version": ..., "ag_ui_version": ...}`. RunStarted: `runId=session_id`, `threadId=session_id` (no `input` — AG-UI's `RunAgentInput` is typed for client-echo, not for our metadata). StateSnapshot: initial near-empty `MarsysRunState`. | Discovered post-validator that `RunStartedEvent.input` is a strongly-typed `RunAgentInput`. Fell back to leading Custom (the brief's original design). Three events in sequence; iterator yields them in order. |
| `FinalResponseEvent` (success=True) | `RunFinished` | `outcome="success"`, `result={"final_response": final_response, "total_duration_ms": total_duration*1000, "total_steps": total_steps}` | Terminal event; iterator yields then closes. |
| `FinalResponseEvent` (success=False) | `RunError` | `message=final_response or "run failed"`, `code="execution_failed"` | Terminal. |
| `CriticalErrorEvent` | `RunError` + then terminal | `message=message`, `code=error_code` | Terminal — close stream after emission. Field names: `CriticalErrorEvent.error_code` exists; verified `coordination/status/events.py:131-143`. |
| `ErrorEvent` | `Custom("marsys.error")` | `value={"agent": agent_name, "error_class": error_class, "message": error_message, "recoverable": recoverable, "retry_count": retry_count}` | Non-terminal — run continues. AG-UI has no "recoverable error" event; Custom keeps it visible without breaking the lifecycle contract. |
| `ResourceLimitEvent` | `Custom("marsys.resource.limit")` | `value={"resource_type": ..., "pool_name": ..., "limit_value": ..., "current_value": ..., "suggestion": ...}` | Non-terminal; system-level constraint signal. Note actual fields are `limit_value`/`current_value` (not `limit`/`current`) per `coordination/status/events.py:174-180`. |

### Step events

| MARSYS event | AG-UI event(s) | Field mapping | Notes |
|---|---|---|---|
| `AgentStartEvent` | `StepStarted` | `stepName=f"{agent_name}#{step_number}"` | Step boundary; `stepName` is the only field AG-UI defines. Step span_id stashed for correlation. |
| `AgentCompleteEvent` | `StepFinished` | `stepName=f"{agent_name}#{step_number}"` | Mirror of StepStarted. |
| `AgentMessagesPreparedEvent` | **internal-only** | — | Raw input messages; not consumer-facing. Used by TraceCollector for content-addressed capture (`d2b600e`). |

### Generation events (the new ones)

| MARSYS event | AG-UI event(s) | Field mapping | Notes |
|---|---|---|---|
| `AssistantMessageEvent` (NEW in this session) | `TextMessageStart` + `TextMessageContent` + `TextMessageEnd` | Start: `messageId=ULID, role="assistant"`. Content: `messageId, delta=content_text`. End: `messageId`. | Triple emitted together because framework lacks token streaming. Future enhancement chunks into multiple Content events. |
| `AgentThinkingEvent` | `ReasoningStart` + `ReasoningMessageContent` + `ReasoningEnd` | Start: `messageId=ULID`. Content: `messageId, delta=thought`. End: `messageId`. | Same triple-as-single-shot pattern. If `thought` is empty, skip the triple. |
| `GenerationEvent` | **metadata attachment** (see notes) | Emit `Custom("marsys.generation.metadata")` with `value={"model": model_name, "provider": provider, "prompt_tokens": ..., "completion_tokens": ..., "reasoning_tokens": ..., "finish_reason": ...}` | Carries cost/latency info that doesn't fit AG-UI's lifecycle events. Custom keeps Spren's cost calculator (`docs/architecture/spren/06-observability.md:99`) fed. |

### Tool call events

| MARSYS event | AG-UI event(s) | Field mapping | Notes |
|---|---|---|---|
| `ToolCallEvent(status="started")` | `ToolCallStart` + `ToolCallArgs` | Start: `toolCallId=ULID, toolCallName=tool_name, parentMessageId=last_assistant_message_id`. Args: `toolCallId, delta=json.dumps(arguments or {})`. | Args as a single delta — same coarse-grained limitation as TextMessageContent. The translator maintains a per-stream `(branch_id, tool_name) → toolCallId` map across the started/completed/failed triple so the same UULID flows through. |
| `ToolCallEvent(status="completed")` | `ToolCallEnd` + `ToolCallResult` | End: `toolCallId`. Result: `messageId=ULID, toolCallId, content=result_summary or "", role="tool"`. | Result content is `result_summary` (already-summarized for trace). Full content lives in the trace. |
| `ToolCallEvent(status="failed")` | `ToolCallEnd` + `ToolCallResult` | End: `toolCallId`. Result: `messageId=ULID, toolCallId, content=result_summary or "tool failed", role="tool"`. | `ToolCallEvent` has NO `error_summary` field (verified `coordination/status/events.py:71-82`). The failed-status emission at `tool_executor.py:349-358` carries no error content; the surrounding `ErrorEvent` (mapped to `Custom("marsys.error")`) is the source of failure detail. The `ToolCallResult.content` falls back to `result_summary` or a generic literal so the AG-UI triple stays well-formed. |

### Branch / orchestration events (marsys-specific → Custom)

| MARSYS event | AG-UI event | Custom name | Value |
|---|---|---|---|
| `BranchCreatedEvent` | `Custom` | `marsys.branch.created` | `{branch_id, branch_name, source_agent, target_agents, trigger_type, parent_branch_id}` |
| `BranchCompletedEvent` | `Custom` | `marsys.branch.completed` | `{branch_id, last_agent, success, total_steps}` |
| `ParallelGroupEvent` | `Custom` | `marsys.parallel.group` | `{group_id, agent_names, status, completed_count, total_count}` |
| `ConvergenceEvent` | `Custom` | `marsys.convergence` | `{parent_branch_id, child_branch_ids, convergence_point, group_id, successful_count, total_count}` |

These are framework-internal lifecycle events with no AG-UI counterpart. AG-UI's `Custom` event is the documented escape hatch (per AG-UI docs + SP-004). Consumers that don't care about branch topology can ignore them; consumers that do (Spren's branch visualizer, Studio's workflow inspector) consume them with a typed schema. The schema lives at `docs/architecture/framework/aggui-custom-events.md` — created in this session.

### User interaction events

| MARSYS event | AG-UI event | Mapping | Notes |
|---|---|---|---|
| `UserInteractionEvent(interaction_type="starting")` | `Custom("marsys.user_interaction.pending")` | `value={"interaction_id": ..., "agent_name": ..., "prompt_summary": ..., "options": ...}` | Spren wires this to a UI prompt; full prompt comes via REST. |
| `UserInteractionEvent(interaction_type="completed")` | `Custom("marsys.user_interaction.resolved")` | `value={"interaction_id": ..., "agent_name": ...}` | |
| `UserInteractionEvent(interaction_type="timeout")` | `Custom("marsys.user_interaction.timeout")` | `value={"interaction_id": ..., "agent_name": ...}` | |

AG-UI's protocol does NOT define a request/response event pair (it's not designed for human-in-the-loop). Custom is the right escape hatch. If AG-UI adds this in a future spec version, we migrate; until then, Custom keeps the contract stable.

### Plan / memory state events

| MARSYS event | AG-UI event(s) | Mapping | Notes |
|---|---|---|---|
| `PlanCreatedEvent` | `StateSnapshot` | `snapshot={..., "plans": {agent_name: {goal, items: [...]}}}` | Full state snapshot on plan create — easier than incremental. |
| `PlanUpdatedEvent` / `PlanItemAddedEvent` / `PlanItemRemovedEvent` / `PlanClearedEvent` | `StateDelta` | `delta=[{"op": "...", "path": "/plans/{agent_name}/items/...", ...}]` | RFC 6902 JSON Patch ops, matching AG-UI's state delta format. |
| `CompactionEvent` | `Custom("marsys.memory.compaction")` | `value={"agent_name": ..., "status": ..., "pre_tokens": ..., "post_tokens": ..., "duration": ...}` | Not workflow state; memory lifecycle. Custom is correct. |
| `MemoryResetEvent` | **internal-only** | — | Framework-internal recovery; not user-visible. |

### Mapping summary

The exhaustive mapping test uses an explicit `EVENT_REGISTRY: set[type] = DISPATCH.keys() | INTERNAL_ONLY | NOT_YET_EMITTED` declared in `coordination/aggui/mapping.py`. The test walks `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, and `agents/memory.py`, discovers every `*Event` class (across multiple base-class lineages — `MemoryResetEvent` does NOT inherit from `StatusEvent`), and asserts each is in `EVENT_REGISTRY`. Reflection over a single base class was rejected — it misses `MemoryResetEvent` and risks missing future events added in a new module.

Three buckets:

- **`DISPATCH`** — events with active mappers. 9 to standard AG-UI events (lifecycle + step + generation + tool + plan state) + 10 to `Custom` (branch / orchestration / error / resource / user interaction / memory compaction).
- **`INTERNAL_ONLY`** — events deliberately dropped. 2 entries: `AgentMessagesPreparedEvent` (raw input messages, used only by TraceCollector for content-addressed capture per commit `d2b600e`), `MemoryResetEvent` (framework-internal recovery; not user-visible).
- **`NOT_YET_EMITTED`** — events defined but never emitted. 2 entries: `ValidationDecisionEvent` (Phase 4 future work), `BranchEvent` (defined in `status/events.py:86` but no emission site found). When emission lands later, the PR author must move the event out of this bucket and into `DISPATCH` or `INTERNAL_ONLY` — the exhaustive test will fail otherwise.

If a new framework event class lands without a mapping disposition, the exhaustive test fails — forcing the change author to update the registry.

---

## State events — `MarsysRunState` schema

`StateSnapshot` fires after `RunStarted` (initial snapshot is empty / minimal). `StateDelta` fires on every significant state change.

The schema is **trimmed to what the translator's actual EventBus subscriptions can derive**. Fields requiring events that don't yet exist on the EventBus (e.g., `arrived_count`, `resolver_branch`, full barrier candidate set) are deferred to a future session where the Orchestrator emits dedicated barrier events. The schema is versioned (`schema_version=1`); the future session bumps to 2 with backward-compat handling.

```python
from pydantic import BaseModel
from typing import Literal


class BranchState(BaseModel):
    branch_id: str
    branch_name: str
    current_agent: str  # updated on each AgentStartEvent for the branch
    status: Literal["RUNNING", "WAITING", "TERMINATED", "FAILED", "ABANDONED"]
    step_count: int
    parent_branch_id: str | None  # carried by BranchCreatedEvent


class BarrierState(BaseModel):
    """Trimmed v0.3 schema — what we can derive from ParallelGroupEvent + ConvergenceEvent.

    Future session adds: arrived_count (during wait window), resolver_branch,
    full candidates set. Those require new framework events.
    """
    barrier_id: str
    status: Literal["OPEN", "FIRED", "CANCELLED"]
    rendezvous_node: str | None
    group_id: str | None         # set for fork barriers (from ParallelGroupEvent.group_id)
    successful_count: int = 0    # populated on fire (from ConvergenceEvent)
    total_count: int = 0         # populated on fire OR from ParallelGroupEvent


class PlanItemState(BaseModel):
    item_id: str
    title: str
    status: Literal["pending", "in_progress", "completed", "abandoned"]


class PlanState(BaseModel):
    agent_name: str
    goal: str | None
    items: list[PlanItemState]


class MarsysRunState(BaseModel):
    schema_version: int = 1
    branches: dict[str, BranchState] = {}
    barriers: dict[str, BarrierState] = {}
    plans: dict[str, PlanState] = {}
    total_steps: int = 0  # incremented on each AgentCompleteEvent
```

State deltas use RFC 6902 JSON Patch operations (AG-UI's documented delta format). The translator maintains the state in-memory per stream; consumers don't need to track state themselves but may — `StateSnapshot` is replayable for catch-up.

**JSON Patch library:** `jsonpatch>=1.33` (RFC 6902, MIT, stdlib-only deps). Shipped under the `aggui` optional-dep group. Hand-rolled patch generation was considered and rejected — RFC 6902 path escaping (`/` and `~` in keys) and `move`/`copy`/`test` ops are fiddly enough to justify the dependency.

---

## High-level scope

In scope:

- `marsys/coordination/aggui/__init__.py` — public exports
- `marsys/coordination/aggui/translator.py` — `AGGUITranslator` (EventBus subscriber) + `AGUIEventStream` (async iterator) + bounded-queue backpressure
- `marsys/coordination/aggui/mapping.py` — event-to-event translation functions (one per source event class)
- `marsys/coordination/aggui/state.py` — `MarsysRunState` Pydantic model + JSON Patch delta computation
- `marsys/coordination/aggui/sse.py` — `aggui_event_to_sse(event) -> str` thin wrapper
- `marsys/coordination/aggui/custom_events.py` — Pydantic models for each `Custom` event's `value` payload (typed; importable by consumers for typecheck)
- `marsys/agents/status/events.py` — add `AssistantMessageEvent` class
- `marsys/agents/agents.py` — emit `AssistantMessageEvent` after `model.arun()` returns (5-10 lines)
- `marsys/coordination/orchestra.py` — wire `AGGUITranslator` construction inside `_wire_event_bus()` (sibling block after `TraceCollector`) when `ExecutionConfig.aggui.enabled` is True (additive; default off; resume-parity)
- `marsys/coordination/config.py` (or wherever `ExecutionConfig` lives) — add `aggui: AGGUIConfig` field with `enabled: bool = False`, `queue_max_size: int = 10000`
- `packages/framework/pyproject.toml` — add `ag-ui-protocol==0.1.18` AND `jsonpatch>=1.33` under `[project.optional-dependencies] aggui` (NOT runtime — keep framework lean for users who don't need AG-UI). `marsys[aggui]` installs both.
- Tests: unit (each mapping function), integration (real 3-agent workflow → assert AG-UI event stream is well-formed and SDK-validates), backpressure (slow consumer doesn't deadlock), exhaustive-mapping (every framework event class has a disposition), doc-drift (generated `aggui-custom-events.md` matches checked-in version)
- `docs/architecture/framework/aggui-custom-events.md` — JSON Schema for every `marsys.*` Custom event. **Auto-generated** from the Pydantic models in `coordination/aggui/custom_events.py` via `scripts/generate_aggui_custom_events_doc.py`. Hand-maintained markdown was considered and rejected — drift between Pydantic models and the doc is inevitable.
- `packages/framework/scripts/generate_aggui_custom_events_doc.py` — walks `CUSTOM_EVENT_REGISTRY`, calls `model.model_json_schema()` for each, writes the markdown. CI invokes this and asserts the checked-in markdown is up to date.
- `CHANGELOG.md` entry

Out of scope:

- HTTP server (consumer-side)
- AG-UI client (consumer-side)
- Multi-run streams (one stream per `run_id`)
- Persistence of AG-UI events (use TelemetrySink)
- Token-level streaming (framework limitation; future session)
- Updating `docs/architecture/spren/06-observability.md` (architecture-doc task tracked separately; doc still says translator is Spren-side and needs to move it to framework)
- Migration tooling for AG-UI v0.x → v1.x

---

## File map

### Files to create

- `packages/framework/src/marsys/coordination/aggui/__init__.py` — public exports: `AGGUITranslator`, `AGUIEventStream`, `aggui_event_to_sse`, `AGGUIConfig`, `MarsysRunState`
- `packages/framework/src/marsys/coordination/aggui/translator.py` — `AGGUITranslator` (subscriber lifecycle) + `AGUIEventStream` (iterator). `AGUIEventStream(translator)` takes the translator directly — no Orchestra parameter. Orchestra is per-run; `run_id` is redundant.
- `packages/framework/src/marsys/coordination/aggui/mapping.py` — `map_event(event) -> Iterable[BaseEvent]` dispatch + `DISPATCH` / `INTERNAL_ONLY` / `NOT_YET_EMITTED` / `EVENT_REGISTRY`
- `packages/framework/src/marsys/coordination/aggui/state.py` — `MarsysRunState` + JSON Patch helpers (uses `jsonpatch>=1.33`)
- `packages/framework/src/marsys/coordination/aggui/sse.py` — `aggui_event_to_sse(event) -> str` (thin wrapper around `ag_ui.encoder.EventEncoder`)
- `packages/framework/src/marsys/coordination/aggui/custom_events.py` — Pydantic models for `value` payloads of every `marsys.*` Custom event + module-level `CUSTOM_EVENT_REGISTRY: dict[str, type[BaseModel]]`. Strict validation on emission (raises on `model_validate` failure).
- `packages/framework/src/marsys/coordination/aggui/config.py` — `AGGUIConfig` dataclass
- `packages/framework/scripts/generate_aggui_custom_events_doc.py` — walks `CUSTOM_EVENT_REGISTRY`, writes `docs/architecture/framework/aggui-custom-events.md` from Pydantic schemas
- `packages/framework/tests/coordination/aggui/__init__.py`
- `packages/framework/tests/coordination/aggui/test_mapping.py` — one test per mapping row
- `packages/framework/tests/coordination/aggui/test_translator.py` — subscriber lifecycle + queue + backpressure
- `packages/framework/tests/coordination/aggui/test_stream.py` — iterator semantics (RunStarted first, terminal events close iterator)
- `packages/framework/tests/coordination/aggui/test_state.py` — `MarsysRunState` + delta computation
- `packages/framework/tests/coordination/aggui/test_integration.py` — real 3-agent run → full event sequence
- `packages/framework/tests/coordination/aggui/test_custom_events.py` — strict validation raises on bad payload
- `packages/framework/tests/coordination/aggui/test_exhaustive_mapping.py` — walks every event module (`coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, `agents/memory.py`), discovers every `*Event` class across multiple base-class lineages (handles `MemoryResetEvent` which is NOT a `StatusEvent` subclass), asserts each is in `EVENT_REGISTRY`
- `packages/framework/tests/coordination/aggui/test_doc_generation.py` — invokes the doc-gen script, diffs against the checked-in `aggui-custom-events.md`, fails on drift
- `docs/architecture/framework/aggui-custom-events.md` — **auto-generated** JSON Schemas for every `marsys.*` Custom event

Every test file in `tests/coordination/aggui/` begins with `pytest.importorskip("ag_ui")` so CI works for contributors who didn't install the optional dep.

### Files to modify

- `packages/framework/src/marsys/coordination/status/events.py` — add `AssistantMessageEvent`
- `packages/framework/src/marsys/agents/agents.py` — emit `AssistantMessageEvent` after `model.arun()` (mirrors `AgentMessagesPreparedEvent` site at `agents.py:3112`)
- `packages/framework/src/marsys/coordination/tracing/collector.py` — add `AssistantMessageEvent` to `_subscribe_to_events` (line 94-118 currently has 13 entries; this is the 14th). New handler `_handle_assistant_message` stores assistant content keyed by `(branch_id, agent_name, step_id)` in the existing message store (content-addressed, mirroring `_handle_agent_messages_prepared` from commit `d2b600e`). Respect `TracingConfig.include_message_content` — if `False`, suppress content.
- `packages/framework/src/marsys/coordination/orchestra.py` — wire `AGGUITranslator` construction inside `_wire_event_bus()` (line 286-338), after the `TraceCollector` block. Gated by `execution_config.aggui.enabled`. **Must be in `_wire_event_bus()` and not directly in `_initialize_components()`** — `_wire_event_bus()` is also called from `resume_session` so the translator gets re-wired on resume.
- `packages/framework/src/marsys/coordination/config.py` — add `aggui: AGGUIConfig = field(default_factory=AGGUIConfig)` to `ExecutionConfig` (line 231-296)
- `packages/framework/pyproject.toml` — add `[project.optional-dependencies] aggui = ["ag-ui-protocol==0.1.18", "jsonpatch>=1.33"]`
- `packages/framework/CHANGELOG.md` — release entry under `[Unreleased]`
- `packages/framework/tests/agents/test_agents.py` — regression for the new `AssistantMessageEvent` emission site
- `packages/framework/tests/coordination/tracing/test_collector.py` — regression for TraceCollector subscribing to `AssistantMessageEvent`
- `docs/implementation/framework/v0.3-spren-support.md` — update Session 06 row from "scoped" to "✅ shipped"

### Files NOT to touch

- TRUNK-CRITICAL: `coordination/execution/orchestrator.py`, `coordination/execution/real_runtime.py`, `coordination/validation/response_validator.py`, `coordination/topology/graph.py`, `coordination/execution/det_nodes.py`, `coordination/orchestra.py` constructor signature (additive to `_wire_event_bus()` body is OK; additive to `_initialize_components()` body is OK; both are non-TRUNK-CRITICAL methods).
- `coordination/tracing/` — AGGUI is a peer subscriber to EventBus, not a sink consumer. No tracing module changes.
- `coordination/event_bus.py` — translator uses existing `subscribe` interface; no EventBus changes.

---

## Detailed plan

### Step 1 — Re-verify the contract surfaces

Before writing code:

```bash
cd /home/rezaho/research_projects/marsys-tracing-work/
source .venv/bin/activate
cd packages/framework

# Baseline test counts
pytest tests/ -x --tb=short 2>&1 | tail -3

# Verify event class inventory against the brief's table
grep -rn 'class.*Event' src/marsys/coordination/status/events.py src/marsys/coordination/tracing/events.py src/marsys/coordination/events.py

# Verify AG-UI SDK version + import paths
pip install 'ag-ui-protocol==0.1.18'
python -c "from ag_ui.core import EventType, RunStartedEvent, TextMessageContentEvent, CustomEvent; print('ok')"
python -c "from ag_ui.encoder import EventEncoder; print('ok')"

# Verify Orchestra's _wire_event_bus and _initialize_components haven't moved
grep -n 'def _wire_event_bus\|def _initialize_components' src/marsys/coordination/orchestra.py
grep -n 'TraceCollector' src/marsys/coordination/orchestra.py
# Verify resume_session still calls _wire_event_bus (Session 03 pattern)
grep -n '_wire_event_bus' src/marsys/coordination/orchestra.py
```

If any check fails, stop and update the brief.

### Step 2 — Add `AssistantMessageEvent` and emit it

Add to `coordination/status/events.py` (alongside `AgentMessagesPreparedEvent`):

```python
@dataclass
class AssistantMessageEvent(StatusEvent):
    """Emitted by Agent._run() immediately after model.arun() returns.

    Carries the assistant's response content (text + optional tool_calls metadata).
    Pairs symmetrically with AgentMessagesPreparedEvent (input → output).
    """
    agent_name: str
    step_number: int
    step_span_id: str
    message_id: str  # ULID
    content: str
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
```

Add emission in `agents/agents.py` immediately after `model.arun()` returns (line ~3129) and BEFORE `Message.from_harmonized_response()` (line ~3132). Mirror the `AgentMessagesPreparedEvent` site at `agents.py:3110-3119`: same gate `if self._step_event_bus is not None and self._step_context`, same context fields (`session_id`, `branch_id`, `step_number`, `step_span_id`). About 10 lines.

**Error path:** the agent's exception handler at `agents.py:3146` emits `ErrorEvent` and returns an error `Message` — `AssistantMessageEvent` does NOT fire on error. The AG-UI mapping must NOT assume a `TextMessageStart/End` triple per step unconditionally; the absence of `AssistantMessageEvent` implicitly means "no assistant content this step." The error is surfaced as `Custom("marsys.error")` via the `ErrorEvent` mapper.

Add to `TraceCollector._subscribe_to_events` in `coordination/tracing/collector.py:94-118` so the trace collector receives the event (currently 13 entries; add this as the 14th). The handler `_handle_assistant_message` stores assistant content keyed by `(branch_id, agent_name, step_id)` in the existing message store — **mirroring the content-addressed input capture pattern landed by commit `d2b600e`** for `_handle_agent_messages_prepared`. Same pattern: hash the content, store body by hash, attach `message_id + content_hash` to the enclosing step span's attributes. **Respect `TracingConfig.include_message_content`** — if `False`, attach `content_hash` only and skip storing the body. Do NOT invent a parallel content-storage path.

### Step 3 — Add the AG-UI translator module

Create `coordination/aggui/` directory.

**`aggui/config.py`**:

```python
from dataclasses import dataclass


@dataclass
class AGGUIConfig:
    enabled: bool = False
    queue_max_size: int = 10000
```

**`aggui/state.py`** — `MarsysRunState` Pydantic model as specified in §State events.

**`aggui/custom_events.py`** — Pydantic models for `marsys.*` Custom values. Importable for typechecking.

**`aggui/mapping.py`** — pure functions, one per event class. Each returns `Iterable[BaseEvent]` (zero or more AG-UI events). The handshake info rides inside `RunStartedEvent.input["_marsys_handshake"]` — NOT a leading Custom.

```python
from typing import Iterable
from ag_ui.core import BaseEvent, RunStartedEvent, ...

def map_execution_start(event: ExecutionStartEvent, ctx: AGGUITranslator) -> Iterable[BaseEvent]:
    yield RunStartedEvent(
        runId=event.session_id,
        threadId=event.session_id,
        input={
            "task_summary": event.task_summary,
            "topology": event.topology_summary,
            "agents": event.agent_names,
            "_marsys_handshake": {
                "schema_version": 1,
                "marsys_version": _marsys_version(),
                "ag_ui_version": _ag_ui_version(),
            },
        },
    )

# Three buckets. Together: DISPATCH.keys() | INTERNAL_ONLY | NOT_YET_EMITTED == EVENT_REGISTRY.
INTERNAL_ONLY: set[type] = {AgentMessagesPreparedEvent, MemoryResetEvent}
NOT_YET_EMITTED: set[type] = {ValidationDecisionEvent, BranchEvent}

DISPATCH: dict[type, Callable] = {
    ExecutionStartEvent: map_execution_start,
    FinalResponseEvent: map_final_response,
    # ... one per event class with an active mapping
}

EVENT_REGISTRY: set[type] = set(DISPATCH.keys()) | INTERNAL_ONLY | NOT_YET_EMITTED
```

**`aggui/translator.py`** — the subscriber + iterator. Lifecycle mirrors `TraceCollector`: subscribe in `__init__`, unsubscribe in `close()`.

```python
class AGGUITranslator:
    """EventBus subscriber that produces AG-UI events into a bounded queue."""

    def __init__(self, event_bus: EventBus, run_id: str, config: AGGUIConfig):
        self.event_bus = event_bus
        self.run_id = run_id
        self.config = config
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_max_size)
        self.state = MarsysRunState()
        self._lagged_count = 0
        self._closed = False
        self._subscribe()

    def _subscribe(self) -> None:
        for event_class in DISPATCH:
            self.event_bus.subscribe(event_class.__name__, self._handle)

    async def _handle(self, event) -> None:
        mapper = DISPATCH[type(event)]
        try:
            for aggui_event in mapper(event, self):
                self._enqueue(aggui_event)
        except Exception:
            logger.exception("AGGUI mapper for %s failed", type(event).__name__)
            # do not propagate — other subscribers (TraceCollector) keep running

    def _enqueue(self, aggui_event: BaseEvent) -> None:
        if self._lagged_count > 0:
            # emit catch-up Custom carrying cumulative drop count BEFORE the next event
            try:
                self.queue.put_nowait(_make_lagged_custom(self._lagged_count))
                self._lagged_count = 0
            except asyncio.QueueFull:
                pass  # still full; keep counting
        try:
            self.queue.put_nowait(aggui_event)
        except asyncio.QueueFull:
            # drop-newest: increment lag counter; preserve prefix coherence
            self._lagged_count += 1

    async def close(self) -> None:
        for event_class in DISPATCH:
            self.event_bus.unsubscribe(event_class.__name__, self._handle)
        self._closed = True


class AGUIEventStream:
    """Async iterator that yields AG-UI events from a translator's queue.

    Takes the translator directly — no Orchestra/run_id indirection.
    Orchestra exposes the translator as a plain attribute (`orchestra.aggui_translator`).
    """

    def __init__(self, translator: AGGUITranslator):
        self.translator = translator

    def __aiter__(self) -> "AGUIEventStream":
        return self

    async def __anext__(self) -> BaseEvent:
        if self.translator._closed and self.translator.queue.empty():
            raise StopAsyncIteration
        return await self.translator.queue.get()
```

**`aggui/sse.py`**:

```python
from ag_ui.encoder import EventEncoder
from ag_ui.core import BaseEvent

_encoder = EventEncoder()

def aggui_event_to_sse(event: BaseEvent) -> str:
    """SSE wire-format for an AG-UI event. Wraps ag_ui.encoder.EventEncoder."""
    return _encoder.encode(event)
```

### Step 4 — Wire translator construction into Orchestra

In `coordination/orchestra.py:_wire_event_bus()` (lines 286-338), after the `TraceCollector` construction block, add (additive — gated by `aggui.enabled`):

```python
if execution_config.aggui.enabled:
    from .aggui.translator import AGGUITranslator
    self.aggui_translator = AGGUITranslator(
        event_bus=self.event_bus,
        run_id=session_id,
        config=execution_config.aggui,
    )
else:
    self.aggui_translator = None
```

**Why `_wire_event_bus()` and not `_initialize_components()` directly:** Session 03 added pause/resume via `resume_session`, which calls `_wire_event_bus()` to attach listeners to the fresh `EventBus`. If the translator is constructed only in `_initialize_components()`, a resumed run produces no AG-UI stream. Putting it in `_wire_event_bus()` ensures the translator is re-wired on resume.

`Orchestra.aggui_translator` is exposed as a **plain attribute** (matching `self.trace_collector`). No `get_aggui_translator(run_id)` method — Orchestra is per-run, so `run_id` is redundant indirection. `AGUIEventStream` takes the translator directly:

```python
orchestra = Orchestra(...)
result_task = asyncio.create_task(orchestra.run(task, topology, ...))

# Spren's FastAPI handler:
async def stream_run(run_id: str):
    translator = orchestra.aggui_translator  # set during _wire_event_bus()
    stream = AGUIEventStream(translator)
    return EventSourceResponse(
        (aggui_event_to_sse(event) async for event in stream)
    )
```

### Step 5 — Tests

Test layout:

```
packages/framework/tests/coordination/aggui/
  __init__.py
  test_mapping.py             # one test per row in §Event mapping table
  test_translator.py          # subscriber lifecycle; queue; backpressure
  test_stream.py              # iterator semantics; terminal close
  test_state.py               # MarsysRunState + JSON Patch deltas
  test_custom_events.py       # strict validation raises on bad payload
  test_integration.py         # real 3-agent run → full event sequence
  test_exhaustive_mapping.py  # every framework event class has a disposition
  test_doc_generation.py      # generated aggui-custom-events.md matches checked-in
```

Every test file begins with `pytest.importorskip("ag_ui")` so the suite still runs cleanly for contributors who haven't installed the optional `aggui` extras.

`test_exhaustive_mapping.py` is the load-bearing regression test. It walks `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, AND `agents/memory.py` (to catch `MemoryResetEvent` which doesn't inherit from `StatusEvent`), finds every `*Event` class across multiple base-class lineages, and asserts each is in `mapping.EVENT_REGISTRY` (i.e. `DISPATCH.keys() | INTERNAL_ONLY | NOT_YET_EMITTED`). New event classes added to the framework without a mapping decision fail the test. When emission lands for a `NOT_YET_EMITTED` event later, the PR author moves it into `DISPATCH` (or explicitly into `INTERNAL_ONLY`); the exhaustive test won't budge but the in-place change is now visible in the diff.

`test_doc_generation.py` runs `scripts/generate_aggui_custom_events_doc.py` and diffs the output against `docs/architecture/framework/aggui-custom-events.md` — fails if drifted. Forces the doc + Pydantic models to stay in sync.

`test_integration.py` runs a 3-agent research workflow (User → Researcher → Writer) end-to-end and asserts the AG-UI event sequence:

- First event is `RunStarted` with `input["_marsys_handshake"]` populated (`{schema_version, marsys_version, ag_ui_version}`)
- Second event is `StateSnapshot` with the initial near-empty `MarsysRunState`
- Final event is `RunFinished` (or `RunError`)
- Every emitted event validates as a real `ag_ui.core.BaseEvent` subclass instance via `BaseEvent.model_validate(event.model_dump())` round-trip
- Every `TextMessageStart` has a matching `TextMessageEnd` (same `messageId`)
- Every `ToolCallStart` has a matching `ToolCallEnd` (same `toolCallId`)
- Subsequent state changes emit `StateDelta` with valid RFC 6902 JSON Patch ops (via `jsonpatch`)
- A `StateDelta` fires when a `branches[*].current_agent` changes (e.g., on `AgentStartEvent`) — surfaces "branch X is now running agent Y"
- SSE round-trip: `aggui_event_to_sse(event)` produces valid `data: {json}\n\n` lines that re-parse via `EventEncoder`

`test_translator.py` covers backpressure:

- Fast producer + slow consumer (`asyncio.sleep(0.1)` per event in consumer) on a 100-event run with `queue_max_size=10` does not deadlock.
- When the queue overflows, `_lagged_count` increments; the next successful put includes a preceding `Custom("marsys.stream.lagged")` event with the cumulative drop count.
- Subscriber failure (mapping raises) does not propagate to other subscribers (TraceCollector keeps running).

### Step 6 — Documentation

`docs/architecture/framework/aggui-custom-events.md` — JSON Schema for every `marsys.*` Custom event (10 schemas as listed in §Event mapping table). Format: one section per Custom name, each with `name`, `description`, and a JSON Schema block. Consumers reading from any language can validate against these.

`CHANGELOG.md` entry under next release:

```
- feat(framework): AG-UI event stream translator at `marsys.coordination.aggui`. Optional dependency via `pip install marsys[aggui]`. New `AssistantMessageEvent` on the EventBus to surface model output. See `docs/implementation/framework/sessions/v0.3.0/06-aggui-translator.md`.
```

### Step 7 — Sign-off

Update this file's "What was actually built" with delta from plan. Update `v0.3-spren-support.md` to add Session 06 row. Note framework release version that ships this feature. Add "Lessons / Surprises" if anything surprised you (especially: AG-UI SDK gotchas, mapping edge cases discovered during real-run integration tests).

---

## Acceptance

- [ ] `marsys.coordination.aggui` package exists with public exports as listed in §File map (`AGGUITranslator`, `AGUIEventStream`, `aggui_event_to_sse`, `AGGUIConfig`, `MarsysRunState`).
- [ ] `AssistantMessageEvent` defined in `coordination/status/events.py` and emitted by `agents.py` after `model.arun()` returns (using the same `_step_event_bus`/`_step_context` gate as `AgentMessagesPreparedEvent`). NOT emitted on the error path. TraceCollector subscribes to it (14th entry); handler stores content via the content-addressed pattern (commit `d2b600e`), respecting `TracingConfig.include_message_content`.
- [ ] `ag-ui-protocol==0.1.18` AND `jsonpatch>=1.33` declared as optional dependencies under `[project.optional-dependencies] aggui`. Framework's mandatory dependency set unchanged.
- [ ] Every framework event class (in `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, AND `agents/memory.py`) has a documented disposition: in `mapping.DISPATCH`, `mapping.INTERNAL_ONLY`, or `mapping.NOT_YET_EMITTED`. Test asserts this exhaustively (new event class added without a disposition fails the test). The exhaustive test handles classes that don't inherit from `StatusEvent` (e.g. `MemoryResetEvent`).
- [ ] `AGUIEventStream(translator)` (takes translator directly — no Orchestra/run_id parameter) yields a valid AG-UI event sequence:
  - [ ] First event is `RunStarted` with `input["_marsys_handshake"] == {"schema_version": 1, "marsys_version": ..., "ag_ui_version": ...}`
  - [ ] Second event is `StateSnapshot` with initial (empty or near-empty) `MarsysRunState`
  - [ ] Terminal event is `RunFinished` or `RunError`; iterator raises `StopAsyncIteration` after queue drains
  - [ ] Every `TextMessage*` triple is well-formed (`Start` → `Content` → `End`, same `messageId`)
  - [ ] Every `ToolCall*` sequence is well-formed (`Start` → `Args` → `End` → `Result`, same `toolCallId`)
- [ ] `Orchestra.aggui_translator` is set inside `_wire_event_bus()` (NOT directly in `_initialize_components()`), so `resume_session` re-wires the translator. Exposed as a plain attribute (no `get_aggui_translator(run_id)` method).
- [ ] `StateSnapshot` fires once after `RunStarted` with initial state. `StateDelta` fires when any branch's `current_agent` changes (on `AgentStartEvent`), when plan/memory state changes, or when barrier state changes (fork creation / convergence fire). All deltas are valid RFC 6902 JSON Patch ops produced via `jsonpatch.make_patch(old.model_dump(), new.model_dump()).patch`.
- [ ] `MarsysRunState.total_steps` is incremented on each `AgentCompleteEvent`.
- [ ] `MarsysRunState.barriers[*]` schema is the trimmed v0.3 set (`barrier_id`, `status`, `rendezvous_node`, `group_id`, `successful_count`, `total_count`). Fields requiring not-yet-emitted events (`arrived_count`, `resolver_branch`) are NOT in the schema for v0.3.
- [ ] Backpressure: slow consumer (100ms sleep per event) on a fast run with `queue_max_size=10` does NOT deadlock. When overflow occurs, `Custom("marsys.stream.lagged")` is emitted on the next successful put, carrying cumulative drop count, and the counter resets. Drop policy is drop-NEWEST (preserves prefix coherence).
- [ ] `aggui_event_to_sse(event)` produces SSE-formatted lines (`data: {json}\n\n`) that re-parse via `ag_ui.encoder.EventEncoder`. Thin wrapper; no custom serialization.
- [ ] `marsys.*` Custom events have Pydantic models in `coordination/aggui/custom_events.py` registered in `CUSTOM_EVENT_REGISTRY`. Strict validation on emission (raises on `model_validate` failure). JSON Schemas in `docs/architecture/framework/aggui-custom-events.md` are **generated** by `scripts/generate_aggui_custom_events_doc.py` and validated against the checked-in version by a CI test.
- [ ] Translator is gated by `ExecutionConfig.aggui.enabled` (default `False`). Existing framework consumers see no behavior change.
- [ ] Framework regression suite green (zero new failures vs. baseline). Document baseline + post-change counts in §"What was actually built."
- [ ] Optional-dependency isolation: with `ag-ui-protocol` + `jsonpatch` uninstalled, the rest of the framework's tests still pass; aggui tests skip cleanly via `pytest.importorskip("ag_ui")`.
- [ ] Integration test: a 3-agent research workflow (User → Researcher → Writer) emits a complete, valid AG-UI event sequence consumable by the `ag-ui-protocol` SDK. Every emitted event passes `BaseEvent.model_validate(event.model_dump())` round-trip.
- [ ] Multi-consumer justification documented in PR description: at minimum Spren, MARSYS Cloud, MARSYS Studio, generic third-party AG-UI client — each shown as a one-paragraph adapter sketch.
- [ ] No Spren type imported in this PR. No `if running under Spren` code paths.
- [ ] CHANGELOG entry under `[Unreleased]`.
- [ ] `docs/implementation/framework/v0.3-spren-support.md` updated — Session 06 row marked ✅ shipped.

---

## Hard rules

### Multi-consumer justification (mandatory)

- [ ] PR description includes a one-paragraph adapter sketch for each of: Spren, MARSYS Cloud, MARSYS Studio, generic third-party AG-UI client. Each shows how the consumer wires the iterator into its transport (Spren → FastAPI SSE; Cloud → its hosted HTTP server; Studio → its WebSocket-to-AG-UI bridge; generic → user's own server). The pattern in each: `stream = AGUIEventStream(orchestra.aggui_translator)`, then `async for event in stream: send(aggui_event_to_sse(event))`.
- [ ] No Spren type imported in this PR.
- [ ] No "if running under Spren" code paths.

### Framework design principles

- DP-001 (pure agent logic): unchanged. Translator observes events; doesn't run agent logic.
- DP-002 (centralized validation): unchanged.
- DP-003 (unified-barrier orchestration): translator subscribes to barrier events but doesn't drive them. No fire-gate changes.
- DP-004 (branch isolation): translator reads events from EventBus; does not mutate branch state.
- DP-005 (topology-driven routing): unchanged.
- DP-006 (adapter pattern): AGGUITranslator IS an adapter — from internal events to an external wire protocol. Same DP-006 spirit as the model adapter pattern.
- DP-007 (format pluggability): AG-UI is one wire format. The framework can ship additional translators later (`coordination/protocol_X/`) without touching this one.

If this feature would force a violation of any DP, escalate before coding.

### No TRUNK-CRITICAL behavioral changes

`Orchestra.__init__` and `Orchestra.run` signatures unchanged. AGGUI is gated by `ExecutionConfig.aggui`. `_wire_event_bus()` body gets an additive block after `TraceCollector` construction (both `__init__` and `resume_session` re-wire correctly).

If implementation requires a non-additive change to `Orchestra`, `Orchestrator`, `RealRuntime`, `ValidationProcessor`, `TopologyGraph`, or `det_nodes.py`, **stop and escalate via `AskUserQuestion`** before any edit.

### Clean code rules (CLAUDE.md SWE anti-patterns)

- Smallest implementation that passes acceptance criteria.
- No backward-compat shims. There is no v0 → v1 AG-UI migration shipped in this session.
- No variant filenames (`_v2`, `_new`, etc.).
- One concern per file: config, state, mapping, translator, sse, custom events, tests by concern.
- No descriptive comments for self-naming code — only WHY when not obvious.
- TODO comments are forbidden — if you can't fix it now, surface as an open question and decide before merging.

### Anti-patterns to defend against (from CLAUDE.md)

- **Silent feature dropping**: if a mapping in §Event mapping table can't be implemented as specified, raise it explicitly — don't quietly drop it from the dispatch table.
- **Legacy retention**: do NOT keep the AG-UI translator at `src/spren/events.py` after this lands. Spren's observability doc currently points there; that doc must update to point at the framework. Spren-side translator file must NOT exist post-merge (Spren is a consumer of this module).
- **Speculative concrete code**: do NOT add custom events that aren't in §Event mapping table just because they "might be useful." A new Custom name requires a new framework event in active use.

---

## Tests (required for "done")

### Unit tests

`test_mapping.py` — one test per row in §Event mapping table. Each test constructs a fixture framework event, calls the mapper, and asserts the AG-UI output:
- Correct event type (e.g., `RunStartedEvent`)
- Correct field values
- For Custom events: correct `name` and `value` schema

`test_state.py`:
- `MarsysRunState` initializes empty.
- Updating a branch produces a JSON Patch delta with one `replace` op.
- Adding a plan produces a JSON Patch delta with one `add` op.
- Removing a plan item produces a JSON Patch delta with one `remove` op.
- Delta + snapshot apply cleanly back to the same snapshot (round-trip).

### Integration tests

`test_translator.py`:
- `AGGUITranslator` subscribes to every event class in `DISPATCH` at construction.
- `AGGUITranslator` unsubscribes on `close()`.
- Mapping that raises → exception logged, other mappers continue.
- Bounded queue: `queue_max_size=10`, 100-event burst → drop-newest, lagged count tracked, next successful put yields a `Custom("marsys.stream.lagged")` event with cumulative count.

`test_stream.py`:
- `AGUIEventStream(orchestra, run_id)` returns an `AsyncIterator`.
- Iterator yields events FIFO from the queue.
- After `RunFinished` is yielded AND the queue is empty, iterator raises `StopAsyncIteration`.
- Slow consumer (100ms per event) on fast run does not deadlock.

`test_integration.py`:
- Run a real 3-agent topology (User → Researcher → Writer) with `ExecutionConfig.aggui.enabled=True`.
- Capture every emitted AG-UI event.
- Assert sequence:
  - First: `RunStarted` with `runId == session_id` AND `input["_marsys_handshake"] == {schema_version: 1, marsys_version, ag_ui_version}`
  - Second: `StateSnapshot` (empty or near-empty initial state)
  - Then a mix of `StepStarted`, `TextMessage*`, `ToolCall*`, `StepFinished`, `StateDelta`, `Custom("marsys.branch.*")`, `Custom("marsys.parallel.group")`, `Custom("marsys.convergence")` in event order
  - Last: `RunFinished` with `outcome="success"` and `result.final_response` matching the Orchestra result
- Every event passes `BaseEvent.model_validate(event.model_dump())` round-trip (validates against the SDK's models)
- SSE round-trip: every event's `aggui_event_to_sse(event)` parses back to the same event via `EventEncoder` (or AG-UI SDK's parser if one exists)
- Every `TextMessage*` triple has matching `messageId`s
- Every `ToolCall*` sequence has matching `toolCallId`s

`test_exhaustive_mapping.py` (the load-bearing regression test):
- Reflectively walk `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`. Discover every class that is a subclass of `StatusEvent` (or whatever the base is).
- Assert each discovered class is in `mapping.DISPATCH` keys OR in `mapping.INTERNAL_ONLY` set.
- If neither: test fails with a clear message naming the unmapped event class.

### Edge-case coverage matrix

| Scenario | Test |
|---|---|
| Handshake rides inside `RunStarted.input["_marsys_handshake"]` (first event is `RunStarted`) | `test_integration.py` |
| RunFinished closes iterator | `test_stream.py` |
| RunError closes iterator | `test_stream.py` |
| TextMessage triple well-formed | `test_integration.py` |
| ToolCall sequence well-formed | `test_integration.py` |
| StateSnapshot then StateDelta in correct order | `test_integration.py` |
| Drop-newest on overflow + lagged Custom | `test_translator.py` |
| Mapping raises does not break other mappers | `test_translator.py` |
| Exhaustive coverage of event classes | `test_exhaustive_mapping.py` |
| SSE round-trip via SDK encoder | `test_integration.py` |
| AssistantMessageEvent emission | `tests/agents/test_agents.py` (regression for the new emission site) |
| TraceCollector subscribes to AssistantMessageEvent | `tests/coordination/tracing/test_collector.py` (regression) |

### Framework regression suite

Entire `packages/framework/tests/` passes with the SAME counts as baseline. Document baseline + post-change counts in "What was actually built."

---

## Resolved decisions (was: open questions)

Resolved 2026-05-12 with the framework implementer and user before implementation. The brief's prior list of open questions is preserved here with each resolution noted.

1. **`AssistantMessageEvent` scope.** **RESOLVED — in scope.** Without this event, AG-UI `TextMessageContent` would be empty, gutting the translator's primary value. Adding the event is ~10 lines at `agents.py:3129`; TraceCollector handler mirrors the content-addressed input capture pattern (commit `d2b600e`). Multi-consumer benefit: LangSmith / Phoenix / Langfuse adapters can stop inferring assistant content from span attributes.

2. **AG-UI version pin.** **RESOLVED — keep `ag-ui-protocol==0.1.18`.** PyPI's latest release on 2026-05-12 is still 0.1.18 (released April 21, 2026). No newer pre-release. Pin tightly.

3. **Backpressure.** **RESOLVED — drop-newest.** Confirmed; preserves prefix coherence of `TextMessageStart/Content/End` triples. Cumulative drop count is signalled via a `Custom("marsys.stream.lagged")` event emitted on the next successful put after overflow.

4. **Custom event Pydantic validation mode.** **RESOLVED — strict.** Custom events that fail Pydantic `model_validate` raise. Catches schema drift fast; one mode = one code path. Consumers that need lenient behavior wrap with try/except at their boundary — not a framework concern.

5. **`StateDelta` granularity.** **RESOLVED — one delta per state change (option a).** Simpler + more accurate. Optimize later if profiling shows it matters for runs with hundreds of state transitions.

6. **Optional dependency vs hard dependency.** **RESOLVED — optional (`marsys[aggui]`).** Keeps the framework lean for Phoenix-only / LangSmith-only / raw-script users who don't need AG-UI.

### Additional decisions made during validator + improver review

7. **Handshake placement.** **RESOLVED IN TWO STAGES.**
   - **Initial decision (pre-implementation):** inside `RunStartedEvent.input["_marsys_handshake"]` (not a leading Custom). Rationale: AG-UI consumers that filter on lifecycle events would silently drop a pre-`RunStarted` Custom.
   - **Revised (during B1.10 testing):** leading `Custom("marsys.aggui.handshake")` event BEFORE `RunStarted`. Verified against `ag-ui-protocol==0.1.18`: `RunStartedEvent.input` is a strongly-typed `RunAgentInput` (requires `thread_id`, `run_id`, `state`, `messages`, `tools`, `context`, `forwarded_props`) designed to echo client request shape — not a free-form metadata pocket. Pinching the handshake into `input` requires synthesizing those required fields with empty values, which is uglier than a leading Custom. Returned to the brief's original design.

8. **Where AGGUITranslator wiring lives.** **RESOLVED — `_wire_event_bus()`** (not `_initialize_components()` directly). `_wire_event_bus()` is called from both `__init__` and `resume_session`; this gives resume parity.

9. **`get_aggui_translator(run_id)` method on `Orchestra`.** **RESOLVED — dropped.** Orchestra is per-run; `run_id` indirection is dead weight. `AGUIEventStream(translator)` takes the translator directly. `orchestra.aggui_translator` is exposed as a plain attribute, matching `orchestra.trace_collector`.

10. **`BranchEvent` + `ValidationDecisionEvent` disposition.** **RESOLVED — `NOT_YET_EMITTED` bucket** (third category alongside `DISPATCH` and `INTERNAL_ONLY`). Both are defined but have no emission sites today. Explicit deferred-mapping forces the future PR author who adds emission to make a mapping decision.

11. **`MarsysRunState.barriers` schema.** **RESOLVED — trimmed.** v0.3 schema: `barrier_id`, `status`, `rendezvous_node`, `group_id`, `successful_count`, `total_count`. Fields requiring not-yet-emitted barrier events (`arrived_count`, `resolver_branch`) are deferred to a future session. `total_steps` is incremented on each `AgentCompleteEvent`.

12. **Custom event docs source-of-truth.** **RESOLVED — generated from Pydantic.** `scripts/generate_aggui_custom_events_doc.py` walks `CUSTOM_EVENT_REGISTRY`, writes `aggui-custom-events.md` from `model.model_json_schema()`. CI test fails if checked-in markdown drifts. Hand-maintained markdown was rejected — drift is inevitable.

13. **JSON Patch generation.** **RESOLVED — adopt `jsonpatch>=1.33` library** (RFC 6902, MIT). Hand-rolling mishandles RFC 6902 path escaping (`/` and `~` in keys). Library is small + stdlib-only deps.

14. **`ToolCallEvent.error_summary` mapping.** **RESOLVED — that field doesn't exist on the event.** Failed-status mapping falls back to `result_summary` or a generic literal. The surrounding `ErrorEvent` (mapped to `Custom("marsys.error")`) is the source of failure detail.

15. **`MemoryResetEvent` discovery in exhaustive test.** **RESOLVED — multi-module walk.** The test walks `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, AND `agents/memory.py`. Reflection over `StatusEvent` subclasses alone would miss `MemoryResetEvent` (it doesn't inherit from `StatusEvent`).

16. **`pytest.importorskip("ag_ui")` in test files.** **RESOLVED — yes.** Every test file in `tests/coordination/aggui/` begins with this line so CI works for contributors who didn't `pip install '.[aggui]'`.

---

## Cross-references

- Consumer (Spren v0.3): `docs/implementation/spren/sessions/v0.3.0/04-run-execution.md` — **does not exist on disk at brief-write time**. Will be written by Spren's architect session. This brief defines the contract Spren consumes. If Spren's plan disagrees with this brief, the brief is authoritative (multi-consumer principle); Spren accommodates.
- Related framework pattern (multi-consumer hooks): [`./02-telemetry-sink-protocol.md`](./02-telemetry-sink-protocol.md)
- Framework Spren-support summary: [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — Session 06 row currently marked "scoped"; this session updates to ✅ shipped.
- AG-UI specification: https://docs.ag-ui.com — current Python SDK at `ag-ui-protocol==0.1.18` (April 2026)
- AG-UI Python SDK reference: https://docs.ag-ui.com/sdk/python/core/overview
- Spren architecture (consumer side): [`../../../../architecture/spren/03-api-design.md`](../../../../architecture/spren/03-api-design.md) at line 117, [`../../../../architecture/spren/06-observability.md`](../../../../architecture/spren/06-observability.md). **Both are stale**: API design doc at line 117 says `marsys.transport.aggui`; observability diagram says the translator lives in Spren. Both must be updated to point at `marsys.coordination.aggui` after this session lands (separate doc-update task).
- Spren design principles: [`../../../../architecture/spren/08-design-principles.md`](../../../../architecture/spren/08-design-principles.md) — SP-004 (AG-UI as wire schema), SP-018 (framework purity)
- Framework architecture overview: [`../../../../architecture/framework/overview.md`](../../../../architecture/framework/overview.md)
- Framework design principles: [`../../../../architecture/framework/design-principles.md`](../../../../architecture/framework/design-principles.md) — DP-006 (adapter pattern; AGGUITranslator is the adapter from internal events to the external wire protocol)

---

## Sign-off

On completion:

1. Update **What was actually built** below with delta from plan
2. Update [`../../v0.3-spren-support.md`](../../v0.3-spren-support.md) — add Session 06 row to the table
3. Note the framework release version that ships this feature
4. Add **Lessons / Surprises** below

### What was actually built (filled by implementer)

**Date:** 2026-05-13. **Framework version target:** next unreleased after `0.2.1-beta`.

**Test counts:**
- Baseline (before Session 06): 1048 passing / 19 failing (pre-existing, unrelated — see "Pre-existing failures" below) / 40 skipped.
- After Session 06: 1117 passing / 19 failing (same pre-existing) / 40 skipped.
- New tests added: **69** — `tests/coordination/aggui/test_state.py` (9), `test_custom_events.py` (7), `test_exhaustive_mapping.py` (4), `test_mapping.py` (29), `test_translator.py` (8), `test_stream.py` (4), `test_integration.py` (7), `test_doc_generation.py` (1).

**AG-UI SDK version:** `ag-ui-protocol==0.1.18` (pinned as planned; verified latest on PyPI on 2026-05-13).

**Files created:**
- Production (`packages/framework/src/marsys/coordination/aggui/`): `__init__.py`, `config.py`, `state.py`, `custom_events.py`, `mapping.py`, `translator.py`, `sse.py`.
- Tests (`packages/framework/tests/coordination/aggui/`): `__init__.py`, `test_state.py`, `test_custom_events.py`, `test_exhaustive_mapping.py`, `test_mapping.py`, `test_translator.py`, `test_stream.py`, `test_integration.py`, `test_doc_generation.py`.
- Script: `packages/framework/scripts/generate_aggui_custom_events_doc.py`.
- Doc (auto-generated): `docs/architecture/framework/aggui-custom-events.md`.

**Files modified:**
- `packages/framework/src/marsys/coordination/status/events.py` — added `AssistantMessageEvent`.
- `packages/framework/src/marsys/agents/agents.py` — emits `AssistantMessageEvent` after `model.arun()` returns (line ~3132–3147, same gate as `AgentMessagesPreparedEvent` at line 3110–3119; not emitted on error path).
- `packages/framework/src/marsys/coordination/tracing/collector.py` — added `AssistantMessageEvent` subscription (14th entry) + `_handle_assistant_message` handler that mirrors the content-addressed pattern from `_handle_agent_messages_prepared` (commit `d2b600e`).
- `packages/framework/src/marsys/coordination/config.py` — added `aggui: AGGUIConfig` field on `ExecutionConfig`; imports `AGGUIConfig` from `coordination.aggui.config`.
- `packages/framework/src/marsys/coordination/orchestra.py` — additive block in `_wire_event_bus()` constructs `AGGUITranslator` when `aggui.enabled` (sibling to `TraceCollector` block; `_wire_event_bus()` is also called from `resume_session` so resumed runs get a fresh translator).
- `packages/framework/pyproject.toml` — added `[project.optional-dependencies] aggui = ["ag-ui-protocol==0.1.18", "jsonpatch>=1.33"]`.
- `packages/framework/tests/agents/test_agent.py` — fixed pre-existing fixture failures (passed `api_key="test"` to three `ModelConfig` constructor sites; unrelated to Session 06 but blocked baseline).
- `packages/framework/CHANGELOG.md` — release entry under `[Unreleased]`.
- `docs/implementation/framework/v0.3-spren-support.md` — Session 06 row marked ✅ shipped with one-line outcome.

**Things done differently from the original plan:**

1. **Handshake placement.** Original (revised) decision: ride inside `RunStartedEvent.input["_marsys_handshake"]`. Reverted to a leading `Custom("marsys.aggui.handshake")` after discovering that `ag-ui-protocol==0.1.18`'s `RunStartedEvent.input` is a strongly-typed `RunAgentInput` (requires `thread_id`, `run_id`, `state`, `messages`, `tools`, `context`, `forwarded_props`) designed to echo a client request shape, not a free-form metadata pocket. Pinching the handshake into `input` would require synthesizing those required fields. The leading-Custom approach is the brief's original design — verified at implementation time as the right call. acceptance.md AC-20 / AC-21 / AC-22 / AC-86 were revised accordingly with `[revised 2026-05-13]` annotations.

2. **`ReasoningMessageStartEvent.role`.** The mapping originally used `role="assistant"` for `AgentThinkingEvent`-derived reasoning events; the AG-UI SDK's `ReasoningMessageStartEvent.role` is a `Literal["reasoning"]` — corrected.

3. **`run_id` parameter on `AGGUITranslator`.** Dropped at construction time — the translator never used it (mappers read `event.session_id` directly). Constructor is now `AGGUITranslator(event_bus, config)`.

4. **Pre-existing test failures triaged, not fixed.** 19 baseline failures across `test_managed_memory.py` (8, `ManagedMemoryConfig` API drift in test fixtures), `test_provider_integration.py::TestOpenAIProvider` (7, missing `OPENAI_API_KEY`), `test_memory_manual.py::test_message_with_images` (1, missing fixture file), `test_agent.py` (2, residual fixture issues beyond the 3 I fixed), and `test_learnable_agents.py` (1, similar pattern). None caused by Session 06; fixing them all would have exploded scope into test-infrastructure maintenance. Surfaced as a known-baseline failure set.

### Lessons / Surprises (filled by implementer)

**The biggest surprise:** AG-UI SDK's `RunStartedEvent.input` is NOT a free-form dict. It's a typed `RunAgentInput` modeling a client's run-start request payload. The brief (and my plan revision) assumed `input` was a generic metadata pocket. Verifying SDK signatures with `model_fields` BEFORE designing the mapping would have caught this — a useful pattern for the next AG-UI session. The handshake-as-leading-Custom design is honest to the protocol; the in-input design was wishful thinking.

**Validator drift was substantial.** Plan-cited line numbers for `orchestra.py` emission sites (`760`, `700`, `207-216`) were off by 100–140 lines; `_initialize_components` had moved entirely and TraceCollector was now constructed in `_wire_event_bus()` (a Session 03 refactor). Re-grepping at implementation time before coding paid for itself — the alternative would have been silently wiring the translator into `_initialize_components` and breaking resume_session for AG-UI streams.

**`MemoryResetEvent` is an outlier.** It does not inherit from `StatusEvent` and lives in `agents/memory.py` rather than `coordination/status/events.py` or `coordination/tracing/events.py`. The exhaustive test's reflective discovery had to walk four modules and accept multiple base-class lineages. Reflection over a single base class would silently miss this event.

**`ToolCallEvent` has no `error_summary` field.** The plan's mapping referenced a field that doesn't exist on the event. The failed-status emission at `tool_executor.py:349-358` carries no error content at all — the surrounding `ErrorEvent` (mapped to `Custom("marsys.error")`) is the real signal. The translator falls back to `result_summary or "tool failed"` so the AG-UI triple stays well-formed even when no error detail is available.

**`BranchEvent` is defined but dormant.** Distinct class from `BranchCreatedEvent` / `BranchCompletedEvent` (different module — `status/events.py` vs `coordination/events.py`). TraceCollector subscribes to it but nothing ever emits it. The `NOT_YET_EMITTED` bucket — a third disposition alongside `DISPATCH` and `INTERNAL_ONLY` — captures this status explicitly; when emission lands later, the PR author must move it out.

**`ValidationDecisionEvent` is in the same bucket** — defined in `coordination/tracing/events.py`, never emitted yet (Phase 4 future work per the plan).

**Auto-generated docs paid off.** The `scripts/generate_aggui_custom_events_doc.py` + `test_doc_generation.py` pair means the markdown lives at the same address as the Pydantic models, and CI catches drift. Hand-maintained would have drifted on the very first follow-up that added a Custom event.

**Optional dependency strategy is clean.** Every test file in `tests/coordination/aggui/` begins with `pytest.importorskip("ag_ui")`. Contributors who didn't `pip install '.[aggui]'` see the suite skip cleanly. Framework's mandatory dependency set is unchanged.

### Pre-existing failures (documented baseline)

These failures existed before Session 06 and are unrelated to this work. None caused by my changes; none address-able within this session's scope. Surfaced for future maintenance:

| Test | Cause |
|---|---|
| `tests/memory/test_managed_memory.py` (8 tests) | `ManagedMemoryConfig.__init__` doesn't accept `max_total_tokens_trigger` — test fixtures out of sync with the runtime API |
| `tests/models/test_provider_integration.py::TestOpenAIProvider` (7 tests) | Need `OPENAI_API_KEY` env var |
| `tests/models/test_provider_integration.py::TestCrossProviderComparison` (1 test) | Same — needs API key |
| `tests/memory/test_memory_manual.py::test_message_with_images` | `FileNotFoundError` on missing fixture image |
| `tests/agents/test_agent.py::TestAgentMemory::test_memory_has_system_message` | Distinct from the 3 fixtures I fixed; different assertion failure |
| `tests/agents/test_learnable_agents.py::TestLearnableAgentMemory::test_memory_initialized_with_instruction` | Similar fixture pattern |
| `tests/memory/test_managed_memory.py::TestManagedMemoryConfig::test_default_config` & `test_custom_config` | Same `ManagedMemoryConfig` API drift |

Fixed by Session 06 (out of pragmatism — same fixture pattern as my new tests): `tests/agents/test_agent.py` — three `ModelConfig` construction sites that lacked `api_key="test"` (the validator requires either an env var or an explicit key).

**Note on `test_memory_has_system_message`:** Before this session, it failed at fixture-setup (same ModelConfig API-key issue). After my api_key fix, the fixture succeeds but the test fails at the assertion `len(system_messages) == 1` (`0 == 1`). The test's underlying premise — that an Agent stores a system message in `memory` at construction time — is no longer how the framework works (the agent constructs the system prompt dynamically during step preparation, at `agents.py:1423-1430`). The test is stale, not broken by Session 06; fixing it would require rewriting the assertion against the actual current behavior, which is out of scope. The failure count remains 19 (same baseline shape, different leaf assertion); the `[Unreleased]` regression bar is preserved.

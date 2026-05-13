# Acceptance criteria — AG-UI Event Stream Translator (Framework Session 06)

Frozen at 2026-05-13T00:00:00Z. The test auditor reads ONLY this file plus the test files. Implementation source is intentionally hidden.

## Functional

### Package surface and exports

- AC-1: A package `marsys.coordination.aggui` exists and is importable when the `aggui` optional dependency group is installed.
- AC-2: `marsys.coordination.aggui` publicly exports `AGGUITranslator`, `AGUIEventStream`, `aggui_event_to_sse`, `AGGUIConfig`, and `MarsysRunState`.
- AC-3: `AGUIEventStream` is constructed by passing an `AGGUITranslator` instance directly (signature: `AGUIEventStream(translator)`). It does NOT accept an `Orchestra` or `run_id` parameter.
- AC-4: There is no `get_aggui_translator(run_id)` method on `Orchestra`. The translator is exposed as a plain attribute `orchestra.aggui_translator`.

### `AssistantMessageEvent` (new EventBus event)

- AC-5: `AssistantMessageEvent` is defined in `marsys.coordination.status.events` (or the matching public module path used by other status events) and is importable.
- AC-6: `AssistantMessageEvent` carries at minimum `agent_name`, `step_number`, `step_span_id`, `message_id`, `content`, optional `tool_calls`, and optional `finish_reason` fields.
- AC-7: When an agent's model call (`model.arun()`) returns successfully, an `AssistantMessageEvent` is emitted on the EventBus with `content` set to the model's response text.
- AC-8: `AssistantMessageEvent` is NOT emitted on the agent error path (when `model.arun()` raises and the agent's exception handler emits `ErrorEvent` instead).
- AC-9: `TraceCollector` subscribes to `AssistantMessageEvent` (its subscription list contains this event type in addition to the previously existing event subscriptions).
- AC-10: When `TracingConfig.include_message_content` is `True`, the trace collector's handling of `AssistantMessageEvent` stores the assistant content via the same content-addressed pattern used for `AgentMessagesPreparedEvent` (content stored by hash; `message_id` + `content_hash` attached to the enclosing step span). When `include_message_content` is `False`, only the `content_hash` is attached and the body is not stored.

### Optional dependency declaration and isolation

- AC-11: `pyproject.toml` declares `ag-ui-protocol==0.1.18` AND `jsonpatch>=1.33` as optional dependencies under the `aggui` extras group (`pip install marsys[aggui]` installs both).
- AC-12: Neither `ag-ui-protocol` nor `jsonpatch` appears in the framework's mandatory (runtime) dependency set.
- AC-13: With `ag-ui-protocol` and `jsonpatch` uninstalled, the rest of the framework's test suite passes; tests in `tests/coordination/aggui/` skip cleanly.
- AC-14: Every test file in `tests/coordination/aggui/` begins with `pytest.importorskip("ag_ui")` so the suite still runs cleanly for contributors who haven't installed the optional `aggui` extras.

### Event mapping — exhaustive registry

- AC-15: Module `marsys.coordination.aggui.mapping` (or the equivalent location used by the aggui package) exposes three sets/dicts named `DISPATCH`, `INTERNAL_ONLY`, and `NOT_YET_EMITTED`, and a derived `EVENT_REGISTRY` equal to `set(DISPATCH.keys()) | INTERNAL_ONLY | NOT_YET_EMITTED`.
- AC-16: `INTERNAL_ONLY` contains exactly `{AgentMessagesPreparedEvent, MemoryResetEvent}` for v0.3.
- AC-17: `NOT_YET_EMITTED` contains exactly `{ValidationDecisionEvent, BranchEvent}` for v0.3.
- AC-18: Every `*Event` class discoverable across `coordination/status/events.py`, `coordination/tracing/events.py`, `coordination/events.py`, AND `agents/memory.py` is present in `EVENT_REGISTRY`. The discovery walks all four modules and handles classes that do NOT inherit from `StatusEvent` (e.g., `MemoryResetEvent`).
- AC-19: Adding a new `*Event` class to any of the four enumerated event modules without registering it in `DISPATCH`, `INTERNAL_ONLY`, or `NOT_YET_EMITTED` causes the exhaustive-mapping test to fail with a message that names the unmapped event class.

### Event mapping — lifecycle events

- AC-20 [revised 2026-05-13]: `ExecutionStartEvent` maps to a three-event sequence in order: (1) a leading `Custom("marsys.aggui.handshake")` event, (2) a `RunStarted` AG-UI event with `runId == session_id` and `threadId == session_id`, (3) a `StateSnapshot` event carrying the initial near-empty `MarsysRunState`.
- AC-21 [revised 2026-05-13]: The `Custom("marsys.aggui.handshake")` value is a dict with keys `schema_version` (int, equal to `1` for v0.3), `marsys_version` (string), and `ag_ui_version` (string).
- AC-22 [revised 2026-05-13]: The handshake is delivered as a leading `Custom("marsys.aggui.handshake")` BEFORE `RunStarted`. Rationale (post-implementation discovery): AG-UI's `RunStartedEvent.input` is a strongly-typed `RunAgentInput` (requires `thread_id`, `run_id`, `state`, `messages`, `tools`, `context`, `forwarded_props`) designed to echo client request shape, not a free-form metadata pocket. The leading-Custom pattern is the documented AG-UI escape hatch for protocol-version metadata. The original session-plan decision to ride inside `RunStarted.input` was based on the false assumption that `input` was a free-form dict.
- AC-23: `FinalResponseEvent` with `success=True` maps to a `RunFinished` AG-UI event with `outcome="success"` and `result` containing `final_response`, `total_duration_ms`, and `total_steps`.
- AC-24: `FinalResponseEvent` with `success=False` maps to a `RunError` AG-UI event with `code="execution_failed"`.
- AC-25: `CriticalErrorEvent` maps to a `RunError` AG-UI event carrying `message` and `code` derived from the event's fields. This is a terminal event — the stream closes after emission.
- AC-26: `ErrorEvent` maps to a single `Custom` event with `name="marsys.error"` whose `value` contains `agent`, `error_class`, `message`, `recoverable`, and `retry_count`. The run is NOT terminated by this mapping.
- AC-27: `ResourceLimitEvent` maps to a single `Custom` event with `name="marsys.resource.limit"` whose `value` contains `resource_type`, `pool_name`, `limit_value`, `current_value`, and `suggestion`.

### Event mapping — step events

- AC-28: `AgentStartEvent` maps to a `StepStarted` AG-UI event with `stepName == f"{agent_name}#{step_number}"`.
- AC-29: `AgentCompleteEvent` maps to a `StepFinished` AG-UI event with `stepName == f"{agent_name}#{step_number}"`.
- AC-30: `AgentMessagesPreparedEvent` produces NO AG-UI output (it is in `INTERNAL_ONLY`).

### Event mapping — generation events

- AC-31: `AssistantMessageEvent` maps to a triple `TextMessageStart` + `TextMessageContent` + `TextMessageEnd`, all sharing the same `messageId`. The `TextMessageStart` has `role="assistant"` and the `TextMessageContent.delta` equals the event's `content`.
- AC-32: `AgentThinkingEvent` with non-empty thought maps to a triple `ReasoningStart` + `ReasoningMessageContent` + `ReasoningEnd`, all sharing the same `messageId`, where `ReasoningMessageContent.delta` equals the thought.
- AC-33: `AgentThinkingEvent` with empty or absent thought emits NO AG-UI events (the triple is skipped).
- AC-34: `GenerationEvent` maps to a single `Custom` event with `name="marsys.generation.metadata"` whose `value` carries `model`, `provider`, `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, and `finish_reason`.

### Event mapping — tool call events

- AC-35: `ToolCallEvent` with `status="started"` maps to `ToolCallStart` + `ToolCallArgs`, where `ToolCallStart` carries `toolCallId`, `toolCallName`, and `parentMessageId` (most recent assistant message id) and `ToolCallArgs.delta` is the JSON-serialized arguments dict.
- AC-36: `ToolCallEvent` with `status="completed"` maps to `ToolCallEnd` + `ToolCallResult`, both using the SAME `toolCallId` that was issued for the matching `started` event. `ToolCallResult.content` equals `result_summary` (or `""` when absent) and `role="tool"`.
- AC-37: `ToolCallEvent` with `status="failed"` maps to `ToolCallEnd` + `ToolCallResult`, both reusing the same `toolCallId`. `ToolCallResult.content` falls back to `result_summary` (when present) or a generic literal (e.g., `"tool failed"`). The translator does NOT read a non-existent `error_summary` field from the event.
- AC-38: The translator maintains a per-stream identity map so that the `started`, `completed`, and `failed` events for the same tool call share a single `toolCallId`.

### Event mapping — branch / orchestration events (marsys-specific → Custom)

- AC-39: `BranchCreatedEvent` maps to `Custom` with `name="marsys.branch.created"` and `value` containing `branch_id`, `branch_name`, `source_agent`, `target_agents`, `trigger_type`, and `parent_branch_id`.
- AC-40: `BranchCompletedEvent` maps to `Custom` with `name="marsys.branch.completed"` and `value` containing `branch_id`, `last_agent`, `success`, and `total_steps`.
- AC-41: `ParallelGroupEvent` maps to `Custom` with `name="marsys.parallel.group"` and `value` containing `group_id`, `agent_names`, `status`, `completed_count`, and `total_count`.
- AC-42: `ConvergenceEvent` maps to `Custom` with `name="marsys.convergence"` and `value` containing `parent_branch_id`, `child_branch_ids`, `convergence_point`, `group_id`, `successful_count`, and `total_count`.

### Event mapping — user interaction events

- AC-43 [revised 2026-05-13]: `UserInteractionEvent` with `interaction_type="starting"` maps to `Custom` with `name="marsys.user_interaction.pending"` and `value` containing `agent_name`, `prompt_summary` (truncated to 200 chars), and `options`. Rationale: the framework's `UserInteractionEvent` (verified `coordination/status/events.py`) does NOT carry an `interaction_id` field — only `agent_name`, `interaction_type`, `prompt`, `options`. Adding a framework-level event field is outside Session 06's scope; future session may introduce it. For correlation, consumers key on `agent_name` + sequence position.
- AC-44 [revised 2026-05-13]: `UserInteractionEvent` with `interaction_type="completed"` maps to `Custom` with `name="marsys.user_interaction.resolved"` and `value` containing `agent_name`. Same `interaction_id` rationale as AC-43.
- AC-45 [revised 2026-05-13]: `UserInteractionEvent` with `interaction_type="timeout"` maps to `Custom` with `name="marsys.user_interaction.timeout"` and `value` containing `agent_name`. Same `interaction_id` rationale as AC-43.

### Event mapping — plan and memory state events

- AC-46: `PlanCreatedEvent` triggers emission of a `StateSnapshot` AG-UI event whose snapshot contains the new plan under `plans[agent_name]` with `goal` and `items`.
- AC-47: `PlanUpdatedEvent`, `PlanItemAddedEvent`, `PlanItemRemovedEvent`, and `PlanClearedEvent` each trigger emission of a `StateDelta` AG-UI event whose `delta` is a list of RFC 6902 JSON Patch operations targeting `/plans/...` paths.
- AC-48: `CompactionEvent` maps to `Custom` with `name="marsys.memory.compaction"` and `value` containing `agent_name`, `status`, `pre_tokens`, `post_tokens`, and `duration`.
- AC-49: `MemoryResetEvent` produces NO AG-UI output (it is in `INTERNAL_ONLY`).

### `MarsysRunState` schema

- AC-50: `MarsysRunState` is a Pydantic model exposing `schema_version` (defaulting to `1`), `branches` (dict[str, BranchState]), `barriers` (dict[str, BarrierState]), `plans` (dict[str, PlanState]), and `total_steps` (int, defaulting to `0`).
- AC-51: `BranchState` carries `branch_id`, `branch_name`, `current_agent`, `status` (one of `"RUNNING"`, `"WAITING"`, `"TERMINATED"`, `"FAILED"`, `"ABANDONED"`), `step_count`, and `parent_branch_id`.
- AC-52: `BarrierState` carries exactly `barrier_id`, `status` (one of `"OPEN"`, `"FIRED"`, `"CANCELLED"`), `rendezvous_node`, `group_id`, `successful_count`, and `total_count`. It does NOT include `arrived_count` or `resolver_branch` (those are deferred to a future schema version).
- AC-53: `PlanState` carries `agent_name`, `goal`, and `items` (list[PlanItemState]); `PlanItemState` carries `item_id`, `title`, and `status` (one of `"pending"`, `"in_progress"`, `"completed"`, `"abandoned"`).
- AC-54: `MarsysRunState.total_steps` is incremented by exactly one each time an `AgentCompleteEvent` is observed by the translator.
- AC-55: When an `AgentStartEvent` is observed, the corresponding branch's `current_agent` in `MarsysRunState.branches[branch_id]` is updated and a `StateDelta` is emitted reflecting the change.

### State snapshot and delta emission

- AC-56: A `StateSnapshot` AG-UI event is emitted exactly once immediately after the initial `RunStarted` event. Its snapshot is the empty-or-near-empty initial `MarsysRunState`.
- AC-57: Subsequent state changes (branch creation, branch agent change, barrier creation/fire, plan changes) emit a `StateDelta` AG-UI event whose `delta` is a valid RFC 6902 JSON Patch list. The patch is produced via `jsonpatch.make_patch(old_state.model_dump(), new_state.model_dump()).patch` (or equivalent that returns a valid RFC 6902 patch).
- AC-58: A `StateSnapshot` is replayable: applying every subsequent `StateDelta` in order yields a state equal to the final in-memory `MarsysRunState`.

### `Custom` event payload validation

- AC-59: Every `marsys.*` Custom event has a corresponding Pydantic model registered in a `CUSTOM_EVENT_REGISTRY` mapping (Custom name -> Pydantic model class).
- AC-60: At emission time, a Custom event's `value` payload is validated against its registered Pydantic model. Validation failure raises (strict mode — the translator does NOT swallow validation errors with try/except as default behavior).

### `Custom` event documentation generation

- AC-61: A script (e.g. `scripts/generate_aggui_custom_events_doc.py`) walks `CUSTOM_EVENT_REGISTRY`, calls `model.model_json_schema()` for each Pydantic model, and writes `docs/architecture/framework/aggui-custom-events.md` as one section per Custom name with JSON Schema blocks.
- AC-62: A test invokes the doc-generation script and diffs the output against the checked-in `docs/architecture/framework/aggui-custom-events.md`. The test FAILS if the markdown drifts from the schemas regenerated by the script.
- AC-63: `docs/architecture/framework/aggui-custom-events.md` exists, is checked in, and contains schemas for every `marsys.*` Custom event the translator can emit.

### Translator lifecycle

- AC-64: At construction, `AGGUITranslator` subscribes to every event class in `DISPATCH` on the provided `EventBus`.
- AC-65: Calling `AGGUITranslator.close()` unsubscribes from every event class it previously subscribed to.
- AC-66: When the mapping function for one event class raises, the exception is logged but does NOT propagate to the EventBus dispatcher, and other subscribers (e.g., `TraceCollector`) continue to receive events normally.

### `AGUIEventStream` async iterator semantics

- AC-67: `AGUIEventStream` is an async iterator: calling `async for event in stream` yields events in FIFO order from the translator's queue.
- AC-68: The first event yielded by the stream for a fresh run is `RunStarted`.
- AC-69: The second event yielded by the stream is `StateSnapshot`.
- AC-70: After a terminal event (`RunFinished` or `RunError`) is yielded AND the translator's queue is empty AND the translator is closed, the next call to `__anext__` raises `StopAsyncIteration`.

### Bounded queue and backpressure

- AC-71: The translator's queue is bounded by `AGGUIConfig.queue_max_size` (default `10000`).
- AC-72: Under overflow (producer outruns consumer beyond `queue_max_size`), the translator drops the NEWEST event (drop-newest policy). It does NOT drop older queued events.
- AC-73: When events have been dropped, the next successful enqueue is preceded by a `Custom` event with `name="marsys.stream.lagged"` whose `value` carries the cumulative count of dropped events accumulated since the last lagged notification.
- AC-74: After the lagged Custom is successfully enqueued, the lagged counter resets to zero.
- AC-75: A producer running at full speed against a slow consumer (e.g. consumer sleeps 100ms per event) on a `queue_max_size=10` queue does NOT deadlock. The run completes; events are either delivered or accounted for via lagged Customs.

### SSE wire format

- AC-76: `aggui_event_to_sse(event)` returns an SSE-formatted string of the form `data: {json}\n\n` (matching the shape produced by `ag_ui.encoder.EventEncoder.encode`).
- AC-77: An AG-UI event passed through `aggui_event_to_sse` and then back through `ag_ui.encoder.EventEncoder` (or equivalent parser) round-trips to a model-equivalent AG-UI event.
- AC-78: `aggui_event_to_sse` is a thin wrapper over the AG-UI SDK encoder — the framework does NOT ship custom SSE serialization logic that bypasses `ag_ui.encoder.EventEncoder`.

### Orchestra wiring and resume parity

- AC-79: `Orchestra` exposes `aggui_translator` as a plain attribute (mirroring `trace_collector`). Its value is `None` when AG-UI is disabled.
- AC-80: When `ExecutionConfig.aggui.enabled` is `True`, `Orchestra.aggui_translator` is set to an `AGGUITranslator` instance after the orchestra is initialised.
- AC-81: When `ExecutionConfig.aggui.enabled` is `False` (the default), `Orchestra.aggui_translator` is `None` and no AG-UI events are produced.
- AC-82: Translator construction happens inside `Orchestra._wire_event_bus()` (NOT directly in `_initialize_components()`). As a consequence, a run resumed via `resume_session` re-attaches a fresh translator to the new `EventBus` and produces a valid AG-UI stream on the resumed run.
- AC-83: `Orchestra.__init__` and `Orchestra.run` public signatures are unchanged by this session.

### Configuration

- AC-84: `ExecutionConfig` exposes an `aggui` field of type `AGGUIConfig` with `aggui.enabled` defaulting to `False` and `aggui.queue_max_size` defaulting to `10000`.

### Integration test — 3-agent workflow

- AC-85: An integration test runs a 3-agent workflow (User → Researcher → Writer) end-to-end with `ExecutionConfig.aggui.enabled=True` and captures the full AG-UI event sequence emitted by the translator.
- AC-86 [revised 2026-05-13]: In the captured sequence, the first event is `Custom("marsys.aggui.handshake")` with `value` populated with `schema_version`, `marsys_version`, and `ag_ui_version`. The SECOND event is `RunStarted` with `runId == session_id`.
- AC-87: In the captured sequence, the second event is `StateSnapshot` carrying a (near-empty) `MarsysRunState`.
- AC-88: In the captured sequence, the last event is `RunFinished` (with `outcome="success"` and `result.final_response` matching the Orchestra result) on the success path, or `RunError` on the failure path.
- AC-89: Every captured AG-UI event passes `BaseEvent.model_validate(event.model_dump())` round-trip against `ag_ui.core`'s SDK Pydantic models.
- AC-90: In the captured sequence, every `TextMessageStart` event has a matching `TextMessageEnd` event sharing the same `messageId`.
- AC-91: In the captured sequence, every `ToolCallStart` event has a matching `ToolCallEnd` event sharing the same `toolCallId`.
- AC-92: In the captured sequence, every event survives the SSE round-trip: `aggui_event_to_sse(event)` produces output that re-parses back to an equivalent AG-UI event via `ag_ui.encoder.EventEncoder`.

## Non-functional

### Multi-consumer / no-Spren coupling (hard rules)

- AC-93: No symbol from `spren` or any Spren-specific package is imported anywhere in `marsys.coordination.aggui` or in the new test files for this session.
- AC-94: There is no conditional branch in the translator or related modules whose condition tests for Spren (no `if running under Spren`, no consumer-name flag, no Spren-specific custom event).
- AC-95: The translator's public surface (`AGGUITranslator`, `AGUIEventStream`, `aggui_event_to_sse`, `AGGUIConfig`, `MarsysRunState`) is consumable by any AG-UI-speaking client; the same `AGUIEventStream(orchestra.aggui_translator)` + `aggui_event_to_sse` pattern works for Spren, MARSYS Cloud, MARSYS Studio, and a generic third-party AG-UI client.

### Optional-dependency isolation

- AC-96: The framework's full regression test suite passes with `ag-ui-protocol` and `jsonpatch` uninstalled. Tests located under `tests/coordination/aggui/` skip cleanly when the optional deps are missing (`pytest.importorskip("ag_ui")` short-circuits them).

### Regression baseline

- AC-97: The pre-existing framework regression suite passes with zero new failures relative to the baseline measured before this session's changes.

### CHANGELOG and shipping artifacts

- AC-98: `packages/framework/CHANGELOG.md` has a new entry under `[Unreleased]` describing the AG-UI translator package, the new `AssistantMessageEvent`, and the optional `marsys[aggui]` install group.
- AC-99: `docs/implementation/framework/v0.3-spren-support.md` is updated to mark Session 06 as shipped.

## Out of scope

The following are explicitly excluded from this session. Tests asserting these behaviors should NOT be added in this session.

- Token-level / chunked LLM streaming (`TextMessageContent` carries the full assistant text in a single delta for v0.3).
- HTTP server / SSE endpoint hosting (consumer concern — Spren / Cloud / Studio wrap the iterator themselves).
- An AG-UI client implementation (consumer concern).
- Multi-run streams from a single translator (one stream per `run_id` in v0.3).
- Persistence of AG-UI events as the storage format (use `TelemetrySink` for span-shaped persistence).
- Migration tooling for AG-UI v0.x → v1.x (handle when AG-UI tags v1).
- `ValidationDecisionEvent` and `BranchEvent` active mapping (both are in `NOT_YET_EMITTED` because they have no emission sites in v0.3).
- Schema fields `arrived_count` and `resolver_branch` on `BarrierState` (deferred to a future framework session that emits dedicated barrier events).
- Updating `docs/architecture/spren/06-observability.md` to reflect the framework-side translator location (tracked as a separate doc-update task).

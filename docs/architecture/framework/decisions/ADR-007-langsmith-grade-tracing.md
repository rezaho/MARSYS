# ADR-007: LangSmith-grade LLM trace capture

**Status**: Accepted
**Date**: 2026-05-09 (initial), 2026-05-10 (Azure / Responses-API parsing amendments)
**Related**: ADR-005 (unified-barrier orchestration — defines the span tree this ADR populates).

## Context

The streaming NDJSON tracing pipeline (introduced in commit `12447f7`, `d21c618`, `ddf68a8`) records `execution / branch / step / generation / tool / compaction` spans, but a generation span historically carried only metadata: model name, provider, finish reason, token counts, and booleans for `has_thinking` / `has_tool_calls`. The actual LLM payload — input messages, advertised tool schemas, sampling parameters, response content, thinking, structured tool_calls, response metadata — was not on the trace. With only metadata you can confirm a model call happened; you cannot debug *why* it produced what it did.

LangSmith / Phoenix / Langfuse / Datadog observability backends are designed around the full payload. To make them useful, we have to capture the full payload first and then ship it.

Two design questions had to be resolved before writing code:

1. **Where in the call chain do we capture the payload?**
2. **How do we propagate the per-step correlation context (step span id, branch id, agent name, event bus) down to the capture point without poisoning every signature?**

A third question — *what on-disk and remote storage formats do we use* — was settled in [§D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite).

## Decision

### D1. Capture at the model-wrapper layer (recording, not reconstruction)

`StepExecutor` already had a hook (`_serialize_input_messages`) that built a snapshot of the agent's input messages before the agent ran, fed via `AgentStartEvent.messages` into the collector and on into `MessageStore`. That hook is **input-only** by construction (the response hasn't happened yet) and is **a reconstruction**: it re-serializes the agent's working memory plus the current request, and a divergence at any later layer — system-prompt prepending, format processors injecting steering tools, retry rewrites, compaction — silently leaves the trace and the wire desynchronized.

For LangSmith-grade debugging we need the literal kwargs the model wrapper handed to the adapter. The wrapper layer is also where:

- The `tools` list has been finalized.
- Sampling parameters are settled.
- The retry loop sits — so retries with rewritten messages each get their own captured pair.
- Tool-internal LLM calls (`web_tools.py`, `BrowserAgent` vision call) and memory compaction (`SummarizationProcessor._run_compaction_model`) flow through, and would otherwise bypass any step-level reconstruction entirely.

We capture there and accept the cost: clean separation between coordination and transport layers is broken (model.py imports tracing lazily). Every other choice we evaluated either missed call paths or introduced silent drift.

### D2. Explicit `TraceContext` kwarg threading — no ContextVars

The original ADR draft used seven `ContextVar`s (`_current_step_span_id`, `_current_branch_id`, `_current_agent_name`, `_current_session_id`, `_current_event_bus`, `_current_call_kind`, `_in_capture`) so the capture decorator could read them off the running task without modifying any signatures. This is convenient — but the cost has bitten us before:

- **Implicit data flow.** A function's signature lies about its dependencies; reasoning becomes "what's set on the running task" instead of "what was passed in."
- **Lifecycle on the honor system.** `set` with no matching `reset` leaks state across tests and between concurrent runs. The original ADR draft documented the leak as a known footgun in §7.1–7.2 and proposed a mitigation (clearing in `finally`) which itself relies on every call site doing the right thing.
- **`run_in_executor` doesn't propagate `ContextVar`s.** `BaseAPIModel.arun` falls back to `loop.run_in_executor(None, sync_run)` when no async adapter is present; the wrapped sync call would see empty trace context and silently bypass capture.
- **Test ergonomics.** Any test touching the model layer needs fixtures resetting every var, or pytest-async's task-scoping bites.
- **The static decoupling is a runtime lie.** The capture helper never appeared on a function signature, but every model wrapper depended on it through these vars. The dependency was real and debug-relevant; hiding it didn't simplify anything, it only made it harder to see.

We replaced the seven vars with a single `TraceContext` frozen dataclass threaded explicitly:

```text
StepExecutor builds TraceContext(step_span_id, branch_id, agent_name, session_id, event_bus, kind="generation")
       │
       ▼
context["trace_ctx"]  ──→  run_context["trace_ctx"]
                                  │
                      agent.run_step(request, run_context)
                                  │
                                  ▼
                      model_kwargs["trace_ctx"]
                                  │
                          agent._run(**model_kwargs)
                                  │
                                  ▼
                   model.arun(messages, ..., trace_ctx=trace_ctx)
                                  │
                                  ▼
                   capture_llm_call(trace_ctx, ...)
```

`Branch` already carried `step_span_id` through `context["step_span_id"]`, so we extended an existing pattern rather than inventing a new one. The forwarding boilerplate is ~30 lines of code total (one merge in `BaseAgent.run_step`, one merge in `InteractiveElementsAgent._run`, one parameter on `_run_compaction_model`, one read in `_forced_summarization_fallback`, one `pop("trace_ctx", None)` in each of the three wrapped `arun` methods). Compaction calls derive `trace_ctx.child(kind="compaction")` so they land as compaction-flavored spans without a separate plumbing path.

`TraceContext.captured` is the re-entrancy guard. When `LearnableModelWrapper.arun` opens a capture and then delegates to `BaseLocalModel.arun`, the inner wrapper would otherwise re-emit. The outer helper passes `cap.inner_ctx` (which is `trace_ctx.mark_captured()`) to the inner call; the inner `capture_llm_call` sees `captured=True` and bypasses. Exactly one request/response pair lands per LLM invocation regardless of wrapper depth.

### D3. Async context manager — not a decorator, not inline

We needed:

| Property | Decorator | Raw inline | Helper fns | **Context manager** |
|---|---|---|---|---|
| Visible at call site | no | yes | yes | **yes** |
| Single source of truth for emit logic | yes | no | yes | **yes** |
| Error path automatic | yes | manual | manual | **yes** |
| Re-entrancy guard centralized | yes | manual | manual | **yes** |

`async with capture_llm_call(...) as cap:` wins on all four. The emit-before-body / emit-after-body pattern is exactly what context managers exist for; the body is the literal call site of the inner adapter, so it's debuggable. `cap.set_response(response)` records the result for the close event (forgetting it produces a response event with `content=None` rather than silent data loss). Re-raise on exception is built into `__aexit__` semantics.

### D4. Reuse the existing event bus

Capture emits `LLMRequestEvent` / `LLMResponseEvent` on the same `AsyncEventBus` already wired into `Orchestra`. The collector subscribes (`_handle_llm_request` opens a generation/compaction span, `_handle_llm_response` closes it and streams via `_stream_span`). No new sink is needed — the NDJSON writer already fans out from `_stream_span`, and any user-supplied `TelemetrySink` (including the new `OtelTraceWriter`) hooks in at the same point.

This means: enabling the new capture pipeline does not require touching any sink. Existing sinks see strictly more spans with strictly more attributes; nothing they parsed before changes shape. Consumers who want only metadata can continue to ignore the new attributes.

### D5. Tool schemas dedup through the same `MessageStore`

Tool schemas can be 30 KB+ (a full coordination-tools catalog plus user tools) and rarely change between steps. Capturing them inline on every span would bloat NDJSON. When `TracingConfig.capture_full_input` is on and a `MessageStore` is configured, the collector hashes tool schemas through the store using a synthetic `[{"role": "tool_schema", "content": tools}]` envelope — a single tool catalogue lands once across all steps. The inline `tools` attribute remains so OTel-bound consumers (which can't follow CAS pointers) still see content.

### D6. Response payload kept inline (for now)

Response content / thinking / tool_calls are kept inline on span attributes, not routed through `MessageStore`. Responses are typically modest in size and almost always unique per call; the dedup payoff doesn't justify a second hashing surface. Promote to the store later if NDJSON line sizes become a problem.

### D7. Keep NDJSON default, add OTel sink, drop SQLite

The original ADR draft planned a SQLite + OTel pair as the storage layer. We compared SQLite vs NDJSON along the two axes that matter for this framework:

- **Compactness** — wash. Both dedup heavy content (NDJSON via the sidecar `messages/` directory; SQLite via a `content` table). Zipped NDJSON is usually smaller than the equivalent SQLite file.
- **Concurrency** — NDJSON wins decisively. File-per-trace + content-addressed sidecar means parallel branches and parallel `Orchestra.run()` invocations contend on **nothing**. SQLite has a single-writer lock — every parallel branch's span emission queues on it. For a framework whose whole point is fan-out, that's a structural mismatch.

Decision:

- **NDJSON stays the default** local store (already wired in `Orchestra.__init__`).
- **`OtelTraceWriter` ships as opt-in `TelemetrySink`** under the new `tracing-otel` extra. Covers LangSmith, Langfuse, Phoenix, Jaeger, Tempo, Datadog over OTLP/HTTP using `gen_ai.*` semantic conventions plus OpenInference attributes for LangSmith's chat-bubble rendering.
- **SQLite writer dropped entirely.** No cross-run-SQL use case on the team; OTel + LangSmith covers vendor querying. Dropping it saves ~500 lines of code we don't have to maintain or refactor to streaming.

Post-Phase-2 storage is two-layer: NDJSON locally for "what happened on this machine," LangSmith remotely for "let me actually look at it."

### D8. Indexed gen_ai prompt + single-blob completion (LangSmith pragmatics)

LangSmith's UI parses two parallel attribute schemes — OpenInference (`llm.input_messages.{i}.message.*`) for the input panel and indexed `gen_ai.prompt.{i}.*` for some integrations. We emit both, plus a `gen_ai.prompt` JSON-blob fallback, because the GenAI semconv is in flux and different backends key off different things.

For the **output** side we deliberately ship only a single-blob `gen_ai.completion` and skip indexed `gen_ai.completion.{n}.*`. LangSmith's renderer ignores the blob fallback when the indexed pattern exists and renders an empty output panel for the `content=null + tool_calls=[…]` shape (the common shape for tool-using agents). The blob carries the full payload so the panel populates. This is captured directly in `_build_output_message`'s docstring along with the strict invariant: the output dict must contain only `{role, content, tool_calls, thinking}` — adding `finish_reason` or `usage` flips the renderer back to a JSON-fields panel. `finish_reason` is shipped on `gen_ai.response.finish_reasons`; usage on `gen_ai.usage.{input,output}_tokens`. A regression test (`test_output_value_is_strict_openai_chat_completion_shape`) asserts the key set never grows.

## Architecture

```text
                Orchestra → step_executor builds TraceContext
                                    │
                                    ▼
              context["trace_ctx"] ──→ run_context["trace_ctx"]
                                            │
                              agent.run_step(request, run_context)
                                            │
                                            ▼
                                model_kwargs["trace_ctx"]
                                            │
                                    agent._run(**model_kwargs)
                                            │
                                            ▼
                          model.arun(messages, ..., trace_ctx=...)
                                            │
                                ┌───────────┴───────────┐
                                │ async with             │
                                │  capture_llm_call(...) │ ──▶ EventBus.emit(LLMRequestEvent)
                                │     as cap:            │      (full payload: messages,
                                │   await adapter.arun() │       tools, sampling)
                                │   cap.set_response(r)  │
                                └───────────┬───────────┘ ──▶ EventBus.emit(LLMResponseEvent)
                                            │                  (content, thinking, tool_calls,
                                            ▼                   reasoning, metadata)
                                    HarmonizedResponse
                                            │
                            ┌───────────────┴────────────┐
                            │ TraceCollector handlers    │
                            │  _handle_llm_request:      │
                            │   open generation span,    │
                            │   route msgs + tools       │
                            │   through MessageStore     │
                            │  _handle_llm_response:     │
                            │   close span, attach       │
                            │   payload, _stream_span()  │
                            └───────────────┬────────────┘
                                            │
                                            ▼
                                    TelemetrySink fan-out
                                    (NDJSONTraceWriter + any user sinks
                                     including OtelTraceWriter)
```

## Trace-tree shape: parallel children under their dispatching step

The previous trace tree placed `branch` spans flat under the `execution` root regardless of how they were spawned. The first live LangSmith run made the limitation obvious — parallel fan-outs were spatially indistinguishable from sequential workflows. We added `BranchCreatedEvent.parent_step_span_id` (additive, optional) and threaded:

```text
RealRuntime.step → StepResult.step_span_id
       │
       ▼
Orchestrator._interpret → Branch.last_step_span_id
       │
       ▼
Orchestrator._handle_parallel_invoke → _spawn(parent_step_span_id=...)
       │
       ▼
BranchCreatedEvent(parent_step_span_id=...)
       │
       ▼
TraceCollector._handle_branch_created → parent_span = step_spans[parent_step_span_id] or trace.root_span
```

Branches dispatched by a step now nest under that step in the trace tree. Entry / initial branches fall back to the execution root (correct for that case). Orchestrators that don't populate the field still produce valid traces; only the nested rendering is degraded.

## Wire issues surfaced once Azure traces reached LangSmith

We discovered two `OpenAIAdapter.harmonize_response` bugs only after running real Azure (Cognitive Services / AI Foundry) traffic and looking at the resulting LangSmith spans:

1. **`content` defaulted to `""` instead of `None`.** OpenAI's convention — and what LangSmith's renderer expects in the assistant-bubble for tool-only responses — is `content: null` when the response is purely tool calls. An empty string flipped the renderer to the JSON-fields panel. Fixed: `content` defaults to `None`, only gets a string when a `message` item is present in the Responses API output array.
2. **Token usage parsing read chat-completions field names** (`prompt_tokens` / `completion_tokens` / `reasoning_tokens`) from a Responses API `usage` block that uses `input_tokens` / `output_tokens` / `output_tokens_details.reasoning_tokens`. Fixed: try Responses-API names first, fall back to chat-completions for endpoint compatibility.

Both fixes are general OpenAI Responses API correctness, not Azure-specific. Azure was the diagnostic vehicle because it surfaced reasoning-token accounting gaps; OpenRouter happened to mask them by passing through old-shape `usage` blocks.

A third fix in this same area: `BaseAPIModel.provider` was a `@property` deriving the lowercased adapter class name (`"asyncopenrouter"` for `AsyncOpenRouterAdapter`), so traced spans carried a non-canonical provider id and broke string equality with the configured provider name. Replaced with stored `self.provider` / `self.model_name` set in `__init__`.

## Consequences

- **Positive:**
  - Generation spans now carry the full LLM payload — debugging matches the wire by definition.
  - Compaction LLM calls and tool-internal LLM calls (web_tools, browser-agent vision) land in the trace under the right step. Previously invisible.
  - Single trace pipeline serves both local NDJSON and remote vendor backends; NDJSON remains the source of truth and is unaffected by vendor outages.
  - Re-entrancy guard makes nested wrappers (`LearnableModelWrapper` → `BaseLocalModel`) emit exactly one pair regardless of depth.
  - Adding a new model wrapper is a fixed pattern (wrap `arun` in `async with capture_llm_call(...)`); easy to lint for in CI later.
  - Adding a new vendor backend is a fixed pattern (`TelemetrySink` subclass under `tracing-otel` if it's OTel-shaped, or standalone if not).

- **Negative:**
  - `marsys.models.models` lazy-imports `marsys.coordination.tracing.capture`. The import is genuinely conditional (it only fires when an `arun` runs under a traced session), but the dependency exists. Models is no longer a leaf module from the tracing module's perspective.
  - Each wrapped `arun` adds ~10 lines of capture boilerplate. Three wrappers today, low maintenance burden, but linting it explicitly is worth doing if the wrapper count grows.
  - The OTel mapping needs to emit three independent attribute surfaces (`gen_ai.*` indexed, OpenInference indexed, JSON-blob fallback) because no single backend reads only one consistently. ~700 lines of mapping code in `OtelTraceWriter`. Each surface is testable in isolation; the cost is mostly comment volume.
  - The strict-output-shape invariant in `_build_output_message` is enforced by a regression test, not a type. Nothing prevents a future contributor from adding `usage` to the output dict and regressing LangSmith rendering. The test is the gate.

- **Risk:**
  - LangSmith's renderer rules are documented largely by reverse-engineering. If they change the rendering pipeline, our chat-bubble pragmatics in `_emit_indexed_prompt_attrs` (the assistant-with-empty-content fallback) may need to follow. Mitigation: live-test scripts under `live_tests/tracing/` exercise the path on every release.
  - `python-ulid` → `blake2b` → fixed-byte int: the deterministic ULID→OTel-ID mapping has a theoretical collision rate of ~2^-64 for span ids and ~2^-128 for trace ids. At our trace volumes (millions of spans per project lifetime, not per run), collisions are not a practical concern. If volumes grow, switch to a 128-bit span id (OTel's W3C trace context allows it but most backends still index on 64).

## Alternatives Rejected

1. **Reconstruction at `step_executor` instead of recording at the wrapper.**
   Rejected per [§D1](#d1-capture-at-the-model-wrapper-layer-recording-not-reconstruction): silently drifts from the wire when format processors / retry rewrites mutate messages between assembly and POST, and misses tool-internal LLM calls and compaction calls entirely.

2. **`ContextVar`-based propagation.**
   Rejected per [§D2](#d2-explicit-tracecontext-kwarg-threading--no-contextvars): implicit data flow, lifecycle leaks, `run_in_executor` boundary loss, test fragility.

3. **`@capture_arun` decorator on each model wrapper's `arun`.**
   Rejected because the decorator pattern hides the emit logic from the call site (we explicitly wanted it visible in the body), and because the kwargs the decorator needs to forward (`messages`, `tools`, sampling) require introspection on the wrapped function — fine for `BaseAPIModel.arun` but uneven across wrappers with different signatures.

4. **SQLite as primary on-disk store.**
   Rejected per [§D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite): single-writer lock contention is a structural mismatch for a framework whose whole point is parallel fan-out. NDJSON's file-per-trace + sidecar `messages/` model has zero contention.

5. **OTel as the only export channel (no NDJSON).**
   Rejected because vendor outages would lose the trace; LangSmith rate limits during incidents would silently drop spans; offline development needs a local store. NDJSON is the durable record; OTel is the push channel for vendor viewers.

6. **Standalone `AzureOpenAIAdapter` with reasoning-model quirk translation.**
   Initially built (~225 lines) when targeting the older `<resource>.openai.azure.com` product, which routes through `/chat/completions` and rejects `max_tokens` / `temperature` on reasoning models. **Reverted** once we tested against the newer Cognitive Services / AI Foundry product (`<resource>.cognitiveservices.azure.com`), which exposes `/openai/v1/responses` directly, accepts Bearer auth, and handles reasoning-model parameters through the standard Responses API shape. Azure now works as a pure config choice on the v1 endpoint with stock `OpenAIAdapter`. The adapter design is preserved in the decision log of the merge notes if a user reports the older endpoint doesn't work.

7. **Indexed `gen_ai.completion.{n}.*` attributes for the response panel.**
   Rejected per [§D8](#d8-indexed-gen_ai-prompt--single-blob-completion-langsmith-pragmatics): LangSmith renders an empty output panel for content=null+tool_calls when indexed completions exist; the single-blob form populates correctly.

## Verification

```text
PYTHONPATH=src python -m pytest tests/coordination/tracing/ \
    --ignore=tests/coordination/tracing/test_streaming_integration.py
```

Latest result: **133 passed, 1 pre-existing Python 3.14 failure** (`test_reader_opens_under_writer_lock_windows`, unrelated to this work).

- 30 new tests: `test_capture_helper.py` (9), `test_collector_llm_handlers.py` (8), `test_otel_writer.py` (14 — construction guards, deterministic parent linking under streaming order, `gen_ai.*` mapping per kind, per-message events, `gen_ai.choice` payload, error status, lifecycle isolation, strict-output-shape invariant).
- 99 pre-existing tracing tests still green.

Live verification: `live_tests/tracing/secret_word_pipeline.py` exercises Phase-1 + Phase-2 invariants on a real LLM run; `live_tests/tracing/deep_research.py` stress-tests AgentPool fan-out, compaction spans, and OTel batch flush.

## References

- Source: `packages/framework/src/marsys/coordination/tracing/capture.py` (~200 LoC).
- Source: `packages/framework/src/marsys/coordination/tracing/trace_context.py` (~45 LoC).
- Source: `packages/framework/src/marsys/coordination/tracing/writers/otel_writer.py` (~700 LoC).
- Source: `packages/framework/src/marsys/coordination/tracing/collector.py` (`_handle_llm_request`, `_handle_llm_response`, `_handle_branch_created`).
- Wire-up: `packages/framework/src/marsys/models/models.py` (three wrapped `arun` methods); `packages/framework/src/marsys/coordination/execution/step_executor.py` (`TraceContext` build); `packages/framework/src/marsys/agents/agents.py`, `agents/browser_agent.py`, `agents/memory_strategies.py`, `agents/memory.py` (forwarding).
- Tests: `packages/framework/tests/coordination/tracing/test_capture_helper.py`, `test_collector_llm_handlers.py`, `test_otel_writer.py`.
- Live tests: `packages/framework/live_tests/tracing/secret_word_pipeline.py`, `deep_research.py`.
- Streaming pipeline: [ADR (none — see commits `12447f7`, `d21c618`, `ddf68a8`)].
- Span tree: [ADR-005](ADR-005-unified-barrier-algorithm.md) (defines `step` / `branch` / `execution` spans this ADR adds children to).
- CHANGELOG entries: [packages/framework/CHANGELOG.md](../../../../packages/framework/CHANGELOG.md) `[Unreleased]`.

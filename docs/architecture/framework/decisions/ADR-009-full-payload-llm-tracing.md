# ADR-009: Full-payload LLM trace capture & vendor-neutral OTel export

**Status**: Accepted
**Date**: 2026-06-04
**Implements**: Framework tracing — full-payload LLM capture at the provider-adapter layer (one event per attempt), built on the streaming NDJSON span pipeline.
**Related**: ADR-005 (unified-barrier algorithm — defines the `execution` / `branch` / `step` span tree this ADR populates with `generation` / `compaction` / `tool` children).

## Context

The streaming NDJSON tracing pipeline records `execution / branch / step / generation / tool / compaction` spans, but a generation span on its own carries only metadata: model name, provider, finish reason, token counts. The information that makes a trace *debuggable* — the input message list, advertised tool schemas, sampling parameters, response content, thinking, structured tool_calls, response metadata — is the LLM payload itself. With only metadata you can confirm a model call happened; you cannot tell why it produced what it did. LLM-observability backends are built around the full payload.

This ADR covers capturing that payload faithfully and shipping it over a transport not tied to any one product. Three questions drive it:

1. **Where in the call chain do we capture the payload?** (§D1)
2. **How do we propagate per-step correlation context — step span id, branch id, agent name, event bus — down to the capture point?** (§D2)
3. **What on-disk and remote storage formats do we use?** (§D7)

## Decision

### D1. Capture inside the provider adapter — recording, not reconstruction, one event per attempt

We record the literal request the adapter is about to send to (and the response it got back from) the provider, emitting directly from inside the adapter's `arun`/`_arun_standard`, rather than reconstructing the call at the step layer or wrapping the whole call once at the model-wrapper layer. Reconstruction is input-only by construction (the response hasn't happened yet) and drifts from the wire whenever a later layer mutates the call — system-prompt prepending, format processors injecting steering tools, retry rewrites, compaction. The trace and the wire silently desynchronize.

The provider adapter is the right layer because it is where:

- the request payload is finalized (`format_request_payload` has run),
- sampling parameters are settled,
- **the retry loop actually sits** (`AsyncBaseAPIAdapter._arun_standard`), so each attempt — a retried 5xx/429 or the final success — records its *own* event with its own timing, rather than the model wrapper collapsing the whole retry sequence into a single record. (`LLMCallEvent.request_id` is unique per attempt, which is what makes a retried call legible in the trace.)
- the success / error outcome of each attempt is known precisely (success after harmonization, error in the retry branches and the terminal `except`).

Tool-internal LLM calls (`web_tools.py`, the `BrowserAgent` vision call) and memory compaction (`SummarizationProcessor._run_compaction_model`) all flow through a model wrapper into an adapter, so they are captured too — the model wrappers forward `trace_ctx` down (§D2) and the adapter emits. A step-level reconstruction misses all of these entirely.

The accepted cost: the clean coordination/transport separation is broken — `models.adapters.base` and `models.adapters.local` lazy-import `coordination.tracing.capture`. (Previously the coupling sat on `models.models`; moving the emit into the adapter moves the import with it.) Every alternative we evaluated either misses call paths, introduces silent drift, or cannot see individual retry attempts.

**Async-only, by construction.** `emit_llm_call` is a coroutine, so capture lives only on the async paths (`AsyncBaseAPIAdapter._arun_standard` / `arun_streaming`, and each local adapter's `arun`). The synchronous `_run_standard` is not traced — and neither was it under the old wrapper design in any way that mattered. The one behavioural gap this opens is `BaseAPIModel.arun`'s `run_in_executor` fallback, taken only when a provider has no `Async*` adapter at all (none of the shipped providers); that path runs the sync adapter and is untraced. Documented in Consequences.

### D2. Explicit `TraceContext` kwarg threading — no ContextVars

The convenient approach is a set of `ContextVar`s (`_current_step_span_id`, `_current_branch_id`, `_current_agent_name`, `_current_session_id`, `_current_event_bus`, `_current_call_kind`, `_in_capture`) read off the running task so capture needs no signature changes. We reject it for three reasons:

- **Lifecycle on the honor system.** `set` with no matching `reset` leaks state across tests and between concurrent runs; the mitigation (clearing in a `finally`) itself relies on every call site doing the right thing.
- **Test ergonomics.** Any test touching the model/adapter layer would need fixtures resetting every var, or pytest-async's task-scoping bites.
- **Greppability.** A named `TraceContext` field on `context` / `run_context` / `model_kwargs` is visible at every coordination hop; a `ContextVar` read at the bottom is not.

(The `run_in_executor` thread-pool boundary, which `ContextVar`s do not cross, would have been the disqualifying reason under the old wrapper design — but capture now lives in the *async* adapter, on the same task as its caller, so `ContextVar`s would in fact have propagated on the traced path. The only executor hop left, `BaseAPIModel.arun`'s sync fallback, is untraced regardless (§D1). So the rejection now rests on the three reasons above, not on the executor boundary.)

We thread a single `TraceContext` frozen dataclass explicitly instead. Crucially, the threading does not stop at the model wrapper: the wrapper forwards it one hop further into the adapter, where the emit happens:

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
                   model.arun(messages, ..., trace_ctx=trace_ctx)   # wrapper forwards, does not emit
                                  │
                                  ▼
              adapter.arun(messages, ..., trace_ctx=trace_ctx)      # pops it, emits per attempt
                                  │
                                  ▼
              emit_llm_call(trace_ctx, ..., response= | error= | error_type=)
```

`Branch` already carried `step_span_id` through `context["step_span_id"]`, so this extends an existing pattern rather than inventing one. Compaction calls derive `trace_ctx.child(kind="compaction")` so they land as compaction-flavored spans without a separate plumbing path.

**The implicit-kwargs caveat — what "explicit" does and does not buy.** The chain is explicit and greppable in the *coordination* layers: `TraceContext` is a named field on `context` / `run_context` / `model_kwargs`. It is **not** advertised on any public `arun` signature — `trace_ctx` rides into the model wrappers as a `**kwargs` key, and from there into the adapters the same way. `BaseAPIModel.arun` pops it and forwards it explicitly to `async_adapter.arun(trace_ctx=…)`; `BaseLocalModel.arun` and `PeftHead.arun` forward it untouched in `**kwargs`; the terminal pop-and-emit happens in the adapter (`AsyncBaseAPIAdapter.arun` / `_arun_standard`, and each local adapter's `arun`). The compaction path reads it via `runtime.get("trace_ctx")`. So no public `arun` signature gains a `trace_ctx` parameter, and correct propagation is a **load-bearing convention, not a type-enforced contract**: every wrapper must forward it and every adapter must pop and emit, and nothing in code checks that. We accept that over a typed parameter to avoid changing the TRUNK-CRITICAL `arun` signatures — but it means the "no implicit data flow" advantage is *partial*: the coordination chain is visible; the two final hops (into the wrapper, then into the adapter) are implicit kwargs keys held in place by convention.

**Re-entrancy guard — retained, currently dormant.** `TraceContext.captured` (and `mark_captured()`) exist to prevent double-emit when one emitter delegates to another: an outer emitter forwards `trace_ctx.mark_captured()` and the inner `emit_llm_call` sees `captured=True` and bypasses. With emission consolidated at the single adapter layer, no path currently nests two emitters — `PeftHead.arun` and `BaseLocalModel.arun` each delegate to exactly one adapter `arun`, which is the sole emitter, so they forward a plain `trace_ctx` and the guard is not exercised. It is kept as forward-looking insurance for any future wrapper that wraps another emitting wrapper.

### D3. Direct emits at each outcome point — not a decorator, not a context manager

The adapter builds the per-call input snapshot once (`_trace_call_base`: model/provider identity, `list(messages)`, tools, images, sampling params) and then calls `emit_llm_call` **directly at each outcome point** in the retry loop, passing a fresh per-attempt `start` each time:

```text
trace_base = self._trace_call_base(messages, **kwargs)        # snapshot once
for attempt in range(max_retries + 1):
    attempt_start = time.time()
    try:
        ... POST ...
        if 5xx and attempt < max:  emit(error_type="ServerError");  sleep; continue
        if 429 and attempt < max:  emit(error_type="RateLimitError"); sleep; continue
        harmonized = harmonize_response(...)
        emit(response=harmonized);  return harmonized              # success
    except (ClientError | Exception) as e:                          # terminal
        result = handle_api_error(e, ...)                           # ErrorResponse | recovered
        emit(error=result) if isinstance(result, ErrorResponse) else emit(response=result)
        return result
```

`emit_llm_call` is the only helper — it builds, populates, and emits one `LLMCallEvent`, choosing `status` from its error inputs (`ok` / `error` / `cancelled`). It accepts an error three ways so a single helper serves every point: a raised exception (`error=exc`), an `ErrorResponse`-like carrier the adapter *returned* (`error=err_response`, recording the provider-native `error_type`/`error`), or an explicit `error_type`/`error_message` pair for a retry branch that has only a status code and no object. The local adapters (single attempt, no retry loop) use the same shape with one `try`/`except` around the `to_thread` call.

A **decorator** was rejected — and the rejection was reaffirmed during implementation — because it hides the emit from the call site, needs introspection to forward kwargs across adapters with different `arun` signatures, and cannot place an emit *inside* the retry loop at each distinct attempt outcome (its whole premise is to wrap the call as one unit). A **context manager** was rejected because there is nothing to "open" at entry: the input snapshot is one local and each emit is terminal for its attempt; a `try`/`except` with explicit success/failure points expresses that more directly than `__aexit__`. There is deliberately no separate "begin" call.

### D4. One self-contained `LLMCallEvent` per attempt, on the existing event bus

Capture emits one self-contained `LLMCallEvent` per *attempt* — each carrying the full input snapshot, the outcome (response or error), and per-attempt timing — on the `AsyncEventBus` already wired into `Orchestra`. A call that retried twice and then succeeded yields three events (two `status="error"`, one `status="ok"`), each with its own `request_id`. The collector's `_handle_llm_call` builds, populates, and closes a generation/compaction span in a single handler and streams it via `_stream_span`; it already treats each event independently (opens and closes a fresh span, appends it to the step's children), so N attempts simply produce N generation spans under the step — no collector change was needed. No new sink is needed either — `NDJSONTraceWriter` and any user-supplied `TelemetrySink` (including `OtelTraceWriter`) already fan out from `_stream_span`.

Self-contained per-attempt events rather than an open/close request-response pair, because:

- **Less downstream state.** No request↔response correlation anywhere: no `request_id → open span` map in the collector, no correlation dict in the AG-UI translator. Each event is self-describing for every consumer, removing a class of in-memory state that must be cleaned up.
- **Faithful input capture preserved.** The adapter snapshots the messages list before the request (`list(messages)`) into `trace_base` and reuses that snapshot for every attempt's emit, so an adapter or retry that mutates the list in place cannot pollute the recorded input — the recording-not-reconstruction property of §D1 holds, and every attempt records the same full input.
- **Retries are legible.** Because each attempt emits, a trace shows the rewritten/retried calls as distinct spans rather than one opaque record — the original motivation for capturing where the retry loop lives (§D1).
- **Inputs survive failures.** Each failure emit carries the captured inputs, so the prompt that triggered a failure is still recorded.

**Accepted trade-off.** A given attempt whose emit cannot complete leaves no record *for that attempt* (earlier attempts are already recorded). The sharp case is cancellation during event-loop shutdown — `emit_llm_call`'s cancel path is best-effort (`asyncio.shield` + swallow). A hard kill loses any in-memory state anyway, and the surviving window is narrow. A consequence is that the collector needs no open-LLM-span finalization backstop. Liveness ("a call is in flight") is not provided by this layer; derive it at the orchestration layer from step/agent events if needed.

### D5. Inputs and tool schemas: content-addressed, ref-only on the span

Full-input capture is **always on when tracing is enabled** — there is no opt-in flag. When the collector is constructed (only when `TracingConfig.enabled`), it builds a `MessageStore` (default `FilesystemMessageStore` under `output_dir/messages/`, or a user-supplied override). Input message lists and tool schemas are hashed into the store and the span carries **only** a content-addressed reference, never an inline copy:

- **Input messages** → `input_messages_ref`: the ordered list of per-message content hashes (`history`), plus an optional `base`/`patch` diff against a prior resolved history (used on the step-span path for append-only sharing; the per-call generation span uses a flat history). Each unique message is stored once, so an append-only branch and its forks share their common prefix.
- **Tool schemas** (30 KB+ for a full coordination-tools catalog, rarely changing between steps) → `tools_ref`, hashed through the same store via a synthetic `[{"role": "tool_schema", "content": tools}]` envelope, so a tool catalogue lands once across all steps.

Identical content deduplicates across every trace sharing an `output_dir`. This keeps the on-disk NDJSON — the source of truth — compact: a span carries hashes, not payloads.

**Sinks that cannot follow a content-addressed pointer rehydrate at publish time.** The collector hands its store to each sink once, via `TelemetrySink.bind_message_store(store)` (a no-op by default). `OtelTraceWriter` overrides it: OTLP can't carry a CAS pointer, so the writer resolves `input_messages_ref` / `tools_ref` to literal content via `store.reconstruct(ref)` when building its attributes. Ref-aware consumers (the NDJSON writer, post-mortem `TraceTree.from_ndjson`) persist the ref verbatim and resolve lazily. Net result: one shape on the span (the ref), and each sink renders the shape it needs without the collector encoding any sink's format. (A defensive inline fallback exists if a collector is ever constructed with no store.)

The removed `TracingConfig.capture_full_input` opt-in flag is what this design replaces: streaming NDJSON, content-addressed dedup, and ref-only spans resolved the disk-overhead concern it once guarded, and `SecretRedactor` (applied once at `_stream_span`) covers privacy. The flag never gated capture (only whether the store was built) and was deleted — capture + dedup are now unconditional when tracing is enabled.

### D6. Response payload kept inline

Response content / thinking / tool_calls stay inline on span attributes, not routed through `MessageStore`. Responses are typically modest and almost always unique per call; a second hashing surface doesn't pay for itself. Promote to the store later if NDJSON line sizes become a problem.

### D7. Keep NDJSON default, add an opt-in OTel sink, no SQLite

NDJSON vs SQLite as the local store, on the two axes that matter for this framework:

- **Compactness** — wash. Both dedup heavy content (NDJSON via the sidecar `messages/` directory; SQLite via a `content` table). Zipped NDJSON is usually smaller than the equivalent SQLite file.
- **Concurrency** — NDJSON wins decisively. File-per-trace + content-addressed sidecar means parallel branches and parallel `Orchestra.run()` invocations contend on **nothing**. SQLite's single-writer lock queues every parallel branch's span emission — a structural mismatch for a framework whose whole point is fan-out.

Decision:

- **NDJSON is the default** local store, wired in `Orchestra.__init__`.
- **`OtelTraceWriter` is an opt-in `TelemetrySink`** under the `tracing-otel` extra, exporting over OTLP/HTTP using `gen_ai.*` semconv plus cross-vendor attribute surfaces (§D8), so one configuration works against any OTLP-aware backend.
- **No SQLite writer.** No cross-run-SQL use case; OTLP export covers remote querying.

Storage is two-layer: NDJSON locally for "what happened on this machine," an OTLP backend remotely for "let me actually look at it." NDJSON remains the durable record and is unaffected by backend outages.

**Private-SDK-API dependency and its version cap.** `OtelTraceWriter._emit_span` constructs SDK spans through underscore-private internals — `opentelemetry.sdk.trace._Span`, `TracerProvider._active_span_processor`, `Tracer._instrumentation_scope` — to inject the deterministic `blake2b(ulid)`→OTel-ID mapping. This is load-bearing: `tracer.start_span()` auto-allocates a random `span_id`, but children close and stream *before* their parents in the streaming close-order, so they must derive `parent.span_id` deterministically, and the public API offers no way to set a chosen ID. The cost is fragility — a future SDK refactor of those internals would break export, and because `publish_span` logs-and-swallows, it would break *silently* (NDJSON keeps working and masks it). Mitigation: the `tracing-otel` extra is pinned `>=1.27.0,<2.0` (api/sdk/exporter ship lockstep on the 1.x train) so a major SDK bump — the likely site of such a refactor — is a deliberate bump-and-verify, not a silent transitive upgrade. **Migration trigger:** if upstream adds a supported way to construct a span with a caller-chosen `span_id` (or a public `IdGenerator` seam that can map per-span), switch to it, delete the private-symbol imports, and raise the cap. Until then, treat any `opentelemetry-sdk` upgrade as gated on a green `test_otel_writer.py` (which asserts deterministic parent linking).

### D8. Vendor-neutral attribute mapping — multiple standard surfaces, no product-specific keys

The exporter emits **only published, cross-vendor conventions** and no attribute keyed to a single product's namespace. The GenAI-semconv *content* layer (prompts/completions) is still in flux and no two backends read the same convention as primary, so we emit the small union of standard surfaces and let each backend pick the one it understands:

- **OTel `gen_ai.*` semconv** — operation name, model, sampling params, token usage, finish reason; indexed `gen_ai.prompt.{i}.*` and the single-blob `gen_ai.prompt` / `gen_ai.completion`; tool definitions as the single JSON-string array `gen_ai.tool.definitions` (the semconv shape — there is no `gen_ai.request.tools.{i}.*` attribute).
- **OpenInference** — `input.value` / `output.value` (+ mime types), `llm.input_messages.{i}.message.*`, `llm.tools.{i}.tool.json_schema` (the available-tools panel), and `openinference.span.kind` (LLM / TOOL / CHAIN). The semconv has no stable span-typing attribute yet, so this cross-vendor key fills the gap.
- **OpenLLMetry** — `llm.request.functions.{i}.*` for tool definitions.

Specific neutrality decisions, each verified against real backend UIs:

- **No product-specific span-kind badge.** Span typing uses the cross-vendor `openinference.span.kind`; backends that ignore a surface read another, so data is never lost.
- **No `gen_ai.{role}.message` / `gen_ai.choice` events.** Some backends rank those events above the cleaner attribute surfaces, and span-event attributes can't carry a structured message/tool-call payload — they render as raw JSON. Prompt/completion content ships via attributes only.
- **Tool calls in the OpenAI wire shape.** Indexed `gen_ai.prompt.{i}.tool_calls.{j}.*` use `id` + `type` + nested `function.name` / `function.arguments`, arguments kept as a JSON *string* (never double-encoded). The nested layout is what OTel ingestions map to tool-call chips; a flat `.name` / `.arguments` leaves the chip unrendered. This matches the OpenInference emission, so it's neutral.
- **Single-blob completion, not indexed.** We ship `gen_ai.completion` as one blob and skip indexed `gen_ai.completion.{n}.*`: some renderers ignore the blob once the indexed form exists and then show an empty output panel for the `content=null + tool_calls` shape (common for tool-using agents). The output dict must contain only `{role, content, tool_calls, thinking}` — adding `finish_reason` / `usage` can flip a renderer to a raw-fields panel; those ship separately on `gen_ai.response.finish_reasons` and `gen_ai.usage.{input,output}_tokens`. A regression test asserts the key set never grows.
- **Null content represented as `""`.** A tool-call-only assistant turn has `content: null` in the OpenAI shape, but OTel scalar attributes cannot be null. Omitting `gen_ai.prompt.{i}.content` can leave a backend unable to classify the turn, so null is coerced to `""`.

**Accepted backend-rendering limitations (do NOT close them by adding product-specific keys):**

1. Some backends render their rich *generation* chat view only from their own native input/output attribute, not from the standard `gen_ai.*` / OpenInference keys. On those, our neutral attributes still populate the observation's input/output completely (all data present and queryable); the platform may just show it as raw attributes rather than chat bubbles.
2. Some backends do not render tool-call chips for assistant turns embedded in the *input history* (only for the live output), regardless of which standard convention carries them.

Both are inherent to how those backends ingest OTLP, not gaps in what we emit. Neutrality was chosen over emitting any product-specific attribute.

**`TelemetrySink` boundary — why a concrete writer ships in-framework.** `sink.py` states the framework "knows nothing about specific backends" and that vendor adapters live outside it. `OtelTraceWriter` is consistent with that: because it emits *only* the cross-vendor surfaces above, it is a vendor-*neutral* sink — the OTLP analogue of the bundled `NDJSONTraceWriter`, not a vendor adapter. Vendor-*specific* concerns (a product's endpoint, auth headers, env-var wiring, or any single-product attribute key) stay on the caller's side: the live test `secret_word_pipeline.py` builds LangSmith / Langfuse presets on top of the generic writer, and a published preset belongs in a dedicated package (`marsys-langsmith`, `marsys-langfuse`, or a downstream consumer's own telemetry package) — never in `src/marsys/`.

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
                          model.arun(messages, ..., trace_ctx=...)   wrapper: forward only
                                            │
                                            ▼
                          adapter.arun(messages, ..., trace_ctx=...)
                                            │
                       ┌────────────────────┴────────────────────┐
                       │ trace_base = snapshot(msgs, tools, ...)  │  snapshot inputs once
                       │ for attempt in 0..max_retries:           │
                       │   t = time.time()                        │
                       │   ... POST ...                           │
                       │   5xx/429 & retrying → emit(error_type=) │ ─▶ EventBus.emit(
                       │   success            → emit(response=r)  │      LLMCallEvent)
                       │   terminal except    → emit(error=…)     │   one self-contained
                       └────────────────────┬────────────────────┘   event PER ATTEMPT:
                                            │                        inputs + outcome +
                                            ▼                        per-attempt timing
                                    HarmonizedResponse | ErrorResponse
                                            │
                            ┌───────────────┴────────────┐
                            │ TraceCollector             │
                            │  _handle_llm_call:         │  (one span per event →
                            │   build generation span,   │   N attempts = N spans
                            │   route msgs + tools       │   under the step)
                            │   through MessageStore,     │
                            │   attach output, close,     │
                            │   _stream_span()            │
                            └───────────────┬────────────┘
                                            │  SecretRedactor runs once here
                                            ▼
                                    TelemetrySink fan-out
                                    (NDJSONTraceWriter + any user sinks,
                                     including OtelTraceWriter)
```

### Trace-tree shape: parallel children under their dispatching step

A `branch` span nests under the `step` that dispatched it, not flat under the `execution` root — otherwise parallel fan-outs are spatially indistinguishable from sequential workflows in a backend UI. `BranchCreatedEvent.parent_step_span_id` (additive, optional) is threaded:

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
TraceCollector._handle_branch_created → parent = step_spans[parent_step_span_id] or trace.root_span
```

Entry / initial branches fall back to the execution root (correct for that case). Orchestrators that don't populate the field still produce valid traces; only the nested rendering degrades.

## Adapter correctness for faithful traces

Three adapter-level behaviors are part of this work because they directly determine what the trace records:

1. **`OpenAIAdapter.harmonize_response` defaults `content` to `None`, not `""`.** OpenAI's convention — and what chat-UI renderers expect in the assistant bubble for tool-only responses — is `content: null` when the response is purely tool calls; an empty string flips some renderers to a raw-fields panel. `content` only becomes a string when a `message` item is present in the Responses-API output array.
2. **Token-usage parsing tries Responses-API names first** (`input_tokens` / `output_tokens` / `output_tokens_details.reasoning_tokens`) and falls back to chat-completions names (`prompt_tokens` / `completion_tokens` / `reasoning_tokens`). The Responses API moved reasoning-token accounting into a nested block; reading the flat names alone yields `null`.
3. **Provider is a stored identity attribute, propagated onto the adapter that emits.** `BaseAPIModel.provider` / `model_name` are set in `__init__` (a `@property` deriving the lowercased adapter class name produced non-canonical ids like `"asyncopenrouter"` that broke equality with the configured provider). Because the emit now happens *inside* the adapter, the model wrappers also stamp the canonical provider onto the adapter instance they create: `BaseAPIModel.__init__` sets `adapter.provider` and `async_adapter.provider`; `BaseLocalModel.__init__` sets `adapter.provider` from its `backend` (`"huggingface"` / `"vllm"`); `PeftHead.__init__` sets `self.model.provider = "huggingface"`. The adapter's `_trace_call_base` reads that attribute (falling back to `_provider_name()` / `backend` for standalone adapter use), so generation spans carry a real provider id rather than the class-name-derived one. Setting `adapter.provider` additionally fixes the long-standing `getattr(self, "provider", "unknown")` fallback in the adapter's own `ErrorResponse` construction.

## Consequences

- **Positive:**
  - Generation spans carry the full LLM payload — debugging matches the wire by definition.
  - Retried calls are legible: each attempt is its own generation span with its own timing and `request_id`, instead of one opaque record per call.
  - Compaction and tool-internal LLM calls (web_tools, browser-agent vision) land in the trace under the right step; previously invisible.
  - One trace pipeline serves both local NDJSON and any OTLP backend; NDJSON is the source of truth and is unaffected by backend outages.
  - The export is strictly vendor-neutral: rich LLM rendering on platforms that read any emitted standard surface, complete data on those that read only the base `gen_ai.*` keys, no lock-in.
  - Adding a new provider adapter inherits capture for free (the emits live in the shared `AsyncBaseAPIAdapter._arun_standard`); a new local adapter follows a fixed pattern (snapshot + one `try`/`except` around the `to_thread` call); adding a new backend is a fixed pattern (`TelemetrySink` subclass).
  - Capture + dedup are automatic when tracing is enabled — no flag to forget, and the on-disk record stays deduped by default.

- **Negative:**
  - `marsys.models.adapters.base` / `…adapters.local` lazy-import `marsys.coordination.tracing.capture`. The import is genuinely conditional (it fires only when an `arun` runs under a traced session), but the adapter layer is no longer a leaf module from tracing's perspective. (The coupling moved here from `models.models`, where it sat under the old wrapper design — it did not appear from nothing.)
  - The emit boilerplate now lives in the adapter retry loop: ~4 emit sites in `_arun_standard` plus a terminal one for the streaming path, and one `try`/`except` in each of the three local adapters. The three model wrappers became thin (they only forward `trace_ctx`).
  - A retried call produces several generation spans under one step rather than one. This is the intended visibility win, but consumers that assumed "one generation span per step" must handle N.
  - `BaseAPIModel.arun`'s `run_in_executor` sync fallback — reachable only for a provider with no `Async*` adapter (none ship today) — is untraced, because `emit_llm_call` is async and the emit now lives in the (skipped) async adapter. The old wrapper design happened to wrap that branch too; this is a narrow regression on an unreachable path.
  - The OTel mapping emits several parallel attribute surfaces because no single backend reads only one consistently. Each surface is testable in isolation; the cost is mostly comment volume.
  - The strict-output-shape invariant in `_build_output_message` is enforced by a regression test, not a type. The test is the gate against a future contributor adding `usage`/`finish_reason` and regressing shape-sensitive renderers.
  - `trace_ctx` propagation through `**kwargs` is convention, not type-enforced (§D2), and now spans two hops (wrapper-forward, adapter-pop). A new wrapper that forgets to forward it, or a new adapter that forgets to pop-and-emit, silently drops capture for that path.

- **Risk:**
  - Backend rendering rules are documented by observation, not formal spec. If a backend changes its ingestion, the chat-bubble pragmatics in `_emit_indexed_prompt_attrs` may need to follow. Mitigation: live-test scripts under `live_tests/tracing/` exercise the path against real backends.
  - The deterministic `ulid → blake2b → fixed-byte int` mapping has a theoretical collision rate of ~2⁻⁶⁴ for span ids, ~2⁻¹²⁸ for trace ids. Not a practical concern at our volumes; if they grow, switch to a 128-bit span id.
  - `OtelTraceWriter` depends on private OTel SDK internals (§D7). Mitigated by the `<2.0` cap and a green-`test_otel_writer.py` gate on any SDK bump.

## Alternatives Rejected

1. **Reconstruction at the step layer instead of recording in the adapter.** Rejected per §D1: drifts from the wire when format processors / retry rewrites mutate messages between assembly and POST, and misses tool-internal and compaction calls entirely.
2. **One emit at the model-wrapper layer wrapping the whole adapter call** (the previous design). Rejected per §D1/§D4: it sits *above* the retry loop, so it collapses every retry into a single record and can never show the individual attempts — the precise thing that makes a flaky/rate-limited call debuggable. Moving the emit into the adapter, where the loop lives, is what buys per-attempt visibility.
3. **`ContextVar`-based propagation.** Rejected per §D2: lifecycle leaks, test fragility, and lost greppability. (The `run_in_executor` boundary, once the disqualifier, no longer applies now that capture is on the async adapter task — but the other reasons stand.)
4. **A `@capture_arun` decorator on each adapter's `arun`.** Rejected per §D3 and reaffirmed during implementation: it hides the emit, needs introspection to forward kwargs across adapters with different signatures, and — fatally — cannot emit *inside* the retry loop at each per-attempt outcome, since its premise is to wrap the call as one unit.
5. **An open/close `LLMRequestEvent` + `LLMResponseEvent` pair.** Rejected per §D4: forces request↔response correlation state into every consumer for no benefit the self-contained per-attempt event doesn't already provide.
6. **SQLite as the primary on-disk store.** Rejected per §D7: single-writer lock contention is a structural mismatch for parallel fan-out; NDJSON's file-per-trace + sidecar `messages/` has zero contention.
7. **OTel as the only export channel (no NDJSON).** Rejected: backend outages would lose the trace, rate limits during incidents would silently drop spans, and offline development needs a local store. NDJSON is durable; OTLP is the push channel.
8. **Inline payload on every span (no content-addressed ref).** Rejected per §D5: defeats the dedup the `MessageStore` exists to provide and bloats the NDJSON source of truth. Ref-only on the span + sink-side rehydration gives both compact storage and full content where a sink needs it.
9. **Product-specific attributes/events to force a particular backend's rich rendering.** Rejected per §D8: the two accepted rendering limitations could be closed by emitting one product's native input/output attribute, but that reintroduces vendor lock-in into a module whose explicit goal is neutrality. The complete data is present regardless; only some visual polish on specific backends is forgone.

## Verification

```text
PYTHONPATH=src python -m pytest tests/coordination/tracing/ \
    --ignore=tests/coordination/tracing/test_streaming_integration.py
```

The tracing suite covers:

- `test_capture_helper.py`, `test_collector_llm_handlers.py` — the `emit_llm_call` path and its three error shapes (exception, returned `ErrorResponse`, explicit `error_type`/`error_message`), snapshot-at-call-start, error/cancel input preservation, ref-only generation-span shaping, and the no-store inline fallback.
- `test_adapter_emission.py` — per-attempt emission driven through `AsyncBaseAPIAdapter.arun` with a fake aiohttp session: a 500→200 retry emits one `status="error"` then one `status="ok"` (full input on both), a handled 4xx emits one error event with the provider-native `error_type`, and absent `trace_ctx` emits nothing.
- `test_otel_writer.py` — construction guards, deterministic parent linking under streaming close-order, `gen_ai.*` mapping per kind, the OpenAI-wire tool-call shape, the single-blob completion + strict-output-shape invariant, the regression lock against `gen_ai.{role}.message` / `gen_ai.choice` events and any product-specific span-kind key, ref rehydration via a bound `MessageStore`, and error/exception status mapping.
- `test_messages.py` — `MessageStore` round-trip, cross-trace dedup, diff-history, and always-on capture (no opt-in flag).
- `test_multi_consumer.py` — the `TelemetrySink` protocol across differently-shaped adapters in parallel, with secrets redacted in every view.

Live verification: `live_tests/tracing/secret_word_pipeline.py` exercises the capture + export invariants on a real LLM run (with `--langsmith` / `--langfuse` presets for two OTLP backends); `live_tests/tracing/deep_research.py` stress-tests `AgentPool` fan-out, compaction spans, and OTel batch flush; `live_tests/tracing/full_input_capture.py` verifies content-addressed capture end-to-end against a real model.

## References

- Capture (emit helper): `coordination/tracing/capture.py` (`emit_llm_call`, `extract_sampling_params`), `coordination/tracing/trace_context.py` (`TraceContext`).
- Emit sites (adapter layer): `models/adapters/base.py` (`AsyncBaseAPIAdapter.arun` / `_arun_standard` / `_trace_call_base`), `models/adapters/local.py` (`_local_trace_base` + the three `arun` methods).
- Events / collector: `coordination/tracing/events.py` (`LLMCallEvent`), `coordination/tracing/collector.py` (`_handle_llm_call`, `_handle_branch_created`, `_build_message_store`).
- Storage / dedup: `coordination/tracing/messages.py` (`MessageStore`, `build_input_messages_ref`), `coordination/tracing/sink.py` (`TelemetrySink.bind_message_store`).
- Export: `coordination/tracing/writers/otel_writer.py`, `coordination/tracing/writers/ndjson_writer.py`.
- Config: `coordination/tracing/config.py` (`TracingConfig`; capture + dedup are unconditional when enabled — no opt-in flag).
- Wire-up / forwarding: `models/models.py` (the three `arun` wrappers forward `trace_ctx`; `__init__` stamps `adapter.provider`), `coordination/execution/step_executor.py` (`TraceContext` build), `agents/agents.py`, `agents/browser_agent.py`, `agents/memory_strategies.py`, `agents/memory.py` (forwarding).
- Tests: `tests/coordination/tracing/{test_capture_helper,test_adapter_emission,test_collector_llm_handlers,test_otel_writer,test_messages,test_multi_consumer}.py`.
- Live tests: `live_tests/tracing/{secret_word_pipeline,deep_research,full_input_capture}.py`.
- Span tree: [ADR-005](ADR-005-unified-barrier-algorithm.md) (defines the `step` / `branch` / `execution` spans this ADR adds children to).
- CHANGELOG: [packages/framework/CHANGELOG.md](../../../../packages/framework/CHANGELOG.md) (records the `capture_full_input` / `include_generation_details` removals).

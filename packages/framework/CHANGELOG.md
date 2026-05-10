# Changelog

All notable changes to MARSYS (Multi-Agent Reasoning Systems) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **`TelemetrySink` ABC** (`packages/framework/src/marsys/coordination/tracing/sink.py`): generic seam for forwarding closed spans to external observability backends (Spren daemon, LangSmith, Phoenix, Langfuse, MARSYS Cloud, custom HTTP). Two abstract async methods: `publish_span(span)` called once per span close, `close()` called once at run end. Adapters live outside the framework as third-party packages and translate the framework's `Span` shape to whatever vendor API they target. Errors are caught + logged at the `TraceCollector` boundary; one bad sink does not stop the run or block other sinks.
- **`SecretRedactor`** (`coordination/tracing/redactor.py`): scrubs known-secret keys from span attribute payloads at the `TraceCollector._stream_span` chokepoint. Default deny-list (case-insensitive, word-boundary match): `api_key, apikey, token, authorization, auth, secret, password, bearer, cookie, session, credential`. Word boundaries treat `_`/`-`/non-alphanumeric chars as separators so `auth_token` redacts but `prompt_tokens` (an LLM token-count metric) does not. Walks `span.attributes`, every `event['attributes']` dict in `span.events`, every `link['attributes']` dict in `span.links`. Mutates in place — all consumers (NDJSON writer, vendor sinks, in-memory `TraceTree`) see the same redacted view. `NoRedactionRedactor` opt-out variant for callers that explicitly accept the leak risk.
- **`TracingConfig.sinks: list = []`** and **`TracingConfig.redactor: SecretRedactor | None = None`** fields. Sinks register alongside the default `NDJSONTraceWriter`; if `redactor` is None the default `SecretRedactor()` instantiates lazily inside `TraceCollector`. Zero new kwargs on `Orchestra` — sinks plumb through `ExecutionConfig.tracing`, eliminating any TRUNK-CRITICAL surface change.
- **Streaming NDJSON tracing writer** (`packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py`): one JSON object per closed span at `{TracingConfig.output_dir}/{trace_id}.ndjson`, written incrementally by `TraceCollector` as each span closes. Mid-run process crashes preserve every span emitted up to the crash; late subscribers tail-follow the file from start to current EOF. Bounded `asyncio.Queue` (default `maxsize=10000`) + dedicated drain task keeps `publish_span` non-blocking on the disk path; drop-oldest on overflow with a `dropped_span_count` metric. After 100 consecutive `OSError`s the writer self-disables (a `stream_event` warning line is emitted, subsequent spans drop into `disabled_dropped_count`). `fsync_per_span` is opt-in (default off).
- **Streaming NDJSON reader** (`coordination/tracing/readers/ndjson_reader.py`): generator yielding span dicts in file order, with `follow=True` tail-follow mode, truncated-trailing-line tolerance, and `completion_status` ∈ `{complete, truncated, crashed}`. Tolerates unknown top-level fields and unknown `attributes` keys (additive-only schema-evolution policy); rejects `schema_version > 1` with `NDJSONVersionError`.
- **`TraceTree.from_ndjson(path)`** classmethod for post-mortem reconstruction. Symmetric with `TraceTree.to_dict()`. Spans whose `parent_span_id` is unknown surface in a new `TraceTree.orphans: List[Span]` field rather than being silently dropped.
- **Per-span streaming hook** (`TelemetrySink.publish_span(span)`) — every closed span fans out from `TraceCollector._stream_span` to every registered sink. Replaces the previous `TraceWriter.write_span` virtual method; the new ABC requires every sink to implement `publish_span` (no virtual default).
- **`OrchestraResult.metadata["tracing"]`** exposes writer counts after `Orchestra.execute()` returns: `total_spans`, `disk_error_count`, `dropped_span_count`, `disabled_dropped_count`, `disabled`. Programmatic consumers (Cloud worker, CI) detect partial / disabled traces from this dict.
- **`coordination/tracing/_ids.py`** — single-source ULID factory (`new_id()`) used by `StatusEvent.event_id`, `Span.span_id`, and `TraceTree.trace_id`.
- **Full-payload LLM capture** at the model-wrapper layer (`coordination/tracing/capture.py`). New `capture_llm_call` async context manager wraps every `model.arun` call (`BaseLocalModel`, `BaseAPIModel` async + thread-fallback, `LearnableModelWrapper`) and emits a paired `LLMRequestEvent` / `LLMResponseEvent` carrying the literal kwargs the wrapper handed to the adapter — recording-not-reconstruction, so the trace matches the wire by definition. Bypasses cleanly outside `Orchestra` (no `trace_ctx`, no event bus, or already-captured frame). Re-entrancy guard via `TraceContext.captured` prevents inner-wrapper double-emit when one wrapper delegates to another (`LearnableModelWrapper` → `BaseLocalModel`). See ADR-007.
- **`TraceContext`** frozen dataclass (`coordination/tracing/trace_context.py`) threaded explicitly through `context["trace_ctx"]` → `run_context["trace_ctx"]` → `model_kwargs` → `model.arun(trace_ctx=…)`. Replaces the implicit-state ContextVars approach of the original ADR draft — the chain is visible in signatures, survives `run_in_executor` boundaries, and tests don't need fixtures resetting per-call vars. `TraceContext.child(kind="compaction")` derives a compaction-flavored context for `_run_compaction_model`.
- **`LLMRequestEvent`** / **`LLMResponseEvent`** (`coordination/tracing/events.py`). Carry the full payload — messages, advertised tool schemas, sampling params, content, thinking, reasoning, structured tool_calls, response metadata, error type/message — keyed by `request_id`. Collector handlers `_handle_llm_request` / `_handle_llm_response` open generation/compaction spans on the active step, route messages and tool schemas through `MessageStore` for content-addressed dedup, and stream the closed span through the existing `_stream_span` fan-out so all sinks see it identically.
- **`OtelTraceWriter`** (`coordination/tracing/writers/otel_writer.py`). `TelemetrySink` subclass that exports each closed MARSYS span as one OpenTelemetry span over OTLP/HTTP. Vendor-neutral — one configuration ships traces to LangSmith, Langfuse, Phoenix (Arize), Jaeger, Tempo, or Datadog. Per-span emission keeps memory bounded under the streaming model. Deterministic `blake2b(ulid, digest_size=…)` hashing maps 26-char ULIDs to 16-byte trace_ids and 8-byte span_ids so children arriving before parents (streaming close-order) still resolve `parent.span_id`. Lazy SDK imports — module is importable without the `tracing-otel` extra; instantiation is the only failure point. Production uses `BatchSpanProcessor`; tests inject `InMemorySpanExporter` via the `_exporter_override` seam. Maps each kind onto `gen_ai.*` semconv plus OpenInference attributes (`langsmith.span.kind`, `input.value` / `output.value`) and indexed prompt/tool/completion attributes for LangSmith's chat-bubble + tool-call-chip rendering.
- **Optional dependency `tracing-otel`** in `pyproject.toml`: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http` (each `>=1.27.0`). Without the extra, `OtelTraceWriter` is importable but raises `ImportError` on instantiation.
- **`BranchCreatedEvent.parent_step_span_id`** (`coordination/events.py`) — span id of the dispatching step. The orchestrator threads `StepResult.step_span_id` → `Branch.last_step_span_id` → the spawn call → `BranchCreatedEvent`, and the collector parents the new branch span under that step. Without this, parallel fan-out renders flat under the execution root in LangSmith. Both `parent_step_span_id` and the existing `parent_branch_id` are optional and additive.
- **`StepResult.step_span_id`** + **`Branch.last_step_span_id`** (`coordination/execution/orchestrator_types.py`) — trace-correlation plumbing for the change above. `RealRuntime.step` stamps the step span id on its result; `Orchestrator._interpret` snapshots it onto the branch; `_handle_parallel_invoke` reads it when spawning children.
- **`live_tests/tracing/secret_word_pipeline.py`** — five-agent live workflow exercising execution / branch / step / generation / compaction / tool spans, with a verification step that asserts the full-payload fields (`input_messages`, `response_content`, `sampling_params`) landed on every generation span. `--langsmith` wires `OtelTraceWriter` for the same run. Supports `--provider {openrouter,azure}`.
- **`live_tests/tracing/deep_research.py`** — tracing stress test adapted from `examples/example_01_Deep_Research.py`. Exercises `AgentPool` fan-out (multiple `BrowserAgent` instances dispatched in parallel), compaction spans (large fetched pages trip the compaction threshold), sustained event-bus pressure, real failure modes (bot-blocked fetches, timeouts), and cross-pool `MessageStore` dedup of overlapping system prompts.

### Changed
- **`TraceCollector` constructor signature**: `writers: List[TraceWriter]` → `sinks: List[TelemetrySink]`. Behaviour is otherwise unchanged — the per-span fan-out and the close-on-shutdown loop work identically. `TraceCollector.writers` attribute renamed to `TraceCollector.sinks`.
- **`TraceCollector._stream_span`**: now applies the configured `SecretRedactor` (`config.redactor` or a default `SecretRedactor()`) to the span before fan-out. Every sink and the in-memory `TraceTree` see identical redacted attributes.
- **`NDJSONTraceWriter`**: now inherits from `TelemetrySink` (was `TraceWriter`). The class name and module path stay the same. `write_span(span)` method renamed to `publish_span(span)`. The vestigial `write(trace)` no-op method is removed. All other behaviour identical.
- **`Orchestra._initialize_components`**: combines the default `NDJSONTraceWriter` with `execution_config.tracing.sinks` (user-supplied list) and registers them via `TraceCollector(sinks=...)`. NDJSON sink is always first; user sinks register after.
- **`Orchestra._collect_tracing_metadata`**: iterates `trace_collector.sinks` (renamed from `writers`). Reads counters from the first sink that exposes them — the NDJSON sink, always first.
- **`StatusEvent.event_id`** factory: `uuid.uuid4()` → ULID via `new_id()`. Type unchanged (`str`); subscribers treating `event_id` as opaque are unaffected. ULIDs are 26-character uppercase Crockford-base32, monotonic-orderable within a process — required by Spren's SSE `Last-Event-ID` resume contract.
- **`Span.span_id`** factory in `create_span()`: same UUID4 → ULID migration. `TraceTree.trace_id` (generated in `_handle_execution_start`) likewise.
- **`Orchestra.execute()`** now calls `await self.trace_collector.close()` after `finalize()` in its `finally` block, bounded by `asyncio.wait_for(timeout=NDJSONTraceWriter.CLOSE_TIMEOUT_SECONDS)` (5.0s). Required for the streaming writer's drain-and-flush lifecycle. Both success and exception paths now assign to a `result` local variable so the finally block can populate `result.metadata["tracing"]` before return — no behaviour change for callers that ignore metadata.
- **`Orchestra.__init__()`** instantiates `NDJSONTraceWriter` instead of `JSONFileTraceWriter` when `TracingConfig.enabled` is true.
- **`BaseAPIModel.provider`** is now a stored instance attribute set in `__init__` instead of a `@property` derived from the adapter class name. The previous property returned values like `"asyncopenrouter"` for async adapters (because the class name is `AsyncOpenRouterAdapter`), which appeared in trace attributes as a non-canonical provider id and broke string equality with the configured provider name. `BaseAPIModel.model_name` is now also stored as an instance attribute for symmetry.
- **`TraceCollector._handle_llm_request`** keeps `input_messages` and `tools` **inline** on the span attributes alongside any `*_ref` from `MessageStore`. Without this, OTel-bound consumers (LangSmith, Phoenix) couldn't see actual content — only the dedup hash. Both/either is intentional: sinks pick whichever fits.
- **`OpenAIAdapter.harmonize_response`** Responses-API parsing fixes:
  - `content` defaults to `None` (not `""`) so tool-only responses match the OpenAI `content: null` convention. An empty string flips LangSmith's renderer from the AI-message bubble to a JSON-fields panel.
  - Token-usage parsing tries Responses-API field names first (`input_tokens` / `output_tokens` / `output_tokens_details.reasoning_tokens`) and falls back to chat-completions names (`prompt_tokens` / `completion_tokens` / `reasoning_tokens`) for endpoint compatibility. The Responses API moved reasoning-token accounting into the nested `output_tokens_details` block; the previous code read the flat names and got `null`.

### Removed
- **`TraceWriter` ABC** (`coordination/tracing/writers/base.py`). The class, the abstract `write(trace: TraceTree)` method, the abstract `close()` method, and the virtual `write_span` hook all removed. Replaced by `TelemetrySink` at `coordination/tracing/sink.py` — sinks have `publish_span(span)` and `close()`, and the vestigial finalize-tree call (`write(trace)`) that no longer carried meaning post-streaming is gone. `TraceWriter` had only one production subclass (`NDJSONTraceWriter`); custom subclasses outside the framework migrate to `TelemetrySink`. Forward-only removal per project `CLAUDE.md` §2.6.
- **`TraceWriter` export** from `coordination.tracing.__init__` and `coordination.tracing.writers.__init__`.
- **`for writer in self.writers: await writer.write(trace)` loop** in `TraceCollector.finalize` (was the only call site of the deleted `write(trace)` method; behavior preserved — the loop only invoked a no-op).
- **`JSONFileTraceWriter`** (`coordination/tracing/writers/json_writer.py`). Forward-only removal per project `CLAUDE.md` §2.6 (clean code is the default; deprecation only on explicit user direction). The new NDJSON writer is the sole on-disk trace format. Users with archived legacy `.json` traces deserialize them with `json.load()` directly — the file shape already mirrors `TraceTree.to_dict()`.
- **`JSONFileTraceWriter`** export from `coordination.tracing.writers.__init__`.
- **`TracingConfig.detail_level`** field. Was only consumed by the legacy `JSONFileTraceWriter._filter_span` which is now deleted; no production code path read it after the streaming writer landed. Callers passing `detail_level=...` must drop the kwarg.
- **`TracingConfig.max_content_length`** field. Was technically still referenced by `TraceCollector._handle_final_response` to truncate `final_response_summary`, but the default value `0` caused any non-empty summary to truncate to `"..."` (the whole content was lost). Bug was hidden because no test emits `FinalResponseEvent`. Truncation block removed; summaries are now stored verbatim. Callers passing `max_content_length=...` must drop the kwarg.
- **Legacy `GenerationEvent` emit** in `StepExecutor` (`coordination/execution/step_executor.py`). The new `LLMRequestEvent` / `LLMResponseEvent` pair is emitted at the model-wrapper layer for every `arun` call and carries strictly more data (full prompt, response content, sampling params, tool schemas), so the legacy summary emit produced a duplicate generation span per call. The `GenerationEvent` dataclass and its collector handler are kept so any external consumer still emitting it continues to work; the framework no longer emits it itself.

### Dependencies
- Added: `python-ulid>=3.1.0,<4` (MIT-licensed; transitive dep `typing_extensions` already present).
- Added (optional, `tracing-otel` extra): `opentelemetry-api>=1.27.0`, `opentelemetry-sdk>=1.27.0`, `opentelemetry-exporter-otlp-proto-http>=1.27.0`. Apache-2.0. Required only for the OTel exporter; framework imports lazily so users without the extra are unaffected.

### Migration notes
- The on-disk trace format changed from `{output_dir}/{session_id}.json` (single tree-shaped JSON) to `{output_dir}/{trace_id}.ndjson` (one closed span per line). Callers that previously consumed `.json` files via `json.load()` should switch to either `NDJSONTraceReader.stream()` (live consumer) or `TraceTree.from_ndjson(path)` (post-mortem); the latter returns a `TraceTree` whose `to_dict()` shape matches the legacy file shape.
- ULID identifiers are 26 chars vs UUID4's 36 chars. Downstream consumers asserting on length 36 will break; consumers treating IDs as opaque strings are unaffected. ULIDs are case-sensitive uppercase.
- **`TraceWriter` → `TelemetrySink` migration.** Custom `TraceWriter` subclasses outside the framework break in this release. Migration recipe: change inheritance to `TelemetrySink`, rename `write_span(span)` to `publish_span(span)`, drop any `write(trace: TraceTree)` override (no longer called by the framework — the streaming `publish_span` path is the sole entry point now). `close()` semantics unchanged.
- **`TraceCollector(writers=...)` → `TraceCollector(sinks=...)`.** Direct consumers passing the kwarg by name update accordingly. Positional callers (none in the framework) are unaffected.
- **External observability backends** (Spren, LangSmith, Phoenix, Langfuse, custom HTTP) now register through `TracingConfig.sinks`. `Orchestra` keeps zero new kwargs; sinks travel with the rest of the tracing config inside `ExecutionConfig`.
- **Enabling full LLM-payload capture.** Set `TracingConfig.capture_full_input=True` to populate the new `input_messages` / `tools` / `response_content` / `response_thinking` / `response_tool_calls` / `sampling_params` attributes on every generation span. With it off, generation spans land but carry only metadata (model name, provider, finish reason, token counts).
- **Shipping traces to LangSmith / Phoenix / Langfuse.** Install the extra (`pip install 'marsys[tracing-otel]'`) and pass an `OtelTraceWriter` in `TracingConfig.sinks`:

  ```python
  from marsys.coordination.tracing import TracingConfig, OtelTraceWriter
  cfg = TracingConfig(
      enabled=True,
      capture_full_input=True,
      sinks=[OtelTraceWriter(
          endpoint="https://api.smith.langchain.com/otel/v1/traces",
          headers={"x-api-key": "ls-..."},
          service_name="my-marsys-app",
      )],
  )
  ```

  Other OTLP/HTTP backends (Langfuse, Phoenix, Jaeger, Tempo, Datadog) work with the same writer — swap `endpoint` and `headers`.
- **Subclasses overriding `BaseAPIModel.provider`.** The `@property` is gone — `provider` is a stored instance attribute. Subclasses that read `self.provider` keep working; subclasses that defined their own `@property` override should switch to setting `self.provider = ...` in `__init__`.

---

## [0.3.0] - 2026-05-02

### Added
- **Unified-barrier orchestration**: single `Orchestrator` event loop owning the entire `Branch` / `Barrier` graph (`src/marsys/coordination/execution/orchestrator.py`). One `Barrier` shape, two creation paths (parallel-fork via `invoke_agent` with multiple invocations; lazy `ensure_barrier` rendezvous at convergence points). Six fire gates evaluated in fixed order: status → ROOT-defer → upstream → pending → vestigial-cancel → ratio. Seven invariants documented in `orchestrator_types.py:17-24`.
- **`RealRuntime`** per-branch driver (`src/marsys/coordination/execution/real_runtime.py`) implementing the orchestrator's `Runtime` Protocol: agent acquisition, memory rehydration, `StepExecutor` invocation, `ValidationProcessor` invocation, coordination-action translation. Detects content-only response loops with configurable thresholds.
- **Three reserved det-node singletons**: `StartNode`, `EndNode`, `UserNode` (`src/marsys/coordination/execution/det_nodes.py`). Non-LLM nodes with explicit lifecycle hooks (`on_workflow_start`, `on_single_invoke`, `on_dispatch`) interacting with the orchestrator only through the narrow `DetNodeContext` Protocol. Reserved names auto-resolve in topology converters.
- **Four topology-gated coordination tools** (`src/marsys/coordination/formats/coordination_tools.py`): `invoke_agent(invocations=[…])` (single or parallel peer dispatch), `terminate_workflow(answer=…)` (gated by edge to `End` det-node), `ask_user(question=…)` (gated by edge to `User` det-node), `end_conversation(summary=…)`. Tool gating computed by `StepExecutor._build_coordination_context` from outgoing topology edges.
- **`AgentInput` abstraction** (`src/marsys/agents/agent_input.py`) at the orchestrator-agent boundary. Aggregates multi-source barrier arrivals into typed-text-block content lists (Anthropic-native, OpenAI-compat) so multi-source convergence packages cleanly into one user `Message`. Factories: `from_text`, `from_message`, `from_messages`, `coerce`, `aggregate`.
- **Retry-tiered steering messages** (`src/marsys/coordination/steering/manager.py:_action_error_prompt`) keyed off retry count: tier 1 generic, tier 2 emphasizes tool choice, tier 3+ names topology peers explicitly. Prevents LLM repetition collapse on content-only loops.
- **Content-only loop detection** with `CONTENT_ONLY_STEERING_THRESHOLD = 2` and `CONTENT_ONLY_HARD_LIMIT = 10` (configurable per-workflow via `ExecutionConfig.content_only_steering_threshold` / `content_only_hard_limit`). On hard-limit breach, `RealRuntime._build_content_only_diagnostic` produces a structured diagnostic naming the agent's available coordination tools, regular tools, last assistant content snippet, and the likely root cause (instruction-topology mismatch).
- **Legacy migration shim** in `Orchestra._apply_legacy_topology_shim` (`src/marsys/coordination/orchestra.py:296`): translates legacy `entry_point` / `exit_points` / `User(Node)` topology metadata into explicit `Start` / `End` / `User` det-node edges with `DeprecationWarning`. Idempotent.
- **Compile-time topology validation** (commit `c3dae26`): reachability + cycle-without-escape rules. Malformed topologies fail at `Orchestra.run` with actionable errors before any agents run.
- **Documentation overhaul (Phases 1–6)**:
  - New ADRs: ADR-005 (unified-barrier algorithm), ADR-006 (deprecation timeline).
  - New concept docs: `concepts/coordination-tools.md`, `concepts/det-nodes.md`. AgentInput section added to `concepts/messages.md`.
  - New component YAMLs: `real-runtime.yaml`, `det-nodes.yaml`, `agent-input.yaml`.
  - Architecture overview rewritten; design-principles DP-002, DP-003, DP-004 refreshed.
  - User-facing docs (`api/execution.md`, `api/validation.md`, `guides/steering-and-error-recovery.md`, `getting-started/first-agent.md`) rewritten around the unified-barrier model.

### Renamed (alias kept for one release; removal target v0.4)
- `return_final_response` (tool name) → `terminate_workflow`. Legacy alias still in `COORDINATION_TOOL_NAMES`; routes through `_validate_terminate_workflow` and returns `ActionType.FINAL_RESPONSE` for back-compat.
- `final_response` (action string) → `terminate_workflow` (tool). `ActionType.FINAL_RESPONSE` retained as enum alias.

### Deprecated (removal target v0.4)
- Topology metadata `entry_point` / `exit_points` — use explicit `StartNode` / `EndNode` det-node edges. Auto-shim handles existing usage with `DeprecationWarning`.
- Legacy `User(Node)` regular nodes — use `UserNode()` det-node directly. Auto-translated by the shim.
- `TopologyGraph.has_user_access(agent)` — superseded by `has_edge_to_endnode(agent)` and `has_edge_to_usernode(agent)`. Retained at `topology/graph.py:818` while internal validators transition.

### Removed
- **`BranchSpawner` / `DynamicBranchSpawner`** classes (commit `bc19b98`, Phase 3 cutover) — replaced by `Orchestrator` event loop.
- **`BranchExecutor`** large class (commit `bc19b98`) — replaced by `RealRuntime.step()` driving `StepExecutor`.
- **Legacy JSON `next_action` response format** (commit `bc19b98`, no shim) — coordination uses native tool calls. Agents emitting `{"next_action": "invoke_agent", "action_input": "X"}` no longer validate.
- **`parallel_invoke` action string** — replaced by `invoke_agent` with multiple invocations in one tool call (orchestrator dispatches concurrently).
- **`wait_and_aggregate` action string** — implicit in the orchestrator's barrier semantics.
- **`CoordinationContext.can_return_final_response`** field shim (commit `82ff393`, step-7 cleanup) — was a `@property` aliasing `can_terminate_workflow`. Constructor kwarg shim also removed. Use `can_terminate_workflow` directly.
- Internal compatibility fallbacks in `step_executor` / `response_validator`: `metadata["exit_points"]` / `["original_exits"]` paths and `has_user_access` fallback in `_build_coordination_context`, `_get_available_actions`, `_validate_terminate_workflow`. The shim guarantees the explicit `End` edge.
- `BaseAgent.can_return_final_response` now reads `has_edge_to_endnode` (was `has_user_access`).

### Changed (internal)
- `ConvergenceEvent` now fires for any multi-arrival barrier (rendezvous AND parallel-invoke fork barriers), not just rendezvous (commit `73ee5a3`). Tags `parent_branch_id = bar.resolver_branch` for trace cross-span links. Rendezvous resolver branches now emit `BranchCreatedEvent` when transitioning `WAITING → RUNNING`.
- `tool_executor` empty-args parsing: pre-checks empty/whitespace strings, maps directly to `{}` without parse attempt (commit `73ee5a3`). Parse failures on non-empty input log at WARNING (was ERROR).

### Documentation
- `Multi-Agent_Patterns.md` annotated with v0.3.0 historical-reference banner pointing at the canonical concept docs and ADRs.
- `examples/README.md` rewritten: removed dead `router_integration_example.py` reference; promoted `examples/00_documentation_examples/` as canonical entry point.
- `examples/router_integration_example.py` deleted (used `BranchExecutor` and JSON `next_action`).
- 17 example files annotated with v0.3.0 migration notes (10 use deprecated `entry_point`/`exit_points` and still work via shim; 7 use removed JSON `next_action` and will not run under current code).
- `examples/notebooks/test_Deep_Research_multi-agent.ipynb` annotated with WILL-NOT-RUN markdown header.
- `examples/01_IP_Valuation/05_BACKWARD_DEDUCTION_AGENT_DESIGN.md` annotated with side-by-side legacy → canonical migration table.

---

## [0.2.1-beta] - 2026-03-01

### Added
- Execution tracing module with hierarchical span trees, generation/validation/convergence events, and JSON trace output
- Active context compaction with multi-stage memory processor pipeline (tool truncation, summarization, backward packing) and payload-error recovery
- Response formats architecture with pluggable format system separating prompt building and response parsing from agents
- Task planning module with agent-callable planning tools, plan lifecycle management, and status events
- Provider adapter refactor: extracted adapters from monolithic `models.py` into `models/adapters/` (OpenAI, Anthropic, Google, OpenRouter, OpenAI-OAuth, Anthropic-OAuth)
- OAuth credential store and CLI (`marsys oauth add/remove/list/set-default`) for profile management with automatic token refresh
- RunFileSystem virtual filesystem with mount-based path resolution for sandboxed agent file operations
- Code execution module with sandboxed Python and shell execution
- CodeExecutionAgent for file operations and code execution tasks
- DataAnalysisAgent with persistent Python session (Jupyter-like) for data analysis workflows
- Shell tools with pattern-based command validation (replacing BashTools)
- ElementDetector for unified element detection with shadow DOM and iframe support
- Browser cursor icons for screenshot annotation
- CompactionEvent and MemoryResetEvent status events
- EventBus propagation through execution pipeline for planning events
- Comprehensive test suites for agents, coordination, communication, memory, and models

### Changed
- Agents module migrated to formats architecture; deprecated methods removed
- LearnableAgent migrated to formats architecture
- ValidationProcessor and StepExecutor integrated with formats module
- BrowserAgent detection mode logic updated; `type_text` renamed to `keyboard_input`
- File operations handlers updated to use RunFileSystem
- Default agent model configurations updated
- Logging configuration improved to reduce noise
- README streamlined for quick onboarding
- Documentation updated across API refs, concept guides, and getting started
- Version bumped to 0.2.1-beta

### Fixed
- `.get` to `.pop` for context kwargs in exception constructors preventing duplicate keyword arguments
- Stale agent error context not cleared between turns in BranchExecutor
- Agent-vs-tool confusion detection in ToolExecutor
- State serialization issues (JSON-safe metadata, session ID propagation)
- MaxBranchDepthRule not checking `spawn_request_metadata`
- HTTP 413 REQUEST_TOO_LARGE not classified as a recoverable error
- `max_steps` not propagated through execution context

---

## [0.1.2] - 2025-12-21

### ✨ Added
- OpenAI Responses API migration with multimodal content support
- xAI provider (replacing Groq)
- Gemini 3 thought signature support for multi-turn tool calling
- Tool calling support for local HuggingFace models
- Browser session persistence and tab management
- `get_page_overview()` and `inspect_element()` browser tools
- Error-category-based steering system
- Provider integration tests (OpenAI, Anthropic, Google, OpenRouter, xAI)

### 🔄 Changed
- Domain updated from marsys.io to marsys.ai
- Topology keys renamed from nodes/edges to agents/flows
- Default convergence policy changed to strict (1.0)
- Examples updated to use Claude Sonnet 4.5 via OpenRouter

### 🔧 Fixed
- Google API 429 rate limit handling
- Cross-document element handle errors in browser agent
- Parallel branch execution and continuation state management
- Vision agent initialization and screenshot handling
- aiohttp session cleanup in auto_run()

---

## [0.1.0-beta] - 2025-01-XX

### 🎉 Initial Beta Release

The first public beta release of MARSYS - Multi-Agent Reasoning Systems framework.

### ✨ Added

#### Core Framework
- **Orchestra API**: High-level coordination system for multi-agent workflows
- **Dynamic Branching**: Runtime parallel execution with automatic branch spawning and convergence
- **Three-Way Topology Definition**: Support for string notation, object-based, and pattern configurations
- **7 Pre-defined Patterns**: Hub-and-spoke, pipeline, mesh, hierarchical, star, ring, and broadcast patterns
- **Flexible Agent System**: BaseAgent class with pure execution model for predictable behavior

#### Advanced Features
- **State Persistence**: Full pause/resume capability with FileStorageBackend
- **Checkpointing System**: Save and restore execution state at critical points
- **User Interaction Nodes**: Built-in human-in-the-loop support for approval workflows
- **Rules Engine**: Flexible constraint system for timeouts, resource limits, and custom logic
- **Agent Pools**: True parallel execution with isolated agent instances
- **Memory Management**: Sophisticated conversation memory with retention policies (single_run, session, persistent)

#### Communication & Monitoring
- **Status Manager**: Real-time execution tracking with configurable verbosity levels
- **Enhanced Terminal**: Rich formatting with colors, tables, and progress indicators
- **Multi-Channel System**: Support for terminal, async, and custom communication channels
- **Error Recovery**: Intelligent error handling with routing to User nodes

#### Agent Capabilities
- **BrowserAgent**: Web automation using Playwright for scraping and interaction
- **Tool Integration**: Automatic OpenAI-compatible schema generation from Python functions
- **Multi-Model Support**: Works with OpenAI, Anthropic, Google, Groq, and local models
- **Vision Model Support**: Integration with vision-language models including Qwen-VL

#### Developer Experience
- **Comprehensive Documentation**: Full docs at marsys.ai with tutorials and examples
- **10 Real-World Examples**: Practical examples covering common patterns
- **Type Safety**: Full type hints throughout the codebase
- **Testing Suite**: Comprehensive test coverage (11 core + 5 integration tests)

### 📦 Package Structure

#### Installation Options
- **Default**: `pip install marsys` - Full framework (core + browser + ui + tools)
- **Local Models**: `pip install marsys[local-models]` - Adds PyTorch + Transformers
- **Production**: `pip install marsys[production]` - Adds vLLM + Flash Attention
- **Development**: `pip install marsys[dev]` - Everything + testing tools

#### Dependency Management
- Cleaned up requirements.txt (removed 30+ unused packages)
- Conservative version updates (only minor/patch updates)
- Modular extras system for optional features
- Transitive dependencies handled automatically

### 🔄 Changed

- **Python Requirement**: Changed from `==3.12.*` to `>=3.12` (requires Python 3.12 or higher)
- **Package Name**: Changed from `multi-agent-ai-learning` to `marsys` for production
- **Dependency Versions**: Updated to latest stable versions (Oct 2025)
  - pydantic: 2.10.6 → 2.11.9
  - psutil: 7.0.0 → 7.1.0
  - aiohttp: 3.9.1 → 3.12.15
  - playwright: 1.51.0 → 1.55.0
  - transformers: 4.49.0 → 4.54.1
  - And 10+ other packages (see requirements.txt)

### 🗑️ Deprecated

- **PyPDF2**: Replaced with `pypdf>=3.0.0` (PyPDF2 is no longer maintained)
- **AutoAWQ**: Package archived by maintainers (use alternatives like AWQ from transformers)

### ⚠️ Known Issues

#### Pending Updates (Requires Testing)
- **python-json-logger v3**: Kept at v2.0.7 (v3.3.0 requires API compatibility testing)

#### Platform-Specific
- **Windows**: WSL recommended for best experience (native Windows support is experimental)
- **vLLM**: Linux-only (not available on macOS/Windows)
- **Flash Attention**: Requires CUDA-compatible GPU

### 🔧 Fixed

- Memory leaks in parallel execution scenarios
- Branch synchronization issues at convergence points
- Agent pool allocation race conditions
- Error propagation in nested branches
- Context passing between sequential agents

### 🛡️ Security

- No known security vulnerabilities
- All dependencies from trusted sources (PyPI)
- No credentials or secrets stored in package

### 📚 Documentation

- Complete framework documentation at [marsys.ai](https://marsys.ai)
- API reference for all public classes and methods
- Architecture guides and design patterns
- 10 fully documented real-world examples
- Migration guides and best practices

### 🙏 Acknowledgments

- Open-source community for invaluable feedback
- Model providers (OpenAI, Anthropic, Google) for powerful APIs
- Early adopters and testers who shaped the framework

---

## Version History Summary

| Version | Date | Status | Highlights |
|---------|------|--------|------------|
| 0.3.0 | 2026-05-02 | Released | Unified-barrier orchestrator, det-nodes, native-tool-call coordination, AgentInput, retry-tiered steering, full docs overhaul |
| 0.2.1-beta | 2026-03-01 | Released | Active context compaction, modular adapters, new agents, RunFileSystem |
| 0.1.2 | 2025-12-21 | Released | OpenAI Responses API, xAI provider, steering system |
| 0.1.0-beta | 2025-01 | Released | Initial public beta with full framework |

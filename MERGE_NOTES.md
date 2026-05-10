# MARSYS tracing — merge plan & progress notes

> **For future agents**: read this top-to-bottom before touching tracing code in
> either repo. It captures the full state of an in-progress merge between two
> divergent tracing implementations and the design decisions made so far.
> Update the **Status** and **Decision log** sections as you progress.

---

## 0. Where things live

Two clones share a workspace at `c:/Users/mehdi/OneDrive/Documents/projects/marsys/`:

| Path | What it is |
|---|---|
| [MARSYS/](MARSYS/) | Mehdi's clone of `rezaho/MARSYS` (`main` branch, base `cfd88f5`). Has uncommitted local work — a SQLite + OTel tracing module documented in [MARSYS/docs/architecture/framework/decisions/ADR-007-langsmith-grade-tracing.md](MARSYS/docs/architecture/framework/decisions/ADR-007-langsmith-grade-tracing.md). |
| [current_branch/MARSYS/](current_branch/MARSYS/) | Same repo, branch `feature/tracing-streaming`. Teammate restructured to a monorepo (`packages/framework/`, `packages/spren/`, `apps/{web,desktop,tui}`) and shipped a different tracing implementation: streaming NDJSON. |

Both diverged from commit `cfd88f5`. Goal: merge Mehdi's full-payload capture
work into `current_branch/MARSYS` without breaking teammate's streaming model.

All implementation work happens in `current_branch/MARSYS/`. The original
`MARSYS/` clone is read-only reference for what Mehdi's design did.

---

## 1. Status

**Phase 1 — full-payload LLM capture: ✅ DONE.** Capture-layer code is wired
into `current_branch/MARSYS/`, with 16 new tests all green and teammate's 98
existing tracing tests still passing. See [§5.1](#51-phase-1--full-payload-capture)
and [§6](#6-test-status).

**Phase 2 — OTel exporter for LangSmith / Phoenix / Langfuse: ✅ DONE.**
[`OtelTraceWriter`](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/writers/otel_writer.py)
shipped as a `TelemetrySink` subclass — per-span emission, `gen_ai.*`
semconv mapping, lazy OTel imports under the new `tracing-otel`
extra. SQLite writer **dropped** (no killer use case once OTel covers
vendor export). 14 new tests green. See [§5.2](#52-phase-2--otel-exporter)
and the [decision log](#decision-log-append-only).

**Phase 3 — remaining merge axes: 🔲 NOT STARTED.** See [§7](#7-open-decisions-not-yet-discussed).

---

## 2. The big picture

Each side built tracing for the same goal (LangSmith-grade observability) but
took opposite paths. The two designs disagree on **almost every architectural
axis**.

| Axis | Mehdi's (in [MARSYS/](MARSYS/)) | Teammate's (in [current_branch/MARSYS/](current_branch/MARSYS/)) |
|---|---|---|
| Repo layout | Flat `src/marsys/` | **Monorepo** with `packages/framework/`, `pnpm`, `cargo`, `uv` |
| Storage | **SQLite** (CAS, blake2b-32) **+ OTel** | **NDJSON** streaming + sidecar `messages/{sha256}.json` |
| Write timing | At-finalize (whole `TraceTree` → write) | **Streaming** — async queue + drain task |
| Writer ABC | `TraceWriter.write(tree)` | `TelemetrySink.publish_span(span)` (per-span) |
| LLM capture point | `@_capture_arun` decorator on 3 model wrappers | `step_executor._serialize_input_messages` |
| What gets captured | **Full**: messages + tool schemas + sampling + content + thinking + reasoning + tool_calls | **Partial**: input messages only — `GenerationEvent` is metrics + booleans, **no response content** |
| Correlation | 7 ContextVars + decorator reads them | `step_span_id` carried in `context["step_span_id"]` dict |
| Per-message dedup | blake2b-32 in `content` table | SHA-256 + `FilesystemMessageStore` + RFC 6902 patches between branches |
| Compaction tagging | `current_call_kind="compaction"` ContextVar | **Not captured at all** |
| Re-entrancy guard | `in_capture` ContextVar | N/A |
| `StepResult.step_span_id` | New field added | Not present |
| `ValidationDecisionEvent` | Mehdi added the missing emit | **Already emits** (handler at `_handle_validation_decision`) |
| Redaction | Pre-hash callable per blob | `SecretRedactor` deny-list, runs once at `_stream_span` |
| Event ID factory | `uuid4` | **ULID** |

**Key insight**: teammate's design has a clean architecture but only captures
*input* messages. Response content / thinking / tool_calls / sampling /
advertised tool schemas are still missing — exactly the gap Mehdi's ADR §1.2
was created to close. So Mehdi's capture mechanism is **load-bearing value**
even after teammate's work.

---

## 3. Decisions locked in

### D1. Capture layer — keep Mehdi's (model-wrapper level)

Discussion explored *recording vs reconstruction*:

- Teammate's reconstructs the input via `step_executor._serialize_input_messages` *before* the agent runs. If anything mutates the messages between assembly and the wire (system-prompt prepending, format processors injecting steering tools, retry rewrites), the trace and the wire **drift silently**.
- Mehdi's records the literal kwargs the model wrapper hands to the adapter. Matches the wire by definition.

For LangSmith-grade debugging, recording is non-negotiable. Mehdi's design also
covers code paths teammate's misses entirely: memory compaction, tool-internal
model calls (`web_tools.py`), within-`arun` retries.

**The cost**: clean separation between coordination and transport layers is
sacrificed (model.py imports tracing lazily). Accepted.

### D2. No ContextVars — explicit kwarg threading via `TraceContext`

Teammate raised legitimate objections to ContextVars:
- Implicit data flow (signature lies about dependencies)
- Lifecycle on the honor system (Mehdi's design admits the leak in ADR §7.1, §7.2)
- Test fragility (need fixtures resetting every var)
- Threading-boundary surprises (`run_in_executor` doesn't propagate them)
- Static decoupling is a runtime lie

**Replaced with**: a single `TraceContext` dataclass, threaded explicitly
through `context["trace_ctx"]` → `run_context["trace_ctx"]` → `model_kwargs`
→ `model.arun(trace_ctx=...)`. Teammate already passes `step_span_id`
through the same context dict, so we extend an existing pattern rather than
inventing a new one. ~30 extra lines of forwarding boilerplate eliminate
all four footguns.

### D3. Async context manager, not decorator

User wanted to keep the trace-emit code visible at the call site (no
`@decorator` magic). Three alternatives compared:

| | Decorator | Raw inline | Helper fns | **Context manager** |
|---|---|---|---|---|
| Visible at call site | no | yes | yes | **yes** |
| Single source of truth for emit logic | yes | no | yes | **yes** |
| Error path automatic | yes | manual | manual | **yes** |
| Re-entrancy guard centralized | yes | manual | manual | **yes** |

Picked **`async with capture_llm_call(...) as cap:`** — visible at every
`arun` call site, single source of truth in [capture.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/capture.py),
no forgettable conditions.

### D4. Adopt teammate's NDJSON pipeline as the persistence path

Capture emits `LLMRequestEvent` / `LLMResponseEvent` on the **same**
`EventBus` teammate already uses. Collector handlers translate them into
generation/compaction child spans on the active step span, then call
teammate's existing `_stream_span(span)` so the NDJSON sink picks them up
unchanged. No new sink needed for Phase 1.

The SQLite + OTel sinks from Mehdi's branch were deferred at the time this
section was written; see [D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite-phase-2)
below for the resolution.

### D5. Tool schemas routed through MessageStore for dedup

Tool schemas can be 30 KB+ and rarely change. Capturing them inline on
every span would bloat NDJSON. Solution: when `TracingConfig.capture_full_input`
is on and a `MessageStore` exists, hash tool schemas through the same store
that handles input messages — a single tool catalogue lands once across all
steps. Falls back to inline when the store is off.

### D6. Response payload kept inline (for now)

Response content/thinking/tool_calls are kept inline in span attributes,
not routed through `MessageStore`. Responses are typically modest and unique
per call (low dedup payoff). Promote to the store later if NDJSON line sizes
become a concern.

### D7. Keep NDJSON default, add OTel sink, drop SQLite (Phase 2)

Discussion compared SQLite vs NDJSON as the single primary store along
two axes the user prioritized:

- **Compactness** — wash. Both dedup heavy content (NDJSON via sidecar
  `messages/`, SQLite via the `content` table). Zipped NDJSON is usually
  smaller than the equivalent SQLite file.
- **Concurrency** — NDJSON wins decisively. File-per-trace + content-addressed
  sidecar means parallel branches and parallel `Orchestra.run()` invocations
  contend on **nothing**. SQLite has a single writer lock — every parallel
  branch's span emission queues on it. For a framework whose whole point is
  fan-out, that's a structural mismatch.

Decision:

- **NDJSON stays the default** (already wired in `Orchestra.__init__`).
- **OtelTraceWriter ships as opt-in `TelemetrySink`** under the new
  `tracing-otel` extra — covers LangSmith, Langfuse, Phoenix, Jaeger,
  Tempo, Datadog via OTLP/HTTP and `gen_ai.*` semantic conventions.
- **SQLite writer is dropped entirely.** No cross-run-SQL use case on
  the team; OTel + LangSmith covers vendor querying. Dropping it saves
  ~500 lines of code we won't have to maintain or refactor to streaming.

The post-Phase-2 storage story is two-layer: NDJSON locally for "what
happened on this machine," LangSmith remotely for "let me actually look
at it."

---

## 4. Architecture after Phase 1

```
Orchestra → step_executor builds TraceContext (no ContextVars)
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
                           model.arun(messages, ..., trace_ctx=...)
                                          │
                              ┌───────────┴───────────┐
                              │ async with             │
                              │  capture_llm_call(...) │ ──▶  EventBus.emit(LLMRequestEvent)
                              │     as cap:            │       (full payload: messages,
                              │   await adapter.arun() │        tools, sampling)
                              │   cap.set_response(r)  │
                              └───────────┬───────────┘ ──▶  EventBus.emit(LLMResponseEvent)
                                          │                  (content, thinking, tool_calls,
                                          ▼                   reasoning, metadata)
                                  HarmonizedResponse
                                          │
                          ┌───────────────┴────────────┐
                          │ TraceCollector handlers    │
                          │  _handle_llm_request:      │
                          │   open generation span,    │
                          │   route messages + tools   │
                          │   through MessageStore     │
                          │  _handle_llm_response:     │
                          │   close span, attach       │
                          │   payload, _stream_span()  │
                          └───────────────┬────────────┘
                                          │
                                          ▼
                                  TelemetrySink fan-out
                                  (NDJSONTraceWriter + any user sinks)
```

**Compaction calls** travel the same path: the parent `TraceContext` is
attached to the `compaction_runtime` dict in
[agents.py](current_branch/MARSYS/packages/framework/src/marsys/agents/agents.py)
at both compaction trigger sites (threshold-driven `compact_if_needed` and
payload-error `compact_for_payload_error`). `_run_compaction_model` derives
`trace_ctx.child(kind="compaction")` so the captured call lands as a
`kind="compaction"` span under the active step.

---

## 5. What has been implemented

Working directory for all changes: `current_branch/MARSYS/packages/framework/`.

### 5.1. Phase 1 — full-payload capture

**New files**:

| File | Lines | Purpose |
|---|---|---|
| `src/marsys/coordination/tracing/trace_context.py` | ~45 | `TraceContext` frozen dataclass + `child()` / `mark_captured()` helpers |
| `src/marsys/coordination/tracing/capture.py` | ~155 | `capture_llm_call` async context manager + `LLMCallCapture` + `extract_sampling_params` |
| `tests/coordination/tracing/test_capture_helper.py` | ~210 | 9 tests — happy path, 4 bypass conditions, error path, re-entrancy guard, compaction kind, sampling extraction |
| `tests/coordination/tracing/test_collector_llm_handlers.py` | ~210 | 7 tests — span open/close/stream, mismatched/orphan handling, compaction kind, error mapping, finalize drain |

### Edits

| File | Change |
|---|---|
| `src/marsys/coordination/tracing/events.py` | Added `LLMRequestEvent` + `LLMResponseEvent` next to existing `GenerationEvent` |
| `src/marsys/coordination/tracing/collector.py` | Added `_handle_llm_request` / `_handle_llm_response`, registered them in `_subscribe_to_events`, added `self.llm_spans: Dict[str, Span]` index, cleaned up at finalize |
| `src/marsys/coordination/tracing/__init__.py` | **Did NOT re-export** `capture` / `events` / `trace_context` — would create a circular import (see [§9.A](#9a-circular-import-trap)). Import them from submodules directly. |
| `src/marsys/models/models.py` | Wrapped all 3 `arun` methods with `async with capture_llm_call(...) as cap:` — `BaseLocalModel.arun` (~line 438), `BaseAPIModel.arun` (~line 698, both async-adapter and thread-fallback branches), `LearnableModelWrapper.arun` (~line 987). The Learnable wrapper forwards `cap.inner_ctx` so the inner wrapper bypasses re-emission. |
| `src/marsys/coordination/execution/step_executor.py` | Built `TraceContext` right after `step_span_id` is generated (~line 348) and put it on both `context["trace_ctx"]` and `run_context["trace_ctx"]` |
| `src/marsys/agents/agents.py` | `run_step` extracts `trace_ctx` from `context` and adds to `model_kwargs` so it flows through `_run` → `**api_model_kwargs` → `model.arun`. Also threaded into `compaction_runtime` at both compaction sites (threshold and payload-error). |
| `src/marsys/agents/browser_agent.py` | `InteractiveElementsAgent._run` merges `trace_ctx` from `**kwargs` into `api_model_kwargs` (without this, browser-agent calls bypass capture even when the rest of the run is traced) |
| `src/marsys/agents/memory_strategies.py` | `_run_compaction_model` accepts `trace_ctx`, derives `child(kind="compaction")`, passes to `model.arun`. Caller in `SummarizationProcessor.reduce` reads from `runtime.get("trace_ctx")`. |
| `src/marsys/agents/memory.py` | `_forced_summarization_fallback` forwards `trace_ctx` from its `runtime` arg into the `_run_compaction_model` call |

### Things deliberately NOT done in Phase 1

- **No `StepResult.step_span_id` field** added — teammate's design routes via
  `context["step_span_id"]` instead. We follow that.
- **No `ValidationDecisionEvent` re-emission** — teammate already has both
  the handler and the emit path. Don't duplicate.
- **No `tracing/__init__.py` re-exports** of `capture` / `events` /
  `trace_context` — would cycle. See [§9.A](#9a-circular-import-trap).

### 5.2. Phase 2 — OTel exporter

**New files**:

| File | Lines | Purpose |
|---|---|---|
| `src/marsys/coordination/tracing/writers/otel_writer.py` | ~365 | `OtelTraceWriter` (`TelemetrySink` subclass). Per-span emission, deterministic ULID→OTel-ID hashing for parent links, `gen_ai.*` semconv mapping, lazy OTel imports. Test seam via `_exporter_override` for `InMemorySpanExporter`. |
| `tests/coordination/tracing/test_otel_writer.py` | ~265 | 14 tests — construction guards, deterministic parent linking under streaming order, `gen_ai.*` mapping for generation/compaction/tool/step/branch/execution kinds, per-message events, `gen_ai.choice` payload, error status, lifecycle isolation. |

**Edits**:

| File | Change |
|---|---|
| `src/marsys/coordination/tracing/collector.py` | `_handle_llm_request` now keeps `input_messages` and `tools` **inline** on the span attributes alongside any `*_ref` from `MessageStore`. Without this, OTel-bound consumers couldn't see actual content (only the dedup hash). The dual-storage is intentional — sinks pick whichever fits. Also passes `prev_history=None` explicitly to `build_input_messages_ref` (positional kwargs are required by that helper). |
| `src/marsys/coordination/tracing/writers/__init__.py` | Re-export `OtelTraceWriter`. Safe because the OTel SDK only loads on instantiation, not on module import. |
| `src/marsys/coordination/tracing/__init__.py` | Re-export `OtelTraceWriter` so users can `from marsys.coordination.tracing import OtelTraceWriter`. |
| `pyproject.toml` | New optional extra `tracing-otel` pulls `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`. |

**SQLite writer**: not ported (per [D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite-phase-2)).
The `MARSYS/` clone's `SQLiteTraceWriter` and its tests are abandoned — do
not bring them across.

**Wiring example**:

```python
from marsys.coordination.tracing import (
    TracingConfig, OtelTraceWriter,
)

cfg = TracingConfig(
    enabled=True,
    capture_full_input=True,
    sinks=[
        OtelTraceWriter(
            endpoint="https://api.smith.langchain.com/otel/v1/traces",
            headers={"x-api-key": "ls-..."},
            service_name="my-marsys-app",
        ),
    ],
)
```

`Orchestra.__init__` already prepends `NDJSONTraceWriter` and extends with
`TracingConfig.sinks`, so no Orchestra change was required.

**Subtle implementation notes**:

- OTel needs 16-byte trace_ids and 8-byte span_ids; teammate's IDs are
  26-char ULIDs. Mapping is `blake2b(ulid, digest_size=8 or 16)` —
  deterministic so children arriving before parents (streaming close-order)
  still resolve their `parent.span_id`.
- The OTel SDK's `_Span.__init__` does **not** accept `start_time` as a
  constructor kwarg (changed in 1.27+). We construct the span then call
  `.start(start_time=...)` and `.end(end_time=...)` separately. If the SDK
  surface shifts again, look there first.
- `BatchSpanProcessor` is used by default (production); `SimpleSpanProcessor`
  is wired only via `_exporter_override` (tests). This split keeps
  production async + batched while tests stay synchronous and assertable.

### Things deliberately NOT done in Phase 2

- **No `tracing.otel: OtelExportConfig` convenience field** on `TracingConfig`.
  Users compose the writer themselves and pass it through `sinks=[...]`.
  Adding a typed convenience config would couple `TracingConfig` to the OTel
  module surface. Trivial to add later if it bites.
- **No SQLite refactor**. Per [D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite-phase-2),
  dropped entirely.
- **No live LangSmith E2E test**. The OTel mapping is verified end-to-end
  inside the SDK via `InMemorySpanExporter`; smoke-testing against a real
  LangSmith endpoint is a manual step. Recipe in [§8](#8-how-to-verify).

### 5.3. Phase 3 — Azure OpenAI via stock `OpenAIAdapter`

**Outcome: zero new framework code.** Azure's Cognitive Services / AI
Foundry endpoints (`<resource>.cognitiveservices.azure.com`) on the v1
inference path expose the OpenAI Responses API at `/openai/v1/responses`
and accept `Authorization: Bearer <api-key value>`. That's exactly what
the framework's `OpenAIAdapter` posts to and how it authenticates, so
Azure works as a pure configuration choice — no adapter, no factory
entry, no env-var map change, no `Literal[...]` constraint update.

**Only edit needed** is in the live test:

| File | Change |
|---|---|
| `live_tests/tracing/secret_word_pipeline.py` | Added `--provider=azure` CLI flag as an *env-var-source selector* (reads `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT`). The resulting `ModelConfig` uses `provider="openai"` because there's nothing Azure-specific to wire. The CLI flag is purely an ergonomic UX so users don't have to know the underlying provider name maps to OpenAI. `_load_env_early()` now uses `load_dotenv(env_file, override=True)` so stale shell-injected env vars don't silently shadow `.env` updates. |

**Verified configuration** (`marsys-development-01-m.cognitiveservices.azure.com`):

```bash
AZURE_OPENAI_API_KEY=<KEY 1 or KEY 2 from "Keys and Endpoint">
AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=<deployment name from Azure Portal>
```

#### What the discovery process taught us (preserve for future agents)

The first Azure resource we tested (`<resource>.openai.azure.com`, the
**older Azure OpenAI Service** product) does NOT expose `/responses`,
and on top of that its v1 deployment was a reasoning model
(`gpt-5.4-mini`) that rejects the legacy `max_tokens` / `temperature`
fields with a 400 `unsupported_parameter`. We spent hours building a
standalone `AzureOpenAIAdapter` to handle those quirks.

Then we switched to a different Azure resource on the **newer
Cognitive Services / AI Foundry product** (`<resource>.cognitiveservices.azure.com`)
and discovered it does expose `/responses`, accepts Bearer auth, and
the reasoning-model handling is built into the OpenAI Responses API
shape — so all the quirk-translation work disappeared.

The `cognitiveservices.azure.com` domain is the strategic Microsoft
direction; `openai.azure.com` is being deprecated. New Azure deployments
should use the newer endpoint and will work with stock `OpenAIAdapter`.

**If a future user reports `openai.azure.com` doesn't work**, the answer
is one of:

1. Migrate the resource to AI Foundry (Microsoft-recommended path).
2. Re-introduce the `AzureOpenAIAdapter` we built on 2026-05-10 — see
   the decision log entry below for the design (api-key header,
   `/chat/completions` URL, reasoning-quirk translation in
   `format_request_payload`). Roughly 225 lines of code.

Either is a one-time decision; we don't carry the adapter today because
no current consumer needs it.

#### Caveats that still apply

- The deployment name is **not** the model name. `gpt-5.4-mini` (a valid
  deployment) ≠ `gpt-4o` (a catalog model without a deployment by that
  name on this resource). The `/openai/v1/models` listing endpoint
  returns the *catalog*, not the actual deployments — so it can mislead.
- Without portal access, the only reliable way to discover an unknown
  deployment name is brute-force probing via `/chat/completions` —
  `200` vs `404 DeploymentNotFound` settles each candidate.
- The MARSYS `ModelAPIError` classifier sometimes maps Azure 400s to
  `MODEL_API_UNKNOWN_ERROR / "Resource Not Found"` in the user-facing
  message — which is misleading. Worth a follow-up to improve error
  mapping for Azure responses (or just to bubble up the raw response
  body).

#### Test status

133 tracing tests passing (1 pre-existing Python 3.14 failure unrelated
to merge work). End-to-end correctness on the new Azure endpoint is
verified by the `live_tests/tracing/secret_word_pipeline.py --provider azure`
recipe in [§8.3](#83-azure-openai-provider-phase-3).

---

## 6. Test status

```
PYTHONPATH=src python -m pytest tests/coordination/tracing/ --ignore=tests/coordination/tracing/test_streaming_integration.py
```

Latest result (after Phase 2): **129 passed, 1 pre-existing failure**.

- **30 new tests, all passing** — 9 capture helper + 8 collector handlers + 14 OTel writer (the 8th collector test was added in Phase 2 to cover the inline-vs-ref dual-storage fix).
- **All 99 of teammate's existing tracing tests still pass**.
- 1 pre-existing failure (`test_reader_opens_under_writer_lock_windows`) due
  to Python 3.14 `asyncio.get_event_loop()` incompat. **Unrelated to merge work**;
  same code on teammate's untouched branch fails identically.

Integration test (`test_streaming_integration.py`) skipped during smoke
testing — needs fuller Orchestra setup. Should be exercised before claiming
the merge is fully validated end-to-end.

### 6.1. Live e2e (manual)

A runnable smoke script lives at
[`live_tests/tracing/secret_word_pipeline.py`](current_branch/MARSYS/packages/framework/live_tests/tracing/secret_word_pipeline.py)
following teammate's `live_tests/` conventions (no `test_` prefix,
`--output-dir` flag, single-line JSON summary, exit 0/1). Five agents
collaborate to discover the secret word "MARS" so the trace exercises
execution / branch / step / generation / tool / convergence span kinds
all together.

It verifies the Phase-1 + Phase-2 invariants on a real LLM run:

- Every generation span carries `input_messages` (inline), `response_content`,
  and `sampling_params`.
- Tool calls and parallel-fan-out branches land in the trace.
- When `--langsmith` is passed and `LANGSMITH_API_KEY` is set, the
  `OtelTraceWriter` is wired and ships spans to LangSmith via OTLP/HTTP.

Run:

```bash
cd packages/framework
python live_tests/tracing/secret_word_pipeline.py --output-dir /tmp/marsys/secret-001
# add --langsmith to enable LangSmith export
# add --print-tree to see the reconstructed TraceTree
```

Costs a few cents in OpenRouter spend per run. Not part of CI.

---

## 7. Open decisions (not yet discussed)

These are the remaining axes from [§2](#2-the-big-picture). Each is its own
mini-conversation worth having before implementation.

### 7.A. Validation event emission

Teammate already emits `ValidationDecisionEvent` somewhere. Mehdi's branch
also added one in `real_runtime.py`. Need to:
1. Confirm teammate's emit path covers all the cases Mehdi was concerned about.
2. If yes, drop the duplicate. If gaps exist, fill them in teammate's tree.

### 7.B. Storage / writers — ✅ RESOLVED in Phase 2

Decision: NDJSON stays default; `OtelTraceWriter` ships as opt-in
`TelemetrySink`; SQLite writer dropped. See [D7](#d7-keep-ndjson-default-add-otel-sink-drop-sqlite-phase-2)
for rationale and [§5.2](#52-phase-2--otel-exporter) for what landed.

### 7.C. Per-message dedup mechanism

Mehdi: blake2b-32 in a `content` SQLite table, per-blob.
Teammate: SHA-256 + `FilesystemMessageStore` + RFC 6902 patches between
branches for git-like diff display.

If teammate's NDJSON stays the source-of-truth (likely), use SHA-256 / sidecar
everywhere and drop blake2b. The current `_handle_llm_request` already calls
teammate's `build_input_messages_ref` for input messages and tool schemas.
**Aligned in Phase 1; nothing more to do unless we add SQLite back.**

### 7.D. Redaction placement

Mehdi: pre-hash on each blob (must be pre-hash, otherwise CAS reference
integrity breaks).
Teammate: `SecretRedactor` deny-list at `_stream_span`, runs once before
fan-out.

Teammate's is cleaner for the streaming pipeline. If we re-add SQLite, the
pre-hash invariant must be honored *inside* the SQLite sink (run the
redactor again before computing the CAS hash). **No conflict in Phase 1.**

### 7.E. Repo restructure

Mehdi's local branch is still flat-layout (`src/marsys/...`); teammate's is
monorepo (`packages/framework/src/marsys/...`). Any further work Mehdi does
in his local clone needs to be relocated. **All Phase 1 work landed directly
in the monorepo layout** — no relocation debt.

### 7.F. Event ID factory

Teammate switched `uuid4()` → `ULID`. Mehdi's branch uses `uuid4`. The
capture helper currently uses `uuid.uuid4()` for `request_id` to keep
parity with Mehdi's design — switching to ULID would be ~5 lines if
teammate prefers consistency. **Trivial, not blocking.**

---

## 8. How to verify

### 8.1. NDJSON capture (Phase 1)

```python
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing import TracingConfig

result = await Orchestra.run(
    task="...",
    topology=topology,
    execution_config=ExecutionConfig(
        tracing=TracingConfig(
            enabled=True,
            output_dir="./traces",
            capture_full_input=True,         # turn on the full-payload pipeline
        ),
    ),
)
```

After the run, the NDJSON file at `./traces/<trace_id>.ndjson` should
contain `kind: "generation"` (and `kind: "compaction"` if compaction fired)
spans with attributes:

- `model_name`, `provider`, `sampling_params`
- `input_messages` (inline) and `input_messages_ref` (CAS pointer into `./traces/messages/<sha>.json`)
- `tools` (inline) and `tools_ref` (CAS pointer)
- `response_content`, `response_thinking`, `response_reasoning`,
  `response_tool_calls`, `response_metadata`
- `duration_ms`, `request_id`, `kind`, `agent_name`

If `response_content` is missing on every span, look at [§9.B](#9b-debugging-missing-payload).

### 8.2. OTel export to LangSmith (Phase 2)

```python
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing import TracingConfig, OtelTraceWriter

result = await Orchestra.run(
    task="...",
    topology=topology,
    execution_config=ExecutionConfig(
        tracing=TracingConfig(
            enabled=True,
            output_dir="./traces",
            capture_full_input=True,
            sinks=[
                OtelTraceWriter(
                    endpoint="https://api.smith.langchain.com/otel/v1/traces",
                    headers={"x-api-key": "ls-..."},
                    service_name="my-marsys-app",
                ),
            ],
        ),
    ),
)
```

Open the LangSmith project — each `Orchestra.run()` should appear as a
trace, with one span per agent step and per LLM call. Generation spans
carry `gen_ai.system` / `gen_ai.request.model` / `gen_ai.request.*`
attributes and per-message events (`gen_ai.system.message`,
`gen_ai.user.message`, `gen_ai.assistant.message`, `gen_ai.tool.message`,
`gen_ai.choice`).

Other OTLP/HTTP backends (Langfuse, Phoenix, Jaeger, Tempo, Datadog) work
the same way — swap `endpoint` and `headers`. Install the extra first:

```
pip install 'marsys[tracing-otel]'
```

If LangSmith shows a trace but the messages look empty, confirm
`capture_full_input=True` is set — without it, only metadata flows.

### 8.3. Azure OpenAI provider (Phase 3)

```bash
# .env (or shell env)
AZURE_OPENAI_API_KEY=<KEY 1 from Azure Portal -> Keys and Endpoint>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=<deployment name from Portal -> Model deployments>
```

```cmd
cd packages\framework
python live_tests\tracing\secret_word_pipeline.py --provider azure --output-dir <out-dir>
```

Add `--langsmith` to also fan out to LangSmith — Azure traces appear in
the same project as OpenRouter ones; the trace shape is identical.

**Common gotchas**:

- `gpt-4o` from the model catalog is *not* the same as a deployment named
  `gpt-4o`. Use the actual deployment name. If the deployment listing
  endpoint is locked down, just brute-force `gpt-5.4-mini`, `gpt-4o`,
  `gpt-4o-mini`, or whatever your team's naming convention is via
  one-off `chat/completions` curls — `200` vs `404 DeploymentNotFound`
  resolves it.
- The endpoint MUST end in `/openai/v1/`. The classic per-deployment
  URL (`.../openai/deployments/<name>/...?api-version=...`) is not
  supported — see [§5.3](#53-phase-3--native-azure-openai-provider).
- The two API keys (KEY 1 / KEY 2) are interchangeable; pick either one.

If you get `[MODEL_API_INVALID_MODEL_ERROR] Resource Not Found`, check
in this order: (1) endpoint shape ends in `/openai/v1/`, (2) deployment
name matches an actual deployment on the resource, (3) auth — verify
with the curl in [§5.3](#53-phase-3--native-azure-openai-provider).

---

## 9. Maintenance notes

### 9.A. Circular import trap

**Do not** add `from .capture import ...`, `from .events import LLMRequestEvent`,
or `from .trace_context import ...` to
[tracing/__init__.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/__init__.py).

The chain that breaks:
```
coordination/__init__.py
  → status/events.py
  → from ..tracing._ids import new_id   # forces tracing/__init__ to load
  → tracing/__init__.py
  → from .capture import ...
  → from .events import LLMRequestEvent
  → from ..status.events import StatusEvent   # ❌ partially initialized
```

Solution: import the new modules directly from their submodules. `__init__`
keeps the original re-exports only.

### 9.B. Debugging missing payload

If `LLMRequestEvent` / `LLMResponseEvent` aren't reaching the collector:

1. Check `trace_ctx` is actually reaching `model.arun`. Print `kwargs.get("trace_ctx")`
   inside one of the wrapped `arun` methods — if `None`, the kwarg dropped somewhere
   in the chain. Common culprits: a new `_run` override that doesn't forward `**kwargs`,
   or a `_get_api_kwargs()` filter that strips it.
2. Check the bus has subscribers — `EventBus.subscribe('LLMRequestEvent', ...)` must
   have run. The collector does this in `_subscribe_to_events`. If the collector
   isn't constructed (e.g., `TracingConfig.enabled=False`), no one's listening.
3. Check the bypass conditions in [capture.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/capture.py):
   - `messages` not a list → bypass (web_tools raw-string path)
   - `event_bus is None` → bypass
   - `trace_ctx.captured` → bypass (re-entrancy guard fired)

### 9.C. Adding a new model wrapper class

If a 4th wrapper is added (e.g., a remote-batch model), wrap its `arun` with
the same `async with capture_llm_call(...)` block as the other three. Pattern:

```python
async def arun(self, messages, *, trace_ctx=None, **kwargs):
    from marsys.coordination.tracing.capture import (
        capture_llm_call, extract_sampling_params,
    )
    async with capture_llm_call(
        trace_ctx,
        model_name=..., provider=...,
        messages=messages, tools=kwargs.get("tools"),
        sampling_params=extract_sampling_params(kwargs),
    ) as cap:
        # If calling another wrapped model.arun: forward cap.inner_ctx
        # If calling an adapter (terminal): don't forward trace_ctx at all
        response = await self.delegate.arun(...)
        cap.set_response(response)
    return response
```

Forgetting this means calls through the new wrapper are silently uncaptured.
Worth a CI lint that fails if any class with an `arun` method doesn't `async
with capture_llm_call` — easy to add later.

### 9.D. Adding a new field to `LLMRequestEvent` / `LLMResponseEvent`

Five layers to touch (intentional — each layer has a different audience):

1. Add field to event dataclass in [events.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/events.py).
2. Populate it in [capture.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/capture.py).
3. Surface on the span in `_handle_llm_request` / `_handle_llm_response` in
   [collector.py](current_branch/MARSYS/packages/framework/src/marsys/coordination/tracing/collector.py).
4. (Optional) If it's heavy, route through `MessageStore` instead of inline.
5. (Optional) If we ever ship `OtelTraceWriter`, map it to a `gen_ai.*`
   attribute there.

---

## 10. Reference

- Mehdi's original design doc: [MARSYS/docs/architecture/framework/decisions/ADR-007-langsmith-grade-tracing.md](MARSYS/docs/architecture/framework/decisions/ADR-007-langsmith-grade-tracing.md)
- Teammate's commit ahead of base: `cfd88f5..HEAD` on `feature/tracing-streaming`
- Common ancestor: `cfd88f5` (Merge PR #29)
- Most-relevant teammate commits:
  - `12447f7` — streaming NDJSON tracing writer + reader
  - `d21c618` — `TelemetrySink` ABC, redact at fan-out
  - `b246f7f` — ULID event IDs
  - `e2e6a17` — relocate marsys to `packages/framework/`
  - `ddf68a8` — content-addressed full-input capture (teammate's input-only version)

---

## Decision log (append-only)

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-09 | Adopt Mehdi's model-wrapper capture, drop ContextVars in favor of explicit `TraceContext` kwarg threading, package the helper as an async context manager (`capture_llm_call`). | Teammate's input-only capture leaves the LangSmith-grade gap open. ContextVars create implicit data flow + lifecycle footguns. Context manager keeps emit logic at the call site without decorator magic. |
| 2026-05-09 | Phase 1 complete: 16 new tests green, no regressions. | See [§5.1](#51-phase-1--full-payload-capture), [§6](#6-test-status). |
| 2026-05-09 | Keep NDJSON as the default local store; ship `OtelTraceWriter` as opt-in `TelemetrySink` for LangSmith / Phoenix / Langfuse export; drop SQLite writer entirely. | NDJSON wins decisively on concurrency (file-per-trace + content-addressed sidecar = zero contention); compactness is a wash; no team need for cross-run SQL. OTel fills the LangSmith-viewing requirement without adding a second local store. |
| 2026-05-09 | Phase 2 complete: `OtelTraceWriter` shipped + 14 new tests green; collector now keeps messages/tools inline alongside CAS refs so OTel-bound consumers see full content. | See [§5.2](#52-phase-2--otel-exporter), [§6](#6-test-status). |
| 2026-05-09 | Ported Mehdi's secret-word e2e from `MARSYS/tests/tracing/test_tracing_e2e.py` to `live_tests/tracing/secret_word_pipeline.py`, rewritten to match teammate's `live_tests/` conventions (no `test_` prefix, `--output-dir`, JSON summary, exit code) and adapted to the Phase-1+2 APIs (no `db_path`, no `OtelExportConfig`; instead `OtelTraceWriter` passed via `sinks=[…]`). Verifies on a real LLM run that every generation span has `input_messages` + `response_content` + `sampling_params` populated. | See [§6.1](#61-live-e2e-manual). |
| 2026-05-09 | First live LangSmith run surfaced six issues; all fixed. (1+2) `BaseAPIModel` had a buggy read-only `provider` property deriving the lowercased adapter class name (`"asyncopenrouter"`); replaced with stored `self.provider` / `self.model_name` instance attributes. (3) Parallel branches landed flat under the execution root; threaded `step_span_id` through `RealRuntime.step` → `OrchestratorStepResult` → `Branch.last_step_span_id` → `_spawn(parent_step_span_id=)` → `BranchCreatedEvent` → collector reparents under the dispatching step. (4) Every LLM call rendered twice in LangSmith because the legacy `GenerationEvent` emit in `step_executor` and the new `LLMRequestEvent`/`LLMResponseEvent` pair both produced spans; removed the legacy emit. (5) Outputs in LangSmith showed only `finish_reason: tool_calls` (empty content panel); ported Mehdi's local OTel attribute mapping which adds `langsmith.span.kind` / `openinference.span.kind`, `input.value` / `output.value` (OpenInference), indexed `gen_ai.prompt.{i}.role/content/tool_calls.{j}.*`, indexed `gen_ai.request.tools.{i}.*` + `llm.request.functions.{i}.*`, and a single-blob `gen_ai.completion` (deliberately not indexed — LangSmith renders empty when content=null+tool_calls under the indexed pattern). (6) AgentB instruction-following failure tightened. Bonus: `--model-name` argparse default was computed before `load_dotenv` ran, ignoring the user's `OPEN_ROUTER_MODEL=…:free` override and silently falling back to a paid model; moved `_load_env_early()` to the top of `parse_args()`. | 17 OTel writer tests + 9 capture + 8 collector all green; 168 pass total in tracing + orchestrator suite. |
| 2026-05-09 | Second LangSmith run: outputs were rendering as a raw "JSON Fields" panel (`finish_reason`, `role`, `tool_calls`, `usage_metadata` flat) instead of the expected AI message bubble with green tool-call chips. Root cause: `_build_output_message` was including `finish_reason` inside the message dict shipped as `output.value`. LangSmith's renderer inspects the exact key set of the parsed output and only renders the AI bubble when the dict matches the strict OpenAI-Chat-Completion shape `{role, content, tool_calls}` (plus optional `thinking`); any extra field flips it to the JSON-fields panel. `finish_reason` belongs on `gen_ai.response.finish_reasons` instead (already emitted), and usage on `gen_ai.usage.{input,output}_tokens` (already emitted). LangSmith overlays both onto the bubble automatically. Added regression test `test_output_value_is_strict_openai_chat_completion_shape` that asserts the output dict's key set is a subset of `{role, content, tool_calls, thinking}`. | 18 OTel tests, 169 pass total in tracing + orchestrator suite. |
| 2026-05-10 | Added native `provider="azure"` support for Azure OpenAI on the v1 inference endpoint (`.../openai/v1/`). Probed: Azure's v1 path exposes `/chat/completions` (NOT `/responses`, which `OpenAIAdapter` posts to) and accepts the api-key value as both an `api-key:` header and an `Authorization: Bearer` token — confirmed against `marsys-development-01-mehdi.openai.azure.com` with the deployment `gpt-5.4-mini`. Initial registration mapped `"azure" → OpenRouterAdapter` on the assumption that the right URL + right auth was sufficient. | First Azure attempt; failed against reasoning models — see next row. |
| 2026-05-10 | Replaced the `OpenRouterAdapter` registration with a standalone `AzureOpenAIAdapter` (~225 lines, no sibling-adapter inheritance). `OpenRouterAdapter` couldn't carry Azure: reasoning-class deployments (gpt-5.x, o-series) reject `max_tokens`, `temperature`, and the OpenRouter-style nested `{"reasoning": {"effort": ...}}` with 400 `unsupported_parameter`. The standalone adapter uses `api-key:` header (Azure-canonical), branches on `_is_reasoning_model(model_name)` to swap `max_tokens` ↔ `max_completion_tokens`, drop sampling params for reasoning models, and emit `reasoning_effort` as a top-level field. Reasoning-token usage parsed from `usage.completion_tokens_details`. Discovery driven by `live_tests/tracing/_debug_azure.py` which bypasses MARSYS's error wrapper to surface Azure's actual 400 body (which the wrapper had been hiding as `MODEL_API_UNKNOWN_ERROR / "Resource Not Found"`). Tracing tests still 133/134 passing. | See [§5.3](#53-phase-3--azure-openai-via-stock-openaiadapter). |
| 2026-05-10 | **Reverted the standalone `AzureOpenAIAdapter`.** User obtained access to a different Azure resource on the newer Cognitive Services / AI Foundry product (`<resource>.cognitiveservices.azure.com` rather than the older `<resource>.openai.azure.com`). The Foundry endpoint exposes `/openai/v1/responses` AND accepts `Authorization: Bearer <api-key value>` — both verified via curl. That's exactly what stock `OpenAIAdapter` posts to, so Azure works as pure config: no adapter, no factory entry, no env-var-map change, no `Literal[...]` constraint update. Deleted: `src/marsys/models/adapters/azure_openai.py`, `live_tests/tracing/_debug_azure.py`. Reverted: `factory.py`, `adapters/__init__.py`, `models.py` imports + `env_var_map` + `Literal` constraint. Live test keeps `--provider=azure` as an env-var-source CLI selector, but the resulting `ModelConfig` uses `provider="openai"`. Future-proofing: if anyone reports an `openai.azure.com` resource (the older product) doesn't work, re-introduce the adapter from this row's predecessor — its design is documented above. | See [§5.3](#53-phase-3--azure-openai-via-stock-openaiadapter); 133 tracing tests still passing. |
| 2026-05-10 | **Fixed two `OpenAIAdapter` Responses-API parsing bugs that surfaced once Azure traces reached LangSmith.** First live Azure run produced traces with `response_content=""` and `response_metadata.usage.{prompt_tokens, completion_tokens}=null`. Two root causes in `harmonize_response`: (1) `content` initialized to `""` rather than `None`, so tool-only responses (no `message` item in the `output` array) carried an empty-string content into the trace — LangSmith flips its renderer to a JSON-fields panel for `content: ""` while it correctly renders the AI-message bubble for `content: null`. (2) Token usage read chat-completions field names (`prompt_tokens` / `completion_tokens` / `reasoning_tokens`) from a Responses API `usage` block that uses `input_tokens` / `output_tokens` / `output_tokens_details.reasoning_tokens`. Fixed both: `content` now defaults to `None`, usage parsing tries Responses-API names first and falls back to chat-completions for back-compat. Re-recreated `live_tests/tracing/_debug_azure.py` (had been deleted in the revert) targeted at `AsyncOpenAIAdapter` — useful diagnostic when MARSYS's wrapper masks Azure responses as `MODEL_API_INVALID_MODEL_ERROR / "Resource Not Found"` (also noted: many user-visible errors are actually 4xx with detailed bodies that the wrapper hides). | 138 tracing+models tests passing (1 pre-existing Python 3.14 failure). |

<!-- Future agents: append your decisions here. -->

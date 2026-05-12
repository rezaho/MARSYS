# 06 — Observability

## What observability covers in Spren

Three first-class signals, all derived from the marsys framework's tracing module (`coordination/tracing/`):

1. **Traces** — full hierarchical span tree per run. Primary user-facing artifact for "why did this happen."
2. **Cost** — tokens × provider rate, aggregated per run + per day + per workflow. First-class from v0.3 (SP-010); not a later retrofit.
3. **Latency / status counters** — basic metrics surfaced as charts on the run history page.

The framework ships one seam: the `TelemetrySink` ABC. Sinks register with `TraceCollector`; the collector calls `publish_span(span)` once per closed span after running it through the configured `SecretRedactor`. The shipped default sink is `NDJSONTraceWriter` (one closed span per line on disk — Spren reads this for SSE replay and run inspection). Spren's own `SprenTelemetrySink` (in-tree at `packages/spren/src/spren/telemetry/`) is a `TelemetrySink` for the `python my_workflow.py → Spren UI` flow, posting the redacted span stream to Spren's REST surface. Vendor adapters (LangSmith, Phoenix, Langfuse, OpenInference) fit the same ABC and are contributor-welcome, but no in-tree shipped adapter beyond `NDJSONTraceWriter` and Spren's own.

## Trace pipeline

```
marsys execution
       │
       ▼
  EventBus (packages/framework/src/marsys/coordination/event_bus.py)
       │
       ├──────────────────────────────────────────────┐
       │                                              ▼
       │                              ┌────────────────────────────────┐
       │                              │ TraceCollector                 │
       │                              │ (single EventBus subscriber;   │
       │                              │  builds hierarchical Span tree)│
       │                              └──────────┬─────────────────────┘
       │                                         │ closed span
       │                                         │
       │                                         ├─► SecretRedactor.redact(span.attributes)
       │                                         │     │
       │                                         │     ├─► NDJSON writer
       │                                         │     │   <data-dir>/data/runs/{id}/trace.ndjson
       │                                         │     │   (one closed span per line, flushed per span)
       │                                         │     │
       │                                         │     └─► TelemetrySink fan-out (Framework v0.4)
       │                                         │           ├─► SprenTelemetrySink
       │                                         │           ├─► (LangSmith / Phoenix / Langfuse adapters
       │                                         │           │    fit same protocol — contributor-welcome)
       │                                         │           └─► ...
       │
       │  ┌────────────────────────────────────────────┐
       └─►│ AG-UI translator (framework-side)          │
          │ marsys.transport.aggui (Framework v0.3 S06)│
          │ subscribes to BranchCreatedEvent /         │
          │ BranchCompletedEvent / generation +        │
          │ tool-call lifecycle; emits AG-UI events    │
          │ as AsyncIterator[AGUIEvent] from           │
          │ AGUIEventStream(orchestra, run_id)         │
          │                                            │
          │ Spren wraps in SSE HTTP endpoint at        │
          │ /v1/runs/{id}/events (Spren v0.3 S04)      │
          └────────────────────────────────────────────┘
```

The NDJSON writer and the `TelemetrySink` fan-out are framework-side, hooked into `TraceCollector` at every `span.close(...)` site (`_handle_branch_completed`, `_handle_agent_complete`, `_handle_tool_call`, `_handle_generation`, `_handle_final_response`, plus the orphan-close loop in `finalize`). They consume closed spans, never raw events.

The AG-UI translator is framework-side (`marsys.transport.aggui`), subscribing to `EventBus` directly for in-flight lifecycle events (`BranchCreatedEvent` / `BranchCompletedEvent` from `coordination/events.py` plus generation + tool-call lifecycle events from `coordination/status/events.py`). It emits AG-UI events as `AsyncIterator[AGUIEvent]` from `AGUIEventStream(orchestra, run_id)`; Spren wraps the iterator in an SSE HTTP endpoint at `/v1/runs/{id}/events`. The translator is reusable by any consumer who wants live UI streaming (MARSYS Cloud's hosted dashboard, third-party AG-UI clients, custom UIs) — Spren is one consumer of a generic seam, matching the `TelemetrySink` pattern.

Different layer, different consumer surface, different cadence — a closed span vs. a live event.

Both surfaces are append-only. Crash mid-run loses neither — the NDJSON file is durable through any crash up to the moment of the crash, and the SSE channel is reconnect-tolerant (clients tail-read the file from start to current EOF, then resume from latest `Last-Event-ID`).

### Why NDJSON (replaces the legacy JSON-at-end writer)

The legacy writer built a single JSON object in memory and wrote it once at run completion. Two problems:

1. **Crash mid-run loses the trace** — entire file fails to write.
2. **No live streaming** — UI can't render until run is done.

The NDJSON writer (`packages/framework/src/marsys/coordination/tracing/writers/ndjson_writer.py`):

- One JSON object per closed span (newline-delimited)
- Each span is appended as it closes via the per-span `write_span` hook on `TraceCollector`; `f.flush()` after every line. `fsync` is opt-in via `fsync_per_span=True` (default off — we trust OS buffer for performance)
- Tail-readable: a late SSE subscriber reads the file from start to current EOF, then switches to live `EventBus` subscription
- Each line carries `schema_version: 1` so future readers can pin against the contract
- A terminal `stream_completed` marker is written on `close()`; missing marker means the run crashed and the reader surfaces `completion_status == "crashed"`
- `event_id` and `span_id` are ULIDs (monotonic-orderable per process), so SSE `Last-Event-ID` resume works without server-side ordering tables

The legacy `JSONFileTraceWriter` is removed; only the NDJSON writer ships from Framework v0.3 onward.

This change is in scope for the framework, not Spren-only — see [`01-system-context.md`](./01-system-context.md) "What Spren provides to the framework."

## Span schema (framework wire format)

Each NDJSON line is a closed span. `start_time` / `end_time` / `ts` are emitted as floats (epoch seconds) by the framework — consumers convert to ISO 8601 for display if needed; the on-disk format does not do this conversion.

```json
{
  "schema_version": 1,
  "ts": 1714220645.123,            // float, writer emission time (epoch seconds)
  "trace_id": "01J9X4ABCDEFGHJKMN", // ULID, uppercase Crockford-base32; same as run_id
  "span_id": "01J9X4ABCDEFGHJKMP",  // ULID per span
  "parent_span_id": "01J9X4ABCDEFGHJKMR" | null,
  "kind": "execution|branch|step|generation|tool",  // lowercase, five kinds total
  "name": "human-readable label",
  "start_time": 1714220641.000,    // float, span start (epoch seconds)
  "end_time": 1714220645.123,      // float, span close (epoch seconds)
  "duration_ms": 4123.0,
  "status": "ok|error",
  "attributes": {                   // kind-specific; see below
    // ...
  },
  "events": [...],                  // optional; validation decisions etc. attached to step spans
  "links": [...]                    // optional; cross-span relationships (e.g. convergence)
}
```

Validation decisions, user-interaction prompts, and the final-response summary are NOT separate `kind` values; they are events / attributes on the surrounding step or root execution span (see `Span.events` and `Span.attributes`).

### Attributes by kind

- `execution` — `{task_summary, topology_summary, agent_names, config_summary, success, total_steps, total_duration, final_response_summary}`
- `branch` — `{branch_name, source_agent, target_agents, trigger_type, branch_type, is_parallel, total_steps, success}`
- `step` — `{agent_name, step_number, request_summary, action_type, next_agents?, success, duration, error?}`
- `generation` — `{agent_name, model_name, provider, prompt_tokens, completion_tokens, reasoning_tokens, response_time_ms, finish_reason, has_thinking, has_tool_calls}`
- `tool` — `{tool_name, agent_name, arguments?, result_summary?}`

Content fields (full prompts / outputs) are excluded by default and gated through `TracingConfig.include_message_content` / `include_tool_results`. Spren reads them from the upstream trace if user clicks "show full" in the UI.

### Diagnostic and terminal lines

Two non-span line types appear in the file. Consumers reading individual spans MUST filter them out by `kind`:

- `kind == "stream_event"` — diagnostic emitted on queue overflow or writer self-disable. `attributes` describes the event (e.g. `{"event": "dropped_span", "dropped_span_count": N}`).
- `kind == "stream_completed"` — terminal marker, exactly once on close. `attributes` summarizes counts: `{total_spans, disk_error_count, dropped_span_count, disabled_dropped_count, disabled}`. Missing marker on EOF means the writer crashed; readers can surface this via `NDJSONTraceReader.completion_status == "crashed"`.

## TelemetrySink ABC (Framework v0.3)

The `TelemetrySink` ABC at `packages/framework/src/marsys/coordination/tracing/sink.py` is the seam an external observability backend uses to receive workflow execution. The ABC is span-shaped (matches what every modern observability vendor consumes), hierarchical via `parent_span_id`, and async (sinks may do network I/O):

```python
class TelemetrySink(ABC):
    @abstractmethod
    async def publish_span(self, span: 'Span') -> None: ...
    @abstractmethod
    async def close(self) -> None: ...
```

Sinks register with `TraceCollector` (NOT `EventBus` directly — the collector already does the span-tree-building work; subscribing raw events would duplicate it). At every `span.close(...)` site, `TraceCollector` runs the closed span through the configured `SecretRedactor` (mutates in place) and fans it out to all registered sinks via `publish_span`. Sink-side exceptions are caught and logged at the framework boundary; a misbehaving sink does not stop the run.

**Orchestra wiring** (sinks are passed via `TracingConfig.sinks`; `Orchestra` builds the final list as `[NDJSONTraceWriter(execution_config.tracing)] + execution_config.tracing.sinks`):

```python
tracing_config = TracingConfig(
    output_dir="./traces",
    sinks=[SprenTelemetrySink(...)],
    redactor=SecretRedactor(extra_deny=["my_custom_key"]),
)
execution_config = ExecutionConfig(tracing=tracing_config)
result = await Orchestra.run(topology, task, execution_config=execution_config)
```

The default `NDJSONTraceWriter` is always added; user-supplied `sinks` are appended. `redactor` defaults to `SecretRedactor()` (the default deny-list); pass `NoRedactionRedactor()` to opt out.

**Spren's adapter** lives in-tree at `packages/spren/src/spren/telemetry/`. It posts the redacted span stream to Spren's REST surface so a `python my_workflow.py` invocation surfaces in the Spren UI alongside runs Spren itself dispatched. **Vendor adapters** (LangSmith, Phoenix / OpenInference, Langfuse) fit the same ABC — span-shaped, hierarchical, async-batched at the sink — so a third-party `marsys-langsmith` / `marsys-phoenix` / `marsys-langfuse` package is a thin translation layer rather than a parallel `TraceCollector`. None ship in-tree.

## SecretRedactor

`ToolCallEvent.arguments: Dict[str, Any]` and other span attribute payloads may carry API keys, OAuth tokens, bearer credentials. The `SecretRedactor` at `packages/framework/src/marsys/coordination/tracing/redactor.py` scrubs them at the fan-out boundary inside `TraceCollector._stream_span`:

- Default deny-list (case-insensitive, **word-boundary match against dict keys**, recursive on nested dicts + `event['attributes']` dicts in `span.events` + `link['attributes']` dicts in `span.links`): `api_key`, `apikey`, `token`, `authorization`, `auth`, `secret`, `password`, `bearer`, `cookie`, `session`, `credential`. Matches replace the value with `[REDACTED]`; structure is preserved.
- Word boundaries treat `_` and `-` (and any non-alphanumeric character) as separators. So `auth_token` and `x_api_key` redact, but `prompt_tokens` (an LLM token-count metric) and `authority` do NOT.
- Composable: `SecretRedactor(extra_deny=["my_custom_key"])` extends the deny-list per-project.
- Explicit opt-in to raw passthrough: `NoRedactionRedactor()` — caller accepts the leak risk.

The redactor mutates spans in place once per fan-out, so **all consumers see the same redacted view** — `NDJSONTraceWriter` writes the redacted span to disk, vendor sinks receive the redacted span over the network, in-memory `TraceTree` viewers read the redacted span. There is no "redacted-for-sinks-but-original-on-disk" mode by design.

## Schema versioning

Every cross-boundary payload (`Span` lines on disk, `Workflow` / `WorkflowRef` / `RunStarted` / `RunCompleted` to sinks, `StateSnapshot` for pause/resume in v0.4) carries `schema_version: int = 1`. Framework bumps on breaking shape changes. Consumers compare on receive — mismatch surfaces a clear warning; the consumer chooses graceful-degrade or hard-error. The framework does NOT enforce; the sink / reader / writer does. v0.4 ships no migration tooling — `IncompatibleSnapshotError` (for snapshots) and a sink-side warning log (for spans) are the expected mismatch paths.

## Cost calculation

`src/spren/cost.py`:

- Maintains a rate table: `{provider, model} → {input_per_1m_usd, output_per_1m_usd, reasoning_per_1m_usd?}`
- Source: a YAML file in the package, updated periodically
- On every `GenerationEvent`:
  - `cost = (prompt_tokens × in_rate + completion_tokens × out_rate + reasoning_tokens × reasoning_rate) / 1_000_000`
- Per run: aggregated into `runs.total_cost_usd`
- Per day / per workflow: derived via SQL queries

Budget enforcement (v0.3: hard cap on a per-day basis):

- Daily budget cap (settings)
- Per-run budget cap (settings)
- Meta-agent reads remaining budget; refuses to dispatch a run that's projected to exceed (using historical-mean cost for the workflow as the projection)

## Run history queries (SQL)

Standard queries powering the UI:

```sql
-- Last 50 runs across all workflows
SELECT id, workflow_id, status, started_at, total_duration_ms, total_cost_usd
FROM runs
ORDER BY created_at DESC LIMIT 50;

-- Cost rollup last 30 days
SELECT date(started_at) AS day, SUM(total_cost_usd) AS spend
FROM runs WHERE started_at >= date('now', '-30 days')
GROUP BY day;

-- Workflows ranked by total cost (helps user identify expensive workflows)
SELECT w.id, w.name, COUNT(r.id) AS runs, SUM(r.total_cost_usd) AS total_spend
FROM workflows w LEFT JOIN runs r ON r.workflow_id = w.id
GROUP BY w.id ORDER BY total_spend DESC;
```

SQLite FTS5 index on `runs.task_input` and `runs.final_response` for "find runs that mentioned X" search.

## Retention

- Default: keep `trace.ndjson` for the last 200 runs OR 7 days, whichever is larger
- Older traces: gzipped + moved to `<data-dir>/data/archived/runs/{id}.ndjson.gz`. The SQLite row stays.
- Archival runs daily on app start
- User-configurable in settings: `retention.runs.max_count`, `retention.runs.max_age_days`, `retention.runs.archive_format`

## Out of scope

- Distributed tracing across processes (single-process by design — Spren is a local daemon).
- In-tree vendor adapter packages (LangSmith / Phoenix / Langfuse). The protocol is shaped to fit them; the actual adapters ship as third-party packages.
- OTLP-over-HTTP export. The `TelemetrySink` ABC is the framework's seam for any wire-format export; an OTLP adapter would be a `TelemetrySink` implementation written against the OTel SDK. Out of scope for this repo.
- Alerting on errors / cost thresholds beyond the per-day budget cap and per-run budget cap (SP-013). Notifications via channels are a v0.4 concern.
- Long-term metrics warehouse. SQL queries on the `runs` table cover v0.3 needs; if a project outgrows them, a `TelemetrySink` to a warehouse is the path.

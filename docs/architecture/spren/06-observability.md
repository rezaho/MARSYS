# 06 — Observability

## What observability covers in Spren

Three first-class signals, all derived from the marsys framework's existing `EventBus` + tracing:

1. **Traces** — full hierarchical span tree per run. Primary user-facing artifact for "why did this happen."
2. **Cost** — tokens × provider rate, aggregated per run + per day + per workflow.
3. **Latency / status counters** — basic metrics surfaced as charts on the run history page.

There is NO separate Prometheus/OTel exporter at this stage. Everything is local; metrics are derived from the same trace files. A later release may add OTel emit for users who want to ship to Langfuse / Phoenix / etc. via the framework's `TelemetrySink` protocol.

## Trace pipeline

```
marsys execution
       │
       ▼
  EventBus (in-process pub/sub, packages/framework/src/marsys/coordination/event_bus.py)
       │ ┌────────────────────────────────────────────┐
       ├─►│ NDJSON writer                              │  (replaces existing json_writer; see below)
       │ │ <data-dir>/data/runs/{id}/trace.ndjson     │
       │ └────────────────────────────────────────────┘
       │
       │ ┌────────────────────────────────────────────┐
       └─►│ AG-UI translator → SSE channel             │
         │ /v1/runs/{id}/events                        │
         └────────────────────────────────────────────┘
```

Both subscribers are append-only. Crash mid-run loses neither — the file is durable and the SSE channel is reconnect-tolerant (clients re-fetch the file then resume from latest event id).

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

The legacy `JSONFileTraceWriter` is removed; only the NDJSON writer ships in v0.4.

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

## What's NOT here in v0.3

- Distributed tracing across processes (we're single-process)
- OpenTelemetry export (v0.3; will use OTel GenAI semantic conventions when stable)
- Alerting on errors / cost thresholds (v0.3)
- Long-term metrics warehouse (v0.3; for now, SQL queries on the runs table are enough)

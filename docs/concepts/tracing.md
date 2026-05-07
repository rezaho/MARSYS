# Tracing

Structured execution traces for multi-agent workflow observability.

## Overview

The tracing module captures the full execution flow of an Orchestra run as a hierarchical span tree, enabling live observability and post-hoc analysis of:

- **Agent steps** — which agents ran, in what order, and how long each took
- **LLM generations** — token usage, model identity, API latency, finish reason per generation
- **Validation decisions** — what action type was determined and what routing was chosen
- **Tool calls** — which tools were invoked, their arguments, results, and duration
- **Branch lifecycle** — branch creation, parallel execution, and convergence
- **Convergence** — which child branches fed into which parent, and what was aggregated

Traces stream as **newline-delimited JSON** (NDJSON), one object per closed span, written incrementally as the run progresses. A late subscriber can tail-follow the file, and a process crash mid-run preserves every span emitted up to the crash.

## Core Components

### TracingConfig

Controls whether tracing is enabled and what content to capture:

```python
from marsys.coordination.tracing.config import TracingConfig

config = TracingConfig(
    enabled=True,
    output_dir="./traces",
    include_generation_details=True,   # token counts, model metadata
    include_message_content=True,       # final response summary on root
    include_tool_results=True,          # tool args and result summaries
)
```

### TraceCollector

EventBus consumer that subscribes to execution events and builds the span tree:

```python
from marsys.coordination.tracing import TraceCollector
```

You don't create this directly — Orchestra creates it automatically when `TracingConfig.enabled=True`, and registers an `NDJSONTraceWriter` as its single writer.

### Span

A single unit of work in the trace. Spans form a tree via `parent_span_id`:

```python
from marsys.coordination.tracing.types import Span
```

Span kinds: `execution`, `branch`, `step`, `generation`, `tool` — all lowercase.

Validation decisions are captured as `events` on step spans (not separate spans). Convergence is captured as `links` and events on both the parent branch span and the convergence step span.

`span_id`, `parent_span_id`, and `trace_id` are ULIDs — 26-character uppercase Crockford-base32 strings, monotonic-orderable per process.

### TraceTree

A complete trace rooted at an execution span:

```python
from marsys.coordination.tracing.types import TraceTree

tree = TraceTree.from_ndjson(pathlib.Path("./traces/01J....ndjson"))
print(tree.root_span.duration_ms, len(tree.orphans))
```

`from_ndjson` is the post-mortem entry point: it walks the file once, indexes spans by `span_id`, and attaches each span as a child of its `parent_span_id`. Spans whose parent was never written (e.g. because the run crashed) populate `tree.orphans` rather than being silently dropped.

### NDJSONTraceWriter

The streaming writer used by Orchestra by default. One JSON object per closed span; lazy file open at `{config.output_dir}/{trace_id}.ndjson` on the first span. The writer enqueues spans on a bounded `asyncio.Queue` and a single dedicated drain task does the disk I/O — `write_span` is non-blocking on the calling path so a slow disk doesn't back-pressure event emission.

```python
from marsys.coordination.tracing import NDJSONTraceWriter
```

A terminal `stream_completed` marker is written on `close()`. Missing marker = process crashed before close, surfaced to readers as `completion_status == "crashed"`.

### NDJSONTraceReader

The streaming reader. Two modes:

```python
from marsys.coordination.tracing import NDJSONTraceReader

reader = NDJSONTraceReader(path)

# Post-mortem: drain the file, get span dicts
for span in reader.stream():
    print(span["kind"], span["name"])
print(reader.completion_status)  # "complete" | "truncated" | "crashed"

# Live: tail-follow a running trace
for span in reader.stream(follow=True):
    print(span["kind"], span["name"])
# Loop exits when the stream_completed marker is read.
```

## Basic Usage

Enable tracing by adding `TracingConfig` to your `ExecutionConfig`:

```python
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing.config import TracingConfig

result = await Orchestra.run(
    task="Research quantum computing",
    topology=topology,
    agent_registry=AgentRegistry,
    execution_config=ExecutionConfig(
        tracing=TracingConfig(
            enabled=True,
            output_dir="./traces",
        ),
    ),
)
```

After the run completes, find the trace at `./traces/{trace_id}.ndjson`. Read writer counts from `result.metadata["tracing"]`:

```python
print(result.metadata["tracing"])
# {'total_spans': 142, 'disk_error_count': 0, 'dropped_span_count': 0,
#  'disabled_dropped_count': 0, 'disabled': False}
```

For `auto_run`, the same config applies since it uses Orchestra internally:

```python
result = await agent.auto_run(
    "Research AI trends",
    max_steps=10,
    execution_config=ExecutionConfig(
        tracing=TracingConfig(enabled=True),
    ),
)
```

## Trace Output

Each line of the NDJSON file is a closed span. Abbreviated example:

```jsonl
{"schema_version":1,"ts":1714220642.5,"trace_id":"01J9X4...","span_id":"01J9X5...","parent_span_id":"01J9X4...","name":"Generation: claude-sonnet-4","kind":"generation","start_time":1714220641.3,"end_time":1714220642.5,"duration_ms":1200.5,"status":"ok","attributes":{"prompt_tokens":150,"completion_tokens":80}}
{"schema_version":1,"ts":1714220645.1,"trace_id":"01J9X4...","span_id":"01J9X6...","parent_span_id":"01J9X4...","name":"Step 0: Coordinator","kind":"step","start_time":1714220641.0,"end_time":1714220645.1,"duration_ms":4123.0,"status":"ok","attributes":{"agent_name":"Coordinator","step_number":0,"action_type":"invoke_agent"},"events":[{"name":"validation_decision","attributes":{"is_valid":true}}]}
{"schema_version":1,"ts":1714220650.0,"kind":"stream_completed","attributes":{"total_spans":42,"disk_error_count":0,"dropped_span_count":0,"disabled_dropped_count":0,"disabled":false}}
```

Reconstructing the tree gives the same shape `TraceTree.to_dict()` produces:

```json
{
  "trace_id": "01J9X4...",
  "session_id": "session-001",
  "metadata": {"task_summary": "...", "agent_names": ["Coordinator", "Researcher", "FactChecker"]},
  "root_span": {
    "name": "Orchestra.run",
    "kind": "execution",
    "duration_ms": 12450.3,
    "status": "ok",
    "children": [
      {"name": "Branch: main", "kind": "branch",
       "links": [{"linked_span_id": "...", "relationship": "convergence"}],
       "children": [
         {"name": "Step 0: Coordinator", "kind": "step",
          "attributes": {"action_type": "invoke_agent"},
          "events": [{"name": "validation_decision", "attributes": {"next_agents": ["Researcher", "FactChecker"]}}],
          "children": [
            {"name": "Generation: claude-sonnet-4", "kind": "generation", "duration_ms": 1200.5,
             "attributes": {"prompt_tokens": 150, "completion_tokens": 80}}
          ]}
       ]}
    ]
  }
}
```

## Span Hierarchy

Each execution produces a tree with this structure:

```
Execution Span (one per Orchestra.run)
├── Branch Span (initial branch, e.g. "main")
│   ├── Step Span (agent step that triggers parallel_invoke)
│   │   ├── Generation Span (LLM call details)
│   │   ├── Tool Span (per tool invocation)
│   │   └── Validation Event (routing decision — stored as event, not span)
│   ├── Step Span (convergence step — receives aggregated results)
│   │   ├── links: [{child_branch_1}, {child_branch_2}]  ← convergence
│   │   ├── events: [convergence, validation_decision]
│   │   └── Generation Span ...
│   └── ...
├── Branch Span (parallel child 1)
│   └── Step Span → Generation → Tool → ...
├── Branch Span (parallel child 2)
│   └── Step Span → Generation → Tool → ...
└── (Convergence links also on parent branch span)
```

## Schema Versioning

Each line carries `schema_version: 1`. The format follows additive-only evolution within v1:

- New top-level fields and new keys inside `attributes` may be added without bumping the version. Readers MUST ignore unknown fields silently.
- Removed or renamed fields, type changes, and semantic changes require a new schema version.
- The reader rejects `schema_version > SUPPORTED_SCHEMA_VERSION` with `NDJSONVersionError`.

## Crash Detection

The writer's lifecycle gives consumers a clear way to detect partial traces:

- File ends with `stream_completed` marker → `reader.completion_status == "complete"`. Healthy run.
- File ends without marker but with full lines → `"crashed"`. Process died before `close()`. Spans before death are preserved.
- File ends with a partial last line (no trailing `\n`) → `"truncated"`. Writer was mid-write or file was copied during a write. Reader yields the N-1 full spans.

Use this together with `OrchestraResult.metadata["tracing"]` (writer counts surfaced by Orchestra after `close()`) to alert on `disabled` or non-zero `dropped_span_count` from a programmatic consumer.

## Best Practices

### Keep `./traces/` out of version control
By default trace files contain full LLM prompts and responses. Add the directory to `.gitignore`.

### Choose a meaningful trace organization
Files are keyed by `trace_id`. If you want a stable filename per logical run, derive a separate index file in your application — the trace_id alone is opaque to humans.

### Be mindful of content sensitivity
Set `include_message_content=False` to skip the final response summary on the root span. Set `include_tool_results=False` to skip tool arguments and result summaries (useful when tools handle credentials or PII). For finer-grained redaction, write a custom `TraceWriter` subclass that scrubs `Span.attributes` in `write_span` before forwarding.

### Use `fsync_per_span=True` only when you need it
The default writer flushes per line and fsyncs once on close — fast and durable enough for typical workloads. `fsync_per_span=True` adds an `os.fsync` per write; only enable when you need durability against kernel-level crashes between span writes.

## Limitations

- One writer per file. Two concurrent runs use distinct trace_ids and produce distinct files; cross-trace mixing into a shared file is not supported.
- No OTel-protocol export to LangSmith / Phoenix / Langfuse. NDJSON-on-disk is the durability half; OTel-on-the-wire will land via a separate `TelemetrySink` protocol.
- Tracing for standalone `agent.run()` calls (outside Orchestra) is not exercised by Orchestra's lifecycle, so the writer's `close()` won't run automatically — call it explicitly if you instantiate the writer manually.

## Related Documentation

- [Tracing API Reference](../api/tracing.md) — complete class and method reference
- [Configuration](../getting-started/configuration.md) — `ExecutionConfig` setup
- [Architecture Overview](../architecture/overview.md) — how tracing fits in the system

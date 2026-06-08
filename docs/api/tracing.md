# Tracing API

Complete API reference for execution tracing and observability in multi-agent workflows.

## Overview

The Tracing API captures the full hierarchy of an Orchestra run — execution, branches, agent steps, LLM generations, tool calls, and validation decisions — and streams it to disk as one JSON object per closed span. Mid-run crashes preserve every span emitted up to the crash; live consumers tail-follow the file as spans arrive.

## Core Classes

### TracingConfig

Configuration for execution tracing.

**Import:**
```python
from marsys.coordination.tracing.config import TracingConfig
```

**Constructor:**
```python
TracingConfig(
    enabled: bool = False,
    output_dir: str = "./traces",
    include_message_content: bool = True,
    include_tool_results: bool = True,
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | `bool` | Enable trace collection | `False` |
| `output_dir` | `str` | Directory for trace NDJSON files | `"./traces"` |
| `include_message_content` | `bool` | Include the final response summary on the root execution span | `True` |
| `include_tool_results` | `bool` | Include tool arguments and result summaries in tool spans | `True` |

### TraceCollector

EventBus consumer that builds hierarchical span trees from execution events. Hooked into the streaming writer via per-span callbacks.

**Import:**
```python
from marsys.coordination.tracing import TraceCollector
```

**Constructor:**
```python
TraceCollector(
    event_bus: EventBus,
    config: TracingConfig,
    writers: Optional[List[TraceWriter]] = None,
)
```

!!! tip "Automatic Creation"
    You don't create `TraceCollector` directly. Orchestra creates it when `TracingConfig.enabled=True` and registers an `NDJSONTraceWriter`.

**Key Methods:**

#### finalize
```python
async def finalize(session_id: str) -> Optional[TraceTree]
```
Finalize the trace for a session. Closes any open spans (marking them as `error`), streams them through `write_span`, and returns the completed `TraceTree`.

#### close
```python
async def close() -> None
```
Drain all writers, write the `stream_completed` marker, and close file handles. Called by Orchestra after `finalize()` in a try/finally bounded by 5 seconds. Idempotent.

### Span

A single unit of work in the execution trace.

**Import:**
```python
from marsys.coordination.tracing.types import Span
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `span_id` | `str` | ULID (26-char uppercase Crockford-base32) |
| `parent_span_id` | `Optional[str]` | Parent span for tree nesting; `None` only on the root execution span |
| `trace_id` | `str` | ULID identifying the run; same on every span in one `Orchestra.run()` |
| `name` | `str` | Human-readable label (e.g., `"Step 3: Researcher"`) |
| `kind` | `str` | Lowercase, one of `execution`, `branch`, `step`, `generation`, `tool` |
| `start_time` | `float` | Epoch seconds |
| `end_time` | `Optional[float]` | Epoch seconds (set on close) |
| `duration_ms` | `Optional[float]` | Computed on close |
| `status` | `str` | `"ok"` or `"error"` |
| `attributes` | `Dict[str, Any]` | Kind-specific data |
| `events` | `List[Dict]` | Instant events attached to the span (e.g. validation decisions) |
| `children` | `List[Span]` | Child spans (in-memory only; the on-disk NDJSON is flat — children are reconstructed by `parent_span_id`) |
| `links` | `List[Dict]` | Cross-branch causal links (e.g. convergence) |

**Key Methods:**

#### close
```python
def close(end_time: Optional[float] = None, status: Optional[str] = None) -> None
```
Close the span, computing `duration_ms` from `start_time` to `end_time`.

#### add_event
```python
def add_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None
```
Add an instant event to this span (e.g. a validation decision).

#### to_dict
```python
def to_dict() -> Dict[str, Any]
```
Serialize the span (and its children) to a nested dict. Used by `TraceTree.to_dict()` for post-mortem reconstruction.

### TraceTree

A complete execution trace rooted at an execution span.

**Import:**
```python
from marsys.coordination.tracing.types import TraceTree
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | ULID; matches `Span.trace_id` on every span in the trace |
| `session_id` | `str` | Orchestra session ID (caller-supplied or `uuid.uuid4()` fallback) |
| `root_span` | `Span` | Root execution span containing the full tree |
| `metadata` | `Dict[str, Any]` | Task summary, agent names, etc. |
| `orphans` | `List[Span]` | Spans whose `parent_span_id` was not present in the file when reconstructed by `from_ndjson`. Empty for healthy runs; non-empty after a crash that left dangling children. |

**Key Methods:**

#### to_dict
```python
def to_dict() -> Dict[str, Any]
```
Serialize the full trace tree for JSON output. `orphans` is included only when non-empty.

#### from_ndjson
```python
@classmethod
def from_ndjson(cls, path: pathlib.Path) -> "TraceTree"
```
Reconstruct a `TraceTree` from an NDJSON trace file. Order-independent: walks the file once, indexes spans by `span_id`, then attaches each span as a child of its `parent_span_id`. The execution-kind span with `parent_span_id is None` becomes the root; spans referencing an unknown parent populate `orphans`.

### TraceWriter

Abstract base for trace output backends.

**Import:**
```python
from marsys.coordination.tracing.writers.base import TraceWriter
```

**Methods:**

| Method | Description |
|--------|-------------|
| `async write(trace: TraceTree) -> None` | Finalize-only hook; called once at `TraceCollector.finalize`. The streaming writer makes this a no-op. |
| `async close() -> None` | Release resources. Must be idempotent. |
| `async write_span(span: Span) -> None` | Streaming hook; called by `TraceCollector` at every span-close site. **Default no-op** for non-streaming writers; streaming writers override. |

### NDJSONTraceWriter

Streaming NDJSON writer. One JSON object per closed span on its own line, plus a terminal `stream_completed` marker. The default writer for `Orchestra` when tracing is enabled.

**Import:**
```python
from marsys.coordination.tracing.writers import NDJSONTraceWriter
```

**Constructor:**
```python
NDJSONTraceWriter(
    config: TracingConfig,
    *,
    fsync_per_span: bool = False,
    queue_maxsize: int = 10000,
)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `config` | `TracingConfig` | Tracing configuration | Required |
| `fsync_per_span` | `bool` | Call `os.fsync` after every line. Default off (the writer flushes per line and fsyncs once on close). | `False` |
| `queue_maxsize` | `int` | Bounded queue size; on overflow the writer drops the oldest queued span and emits a `stream_event` diagnostic line. | `10000` |

**Output:** `{config.output_dir}/{trace_id}.ndjson` — one file per `Orchestra.run()`.

**Lifecycle:**
- File is opened lazily on the first `write_span` call (the `trace_id` derives from the first span).
- A second `write_span` with a different `trace_id` raises `ValueError` — a writer is bound to one trace.
- `write_span` enqueues the span for the dedicated drain task and returns immediately. Disk I/O happens off the calling path.
- On `OSError`: the disk-error counter increments and the writer logs at WARNING. After 100 consecutive errors the writer self-disables and subsequent spans drop into `disabled_dropped_count`. Throttled logging avoids flooding.
- `close()` sends a sentinel to the drain task, waits up to 5 s for it to finish, writes the `stream_completed` marker, fsyncs, and closes the file. Idempotent.

**Introspection properties:**
| Property | Type | Description |
|----------|------|-------------|
| `total_spans` | `int` | Spans successfully written |
| `disk_error_count` | `int` | Cumulative `OSError`s on writes |
| `consecutive_disk_errors` | `int` | Resets on success; triggers self-disable at 100 |
| `dropped_span_count` | `int` | Spans dropped due to queue overflow |
| `disabled_dropped_count` | `int` | Spans dropped after self-disable |
| `disabled` | `bool` | Whether the writer self-disabled |

### NDJSONTraceReader

Streaming reader with tail-follow support. Used for live consumption (Spren SSE, hosted observability) and post-mortem analysis.

**Import:**
```python
from marsys.coordination.tracing import NDJSONTraceReader, NDJSONVersionError
```

**Constructor:**
```python
NDJSONTraceReader(path: pathlib.Path)
```

**Methods:**

#### stream
```python
def stream(follow: bool = False, poll_interval: float = 0.1) -> Iterator[dict]
```
Yield span dicts in file order. Marker / diagnostic lines (`kind == "stream_completed"`, `kind == "stream_event"`) are not yielded but DO update `completion_status`. With `follow=True`, polls for new bytes after EOF and stops when the `stream_completed` marker is read or the consumer breaks out.

#### completion_status
```python
@property
def completion_status() -> Literal["complete", "truncated", "crashed"]
```
File completion state, meaningful after `stream()` returns:
- `complete` — file ended with a `stream_completed` marker.
- `truncated` — final line was partial (writer crashed mid-write or file was copied during a write). The reader yields the N-1 full spans and exposes `truncated_line_count`.
- `crashed` — file ended cleanly but no marker was written (process died after the last full line).

#### truncated_line_count
```python
@property
def truncated_line_count() -> int
```
Count of unparseable / partial lines encountered.

#### Schema-version handling
Lines with `schema_version > 1` raise `NDJSONVersionError`. Unknown top-level fields and unknown `attributes` keys are silently ignored — additive-only schema evolution within v1.

### OrchestraResult.metadata["tracing"]

After `Orchestra.run()` returns, the result's `metadata["tracing"]` dict surfaces writer state for programmatic consumers (Cloud worker, CI):

```python
{
  "total_spans": 142,
  "disk_error_count": 0,
  "dropped_span_count": 0,
  "disabled_dropped_count": 0,
  "disabled": False,
}
```

Use this to detect partial / disabled traces without instantiating the writer yourself.

## Wire Format

One JSON object per closed span, separated by `\n`:

```json
{
  "schema_version": 1,
  "ts": 1714220645.123,
  "trace_id": "01J9X4ABCDEFGHJKMNPQRSTVWX",
  "span_id":  "01J9X4ABCDEFGHJKMNPQRSTVWY",
  "parent_span_id": "01J9X4ABCDEFGHJKMNPQRSTVWZ",
  "name": "Step 3: Researcher",
  "kind": "step",
  "start_time": 1714220641.000,
  "end_time":   1714220645.123,
  "duration_ms": 4123.0,
  "status": "ok",
  "attributes": {"agent_name": "Researcher", "step_number": 3, "action_type": "invoke_agent"},
  "events": [{"name": "validation_decision", "timestamp": 1714220645.0, "attributes": {"is_valid": true}}],
  "links":  [{"linked_span_id": "01J9X4...", "relationship": "convergence"}]
}
```

Field summary:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `int` | Pin against changes; bumped only on breaking format change. Currently `1`. |
| `ts` | `float` | Wall-clock at the moment this line was written by the writer (epoch seconds). |
| `trace_id`, `span_id`, `parent_span_id` | `str` (ULID) | 26-char uppercase Crockford-base32; monotonic-orderable per process. |
| `kind` | `str` | Lowercase, one of `execution \| branch \| step \| generation \| tool`. |
| `start_time` / `end_time` / `duration_ms` | `float` | Epoch seconds (timestamps stay floats; consumers convert to ISO 8601 if needed). |
| `attributes` | `object` | Kind-specific. |
| `events`, `links` | `array` | Optional; present only when the span has them. |

### Marker and diagnostic lines

The writer emits two non-span line types that consumers reading individual spans MUST filter out by `kind`:

- `kind == "stream_event"` — diagnostic on queue overflow or writer self-disable. The `attributes` payload describes the event (e.g. `{"event": "dropped_span", "dropped_span_count": 7}` or `{"event": "writer_disabled", "consecutive_failures": 100}`).
- `kind == "stream_completed"` — terminal marker, exactly once on `close()`. `attributes` summarizes the run: `{total_spans, disk_error_count, dropped_span_count, disabled_dropped_count, disabled}`. Missing marker on EOF means the process crashed before close — readers expose this via `completion_status == "crashed"`.

## Trace Events

Events emitted by execution components and consumed by `TraceCollector`. All extend `StatusEvent` (which carries `event_id`, `timestamp`, `branch_id`, etc.). `event_id` is a ULID.

### ExecutionStartEvent
| Field | Type | Description |
|-------|------|-------------|
| `task_summary` | `str` | Truncated task description |
| `topology_summary` | `Dict` | Node and edge counts |
| `agent_names` | `List[str]` | Agents in the topology |
| `config_summary` | `Dict` | Key config values |

### GenerationEvent
| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Agent that ran the generation |
| `step_number` | `int` | Step index in the branch |
| `step_span_id` | `str` | Correlation ID for the step span |
| `model_name`, `provider` | `str` | Model and provider identifiers |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens` | `Optional[int]` | Token counts |
| `response_time_ms` | `Optional[float]` | API latency in milliseconds |
| `finish_reason` | `Optional[str]` | Why the model stopped |
| `has_thinking`, `has_tool_calls` | `bool` | Content flags |

### ValidationDecisionEvent
| Field | Type | Description |
|-------|------|-------------|
| `agent_name`, `step_number`, `step_span_id` | | Step correlation |
| `is_valid` | `bool` | Whether validation passed |
| `action_type` | `str` | Determined action (e.g., `invoke_agent`, `terminate_workflow`) |
| `next_agents` | `List[str]` | Target agent(s) for routing |
| `error_category` | `Optional[str]` | Error classification if invalid |
| `is_tool_continuation` | `bool` | Whether this is a tool continuation bypass |

### ConvergenceEvent
Emitted when parallel branches converge. The collector attaches links and events to both the parent branch span and the next step span on that branch.

| Field | Type | Description |
|-------|------|-------------|
| `parent_branch_id` | `str` | Branch receiving converged results |
| `child_branch_ids` | `List[str]` | Branches that converged |
| `convergence_point` | `str` | Convergence node name |
| `group_id` | `str` | Parallel group identifier |
| `successful_count`, `total_count` | `int` | Branch counts |

## Usage Patterns

### Enable tracing
```python
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing.config import TracingConfig

config = ExecutionConfig(
    tracing=TracingConfig(enabled=True, output_dir="./traces"),
)
```

### Read a completed trace (post-mortem)
```python
from marsys.coordination.tracing import TraceTree

tree = TraceTree.from_ndjson(pathlib.Path("./traces/01J9X4ABCDEFGHJKMNPQRSTVWX.ndjson"))
print(tree.root_span.duration_ms, len(tree.orphans))
```

### Tail-follow a live trace
```python
from marsys.coordination.tracing import NDJSONTraceReader

reader = NDJSONTraceReader(pathlib.Path("./traces/01J9X4....ndjson"))
for span in reader.stream(follow=True):
    print(span["kind"], span["name"], span.get("duration_ms"))
# Loop exits on stream_completed marker; status is then "complete".
```

### Custom writer (extending the ABC)
```python
from marsys.coordination.tracing.writers.base import TraceWriter
from marsys.coordination.tracing.types import Span, TraceTree

class MyStreamingWriter(TraceWriter):
    async def write(self, trace: TraceTree) -> None:
        pass  # finalize hook unused; everything streams via write_span

    async def write_span(self, span: Span) -> None:
        await self._send_to_backend(span.to_dict())

    async def close(self) -> None:
        await self._flush()
```

### Detect partial traces from OrchestraResult
```python
result = await Orchestra.run(task=..., topology=..., execution_config=cfg)
tracing = result.metadata.get("tracing", {})
if tracing.get("disabled") or tracing.get("dropped_span_count"):
    logger.warning("trace incomplete: %s", tracing)
```

## Best Practices

- Use `Orchestra.run` with `TracingConfig(enabled=True)` — `Orchestra` wires the streaming writer and lifecycle for you.
- Add `./traces/` to `.gitignore`. Trace files contain LLM prompts and responses by default.
- Set `include_message_content=False` if your prompts contain secrets; or apply redaction at the consumer side after reading.
- For long-running workloads on flaky storage, set `fsync_per_span=True` to fsync every line — slower but durable across kernel-level crashes.
- For privacy-sensitive deployments, consider a custom `TraceWriter` subclass that scrubs `attributes` before writing.

## Migration from JSON-at-end traces

Pre-v0.4 trace files were single JSON documents at `{output_dir}/{session_id}.json`. The new on-disk format is one NDJSON line per span at `{output_dir}/{trace_id}.ndjson`. Identifiers changed from UUID4 (36 chars) to ULID (26 chars uppercase).

Archived legacy `.json` files can still be loaded with `json.load()` — the file shape mirrors `TraceTree.to_dict()`, so the same tree-walking code works. There is no automatic migration utility; if you need NDJSON for legacy traces, rewrite them with a small script that calls `Span.to_dict()` per node and emits one line each.

## Related Documentation

- [Tracing Concepts](../concepts/tracing.md) — overview and span hierarchy
- [Configuration API](configuration.md) — `ExecutionConfig` reference
- [Architecture Overview](../architecture/overview.md) — how tracing fits in the system

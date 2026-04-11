# Tracing API

Complete API reference for execution tracing and observability in multi-agent workflows.

## 🎯 Overview

The Tracing API provides structured execution traces that capture the full hierarchy of an Orchestra run — from top-level execution through branches, agent steps, LLM generations, tool calls, and validation decisions.

## 📦 Core Classes

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
    detail_level: str = "standard",
    include_generation_details: bool = True,
    include_message_content: bool = False,
    include_tool_results: bool = True,
    max_content_length: int = 500
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | `bool` | Enable trace collection | `False` |
| `output_dir` | `str` | Directory for trace JSON files | `"./traces"` |
| `detail_level` | `str` | `"minimal"`, `"standard"`, or `"verbose"` | `"standard"` |
| `include_generation_details` | `bool` | Include token counts, model info in generation spans | `True` |
| `include_message_content` | `bool` | Include full prompt/response content (sensitive) | `False` |
| `include_tool_results` | `bool` | Include tool call results in tool spans | `True` |
| `max_content_length` | `int` | Truncation length for string attributes in `standard` mode | `500` |

### TraceCollector

EventBus consumer that builds hierarchical span trees from execution events.

**Import:**
```python
from marsys.coordination.tracing import TraceCollector
```

**Constructor:**
```python
TraceCollector(
    event_bus: EventBus,
    config: TracingConfig,
    writers: Optional[List[TraceWriter]] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `event_bus` | `EventBus` | Event bus to subscribe to | Required |
| `config` | `TracingConfig` | Tracing configuration | Required |
| `writers` | `List[TraceWriter]` | Output backends for completed traces | `[]` |

!!! tip "Automatic Creation"
    You don't create `TraceCollector` directly. Orchestra creates it when `TracingConfig.enabled=True`.

**Key Methods:**

#### finalize
```python
async def finalize(session_id: str) -> Optional[TraceTree]
```
Finalize the trace for a session. Closes any open spans (marking them as error), computes durations, and writes via all registered writers. Called automatically by Orchestra in a `try/finally` block.

#### close
```python
async def close() -> None
```
Shut down all writers and release resources.

### Span

A single unit of work in the execution trace.

**Import:**
```python
from marsys.coordination.tracing.types import Span
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `span_id` | `str` | Unique identifier (UUID) |
| `parent_span_id` | `Optional[str]` | Parent span for tree nesting |
| `trace_id` | `str` | Session-level trace identifier |
| `name` | `str` | Human-readable name (e.g., `"Step 3: Researcher"`) |
| `kind` | `str` | `execution`, `branch`, `step`, `generation`, `tool`, or `validation` |
| `start_time` | `float` | Epoch seconds |
| `end_time` | `Optional[float]` | Epoch seconds (set on close) |
| `duration_ms` | `Optional[float]` | Computed on close |
| `status` | `str` | `"ok"` or `"error"` |
| `attributes` | `Dict[str, Any]` | Kind-specific data |
| `events` | `List[Dict]` | Instant events attached to this span |
| `children` | `List[Span]` | Child spans |
| `links` | `List[Dict]` | Cross-branch causal links |

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
Add an instant event (e.g., validation decision) to this span.

#### to_dict
```python
def to_dict() -> Dict[str, Any]
```
Serialize the span tree to a nested dict for JSON output.

### TraceTree

A complete execution trace rooted at an execution span.

**Import:**
```python
from marsys.coordination.tracing.types import TraceTree
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | Unique trace identifier |
| `session_id` | `str` | Orchestra session ID |
| `root_span` | `Span` | Root execution span containing the full tree |
| `metadata` | `Dict[str, Any]` | Task summary, agent names, etc. |

#### to_dict
```python
def to_dict() -> Dict[str, Any]
```
Serialize the full trace tree for JSON output.

### TraceWriter

Abstract base for trace output backends.

**Import:**
```python
from marsys.coordination.tracing.writers.base import TraceWriter
```

**Methods:**

| Method | Description |
|--------|-------------|
| `async write(trace: TraceTree) -> None` | Write a completed trace |
| `async close() -> None` | Release resources |

### JSONFileTraceWriter

Writes traces as structured JSON files.

**Import:**
```python
from marsys.coordination.tracing.writers import JSONFileTraceWriter
```

**Constructor:**
```python
JSONFileTraceWriter(config: TracingConfig)
```

Output: `{config.output_dir}/{session_id}.json`

The writer respects `detail_level` from the config:
- **minimal**: Span hierarchy and timing only
- **standard**: All attributes, strings truncated to `max_content_length`
- **verbose**: Full content, no truncation

## 📦 Trace Events

Events emitted by execution components and consumed by `TraceCollector`. All extend `StatusEvent`.

### ExecutionStartEvent

Emitted when `Orchestra.execute()` begins.

| Field | Type | Description |
|-------|------|-------------|
| `task_summary` | `str` | Truncated task description |
| `topology_summary` | `Dict` | Node and edge counts |
| `agent_names` | `List[str]` | Agents in the topology |
| `config_summary` | `Dict` | Key config values |

### GenerationEvent

Emitted after each LLM generation completes.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Agent that ran the generation |
| `step_number` | `int` | Step index in the branch |
| `step_span_id` | `str` | Correlation ID for the step span |
| `model_name` | `str` | Model identifier |
| `provider` | `str` | Provider name |
| `prompt_tokens` | `Optional[int]` | Input tokens |
| `completion_tokens` | `Optional[int]` | Output tokens |
| `reasoning_tokens` | `Optional[int]` | Reasoning tokens (o1/o3) |
| `response_time_ms` | `Optional[float]` | API latency in milliseconds |
| `finish_reason` | `Optional[str]` | Why the model stopped |
| `has_thinking` | `bool` | Whether thinking content was present |
| `has_tool_calls` | `bool` | Whether tool calls were requested |

### ValidationDecisionEvent

Emitted after response validation determines the next action.

| Field | Type | Description |
|-------|------|-------------|
| `agent_name` | `str` | Agent whose response was validated |
| `step_number` | `int` | Step index |
| `step_span_id` | `str` | Step span correlation |
| `is_valid` | `bool` | Whether validation passed |
| `action_type` | `str` | Determined action (e.g., `invoke_agent`, `final_response`) |
| `next_agents` | `List[str]` | Target agent(s) for routing |
| `error_category` | `Optional[str]` | Error classification if invalid |
| `is_tool_continuation` | `bool` | Whether this is a tool continuation bypass |

### ConvergenceEvent

Emitted when parallel branches converge.

| Field | Type | Description |
|-------|------|-------------|
| `parent_branch_id` | `str` | Branch receiving converged results |
| `child_branch_ids` | `List[str]` | Branches that converged |
| `convergence_point` | `str` | Convergence node name |
| `group_id` | `str` | Parallel group identifier |
| `successful_count` | `int` | Successfully completed branches |
| `total_count` | `int` | Total branches in group |

## 🎨 Usage Patterns

### Enable Tracing

```python
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.tracing.config import TracingConfig

config = ExecutionConfig(
    tracing=TracingConfig(enabled=True, output_dir="./traces"),
)
```

### Custom Writer

```python
from marsys.coordination.tracing.writers.base import TraceWriter
from marsys.coordination.tracing.types import TraceTree

class MyTraceWriter(TraceWriter):
    async def write(self, trace: TraceTree) -> None:
        data = trace.to_dict()
        # Send to your observability backend
        await send_to_backend(data)

    async def close(self) -> None:
        pass
```

### Access Trace Programmatically

```python
# After Orchestra.execute(), if you have access to the Orchestra instance:
if orchestra.trace_collector:
    trace = await orchestra.trace_collector.finalize(session_id)
    if trace:
        trace_dict = trace.to_dict()
```

## 📋 Best Practices

- ✅ Use `detail_level="minimal"` in production for low overhead
- ✅ Use `detail_level="standard"` (default) during development
- ✅ Keep `include_message_content=False` unless investigating specific issues
- ✅ Use meaningful `session_id` values for trace file identification
- ✅ Add `./traces/` to `.gitignore`
- ❌ Don't enable `verbose` in production — trace files can be very large
- ❌ Don't commit trace files containing sensitive prompt/response content

## 🚦 Related Documentation

- [Tracing Concepts](../concepts/tracing.md) - Overview and span hierarchy
- [Configuration API](configuration.md) - ExecutionConfig reference
- [Architecture Overview](../architecture/overview.md) - How tracing fits in the system
- [State Management API](state.md) - Related persistence module

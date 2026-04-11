# Tracing

Structured execution traces for multi-agent workflow observability.

## Overview

The tracing module captures the full execution flow of an Orchestra run as a hierarchical span tree, enabling post-hoc analysis of:

- **Agent steps**: Which agents ran, in what order, and how long each took
- **LLM generations**: Token usage, model identity, API latency, and finish reason per generation
- **Validation decisions**: What action type was determined and what routing was chosen
- **Tool calls**: Which tools were invoked, their arguments, results, and duration
- **Branch lifecycle**: Branch creation, parallel execution, and convergence
- **Convergence**: Which child branches fed into which parent, and what was aggregated

Traces are written as structured JSON files, one per execution session.

## Core Components

### TracingConfig

Controls whether tracing is enabled and what detail level to capture:

```python
from marsys.coordination.tracing.config import TracingConfig

config = TracingConfig(
    enabled=True,
    output_dir="./traces",
    detail_level="verbose",  # minimal | standard | verbose
)
```

### TraceCollector

EventBus consumer that subscribes to execution events and builds a span tree:

```python
from marsys.coordination.tracing import TraceCollector
```

You don't create this directly — Orchestra creates it automatically when `TracingConfig.enabled=True`.

### Span

A single unit of work in the trace. Spans form a tree via `parent_span_id`:

```python
from marsys.coordination.tracing.types import Span
```

Span kinds: `execution`, `branch`, `step`, `generation`, `tool`.

Validation decisions are captured as events on step spans (not separate spans). Convergence is captured as links and events on both the parent branch span and the convergence step span.

### TraceTree

A complete trace rooted at an execution span:

```python
from marsys.coordination.tracing.types import TraceTree
```

### TraceWriter

Abstract base for output backends. `JSONFileTraceWriter` is the built-in implementation:

```python
from marsys.coordination.tracing.writers import JSONFileTraceWriter
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

After execution, find the trace at `./traces/{session_id}.json`.

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

The JSON output is a tree of spans. Here is an abbreviated example:

```json
{
  "trace_id": "5228247d-...",
  "session_id": "session-001",
  "metadata": {
    "task_summary": "Research quantum computing",
    "agent_names": ["Coordinator", "Researcher", "FactChecker"]
  },
  "root_span": {
    "name": "Orchestra.run",
    "kind": "execution",
    "duration_ms": 12450.3,
    "status": "ok",
    "children": [
      {
        "name": "Branch: main",
        "kind": "branch",
        "links": [
          {"linked_span_id": "...", "relationship": "convergence"}
        ],
        "children": [
          {
            "name": "Step 0: Coordinator",
            "kind": "step",
            "attributes": { "action_type": "parallel_invoke" },
            "events": [{"name": "validation_decision", "attributes": {"next_agents": ["Researcher", "FactChecker"]}}],
            "children": [
              {"name": "Generation: claude-sonnet-4-20250514", "kind": "generation", "duration_ms": 1200.5, "attributes": {"prompt_tokens": 150, "completion_tokens": 80}}
            ]
          },
          {
            "name": "Step 1: Coordinator",
            "kind": "step",
            "attributes": { "action_type": "final_response" },
            "links": [
              {"linked_span_id": "researcher-branch-id", "relationship": "convergence"},
              {"linked_span_id": "factchecker-branch-id", "relationship": "convergence"}
            ],
            "events": [{"name": "convergence", "attributes": {"successful_count": 2, "total_count": 2}}]
          }
        ]
      },
      {
        "name": "Branch: Researcher",
        "kind": "branch",
        "attributes": { "trigger_type": "parallel" },
        "children": ["... step spans with generation/tool children ..."]
      },
      {
        "name": "Branch: FactChecker",
        "kind": "branch",
        "attributes": { "trigger_type": "parallel" },
        "children": ["... step spans with generation/tool children ..."]
      }
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

## Detail Levels

| Level | What's captured | Use case |
|-------|----------------|----------|
| `minimal` | Span hierarchy + timing only, no attributes | Performance profiling |
| `standard` | All spans with attributes, content truncated to `max_content_length` | Production with size limits |
| `verbose` | Everything including full message content | Default, full visibility during development |

## Best Practices

### 1. Use Meaningful Session IDs

```python
context = {"session_id": "research_quantum_2026_04"}
```

This makes trace files easy to find and correlate.

### 2. Choose the Right Detail Level

Use `verbose` (default) during development for full visibility. Use `minimal` in production for low overhead. Use `standard` with `max_content_length` when you want attributes but need to control file size.

### 3. Be Mindful of Content Sensitivity

By default, trace files contain full LLM prompts and responses. Keep trace output directories secure and excluded from version control. Set `include_message_content=False` or use `detail_level="minimal"` to exclude content.

## Limitations

- Only `JSONFileTraceWriter` is implemented (Chrome Trace Format export planned for follow-up)
- Traces are written on completion — no streaming/incremental output during execution
- Tracing for standalone `agent.run()` calls (outside Orchestra) is not supported

## Related Documentation

- [Tracing API Reference](../api/tracing.md) - Complete class and method reference
- [Configuration](../getting-started/configuration.md) - ExecutionConfig setup
- [Architecture Overview](../architecture/overview.md) - How tracing fits in the system

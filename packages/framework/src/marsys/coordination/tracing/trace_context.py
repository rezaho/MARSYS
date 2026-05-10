"""Per-step tracing context object threaded through ``model.arun`` calls.

Built once in ``step_executor`` at step boundary and propagated explicitly
via the ``trace_ctx`` kwarg through the agent → model wrapper chain.
This avoids the implicit-state cost of ContextVars while still giving the
``capture_llm_call`` helper everything it needs to correlate an LLM call
with its parent step span.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class TraceContext:
    """Carries the correlation ids and event bus needed to emit LLM events.

    ``kind`` distinguishes a normal generation call from a compaction call so
    the collector can place each into the correct span kind. ``captured`` is
    the re-entrancy guard: an outer wrapper sets it to ``True`` before
    delegating to an inner wrapper, so the inner ``capture_llm_call`` skips
    re-emission and only one request/response pair lands per LLM invocation.
    """

    step_span_id: str
    branch_id: Optional[str]
    agent_name: str
    session_id: str
    event_bus: Any                       # AsyncEventBus — typed Any to avoid an import cycle
    kind: str = "generation"             # "generation" | "compaction"
    captured: bool = False               # re-entrancy guard

    def child(self, *, kind: Optional[str] = None) -> "TraceContext":
        """Derive a child context, e.g. ``parent.child(kind="compaction")``.

        ``captured`` is reset to ``False`` so the child call gets its own
        request/response pair.
        """
        return replace(self, kind=kind or self.kind, captured=False)

    def mark_captured(self) -> "TraceContext":
        """Return a copy with ``captured=True`` for the inner-wrapper hop."""
        return replace(self, captured=True)

"""
Tracing module for MARSYS observability.

Provides structured execution traces by consuming EventBus events
and building hierarchical span trees. Traces capture the full
execution flow: agent steps, LLM generations, tool calls,
validation decisions, branching, and convergence.
"""

from .config import TracingConfig
from .types import Span, TraceTree
from .collector import TraceCollector
from .readers import NDJSONTraceReader, NDJSONVersionError
from .writers import NDJSONTraceWriter, TraceWriter

__all__ = [
    "TracingConfig",
    "Span",
    "TraceTree",
    "TraceCollector",
    "TraceWriter",
    "NDJSONTraceWriter",
    "NDJSONTraceReader",
    "NDJSONVersionError",
]

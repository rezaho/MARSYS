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
from .messages import (
    FilesystemMessageStore,
    InMemoryMessageStore,
    MessageStore,
    build_input_messages_ref,
    compute_message_hash,
)
from .readers import NDJSONTraceReader, NDJSONVersionError
from .redactor import NoRedactionRedactor, SecretRedactor
from .sink import TelemetrySink
from .writers import NDJSONTraceWriter, OtelTraceWriter

# NOTE: ``trace_context``, ``capture``, and the LLM event classes in
# ``events`` are intentionally NOT re-exported here — eager re-export
# triggers a circular import via ``coordination.status.events``.
# Import them directly from their submodules.

__all__ = [
    "TracingConfig",
    "Span",
    "TraceTree",
    "TraceCollector",
    "TelemetrySink",
    "SecretRedactor",
    "NoRedactionRedactor",
    "MessageStore",
    "InMemoryMessageStore",
    "FilesystemMessageStore",
    "compute_message_hash",
    "build_input_messages_ref",
    "NDJSONTraceWriter",
    "OtelTraceWriter",
    "NDJSONTraceReader",
    "NDJSONVersionError",
]

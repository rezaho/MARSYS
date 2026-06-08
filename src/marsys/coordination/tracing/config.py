"""
Configuration for the tracing module.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .sink import TelemetrySink
    from .redactor import SecretRedactor
    from .messages import MessageStore


@dataclass
class TracingConfig:
    """
    Configuration for execution tracing.

    Controls whether traces are collected and where to write trace output,
    plus per-event-kind toggles for content capture, plus the list of
    user-supplied TelemetrySinks and the SecretRedactor that runs once
    at the fan-out boundary in TraceCollector._stream_span.

    Full-input capture is always on when tracing is enabled: each agent step
    and LLM call records its full input message list as content-addressed
    hashes via the :class:`~marsys.coordination.tracing.messages.MessageStore`.
    Identical messages dedup automatically across all traces sharing the same
    ``output_dir``, and generation spans carry a compact ref instead of an
    inline copy (sinks that can't follow a ref rehydrate at publish time).
    """

    enabled: bool = False
    output_dir: str = "./traces"
    include_message_content: bool = True
    include_tool_results: bool = True
    sinks: List['TelemetrySink'] = field(default_factory=list)
    redactor: Optional['SecretRedactor'] = None

    # Optional override of the default ``FilesystemMessageStore``.
    # Plug in S3/Redis/etc. backends by subclassing ``MessageStore``.
    message_store: Optional['MessageStore'] = None

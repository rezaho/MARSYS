"""
Configuration for the tracing module.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .sink import TelemetrySink
    from .redactor import SecretRedactor


@dataclass
class TracingConfig:
    """
    Configuration for execution tracing.

    Controls whether traces are collected and where to write trace output,
    plus per-event-kind toggles for content capture, plus the list of
    user-supplied TelemetrySinks and the SecretRedactor that runs once
    at the fan-out boundary in TraceCollector._stream_span.
    """

    enabled: bool = False
    output_dir: str = "./traces"
    include_generation_details: bool = True
    include_message_content: bool = True
    include_tool_results: bool = True
    sinks: List['TelemetrySink'] = field(default_factory=list)
    redactor: Optional['SecretRedactor'] = None

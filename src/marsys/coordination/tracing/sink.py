"""
TelemetrySink — async streaming consumer for closed spans.

The framework's `TraceCollector` calls `publish_span(span)` once per span
close. Adapters live outside the framework (spren.telemetry, marsys-langsmith,
marsys-phoenix, marsys-langfuse). The framework knows about TelemetrySink;
it knows nothing about specific vendor backends.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Span


class TelemetrySink(ABC):
    """
    Async streaming consumer for closed spans.

    TraceCollector calls publish_span once per span close. Adapters may do
    network I/O, queue/batch internally, and may take time to shut down.
    Exceptions raised from publish_span are caught + logged at the framework
    boundary; a misbehaving sink does not stop the run.

    Lifecycle:
      * publish_span(span) — called once per closed span. May do I/O.
      * close() — called once at run end. Flush pending data and release
        resources. Idempotent. Bounded by Orchestra's close timeout.
    """

    @abstractmethod
    async def publish_span(self, span: 'Span') -> None:
        """Forward a closed span to the backend. May do network I/O."""

    @abstractmethod
    async def close(self) -> None:
        """Flush pending data and release resources. Idempotent."""

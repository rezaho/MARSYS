"""
Abstract base for trace writers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Span, TraceTree


class TraceWriter(ABC):
    """
    Base class for trace output backends.

    Implementations persist or export trace data. Two flavours are supported:

    - **Finalize-only writers** override only ``write`` and ``close`` and
      receive the full ``TraceTree`` once at the end of a run.
    - **Streaming writers** additionally override ``write_span`` to emit
      data as each span closes. ``TraceCollector`` calls ``write_span``
      from every span-close site (root, branch, step, tool, generation,
      and the orphan-finalize pass).
    """

    @abstractmethod
    async def write(self, trace: 'TraceTree') -> None:
        """Write a completed trace."""

    @abstractmethod
    async def close(self) -> None:
        """Release any resources held by this writer."""

    async def write_span(self, span: 'Span') -> None:
        """Stream a single closed span. Default no-op for non-streaming writers.

        Streaming writers override; ``TraceCollector`` calls this from every
        span-close site. Errors are caught by the collector and logged — they
        must NOT propagate back into the event-handler call path.
        """
        return None

"""
TelemetrySink — async streaming consumer for closed spans.

The framework's `TraceCollector` calls `publish_span(span)` once per span
close. Concrete adapters (e.g. the bundled `OtelTraceWriter`, or any
out-of-tree backend adapter) implement this interface. The framework knows
about TelemetrySink; it knows nothing about specific backends.

Boundary: vendor-*neutral* sinks ship in-framework — `NDJSONTraceWriter`
(local disk) and `OtelTraceWriter` (OTLP/HTTP emitting only cross-vendor
conventions: `gen_ai.*` semconv, OpenInference, OpenLLMetry). Vendor-
*specific* presets — a product's endpoint, auth headers, env-var wiring, or
any attribute keyed to a single product's namespace — live outside the
framework, in the caller or a dedicated adapter package (a downstream
consumer's telemetry adapter, marsys-langsmith, marsys-phoenix,
marsys-langfuse). See
`live_tests/tracing/secret_word_pipeline.py` for caller-side LangSmith /
Langfuse presets built on top of the generic `OtelTraceWriter`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .messages import MessageStore
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

    def bind_message_store(self, store: 'MessageStore') -> None:
        """Optionally receive the collector's content-addressed message store.

        Called once by ``TraceCollector`` during setup, only when a store is
        configured. Generation/compaction spans then carry a content-addressed
        ``*_ref`` instead of inline message content (dedup). Sinks that cannot
        follow a ref — e.g. an OTLP exporter that must emit literal content —
        override this to capture the store and rehydrate via
        ``store.reconstruct(ref)`` at ``publish_span`` time.

        The default is a no-op: ref-aware consumers (the NDJSON writer plus
        post-mortem readers) persist the ref verbatim and resolve it lazily,
        so they need nothing here.
        """

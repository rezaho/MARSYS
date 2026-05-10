"""
Trace writers for persisting execution traces.
"""

from .ndjson_writer import NDJSONTraceWriter
# Safe to import without the 'tracing-otel' extra — OTel SDK only
# loads on instantiation.
from .otel_writer import OtelTraceWriter

__all__ = [
    "NDJSONTraceWriter",
    "OtelTraceWriter",
]

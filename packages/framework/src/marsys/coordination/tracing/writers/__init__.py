"""
Trace writers for persisting execution traces.
"""

from .base import TraceWriter
from .ndjson_writer import NDJSONTraceWriter

__all__ = [
    "TraceWriter",
    "NDJSONTraceWriter",
]

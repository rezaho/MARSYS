"""
Trace writers for persisting execution traces.
"""

from .base import TraceWriter
from .json_writer import JSONFileTraceWriter

__all__ = [
    "TraceWriter",
    "JSONFileTraceWriter",
]

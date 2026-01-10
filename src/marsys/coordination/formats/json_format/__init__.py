"""
JSON response format for MARSYS.

This module provides JSON-based response formatting and processing.
"""

from .format import JSONResponseFormat
from .processor import StructuredJSONProcessor

__all__ = [
    "JSONResponseFormat",
    "StructuredJSONProcessor",
]

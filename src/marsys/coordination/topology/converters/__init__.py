"""
Converters for transforming various topology formats to the canonical Topology class.

This package provides converters for the three ways to define multi-agent systems:
1. String notation (simple dictionaries with strings)
2. Object notation (mixed objects and strings)
3. Pattern configuration (pre-defined patterns)
"""

from .string_converter import StringNotationConverter
from .object_converter import ObjectNotationConverter
from .pattern_converter import PatternConfigConverter
from .parsing import parse_node, parse_edge, parse_rule

__all__ = [
    'StringNotationConverter',
    'ObjectNotationConverter', 
    'PatternConfigConverter',
    'parse_node',
    'parse_edge',
    'parse_rule'
]
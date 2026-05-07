"""
Configuration for the tracing module.
"""

from dataclasses import dataclass


@dataclass
class TracingConfig:
    """
    Configuration for execution tracing.

    Controls whether traces are collected and where to write trace output,
    plus per-event-kind toggles for content capture.
    """

    enabled: bool = False
    output_dir: str = "./traces"
    include_generation_details: bool = True
    include_message_content: bool = True
    include_tool_results: bool = True

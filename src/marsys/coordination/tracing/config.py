"""
Configuration for the tracing module.
"""

from dataclasses import dataclass, field


@dataclass
class TracingConfig:
    """
    Configuration for execution tracing.

    Controls whether traces are collected, what detail level to capture,
    and where to write trace output.
    """

    enabled: bool = False
    output_dir: str = "./traces"

    # Detail levels:
    #   minimal  - span hierarchy + timing only (no attributes content)
    #   standard - all spans with attributes, content truncated to max_content_length
    #   verbose  - everything including full message content
    detail_level: str = "standard"

    include_generation_details: bool = True
    include_message_content: bool = False  # Off by default — can be large/sensitive
    include_tool_results: bool = True
    max_content_length: int = 500

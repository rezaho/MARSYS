"""
JSON file trace writer.

Writes one JSON file per execution trace to the configured output directory.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import TraceWriter

if TYPE_CHECKING:
    from ..config import TracingConfig
    from ..types import Span, TraceTree

logger = logging.getLogger(__name__)


class JSONFileTraceWriter(TraceWriter):
    """
    Writes traces as structured JSON files.

    Output: {output_dir}/{session_id}.json
    Respects detail_level from TracingConfig for content filtering.
    """

    def __init__(self, config: 'TracingConfig'):
        self.config = config
        self.output_dir = config.output_dir

    async def write(self, trace: 'TraceTree') -> None:
        """Write trace tree as a JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)

        file_path = os.path.join(self.output_dir, f"{trace.session_id}.json")

        # Filter based on detail level
        trace_dict = self._filter_trace(trace)

        with open(file_path, 'w') as f:
            json.dump(trace_dict, f, indent=2, default=str)

        logger.info(f"Trace written to {file_path}")

    async def close(self) -> None:
        """No resources to release."""
        pass

    def _filter_trace(self, trace: 'TraceTree') -> Dict[str, Any]:
        """Apply detail_level filtering to the trace dict."""
        result = {
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "metadata": trace.metadata,
            "root_span": self._filter_span(trace.root_span),
        }
        return result

    def _filter_span(self, span: 'Span') -> Dict[str, Any]:
        """Filter a single span based on detail_level."""
        detail = self.config.detail_level

        result: Dict[str, Any] = {
            "span_id": span.span_id,
            "name": span.name,
            "kind": span.kind,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "status": span.status,
        }

        if span.parent_span_id:
            result["parent_span_id"] = span.parent_span_id

        if detail == "minimal":
            # Hierarchy + timing only — skip attributes, events, links
            pass
        elif detail == "standard":
            result["attributes"] = self._truncate_attributes(span.attributes)
            if span.events:
                result["events"] = span.events
            if span.links:
                result["links"] = span.links
        elif detail == "verbose":
            result["attributes"] = span.attributes
            if span.events:
                result["events"] = span.events
            if span.links:
                result["links"] = span.links

        if span.children:
            result["children"] = [self._filter_span(child) for child in span.children]

        return result

    def _truncate_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate string values in attributes to max_content_length (0 = no truncation)."""
        max_len = self.config.max_content_length
        if not max_len:
            return attributes
        truncated = {}
        for key, value in attributes.items():
            if isinstance(value, str) and len(value) > max_len:
                truncated[key] = value[:max_len] + "..."
            else:
                truncated[key] = value
        return truncated

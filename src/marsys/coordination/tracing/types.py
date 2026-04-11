"""
Core data types for the tracing module.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class Span:
    """
    A single unit of work in the execution trace.

    Spans form a tree: an execution span contains branch spans,
    which contain step spans, which contain generation/tool/validation spans.

    The `kind` field determines what `attributes` contain:
      - execution: task, topology_summary, config_summary
      - branch: branch_type, agents, completion_condition, parent_branch_id, group_id
      - step: agent_name, step_number, action_type, next_agent, retry_count
      - generation: model_name, provider, prompt_tokens, completion_tokens, response_time_ms
      - tool: tool_name, arguments_summary, result_summary
      - validation: is_valid, action_type, next_agents, error_category
    """

    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    kind: str  # execution | branch | step | generation | tool | validation
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok | error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    children: List['Span'] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)

    def close(self, end_time: Optional[float] = None, status: Optional[str] = None) -> None:
        """Close this span, computing duration."""
        self.end_time = end_time or time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if status:
            self.status = status

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an instant event to this span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def add_link(self, linked_span_id: str, relationship: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a causal link to another span (e.g., convergence)."""
        self.links.append({
            "linked_span_id": linked_span_id,
            "relationship": relationship,
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON output."""
        result = {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.events:
            result["events"] = self.events
        if self.links:
            result["links"] = self.links
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result


@dataclass
class TraceTree:
    """
    A complete execution trace rooted at an execution span.

    One TraceTree is produced per Orchestra.run() invocation.
    """

    trace_id: str
    session_id: str
    root_span: Span
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full trace tree for JSON output."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "root_span": self.root_span.to_dict(),
        }


def create_span(
    trace_id: str,
    name: str,
    kind: str,
    parent_span_id: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    start_time: Optional[float] = None,
) -> Span:
    """Factory function for creating spans with generated IDs."""
    return Span(
        span_id=str(uuid.uuid4()),
        parent_span_id=parent_span_id,
        trace_id=trace_id,
        name=name,
        kind=kind,
        start_time=start_time or time.time(),
        attributes=attributes or {},
    )

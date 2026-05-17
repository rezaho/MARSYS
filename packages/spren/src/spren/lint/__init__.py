"""Workflow lint surface — non-blocking findings for the canvas UI.

This is a **pure Spren linter** (SP-018): it makes ZERO framework calls — it
does not invoke ``TopologyGraph.validate_workflow`` or any other framework
function. It computes its findings — unreachable nodes, cycles-without-escape,
dangling edges, unknown tools, missing agent references, missing required
fields — from the canonical ``WorkflowDefinition`` directly, reading the
framework ``NodeSpec.kind`` field, producing structured ``LintFinding`` rows
the canvas can pin to specific nodes/edges.
"""
from .workflow_linter import lint_workflow

__all__ = ["lint_workflow"]

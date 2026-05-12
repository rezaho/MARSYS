"""Workflow lint surface — non-blocking findings for the canvas UI.

The framework's ``TopologyGraph.validate_workflow()`` operates over det-nodes
(``StartNode`` / ``EndNode`` / ``UserNode``) and is the production-runtime
guard. Spren's v0.3 workflows do NOT use det-nodes (the topology model is
``NodeType.AGENT``/``USER``/``SYSTEM``/``TOOL`` only — see
``packages/spren/src/spren/models/topology.py``), so a direct call into the
framework validator would either silently no-op (its ``if not self.det_nodes:
return`` guard) or yield messages referencing concepts the user can't see in
the canvas.

Instead, this module computes the same *category* of findings — unreachable
nodes, cycles-without-escape, dangling edges, unknown tools, missing agent
references — from the ``WorkflowDefinition`` directly, producing structured
``LintFinding`` rows the canvas can pin to specific nodes/edges. When
Framework Session 05 ships structured findings on a Pydantic boundary, this
module shrinks to a thin adapter.
"""
from .workflow_linter import lint_workflow

__all__ = ["lint_workflow"]

"""Tests for ``StepExecutor._build_coordination_context`` det-node
filtering on ``next_agents``.

Det-nodes (Start / End / User) are routed via dedicated coordination
tools (``terminate_workflow``, ``ask_user``), NOT via ``invoke_agent``.
Before this fix, ``next_agents`` carried whatever ``topology_graph
.get_next_agents()`` returned — including det-nodes. Two downstream
bugs surfaced from this:

1. ``BaseResponseFormat._build_peer_agent_instructions`` listed
   det-nodes like ``End 1`` under "Available peer agents", competing
   with the WORKFLOW COMPLETION block's instruction to call
   ``terminate_workflow``. Conflicting signal to the LLM.
2. ``CoordinationToolSchemaBuilder._build_invoke_agent_schema`` filtered
   det-nodes via lowercase-name match against
   ``("user", "start", "end")``. A canvas-named ``End 1`` slipped
   through and ended up in the ``invoke_agent`` enum — making it a
   valid invocation target even though it's a det-node.

Fix at ``step_executor.py``: filter ``next_agents`` through
``topology_graph.is_det_node()`` at the build site so det-nodes never
enter ``CoordinationContext.next_agents``. Downstream consumers
inherit the fix.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_fake_topology(*, next_agents: list[str], det_nodes: set[str]):
    """Stub ``TopologyGraph``-like object."""
    graph = MagicMock()
    graph.get_next_agents.return_value = next_agents
    graph.is_det_node.side_effect = lambda name: name in det_nodes
    graph.has_edge_to_endnode.return_value = any(
        n in det_nodes and "end" in n.lower() for n in next_agents
    )
    graph.has_edge_to_usernode.return_value = any(
        n in det_nodes and "user" in n.lower() for n in next_agents
    )
    return graph


class TestDetNodeFilter:
    """Det-nodes (Start, End, User — including canvas-named variants
    like ``End 1``) are filtered out of ``CoordinationContext
    .next_agents`` at the build site."""

    def test_end_node_filtered_from_next_agents(self):
        """``Start → Agent → End 1`` — Agent's ``next_agents`` should
        contain only true peers (none), not ``End 1`` itself."""
        from marsys.coordination.execution.step_executor import StepExecutor

        graph = _make_fake_topology(
            next_agents=["End 1"],
            det_nodes={"End 1"},
        )
        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=graph,
        )
        assert ctx.next_agents == []
        # The gating flag still derived correctly via
        # ``has_edge_to_endnode`` — separate signal from the peer list.
        assert ctx.can_terminate_workflow is True

    def test_canvas_named_end_filtered(self):
        """Canvas-naming gives det-nodes friendlier names like ``End 1``
        (with a number). The filter uses ``is_det_node`` (topology-
        backed) not lowercase exact match — so ``End 1`` is excluded
        the same way as ``End`` would be."""
        from marsys.coordination.execution.step_executor import StepExecutor

        graph = _make_fake_topology(
            next_agents=["Worker", "End 1"],
            det_nodes={"End 1"},
        )
        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=graph,
        )
        # Real peer ``Worker`` survives; det-node ``End 1`` filtered.
        assert ctx.next_agents == ["Worker"]

    def test_user_node_filtered(self):
        """``User`` node is also a det-node, routed via ``ask_user``."""
        from marsys.coordination.execution.step_executor import StepExecutor

        graph = _make_fake_topology(
            next_agents=["User", "Helper"],
            det_nodes={"User"},
        )
        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=graph,
        )
        assert ctx.next_agents == ["Helper"]

    def test_no_det_nodes_means_unfiltered_passthrough(self):
        """When no nodes are det-nodes, all peers pass through."""
        from marsys.coordination.execution.step_executor import StepExecutor

        graph = _make_fake_topology(
            next_agents=["A", "B", "C"],
            det_nodes=set(),
        )
        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=graph,
        )
        assert ctx.next_agents == ["A", "B", "C"]

    def test_graph_without_is_det_node_does_not_break(self):
        """Backward-compat: a graph that doesn't expose
        ``is_det_node`` (older / mocked graphs in legacy tests) still
        returns the unfiltered list rather than crashing."""
        from marsys.coordination.execution.step_executor import StepExecutor

        graph = MagicMock(spec=['get_next_agents', 'has_edge_to_endnode',
                                'has_edge_to_usernode'])
        graph.get_next_agents.return_value = ["Worker", "End 1"]
        graph.has_edge_to_endnode.return_value = True
        graph.has_edge_to_usernode.return_value = False
        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=graph,
        )
        # Without ``is_det_node`` available, no filtering — pre-fix
        # behavior preserved.
        assert ctx.next_agents == ["Worker", "End 1"]

    def test_no_topology_graph_returns_empty_context(self):
        """``topology_graph is None`` short-circuits to empty
        CoordinationContext — pre-existing behavior."""
        from marsys.coordination.execution.step_executor import StepExecutor

        ctx = StepExecutor._build_coordination_context(
            MagicMock(),  # self is unused in this method
            agent_name="Agent",
            topology_graph=None,
        )
        assert ctx.next_agents == []
        assert ctx.can_terminate_workflow is False


class TestPromptDoesNotListEndAsPeer:
    """Integration: after the filter, the rendered system prompt's
    PEER AGENT DELEGATION section does NOT list det-nodes."""

    def test_peer_section_excludes_end_after_filter(self):
        """When ``next_agents`` is empty after filtering, the peer
        section emits nothing — no "Available peer agents: - End 1"
        line that would compete with WORKFLOW COMPLETION."""
        from marsys.coordination.formats.context import (
            AgentContext,
            CoordinationContext,
            SystemPromptContext,
        )
        from marsys.coordination.formats.json_format.format import (
            JSONResponseFormat,
        )

        fmt = JSONResponseFormat()
        # Simulate post-filter state: End 1 removed, no real peers.
        ctx = SystemPromptContext(
            agent=AgentContext(
                name="Agent",
                goal="g",
                instruction="Do the thing.",
            ),
            coordination=CoordinationContext(
                next_agents=[],
                can_terminate_workflow=True,
                can_ask_user=False,
            ),
        )
        prompt = fmt.build_complete_system_prompt(ctx)
        # No peer-delegation block when there are no real peers.
        assert "--- PEER AGENT DELEGATION ---" not in prompt
        # But the workflow-completion block IS present (End edge exists).
        assert "--- WORKFLOW COMPLETION ---" in prompt
        assert "terminate_workflow" in prompt

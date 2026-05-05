"""
Integration tests for Router with other coordination components.

Tests the complete flow from response validation through routing decisions.
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock

from marsys.coordination.routing import Router, RoutingContext
from marsys.coordination.validation.response_validator import (
    ValidationProcessor,
    ValidationResult,
    ActionType
)
from marsys.coordination.validation.types import AgentInvocation
from marsys.coordination.topology.graph import TopologyGraph
from marsys.coordination.topology.analyzer import TopologyAnalyzer
from marsys.coordination.branches.types import (
    ExecutionBranch, 
    BranchType, 
    BranchTopology,
    BranchState,
    BranchStatus,
    ExecutionState
)
from marsys.agents import BaseAgent


class TestRouterIntegration:
    """Integration tests for Router with real components."""
    
    @pytest.fixture
    def complex_topology(self):
        """Create a complex topology for testing."""
        return {"nodes": [
                "User", "Orchestrator", "Planner", 
                "Researcher", "Writer", "Editor",
                "Reviewer", "Publisher"
            ], "edges": [
                "User -> Orchestrator",
                "Orchestrator -> Planner",
                "Planner -> Researcher",
                "Planner -> Writer",
                "Researcher -> Writer",
                "Writer <-> Editor",  # Conversation loop
                "Editor -> Reviewer",
                "Writer -> Reviewer",
                "Reviewer -> Publisher",
                "Reviewer -> Planner"   # Feedback loop
            ], "rules": [
                "parallel(Researcher, Writer)",
                "max_turns(Writer <-> Editor, 5)"
            ]}
    
    @pytest.fixture
    def topology_graph(self, complex_topology):
        """Build topology graph from definition."""
        analyzer = TopologyAnalyzer()
        return analyzer.analyze(complex_topology)
    
    @pytest.fixture
    def router(self, topology_graph):
        """Create Router with topology."""
        return Router(topology_graph)
    
    @pytest.fixture
    def validator(self, topology_graph):
        """Create ValidationProcessor with topology."""
        return ValidationProcessor(topology_graph)
    
    @pytest.mark.asyncio
    async def test_sequential_flow(self, router, validator):
        """Test sequential agent invocation flow."""
        # Setup
        branch = ExecutionBranch(
            id="main",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Planner"],
                entry_agent="Planner"
            )
        )
        
        context = RoutingContext(
            current_branch_id="main",
            current_agent="Planner",
            conversation_history=[],
            branch_agents=["Planner"]
        )
        
        # Planner invokes Researcher
        response = {
            "thinking": "I need to gather information first",
            "next_action": "invoke_agent",
            "action_input": "Research AI ethics and safety"
        }
        
        # Create mock agent and execution state
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Planner"
        exec_state = ExecutionState(
            session_id="test_session",
            current_step=1,
            status="running"
        )
        
        # Skip validation and create result directly for this test
        # (The validator would check topology constraints which we're testing separately)
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response=response,
            invocations=[AgentInvocation(agent_name="Researcher", request="Research AI ethics and safety")]
        )

        # Route
        decision = await router.route(validation, branch, context)

        # Verify
        assert decision.should_continue is True
        assert len(decision.next_steps) == 1
        assert decision.next_steps[0].agent_name == "Researcher"
    
    @pytest.mark.asyncio
    async def test_parallel_execution_flow(self, router, validator):
        """Test parallel agent execution flow."""
        # Setup
        branch = ExecutionBranch(
            id="planner_branch",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Planner"],
                entry_agent="Planner"
            )
        )
        
        context = RoutingContext(
            current_branch_id="planner_branch",
            current_agent="Planner",
            conversation_history=[],
            branch_agents=["Planner"]
        )
        
        # Planner invokes both Researcher and Writer in parallel
        response = {
            "thinking": "Let's research and draft in parallel",
            "next_action": "parallel_invoke",
            "agents": ["Researcher", "Writer"],
            "action_input": {
                "Researcher": "Find latest AI safety research",
                "Writer": "Draft introduction based on outline"
            }
        }
        
        # Manually create validation result for parallel
        # (In real system, ValidationProcessor would handle this)
        from marsys.coordination.validation.response_validator import ValidationResult, ActionType
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.PARALLEL_INVOKE,
            parsed_response=response,
            invocations=[
                AgentInvocation(agent_name="Researcher", request="Find latest AI safety research"),
                AgentInvocation(agent_name="Writer", request="Draft introduction based on outline")
            ]
        )
        
        # Route
        decision = await router.route(validation, branch, context)
        
        # Verify
        assert decision.should_continue is False  # Parent waits
        assert decision.should_wait is True
        assert len(decision.child_branch_specs) == 2
        
        # Check branch specs
        researcher_spec = next(s for s in decision.child_branch_specs if s.entry_agent == "Researcher")
        writer_spec = next(s for s in decision.child_branch_specs if s.entry_agent == "Writer")
        
        assert researcher_spec.initial_request == "Find latest AI safety research"
        assert writer_spec.initial_request == "Draft introduction based on outline"
    
    @pytest.mark.asyncio
    async def test_no_auto_conversation_detection(self, router, validator):
        """Test that router does NOT auto-convert simple branches to conversation."""
        # Start with Writer in a simple branch
        branch = ExecutionBranch(
            id="writer_branch",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Writer"],
                entry_agent="Writer"
            )
        )
        
        context = RoutingContext(
            current_branch_id="writer_branch",
            current_agent="Writer",
            conversation_history=[],
            branch_agents=["Writer"]
        )
        
        # Writer invokes Editor (should NOT start conversation)
        response = {
            "next_action": "invoke_agent",
            "action_input": "Please review this draft for clarity"
        }
        
        # Create mock agent
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Writer"
        exec_state = ExecutionState(
            session_id="test_session",
            current_step=1,
            status="running"
        )
        
        # Create validation result directly
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response=response,
            invocations=[AgentInvocation(agent_name="Editor", request="Please review this draft for clarity")]
        )
        
        # Route
        decision = await router.route(validation, branch, context)
        
        # Verify NO conversation detected - should be sequential
        assert decision.should_continue is True
        assert decision.metadata["routing_type"] == "sequential_agent"
        assert branch.type == BranchType.SIMPLE  # Branch type unchanged!
        
        # Now test actual conversation branch
        from marsys.coordination.branches.types import ConversationPattern
        conv_branch = ExecutionBranch(
            id="conv_branch",
            type=BranchType.CONVERSATION,  # Already a conversation
            topology=BranchTopology(
                agents=["Writer", "Editor"],
                entry_agent="Writer",
                conversation_pattern=ConversationPattern.DIALOGUE
            )
        )
        
        context.current_agent = "Editor"
        context.conversation_turns = 1
        
        response = {
            "next_action": "invoke_agent",
            "action_input": "I've made some suggestions, please revise"
        }
        
        # Create mock agent for Editor
        mock_editor = Mock(spec=BaseAgent)
        mock_editor.name = "Editor"
        
        # Create validation result directly
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response=response,
            invocations=[AgentInvocation(agent_name="Writer", request="I've made some suggestions, please revise")]
        )
        
        # Route within conversation branch
        decision = await router.route(validation, conv_branch, context)
        
        # NOW it should be conversation continuation
        assert decision.should_continue is True
        assert decision.metadata["routing_type"] == "conversation_continuation"
        assert decision.next_steps[0].metadata["conversation_turn"] == 2
    
    @pytest.mark.asyncio
    async def test_invalid_transition_handling(self, router, validator):
        """Test handling of invalid transitions."""
        # Setup
        branch = ExecutionBranch(
            id="test",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Researcher"],
                entry_agent="Researcher"
            )
        )
        
        context = RoutingContext(
            current_branch_id="test",
            current_agent="Researcher",
            conversation_history=[],
            branch_agents=["Researcher"]
        )
        
        # Researcher tries to invoke Publisher directly (not allowed)
        response = {
            "next_action": "invoke_agent",
            "action_input": "Publish the findings"
        }
        
        # Create mock agent
        mock_agent = Mock(spec=BaseAgent)
        mock_agent.name = "Researcher"
        exec_state = ExecutionState(
            session_id="test_session",
            current_step=1,
            status="running"
        )
        
        # Create invalid validation result
        validation = ValidationResult(
            is_valid=False,
            action_type=ActionType.INVOKE_AGENT,
            error_message="Agent Researcher cannot invoke Publisher",
            parsed_response=response,
            invocations=[AgentInvocation(agent_name="Publisher", request="Publish the findings")]
        )
        
        # Route
        decision = await router.route(validation, branch, context)
        
        # Should not continue
        assert decision.should_continue is False
        assert "cannot invoke" in decision.completion_reason
        
        # Check for alternative
        alternative = router.suggest_alternative_route(
            "Researcher",
            "Publisher",
            context
        )
        assert alternative == "Writer"  # Should suggest Writer
    
    @pytest.mark.asyncio
    async def test_complex_workflow(self, router, validator, topology_graph):
        """Test a complex multi-step workflow."""
        results = []
        
        # Step 1: Orchestrator -> Planner
        branch = ExecutionBranch(
            id="main",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Orchestrator"],
                entry_agent="Orchestrator"
            )
        )
        
        context = RoutingContext(
            current_branch_id="main",
            current_agent="Orchestrator",
            conversation_history=[],
            branch_agents=["Orchestrator"]
        )
        
        # Orchestrator delegates to Planner
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response={"action_input": "Plan the article"},
            invocations=[AgentInvocation(agent_name="Planner", request="Plan the article")]
        )
        
        decision = await router.route(validation, branch, context)
        results.append(("Orchestrator->Planner", decision))
        
        # Step 2: Planner initiates parallel execution
        context.current_agent = "Planner"
        context.branch_agents.append("Planner")
        
        validation = ValidationResult(
            is_valid=True,
            action_type=ActionType.PARALLEL_INVOKE,
            parsed_response={
                "action_input": {
                    "Researcher": "Research topic X",
                    "Writer": "Draft outline"
                }
            },
            invocations=[
                AgentInvocation(agent_name="Researcher", request="Research topic X"),
                AgentInvocation(agent_name="Writer", request="Draft outline")
            ]
        )
        
        decision = await router.route(validation, branch, context)
        results.append(("Planner->Parallel", decision))
        
        # Verify workflow
        assert results[0][1].should_continue is True
        assert results[1][1].should_wait is True
        assert len(results[1][1].child_branch_specs) == 2
        
        # Verify topology constraints were respected
        for step_name, decision in results:
            if decision.should_continue or decision.should_wait:
                assert decision.completion_reason is None
            else:
                assert decision.completion_reason is not None
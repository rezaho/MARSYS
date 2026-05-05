"""
Tests for the Router component in the coordination system.

Tests cover various routing scenarios including sequential invocation,
parallel execution, tool calls, conversation patterns, and error handling.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock

from marsys.coordination.routing import (
    Router, 
    RoutingDecision, 
    ExecutionStep, 
    StepType,
    BranchSpec,
    RoutingContext
)
from marsys.coordination.validation.response_validator import ValidationResult, ActionType
from marsys.coordination.validation.types import AgentInvocation
from marsys.coordination.branches.types import (
    ExecutionBranch,
    BranchType,
    BranchTopology,
    BranchState,
    BranchStatus
)
from marsys.coordination.topology.graph import TopologyGraph


class TestRouter:
    """Test suite for the Router component."""
    
    @pytest.fixture
    def mock_topology_graph(self):
        """Create a mock topology graph for testing."""
        graph = Mock(spec=TopologyGraph)
        
        # Setup basic graph structure
        graph.adjacency = {
            "Agent1": ["Agent2", "Agent3"],
            "Agent2": ["Agent3", "Agent4"],
            "Agent3": ["Agent2", "Agent4"],  # Bidirectional with Agent2
            "Agent4": []
        }
        
        # Mock methods
        graph.has_edge = MagicMock(side_effect=lambda f, t: t in graph.adjacency.get(f, []))
        graph.is_in_conversation_loop = MagicMock(
            side_effect=lambda a, b: (a == "Agent2" and b == "Agent3") or 
                                    (a == "Agent3" and b == "Agent2")
        )
        graph.get_next_agents = MagicMock(side_effect=lambda a: graph.adjacency.get(a, []))
        
        return graph
    
    @pytest.fixture
    def router(self, mock_topology_graph):
        """Create a Router instance with mock topology."""
        return Router(mock_topology_graph)
    
    @pytest.fixture
    def simple_branch(self):
        """Create a simple execution branch for testing."""
        return ExecutionBranch(
            id="test_branch_1",
            name="Test Branch",
            type=BranchType.SIMPLE,
            topology=BranchTopology(
                agents=["Agent1"],
                entry_agent="Agent1"
            ),
            state=BranchState(status=BranchStatus.RUNNING)
        )
    
    @pytest.fixture
    def conversation_branch(self):
        """Create a conversation branch for testing."""
        from marsys.coordination.branches.types import ConversationPattern
        topology = BranchTopology(
            agents=["Agent2", "Agent3"],
            entry_agent="Agent2",
            allowed_transitions={
                "Agent2": ["Agent3"],
                "Agent3": ["Agent2"]
            },
            conversation_pattern=ConversationPattern.DIALOGUE
        )
        return ExecutionBranch(
            id="conv_branch_1",
            name="Conversation Branch",
            type=BranchType.CONVERSATION,
            topology=topology,
            state=BranchState(status=BranchStatus.RUNNING)
        )
    
    @pytest.fixture
    def routing_context(self):
        """Create a basic routing context."""
        return RoutingContext(
            current_branch_id="test_branch_1",
            current_agent="Agent1",
            conversation_history=[],
            branch_agents=["Agent1"]
        )
    
    @pytest.mark.asyncio
    async def test_route_agent_invocation_sequential(self, router, simple_branch, routing_context):
        """Test routing for sequential agent invocation."""
        # Create validation result for agent invocation
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response={
                "next_action": "invoke_agent",
                "action_input": "Please analyze this data"
            },
            invocations=[AgentInvocation(agent_name="Agent2", request="Please analyze this data")]
        )

        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)

        # Verify the decision
        assert isinstance(decision, RoutingDecision)
        assert decision.should_continue is True
        assert decision.should_wait is False
        assert len(decision.next_steps) == 1

        # Verify the execution step
        step = decision.next_steps[0]
        assert step.step_type == StepType.AGENT
        assert step.agent_name == "Agent2"
        assert step.request == "Please analyze this data"
        assert step.metadata["from_agent"] == "Agent1"
    
    @pytest.mark.asyncio
    async def test_route_parallel_invocation(self, router, simple_branch, routing_context):
        """Test routing for parallel agent invocation."""
        # Create validation result for parallel invocation
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.PARALLEL_INVOKE,
            parsed_response={
                "next_action": "parallel_invoke",
                "action_input": {
                    "Agent2": "Task for Agent2",
                    "Agent3": "Task for Agent3"
                }
            },
            invocations=[
                AgentInvocation(agent_name="Agent2", request="Task for Agent2"),
                AgentInvocation(agent_name="Agent3", request="Task for Agent3"),
            ]
        )

        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)

        # Verify the decision
        assert decision.should_continue is False  # Parent pauses
        assert decision.should_wait is True       # Wait for children
        assert len(decision.child_branch_specs) == 2
        
        # Verify branch specifications
        branch_spec1 = decision.child_branch_specs[0]
        assert branch_spec1.agents == ["Agent2"]
        assert branch_spec1.entry_agent == "Agent2"
        assert branch_spec1.initial_request == "Task for Agent2"
        
        branch_spec2 = decision.child_branch_specs[1]
        assert branch_spec2.agents == ["Agent3"]
        assert branch_spec2.entry_agent == "Agent3"
        assert branch_spec2.initial_request == "Task for Agent3"
    
    @pytest.mark.asyncio
    async def test_route_simple_agent_invocation_no_auto_convert(self, router, routing_context):
        """Test that router does NOT auto-convert simple branches to conversation."""
        # Create a simple branch
        branch = ExecutionBranch(
            id="branch_1",
            type=BranchType.SIMPLE,
            topology=BranchTopology(agents=["Agent2"], entry_agent="Agent2")
        )
        
        # Update context for Agent2
        routing_context.current_agent = "Agent2"
        
        # Create validation result for agent invocation
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response={
                "next_action": "invoke_agent",
                "action_input": "Let's discuss this"
            },
            invocations=[AgentInvocation(agent_name="Agent3", request="Let's discuss this")]
        )

        # Route the decision
        decision = await router.route(validation_result, branch, routing_context)

        # Verify the decision - should be sequential, NOT conversation
        assert decision.should_continue is True
        assert decision.metadata["routing_type"] == "sequential_agent"
        
        # Verify branch remains SIMPLE type (NOT converted)
        assert branch.type == BranchType.SIMPLE
        
        # Verify the execution step
        step = decision.next_steps[0]
        assert step.step_type == StepType.AGENT
        assert step.agent_name == "Agent3"
        assert step.metadata["action_type"] == "invoke_agent"
        assert "conversation_turn" not in step.metadata  # Not a conversation
    
    @pytest.mark.asyncio
    async def test_route_conversation_continuation_existing_conversation(self, router, conversation_branch, routing_context):
        """Test routing within an existing conversation branch."""
        # Update context for conversation
        routing_context.current_agent = "Agent2"
        routing_context.conversation_turns = 2
        
        # Create validation result for continuing conversation
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response={
                "next_action": "invoke_agent",
                "action_input": "I agree with your point"
            },
            invocations=[AgentInvocation(agent_name="Agent3", request="I agree with your point")]
        )

        # Route the decision
        decision = await router.route(validation_result, conversation_branch, routing_context)

        # Verify the decision - should be conversation continuation
        assert decision.should_continue is True
        assert decision.metadata["routing_type"] == "conversation_continuation"
        
        # Verify branch remains CONVERSATION type
        assert conversation_branch.type == BranchType.CONVERSATION
        
        # Verify the execution step
        step = decision.next_steps[0]
        assert step.step_type == StepType.AGENT
        assert step.agent_name == "Agent3"
        assert step.metadata["conversation_turn"] == 3  # Incremented from context
        assert step.metadata["action_type"] == "conversation_continuation"
    
    @pytest.mark.asyncio
    async def test_route_final_response(self, router, simple_branch, routing_context):
        """Test routing for final response."""
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.FINAL_RESPONSE,
            parsed_response={
                "next_action": "final_response",
                "final_response": "Analysis complete. The result is X."
            }
        )
        
        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)
        
        # Verify the decision
        assert decision.should_continue is False
        assert decision.completion_reason == "Agent provided final response"
        
        # Verify the completion step
        step = decision.next_steps[0]
        assert step.step_type == StepType.COMPLETE
        assert step.request == "Analysis complete. The result is X."
    
    @pytest.mark.asyncio
    async def test_route_invalid_transition(self, router, simple_branch, routing_context):
        """Test routing with invalid agent transition."""
        # Try to invoke Agent4 from Agent1 (not allowed)
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.INVOKE_AGENT,
            parsed_response={"next_action": "invoke_agent"},
            invocations=[AgentInvocation(agent_name="Agent4", request="some task")]
        )

        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)

        # Should complete due to invalid transition
        assert decision.should_continue is False
        assert "not allowed" in decision.completion_reason
    
    @pytest.mark.asyncio
    async def test_route_invalid_result_with_retry(self, router, simple_branch, routing_context):
        """Test routing with invalid result that suggests retry."""
        validation_result = ValidationResult(
            is_valid=False,
            error_message="Invalid JSON format",
            retry_suggestion="Please provide a valid JSON response with 'next_action' field"
        )
        
        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)
        
        # Should continue with retry
        assert decision.should_continue is True
        assert decision.metadata["routing_type"] == "retry"
        
        # Verify retry step
        step = decision.next_steps[0]
        assert step.step_type == StepType.AGENT
        assert step.agent_name == "Agent1"  # Same agent retries
        assert step.request == validation_result.retry_suggestion
    
    @pytest.mark.asyncio
    async def test_suggest_alternative_route(self, router, routing_context):
        """Test alternative route suggestion."""
        # Agent1 can go to Agent2 or Agent3
        alternative = router.suggest_alternative_route("Agent1", "Agent2", routing_context)
        assert alternative == "Agent3"
        
        # If Agent3 is also visited, no alternatives
        routing_context.branch_agents = ["Agent1", "Agent3"]
        alternative = router.suggest_alternative_route("Agent1", "Agent2", routing_context)
        assert alternative is None
    
    @pytest.mark.asyncio
    async def test_parallel_invocation_validation_failure(self, router, simple_branch, routing_context):
        """Test parallel invocation with invalid transitions."""
        # Try parallel invocation with Agent4 (not reachable from Agent1)
        validation_result = ValidationResult(
            is_valid=True,
            action_type=ActionType.PARALLEL_INVOKE,
            parsed_response={"next_action": "parallel_invoke"},
            invocations=[
                AgentInvocation(agent_name="Agent2", request="task2"),
                AgentInvocation(agent_name="Agent4", request="task4"),
            ]  # Agent4 not reachable
        )

        # Route the decision
        decision = await router.route(validation_result, simple_branch, routing_context)

        # Should fail due to invalid transition
        assert decision.should_continue is False
        assert "Invalid transitions" in decision.completion_reason
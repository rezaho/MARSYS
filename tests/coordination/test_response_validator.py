"""
Tests for ValidationProcessor - coordination tool validation and error message handling.

Tests cover:
- validate_coordination_action(): topology checks for invoke_agent, return_final_response, end_conversation
- process_error_message(): API error classification and routing
- ValidationResult dataclass behavior
"""

import pytest
import json
from unittest.mock import Mock, MagicMock

from marsys.coordination.validation.response_validator import (
    ValidationProcessor,
    ValidationResult,
    ActionType,
)
from marsys.coordination.validation.types import AgentInvocation, ValidationErrorCategory
from marsys.coordination.topology.graph import TopologyGraph
from marsys.coordination.branches.types import (
    BranchTopology, BranchType, BranchState, BranchStatus,
    ExecutionBranch, ExecutionState,
)
from marsys.agents.memory import Message


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def topology_graph():
    """Topology: Agent1 → Agent2, Agent3; Agent2 → Agent1; Agent3 → Agent1.
    Agent1 is an exit point (has user access)."""
    graph = Mock(spec=TopologyGraph)
    edges = {
        "Agent1": ["Agent2", "Agent3"],
        "Agent2": ["Agent1"],
        "Agent3": ["Agent1"],
    }
    graph.get_next_agents = MagicMock(side_effect=lambda a: edges.get(a, []))
    graph.has_user_access = MagicMock(side_effect=lambda a: a == "Agent1")
    return graph


@pytest.fixture
def validator(topology_graph):
    return ValidationProcessor(topology_graph)


@pytest.fixture
def agent1():
    from marsys.agents import BaseAgent
    agent = Mock(spec=BaseAgent)
    agent.name = "Agent1"
    return agent


@pytest.fixture
def agent2():
    from marsys.agents import BaseAgent
    agent = Mock(spec=BaseAgent)
    agent.name = "Agent2"
    return agent


@pytest.fixture
def simple_branch():
    branch = Mock(spec=ExecutionBranch)
    branch.type = BranchType.SIMPLE
    branch.id = "test_branch"
    branch.state = Mock(spec=BranchState)
    branch.state.status = BranchStatus.RUNNING
    return branch


@pytest.fixture
def conversation_branch():
    branch = Mock(spec=ExecutionBranch)
    branch.type = BranchType.CONVERSATION
    branch.id = "test_conv_branch"
    branch.state = Mock(spec=BranchState)
    branch.state.status = BranchStatus.RUNNING
    return branch


@pytest.fixture
def exec_state():
    return ExecutionState(session_id="test-session", current_step=1, status="running")


# ── validate_coordination_action tests ───────────────────────────

class TestValidateCoordinationAction:

    @pytest.mark.asyncio
    async def test_invoke_agent_single_target(self, validator, agent1, simple_branch, exec_state):
        """Single invocation → INVOKE_AGENT action type."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={"invocations": [{"agent_name": "Agent2", "request": "analyze data"}]},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is True
        assert result.action_type == ActionType.INVOKE_AGENT
        assert len(result.invocations) == 1
        assert result.invocations[0].agent_name == "Agent2"
        assert result.invocations[0].request == "analyze data"
        assert result.next_agent == "Agent2"

    @pytest.mark.asyncio
    async def test_invoke_agent_parallel(self, validator, agent1, simple_branch, exec_state):
        """Multiple invocations → PARALLEL_INVOKE action type."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={"invocations": [
                {"agent_name": "Agent2", "request": "task A"},
                {"agent_name": "Agent3", "request": "task B"},
            ]},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is True
        assert result.action_type == ActionType.PARALLEL_INVOKE
        assert len(result.invocations) == 2
        assert set(result.next_agents) == {"Agent2", "Agent3"}

    @pytest.mark.asyncio
    async def test_invoke_agent_topology_denied(self, validator, agent2, simple_branch, exec_state):
        """Agent2 tries to invoke Agent3 but topology only allows Agent2 → Agent1."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={"invocations": [{"agent_name": "Agent3", "request": "task"}]},
            agent=agent2, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False
        assert result.error_category == ValidationErrorCategory.PERMISSION_ERROR.value
        assert "Agent3" in result.error_message

    @pytest.mark.asyncio
    async def test_invoke_agent_empty_invocations(self, validator, agent1, simple_branch, exec_state):
        """Empty invocations array → ACTION_ERROR."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={"invocations": []},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False
        assert result.error_category == ValidationErrorCategory.ACTION_ERROR.value

    @pytest.mark.asyncio
    async def test_invoke_agent_missing_invocations_key(self, validator, agent1, simple_branch, exec_state):
        """Missing invocations key entirely."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_return_final_response_with_access(self, validator, agent1, simple_branch, exec_state):
        """Agent1 has user access → FINAL_RESPONSE allowed."""
        result = await validator.validate_coordination_action(
            action="return_final_response",
            data={"response": "The answer is 42."},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is True
        assert result.action_type == ActionType.FINAL_RESPONSE
        assert result.parsed_response["final_response"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_return_final_response_without_access(self, validator, agent2, simple_branch, exec_state):
        """Agent2 has no user access → PERMISSION_ERROR."""
        result = await validator.validate_coordination_action(
            action="return_final_response",
            data={"response": "done"},
            agent=agent2, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False
        assert result.error_category == ValidationErrorCategory.PERMISSION_ERROR.value

    @pytest.mark.asyncio
    async def test_return_final_response_structured_output(self, validator, agent1, simple_branch, exec_state):
        """Structured output schema — data dict without 'response' key is the response."""
        result = await validator.validate_coordination_action(
            action="return_final_response",
            data={"title": "Report", "findings": ["a", "b"]},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is True
        assert result.action_type == ActionType.FINAL_RESPONSE
        # The entire data dict becomes the response
        assert result.parsed_response["final_response"] == {"title": "Report", "findings": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_end_conversation_in_conversation_branch(self, validator, agent1, conversation_branch, exec_state):
        """end_conversation in conversation branch → allowed."""
        result = await validator.validate_coordination_action(
            action="end_conversation",
            data={"summary": "We reached consensus."},
            agent=agent1, branch=conversation_branch, exec_state=exec_state,
        )
        assert result.is_valid is True
        assert result.action_type == ActionType.END_CONVERSATION

    @pytest.mark.asyncio
    async def test_end_conversation_in_simple_branch(self, validator, agent1, simple_branch, exec_state):
        """end_conversation in simple branch → rejected."""
        result = await validator.validate_coordination_action(
            action="end_conversation",
            data={"summary": "done"},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_unknown_action(self, validator, agent1, simple_branch, exec_state):
        """Unknown coordination action → ACTION_ERROR."""
        result = await validator.validate_coordination_action(
            action="do_something_weird",
            data={},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        assert result.is_valid is False
        assert result.error_category == ValidationErrorCategory.ACTION_ERROR.value

    @pytest.mark.asyncio
    async def test_invocation_instance_ids_unique(self, validator, agent1, simple_branch, exec_state):
        """Each AgentInvocation gets a unique instance_id."""
        result = await validator.validate_coordination_action(
            action="invoke_agent",
            data={"invocations": [
                {"agent_name": "Agent2", "request": "a"},
                {"agent_name": "Agent3", "request": "b"},
            ]},
            agent=agent1, branch=simple_branch, exec_state=exec_state,
        )
        ids = [inv.instance_id for inv in result.invocations]
        assert len(set(ids)) == 2  # All unique


# ── process_error_message tests ──────────────────────────────────

class TestProcessErrorMessage:

    @pytest.mark.asyncio
    async def test_auto_retry_transient_error(self, validator, agent1, simple_branch, exec_state):
        """Transient error (timeout) with retryable flag → AUTO_RETRY."""
        error_content = json.dumps({
            "error": "Request timed out",
            "classification": "timeout",
            "is_retryable": True,
            "provider": "anthropic",
        })
        msg = Message(role="error", content=error_content)

        result = await validator.process_error_message(msg, agent1, simple_branch, exec_state)

        assert result.is_valid is True
        assert result.action_type == ActionType.AUTO_RETRY
        assert result.parsed_response["error_info"]["classification"] == "timeout"

    @pytest.mark.asyncio
    async def test_terminal_error_auth_failed(self, validator, agent1, simple_branch, exec_state):
        """Auth failure → TERMINAL_ERROR routed to User."""
        error_content = json.dumps({
            "error": "Invalid API key",
            "classification": "authentication_failed",
            "is_retryable": False,
        })
        msg = Message(role="error", content=error_content)

        result = await validator.process_error_message(msg, agent1, simple_branch, exec_state)

        assert result.is_valid is True
        assert result.action_type == ActionType.TERMINAL_ERROR
        assert len(result.invocations) == 1
        assert result.invocations[0].agent_name == "User"
        error_details = result.invocations[0].request["error_details"]
        assert "termination_reason" in error_details

    @pytest.mark.asyncio
    async def test_error_recovery_unknown_error(self, validator, agent1, simple_branch, exec_state):
        """Unknown classification → ERROR_RECOVERY routed to User."""
        error_content = json.dumps({
            "error": "Something unexpected happened",
            "classification": "unknown",
            "is_retryable": False,
        })
        msg = Message(role="error", content=error_content)

        result = await validator.process_error_message(msg, agent1, simple_branch, exec_state)

        assert result.is_valid is True
        assert result.action_type == ActionType.ERROR_RECOVERY
        assert len(result.invocations) == 1
        assert result.invocations[0].agent_name == "User"

    @pytest.mark.asyncio
    async def test_non_error_message_rejected(self, validator, agent1, simple_branch, exec_state):
        """Regular assistant message → rejected by process_error_message."""
        msg = Message(role="assistant", content="Hello!")

        result = await validator.process_error_message(msg, agent1, simple_branch, exec_state)

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_error_recovery_insufficient_credits(self, validator, agent1, simple_branch, exec_state):
        """Insufficient credits → ERROR_RECOVERY with suggested actions."""
        error_content = json.dumps({
            "error": "Insufficient credits",
            "classification": "insufficient_credits",
            "is_retryable": False,
            "suggested_action": "Check your billing dashboard",
        })
        msg = Message(role="error", content=error_content)

        result = await validator.process_error_message(msg, agent1, simple_branch, exec_state)

        assert result.is_valid is True
        assert result.action_type == ActionType.ERROR_RECOVERY
        error_details = result.invocations[0].request["error_details"]
        assert "suggested_actions" in error_details
        assert len(error_details["suggested_actions"]) >= 2  # original + "Add credits" + "retry"


# ── ValidationResult tests ───────────────────────────────────────

class TestValidationResult:

    def test_defaults(self):
        result = ValidationResult(is_valid=True)
        assert result.action_type is None
        assert result.next_agents == []
        assert result.tool_calls == []
        assert result.invocations == []

    def test_auto_populate_next_agent(self):
        """Single invocation auto-populates next_agent."""
        result = ValidationResult(
            is_valid=True,
            invocations=[AgentInvocation(agent_name="Agent2", request="task")],
        )
        assert result.next_agent == "Agent2"

    def test_next_agents_property(self):
        result = ValidationResult(
            is_valid=True,
            invocations=[
                AgentInvocation(agent_name="A", request="x"),
                AgentInvocation(agent_name="B", request="y"),
            ],
        )
        assert set(result.next_agents) == {"A", "B"}

    def test_error_result(self):
        result = ValidationResult(
            is_valid=False,
            error_message="Something went wrong",
            retry_suggestion="Try again",
            error_category=ValidationErrorCategory.PERMISSION_ERROR.value,
        )
        assert result.is_valid is False
        assert "wrong" in result.error_message

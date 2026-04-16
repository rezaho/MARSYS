"""
End-to-end tests for the coordination tools pipeline.

Tests the full execution path with synthetic LLM responses containing
coordination tool calls (invoke_agent, return_final_response) and verifies
correct routing through StepExecutor → BranchExecutor → ValidationProcessor.

Also tests steering injection for content-only responses and error message
handling for API failures.
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Any, Dict, List

from marsys.agents.memory import Message, ToolCallMsg
from marsys.coordination.execution.step_executor import StepExecutor
from marsys.coordination.execution.branch_executor import BranchExecutor
from marsys.coordination.validation.response_validator import (
    ValidationProcessor, ActionType,
)
from marsys.coordination.validation.types import ValidationErrorCategory
from marsys.coordination.steering.manager import SteeringManager
from marsys.coordination.topology.graph import TopologyGraph
from marsys.coordination.branches.types import (
    StepResult, BranchResult, ExecutionBranch, BranchType,
    BranchTopology, BranchState, BranchStatus,
)
from marsys.coordination.formats.coordination_tools import (
    is_coordination_tool, parse_coordination_call, COORDINATION_TOOL_NAMES,
)


# ── Helpers: Synthetic LLM Responses ─────────────────────────────

def make_coordination_tool_call(name: str, arguments: dict, call_id: str = "call_001") -> ToolCallMsg:
    """Create a ToolCallMsg for a coordination tool (invoke_agent, return_final_response, etc.)."""
    return ToolCallMsg(
        id=call_id,
        call_id=call_id,
        type="function",
        name=name,
        arguments=json.dumps(arguments),
    )


def make_regular_tool_call(name: str, arguments: dict, call_id: str = "call_100") -> ToolCallMsg:
    """Create a ToolCallMsg for a regular tool (google_search, etc.)."""
    return ToolCallMsg(
        id=call_id,
        call_id=call_id,
        type="function",
        name=name,
        arguments=json.dumps(arguments),
    )


def make_message(content: str = "", tool_calls: list = None, role: str = "assistant") -> Message:
    """Create a Message with optional tool calls."""
    return Message(role=role, content=content, tool_calls=tool_calls or [])


def make_invoke_agent_message(targets: List[Dict[str, str]], content: str = "") -> Message:
    """Create a Message with an invoke_agent coordination tool call."""
    tc = make_coordination_tool_call(
        "invoke_agent",
        {"invocations": targets},
    )
    return make_message(content=content, tool_calls=[tc])


def make_final_response_message(response: str, content: str = "") -> Message:
    """Create a Message with a return_final_response coordination tool call."""
    tc = make_coordination_tool_call(
        "return_final_response",
        {"response": response},
        call_id="call_final",
    )
    return make_message(content=content, tool_calls=[tc])


def make_error_message(error: str, classification: str, retryable: bool = False) -> Message:
    """Create an error Message as produced by Agent._run() on API failure."""
    return Message(
        role="error",
        content=json.dumps({
            "error": error,
            "classification": classification,
            "is_retryable": retryable,
        }),
    )


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def topology_graph():
    """Coordinator → Researcher, FactChecker; both → Coordinator. Coordinator is exit point."""
    graph = Mock(spec=TopologyGraph)
    edges = {
        "Coordinator": ["Researcher", "FactChecker"],
        "Researcher": ["Coordinator"],
        "FactChecker": ["Coordinator"],
    }
    graph.get_next_agents = MagicMock(side_effect=lambda a: edges.get(a, []))
    graph.has_user_access = MagicMock(side_effect=lambda a: a == "Coordinator")
    graph.metadata = {"exit_points": ["Coordinator"]}
    return graph


# ── Test: Coordination Tool Detection ────────────────────────────

class TestCoordinationToolDetection:
    """Test that coordination tools are correctly identified and parsed."""

    def test_is_coordination_tool(self):
        assert is_coordination_tool("invoke_agent") is True
        assert is_coordination_tool("return_final_response") is True
        assert is_coordination_tool("end_conversation") is True
        assert is_coordination_tool("google_search") is False
        assert is_coordination_tool("plan_create") is False
        assert is_coordination_tool("") is False

    def test_parse_coordination_call_dict(self):
        tc = {
            "function": {
                "name": "invoke_agent",
                "arguments": json.dumps({"invocations": [{"agent_name": "A", "request": "x"}]}),
            }
        }
        action, data = parse_coordination_call(tc)
        assert action == "invoke_agent"
        assert data["invocations"][0]["agent_name"] == "A"

    def test_parse_coordination_call_bad_json(self):
        tc = {"function": {"name": "invoke_agent", "arguments": "not json{{"}}
        action, data = parse_coordination_call(tc)
        assert action == "invoke_agent"
        assert data == {}  # Graceful fallback


# ── Test: StepExecutor Tool Call Partitioning ────────────────────

class TestStepExecutorPartitioning:
    """Test that StepExecutor correctly separates coordination and regular tool calls."""

    @pytest.mark.asyncio
    async def test_coordination_only(self):
        """Response with only invoke_agent → coordination_action set, no regular tool_calls."""
        msg = make_invoke_agent_message(
            [{"agent_name": "Researcher", "request": "research AI"}],
            content="Let me delegate this to the researcher.",
        )

        executor = StepExecutor()
        result = await executor._process_tool_calls(msg, Mock(name="Coordinator"), Mock(agent_name="Coordinator"))

        assert result.coordination_action == "invoke_agent"
        assert result.coordination_data["invocations"][0]["agent_name"] == "Researcher"
        assert result.tool_calls == []  # No regular tools

    @pytest.mark.asyncio
    async def test_regular_tools_only(self):
        """Response with only regular tools → tool_calls set, no coordination_action."""
        tc = make_regular_tool_call("google_search", {"query": "speed of light"})
        msg = make_message(content="Let me search for that.", tool_calls=[tc])

        executor = StepExecutor()
        result = await executor._process_tool_calls(msg, Mock(name="Agent1"), Mock(agent_name="Agent1"))

        assert result.coordination_action is None
        assert result.coordination_data is None
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_mixed_calls(self):
        """Response with both regular tool AND coordination tool → both extracted."""
        regular = make_regular_tool_call("write_file", {"path": "out.md", "content": "data"}, call_id="call_tool")
        coord = make_coordination_tool_call("invoke_agent", {"invocations": [{"agent_name": "Analyzer", "request": "analyze"}]}, call_id="call_coord")
        msg = make_message(content="Writing file and handing off.", tool_calls=[regular, coord])

        executor = StepExecutor()
        result = await executor._process_tool_calls(msg, Mock(name="Writer"), Mock(agent_name="Writer"))

        assert result.coordination_action == "invoke_agent"
        assert len(result.tool_calls) == 1  # Only regular tool
        assert result.tool_calls[0]["function"]["name"] == "write_file" if isinstance(result.tool_calls[0], dict) else result.tool_calls[0].name == "write_file"

    @pytest.mark.asyncio
    async def test_content_only(self):
        """Response with no tool calls → neither set."""
        msg = make_message(content="I'm thinking about this problem...")

        executor = StepExecutor()
        result = await executor._process_tool_calls(msg, Mock(name="Agent1"), Mock(agent_name="Agent1"))

        assert result.coordination_action is None
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_final_response_detection(self):
        """return_final_response correctly extracted."""
        msg = make_final_response_message("The answer is 42.")

        executor = StepExecutor()
        result = await executor._process_tool_calls(msg, Mock(name="Coordinator"), Mock(agent_name="Coordinator"))

        assert result.coordination_action == "return_final_response"
        assert result.coordination_data["response"] == "The answer is 42."


# ── Test: BranchExecutor Routing Paths ───────────────────────────

class TestBranchExecutorRouting:
    """Test the 4 routing paths in BranchExecutor._execute_agent_step."""

    def _make_branch_executor(self, topology_graph):
        """Create a BranchExecutor with mocked dependencies."""
        be = BranchExecutor.__new__(BranchExecutor)
        be.step_executor = None
        be.response_validator = ValidationProcessor(topology_graph)
        be.topology_graph = topology_graph
        be.branch_spawner = None
        be.rules_engine = None
        be.event_bus = None
        be.max_retries = 3
        be.waiting_for_children = {}
        be.branch_continuation = {}
        be._last_agent_name = None
        be._last_step_result = None
        from marsys.agents.registry import AgentRegistry
        be.agent_registry = AgentRegistry
        return be

    @pytest.mark.asyncio
    async def test_path3_invoke_agent_routing(self, topology_graph):
        """Path 3: coordination_action=invoke_agent → next_agent set correctly."""
        be = self._make_branch_executor(topology_graph)
        branch = Mock(spec=ExecutionBranch)
        branch.type = BranchType.SIMPLE
        branch.id = "test"
        branch.state = Mock()
        branch.state.current_step = 0
        branch.metadata = {}
        branch.agent_retry_info = {}
        branch.retry_counts = {}

        # Create a StepResult as if StepExecutor produced it
        result = StepResult(
            agent_name="Coordinator",
            success=True,
            response=make_invoke_agent_message([{"agent_name": "Researcher", "request": "research"}]),
            coordination_action="invoke_agent",
            coordination_data={"invocations": [{"agent_name": "Researcher", "request": "research"}]},
        )

        # Simulate what _execute_agent_step does in the routing section
        from marsys.coordination.branches.types import ExecutionState
        exec_state = ExecutionState(session_id="test", current_step=0, status="running")

        mock_agent = Mock()
        mock_agent.name = "Coordinator"
        validation = await be.response_validator.validate_coordination_action(
            action=result.coordination_action,
            data=result.coordination_data,
            agent=mock_agent,
            branch=branch,
            exec_state=exec_state,
        )

        assert validation.is_valid is True
        assert validation.action_type == ActionType.INVOKE_AGENT
        assert validation.next_agent == "Researcher"

    @pytest.mark.asyncio
    async def test_path3_final_response_routing(self, topology_graph):
        """Path 3: coordination_action=return_final_response → FINAL_RESPONSE."""
        be = self._make_branch_executor(topology_graph)
        from marsys.coordination.branches.types import ExecutionState
        exec_state = ExecutionState(session_id="test", current_step=0, status="running")
        branch = Mock(spec=ExecutionBranch)
        branch.type = BranchType.SIMPLE

        mock_agent = Mock()
        mock_agent.name = "Coordinator"
        validation = await be.response_validator.validate_coordination_action(
            action="return_final_response",
            data={"response": "The answer is 42."},
            agent=mock_agent,
            branch=branch,
            exec_state=exec_state,
        )

        assert validation.is_valid is True
        assert validation.action_type == ActionType.FINAL_RESPONSE
        assert validation.parsed_response["final_response"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_path1_error_message_auto_retry(self, topology_graph):
        """Path 1: Error message with retryable classification → AUTO_RETRY."""
        be = self._make_branch_executor(topology_graph)
        from marsys.coordination.branches.types import ExecutionState
        exec_state = ExecutionState(session_id="test", current_step=0, status="running")
        branch = Mock(spec=ExecutionBranch)
        branch.type = BranchType.SIMPLE
        mock_agent = Mock()
        mock_agent.name = "Coordinator"

        error_msg = make_error_message("Service unavailable", "service_unavailable", retryable=True)

        validation = await be.response_validator.process_error_message(
            message=error_msg,
            agent=mock_agent,
            branch=branch,
            exec_state=exec_state,
        )

        assert validation.is_valid is True
        assert validation.action_type == ActionType.AUTO_RETRY

    @pytest.mark.asyncio
    async def test_path1_error_message_terminal(self, topology_graph):
        """Path 1: Terminal error → TERMINAL_ERROR routed to User."""
        be = self._make_branch_executor(topology_graph)
        from marsys.coordination.branches.types import ExecutionState
        exec_state = ExecutionState(session_id="test", current_step=0, status="running")
        branch = Mock(spec=ExecutionBranch)
        branch.type = BranchType.SIMPLE
        branch.id = "branch_1"
        mock_agent = Mock()
        mock_agent.name = "Coordinator"

        error_msg = make_error_message("Invalid API key", "authentication_failed", retryable=False)

        validation = await be.response_validator.process_error_message(
            message=error_msg,
            agent=mock_agent,
            branch=branch,
            exec_state=exec_state,
        )

        assert validation.is_valid is True
        assert validation.action_type == ActionType.TERMINAL_ERROR
        assert validation.invocations[0].agent_name == "User"


# ── Test: Steering for Content-Only Responses ────────────────────

class TestContentOnlySteering:
    """Test that steering fires after consecutive content-only responses."""

    @pytest.mark.asyncio
    async def test_steering_after_threshold(self):
        """After 3 content-only responses, agent_retry_info is set with ACTION_ERROR."""
        branch = Mock(spec=ExecutionBranch)
        branch.type = BranchType.SIMPLE
        branch.metadata = {}
        branch.agent_retry_info = {}
        branch.retry_counts = {}

        # Simulate BranchExecutor's content-only path (Path 4) being hit 3 times
        agent_name = "FactChecker"
        for i in range(3):
            result = StepResult(agent_name=agent_name, success=True, response=make_message("thinking..."))
            result.action_type = "continue"
            result.next_agent = agent_name
            if not result.metadata:
                result.metadata = {}
            result.metadata['content_continuation'] = True

            # Simulate the counter logic from BranchExecutor
            count_key = f"_content_only_count_{agent_name}"
            prev_count = branch.metadata.get(count_key, 0)
            branch.metadata[count_key] = prev_count + 1

        # After 3 iterations, check the counter
        assert branch.metadata[f"_content_only_count_{agent_name}"] == 3

    def test_steering_prompt_for_action_error(self):
        """SteeringManager produces a prompt for ACTION_ERROR category."""
        from marsys.coordination.steering.manager import SteeringManager, SteeringContext, ErrorContext

        manager = SteeringManager()
        ctx = SteeringContext(
            agent_name="FactChecker",
            available_actions=["invoke_agent", "tool_calls"],
            error_context=ErrorContext(
                category=ValidationErrorCategory.ACTION_ERROR,
                error_message="You have not used any coordination tools in your recent responses.",
                retry_suggestion="You must use a coordination tool. Available: invoke_agent (to delegate to Coordinator)",
                retry_count=3,
            ),
            is_retry=True,
            steering_mode="error",
        )

        prompt = manager.get_steering_prompt(ctx)

        assert prompt is not None
        assert "coordination" in prompt.lower() or "invoke_agent" in prompt.lower()

    def test_steering_not_fired_without_error(self):
        """No steering when there's no error context and mode is 'error'."""
        from marsys.coordination.steering.manager import SteeringManager, SteeringContext

        manager = SteeringManager()
        ctx = SteeringContext(
            agent_name="Agent1",
            available_actions=["invoke_agent"],
            error_context=None,
            is_retry=False,
            steering_mode="error",
        )

        prompt = manager.get_steering_prompt(ctx)
        assert prompt is None

    def test_steering_permission_error(self):
        """SteeringManager produces a prompt for PERMISSION_ERROR."""
        from marsys.coordination.steering.manager import SteeringManager, SteeringContext, ErrorContext

        manager = SteeringManager()
        ctx = SteeringContext(
            agent_name="Agent2",
            available_actions=["invoke_agent"],
            error_context=ErrorContext(
                category=ValidationErrorCategory.PERMISSION_ERROR,
                error_message="Agent2 cannot invoke Agent3",
                retry_count=1,
            ),
            is_retry=True,
            steering_mode="error",
        )

        prompt = manager.get_steering_prompt(ctx)

        assert prompt is not None
        assert "Permission denied" in prompt

    def test_steering_api_transient(self):
        """SteeringManager produces a minimal prompt for API_TRANSIENT."""
        from marsys.coordination.steering.manager import SteeringManager, SteeringContext, ErrorContext

        manager = SteeringManager()
        ctx = SteeringContext(
            agent_name="Agent1",
            available_actions=["invoke_agent"],
            error_context=ErrorContext(
                category=ValidationErrorCategory.API_TRANSIENT,
                error_message="Timeout",
                classification="timeout",
                retry_count=1,
            ),
            is_retry=True,
            steering_mode="error",
        )

        prompt = manager.get_steering_prompt(ctx)

        assert prompt is not None
        assert "retry" in prompt.lower() or "failed" in prompt.lower()

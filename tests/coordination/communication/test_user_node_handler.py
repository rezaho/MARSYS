"""
Tests for the UserNodeHandler.

Tests cover:
- User node execution handling
- Communication mode detection
- Calling agent determination
- Resume agent determination
- Interaction types
- Sync/async mode handling
- Error recovery handling
- Task formatting
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import time
import uuid

from marsys.coordination.communication.core import (
    CommunicationMode,
    UserInteraction,
)
from marsys.coordination.communication.manager import CommunicationManager
from marsys.coordination.communication.user_node_handler import UserNodeHandler
from marsys.coordination.branches.types import StepResult, BranchStatus


# ==============================================================================
# Mock Objects
# ==============================================================================


class MockBranch:
    """Mock ExecutionBranch for testing."""

    def __init__(self, branch_id: str = "test-branch"):
        self.id = branch_id
        self.state = MockBranchState()
        self.topology = MockTopology()
        self._execution_trace = []
        self.metadata = {}


class MockBranchState:
    """Mock branch state."""

    def __init__(self):
        self.step_count = 0
        self.memory = []
        self.awaiting_user_response = False
        self.interaction_id = None
        self.calling_agent = None
        self.resume_agent = None
        self.user_wait_start_time = None
        self.total_user_wait_time = 0.0
        self.interaction_context = {}
        self.status = BranchStatus.RUNNING


class MockTopology:
    """Mock topology."""

    def __init__(self):
        self.agents = ["User", "Coordinator", "Worker"]
        self.allowed_transitions = {
            "User": ["Coordinator"],
            "Coordinator": ["User", "Worker"],
            "Worker": ["Coordinator"],
        }
        self.current_agent = None


class MockStep:
    """Mock execution step."""

    def __init__(self, agent_name: str, action_type: str = "invoke_agent", success: bool = True):
        self.agent_name = agent_name
        self.action_type = action_type
        self.success = success


class MockCommunicationManager:
    """Mock CommunicationManager for testing."""

    def __init__(self):
        self.handled_interactions = []
        self.response = "mock_user_response"

    async def handle_interaction(self, interaction):
        self.handled_interactions.append(interaction)
        return self.response


# ==============================================================================
# UserNodeHandler Tests
# ==============================================================================


class TestUserNodeHandlerInitialization:
    """Tests for UserNodeHandler initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        comm_manager = CommunicationManager()
        handler = UserNodeHandler(comm_manager)

        assert handler.communication_manager == comm_manager
        assert handler.event_bus is None

    def test_initialization_with_event_bus(self):
        """Test initialization with event bus."""
        comm_manager = CommunicationManager()
        event_bus = Mock()
        handler = UserNodeHandler(comm_manager, event_bus=event_bus)

        assert handler.event_bus == event_bus


class TestCommunicationModeDetection:
    """Tests for communication mode detection."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_determine_sync_mode(self, handler):
        """Test detecting sync mode."""
        context = {"communication_mode": "sync"}
        mode = handler._determine_communication_mode(context)
        assert mode == CommunicationMode.SYNC

    def test_determine_async_pubsub_mode(self, handler):
        """Test detecting async pub/sub mode."""
        context = {"communication_mode": "async_pubsub"}
        mode = handler._determine_communication_mode(context)
        assert mode == CommunicationMode.ASYNC_PUBSUB

    def test_determine_async_queue_mode(self, handler):
        """Test detecting async queue mode."""
        context = {"communication_mode": "async_queue"}
        mode = handler._determine_communication_mode(context)
        assert mode == CommunicationMode.ASYNC_QUEUE

    def test_default_mode_is_sync(self, handler):
        """Test default mode is sync."""
        context = {}
        mode = handler._determine_communication_mode(context)
        assert mode == CommunicationMode.SYNC

    def test_invalid_mode_defaults_to_sync(self, handler):
        """Test invalid mode defaults to sync."""
        context = {"communication_mode": "invalid_mode"}
        mode = handler._determine_communication_mode(context)
        assert mode == CommunicationMode.SYNC


class TestCallingAgentDetection:
    """Tests for calling agent detection."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_get_calling_agent_from_trace(self, handler):
        """Test getting calling agent from execution trace."""
        branch = MockBranch()
        branch._execution_trace = [
            MockStep("Coordinator"),
            MockStep("Worker"),
        ]

        agent = handler._get_calling_agent(branch)

        assert agent == "Worker"

    def test_get_calling_agent_skips_user(self, handler):
        """Test that User is skipped in trace."""
        branch = MockBranch()
        branch._execution_trace = [
            MockStep("Coordinator"),
            MockStep("User"),
        ]

        agent = handler._get_calling_agent(branch)

        assert agent == "Coordinator"

    def test_get_calling_agent_from_state(self, handler):
        """Test getting calling agent from branch state."""
        branch = MockBranch()
        branch._execution_trace = []
        branch.state.calling_agent = "StateAgent"

        agent = handler._get_calling_agent(branch)

        assert agent == "StateAgent"

    def test_get_calling_agent_entry_point(self, handler):
        """Test entry point returns System."""
        branch = MockBranch()
        branch._execution_trace = []
        branch.state.calling_agent = None

        agent = handler._get_calling_agent(branch)

        assert agent == "System"


class TestResumeAgentDetermination:
    """Tests for resume agent determination."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_resume_to_calling_agent(self, handler):
        """Test resuming to calling agent when in transitions."""
        branch = MockBranch()

        resume = handler._determine_resume_agent(branch, "Coordinator")

        assert resume == "Coordinator"

    def test_resume_to_different_agent(self, handler):
        """Test resuming to different agent from transitions."""
        branch = MockBranch()
        branch.topology.allowed_transitions["User"] = ["Worker", "Coordinator"]

        # Should return first non-calling agent
        resume = handler._determine_resume_agent(branch, "Worker")

        # Coordinator is first in transitions after skipping Worker
        assert resume in ["Worker", "Coordinator"]

    def test_resume_default_to_calling(self, handler):
        """Test default resume is calling agent."""
        branch = MockBranch()
        branch.topology.allowed_transitions = {}

        resume = handler._determine_resume_agent(branch, "TestAgent")

        assert resume == "TestAgent"


class TestInteractionTypeDetermination:
    """Tests for interaction type determination."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_determine_type_from_dict(self, handler):
        """Test determining type from dict with explicit type."""
        message = {"interaction_type": "choice"}
        int_type = handler._determine_interaction_type(message)
        assert int_type == "choice"

    def test_determine_type_choice(self, handler):
        """Test determining choice type from options."""
        message = {"options": ["A", "B", "C"]}
        int_type = handler._determine_interaction_type(message)
        assert int_type == "choice"

    def test_determine_type_confirmation(self, handler):
        """Test determining confirmation type."""
        message = {"confirm": True}
        int_type = handler._determine_interaction_type(message)
        assert int_type == "confirmation"

    def test_determine_type_notification(self, handler):
        """Test determining notification type."""
        message = {"notify": "Important message"}
        int_type = handler._determine_interaction_type(message)
        assert int_type == "notification"

    def test_determine_type_default_question(self, handler):
        """Test default type is question."""
        message = "Simple string message"
        int_type = handler._determine_interaction_type(message)
        assert int_type == "question"

    def test_determine_type_none_message(self, handler):
        """Test handling None message."""
        int_type = handler._determine_interaction_type(None)
        assert int_type == "input"


class TestSyncModeHandling:
    """Tests for sync mode interaction handling."""

    @pytest.fixture
    def mock_comm_manager(self):
        """Create a mock communication manager."""
        return MockCommunicationManager()

    @pytest.fixture
    def handler(self, mock_comm_manager):
        """Create a handler for testing."""
        return UserNodeHandler(mock_comm_manager)

    @pytest.mark.asyncio
    async def test_sync_mode_returns_response(self, handler, mock_comm_manager):
        """Test sync mode returns user response."""
        branch = MockBranch()
        branch.state.calling_agent = "TestAgent"

        interaction = UserInteraction(
            interaction_id="test-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="What do you want?",
            communication_mode=CommunicationMode.SYNC,
            calling_agent="TestAgent",
            resume_agent="TestAgent",
        )

        context = {"session_id": "session-1"}
        result = await handler._handle_sync_mode(interaction, context, branch)

        assert result.success is True
        assert result.response == "mock_user_response"
        assert result.action_type == "user_response"
        assert result.next_agent == "TestAgent"

    @pytest.mark.asyncio
    async def test_sync_mode_clears_waiting_state(self, handler, mock_comm_manager):
        """Test sync mode clears waiting state."""
        branch = MockBranch()
        branch.state.awaiting_user_response = True
        branch.state.user_wait_start_time = time.time() - 1.0

        interaction = UserInteraction(
            interaction_id="test-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="Test",
            communication_mode=CommunicationMode.SYNC,
            calling_agent="TestAgent",
            resume_agent="TestAgent",
        )

        context = {"session_id": "session-1"}
        await handler._handle_sync_mode(interaction, context, branch)

        assert branch.state.awaiting_user_response is False
        assert branch.state.total_user_wait_time > 0


class TestAsyncModeHandling:
    """Tests for async mode interaction handling."""

    @pytest.fixture
    def mock_comm_manager(self):
        """Create a mock communication manager."""
        return MockCommunicationManager()

    @pytest.fixture
    def handler(self, mock_comm_manager):
        """Create a handler for testing."""
        return UserNodeHandler(mock_comm_manager)

    @pytest.mark.asyncio
    async def test_async_pubsub_returns_pending(self, handler, mock_comm_manager):
        """Test async pub/sub returns pending status."""
        branch = MockBranch()

        interaction = UserInteraction(
            interaction_id="test-async-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="Async test",
            communication_mode=CommunicationMode.ASYNC_PUBSUB,
            calling_agent="TestAgent",
            topic="test_topic",
        )

        context = {"session_id": "session-1"}
        result = await handler._handle_async_pubsub_mode(interaction, context, branch)

        assert result.success is True
        assert result.action_type == "async_pending"
        assert result.parsed_response["mode"] == "async_pubsub"
        assert result.parsed_response["branch_paused"] is True

    @pytest.mark.asyncio
    async def test_async_pubsub_pauses_branch(self, handler, mock_comm_manager):
        """Test async pub/sub pauses branch."""
        branch = MockBranch()

        interaction = UserInteraction(
            interaction_id="test-async-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="Async test",
            communication_mode=CommunicationMode.ASYNC_PUBSUB,
            calling_agent="TestAgent",
        )

        context = {"session_id": "session-1"}
        await handler._handle_async_pubsub_mode(interaction, context, branch)

        assert branch.state.status == BranchStatus.PAUSED

    @pytest.mark.asyncio
    async def test_async_queue_returns_queued(self, handler, mock_comm_manager):
        """Test async queue returns queued status."""
        branch = MockBranch()

        interaction = UserInteraction(
            interaction_id="test-queue-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="Queue test",
            communication_mode=CommunicationMode.ASYNC_QUEUE,
            calling_agent="TestAgent",
            queue_name="work_queue",
        )

        context = {"session_id": "session-1"}
        result = await handler._handle_async_queue_mode(interaction, context, branch)

        assert result.success is True
        assert result.action_type == "async_queued"
        assert result.parsed_response["mode"] == "async_queue"


class TestTaskFormatting:
    """Tests for task formatting."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_format_simple_string_task(self, handler):
        """Test formatting simple string task."""
        result = handler._format_task_for_user("Research AI")

        assert "Task: Research AI" in result
        assert "Please provide your input:" in result

    def test_format_multiline_task(self, handler):
        """Test formatting multiline task."""
        task = "Line 1\nLine 2\nLine 3"
        result = handler._format_task_for_user(task)

        assert "Task Description:" in result
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_format_non_string_task(self, handler):
        """Test formatting non-string task."""
        task = {"action": "research", "topic": "AI"}
        result = handler._format_task_for_user(task)

        assert "Task:" in result


class TestErrorRecoveryHandling:
    """Tests for error recovery handling."""

    @pytest.fixture
    def mock_comm_manager(self):
        """Create a mock communication manager."""
        manager = MockCommunicationManager()
        manager.response = "1"  # Default to retry
        return manager

    @pytest.fixture
    def handler(self, mock_comm_manager):
        """Create a handler for testing."""
        return UserNodeHandler(mock_comm_manager)

    def test_format_fixable_error(self, handler):
        """Test formatting fixable error."""
        error_info = {
            "failed_agent": "BrowserAgent",
            "message": "Connection timeout",
            "category": "timeout_error",
            "suggested_actions": ["Wait and retry", "Check network"],
        }

        formatted = handler._format_fixable_error(error_info)

        assert formatted["type"] == "error_recovery"
        assert "options" in formatted
        assert "Retry" in formatted["options"]
        assert "Skip" in formatted["options"]
        assert "Abort" in formatted["options"]

    def test_format_terminal_error(self, handler):
        """Test formatting terminal error."""
        error_info = {
            "failed_agent": "APIAgent",
            "message": "Invalid API key",
            "category": "api_error",
            "termination_reason": "Authentication cannot be recovered",
        }

        formatted = handler._format_terminal_error(error_info)

        assert formatted["type"] == "terminal_error"
        assert formatted["options"] is None
        assert formatted["acknowledge_only"] is True

    @pytest.mark.asyncio
    async def test_handle_fixable_error_retry(self, handler, mock_comm_manager):
        """Test handling fixable error with retry choice."""
        mock_comm_manager.response = "1"  # Retry

        branch = MockBranch()
        error_info = {
            "failed_agent": "TestAgent",
            "message": "Temporary error",
            "category": "timeout_error",
        }
        context = {"session_id": "session-1", "retry_context": {"agent_name": "TestAgent"}}

        result = await handler._handle_fixable_error(branch, error_info, context)

        assert result.action_type == "retry_failed_step"
        assert result.metadata.get("retry_requested") is True

    @pytest.mark.asyncio
    async def test_handle_fixable_error_skip(self, handler, mock_comm_manager):
        """Test handling fixable error with skip choice."""
        mock_comm_manager.response = "skip"

        branch = MockBranch()
        error_info = {
            "failed_agent": "TestAgent",
            "message": "Error",
        }
        context = {"session_id": "session-1"}

        result = await handler._handle_fixable_error(branch, error_info, context)

        assert result.action_type == "skip_failed_step"

    @pytest.mark.asyncio
    async def test_handle_fixable_error_abort(self, handler, mock_comm_manager):
        """Test handling fixable error with abort choice."""
        mock_comm_manager.response = "abort"

        branch = MockBranch()
        error_info = {
            "failed_agent": "TestAgent",
            "message": "Error",
        }
        context = {"session_id": "session-1"}

        result = await handler._handle_fixable_error(branch, error_info, context)

        assert result.action_type == "abort_execution"
        assert result.success is False


class TestInteractionContextStorage:
    """Tests for interaction context storage."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_store_interaction_context(self, handler):
        """Test storing interaction context."""
        branch = MockBranch()
        interaction = UserInteraction(
            interaction_id="ctx-test-1",
            branch_id=branch.id,
            session_id="session-1",
            incoming_message="Test",
            calling_agent="TestAgent",
            resume_agent="CoordinatorAgent",
            communication_mode=CommunicationMode.SYNC,
        )

        handler._store_interaction_context(branch, interaction)

        assert branch.state.awaiting_user_response is True
        assert branch.state.interaction_id == "ctx-test-1"
        assert branch.state.calling_agent == "TestAgent"
        assert branch.state.resume_agent == "CoordinatorAgent"
        assert branch.state.user_wait_start_time is not None


class TestExecutionSummary:
    """Tests for execution summary extraction."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_get_execution_summary(self, handler):
        """Test getting execution summary."""
        branch = MockBranch()
        branch._execution_trace = [
            MockStep("Agent1", "invoke_agent", True),
            MockStep("Agent2", "call_tool", True),
            MockStep("Agent3", "invoke_agent", False),
        ]

        summary = handler._get_execution_summary(branch)

        assert len(summary) == 3
        assert summary[0]["agent"] == "Agent1"
        assert summary[1]["action"] == "call_tool"
        assert summary[2]["success"] is False

    def test_get_execution_summary_limited(self, handler):
        """Test execution summary is limited to last 5 steps."""
        branch = MockBranch()
        branch._execution_trace = [
            MockStep(f"Agent{i}") for i in range(10)
        ]

        summary = handler._get_execution_summary(branch)

        assert len(summary) == 5
        assert summary[0]["agent"] == "Agent5"  # Starts from 5th agent

    def test_get_execution_summary_empty(self, handler):
        """Test execution summary with empty trace."""
        branch = MockBranch()
        branch._execution_trace = []

        summary = handler._get_execution_summary(branch)

        assert summary == []


class TestUserFirstMode:
    """Tests for user-first mode handling."""

    @pytest.fixture
    def handler(self):
        """Create a handler for testing."""
        return UserNodeHandler(MockCommunicationManager())

    def test_combine_pending_task_with_dict(self, handler):
        """Test combining dict pending task with user response."""
        result = StepResult(
            agent_name="User",
            response="User input",
            action_type="user_response",
            success=True,
        )
        context = {"pending_task": {"action": "research", "topic": "AI"}}

        combined = handler._combine_pending_task_with_user_response(result, context)

        assert "action" in combined.response
        assert "topic" in combined.response
        assert combined.response["user_response"] == "User input"

    def test_combine_pending_task_with_string(self, handler):
        """Test combining string pending task with user response."""
        result = StepResult(
            agent_name="User",
            response="User input",
            action_type="user_response",
            success=True,
        )
        context = {"pending_task": "Research AI trends"}

        combined = handler._combine_pending_task_with_user_response(result, context)

        assert combined.response["initial_task"] == "Research AI trends"
        assert combined.response["user_response"] == "User input"

    def test_combine_removes_pending_task(self, handler):
        """Test that pending_task is removed after combining."""
        result = StepResult(
            agent_name="User",
            response="User input",
            action_type="user_response",
            success=True,
        )
        context = {"pending_task": "Task"}

        handler._combine_pending_task_with_user_response(result, context)

        assert "pending_task" not in context


class TestHandleUserNode:
    """Tests for the main handle_user_node method."""

    @pytest.fixture
    def mock_comm_manager(self):
        """Create a mock communication manager."""
        return MockCommunicationManager()

    @pytest.fixture
    def handler(self, mock_comm_manager):
        """Create a handler for testing."""
        return UserNodeHandler(mock_comm_manager)

    @pytest.mark.asyncio
    async def test_handle_user_node_basic(self, handler, mock_comm_manager):
        """Test basic user node handling."""
        branch = MockBranch()
        branch.state.calling_agent = "TestAgent"
        context = {"session_id": "session-1"}

        result = await handler.handle_user_node(
            branch,
            "What would you like to do?",
            context
        )

        assert result.success is True
        assert result.agent_name == "User"
        assert len(mock_comm_manager.handled_interactions) == 1

    @pytest.mark.asyncio
    async def test_handle_user_node_with_error_recovery(self, handler, mock_comm_manager):
        """Test user node with error recovery message."""
        mock_comm_manager.response = "1"  # Retry

        branch = MockBranch()
        context = {
            "session_id": "session-1",
            "retry_context": {"agent_name": "FailedAgent"},
        }

        error_message = {
            "error_recovery": True,
            "error_details": {
                "failed_agent": "FailedAgent",
                "message": "Connection failed",
            }
        }

        result = await handler.handle_user_node(branch, error_message, context)

        assert result.action_type == "retry_failed_step"

    @pytest.mark.asyncio
    async def test_handle_user_node_with_terminal_error(self, handler, mock_comm_manager):
        """Test user node with terminal error message."""
        branch = MockBranch()
        context = {"session_id": "session-1"}

        error_message = {
            "error_type": "terminal",
            "error_details": {
                "failed_agent": "FailedAgent",
                "message": "Invalid configuration",
            }
        }

        result = await handler.handle_user_node(branch, error_message, context)

        assert result.action_type == "terminal_error"
        assert result.success is False

    @pytest.mark.asyncio
    async def test_handle_user_node_system_interaction(self, handler, mock_comm_manager):
        """Test user node with System as calling agent."""
        branch = MockBranch()
        branch._execution_trace = []  # No previous agent
        context = {"session_id": "session-1"}

        result = await handler.handle_user_node(
            branch,
            "Task description here",
            context
        )

        # Should format the task for display
        interaction = mock_comm_manager.handled_interactions[0]
        assert "Task:" in str(interaction.incoming_message) or "Task Description:" in str(interaction.incoming_message)

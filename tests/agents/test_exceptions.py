"""
Tests for the marsys.agents.exceptions module.

This module tests:
- AgentFrameworkError base class
- All specialized exception classes
- Exception attributes and formatting
"""

import pytest

from marsys.agents.exceptions import (
    AgentFrameworkError,
    AgentConfigurationError,
    AgentError,
    AgentPermissionError,
    AgentLimitError,
    MessageError,
    MessageFormatError,
    MessageContentError,
    ActionValidationError,
    ToolCallError,
    ResourceError,
    PoolExhaustedError,
    TimeoutError as AgentTimeoutError,
    ModelError,
    ModelResponseError,
    ModelAPIError,
    BrowserError,
    ToolExecutionError,
)


# =============================================================================
# AgentFrameworkError Tests
# =============================================================================

class TestAgentFrameworkError:
    """Tests for the base AgentFrameworkError class."""

    def test_basic_creation(self):
        """Test creating basic exception."""
        error = AgentFrameworkError("Something went wrong")

        assert "Something went wrong" in str(error)

    def test_with_error_code(self):
        """Test exception with error code."""
        error = AgentFrameworkError(
            "Test error",
            error_code="ERR001"
        )

        assert error.error_code == "ERR001"

    def test_with_agent_name(self):
        """Test exception with agent name."""
        error = AgentFrameworkError(
            "Test error",
            agent_name="MyAgent"
        )

        assert error.agent_name == "MyAgent"

    def test_with_all_attributes(self):
        """Test exception with all optional attributes."""
        error = AgentFrameworkError(
            "Test error",
            error_code="ERR001",
            agent_name="MyAgent",
            task_id="task_123",
            context={"key": "value"},
            user_message="User-friendly message",
            suggestion="Try this fix"
        )

        assert error.error_code == "ERR001"
        assert error.agent_name == "MyAgent"
        assert error.task_id == "task_123"
        assert error.context == {"key": "value"}
        assert error.user_message == "User-friendly message"
        assert error.suggestion == "Try this fix"

    def test_to_dict(self):
        """Test converting exception to dictionary."""
        error = AgentFrameworkError(
            "Test error",
            error_code="TEST001",
            agent_name="TestAgent"
        )

        result = error.to_dict()

        assert result["error_type"] == "AgentFrameworkError"
        assert result["error_code"] == "TEST001"
        assert result["agent_name"] == "TestAgent"

    def test_message_property(self):
        """Test message property returns developer_message."""
        error = AgentFrameworkError("Developer message")

        assert error.message == "Developer message"


# =============================================================================
# AgentConfigurationError Tests
# =============================================================================

class TestAgentConfigurationError:
    """Tests for AgentConfigurationError."""

    def test_basic_creation(self):
        """Test creating basic configuration error."""
        error = AgentConfigurationError("Invalid configuration")

        assert "Invalid configuration" in str(error)
        assert isinstance(error, AgentError)
        assert isinstance(error, AgentFrameworkError)

    def test_with_config_field(self):
        """Test exception with config_field."""
        error = AgentConfigurationError(
            "Invalid value",
            config_field="max_tokens"
        )

        assert error.config_field == "max_tokens"

    def test_with_config_value(self):
        """Test exception with config_value."""
        error = AgentConfigurationError(
            "Invalid value",
            config_field="max_tokens",
            config_value=-100
        )

        assert error.config_value == -100

    def test_is_catchable_as_base_error(self):
        """Test that error can be caught as base error."""
        error = AgentConfigurationError("Test")

        with pytest.raises(AgentFrameworkError):
            raise error


# =============================================================================
# AgentPermissionError Tests
# =============================================================================

class TestAgentPermissionError:
    """Tests for AgentPermissionError."""

    def test_basic_creation(self):
        """Test creating basic permission error."""
        error = AgentPermissionError("Permission denied")

        assert "Permission denied" in str(error)
        assert isinstance(error, AgentError)

    def test_with_target_agent(self):
        """Test exception with target_agent."""
        error = AgentPermissionError(
            "Cannot invoke",
            target_agent="RestrictedAgent"
        )

        assert error.target_agent == "RestrictedAgent"

    def test_with_allowed_agents(self):
        """Test exception with allowed_agents."""
        error = AgentPermissionError(
            "Not allowed",
            target_agent="Agent1",
            allowed_agents=["Agent2", "Agent3"]
        )

        assert error.allowed_agents == ["Agent2", "Agent3"]


# =============================================================================
# Message Error Tests
# =============================================================================

class TestMessageErrors:
    """Tests for Message-related errors."""

    def test_message_error_basic(self):
        """Test creating basic message error."""
        error = MessageError("Invalid message")

        assert "Invalid message" in str(error)
        assert isinstance(error, AgentFrameworkError)

    def test_message_format_error(self):
        """Test MessageFormatError."""
        error = MessageFormatError(
            "Invalid JSON",
            invalid_content='{"incomplete":',
            expected_format="JSON"
        )

        assert error.invalid_content == '{"incomplete":'
        assert error.expected_format == "JSON"

    def test_message_content_error(self):
        """Test MessageContentError."""
        error = MessageContentError(
            "Wrong type",
            content_type="string",
            expected_type="dict"
        )

        assert error.content_type == "string"
        assert error.expected_type == "dict"

    def test_action_validation_error(self):
        """Test ActionValidationError."""
        error = ActionValidationError(
            "Invalid action",
            action="unknown_action",
            valid_actions=["call_tool", "invoke_agent", "final_response"]
        )

        assert error.action == "unknown_action"
        assert error.valid_actions == ["call_tool", "invoke_agent", "final_response"]


# =============================================================================
# Tool Error Tests
# =============================================================================

class TestToolErrors:
    """Tests for tool-related errors."""

    def test_tool_call_error(self):
        """Test ToolCallError."""
        error = ToolCallError(
            "Tool not found",
            tool_name="nonexistent_tool",
            available_tools=["search", "fetch"]
        )

        assert error.tool_name == "nonexistent_tool"
        assert error.available_tools == ["search", "fetch"]

    def test_tool_execution_error(self):
        """Test ToolExecutionError."""
        error = ToolExecutionError(
            "Execution failed",
            tool_name="search",
            tool_args={"query": "test"},
            execution_error="Connection timeout"
        )

        assert error.tool_name == "search"
        assert error.tool_args == {"query": "test"}
        assert error.execution_error == "Connection timeout"


# =============================================================================
# Resource Error Tests
# =============================================================================

class TestResourceError:
    """Tests for ResourceError."""

    def test_basic_creation(self):
        """Test creating basic resource error."""
        error = ResourceError("Resource unavailable")

        assert "Resource unavailable" in str(error)
        assert isinstance(error, AgentFrameworkError)


# =============================================================================
# PoolExhaustedError Tests
# =============================================================================

class TestPoolExhaustedError:
    """Tests for PoolExhaustedError."""

    def test_basic_creation(self):
        """Test creating basic pool exhausted error."""
        error = PoolExhaustedError("No instances available")

        assert "No instances available" in str(error)
        assert isinstance(error, ResourceError)
        assert isinstance(error, AgentFrameworkError)

    def test_with_pool_name(self):
        """Test exception with pool_name."""
        error = PoolExhaustedError(
            "Pool exhausted",
            pool_name="BrowserPool"
        )

        assert error.pool_name == "BrowserPool"

    def test_with_instance_counts(self):
        """Test exception with instance counts."""
        error = PoolExhaustedError(
            "All busy",
            pool_name="BrowserPool",
            total_instances=3,
            allocated_instances=3
        )

        assert error.total_instances == 3
        assert error.allocated_instances == 3

    def test_with_requested_count(self):
        """Test exception with requested_count."""
        error = PoolExhaustedError(
            "Not enough",
            pool_name="TestPool",
            total_instances=2,
            allocated_instances=2,
            requested_count=1
        )

        assert error.requested_count == 1


# =============================================================================
# TimeoutError Tests
# =============================================================================

class TestTimeoutError:
    """Tests for TimeoutError (AgentTimeoutError)."""

    def test_basic_creation(self):
        """Test creating basic timeout error."""
        error = AgentTimeoutError("Operation timed out")

        assert "timed out" in str(error)
        assert isinstance(error, ResourceError)
        assert isinstance(error, AgentFrameworkError)

    def test_with_timeout_seconds(self):
        """Test exception with timeout_seconds."""
        error = AgentTimeoutError(
            "Timeout",
            timeout_seconds=30.0
        )

        assert error.timeout_seconds == 30.0

    def test_with_operation(self):
        """Test exception with operation."""
        error = AgentTimeoutError(
            "Operation timeout",
            operation="tool_execution",
            timeout_seconds=60.0
        )

        assert error.operation == "tool_execution"


# =============================================================================
# Model Error Tests
# =============================================================================

class TestModelErrors:
    """Tests for Model-related errors."""

    def test_model_error_basic(self):
        """Test ModelError base class."""
        error = ModelError("Model error")

        assert "Model error" in str(error)
        assert isinstance(error, AgentFrameworkError)

    def test_model_response_error(self):
        """Test ModelResponseError."""
        error = ModelResponseError(
            "Invalid response format",
            response_type="json",
            expected_fields=["content", "role"],
            missing_fields=["content"]
        )

        assert error.response_type == "json"
        assert error.expected_fields == ["content", "role"]
        assert error.missing_fields == ["content"]

    def test_model_api_error(self):
        """Test ModelAPIError."""
        error = ModelAPIError(
            "Rate limit exceeded",
            provider="openai",
            status_code=429,
            classification="rate_limit",
            is_retryable=True,
            retry_after=60
        )

        assert error.provider == "openai"
        assert error.status_code == 429
        assert error.is_retryable is True
        assert error.retry_after == 60


# =============================================================================
# Browser Error Tests
# =============================================================================

class TestBrowserError:
    """Tests for BrowserError."""

    def test_basic_creation(self):
        """Test creating basic browser error."""
        error = BrowserError("Browser error")

        assert "Browser error" in str(error)
        assert isinstance(error, AgentFrameworkError)


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================

class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_all_inherit_from_base(self):
        """Test all custom exceptions inherit from AgentFrameworkError."""
        exceptions = [
            AgentConfigurationError("test"),
            AgentPermissionError("test"),
            AgentLimitError("test"),
            MessageError("test"),
            MessageFormatError("test"),
            ToolCallError("test"),
            ToolExecutionError("test"),
            ResourceError("test"),
            PoolExhaustedError("test"),
            AgentTimeoutError("test"),
            ModelError("test"),
            BrowserError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AgentFrameworkError)
            assert isinstance(exc, Exception)

    def test_pool_exhausted_inherits_from_resource(self):
        """Test PoolExhaustedError inherits from ResourceError."""
        error = PoolExhaustedError("test")

        assert isinstance(error, ResourceError)

    def test_timeout_inherits_from_resource(self):
        """Test TimeoutError inherits from ResourceError."""
        error = AgentTimeoutError("test")

        assert isinstance(error, ResourceError)

    def test_can_catch_by_base_type(self):
        """Test exceptions can be caught by base type."""
        specific_error = PoolExhaustedError("test")

        # Should be catchable as ResourceError
        with pytest.raises(ResourceError):
            raise specific_error

        # Should also be catchable as AgentFrameworkError
        with pytest.raises(AgentFrameworkError):
            raise PoolExhaustedError("test")

    def test_agent_errors_inherit_from_agent_error(self):
        """Test agent-specific errors inherit from AgentError."""
        errors = [
            AgentConfigurationError("test"),
            AgentPermissionError("test"),
            AgentLimitError("test"),
        ]

        for err in errors:
            assert isinstance(err, AgentError)

    def test_message_errors_inherit_from_message_error(self):
        """Test message-related errors inherit from MessageError."""
        errors = [
            MessageFormatError("test"),
            MessageContentError("test"),
            ActionValidationError("test"),
            ToolCallError("test"),
        ]

        for err in errors:
            assert isinstance(err, MessageError)

    def test_model_errors_inherit_from_model_error(self):
        """Test model-related errors inherit from ModelError."""
        errors = [
            ModelResponseError("test"),
            ModelAPIError("test"),
        ]

        for err in errors:
            assert isinstance(err, ModelError)

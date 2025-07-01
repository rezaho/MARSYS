"""
Multi-Agent Framework Exception Hierarchy

This module defines a comprehensive exception hierarchy for the multi-agent framework,
providing specific error types for different categories of issues with rich context
and standardized error handling.

The hierarchy is designed to:
1. Provide granular error handling for different error categories
2. Include rich context information (agent names, task IDs, timestamps)
3. Enable programmatic error recovery and handling
4. Maintain consistent error message formats
5. Support future extensibility
"""

import time
from typing import Any, Dict, List, Optional, Union


class AgentFrameworkError(Exception):
    """
    Base exception class for all multi-agent framework errors.
    
    Provides common error context and standardized error information
    that all framework-specific exceptions inherit.
    
    Attributes:
        error_code: Unique error code for programmatic handling
        agent_name: Name of the agent where error occurred (if applicable)
        task_id: Task ID where error occurred (if applicable)
        timestamp: When the error occurred
        context: Additional context information
        user_message: User-friendly error message
        developer_message: Detailed technical error message
        suggestion: Suggested fix or next steps (if applicable)
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "AGENT_FRAMEWORK_ERROR",
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        """
        Initialize framework error with rich context.
        
        Args:
            message: Technical error message for developers
            error_code: Unique error code for programmatic handling
            agent_name: Name of agent where error occurred
            task_id: Task ID where error occurred
            context: Additional context information
            user_message: User-friendly error message
            suggestion: Suggested fix or next steps
        """
        super().__init__(message)
        self.error_code = error_code
        self.agent_name = agent_name
        self.task_id = task_id
        self.timestamp = time.time()
        self.context = context or {}
        self.user_message = user_message or message
        self.developer_message = message
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.developer_message,
            "user_message": self.user_message,
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "context": self.context,
            "suggestion": self.suggestion,
        }
    
    def __str__(self) -> str:
        """String representation with context."""
        parts = [f"[{self.error_code}]"]
        if self.agent_name:
            parts.append(f"Agent:{self.agent_name}")
        if self.task_id:
            parts.append(f"Task:{self.task_id[:8]}...")
        parts.append(self.developer_message)
        return " ".join(parts)


# =============================================================================
# MESSAGE HANDLING ERRORS
# =============================================================================

class MessageError(AgentFrameworkError):
    """Base class for message handling and validation errors."""
    
    def __init__(self, message: str, **kwargs):
        # Extract error_code to avoid duplicate parameter
        error_code = kwargs.pop("error_code", "MESSAGE_ERROR")
        super().__init__(
            message,
            error_code=error_code,
            **kwargs
        )


class MessageFormatError(MessageError):
    """
    Raised when message format is invalid (JSON parsing, structure issues).
    
    Examples:
    - Invalid JSON syntax
    - Multiple concatenated JSON objects
    - Wrong data types for message fields
    """
    
    def __init__(
        self,
        message: str,
        invalid_content: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        self.invalid_content = invalid_content
        self.expected_format = expected_format
        
        context = kwargs.get("context", {})
        if invalid_content:
            context["invalid_content"] = invalid_content[:200] + "..." if len(invalid_content) > 200 else invalid_content
        if expected_format:
            context["expected_format"] = expected_format
        
        super().__init__(
            message,
            error_code="MESSAGE_FORMAT_ERROR",
            context=context,
            user_message="The message format is invalid. Please check the JSON structure.",
            suggestion="Ensure your response is valid JSON with proper structure and no concatenated objects.",
            **kwargs
        )


class MessageContentError(MessageError):
    """
    Raised when message content is invalid (empty, wrong type, malformed).
    
    Examples:
    - Empty content when content required
    - Wrong content type (expected dict, got string)
    - Missing required content fields
    """
    
    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs
    ):
        self.content_type = content_type
        self.expected_type = expected_type
        
        context = kwargs.get("context", {})
        if content_type:
            context["actual_type"] = content_type
        if expected_type:
            context["expected_type"] = expected_type
        
        super().__init__(
            message,
            error_code="MESSAGE_CONTENT_ERROR",
            context=context,
            user_message="The message content is invalid or empty.",
            suggestion="Provide valid content with the expected type and structure.",
            **kwargs
        )


class ActionValidationError(MessageError):
    """
    Raised when action validation fails (invalid next_action, action_input).
    
    Examples:
    - Invalid next_action value
    - Missing action_input
    - Wrong action_input structure for specific actions
    """
    
    def __init__(
        self,
        message: str,
        action: Optional[str] = None,
        valid_actions: Optional[List[str]] = None,
        action_input: Optional[Any] = None,
        **kwargs
    ):
        self.action = action
        self.valid_actions = valid_actions
        self.action_input = action_input
        
        context = kwargs.get("context", {})
        if action:
            context["provided_action"] = action
        if valid_actions:
            context["valid_actions"] = valid_actions
        if action_input is not None:
            context["action_input_type"] = type(action_input).__name__
        
        super().__init__(
            message,
            error_code="ACTION_VALIDATION_ERROR",
            context=context,
            user_message="The action or action input is invalid.",
            suggestion="Use a valid next_action and provide properly structured action_input.",
            **kwargs
        )


class ToolCallError(MessageError):
    """
    Raised when tool call format or execution is invalid.
    
    Examples:
    - Invalid tool call structure
    - Missing required tool call fields
    - Tool not found or not callable
    """
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
        tool_call_index: Optional[int] = None,
        **kwargs
    ):
        self.tool_name = tool_name
        self.available_tools = available_tools
        self.tool_call_index = tool_call_index
        
        context = kwargs.get("context", {})
        if tool_name:
            context["tool_name"] = tool_name
        if available_tools:
            context["available_tools"] = available_tools
        if tool_call_index is not None:
            context["tool_call_index"] = tool_call_index
        
        super().__init__(
            message,
            error_code="TOOL_CALL_ERROR",
            context=context,
            user_message="The tool call is invalid or the tool is not available.",
            suggestion="Check tool name spelling and ensure proper tool call structure.",
            **kwargs
        )


class SchemaValidationError(MessageError):
    """
    Raised when input/output schema validation fails.
    
    Examples:
    - Input doesn't match agent's input schema
    - Output doesn't match agent's output schema
    - Schema validation library errors
    """
    
    def __init__(
        self,
        message: str,
        schema_type: Optional[str] = None,  # "input" or "output"
        validation_path: Optional[str] = None,
        provided_data: Optional[Any] = None,
        **kwargs
    ):
        self.schema_type = schema_type
        self.validation_path = validation_path
        self.provided_data = provided_data
        
        context = kwargs.get("context", {})
        if schema_type:
            context["schema_type"] = schema_type
        if validation_path:
            context["validation_path"] = validation_path
        if provided_data is not None:
            context["data_type"] = type(provided_data).__name__
            context["data_preview"] = str(provided_data)[:100] + "..." if len(str(provided_data)) > 100 else str(provided_data)
        
        super().__init__(
            message,
            error_code="SCHEMA_VALIDATION_ERROR",
            context=context,
            user_message="The data doesn't match the required schema format.",
            suggestion="Check the expected schema format and adjust your data accordingly.",
            **kwargs
        )


# =============================================================================
# AGENT IMPLEMENTATION & LIFECYCLE ERRORS
# =============================================================================

class AgentError(AgentFrameworkError):
    """Base class for agent implementation and lifecycle errors."""
    
    def __init__(self, message: str, **kwargs):
        # Extract error_code to avoid duplicate parameter
        error_code = kwargs.pop("error_code", "AGENT_ERROR")
        super().__init__(
            message,
            error_code=error_code,
            **kwargs
        )


class AgentImplementationError(AgentError):
    """
    Raised when agent implementation is incorrect or incomplete.
    
    Examples:
    - Missing abstract method implementations
    - Wrong return types from _run methods
    - Invalid Message object creation
    """
    
    def __init__(
        self,
        message: str,
        method_name: Optional[str] = None,
        expected_return_type: Optional[str] = None,
        actual_return_type: Optional[str] = None,
        **kwargs
    ):
        self.method_name = method_name
        self.expected_return_type = expected_return_type
        self.actual_return_type = actual_return_type
        
        context = kwargs.get("context", {})
        if method_name:
            context["method_name"] = method_name
        if expected_return_type:
            context["expected_return_type"] = expected_return_type
        if actual_return_type:
            context["actual_return_type"] = actual_return_type
        
        super().__init__(
            message,
            error_code="AGENT_IMPLEMENTATION_ERROR",
            context=context,
            user_message="There's an implementation issue with the agent.",
            suggestion="Check agent implementation for missing methods or wrong return types.",
            **kwargs
        )


class AgentConfigurationError(AgentError):
    """
    Raised when agent configuration is invalid or incomplete.
    
    Examples:
    - Invalid model configuration
    - Missing required configuration fields
    - Incompatible configuration combinations
    """
    
    def __init__(
        self,
        message: str,
        config_field: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        self.config_field = config_field
        self.config_value = config_value
        
        context = kwargs.get("context", {})
        if config_field:
            context["config_field"] = config_field
        if config_value is not None:
            context["config_value"] = str(config_value)
        
        super().__init__(
            message,
            error_code="AGENT_CONFIGURATION_ERROR",
            context=context,
            user_message="The agent configuration is invalid.",
            suggestion="Check agent configuration for missing or invalid fields.",
            **kwargs
        )


class AgentPermissionError(AgentError):
    """
    Raised when agent permission/access is denied.
    
    Examples:
    - Agent not in allowed_peers list
    - Attempting to invoke non-existent agent
    - Access control violations
    """
    
    def __init__(
        self,
        message: str,
        target_agent: Optional[str] = None,
        allowed_agents: Optional[List[str]] = None,
        **kwargs
    ):
        self.target_agent = target_agent
        self.allowed_agents = allowed_agents
        
        context = kwargs.get("context", {})
        if target_agent:
            context["target_agent"] = target_agent
        if allowed_agents:
            context["allowed_agents"] = allowed_agents
        
        super().__init__(
            message,
            error_code="AGENT_PERMISSION_ERROR",
            context=context,
            user_message="Permission denied for agent interaction.",
            suggestion="Check allowed_peers configuration or verify agent exists.",
            **kwargs
        )


class AgentLimitError(AgentError):
    """
    Raised when agent execution limits are exceeded.
    
    Examples:
    - Maximum depth limit exceeded
    - Maximum interaction count exceeded
    - Maximum steps in auto_run exceeded
    """
    
    def __init__(
        self,
        message: str,
        limit_type: Optional[str] = None,  # "depth", "interactions", "steps"
        current_value: Optional[int] = None,
        limit_value: Optional[int] = None,
        **kwargs
    ):
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value
        
        context = kwargs.get("context", {})
        if limit_type:
            context["limit_type"] = limit_type
        if current_value is not None:
            context["current_value"] = current_value
        if limit_value is not None:
            context["limit_value"] = limit_value
        
        super().__init__(
            message,
            error_code="AGENT_LIMIT_ERROR",
            context=context,
            user_message=f"Agent execution limit exceeded ({limit_type}).",
            suggestion="Consider increasing limits or simplifying the task.",
            **kwargs
        )


# =============================================================================
# MODEL & API ERRORS
# =============================================================================

class ModelError(AgentFrameworkError):
    """Base class for model and API response errors."""
    
    def __init__(self, message: str, **kwargs):
        # Extract error_code to avoid duplicate parameter
        error_code = kwargs.pop("error_code", "MODEL_ERROR")
        super().__init__(
            message,
            error_code=error_code,
            **kwargs
        )


class ModelResponseError(ModelError):
    """
    Raised when model response format is invalid or incomplete.
    
    Examples:
    - Invalid response format from model
    - Missing required response fields
    - Response validation failures
    """
    
    def __init__(
        self,
        message: str,
        response_type: Optional[str] = None,
        expected_fields: Optional[List[str]] = None,
        missing_fields: Optional[List[str]] = None,
        response_content: Optional[Any] = None,
        **kwargs
    ):
        self.response_type = response_type
        self.expected_fields = expected_fields
        self.missing_fields = missing_fields
        self.response_content = response_content
        
        context = kwargs.get("context", {})
        if response_type:
            context["response_type"] = response_type
        if expected_fields:
            context["expected_fields"] = expected_fields
        if missing_fields:
            context["missing_fields"] = missing_fields
        if response_content is not None:
            context["response_content"] = str(response_content)[:500]  # Limit size for logging
        
        super().__init__(
            message,
            error_code="MODEL_RESPONSE_ERROR",
            context=context,
            user_message="The model response format is invalid.",
            suggestion="Check model configuration and response validation logic.",
            **kwargs
        )


class ModelTokenLimitError(ModelError):
    """
    Raised when model token limits are exceeded.
    
    Examples:
    - Input exceeds model's context limit
    - Output truncated due to max_tokens limit
    - Token budget exceeded
    """
    
    def __init__(
        self,
        message: str,
        token_count: Optional[int] = None,
        token_limit: Optional[int] = None,
        limit_type: Optional[str] = None,  # "input", "output", "context"
        **kwargs
    ):
        self.token_count = token_count
        self.token_limit = token_limit
        self.limit_type = limit_type
        
        context = kwargs.get("context", {})
        if token_count is not None:
            context["token_count"] = token_count
        if token_limit is not None:
            context["token_limit"] = token_limit
        if limit_type:
            context["limit_type"] = limit_type
        
        super().__init__(
            message,
            error_code="MODEL_TOKEN_LIMIT_ERROR",
            context=context,
            user_message="Token limit exceeded for model operation.",
            suggestion="Reduce input size or increase max_tokens limit.",
            **kwargs
        )


class ModelAPIError(ModelError):
    """
    Raised when API connection or authentication issues occur.
    
    Examples:
    - API authentication failures
    - Network connection issues
    - API rate limiting
    - Invalid API endpoints
    """
    
    def __init__(
        self,
        message: str,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        api_error_code: Optional[str] = None,
        **kwargs
    ):
        self.api_endpoint = api_endpoint
        self.status_code = status_code
        self.api_error_code = api_error_code
        
        context = kwargs.get("context", {})
        if api_endpoint:
            context["api_endpoint"] = api_endpoint
        if status_code is not None:
            context["status_code"] = status_code
        if api_error_code:
            context["api_error_code"] = api_error_code
        
        super().__init__(
            message,
            error_code="MODEL_API_ERROR",
            context=context,
            user_message="API connection or authentication failed.",
            suggestion="Check API credentials, network connection, and endpoint configuration.",
            **kwargs
        )


# =============================================================================
# BROWSER & TOOL ERRORS
# =============================================================================

class BrowserError(AgentFrameworkError):
    """Base class for browser-related errors."""
    
    def __init__(self, message: str, **kwargs):
        # Extract error_code to avoid duplicate parameter
        error_code = kwargs.pop("error_code", "BROWSER_ERROR")
        super().__init__(
            message,
            error_code=error_code,
            **kwargs
        )


class BrowserNotInitializedError(BrowserError):
    """
    Raised when browser operations are attempted before initialization.
    """
    
    def __init__(self, operation: Optional[str] = None, **kwargs):
        self.operation = operation
        
        context = kwargs.get("context", {})
        if operation:
            context["attempted_operation"] = operation
        
        message = f"Browser not initialized for operation: {operation}" if operation else "Browser not initialized"
        
        super().__init__(
            message,
            error_code="BROWSER_NOT_INITIALIZED_ERROR",
            context=context,
            user_message="Browser needs to be initialized before use.",
            suggestion="Call browser initialization methods before attempting browser operations.",
            **kwargs
        )


class BrowserConnectionError(BrowserError):
    """
    Raised when browser connection or setup fails.
    
    Examples:
    - Browser executable not found
    - Missing browser dependencies
    - Browser launch failures
    """
    
    def __init__(
        self,
        message: str,
        browser_type: Optional[str] = None,
        install_command: Optional[str] = None,
        **kwargs
    ):
        self.browser_type = browser_type
        self.install_command = install_command
        
        context = kwargs.get("context", {})
        if browser_type:
            context["browser_type"] = browser_type
        if install_command:
            context["install_command"] = install_command
        
        super().__init__(
            message,
            error_code="BROWSER_CONNECTION_ERROR",
            context=context,
            user_message="Failed to connect or initialize browser.",
            suggestion=f"Try running: {install_command}" if install_command else "Check browser installation and dependencies.",
            **kwargs
        )


class ToolExecutionError(AgentFrameworkError):
    """
    Raised when tool execution fails.
    
    Examples:
    - Tool function execution errors
    - Invalid tool arguments
    - Tool timeout or resource issues
    """
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        execution_error: Optional[str] = None,
        **kwargs
    ):
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.execution_error = execution_error
        
        context = kwargs.get("context", {})
        if tool_name:
            context["tool_name"] = tool_name
        if tool_args:
            context["tool_args"] = str(tool_args)
        if execution_error:
            context["execution_error"] = execution_error
        
        super().__init__(
            message,
            error_code="TOOL_EXECUTION_ERROR",
            context=context,
            user_message="Tool execution failed.",
            suggestion="Check tool arguments and ensure tool is available and functional.",
            **kwargs
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_from_exception(
    original_exception: Exception,
    error_class: type = AgentFrameworkError,
    **kwargs
) -> AgentFrameworkError:
    """
    Convert a generic exception to a framework-specific error.
    
    Args:
        original_exception: The original exception to convert
        error_class: The framework error class to use
        **kwargs: Additional context for the error
        
    Returns:
        Framework-specific error with original exception information
    """
    context = kwargs.get("context", {})
    context["original_exception_type"] = type(original_exception).__name__
    context["original_exception_message"] = str(original_exception)
    
    message = kwargs.get("message", f"Converted from {type(original_exception).__name__}: {str(original_exception)}")
    
    return error_class(
        message=message,
        context=context,
        **kwargs
    )


def get_error_summary(error: AgentFrameworkError) -> Dict[str, Any]:
    """
    Get a summary of error information for logging/reporting.
    
    Args:
        error: Framework error to summarize
        
    Returns:
        Dictionary with error summary information
    """
    return {
        "error_code": error.error_code,
        "error_type": type(error).__name__,
        "agent_name": error.agent_name,
        "task_id": error.task_id,
        "user_message": error.user_message,
        "suggestion": error.suggestion,
        "timestamp": error.timestamp,
    } 
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
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorAction(Enum):
    """What action can be taken for this error."""

    # User can potentially fix and retry
    USER_FIXABLE = "user_fixable"

    # Cannot be fixed on-the-fly, must terminate
    TERMINAL = "terminal"

    # System should retry automatically (no user interaction)
    AUTO_RETRY = "auto_retry"


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
    Enhanced API error with provider-specific error classification.

    Supports detection of:
    - Insufficient credits/quota
    - Rate limiting
    - Authentication failures
    - Network issues
    - Service availability
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        api_error_code: Optional[str] = None,
        api_error_type: Optional[str] = None,
        classification: Optional[str] = None,  # APIErrorClassification value
        is_retryable: bool = False,
        retry_after: Optional[int] = None,
        suggested_action: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.provider = provider
        self.api_endpoint = api_endpoint
        self.status_code = status_code
        self.api_error_code = api_error_code
        self.api_error_type = api_error_type
        self.classification = classification or APIErrorClassification.UNKNOWN.value
        self.is_retryable = is_retryable
        self.retry_after = retry_after
        self.raw_response = raw_response

        context = kwargs.get("context", {})
        context.update({
            "provider": provider,
            "api_endpoint": api_endpoint,
            "status_code": status_code,
            "api_error_code": api_error_code,
            "api_error_type": api_error_type,
            "classification": self.classification,
            "is_retryable": is_retryable,
            "retry_after": retry_after
        })

        # Auto-generate suggested action if not provided
        if not suggested_action:
            if self.classification == APIErrorClassification.INSUFFICIENT_CREDITS.value:
                provider_actions = {
                    "openai": "Add credits at https://platform.openai.com/billing",
                    "anthropic": "Add credits at https://console.anthropic.com/billing",
                    "google": "Enable billing or upgrade from free tier at https://console.cloud.google.com",
                    "openrouter": "Add credits at https://openrouter.ai/credits",
                    "xai": "Check credits at https://console.x.ai/billing"
                }
                suggested_action = provider_actions.get(provider, f"Add credits to your {provider} account")
            elif self.classification == APIErrorClassification.RATE_LIMIT.value:
                suggested_action = f"Wait {retry_after} seconds before retrying" if retry_after else "Wait before retrying or upgrade your plan"
            elif self.classification == APIErrorClassification.AUTHENTICATION_FAILED.value:
                suggested_action = f"Check your {provider} API key configuration"
            elif self.classification == APIErrorClassification.SERVICE_UNAVAILABLE.value:
                suggested_action = "Service temporarily unavailable. Please try again later."

        super().__init__(
            message,
            error_code=f"MODEL_API_{self.classification.upper()}_ERROR",
            context=context,
            user_message=message,
            suggestion=suggested_action,
            **kwargs
        )

    def is_critical(self) -> bool:
        """Check if this is a critical error that cannot be retried."""
        return self.classification in [
            APIErrorClassification.INSUFFICIENT_CREDITS.value,
            APIErrorClassification.AUTHENTICATION_FAILED.value,
            APIErrorClassification.PERMISSION_DENIED.value,
            APIErrorClassification.INVALID_MODEL.value
        ]

    @property
    def message(self) -> str:
        """Provide message property for backward compatibility."""
        return self.developer_message

    @property
    def suggested_action(self) -> str:
        """Provide suggested_action property for backward compatibility."""
        return self.suggestion if hasattr(self, 'suggestion') else None

    @classmethod
    def from_provider_response(
        cls,
        provider: str,
        response: Optional[Any] = None,
        exception: Optional[Exception] = None
    ) -> 'ModelAPIError':
        """
        Factory method to create ModelAPIError from provider response.
        Analyzes response and creates appropriately classified error.
        """
        # Default values
        # Try to extract error message from exception
        if exception:
            # For aiohttp.ClientResponseError
            if hasattr(exception, 'message') and exception.message:
                message = exception.message
            # For general exceptions with args
            elif hasattr(exception, 'args') and exception.args:
                message = str(exception.args[0]) if exception.args else str(exception)
            else:
                message = str(exception)
        else:
            message = "API Error"

        # Try to get status code from response first, then from exception
        status_code = None
        if response and hasattr(response, 'status_code'):
            status_code = response.status_code
        elif exception:
            # For aiohttp.ClientResponseError
            if hasattr(exception, 'status'):
                status_code = exception.status
            # For requests.HTTPError
            elif hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
                status_code = exception.response.status_code

        classification = APIErrorClassification.UNKNOWN.value
        is_retryable = False
        retry_after = None
        api_error_code = None
        api_error_type = None
        raw_response = None

        # Try to get raw response data
        if response:
            try:
                raw_response = response.json() if hasattr(response, 'json') else None
            except:
                raw_response = None

        # Provider-specific error parsing based on status code
        # This needs to work even when response is None (just using status_code)
        if status_code:
            if provider == "openai":
                error_data = raw_response.get("error", {}) if raw_response else {}
                if error_data:
                    message = error_data.get("message", message)
                    api_error_code = error_data.get("code")
                    api_error_type = error_data.get("type")

                if status_code == 429:
                    if api_error_type == "insufficient_quota":
                        classification = APIErrorClassification.INSUFFICIENT_CREDITS.value
                    else:
                        classification = APIErrorClassification.RATE_LIMIT.value
                        is_retryable = True
                        retry_after = int(response.headers.get("retry-after", 60)) if hasattr(response, 'headers') else 60
                elif status_code == 401:
                        classification = APIErrorClassification.AUTHENTICATION_FAILED.value
                elif status_code == 404:
                        classification = APIErrorClassification.INVALID_MODEL.value
                elif status_code >= 500:
                        classification = APIErrorClassification.SERVICE_UNAVAILABLE.value
                        is_retryable = True

            elif provider == "anthropic":
                error_data = raw_response.get("error", {}) if raw_response else {}
                if error_data:
                    message = error_data.get("message", message)
                    api_error_type = error_data.get("type")

                if status_code == 429:
                        classification = APIErrorClassification.RATE_LIMIT.value
                        is_retryable = True
                        retry_after = int(response.headers.get("retry-after", 60)) if hasattr(response, 'headers') else 60
                elif status_code == 400 and "credit balance" in message.lower():
                        classification = APIErrorClassification.INSUFFICIENT_CREDITS.value
                elif status_code == 401:
                        classification = APIErrorClassification.AUTHENTICATION_FAILED.value
                elif status_code >= 500:
                        classification = APIErrorClassification.SERVICE_UNAVAILABLE.value
                        is_retryable = True

            elif provider == "google":
                error_data = raw_response.get("error", {}) if raw_response else {}
                if error_data:
                    message = error_data.get("message", message)
                    api_error_code = error_data.get("code")

                    if status_code == 429 or (error_data.get("status") == "RESOURCE_EXHAUSTED"):
                        if "quota" in message.lower():
                            classification = APIErrorClassification.INSUFFICIENT_CREDITS.value
                        else:
                            classification = APIErrorClassification.RATE_LIMIT.value
                        is_retryable = True
                        retry_after = 60
                elif status_code == 401:
                        classification = APIErrorClassification.AUTHENTICATION_FAILED.value
                elif status_code >= 500:
                        classification = APIErrorClassification.SERVICE_UNAVAILABLE.value
                        is_retryable = True

            elif provider == "openrouter":
                error_data = raw_response.get("error", {}) if raw_response else {}
                if error_data:
                    message = error_data.get("message", message)

                if status_code == 402:
                    classification = APIErrorClassification.INSUFFICIENT_CREDITS.value
                elif status_code == 429:
                    classification = APIErrorClassification.RATE_LIMIT.value
                    is_retryable = True
                    reset_time = response.headers.get("X-RateLimit-Reset") if hasattr(response, 'headers') else None
                    if reset_time:
                        import time
                        retry_after = max(int(reset_time) - int(time.time()), 1)
                elif status_code == 401:
                    classification = APIErrorClassification.AUTHENTICATION_FAILED.value

            elif provider == "xai":
                if status_code == 429:
                    classification = APIErrorClassification.RATE_LIMIT.value
                    is_retryable = True
                    retry_after = 120  # xAI: 20 requests per 2 hours for free tier
                elif status_code == 401:
                    classification = APIErrorClassification.AUTHENTICATION_FAILED.value

        return cls(
            message=message,
            provider=provider,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_type=api_error_type,
            classification=classification,
            is_retryable=is_retryable,
            retry_after=retry_after,
            raw_response=raw_response
        )

    def get_error_action(self) -> ErrorAction:
        """Determine what action can be taken for this error."""

        # Errors that user can fix on-the-fly
        if self.classification in [
            APIErrorClassification.INSUFFICIENT_CREDITS.value,
            APIErrorClassification.RATE_LIMIT.value,
            APIErrorClassification.SERVICE_UNAVAILABLE.value,
        ]:
            return ErrorAction.USER_FIXABLE

        # Errors that require config/code changes - terminal
        elif self.classification in [
            APIErrorClassification.AUTHENTICATION_FAILED.value,
            APIErrorClassification.PERMISSION_DENIED.value,
            APIErrorClassification.INVALID_MODEL.value,
            APIErrorClassification.INVALID_REQUEST.value,
            APIErrorClassification.VALIDATION_ERROR.value,
        ]:
            return ErrorAction.TERMINAL

        # Network errors and timeouts are fixable by user
        elif self.classification in [
            APIErrorClassification.NETWORK_ERROR.value,
            APIErrorClassification.TIMEOUT.value,
        ]:
            return ErrorAction.USER_FIXABLE  # User can check connection and retry

        return ErrorAction.TERMINAL  # Default to terminal for unknown


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

    def get_error_action(self) -> ErrorAction:
        """Browser connection errors are fixable - user can install/fix browser."""
        return ErrorAction.USER_FIXABLE


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
# API ERROR CLASSIFICATION
# =============================================================================

class APIErrorClassification(Enum):
    """Classification of API errors for intelligent error handling."""

    # Critical (non-retryable)
    INSUFFICIENT_CREDITS = "insufficient_credits"
    AUTHENTICATION_FAILED = "authentication_failed"
    INVALID_MODEL = "invalid_model"
    PERMISSION_DENIED = "permission_denied"

    # Temporary (retryable)
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"

    # Warning (degraded but functional)
    PARTIAL_FAILURE = "partial_failure"
    DEPRECATED_API = "deprecated_api"

    # Request issues
    INVALID_REQUEST = "invalid_request"
    VALIDATION_ERROR = "validation_error"

    UNKNOWN = "unknown"


# =============================================================================
# COORDINATION & ORCHESTRATION ERRORS
# =============================================================================

class CoordinationError(AgentFrameworkError):
    """Base class for coordination and orchestration errors."""

    def __init__(self, message: str, **kwargs):
        error_code = kwargs.pop("error_code", "COORDINATION_ERROR")
        super().__init__(message, error_code=error_code, **kwargs)


class TopologyError(CoordinationError):
    """
    Raised when topology structure is invalid or inconsistent.

    Examples:
    - No entry agents found
    - Cycles detected where not allowed
    - Missing required nodes
    - Invalid edge definitions
    - Disconnected components
    """

    def __init__(
        self,
        message: str,
        topology_issue: Optional[str] = None,
        affected_nodes: Optional[List[str]] = None,
        graph_structure: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.topology_issue = topology_issue
        self.affected_nodes = affected_nodes
        self.graph_structure = graph_structure

        context = kwargs.get("context", {})
        if topology_issue:
            context["topology_issue"] = topology_issue
        if affected_nodes:
            context["affected_nodes"] = affected_nodes
        if graph_structure:
            context["graph_info"] = {
                "node_count": len(graph_structure.get("nodes", [])),
                "edge_count": len(graph_structure.get("edges", []))
            }

        super().__init__(
            message,
            error_code="TOPOLOGY_ERROR",
            context=context,
            user_message="The agent network configuration is invalid.",
            suggestion="Check node connections and ensure proper topology structure.",
            **kwargs
        )


class RoutingError(CoordinationError):
    """
    Raised when routing decisions fail or no valid path exists.
    """

    def __init__(
        self,
        message: str,
        current_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
        routing_type: Optional[str] = None,
        available_paths: Optional[List[str]] = None,
        **kwargs
    ):
        self.current_agent = current_agent
        self.target_agent = target_agent
        self.routing_type = routing_type
        self.available_paths = available_paths

        context = kwargs.get("context", {})
        if current_agent:
            context["current_agent"] = current_agent
        if target_agent:
            context["target_agent"] = target_agent
        if routing_type:
            context["routing_type"] = routing_type
        if available_paths:
            context["available_paths"] = available_paths

        super().__init__(
            message,
            error_code="ROUTING_ERROR",
            context=context,
            user_message="Cannot find valid routing path between agents.",
            suggestion="Verify agent connections and routing configuration.",
            **kwargs
        )


class BranchExecutionError(CoordinationError):
    """Raised when branch execution fails or convergence issues occur."""

    def __init__(
        self,
        message: str,
        branch_id: Optional[str] = None,
        branch_type: Optional[str] = None,
        parent_branch: Optional[str] = None,
        child_branches: Optional[List[str]] = None,
        **kwargs
    ):
        self.branch_id = branch_id
        self.branch_type = branch_type
        self.parent_branch = parent_branch
        self.child_branches = child_branches

        context = kwargs.get("context", {})
        if branch_id:
            context["branch_id"] = branch_id
        if branch_type:
            context["branch_type"] = branch_type
        if parent_branch:
            context["parent_branch"] = parent_branch
        if child_branches:
            context["child_count"] = len(child_branches)

        super().__init__(
            message,
            error_code="BRANCH_EXECUTION_ERROR",
            context=context,
            user_message="Execution branch encountered an error.",
            suggestion="Check branch convergence and parallel execution settings.",
            **kwargs
        )


class ParallelExecutionError(CoordinationError):
    """Raised when parallel agent invocation fails."""

    def __init__(
        self,
        message: str,
        failed_agents: Optional[List[str]] = None,
        successful_agents: Optional[List[str]] = None,
        **kwargs
    ):
        self.failed_agents = failed_agents
        self.successful_agents = successful_agents

        context = kwargs.get("context", {})
        if failed_agents:
            context["failed_agents"] = failed_agents
        if successful_agents:
            context["successful_agents"] = successful_agents

        super().__init__(
            message,
            error_code="PARALLEL_EXECUTION_ERROR",
            context=context,
            user_message="Parallel agent execution failed.",
            suggestion="Check individual agent failures and retry logic.",
            **kwargs
        )


# =============================================================================
# STATE MANAGEMENT ERRORS
# =============================================================================

class StateError(AgentFrameworkError):
    """Base class for state management errors."""

    def __init__(self, message: str, **kwargs):
        error_code = kwargs.pop("error_code", "STATE_ERROR")
        super().__init__(message, error_code=error_code, **kwargs)


class SessionNotFoundError(StateError):
    """Raised when session doesn't exist or has expired."""

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        last_activity: Optional[float] = None,
        **kwargs
    ):
        self.session_id = session_id
        self.last_activity = last_activity

        context = kwargs.get("context", {})
        if session_id:
            context["session_id"] = session_id
        if last_activity:
            import time
            context["inactive_duration"] = time.time() - last_activity

        super().__init__(
            message,
            error_code="SESSION_NOT_FOUND_ERROR",
            context=context,
            user_message="Session not found or has expired.",
            suggestion="Start a new session or check session ID.",
            **kwargs
        )


class CheckpointError(StateError):
    """Raised when checkpoint save/restore operations fail."""

    def __init__(
        self,
        message: str,
        checkpoint_id: Optional[str] = None,
        operation: Optional[str] = None,  # "save", "restore", "delete"
        storage_backend: Optional[str] = None,
        **kwargs
    ):
        self.checkpoint_id = checkpoint_id
        self.operation = operation
        self.storage_backend = storage_backend

        context = kwargs.get("context", {})
        if checkpoint_id:
            context["checkpoint_id"] = checkpoint_id
        if operation:
            context["operation"] = operation
        if storage_backend:
            context["storage_backend"] = storage_backend

        super().__init__(
            message,
            error_code="CHECKPOINT_ERROR",
            context=context,
            user_message=f"Checkpoint {operation} operation failed." if operation else "Checkpoint operation failed.",
            suggestion="Check storage permissions and available space.",
            **kwargs
        )


class StateCorruptionError(StateError):
    """Raised when state data is corrupted or invalid."""

    def __init__(
        self,
        message: str,
        state_type: Optional[str] = None,
        corruption_details: Optional[str] = None,
        **kwargs
    ):
        self.state_type = state_type
        self.corruption_details = corruption_details

        context = kwargs.get("context", {})
        if state_type:
            context["state_type"] = state_type
        if corruption_details:
            context["corruption_details"] = corruption_details

        super().__init__(
            message,
            error_code="STATE_CORRUPTION_ERROR",
            context=context,
            user_message="State data is corrupted or invalid.",
            suggestion="Restore from backup or reinitialize state.",
            **kwargs
        )


class StateLockError(StateError):
    """Raised when state lock cannot be acquired."""

    def __init__(
        self,
        message: str,
        lock_holder: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        self.lock_holder = lock_holder
        self.timeout_seconds = timeout_seconds

        context = kwargs.get("context", {})
        if lock_holder:
            context["lock_holder"] = lock_holder
        if timeout_seconds:
            context["timeout_seconds"] = timeout_seconds

        super().__init__(
            message,
            error_code="STATE_LOCK_ERROR",
            context=context,
            user_message="Cannot acquire state lock.",
            suggestion="Wait for other operations to complete or increase timeout.",
            **kwargs
        )


# =============================================================================
# RESOURCE & POOL ERRORS
# =============================================================================

class ResourceError(AgentFrameworkError):
    """Base class for resource-related errors."""

    def __init__(self, message: str, **kwargs):
        error_code = kwargs.pop("error_code", "RESOURCE_ERROR")
        super().__init__(message, error_code=error_code, **kwargs)


class PoolExhaustedError(ResourceError):
    """Raised when agent pool has no available instances."""

    def __init__(
        self,
        message: str,
        pool_name: Optional[str] = None,
        total_instances: Optional[int] = None,
        allocated_instances: Optional[int] = None,
        requested_count: Optional[int] = None,
        **kwargs
    ):
        self.pool_name = pool_name
        self.total_instances = total_instances
        self.allocated_instances = allocated_instances
        self.requested_count = requested_count

        context = kwargs.get("context", {})
        if pool_name:
            context["pool_name"] = pool_name
        if total_instances is not None:
            context["total_instances"] = total_instances
        if allocated_instances is not None:
            context["allocated_instances"] = allocated_instances
            context["available_instances"] = (total_instances or 0) - allocated_instances
        if requested_count is not None:
            context["requested_count"] = requested_count

        super().__init__(
            message,
            error_code="POOL_EXHAUSTED_ERROR",
            context=context,
            user_message="All agent instances are currently busy.",
            suggestion="Wait for instances to become available or increase pool size.",
            **kwargs
        )

    def get_error_action(self) -> ErrorAction:
        """Pool exhaustion is fixable - user can wait for resources."""
        return ErrorAction.USER_FIXABLE


class TimeoutError(ResourceError):
    """Raised when operation exceeds timeout limit."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        **kwargs
    ):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds

        context = kwargs.get("context", {})
        if operation:
            context["operation"] = operation
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if elapsed_seconds is not None:
            context["elapsed_seconds"] = elapsed_seconds

        super().__init__(
            message,
            error_code="TIMEOUT_ERROR",
            context=context,
            user_message=f"Operation timed out after {timeout_seconds}s." if timeout_seconds else "Operation timed out.",
            suggestion="Increase timeout or optimize operation performance.",
            **kwargs
        )

    def get_error_action(self) -> ErrorAction:
        """Timeout errors are fixable - user can check connection and retry."""
        return ErrorAction.USER_FIXABLE


class ResourceLimitError(ResourceError):
    """Raised when resource limits are exceeded."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,  # "memory", "cpu", "disk"
        limit: Optional[Any] = None,
        current_usage: Optional[Any] = None,
        **kwargs
    ):
        self.resource_type = resource_type
        self.limit = limit
        self.current_usage = current_usage

        context = kwargs.get("context", {})
        if resource_type:
            context["resource_type"] = resource_type
        if limit is not None:
            context["limit"] = str(limit)
        if current_usage is not None:
            context["current_usage"] = str(current_usage)

        super().__init__(
            message,
            error_code="RESOURCE_LIMIT_ERROR",
            context=context,
            user_message="Resource limit exceeded.",
            suggestion="Free up resources or increase limits.",
            **kwargs
        )


class QuotaExceededError(ResourceError):
    """Raised when usage quotas are exceeded."""

    def __init__(
        self,
        message: str,
        quota_type: Optional[str] = None,
        quota_limit: Optional[int] = None,
        current_usage: Optional[int] = None,
        reset_time: Optional[float] = None,
        **kwargs
    ):
        self.quota_type = quota_type
        self.quota_limit = quota_limit
        self.current_usage = current_usage
        self.reset_time = reset_time

        context = kwargs.get("context", {})
        if quota_type:
            context["quota_type"] = quota_type
        if quota_limit is not None:
            context["quota_limit"] = quota_limit
        if current_usage is not None:
            context["current_usage"] = current_usage
        if reset_time is not None:
            context["reset_time"] = reset_time

        super().__init__(
            message,
            error_code="QUOTA_EXCEEDED_ERROR",
            context=context,
            user_message="Usage quota exceeded.",
            suggestion="Wait for quota reset or upgrade your plan.",
            **kwargs
        )


# =============================================================================
# COMMUNICATION ERRORS
# =============================================================================

class CommunicationError(AgentFrameworkError):
    """Base class for communication-related errors."""

    def __init__(self, message: str, **kwargs):
        error_code = kwargs.pop("error_code", "COMMUNICATION_ERROR")
        super().__init__(message, error_code=error_code, **kwargs)


class ChannelNotFoundError(CommunicationError):
    """Raised when communication channel is not found."""

    def __init__(
        self,
        message: str,
        channel_id: Optional[str] = None,
        available_channels: Optional[List[str]] = None,
        **kwargs
    ):
        self.channel_id = channel_id
        self.available_channels = available_channels

        context = kwargs.get("context", {})
        if channel_id:
            context["channel_id"] = channel_id
        if available_channels:
            context["available_channels"] = available_channels

        super().__init__(
            message,
            error_code="CHANNEL_NOT_FOUND_ERROR",
            context=context,
            user_message="Communication channel not found.",
            suggestion="Check channel ID and available channels.",
            **kwargs
        )


class UserInteractionError(CommunicationError):
    """Raised when user interaction fails."""

    def __init__(
        self,
        message: str,
        interaction_type: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        self.interaction_type = interaction_type
        self.timeout = timeout

        context = kwargs.get("context", {})
        if interaction_type:
            context["interaction_type"] = interaction_type
        if timeout is not None:
            context["timeout"] = timeout

        super().__init__(
            message,
            error_code="USER_INTERACTION_ERROR",
            context=context,
            user_message="User interaction failed.",
            suggestion="Ensure user is available and channel is configured.",
            **kwargs
        )


class ChannelConnectionError(CommunicationError):
    """Raised when channel connection fails."""

    def __init__(
        self,
        message: str,
        channel_type: Optional[str] = None,
        connection_details: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.channel_type = channel_type
        self.connection_details = connection_details

        context = kwargs.get("context", {})
        if channel_type:
            context["channel_type"] = channel_type
        if connection_details:
            context["connection_details"] = connection_details

        super().__init__(
            message,
            error_code="CHANNEL_CONNECTION_ERROR",
            context=context,
            user_message="Failed to connect to communication channel.",
            suggestion="Check channel configuration and network connectivity.",
            **kwargs
        )


class MessageRoutingError(CommunicationError):
    """Raised when inter-agent message routing fails."""

    def __init__(
        self,
        message: str,
        source_agent: Optional[str] = None,
        target_agent: Optional[str] = None,
        message_id: Optional[str] = None,
        **kwargs
    ):
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.message_id = message_id

        context = kwargs.get("context", {})
        if source_agent:
            context["source_agent"] = source_agent
        if target_agent:
            context["target_agent"] = target_agent
        if message_id:
            context["message_id"] = message_id

        super().__init__(
            message,
            error_code="MESSAGE_ROUTING_ERROR",
            context=context,
            user_message="Message routing between agents failed.",
            suggestion="Check agent connectivity and message format.",
            **kwargs
        )


# =============================================================================
# WORKFLOW ERRORS
# =============================================================================

class WorkflowError(AgentFrameworkError):
    """Base class for workflow-related errors."""

    def __init__(self, message: str, **kwargs):
        error_code = kwargs.pop("error_code", "WORKFLOW_ERROR")
        super().__init__(message, error_code=error_code, **kwargs)


class WorkflowConfigurationError(WorkflowError):
    """Raised when workflow configuration is invalid."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        config_issue: Optional[str] = None,
        **kwargs
    ):
        self.workflow_name = workflow_name
        self.config_issue = config_issue

        context = kwargs.get("context", {})
        if workflow_name:
            context["workflow_name"] = workflow_name
        if config_issue:
            context["config_issue"] = config_issue

        super().__init__(
            message,
            error_code="WORKFLOW_CONFIGURATION_ERROR",
            context=context,
            user_message="Workflow configuration is invalid.",
            suggestion="Check workflow definition and required parameters.",
            **kwargs
        )


class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        step_name: Optional[str] = None,
        step_number: Optional[int] = None,
        **kwargs
    ):
        self.workflow_name = workflow_name
        self.step_name = step_name
        self.step_number = step_number

        context = kwargs.get("context", {})
        if workflow_name:
            context["workflow_name"] = workflow_name
        if step_name:
            context["step_name"] = step_name
        if step_number is not None:
            context["step_number"] = step_number

        super().__init__(
            message,
            error_code="WORKFLOW_EXECUTION_ERROR",
            context=context,
            user_message="Workflow execution failed.",
            suggestion="Check workflow logs and individual step failures.",
            **kwargs
        )


class WorkflowStateError(WorkflowError):
    """Raised when workflow state is invalid."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        current_state: Optional[str] = None,
        expected_state: Optional[str] = None,
        **kwargs
    ):
        self.workflow_name = workflow_name
        self.current_state = current_state
        self.expected_state = expected_state

        context = kwargs.get("context", {})
        if workflow_name:
            context["workflow_name"] = workflow_name
        if current_state:
            context["current_state"] = current_state
        if expected_state:
            context["expected_state"] = expected_state

        super().__init__(
            message,
            error_code="WORKFLOW_STATE_ERROR",
            context=context,
            user_message="Workflow state is invalid.",
            suggestion="Reset workflow or restore from checkpoint.",
            **kwargs
        )


class WorkflowTimeoutError(WorkflowError):
    """Raised when workflow execution times out."""

    def __init__(
        self,
        message: str,
        workflow_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        **kwargs
    ):
        self.workflow_name = workflow_name
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds

        context = kwargs.get("context", {})
        if workflow_name:
            context["workflow_name"] = workflow_name
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        if elapsed_seconds is not None:
            context["elapsed_seconds"] = elapsed_seconds

        super().__init__(
            message,
            error_code="WORKFLOW_TIMEOUT_ERROR",
            context=context,
            user_message=f"Workflow timed out after {timeout_seconds}s." if timeout_seconds else "Workflow timed out.",
            suggestion="Increase workflow timeout or optimize individual steps.",
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
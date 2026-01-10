"""
Shared response processors for MARSYS.

This module contains:
1. ResponseProcessor - Base class for all format processors
2. Shared processors that are format-agnostic (ErrorMessageProcessor, ToolCallProcessor)

Format-specific processors (like StructuredJSONProcessor) live in their format subdirectory.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseProcessor(ABC):
    """
    Base class for response format processors.

    Each processor implements the ability to check if it can handle
    a response and to extract structured data from it.
    """

    @abstractmethod
    def can_process(self, response: Any) -> bool:
        """
        Check if this processor can handle the response.

        Args:
            response: The response to check (could be Message, dict, str, etc.)

        Returns:
            True if this processor can handle the response
        """
        pass

    @abstractmethod
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Process the response and extract structured data.

        Args:
            response: The response to process

        Returns:
            Dictionary with extracted data, or None if processing failed
        """
        pass

    @abstractmethod
    def priority(self) -> int:
        """
        Return priority for processor ordering (higher = earlier).

        Processors are sorted by priority and the first one that can
        process a response is used.

        Returns:
            Integer priority value
        """
        pass


class ErrorMessageProcessor(ResponseProcessor):
    """
    Handles error Messages from agents (role='error').

    This processor handles Messages with role='error' and extracts
    error information for appropriate error handling and recovery.
    """

    def can_process(self, response: Any) -> bool:
        """Check if this is an error Message."""
        # Import here to avoid circular dependency
        from ...agents.memory import Message

        return isinstance(response, Message) and response.role == "error"

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract error information from the error Message."""
        from ...agents.memory import Message

        if not isinstance(response, Message):
            return None

        try:
            error_content = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            error_content = {"error": str(response.content)}

        classification = error_content.get("classification", "unknown")
        is_retryable = error_content.get("is_retryable", False)

        # Determine action type based on classification
        from ...agents.exceptions import APIErrorClassification

        auto_retry_classifications = [
            APIErrorClassification.TIMEOUT.value,
            APIErrorClassification.NETWORK_ERROR.value,
            APIErrorClassification.SERVICE_UNAVAILABLE.value,
        ]

        terminal_classifications = [
            APIErrorClassification.AUTHENTICATION_FAILED.value,
            APIErrorClassification.PERMISSION_DENIED.value,
            APIErrorClassification.INVALID_MODEL.value,
            APIErrorClassification.INVALID_REQUEST.value,
        ]

        if classification in auto_retry_classifications and is_retryable:
            action_type = "auto_retry"
        elif classification in terminal_classifications:
            action_type = "terminal_error"
        else:
            action_type = "error_recovery"

        return {
            "next_action": action_type,
            "error_info": {
                "message": error_content.get("error", "Unknown error"),
                "classification": classification,
                "provider": error_content.get("provider"),
                "is_retryable": is_retryable,
                "retry_after": error_content.get("retry_after"),
                "suggested_action": error_content.get("suggested_action"),
                "raw_content": error_content,
            },
        }

    def priority(self) -> int:
        return 100  # Highest priority - errors should be handled first


class ToolCallProcessor(ResponseProcessor):
    """
    Handles responses with native tool_calls.

    NOTE: This processor is used by StepExecutor for native tool call extraction,
    NOT by ValidationProcessor for response validation. Tools use the model's
    native tool calling mechanism (response.tool_calls field), not via
    next_action in the response content.
    """

    def can_process(self, response: Any) -> bool:
        """Check for tool_calls in Message or dict response."""
        if hasattr(response, "tool_calls") and response.tool_calls:
            return True
        if isinstance(response, dict):
            return "tool_calls" in response and isinstance(
                response["tool_calls"], list
            )
        return False

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract tool_calls from response."""
        try:
            if hasattr(response, "tool_calls"):
                tool_calls = response.tool_calls
                content = response.content if hasattr(response, "content") else ""
            elif isinstance(response, dict):
                tool_calls = response.get("tool_calls", [])
                content = response.get("content", "")
            else:
                return None

            return {
                "next_action": "call_tool",
                "tool_calls": tool_calls,
                "content": content,
                "raw_response": response,
            }
        except Exception as e:
            logger.error(f"Failed to process tool calls: {e}")
            return None

    def priority(self) -> int:
        return 90  # High priority - tool calls should be detected early

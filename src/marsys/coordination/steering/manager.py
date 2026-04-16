"""
Centralized steering prompt construction for agent guidance.

The steering system provides transient, context-aware prompts to guide agents
during execution, completely separate from persistent memory.
"""

from dataclasses import dataclass
from typing import Optional, List
import logging

from ..validation.types import ValidationErrorCategory

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Structured error context for steering decisions."""
    category: ValidationErrorCategory
    error_message: str
    retry_suggestion: Optional[str] = None
    retry_count: int = 0
    classification: Optional[str] = None  # For API errors
    failed_action: Optional[str] = None   # What action was attempted


@dataclass
class SteeringContext:
    """Full context for steering prompt generation."""
    agent_name: str
    available_actions: List[str]          # From topology
    error_context: Optional[ErrorContext] = None
    is_retry: bool = False
    steering_mode: str = "error"


class SteeringManager:
    """
    Centralized steering prompt construction.

    Responsible for:
    - Deciding when to inject steering based on mode and error state
    - Building error-specific, concise prompts
    - Providing format reminders without polluting agent memory
    """

    def __init__(self):
        """Initialize steering manager with statistics tracking."""
        self.stats = {
            "total_injections": 0,
            "by_mode": {"error": 0, "auto": 0, "always": 0},
            "by_category": {}
        }

    def get_steering_prompt(self, context: SteeringContext) -> Optional[str]:
        """
        Decide whether to inject steering and construct appropriate prompt.

        Args:
            context: Full steering context with error info and mode

        Returns:
            Transient steering prompt string, or None if no steering needed
        """
        mode = context.steering_mode

        # Decision matrix
        if mode == "never" or mode == "error":
            # "never" aliased to "error" for backward compatibility
            if not context.error_context:
                return None
            prompt = self._build_error_prompt(context)
            if prompt:
                self._log_injection(mode, context)
            return prompt

        elif mode == "auto":
            # On any retry
            if context.is_retry:
                if context.error_context:
                    prompt = self._build_error_prompt(context)
                else:
                    prompt = self._build_generic_prompt(context)

                if prompt:
                    self._log_injection(mode, context)
                return prompt
            return None

        elif mode == "always":
            # Every step
            if context.error_context:
                prompt = self._build_error_prompt(context)
            else:
                prompt = self._build_generic_prompt(context)

            if prompt:
                self._log_injection(mode, context)
            return prompt

        logger.warning(f"Unknown steering mode '{mode}', defaulting to no steering")
        return None

    def _log_injection(self, mode: str, context: SteeringContext):
        """Log steering injection for statistics."""
        self.stats["total_injections"] += 1
        self.stats["by_mode"][mode] = self.stats["by_mode"].get(mode, 0) + 1

        if context.error_context:
            category = context.error_context.category.value
            if category not in self.stats["by_category"]:
                self.stats["by_category"][category] = 0
            self.stats["by_category"][category] += 1

            logger.info(
                f"Steering injected for {context.agent_name} "
                f"(mode={mode}, category={category}, retry={context.error_context.retry_count})"
            )
        else:
            logger.info(f"Steering injected for {context.agent_name} (mode={mode}, generic)")

    def get_stats(self) -> dict:
        """Get steering injection statistics."""
        return self.stats.copy()

    def _build_error_prompt(self, context: SteeringContext) -> str:
        """Build error-category-specific steering prompt."""
        error = context.error_context
        category = error.category

        if category == ValidationErrorCategory.PERMISSION_ERROR:
            return self._permission_error_prompt(context)
        elif category == ValidationErrorCategory.ACTION_ERROR:
            return self._action_error_prompt(context)
        elif category == ValidationErrorCategory.API_TRANSIENT:
            return self._api_transient_prompt(context)
        else:
            return self._generic_error_prompt(context)

    def _permission_error_prompt(self, context: SteeringContext) -> str:
        """Permission/topology violation guidance."""
        error = context.error_context
        coord_tools = [a for a in context.available_actions if a != "tool_calls"]

        prompt = f"Permission denied: {error.error_message}\n\n"
        prompt += f"Available coordination tools: {', '.join(coord_tools)}\n"
        prompt += "Please use one of the available tools to proceed."

        return prompt

    def _action_error_prompt(self, context: SteeringContext) -> str:
        """Invalid action guidance."""
        error = context.error_context
        coord_tools = [a for a in context.available_actions if a != "tool_calls"]

        prompt = f"Invalid action: {error.error_message}\n\n"
        prompt += f"Available coordination tools: {', '.join(coord_tools)}\n"

        if error.retry_suggestion:
            prompt += f"\n{error.retry_suggestion}"

        return prompt

    def _api_transient_prompt(self, context: SteeringContext) -> str:
        """API transient error - NO format reminders, just retry notice."""
        error = context.error_context
        classification = error.classification or "API error"

        # Keep it minimal - don't clutter with format instructions
        return f"""Previous API call failed: {classification}. Retrying automatically.

Please proceed with your intended action."""

    def _generic_error_prompt(self, context: SteeringContext) -> str:
        """Fallback for uncategorized errors."""
        error = context.error_context

        return f"""Error: {error.error_message}

{error.retry_suggestion or 'Please try again with a different approach.'}"""

    def _build_generic_prompt(self, context: SteeringContext) -> str:
        """Generic steering prompt (no specific error)."""
        coord_tools = [a for a in context.available_actions if a != "tool_calls"]
        has_tools = "tool_calls" in context.available_actions

        if not coord_tools and not has_tools:
            return ""

        parts = ["Use your available tools to take action:"]
        if "invoke_agent" in coord_tools:
            parts.append("- `invoke_agent`: Delegate to a peer agent")
        if "return_final_response" in coord_tools:
            parts.append("- `return_final_response`: Return your final answer")
        if has_tools:
            parts.append("- Your regular tools for task execution")

        return "\n".join(parts)

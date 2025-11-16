"""
Response validator for centralized response processing.

This module is the CENTRAL hub for ALL response parsing in the coordination system.
It supports multiple response formats and validates actions based on topology permissions.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from ..branches.types import ExecutionBranch, ExecutionState, StepResult
from ..topology.graph import TopologyGraph
from .types import AgentInvocation, ValidationErrorCategory, ValidationError

if TYPE_CHECKING:
    from ...agents import BaseAgent

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Supported action types in agent responses."""
    INVOKE_AGENT = "invoke_agent"          # Sequential agent invocation
    PARALLEL_INVOKE = "parallel_invoke"    # Parallel agent execution (NEW!)
    CALL_TOOL = "call_tool"               # Tool execution
    FINAL_RESPONSE = "final_response"      # Complete execution
    END_CONVERSATION = "end_conversation"  # End conversation branch
    WAIT_AND_AGGREGATE = "wait_and_aggregate"  # Wait for parallel results
    ERROR_RECOVERY = "error_recovery"      # Route to user for error recovery
    TERMINAL_ERROR = "terminal_error"      # Route to user for terminal error display
    AUTO_RETRY = "auto_retry"              # Automatic retry without user intervention


@dataclass
class ValidationResult:
    """Result of response validation with all routing decisions."""
    # Existing validation fields
    is_valid: bool
    action_type: Optional[ActionType] = None
    parsed_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_suggestion: Optional[str] = None
    error_category: Optional[str] = None  # NEW: Error category for steering (from ErrorCategory enum)
    invocations: List['AgentInvocation'] = None  # Complete invocation data for invoke/parallel_invoke
    tool_calls: List[Dict[str, Any]] = None

    # NEW decision fields (moved from StepResult)
    next_agent: Optional[str] = None  # Single next agent for sequential flow
    should_end_branch: bool = False  # Branch should complete
    requires_tool_continuation: bool = False  # Agent continues after tools
    final_response: Optional[Any] = None  # Final response content

    def __post_init__(self):
        if self.invocations is None:
            self.invocations = []
        if self.tool_calls is None:
            self.tool_calls = []

        # Auto-populate next_agent for single invocation
        if not self.next_agent and self.invocations and len(self.invocations) == 1:
            self.next_agent = self.invocations[0].agent_name

    @property
    def next_agents(self) -> List[str]:
        """Backward compatibility property to get agent names from invocations."""
        return [inv.agent_name for inv in self.invocations] if self.invocations else []


class ResponseProcessor(ABC):
    """Base class for response format processors."""
    
    @abstractmethod
    def can_process(self, response: Any) -> bool:
        """Check if this processor can handle the response."""
        pass
    
    @abstractmethod
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Process the response and extract structured data."""
        pass
    
    @abstractmethod
    def priority(self) -> int:
        """Return priority for processor ordering (higher = earlier)."""
        pass


class StructuredJSONProcessor(ResponseProcessor):
    """Handles JSON responses with next_action/action_input structure."""
    
    def _extract_json_from_code_block(self, text: str) -> Optional[str]:
        """
        Extract JSON from markdown code blocks or plain JSON.

        Handles nested code blocks inside JSON string values by using greedy matching
        to find the LAST closing delimiter.
        """
        # Try markdown blocks with greedy matching (```json {...}``` or ```{...}```)
        # Use greedy .* instead of non-greedy .*? to match until the LAST ```
        # This handles cases where JSON contains nested code blocks like:
        #   ```json
        #   {"response": "text with ```python\ncode\n``` inside"}
        #   ```
        code_block_pattern = r'```(?:json)?\s*\n?(.*)```'
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no markdown blocks found, check if the entire text is valid JSON
        # This handles plain JSON responses without markdown formatting
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            try:
                # Validate it's actually JSON before returning
                json.loads(text_stripped)
                return text_stripped
            except json.JSONDecodeError:
                pass

        return None
    
    def _extract_first_json(self, text: str) -> Optional[str]:
        """
        Extract the first valid JSON object from text that may contain multiple JSONs.
        Handles cases where LLM returns multiple concatenated JSON objects.
        """
        # Track braces to find complete JSON objects
        brace_count = 0
        start_idx = None
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    # Found a complete JSON object
                    json_str = text[start_idx:i+1]
                    try:
                        # Validate it's valid JSON
                        json.loads(json_str)
                        # Check if there's more JSON after this
                        remaining = text[i+1:].strip()
                        if remaining and remaining[0] == '{':
                            # Count how many additional JSON objects there might be
                            additional_count = remaining.count('{"')
                            logger.info(f"Multiple JSON actions detected ({additional_count + 1} total). Processing first action, others will be ignored.")
                            logger.debug("Note: Currently the system processes one action at a time. Consider splitting complex workflows into sequential steps.")
                        return json_str
                    except json.JSONDecodeError:
                        # Not valid JSON, continue searching
                        continue
        
        return None
    
    def can_process(self, response: Any) -> bool:
        """Check for expected JSON structure."""
        if isinstance(response, dict):
            return "next_action" in response
        if isinstance(response, str):
            # First try to extract from code block
            json_str = self._extract_json_from_code_block(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "next_action" in data:
                        return True
                    # If parsed but no next_action, continue to next method
                except json.JSONDecodeError:
                    # Parsing failed, continue to next method
                    pass

            # Try to extract first JSON if multiple JSONs present
            json_str = self._extract_first_json(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "next_action" in data:
                        return True
                    # If parsed but no next_action, continue to next method
                except json.JSONDecodeError:
                    # Parsing failed, continue to next method
                    pass

            # Finally try direct JSON parsing
            try:
                data = json.loads(response)
                return isinstance(data, dict) and "next_action" in data
            except json.JSONDecodeError:
                # All methods failed
                return False
        return False
    
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON structure."""
        try:
            if isinstance(response, str):
                # First try to extract from code block
                json_str = self._extract_json_from_code_block(response)
                if json_str:
                    data = json.loads(json_str)
                else:
                    # Try to extract first JSON if multiple JSONs present
                    json_str = self._extract_first_json(response)
                    if json_str:
                        data = json.loads(json_str)
                    else:
                        # Try direct JSON parsing as last resort
                        data = json.loads(response)
            else:
                data = response
            
            result = {
                "next_action": data.get("next_action"),
               "raw_response": data
            }
            
            # Handle different action types
            if data.get("next_action") == "invoke_agent":
                action_input = data.get("action_input", {})
                
                # Check for unified array format (new standard)
                if isinstance(action_input, list):
                    try:
                        # Parse invocations using Pydantic
                        invocations = []
                        for idx, item in enumerate(action_input):
                            if isinstance(item, dict) and "agent_name" in item:
                                # Create AgentInvocation for validation
                                invocation = AgentInvocation(
                                    agent_name=item["agent_name"],
                                    request=item.get("request", {}),
                                    instance_id=f"{item['agent_name']}_{idx}_{uuid.uuid4().hex[:8]}"
                                )
                                # Keep as AgentInvocation object for type safety
                                invocations.append(invocation)
                            else:
                                raise ValueError(f"Invalid invocation format at index {idx}: missing 'agent_name'")
                        
                        # Store validated invocations as objects
                        result["invocations"] = invocations
                        
                        # For backward compatibility, also populate agent_requests
                        result["agent_requests"] = {}
                        for inv in invocations:
                            # Store by instance_id to avoid overwrites
                            result["agent_requests"][inv.instance_id] = inv.request
                        
                        # For backward compatibility with single agent flow
                        if len(invocations) == 1:
                            # Only apply backward compatibility for truly single invocation
                            if not hasattr(invocations[0], 'request'):
                                logger.error(f"Invocation object missing 'request' property: {type(invocations[0])}")
                                return None
                            result["target_agent"] = invocations[0].agent_name
                            result["action_input"] = invocations[0].request
                    
                    except Exception as e:
                        # Log error and return None to let process_response handle it properly
                        logger.warning(f"Failed to parse agent invocations: {str(e)}")
                        return None
                
                # Single dict format - convert to single-item list
                elif isinstance(action_input, dict):
                    if "agent_name" in action_input:
                        # Convert single dict to list format
                        invocation = AgentInvocation(
                            agent_name=action_input["agent_name"],
                            request=action_input.get("request", {}),
                            instance_id=f"{action_input['agent_name']}_0_{uuid.uuid4().hex[:8]}"
                        )
                        # Keep as AgentInvocation object
                        result["invocations"] = [invocation]
                        
                        # For backward compatibility
                        result["target_agent"] = invocation.agent_name
                        result["action_input"] = invocation.request
                        result["agent_requests"] = {invocation.instance_id: invocation.request}
                    else:
                        # Invalid format - return error
                        return {
                            "next_action": "validation_error",
                            "error": "Invalid invocation format",
                            "details": ["Dictionary must contain 'agent_name' field"],
                            "retry_suggestion": "Use format: {'agent_name': 'agent', 'request': {...}} or array format for multiple agents",
                            "raw_response": data
                        }
                else:
                    # Invalid format - not array or proper dict
                    return {
                        "next_action": "validation_error",
                        "error": "Invalid action_input format",
                        "details": [f"Expected array or dict with 'agent_name', got {type(action_input).__name__}"],
                        "retry_suggestion": "Use array format: [{'agent_name': 'agent', 'request': {...}}] for agent invocations",
                        "raw_response": data
                    }
                
            elif data.get("next_action") == "parallel_invoke":
                # Legacy parallel_invoke support for backward compatibility
                result["target_agents"] = data.get("agents", [])
                result["wait_for_all"] = data.get("wait_for_all", True)
                result["is_parallel"] = True
                
                # Extract agent requests if provided
                result["agent_requests"] = {}
                action_input = data.get("action_input", {})
                if isinstance(action_input, dict):
                    for agent in result["target_agents"]:
                        if agent in action_input:
                            result["agent_requests"][agent] = action_input[agent]
            elif data.get("next_action") == "call_tool":
                # IMPORTANT: Tool calls should ONLY come through native Message.tool_calls,
                # NOT through content with next_action="call_tool".
                # If LLM puts tool calls in content, it's a mistake - send error back.

                # Extract what they tried to do for error message
                action_input = data.get("action_input", {})
                tool_calls = action_input.get("tool_calls") if isinstance(action_input, dict) else []
                if not tool_calls:
                    tool_calls = data.get("tool_calls", [])

                return {
                    "next_action": "validation_error",
                    "error": "Tool calls must use native tool_calls format, not content with next_action='call_tool'",
                    "details": [
                        "Your model should return tool calls using the native tool_calls field in the response",
                        "Do NOT manually construct tool calls in the content field",
                        f"You attempted to call {len(tool_calls)} tool(s) incorrectly"
                    ],
                    "retry_suggestion": "Use the model's native tool calling format. Tool calls will be automatically detected and executed.",
                    "raw_response": data
                }
            elif data.get("next_action") == "final_response":
                # Check for response in action_input first (new format)
                action_input = data.get("action_input", {})
                if isinstance(action_input, dict):
                    if "response" in action_input:
                        result["final_response"] = action_input["response"]
                    elif "report" in action_input:
                        result["final_response"] = action_input["report"]
                    else:
                        # Fallback to old format
                        result["final_response"] = data.get("final_response", data.get("content", ""))
                else:
                    # Fallback to old format
                    result["final_response"] = data.get("final_response", data.get("content", ""))
            
            # Include all fields from original data
            result["content"] = data.get("content", "")
            result["message"] = data.get("message", "")  # Extract message field
            # action_input is already handled above per action type
            
            # Preserve any additional fields
            for key, value in data.items():
                if key not in result:
                    result[key] = value
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed at position {e.pos}: {e.msg}")
            logger.debug(f"Failed JSON string (first 500 chars): {json_str[:500] if 'json_str' in locals() else 'None'}")
            
            # Try to extract action type at minimum
            action_match = re.search(r'"next_action"\s*:\s*"([^"]+)"', str(response))
            if action_match:
                action = action_match.group(1)
                logger.info(f"Extracted action type despite JSON error: {action}")
                return {
                    "next_action": action,
                    "parse_error": str(e),
                    "partial_parse": True,
                    "raw_response": response
                }
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing response: {type(e).__name__}: {e}")
            return None
    
    def priority(self) -> int:
        return 100


class ToolCallProcessor(ResponseProcessor):
    """
    Handles responses with native tool_calls.

    IMPORTANT: This processor is used by StepExecutor for native tool call extraction.
    It should NOT be registered in ValidationProcessor.processors list.
    ValidationProcessor should NEVER validate tool_calls - they are pre-processed by StepExecutor.
    """

    def can_process(self, response: Any) -> bool:
        """Check for tool_calls in Message or dict response."""
        # Support Message objects (primary use case for StepExecutor)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return True
        # Support dict responses (backward compatibility)
        if isinstance(response, dict):
            return "tool_calls" in response and isinstance(response["tool_calls"], list)
        return False

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract tool_calls from Message or dict response."""
        try:
            # Extract tool_calls from Message object
            if hasattr(response, 'tool_calls'):
                tool_calls = response.tool_calls
                content = response.content if hasattr(response, 'content') else ""
            # Extract from dict (backward compatibility)
            elif isinstance(response, dict):
                tool_calls = response.get("tool_calls", [])
                content = response.get("content", "")
            else:
                return None

            return {
                "next_action": "call_tool",
                "tool_calls": tool_calls,
                "content": content,
                "raw_response": response
            }
        except Exception as e:
            logger.error(f"Failed to process tool calls: {e}")
            return None

    def priority(self) -> int:
        return 90


class NaturalLanguageProcessor(ResponseProcessor):
    """Extracts actions from natural language using patterns."""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.patterns = {
            "invoke_agent": re.compile(
                r"(?:invoke|call|ask|consult|delegate to|hand off to|pass to)\s+(\w+)",
                re.IGNORECASE
            ),
            "parallel_invoke": re.compile(
                r"(?:run|execute|invoke)\s+(?:both|all|simultaneously|in parallel).*?(\w+)\s+and\s+(\w+)",
                re.IGNORECASE
            ),
            "final_response": re.compile(
                r"\b(?:my final answer is|this is my final response|final report:|conclusion:|here is my final answer|here's my final response)\b",
                re.IGNORECASE
            ),
            "end_conversation": re.compile(
                r"(?:end conversation|conversation complete|stop discussing)",
                re.IGNORECASE
            )
        }
    
    def can_process(self, response: Any) -> bool:
        """Process strings and dicts with content field."""
        if isinstance(response, str):
            return True
        if isinstance(response, dict) and "content" in response:
            return True
        return False
    
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract intent from natural language."""
        try:
            # Convert to string
            if isinstance(response, dict):
                text = response.get("content", "") or str(response)
            else:
                text = str(response)
            
            # Check patterns in priority order
            
            # Check for parallel invocation first
            parallel_match = self.patterns["parallel_invoke"].search(text)
            if parallel_match:
                agents = [parallel_match.group(1), parallel_match.group(2)]
                return {
                    "next_action": "parallel_invoke",
                    "target_agents": agents,
                    "wait_for_all": True,
                    "content": text,
                    "raw_response": response
                }
            
            # Check for single agent invocation
            invoke_match = self.patterns["invoke_agent"].search(text)
            if invoke_match:
                return {
                    "next_action": "invoke_agent",
                    "target_agent": invoke_match.group(1),
                    "content": text,
                    "raw_response": response
                }
            
            # Check for conversation end
            if self.patterns["end_conversation"].search(text):
                return {
                    "next_action": "end_conversation",
                    "content": text,
                    "raw_response": response
                }
            
            # Check for final response
            if self.patterns["final_response"].search(text):
                return {
                    "next_action": "final_response",
                    "content": text,
                    "raw_response": response
                }
            
            # No patterns matched - cannot determine action
            logger.debug(f"NaturalLanguageProcessor: No patterns matched in text: {text[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Failed to process natural language: {e}")
            return None
    
    def priority(self) -> int:
        return 10  # Lowest priority


class ErrorMessageProcessor(ResponseProcessor):
    """Handles error Messages from agents (role='error')."""

    def can_process(self, response: Any) -> bool:
        """Check if this is an error Message."""
        from ...agents.memory import Message
        return isinstance(response, Message) and response.role == "error"

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract error information and determine recovery action."""
        from ...agents.memory import Message

        if not isinstance(response, Message):
            return None

        try:
            error_content = json.loads(response.content)
        except:
            error_content = {"error": str(response.content)}

        # Use preserved classification data if available
        classification = error_content.get('classification', 'unknown')
        is_retryable = error_content.get('is_retryable', False)

        # Map classification to action type
        from ...agents.exceptions import APIErrorClassification

        # Fixable errors that user can resolve
        fixable_classifications = [
            APIErrorClassification.INSUFFICIENT_CREDITS.value,
            APIErrorClassification.RATE_LIMIT.value,
            APIErrorClassification.SERVICE_UNAVAILABLE.value,
            APIErrorClassification.NETWORK_ERROR.value,
            APIErrorClassification.TIMEOUT.value
        ]

        # Terminal errors that require config changes
        terminal_classifications = [
            APIErrorClassification.AUTHENTICATION_FAILED.value,
            APIErrorClassification.PERMISSION_DENIED.value,
            APIErrorClassification.INVALID_MODEL.value,
            APIErrorClassification.INVALID_REQUEST.value
        ]

        # Auto-retry classifications (retry automatically before user intervention)
        auto_retry_classifications = [
            APIErrorClassification.TIMEOUT.value,
            APIErrorClassification.NETWORK_ERROR.value,
            APIErrorClassification.SERVICE_UNAVAILABLE.value
        ]

        if classification in auto_retry_classifications and is_retryable:
            # Auto-retry these errors (BranchExecutor will handle retry logic)
            action_type = "auto_retry"
        elif classification in fixable_classifications:
            action_type = "error_recovery"  # Route to user for fixing after retries exhausted
        elif classification in terminal_classifications:
            action_type = "terminal_error"  # Display and exit
        else:
            # Unknown errors default to terminal
            action_type = "terminal_error"

        error_info = {
            "message": error_content.get('error', 'Unknown error'),
            "classification": classification,
            "provider": error_content.get('provider'),
            "is_retryable": is_retryable,
            "retry_after": error_content.get('retry_after'),
            "suggested_action": error_content.get('suggested_action'),
            "error_type": error_content.get('error_type', ''),
            "raw_content": error_content
        }

        # Log for debugging
        logger.debug(f"ErrorMessageProcessor created error_info: {error_info}")

        return {
            "next_action": action_type,
            "error_info": error_info
        }

    def priority(self) -> int:
        return 100  # Highest priority - check for errors first


class ValidationProcessor:
    """Central hub for ALL response processing."""
    
    def __init__(self, topology_graph: TopologyGraph):
        self.topology_graph = topology_graph
        
        # Initialize processors in priority order
        # NOTE: ToolCallProcessor is NOT included - it's used only by StepExecutor
        # for native tool call extraction, NOT for response validation
        self.processors: List[ResponseProcessor] = sorted([
            ErrorMessageProcessor(),  # Highest priority for API errors
            StructuredJSONProcessor(),
            # ToolCallProcessor(),  # REMOVED - used only by StepExecutor, not ValidationProcessor
            # NaturalLanguageProcessor()
        ], key=lambda p: p.priority(), reverse=True)
        
        # Action validators
        self.action_validators = {
            ActionType.INVOKE_AGENT: self._validate_agent_invocation,
            ActionType.PARALLEL_INVOKE: self._validate_parallel_invocation,
            ActionType.CALL_TOOL: self._validate_tool_call,
            ActionType.FINAL_RESPONSE: self._validate_final_response,
            ActionType.END_CONVERSATION: self._validate_end_conversation,
            ActionType.WAIT_AND_AGGREGATE: self._validate_wait_aggregate,
            ActionType.ERROR_RECOVERY: self._validate_error_recovery,
            ActionType.TERMINAL_ERROR: self._validate_terminal_error
        }
    
    def _get_allowed_actions(self, agent: BaseAgent, branch: ExecutionBranch = None) -> List[str]:
        """Get list of actions this agent is allowed to perform based on topology."""
        allowed = []
        
        # Check if agent can invoke other agents
        next_agents = self.topology_graph.get_next_agents(agent.name)
        if next_agents:
            allowed.append("invoke_agent")

        # Check if agent has tools
        # NOTE: We use "tool_calls" (not "call_tool") to match native model API terminology
        if hasattr(agent, 'tools') and agent.tools:
            allowed.append("tool_calls")

        # Check if agent can return final response
        # CHANGED: Only allow final_response for User access
        # REMOVED: is_child_branch check - child branches must flow to convergence points
        has_user_access = self.topology_graph.has_user_access(agent.name)
        
        if has_user_access:
            allowed.append("final_response")
        else:
            logger.info(f"VALIDATION: {agent.name} NOT allowed final_response - next_agents: {next_agents}")
        
        return allowed
    
    async def process_response(
        self,
        raw_response: Any,
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """
        Main entry point for response processing.

        This method:
        1. Extracts content from Message objects
        2. Tries each processor to parse the response content
        3. Validates the extracted action
        4. Checks topology permissions
        5. Returns structured validation result

        Args:
            raw_response: Agent response (MUST be Message object)
            agent: The agent that generated the response
            branch: Current execution branch
            exec_state: Current execution state

        Returns:
            ValidationResult with routing decisions

        Raises:
            TypeError: If raw_response is not a Message object
        """
        from ...agents.memory import Message

        # CRITICAL: Only accept Message objects
        # This enforces the standardized interface throughout the system
        if not isinstance(raw_response, Message):
            raise TypeError(
                f"ValidationProcessor requires Message object from agent {agent.name}, "
                f"got {type(raw_response).__name__}. "
                f"All agents must return Message objects from their _run() methods."
            )

        # Extract content for parsing
        # NOTE: If Message has tool_calls, BranchExecutor should skip validation (line 866)
        # We only validate content for next_action decisions (invoke_agent, final_response)
        content_to_parse = raw_response.content

        # Handle empty responses - provide clear guidance based on available actions
        allowed_actions = self._get_allowed_actions(agent, branch)

        # Build guidance message based on what's available
        guidance_parts = []

        # Tool calls (native format)
        if "tool_calls" in allowed_actions:
            guidance_parts.append("use native tool_calls format to call tools")

        # Agent invocations or final response (content field with next_action)
        content_actions = []
        if "invoke_agent" in allowed_actions:
            content_actions.append('"invoke_agent"')
        if "final_response" in allowed_actions:
            content_actions.append('"final_response"')

        if content_actions:
            actions_str = " or ".join(content_actions)
            guidance_parts.append(f'return content with "next_action" set to {actions_str} and proper "action_input"')

        # Combine into format hint
        if len(guidance_parts) == 2:
            format_hint = f"You can either: {guidance_parts[0]}, OR {guidance_parts[1]}"
        elif len(guidance_parts) == 1:
            format_hint = f"You must: {guidance_parts[0]}"
        else:
            format_hint = "Provide a valid response"

        # Check for empty content
        if content_to_parse is None:
            return ValidationResult(
                is_valid=False,
                error_message="Response content is None",
                retry_suggestion=f"Your response was empty. {format_hint}",
                error_category=ValidationErrorCategory.FORMAT_ERROR.value
            )

        if isinstance(content_to_parse, str) and not content_to_parse.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Response content is empty string",
                retry_suggestion=f"Your response was an empty string. {format_hint}",
                error_category=ValidationErrorCategory.FORMAT_ERROR.value
            )

        if isinstance(content_to_parse, dict) and not content_to_parse:
            return ValidationResult(
                is_valid=False,
                error_message="Response content is empty dictionary",
                retry_suggestion=f"Your response was an empty dictionary. {format_hint}",
                error_category=ValidationErrorCategory.FORMAT_ERROR.value
            )

        # 1. Try each processor to parse response content
        parsed = None
        for processor in self.processors:
            if processor.can_process(content_to_parse):
                parsed = processor.process(content_to_parse)
                if parsed:
                    logger.debug(f"Response processed by {processor.__class__.__name__}")
                    break
        
        if not parsed:
            # Try to detect if this looks like a successful response that couldn't be parsed
            response_str = str(content_to_parse)
            # allowed_actions already retrieved above

            # Check if it looks like an agent trying to invoke another agent
            if "invoke_agent" in response_str and "summarizer_agent" in response_str:
                logger.warning("Response appears to invoke an agent but couldn't be parsed - likely JSON escaping issue")
                return ValidationResult(
                    is_valid=False,
                    error_message="JSON parsing failed - likely due to unescaped special characters in content",
                    retry_suggestion=f"Your JSON had unescaped special characters. Please ensure all quotes, newlines, and special characters are properly escaped. Valid actions: {', '.join(allowed_actions)}",
                    error_category=ValidationErrorCategory.FORMAT_ERROR.value
                )

            # Check if it's a tool response that needs processing
            elif "tool_calls" in response_str:
                return ValidationResult(
                    is_valid=False,
                    error_message="Tool response detected but couldn't be parsed",
                    retry_suggestion=f"Your tool response format was incorrect. Please ensure it follows proper JSON structure. Valid actions: {', '.join(allowed_actions)}",
                    error_category=ValidationErrorCategory.FORMAT_ERROR.value
                )
            
            # Generic fallback - provide dynamic format instructions based on allowed actions
            # Note: allowed_actions already retrieved above

            # Build clear guidance based on what's available
            guidance_parts = []

            # Tool calls (native format)
            if "tool_calls" in allowed_actions:
                guidance_parts.append("use native tool_calls format to call tools")

            # Agent invocations or final response (content field with next_action)
            content_actions = []
            if "invoke_agent" in allowed_actions:
                content_actions.append('"invoke_agent"')
            if "final_response" in allowed_actions:
                content_actions.append('"final_response"')

            if content_actions:
                actions_str = " or ".join(content_actions)
                guidance_parts.append(f'return content with "next_action" set to {actions_str} and proper "action_input"')

            # Combine into format hint
            if len(guidance_parts) == 2:
                format_hint = f"You can either: {guidance_parts[0]}, OR {guidance_parts[1]}"
            elif len(guidance_parts) == 1:
                format_hint = f"You must: {guidance_parts[0]}"
            else:
                format_hint = "Provide a valid response"

            return ValidationResult(
                is_valid=False,
                error_message="Could not parse response format - no processor could handle it",
                retry_suggestion=f'Your response format was not recognized. {format_hint}',
                error_category=ValidationErrorCategory.FORMAT_ERROR.value
            )
        
        # 2. Determine action type
        action_str = parsed.get("next_action")
        # Don't default to "continue" - require explicit action
        if not action_str:
            allowed_actions = self._get_allowed_actions(agent, branch)
            return ValidationResult(
                is_valid=False,
                error_message="No action specified in response",
                retry_suggestion=f"You didn't specify 'next_action'. You must choose one of: {', '.join(allowed_actions)}",
                error_category=ValidationErrorCategory.ACTION_ERROR.value
            )
        
        try:
            # Direct mapping without "continue" fallback
            action_type = ActionType(action_str)
        except ValueError:
            allowed_actions = self._get_allowed_actions(agent, branch)
            valid_action_types = []
            tool_calls_available = False

            for action in allowed_actions:
                if action == "invoke_agent":
                    valid_action_types.append(ActionType.INVOKE_AGENT.value)
                elif action == "tool_calls":
                    # Don't add "call_tool" to valid actions - tools are called via native response.tool_calls
                    # Just track that tools are available for the hint message
                    tool_calls_available = True
                elif action == "final_response":
                    valid_action_types.append(ActionType.FINAL_RESPONSE.value)

            # Build helpful error message
            error_msg = f"'{action_str}' is not a valid action."
            if valid_action_types:
                error_msg += f" Valid next_action values are: {valid_action_types}."
            if tool_calls_available:
                error_msg += " To use tools, return them in the native tool_calls field (not as next_action)."

            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown action type: {action_str}",
                retry_suggestion=error_msg,
                error_category=ValidationErrorCategory.ACTION_ERROR.value
            )
        
        # 3. Validate specific action
        if action_type in self.action_validators:
            validation_result = await self.action_validators[action_type](
                parsed, agent, branch, exec_state
            )
            
            # Add parsed response to result
            if validation_result.is_valid:
                validation_result.parsed_response = parsed
                # Only set action_type if validator didn't already set it
                if validation_result.action_type is None:
                    validation_result.action_type = action_type
            
            return validation_result
        
        # Default validation for unhandled action types
        return ValidationResult(
            is_valid=True,
            action_type=action_type,
            parsed_response=parsed
        )
    
    async def _validate_agent_invocation(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate agent invocation (single or multiple)."""
        # Get invocations directly from parsed response
        invocations = parsed.get("invocations", [])
        
        if not invocations:
            return ValidationResult(
                is_valid=False,
                error_message="Missing invocations for agent invocation",
                retry_suggestion="You indicated 'invoke_agent' but didn't specify which agent. You must provide the agent_name and request.",
                error_category=ValidationErrorCategory.FORMAT_ERROR.value
            )
        
        # Extract agent names from invocations for validation
        target_agents = [inv.agent_name for inv in invocations]
        
        # Check topology permissions
        next_agents = self.topology_graph.get_next_agents(agent.name)
        invalid_targets = []
        for target in target_agents:
            if target not in next_agents:
                invalid_targets.append(target)
        
        if invalid_targets:
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent {agent.name} cannot invoke: {invalid_targets}",
                retry_suggestion=f"You cannot invoke {invalid_targets}. Your available agents are: {next_agents}",
                error_category=ValidationErrorCategory.PERMISSION_ERROR.value
            )
        
        # Determine action type based on number of invocations
        # Multiple invocations -> PARALLEL_INVOKE, single invocation -> INVOKE_AGENT
        action_type = ActionType.PARALLEL_INVOKE if len(invocations) > 1 else ActionType.INVOKE_AGENT
        
        return ValidationResult(
            is_valid=True,
            invocations=invocations,  # Pass complete invocation data
            action_type=action_type
        )
    
    async def _validate_parallel_invocation(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate parallel agent invocation (DEPRECATED - use _validate_agent_invocation)."""
        # Get invocations from parsed response
        invocations = parsed.get("invocations", [])
        
        if not invocations or len(invocations) < 2:
            return ValidationResult(
                is_valid=False,
                error_message="Parallel invocation requires at least 2 invocations",
                retry_suggestion="Parallel execution requires at least 2 agents. Please specify multiple agents to run simultaneously."
            )
        
        # Extract agent names from invocations for validation
        target_agents = [inv.agent_name for inv in invocations]
        
        # Check topology permissions for each target
        next_agents = self.topology_graph.get_next_agents(agent.name)
        invalid_targets = []
        for target in target_agents:
            if target not in next_agents:
                invalid_targets.append(target)
        
        if invalid_targets:
            return ValidationResult(
                is_valid=False,
                error_message=f"Cannot invoke agents: {invalid_targets}",
                retry_suggestion=f"You cannot invoke {invalid_targets}. Available agents for parallel execution: {next_agents}"
            )
        
        # Check if parallel execution is allowed
        # (Could add rules here about which agents can be run in parallel)
        
        return ValidationResult(
            is_valid=True,
            action_type=ActionType.PARALLEL_INVOKE,
            invocations=invocations  # Pass complete invocation data
        )
    
    async def _validate_tool_call(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate tool call request."""
        tool_calls = parsed.get("tool_calls", [])
        
        # Allow empty tool_calls array (might be a valid case)
        if tool_calls is None:
            return ValidationResult(
                is_valid=False,
                error_message="No tool calls specified",
                retry_suggestion="You indicated a tool action but didn't provide the tool_calls array. You must specify which tools to call."
            )
        
        # Validate tool call format
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return ValidationResult(
                    is_valid=False,
                    error_message="Invalid tool call format",
                    retry_suggestion="Your tool call format is incorrect. Each tool call must have 'id' and 'function' fields."
                )
            
            if "function" not in tool_call:
                return ValidationResult(
                    is_valid=False,
                    error_message="Tool call missing function",
                    retry_suggestion="Your tool call is missing the 'function' field. Each tool call must include this field."
                )
        
        return ValidationResult(
            is_valid=True,
            tool_calls=tool_calls
        )
    
    async def _validate_final_response(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """
        Validate final_response action.
        
        Checks:
        1. Agent is allowed to return final response (connected to User)
        2. Response format is valid
        """
        # Check if agent can return final response
        if self.topology_graph:
            if not self.topology_graph.has_user_access(agent.name):
                # Get the agent's allowed next agents for better error message
                next_agents = self.topology_graph.get_next_agents(agent.name)
                
                logger.warning(
                    f"Agent '{agent.name}' attempted final_response but is not connected to User. "
                    f"Agent's next options: {next_agents}"
                )
                
                return ValidationResult(
                    is_valid=False,
                    error_message=(
                        f"Agent '{agent.name}' cannot use final_response action. "
                        f"Only agents with edges to User nodes can complete execution. "
                        f"This agent can invoke: {next_agents}"
                    ),
                    retry_suggestion=(
                        f"You cannot use 'final_response' from this agent. You must invoke one of: {next_agents}"
                    ),
                    error_category=ValidationErrorCategory.PERMISSION_ERROR.value
                )
        else:
            logger.debug(f"Agent '{agent.name}' is allowed to use final_response (has User access)")
        
        # Validate response format
        action_input = parsed.get("action_input", {})
        if not isinstance(action_input, dict):
            return ValidationResult(
                is_valid=False,
                error_message="final_response action_input must be a dictionary",
                retry_suggestion="Format your final_response like this: {\"action_input\": {\"response\": \"your answer\"}}"
            )
        
        if "response" not in action_input and "report" not in action_input:
            return ValidationResult(
                is_valid=False,
                error_message="final_response action_input must contain 'response' or 'report' field",
                retry_suggestion="Format your final_response like this: {\"action_input\": {\"response\": \"your answer\"}}"
            )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_end_conversation(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate conversation end."""
        # Check if we're in a conversation branch
        if branch.type.value != "conversation":
            allowed_actions = self._get_allowed_actions(agent, branch)
            if "final_response" in allowed_actions:
                retry_suggestion = "Use 'final_response' to complete the execution."
            else:
                next_agents = self.topology_graph.get_next_agents(agent.name)
                if next_agents:
                    retry_suggestion = f"Invoke one of these agents to continue: {next_agents}"
                else:
                    retry_suggestion = "No valid actions available. Check your allowed actions."
            
            return ValidationResult(
                is_valid=False,
                error_message="Cannot end conversation in non-conversation branch",
                retry_suggestion=retry_suggestion
            )
        
        return ValidationResult(is_valid=True)
    
    async def _validate_wait_aggregate(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate wait and aggregate request."""
        # This would check if there are active sub-branches to wait for
        # For now, always valid
        return ValidationResult(is_valid=True)

    async def _validate_error_recovery(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate error recovery action - routes to User node."""
        error_info = parsed.get("error_info", {})
        logger.debug(f"_validate_error_recovery received error_info: {error_info}")

        # Build suggested actions list
        suggested_actions = []

        # Include the original suggested_action if available
        if error_info.get("suggested_action"):
            suggested_actions.append(error_info["suggested_action"])

        # Add additional actions based on classification
        if error_info.get("classification") == "insufficient_credits":
            if "Add credits" not in str(error_info.get("suggested_action", "")):
                suggested_actions.append("Add credits to your account")
            suggested_actions.append("Then retry to continue from where you left off")

        # Set the suggested_actions list
        if suggested_actions:
            error_info["suggested_actions"] = suggested_actions

        # Create invocation for User node
        user_invocation = AgentInvocation(
            agent_name="User",
            request={
                "error_details": error_info,
                "retry_context": {
                    "agent_name": agent.name,
                    "branch_id": branch.id
                }
            }
        )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.ERROR_RECOVERY,
            invocations=[user_invocation],  # Add invocation for User
            parsed_response={
                **parsed,
                "target_agent": "User",  # Always route to User
                "error_details": error_info,
                "retry_context": {
                    "agent_name": agent.name,
                    "branch_id": branch.id
                }
            }
        )

    async def _validate_terminal_error(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate terminal error action - shows error then exits."""
        error_info = parsed.get("error_info", {})

        # Add termination reasons
        if error_info.get("classification") == "authentication_failed":
            error_info["termination_reason"] = (
                "Authentication failed. This requires updating your API key "
                "in the configuration file. Cannot be fixed while running."
            )

        # Create invocation for User node
        user_invocation = AgentInvocation(
            agent_name="User",
            request={
                "error_details": error_info
            }
        )

        return ValidationResult(
            is_valid=True,
            action_type=ActionType.TERMINAL_ERROR,
            invocations=[user_invocation],  # Add invocation for User
            parsed_response={
                **parsed,
                "target_agent": "User",  # Route to User for display
                "error_details": error_info
            }
        )

    def add_processor(self, processor: ResponseProcessor) -> None:
        """Add a custom response processor."""
        self.processors.append(processor)
        self.processors.sort(key=lambda p: p.priority(), reverse=True)
    
    def register_action_validator(
        self,
        action_type: ActionType,
        validator: callable
    ) -> None:
        """Register a custom action validator."""
        self.action_validators[action_type] = validator

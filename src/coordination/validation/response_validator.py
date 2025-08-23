"""
Response validator for centralized response processing.

This module is the CENTRAL hub for ALL response parsing in the coordination system.
It supports multiple response formats and validates actions based on topology permissions.
"""

from __future__ import annotations

import json
import re
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING
from enum import Enum

from ..branches.types import ExecutionBranch, StepResult, ExecutionState
from ..topology.graph import TopologyGraph
from .types import AgentInvocation, ValidationError

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


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    action_type: Optional[ActionType] = None
    parsed_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_suggestion: Optional[str] = None
    next_agents: List[str] = None  # For invoke/parallel_invoke
    tool_calls: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.next_agents is None:
            self.next_agents = []
        if self.tool_calls is None:
            self.tool_calls = []


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
        """Extract JSON from markdown code blocks or plain JSON."""
        # Try markdown blocks first (```json {...}``` or ```{...}```)
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)```'
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
                    return isinstance(data, dict) and "next_action" in data
                except json.JSONDecodeError:
                    return False
            
            # Try to extract first JSON if multiple JSONs present
            json_str = self._extract_first_json(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    return isinstance(data, dict) and "next_action" in data
                except json.JSONDecodeError:
                    return False
            
            # Finally try direct JSON parsing
            try:
                data = json.loads(response)
                return isinstance(data, dict) and "next_action" in data
            except json.JSONDecodeError:
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
                                # Create AgentInvocation with validation
                                invocation = AgentInvocation(
                                    agent_name=item["agent_name"],
                                    request=item.get("request", {}),
                                    instance_id=f"{item['agent_name']}_{idx}_{uuid.uuid4().hex[:8]}"
                                )
                                invocations.append(invocation)
                            else:
                                raise ValueError(f"Invalid invocation format at index {idx}: missing 'agent_name'")
                        
                        # Store validated invocations directly as list
                        result["invocations"] = invocations
                        result["target_agents"] = [inv.agent_name for inv in invocations]
                        
                        # For backward compatibility, also populate agent_requests
                        result["agent_requests"] = {}
                        for inv in invocations:
                            # Store by instance_id to avoid overwrites
                            result["agent_requests"][inv.instance_id] = inv.request
                        
                        # For backward compatibility with single agent flow
                        if len(result["target_agents"]) == 1:
                            result["target_agent"] = result["target_agents"][0]
                            result["action_input"] = invocations[0].request
                    
                    except Exception as e:
                        # Return detailed error to agent
                        return {
                            "next_action": "validation_error", 
                            "error": f"Failed to parse agent invocations: {str(e)}",
                            "retry_suggestion": "Ensure action_input is an array of objects with 'agent_name' and 'request' fields. Do not proceed with parallel invocations until this is fixed.",
                            "raw_response": data
                        }
                
                # Single dict format - convert to single-item list
                elif isinstance(action_input, dict):
                    if "agent_name" in action_input:
                        # Convert single dict to list format
                        invocation = AgentInvocation(
                            agent_name=action_input["agent_name"],
                            request=action_input.get("request", {}),
                            instance_id=f"{action_input['agent_name']}_0_{uuid.uuid4().hex[:8]}"
                        )
                        result["invocations"] = [invocation]
                        result["target_agents"] = [invocation.agent_name]
                        
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
                # Extract tool calls from action_input if needed
                action_input = data.get("action_input", {})
                if "tool_calls" in action_input:
                    tool_calls = action_input["tool_calls"]
                else:
                    tool_calls = data.get("tool_calls", [])
                
                # Clean up tool names - remove "functions." prefix if present
                cleaned_tool_calls = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        func_data = tool_call["function"]
                        if isinstance(func_data, dict) and "name" in func_data:
                            # Strip "functions." prefix if present
                            tool_name = func_data["name"]
                            if isinstance(tool_name, str) and tool_name.startswith("functions."):
                                func_data["name"] = tool_name.replace("functions.", "", 1)
                    cleaned_tool_calls.append(tool_call)
                
                result["tool_calls"] = cleaned_tool_calls
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
    """Handles responses with tool_calls array."""
    
    def can_process(self, response: Any) -> bool:
        """Check for tool_calls in response."""
        if isinstance(response, dict):
            return "tool_calls" in response and isinstance(response["tool_calls"], list)
        return False
    
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Convert to standard action format."""
        try:
            tool_calls = response.get("tool_calls", [])
            
            return {
                "next_action": "call_tool",
                "tool_calls": tool_calls,
                "content": response.get("content", ""),
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
                r"(?:final|complete|done|finished|concluded|my answer is)",
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


class ValidationProcessor:
    """Central hub for ALL response processing."""
    
    def __init__(self, topology_graph: TopologyGraph):
        self.topology_graph = topology_graph
        
        # Initialize processors in priority order
        self.processors: List[ResponseProcessor] = sorted([
            StructuredJSONProcessor(),
            ToolCallProcessor(),
            NaturalLanguageProcessor()
        ], key=lambda p: p.priority(), reverse=True)
        
        # Action validators
        self.action_validators = {
            ActionType.INVOKE_AGENT: self._validate_agent_invocation,
            ActionType.PARALLEL_INVOKE: self._validate_parallel_invocation,
            ActionType.CALL_TOOL: self._validate_tool_call,
            ActionType.FINAL_RESPONSE: self._validate_final_response,
            ActionType.END_CONVERSATION: self._validate_end_conversation,
            ActionType.WAIT_AND_AGGREGATE: self._validate_wait_aggregate
        }
    
    def _get_allowed_actions(self, agent: BaseAgent) -> List[str]:
        """Get list of actions this agent is allowed to perform based on topology."""
        allowed = []
        
        # Check if agent can invoke other agents
        next_agents = self.topology_graph.get_next_agents(agent.name)
        if next_agents:
            allowed.append("invoke_agent")
        
        # Check if agent has tools
        if hasattr(agent, 'tools') and agent.tools:
            allowed.append("call_tool")
        
        # Check if agent can return final response
        if self.topology_graph.has_user_access(agent.name):
            allowed.append("final_response")
        
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
        1. Tries each processor to parse the response
        2. Validates the extracted action
        3. Checks topology permissions
        4. Returns structured validation result
        """
        
        # Handle empty responses
        if raw_response is None:
            return ValidationResult(
                is_valid=False,
                error_message="Response is None",
                retry_suggestion="Please provide a response"
            )
        
        if isinstance(raw_response, str) and not raw_response.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Response is empty string",
                retry_suggestion="Please provide a non-empty response"
            )
        
        if isinstance(raw_response, dict) and not raw_response:
            return ValidationResult(
                is_valid=False,
                error_message="Response is empty dictionary",
                retry_suggestion="Please provide a response with content"
            )
        
        # 1. Try each processor to parse response
        parsed = None
        for processor in self.processors:
            if processor.can_process(raw_response):
                parsed = processor.process(raw_response)
                if parsed:
                    logger.debug(f"Response processed by {processor.__class__.__name__}")
                    break
        
        if not parsed:
            # Try to detect if this looks like a successful response that couldn't be parsed
            response_str = str(raw_response)
            
            # Check if it looks like an agent trying to invoke another agent
            if "invoke_agent" in response_str and "summarizer_agent" in response_str:
                logger.warning("Response appears to invoke an agent but couldn't be parsed - likely JSON escaping issue")
                return ValidationResult(
                    is_valid=False,
                    error_message="JSON parsing failed - likely due to unescaped special characters in content",
                    retry_suggestion="Please retry with properly escaped JSON. Ensure all quotes, newlines, and special characters in the content are properly escaped."
                )
            
            # Check if it's a tool response that needs processing
            elif "tool_calls" in response_str:
                return ValidationResult(
                    is_valid=False,
                    error_message="Tool response detected but couldn't be parsed",
                    retry_suggestion="Please ensure your tool response is properly formatted JSON"
                )
            
            # Generic fallback
            return ValidationResult(
                is_valid=False,
                error_message="Could not parse response format - no processor could handle it",
                retry_suggestion='Please respond with a valid JSON format: {"next_action": "invoke_agent", "action_input": [{"agent_name": "agent_name", "request": "your request"}]}'
            )
        
        # 2. Determine action type
        action_str = parsed.get("next_action")
        
        # Don't default to "continue" - require explicit action
        if not action_str:
            allowed_actions = self._get_allowed_actions(agent)
            return ValidationResult(
                is_valid=False,
                error_message="No action specified in response",
                retry_suggestion=f"Please specify 'next_action' with one of: {', '.join(allowed_actions)}"
            )
        
        try:
            # Direct mapping without "continue" fallback
            action_type = ActionType(action_str)
        except ValueError:
            allowed_actions = self._get_allowed_actions(agent)
            valid_action_types = []
            for action in allowed_actions:
                if action == "invoke_agent":
                    valid_action_types.append(ActionType.INVOKE_AGENT.value)
                elif action == "call_tool":
                    valid_action_types.append(ActionType.CALL_TOOL.value)
                elif action == "final_response":
                    valid_action_types.append(ActionType.FINAL_RESPONSE.value)
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown action type: {action_str}",
                retry_suggestion=f"Valid actions for {agent.name} are: {valid_action_types}"
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
        target_agents = parsed.get("target_agents", [])
        
        # Fall back to single target_agent for backward compatibility
        if not target_agents and "target_agent" in parsed:
            target_agents = [parsed["target_agent"]]
        
        if not target_agents:
            return ValidationResult(
                is_valid=False,
                error_message="Missing target agent(s) for invocation",
                retry_suggestion="Specify which agent(s) to invoke"
            )
        
        # Check topology permissions for all targets
        next_agents = self.topology_graph.get_next_agents(agent.name)
        invalid_targets = []
        for target in target_agents:
            if target not in next_agents:
                invalid_targets.append(target)
        
        if invalid_targets:
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent {agent.name} cannot invoke: {invalid_targets}",
                retry_suggestion=f"Valid targets: {next_agents}"
            )
        
        # Determine action type based on number of target agents
        # Multiple agents -> PARALLEL_INVOKE, single agent -> INVOKE_AGENT
        action_type = ActionType.PARALLEL_INVOKE if len(target_agents) > 1 else ActionType.INVOKE_AGENT
        
        return ValidationResult(
            is_valid=True,
            next_agents=target_agents,
            action_type=action_type
        )
    
    async def _validate_parallel_invocation(
        self,
        parsed: Dict[str, Any],
        agent: BaseAgent,
        branch: ExecutionBranch,
        exec_state: ExecutionState
    ) -> ValidationResult:
        """Validate parallel agent invocation (NEW!)."""
        target_agents = parsed.get("target_agents", [])
        
        if not target_agents or len(target_agents) < 2:
            return ValidationResult(
                is_valid=False,
                error_message="Parallel invocation requires at least 2 agents",
                retry_suggestion="Specify multiple agents to run in parallel"
            )
        
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
                retry_suggestion=f"Valid targets: {next_agents}"
            )
        
        # Check if parallel execution is allowed
        # (Could add rules here about which agents can be run in parallel)
        
        return ValidationResult(
            is_valid=True,
            next_agents=target_agents
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
                retry_suggestion="Provide tool calls in the correct format"
            )
        
        # Validate tool call format
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                return ValidationResult(
                    is_valid=False,
                    error_message="Invalid tool call format",
                    retry_suggestion="Tool calls must be dictionaries with 'id' and 'function'"
                )
            
            if "function" not in tool_call:
                return ValidationResult(
                    is_valid=False,
                    error_message="Tool call missing function",
                    retry_suggestion="Each tool call needs a 'function' field"
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
                        f"Use 'invoke_agent' to pass control to one of: {next_agents}, "
                        f"or 'call_tool' if you have tools available."
                    )
                )
        else:
            logger.debug(f"Agent '{agent.name}' is allowed to use final_response (has User access)")
        
        # Validate response format
        action_input = parsed.get("action_input", {})
        if not isinstance(action_input, dict):
            return ValidationResult(
                is_valid=False,
                error_message="final_response action_input must be a dictionary",
                retry_suggestion="Use format: {\"action_input\": {\"response\": \"your answer\"}}"
            )
        
        if "response" not in action_input and "report" not in action_input:
            return ValidationResult(
                is_valid=False,
                error_message="final_response action_input must contain 'response' or 'report' field",
                retry_suggestion="Use format: {\"action_input\": {\"response\": \"your answer\"}}"
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
            allowed_actions = self._get_allowed_actions(agent)
            if "final_response" in allowed_actions:
                retry_suggestion = "Use 'final_response' to complete execution"
            else:
                next_agents = self.topology_graph.get_next_agents(agent.name)
                if next_agents:
                    retry_suggestion = f"Use 'invoke_agent' to pass control to one of: {next_agents}"
                else:
                    retry_suggestion = "This agent cannot end conversation (no valid actions available)"
            
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

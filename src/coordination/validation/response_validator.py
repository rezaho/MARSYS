"""
Response validator for centralized response processing.

This module is the CENTRAL hub for ALL response parsing in the coordination system.
It supports multiple response formats and validates actions based on topology permissions.
"""

from __future__ import annotations

import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING
from enum import Enum

from ..branches.types import ExecutionBranch, StepResult, ExecutionState
from ..topology.graph import TopologyGraph

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
    
    def can_process(self, response: Any) -> bool:
        """Check for expected JSON structure."""
        if isinstance(response, dict):
            return "next_action" in response
        if isinstance(response, str):
            try:
                data = json.loads(response)
                return isinstance(data, dict) and "next_action" in data
            except:
                return False
        return False
    
    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON structure."""
        try:
            if isinstance(response, str):
                data = json.loads(response)
            else:
                data = response
            
            result = {
                "next_action": data.get("next_action"),
                "raw_response": data
            }
            
            # Handle different action types
            if data.get("next_action") == "invoke_agent":
                # Check for new format first
                if "target_agent" in data:
                    result["target_agent"] = data["target_agent"]
                    # Preserve action_input as data
                    result["action_input"] = data.get("action_input", {})
                else:
                    # Legacy format - action_input is agent name
                    result["target_agent"] = data.get("action_input")
                    result["action_input"] = ""  # No data in legacy format
            elif data.get("next_action") == "parallel_invoke":
                # NEW: Handle parallel invocation
                result["target_agents"] = data.get("agents", [])
                result["wait_for_all"] = data.get("wait_for_all", True)
            elif data.get("next_action") == "call_tool":
                result["tool_calls"] = data.get("tool_calls", [])
            elif data.get("next_action") == "final_response":
                result["final_response"] = data.get("final_response", data.get("content", ""))
            
            # Include all fields from original data
            result["content"] = data.get("content", "")
            # action_input is already handled above per action type
            
            # Preserve any additional fields
            for key, value in data.items():
                if key not in result:
                    result[key] = value
            return result
            
        except Exception as e:
            logger.error(f"Failed to process JSON response: {e}")
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
            
            # Default to continue execution
            return {
                "next_action": "continue",
                "content": text,
                "raw_response": response
            }
            
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
            return ValidationResult(
                is_valid=False,
                error_message="Could not parse response format",
                retry_suggestion="Please respond with a valid action format"
            )
        
        # 2. Determine action type
        action_str = parsed.get("next_action", "continue")
        try:
            # Map string to enum, with fallback
            if action_str == "continue":
                action_type = ActionType.FINAL_RESPONSE
            else:
                action_type = ActionType(action_str)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown action type: {action_str}",
                retry_suggestion=f"Valid actions are: {[a.value for a in ActionType]}"
            )
        
        # 3. Validate specific action
        if action_type in self.action_validators:
            validation_result = await self.action_validators[action_type](
                parsed, agent, branch, exec_state
            )
            
            # Add parsed response to result
            if validation_result.is_valid:
                validation_result.parsed_response = parsed
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
        """Validate single agent invocation."""
        target_agent = parsed.get("target_agent")
        
        if not target_agent:
            return ValidationResult(
                is_valid=False,
                error_message="Missing target agent for invocation",
                retry_suggestion="Specify which agent to invoke"
            )
        
        # Check topology permissions
        # Check if target_agent is in the next agents from current agent
        next_agents = self.topology_graph.get_next_agents(agent.name)
        if target_agent not in next_agents:
            return ValidationResult(
                is_valid=False,
                error_message=f"Agent {agent.name} cannot invoke {target_agent}",
                retry_suggestion=f"Valid targets: {next_agents}"
            )
        
        return ValidationResult(
            is_valid=True,
            next_agents=[target_agent]
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
        """Validate final response."""
        # Final responses are always valid
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
            return ValidationResult(
                is_valid=False,
                error_message="Cannot end conversation in non-conversation branch",
                retry_suggestion="Use 'final_response' to complete execution"
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
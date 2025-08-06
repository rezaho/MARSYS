"""
Step executor for stateless agent execution.

This module handles the execution of individual agent steps in a stateless manner,
separating the pure agent logic from memory management, tool execution, and other
infrastructure concerns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
from collections import defaultdict

from ...agents.memory import Message
from ..branches.types import StepResult
from .tool_executor import RealToolExecutor

if TYPE_CHECKING:
    from ...agents import BaseAgent
    from ..communication.user_node_handler import UserNodeHandler

logger = logging.getLogger(__name__)


@dataclass
class StepContext:
    """Context for a single step execution."""
    session_id: str
    branch_id: str
    step_number: int
    agent_name: str
    memory: List[Dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_request_context(self) -> Dict[str, Any]:
        """Convert to request context for agent.run_step()."""
        return {
            "task_id": self.session_id,
            "step_number": self.step_number,
            "branch_id": self.branch_id,
            "max_depth": 5,
            "max_interactions": 10,
            "depth": 0,
            "interaction_count": 0,
            **self.metadata
        }


@dataclass 
class ToolExecutionResult:
    """Result of executing tool calls."""
    success: bool
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    total_duration: float = 0.0


class StepExecutor:
    """
    Executes individual agent steps in a stateless manner.
    
    This component is responsible for:
    1. Preparing agent memory
    2. Calling the agent's pure _run() method
    3. Handling tool execution separately
    4. Managing retries and error handling
    """
    
    def __init__(
        self,
        tool_executor: Optional['RealToolExecutor'] = None,
        user_node_handler: Optional['UserNodeHandler'] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        # Use RealToolExecutor by default if none provided
        self.tool_executor = tool_executor or RealToolExecutor()
        self.user_node_handler = user_node_handler
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Track execution statistics
        self.stats = defaultdict(lambda: {
            "total_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "retried_steps": 0,
            "total_duration": 0.0,
            "tool_executions": 0
        })
    
    async def execute_step(
        self,
        agent: Union['BaseAgent', str],
        request: Any,
        memory: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> StepResult:
        """
        Execute a single agent step.
        
        Args:
            agent: The agent to execute
            request: The request/prompt for the agent
            memory: The agent's memory (conversation history)
            context: Additional execution context
            
        Returns:
            StepResult with execution outcome
        """
        start_time = time.time()
        
        # Check if this is a User node
        if isinstance(agent, str) and agent.lower() == "user":
            if self.user_node_handler and context.get("branch"):
                return await self.user_node_handler.handle_user_node(
                    branch=context["branch"],
                    incoming_message=request,
                    context=context
                )
            else:
                # No handler - skip User node
                return StepResult(
                    agent_name="User",
                    response="User interaction skipped",
                    action_type="skip",
                    success=True
                )
        
        # Normal agent handling
        agent_name = agent.name if hasattr(agent, 'name') else str(agent)
        
        # Create step context
        step_context = StepContext(
            session_id=context.get("session_id", ""),
            branch_id=context.get("branch_id", ""),
            step_number=context.get("step_number", 1),
            agent_name=agent_name,
            memory=[],  # Empty - not used for injection anymore
            tools_enabled=context.get("tools_enabled", True),
            metadata=context
        )
        
        logger.info(f"Executing step for agent '{agent_name}' "
                   f"(session: {step_context.session_id}, "
                   f"branch: {step_context.branch_id}, "
                   f"step: {step_context.step_number})")
        
        # Execute with retries
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute pure agent logic (agent maintains its own memory)
                agent_response = await agent.run_step(
                    request,
                    {"request_context": step_context.to_request_context()}
                )
                
                # Extract the actual response and context from the agent result
                if isinstance(agent_response, dict):
                    raw_response = agent_response.get('response', agent_response)
                    context_selection = agent_response.get('context_selection')
                else:
                    raw_response = agent_response
                    context_selection = None
                
                # 3. Process response
                step_result = await self._process_agent_response(
                    agent,
                    raw_response,
                    step_context,
                    context_selection
                )
                
                # 4. Handle tool execution if needed
                if step_result.tool_calls and self.tool_executor and step_context.tools_enabled:
                    tool_result = await self._execute_tools(
                        step_result.tool_calls,
                        agent,
                        step_context
                    )
                    
                    if not tool_result.success:
                        step_result.success = False
                        step_result.error = f"Tool execution failed: {tool_result.error}"
                        step_result.requires_retry = True
                    else:
                        # Add tool results to response
                        step_result.tool_results = tool_result.tool_results
                        self.stats[agent_name]["tool_executions"] += len(tool_result.tool_results)
                        
                        # NOTE: We do NOT re-run the agent here. The agent will continue in the next step
                        # to process the tool results. This maintains the one-call-per-step constraint.
                        
                        if tool_result.tool_results:
                            # Separate tool results by origin
                            response_origin_results = [tr for tr in tool_result.tool_results if tr.get('_origin') == 'response']
                            content_origin_results = [tr for tr in tool_result.tool_results if tr.get('_origin') == 'content']
                            
                            # Add tool results directly to agent's memory (source of truth)
                            if hasattr(agent, 'memory') and hasattr(agent.memory, 'add'):
                                # Process response-origin tools first (OpenAI requirement)
                                for tr in response_origin_results:
                                    agent.memory.add(
                                        role="tool",  # Correct role for native tool calls
                                        content=json.dumps(tr['result']),
                                        tool_call_id=tr['tool_call_id'],
                                        name=tr['tool_name']
                                    )
                                
                                # Process content-origin tools differently
                                if content_origin_results:
                                    # Add a bridging assistant message containing tool results
                                    tool_results_summary = []
                                    for tr in content_origin_results:
                                        tool_results_summary.append({
                                            "tool": tr['tool_name'],
                                            "result": tr['result']
                                        })
                                    
                                    bridge_content = {
                                        "tool_execution_results": tool_results_summary
                                    }
                                    
                                    agent.memory.add(
                                        role="assistant",
                                        content=json.dumps(bridge_content),
                                        name=agent.name
                                    )
                            
                            # Create tool messages for memory updates (for context sync later)
                            tool_messages = []
                            
                            # Add response-origin tool messages
                            for tr in response_origin_results:
                                tool_messages.append({
                                    "role": "tool",
                                    "content": json.dumps(tr['result']),
                                    "tool_call_id": tr['tool_call_id'],
                                    "name": tr['tool_name']
                                })
                            
                            # Add bridge message for content-origin tools
                            if content_origin_results:
                                tool_messages.append({
                                    "role": "assistant",
                                    "content": json.dumps(bridge_content),
                                    "name": agent.name
                                })
                            
                            # Add tool responses to step result memory updates
                            # This ensures they propagate to branch memory
                            if hasattr(step_result, 'memory_updates') and step_result.memory_updates:
                                step_result.memory_updates.extend(tool_messages)
                            else:
                                # This case shouldn't happen, but handle it safely
                                step_result.memory_updates = tool_messages.copy()
                            
                            # Mark that agent needs to continue processing tool results
                            step_result.next_agent = agent_name  # Continue with same agent
                            if not hasattr(step_result, 'metadata'):
                                step_result.metadata = {}
                            step_result.metadata['tool_continuation'] = True
                            step_result.metadata['has_tool_results'] = True
                            
                            logger.info(f"Agent '{agent_name}' executed {len(tool_result.tool_results)} tools - will continue in next step")
                
                # Update statistics
                duration = time.time() - start_time
                self._update_stats(agent_name, step_result, retry_count, duration)
                
                # Sync agent memory to context (agent is source of truth)
                if step_result.success and hasattr(agent, 'memory') and hasattr(agent.memory, 'retrieve_all'):
                    # Get full agent memory state
                    agent_memory_state = agent.memory.retrieve_all()
                    
                    # Update step result with complete memory state
                    step_result.memory_updates = agent_memory_state
                    
                    # Log the sync
                    logger.debug(f"Synced {len(agent_memory_state)} messages from agent memory to context")
                
                return step_result
                
            except Exception as e:
                logger.error(f"Error executing agent '{agent_name}' "
                           f"(retry {retry_count}/{self.max_retries}): {e}")
                last_error = str(e)
                retry_count += 1
                
                if retry_count <= self.max_retries:
                    await asyncio.sleep(self.retry_delay * retry_count)
        
        # All retries exhausted
        duration = time.time() - start_time
        failed_result = StepResult(
            agent_name=agent_name,
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            requires_retry=False  # Don't retry further
        )
        
        self._update_stats(agent_name, failed_result, retry_count, duration)
        return failed_result
    
    
    async def _process_agent_response(
        self,
        agent: BaseAgent,
        raw_response: Any,
        context: StepContext,
        context_selection: Optional[Dict[str, Any]] = None
    ) -> StepResult:
        """
        Process the raw agent response into a StepResult.
        
        This handles different response formats and extracts relevant information.
        """
        # Start with basic result
        result = StepResult(
            agent_name=context.agent_name,
            success=True,
            response=raw_response
        )
        
        # Handle different response types
        if isinstance(raw_response, dict):
            # Check for error responses
            if raw_response.get("error"):
                result.success = False
                result.error = raw_response.get("error")
                result.requires_retry = raw_response.get("retry", True)
                return result
            
            # Extract tool calls if present
            if "tool_calls" in raw_response:
                result.tool_calls = raw_response["tool_calls"]
                # Mark these as response-origin tool calls
                if isinstance(result.tool_calls, list):
                    for tc in result.tool_calls:
                        if isinstance(tc, dict):
                            tc['_origin'] = 'response'
            
            # Extract next action if present
            if "next_action" in raw_response:
                result.action_type = raw_response["next_action"]
                
                # Handle call_tool action
                if result.action_type == "call_tool" and "action_input" in raw_response:
                    action_input = raw_response["action_input"]
                    if isinstance(action_input, dict) and "tool_calls" in action_input:
                        # Merge tool calls from action_input with any existing tool calls
                        content_tool_calls = action_input["tool_calls"]
                        # Mark content-based tool calls with their origin
                        for tc in content_tool_calls:
                            if isinstance(tc, dict):
                                tc['_origin'] = 'content'
                        
                        if result.tool_calls:
                            # Convert to list if needed and merge
                            existing = result.tool_calls if isinstance(result.tool_calls, list) else [result.tool_calls]
                            result.tool_calls = existing + content_tool_calls
                        else:
                            result.tool_calls = content_tool_calls
                
                # Extract next agent for invoke_agent action
                elif result.action_type == "invoke_agent" and "action_input" in raw_response:
                    action_input = raw_response["action_input"]
                    
                    # Handle unified array format
                    if isinstance(action_input, list) and len(action_input) > 0:
                        # For single agent, extract the first one
                        first_agent = action_input[0]
                        if isinstance(first_agent, dict) and "agent_name" in first_agent:
                            result.next_agent = first_agent["agent_name"]
                        else:
                            # Fallback if array contains strings
                            result.next_agent = str(first_agent)
                    # Handle legacy format (string or dict)
                    elif isinstance(action_input, dict):
                        # Enhanced format with agent_name
                        if "agent_name" in action_input:
                            result.next_agent = action_input["agent_name"]
                        else:
                            # Very old format where action_input is agent name
                            result.next_agent = str(action_input)
                    else:
                        # Legacy format - action_input is agent name
                        result.next_agent = str(action_input)
            
            # Store parsed response
            result.parsed_response = raw_response
            
            # Store context selection if available
            if context_selection:
                result.context_selection = context_selection
            
        elif isinstance(raw_response, Message):
            # Handle Message objects from agents
            result.response = raw_response.content
            if raw_response.tool_calls:
                result.tool_calls = raw_response.tool_calls
                # Mark these as response-origin since they came from the agent's response
                if isinstance(result.tool_calls, list):
                    for tc in result.tool_calls:
                        if isinstance(tc, dict) and '_origin' not in tc:
                            tc['_origin'] = 'response'
                
                # ENSURE parsed_response exists for tool calls with proper structure
                if not result.parsed_response:
                    result.parsed_response = {
                        'next_action': 'call_tool',  # Add required next_action field
                        'tool_calls': result.tool_calls,
                        '_has_tool_calls': True,
                        'action_input': {
                            'tool_calls': result.tool_calls
                        }
                    }
            
            # Try to parse JSON content if it's a string that looks like JSON
            if isinstance(raw_response.content, str):
                content = raw_response.content.strip()
                
                # Check for markdown code blocks
                if content.startswith('```'):
                    # Extract JSON from markdown
                    import re
                    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1).strip()
                
                # Try to parse as JSON
                if content.startswith('{') and content.endswith('}'):
                    try:
                        parsed = json.loads(content)
                        
                        # Extract structured action information
                        if isinstance(parsed, dict) and "next_action" in parsed:
                            result.action_type = parsed["next_action"]
                            result.parsed_response = parsed
                            
                            # Handle call_tool action
                            if parsed["next_action"] == "call_tool" and "action_input" in parsed:
                                action_input = parsed["action_input"]
                                if isinstance(action_input, dict) and "tool_calls" in action_input:
                                    # Merge tool calls from content with existing Message.tool_calls
                                    content_tool_calls = action_input["tool_calls"]
                                    # Mark content-based tool calls with their origin
                                    for tc in content_tool_calls:
                                        if isinstance(tc, dict):
                                            tc['_origin'] = 'content'
                                    
                                    if result.tool_calls:
                                        # Convert to list if needed and merge
                                        existing = result.tool_calls if isinstance(result.tool_calls, list) else [result.tool_calls]
                                        # Mark existing tool calls (from response.tool_calls) with their origin
                                        for tc in existing:
                                            if isinstance(tc, dict) and '_origin' not in tc:
                                                tc['_origin'] = 'response'
                                        result.tool_calls = existing + content_tool_calls
                                    else:
                                        result.tool_calls = content_tool_calls
                            
                            # Update the agent's memory with merged tool_calls
                            if result.tool_calls and hasattr(agent, 'memory') and hasattr(agent.memory, 'memory'):
                                # Convert any dict tool calls to ToolCallMsg objects
                                from src.agents.memory import ToolCallMsg
                                normalized_tool_calls = []
                                for tc in result.tool_calls:
                                    if isinstance(tc, dict):
                                        normalized_tool_calls.append(ToolCallMsg.from_dict(tc))
                                    else:
                                        normalized_tool_calls.append(tc)
                                
                                # Find the last assistant message in agent's memory
                                for i in range(len(agent.memory.memory) - 1, -1, -1):
                                    msg = agent.memory.memory[i]
                                    if hasattr(msg, 'role') and msg.role == 'assistant':
                                        # Update its tool_calls with the normalized list
                                        msg.tool_calls = normalized_tool_calls
                                        break
                            
                            # Handle invoke_agent action
                            elif parsed["next_action"] == "invoke_agent" and "action_input" in parsed:
                                action_input = parsed["action_input"]
                                if isinstance(action_input, dict) and "agent_name" in action_input:
                                    result.next_agent = action_input["agent_name"]
                                elif isinstance(action_input, str):
                                    result.next_agent = action_input
                                    
                    except json.JSONDecodeError:
                        # Not valid JSON, keep as string
                        pass
            
            # Create memory update
            # IMPORTANT: Only include response-origin tool calls in the message's tool_calls field
            # Content-origin tool calls will be handled separately to avoid OpenAI API errors
            response_origin_tool_calls = None
            if result.tool_calls:
                # Separate tool calls by origin
                response_origin_tool_calls = [
                    tc for tc in result.tool_calls 
                    if isinstance(tc, dict) and tc.get('_origin') == 'response'
                ]
                # If no response-origin tool calls, don't include tool_calls field at all
                if not response_origin_tool_calls:
                    response_origin_tool_calls = None
            
            memory_update = {
                "role": raw_response.role,
                "content": raw_response.content,
                "name": raw_response.name or context.agent_name,
                "timestamp": time.time()
            }
            
            # Only add tool_calls if we have response-origin ones
            if response_origin_tool_calls:
                memory_update["tool_calls"] = response_origin_tool_calls
                
            result.memory_updates = [memory_update]
        
        else:
            # Handle string or other responses
            result.response = str(raw_response)
            result.memory_updates = [{
                "role": "assistant",
                "content": str(raw_response),
                "name": context.agent_name,
                "timestamp": time.time()
            }]
        
        # TOOL CONTINUATION RULE:
        # When an agent returns tool_calls, the agent MUST continue in the next step
        # This applies regardless of whether next_agent is specified
        if result.tool_calls and not result.next_agent:
            result.next_agent = context.agent_name
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata['tool_continuation'] = True
            result.metadata['has_tool_calls'] = True
            
            # Ensure parsed_response has continuation markers for branch spawner
            if not result.parsed_response:
                result.parsed_response = {}
            if isinstance(result.parsed_response, dict):
                result.parsed_response['_tool_continuation'] = True
                result.parsed_response['_pending_tools_count'] = len(result.tool_calls)
            logger.info(f"Agent '{context.agent_name}' executed {len(result.tool_calls)} tools - will continue in next step")
        
        # INVALID RESPONSE HANDLING:
        # Special case: content=None with tool_calls is valid
        if result.tool_calls and (result.response is None or result.response == "None" or result.response == ""):
            # This is valid - agent wants to execute tools without additional content
            # Tool continuation is already handled above
            pass
        # If no action type, no tools, and no next agent specified, this is an invalid response
        elif not result.action_type and not result.tool_calls and not result.next_agent:
            result.next_agent = context.agent_name  # Stay with same agent
            result.requires_retry = True
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata['invalid_response'] = True
            result.metadata['retry_reason'] = 'format_error'
            
            # Generate format instructions
            format_instructions = """
Your response must be a JSON object with one of these formats:

1. To invoke another agent:
{"next_action": "invoke_agent", "action_input": {"agent_name": "AgentName", "request": "your request"}}

2. To call tools:
{"next_action": "call_tool", "action_input": {"tool_calls": [...]}}

3. To provide final response:
{"next_action": "final_response", "action_input": {"response": "your final answer"}}

Do not provide thoughts or explanations outside this JSON structure."""
            
            # Try to get agent-specific format instructions if available
            if hasattr(agent, '_get_response_guidelines'):
                try:
                    format_instructions = agent._get_response_guidelines()
                except:
                    pass  # Use default format
            elif hasattr(agent, '_get_unified_response_format'):
                try:
                    format_instructions = agent._get_unified_response_format()
                except:
                    pass  # Use default format
            
            result.error = f"Invalid response format. {format_instructions}"
            logger.warning(f"Agent '{context.agent_name}' provided invalid response - will retry with format instructions")
        
        return result
    
    async def _execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        agent: BaseAgent,
        context: StepContext
    ) -> ToolExecutionResult:
        """
        Execute tool calls separately from agent logic.
        
        This maintains the separation of concerns where agents don't
        directly execute tools.
        """
        if not self.tool_executor:
            return ToolExecutionResult(
                success=False,
                error="No tool executor configured"
            )
        
        start_time = time.time()
        
        try:
            # Execute tools
            tool_results = await self.tool_executor.execute_tools(
                tool_calls=tool_calls,
                agent=agent,
                context={
                    "session_id": context.session_id,
                    "branch_id": context.branch_id,
                    "agent_name": context.agent_name
                }
            )
            
            duration = time.time() - start_time
            
            return ToolExecutionResult(
                success=True,
                tool_results=tool_results,
                total_duration=duration
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolExecutionResult(
                success=False,
                error=str(e),
                total_duration=time.time() - start_time
            )
    
    def _update_stats(
        self,
        agent_name: str,
        result: StepResult,
        retry_count: int,
        duration: float
    ) -> None:
        """Update execution statistics."""
        stats = self.stats[agent_name]
        stats["total_steps"] += 1
        
        if result.success:
            stats["successful_steps"] += 1
        else:
            stats["failed_steps"] += 1
        
        if retry_count > 0:
            stats["retried_steps"] += 1
        
        stats["total_duration"] += duration
    
    def get_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Args:
            agent_name: Specific agent to get stats for, or None for all
            
        Returns:
            Statistics dictionary
        """
        if agent_name:
            return dict(self.stats.get(agent_name, {}))
        return dict(self.stats)
    
    async def execute_parallel_steps(
        self,
        steps: List[Tuple[BaseAgent, Any, List[Dict[str, Any]], Dict[str, Any]]]
    ) -> List[StepResult]:
        """
        Execute multiple agent steps in parallel.
        
        Args:
            steps: List of (agent, request, memory, context) tuples
            
        Returns:
            List of StepResults in the same order as input
        """
        tasks = []
        for agent, request, memory, context in steps:
            task = self.execute_step(agent, request, memory, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to StepResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_name = steps[i][0].name
                processed_results.append(StepResult(
                    agent_name=agent_name,
                    success=False,
                    error=f"Parallel execution failed: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
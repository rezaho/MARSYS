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
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from collections import defaultdict

from ...agents.memory import Message
from ..branches.types import StepResult
from .tool_executor import RealToolExecutor

if TYPE_CHECKING:
    from ...agents import BaseAgent

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
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        # Use RealToolExecutor by default if none provided
        self.tool_executor = tool_executor or RealToolExecutor()
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
        agent: BaseAgent,
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
        agent_name = agent.name
        
        # Create step context
        step_context = StepContext(
            session_id=context.get("session_id", ""),
            branch_id=context.get("branch_id", ""),
            step_number=context.get("step_number", 1),
            agent_name=agent_name,
            memory=memory,
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
                # 1. Prepare agent memory (inject current memory state)
                await self._prepare_agent_memory(agent, step_context)
                
                # 2. Execute pure agent logic
                agent_response = await agent.run_step(
                    request,
                    {"request_context": step_context.to_request_context()}
                )
                
                # Extract the actual response from the agent result
                if isinstance(agent_response, dict) and 'response' in agent_response:
                    raw_response = agent_response['response']
                else:
                    raw_response = agent_response
                
                # 3. Process response
                step_result = await self._process_agent_response(
                    agent,
                    raw_response,
                    step_context
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
                        
                        # IMPORTANT: Re-run agent to process tool results
                        # This allows the agent to generate a final response based on tool data
                        if tool_result.tool_results:
                            # Add tool results to context memory as tool messages
                            tool_messages = []
                            for tr in tool_result.tool_results:
                                tool_messages.append({
                                    "role": "tool",
                                    "content": json.dumps(tr['result']),
                                    "tool_call_id": tr['tool_call_id'],
                                    "name": tr['tool_name']
                                })
                            
                            # Update context memory with tool results
                            step_context.memory.extend(tool_messages)
                            
                            # Re-run agent with updated memory
                            logger.info(f"Re-running agent '{agent_name}' with {len(tool_result.tool_results)} tool results")
                            
                            # Prepare agent memory again with tool results
                            await self._prepare_agent_memory(agent, step_context)
                            
                            # Create a prompt to process tool results
                            process_prompt = (
                                "Based on the tool results above, provide your analysis and final response. "
                                "Synthesize the information from the tools into a comprehensive answer."
                            )
                            
                            # Execute agent again to process tool results
                            try:
                                final_response = await agent.run_step(
                                    process_prompt,
                                    {"request_context": step_context.to_request_context()}
                                )
                                
                                # Update step result with final response
                                if isinstance(final_response, dict) and 'response' in final_response:
                                    # Extract the actual response from agent result
                                    final_raw_response = final_response['response']
                                else:
                                    final_raw_response = final_response
                                
                                # Process the final response
                                final_step_result = await self._process_agent_response(
                                    agent,
                                    final_raw_response,
                                    step_context
                                )
                                
                                # Update the original step result with the final processed response
                                step_result.response = final_step_result.response
                                step_result.parsed_response = final_step_result.parsed_response
                                step_result.action_type = final_step_result.action_type
                                step_result.next_agent = final_step_result.next_agent
                                
                                # Add final response to memory updates
                                if final_step_result.memory_updates:
                                    if not step_result.memory_updates:
                                        step_result.memory_updates = []
                                    step_result.memory_updates.extend(final_step_result.memory_updates)
                                    
                                logger.info(f"Agent '{agent_name}' successfully processed tool results")
                                
                            except Exception as e:
                                logger.error(f"Error re-running agent '{agent_name}' with tool results: {e}")
                                # Keep the original response with tool results
                
                # Update statistics
                duration = time.time() - start_time
                self._update_stats(agent_name, step_result, retry_count, duration)
                
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
    
    async def _prepare_agent_memory(
        self,
        agent: BaseAgent,
        context: StepContext
    ) -> None:
        """
        Prepare agent memory for execution.
        
        This injects the current memory state into the agent's memory system
        without the agent knowing about it (maintaining pure _run()).
        """
        if hasattr(agent, 'memory') and hasattr(agent.memory, 'memory'):
            # Clear existing memory and repopulate
            if hasattr(agent.memory, 'reset_memory'):
                agent.memory.reset_memory()
            
            # Convert memory format and add to agent's memory
            for msg in context.memory:
                if isinstance(msg, dict):
                    # Convert dict to Message object
                    message = Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        name=msg.get("name"),
                        tool_calls=msg.get("tool_calls")
                    )
                    
                    # Add to memory using the add_message method
                    if hasattr(agent.memory, 'add_message'):
                        agent.memory.add_message(message)
                    elif hasattr(agent.memory, 'memory') and isinstance(agent.memory.memory, list):
                        # Direct append if no add_message method
                        agent.memory.memory.append(message)
            
            logger.debug(f"Prepared {len(context.memory)} messages for agent '{context.agent_name}'")
    
    async def _process_agent_response(
        self,
        agent: BaseAgent,
        raw_response: Any,
        context: StepContext
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
            
            # Extract next action if present
            if "next_action" in raw_response:
                result.action_type = raw_response["next_action"]
                
                # Handle call_tool action
                if result.action_type == "call_tool" and "action_input" in raw_response:
                    action_input = raw_response["action_input"]
                    if isinstance(action_input, dict) and "tool_calls" in action_input:
                        result.tool_calls = action_input["tool_calls"]
                
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
            
        elif isinstance(raw_response, Message):
            # Handle Message objects from agents
            result.response = raw_response.content
            if raw_response.tool_calls:
                result.tool_calls = raw_response.tool_calls
            
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
                                    result.tool_calls = action_input["tool_calls"]
                            
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
            result.memory_updates = [{
                "role": raw_response.role,
                "content": raw_response.content,
                "name": raw_response.name or context.agent_name,
                "tool_calls": raw_response.tool_calls,
                "timestamp": time.time()
            }]
        
        else:
            # Handle string or other responses
            result.response = str(raw_response)
            result.memory_updates = [{
                "role": "assistant",
                "content": str(raw_response),
                "name": context.agent_name,
                "timestamp": time.time()
            }]
        
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
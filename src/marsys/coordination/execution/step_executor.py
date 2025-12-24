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
from ..branches.types import StepResult, ExecutionBranch
from .tool_executor import RealToolExecutor
from ..steering.manager import SteeringManager, SteeringContext, ErrorContext
from ..validation.types import ValidationErrorCategory
from ..formats.context import CoordinationContext

# Enhanced error handling imports
from ...agents.exceptions import (
    # Base error
    AgentFrameworkError,

    # Model/API errors
    ModelAPIError,

    # Message errors (includes ToolCallError)
    ToolCallError,

    # Resource errors
    PoolExhaustedError,
    TimeoutError as FrameworkTimeoutError,  # Avoid conflict with asyncio.TimeoutError
    ResourceLimitError,
    QuotaExceededError,

    # Coordination errors
    TopologyError,
    RoutingError,
    BranchExecutionError,

    # Agent errors
    AgentConfigurationError,
    AgentPermissionError,
    AgentLimitError,

    # Browser errors
    VisionAgentNotConfiguredError,

    # State errors
    StateError,
    SessionNotFoundError,

    # Utility
    create_error_from_exception
)

if TYPE_CHECKING:
    from ...agents import BaseAgent
    from ..communication.user_node_handler import UserNodeHandler
    from ..topology import TopologyGraph
    from ..event_bus import EventBus
    from ..formats import SystemPromptBuilder

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
    branch: Optional['ExecutionBranch'] = None  # FIX 3: Added for dynamic instructions
    topology_graph: Optional['TopologyGraph'] = None  # FIX 3: Added for dynamic instructions
    
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
        event_bus: Optional['EventBus'] = None,
        system_prompt_builder: Optional['SystemPromptBuilder'] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0
    ):
        # Pass event_bus to tool executor
        self.tool_executor = tool_executor or RealToolExecutor(event_bus=event_bus)
        self.user_node_handler = user_node_handler
        self.event_bus = event_bus
        self.system_prompt_builder = system_prompt_builder
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize steering manager
        self.steering_manager = SteeringManager()

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

        # Emit start event if event_bus is available
        if self.event_bus:
            from ..status.events import AgentStartEvent

            # Create request summary
            request_summary = str(request)[:100] if request else None

            await self.event_bus.emit(AgentStartEvent(
                session_id=context.get("session_id", "unknown"),
                branch_id=context.get("branch_id"),
                agent_name=agent_name,
                request_summary=request_summary
            ))

        # Create step context
        step_context = StepContext(
            session_id=context.get("session_id", ""),
            branch_id=context.get("branch_id", ""),
            step_number=context.get("step_number", 1),
            agent_name=agent_name,
            memory=[],  # Empty - not used for injection anymore
            tools_enabled=context.get("tools_enabled", True),
            metadata=context,
            branch=context.get("branch"),  # FIX 3: Pass branch for dynamic instructions
            topology_graph=context.get("topology_graph")  # FIX 3: Pass topology for dynamic instructions
        )
        
        logger.info(f"Executing step for agent '{agent_name}' "
                   f"(session: {step_context.session_id}, "
                   f"branch: {step_context.branch_id}, "
                   f"step: {step_context.step_number})")
        
        # Generate dynamic format instructions if we have topology and branch
        topology_graph = context.get("topology_graph")
        branch = context.get("branch")
        
        # Extract execution config from context
        execution_config = context.get('execution_config')
        if not execution_config:
            from ..config import ExecutionConfig
            execution_config = ExecutionConfig()  # Default: auto mode
        
        # Execute with retries
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                # Execute pure agent logic (agent maintains its own memory)
                run_context = {"request_context": step_context.to_request_context()}

                # Mark as continuation if this is a tool continuation
                if context.get('tool_continuation'):
                    run_context["is_continuation"] = True

                # Mark as continuation if this is a step-executor level retry (framework errors)
                if retry_count > 0:
                    run_context["is_continuation"] = True

                # Mark as continuation if this is an agent-level retry (validation errors from branch_executor)
                agent_retry_count = context.get('metadata', {}).get('agent_retry_count', 0)
                if agent_retry_count > 0:
                    run_context["is_continuation"] = True

                # Build coordination context for the new formats architecture
                coordination_context = self._build_coordination_context(
                    agent_name,
                    topology_graph
                )

                # Add coordination context and system prompt builder to run_context
                run_context["coordination_context"] = coordination_context
                if self.system_prompt_builder:
                    run_context["system_prompt_builder"] = self.system_prompt_builder

                # Add steering based on config using SteeringManager
                agent_retry_count = context.get('metadata', {}).get('agent_retry_count', 0)
                agent_error_context_dict = context.get('metadata', {}).get('agent_error_context')

                # Build ErrorContext if present
                error_context = None
                if agent_error_context_dict:
                    try:
                        error_context = ErrorContext(
                            category=ValidationErrorCategory(agent_error_context_dict['category']),
                            error_message=agent_error_context_dict['error_message'],
                            retry_suggestion=agent_error_context_dict.get('retry_suggestion'),
                            retry_count=agent_error_context_dict['retry_count'],
                            classification=agent_error_context_dict.get('classification'),
                            failed_action=agent_error_context_dict.get('failed_action')
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Failed to parse error context for {agent_name}: {e}")

                # Determine retry flags
                is_retry = (retry_count > 0) or (agent_retry_count > 0)
                has_error = error_context is not None

                # Check if we should apply steering
                if execution_config.should_apply_steering(is_retry, has_error):
                    # Get available actions from topology
                    available_actions = self._get_available_actions(agent, step_context)

                    # Build steering context
                    steering_ctx = SteeringContext(
                        agent_name=agent_name,
                        available_actions=available_actions,
                        error_context=error_context,
                        is_retry=is_retry,
                        steering_mode=execution_config.steering_mode
                    )

                    # Get steering prompt from manager
                    steering_prompt = self.steering_manager.get_steering_prompt(steering_ctx)

                    if steering_prompt:
                        run_context["steering_prompt"] = steering_prompt
                        category_str = error_context.category.value if error_context else "none"
                        logger.debug(
                            f"Injected steering for {agent_name} "
                            f"(mode: {execution_config.steering_mode}, category: {category_str}, "
                            f"retry: {agent_retry_count})"
                        )
                
                agent_response = await agent.run_step(
                    request,
                    run_context
                )
                
                # Extract the actual response and context from the agent result
                if isinstance(agent_response, dict):
                    raw_response = agent_response.get('response', agent_response)
                    context_selection = agent_response.get('context_selection')
                else:
                    raw_response = agent_response
                    context_selection = None

                # Emit thinking event if there's a thought
                if self.event_bus and isinstance(raw_response, dict):
                    thought = raw_response.get("thought") or raw_response.get("reasoning")
                    if thought:
                        from ..status.events import AgentThinkingEvent

                        await self.event_bus.emit(AgentThinkingEvent(
                            session_id=context.get("session_id", "unknown"),
                            branch_id=context.get("branch_id"),
                            agent_name=agent_name,
                            thought=thought,
                            action_type=raw_response.get("next_action")
                        ))

                # 3. Process tool calls only
                step_result = await self._process_tool_calls(
                    raw_response,
                    agent,
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
                        
                        # NOTE: We do NOT re-run the agent here. The agent will continue in the next step
                        # to process the tool results. This maintains the one-call-per-step constraint.
                        
                        if tool_result.tool_results:
                            # Add tool results to agent's memory (source of truth)
                            # CRITICAL: OpenAI API requires ALL tool messages (role="tool") to come immediately
                            # after the assistant message with tool_calls, BEFORE any user messages.
                            # We must batch: 1) all tool messages first, 2) then all user messages
                            if hasattr(agent, 'memory') and hasattr(agent.memory, 'add'):
                                # Phase 1: Collect tool messages and user messages separately
                                tool_messages = []  # Messages with role="tool"
                                user_messages = []  # Messages with role="user" for multimodal/array content

                                for tr in tool_result.tool_results:
                                    origin = tr.get('_origin', 'response')
                                    result = tr['result']

                                    if origin == 'response':
                                        # Response-origin: Standard OpenAI format with role="tool"

                                        # Check if result is ToolResponse object
                                        from marsys.environment.tool_response import ToolResponse

                                        if isinstance(result, ToolResponse):
                                            # Two-message pattern for ToolResponse:
                                            # 1. Tool message (role="tool"): metadata/status
                                            # 2. User message (role="user"): actual content from to_content_array()

                                            # Message 1: Tool message with metadata
                                            tool_content = result.get_metadata_str()
                                            tool_messages.append({
                                                'role': 'tool',
                                                'content': tool_content,
                                                'tool_call_id': tr['tool_call_id'],
                                                'name': tr['tool_name']
                                            })

                                            # Message 2: User message with actual content
                                            # Convert ToolResponse to content array (typed array format)
                                            content_array = result.to_content_array()

                                            # If content is a list (multimodal), prepend reference block
                                            if isinstance(content_array, list):
                                                reference_block = {
                                                    "type": "text",
                                                    "text": f"[Tool response content for tool_call_id: {tr['tool_call_id']}]"
                                                }
                                                content_array = [reference_block] + content_array

                                            user_messages.append({
                                                'role': 'user',
                                                'content': content_array
                                            })
                                        else:
                                            # Single-message pattern for string results
                                            # Result is already stringified by tool_executor
                                            tool_messages.append({
                                                'role': 'tool',
                                                'content': str(result),  # Ensure it's a string
                                                'tool_call_id': tr['tool_call_id'],
                                                'name': tr['tool_name']
                                            })
                                    else:  # origin == 'content'
                                        # Content-origin: Must use role="user" (OpenAI API restriction)
                                        # tr['result'] is already formatted by to_content_array()
                                        user_messages.append({
                                            'role': 'user',
                                            'content': tr['result']
                                        })

                                # Phase 2: Add all tool messages first (MUST come immediately after tool_calls)
                                for tool_msg in tool_messages:
                                    agent.memory.add(**tool_msg)

                                # Phase 3: Add all user messages second
                                for user_msg in user_messages:
                                    agent.memory.add(**user_msg)

                            # Mark that agent needs to continue processing tool results
                            # BranchExecutor will set next_agent based on this metadata
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
                    agent_memory_state = agent.memory.get_messages()
                    
                    # Update step result with complete memory state
                    step_result.memory_updates = agent_memory_state
                    
                    # Log the sync
                    logger.debug(f"Synced {len(agent_memory_state)} messages from agent memory to context")
                
                # FIX: Extract saved context for passing to next agent
                if hasattr(agent, '_context_selector'):
                    messages = agent.memory.get_messages() if hasattr(agent, 'memory') else []
                    saved_context = agent._context_selector.get_saved_context(messages)
                    if saved_context:
                        # Store in step result for passing to next agent
                        step_result.saved_context = saved_context
                        logger.debug(f"Extracted saved context from {agent_name}: {list(saved_context.keys())}")

                # Emit completion event
                if self.event_bus:
                    from ..status.events import AgentCompleteEvent

                    await self.event_bus.emit(AgentCompleteEvent(
                        session_id=context.get("session_id", "unknown"),
                        branch_id=context.get("branch_id"),
                        agent_name=agent_name,
                        success=step_result.success,
                        duration=time.time() - start_time,
                        next_action=None,  # Action type not yet determined - set by BranchExecutor after validation
                        error=step_result.error
                    ))

                return step_result

            # Enhanced error handling with unified handler
            except ModelAPIError as e:
                # Model API errors will be caught by Agent._run() and converted to error Messages
                # which will then be handled by ValidationProcessor
                logger.error(f"Model API error in {agent_name}: {str(e)}")
                raise

            except PoolExhaustedError as e:
                # Resource errors are terminal
                logger.error(f"Pool exhausted for {agent_name}: {str(e)}")
                raise

            except FrameworkTimeoutError as e:
                # Timeouts are terminal
                logger.error(f"Timeout in {agent_name}: {str(e)}")
                raise

            except AgentConfigurationError as e:
                # Configuration errors are terminal - route to user display
                logger.error(f"Configuration error in {agent_name}: {str(e)}")

                # If user node handler available, display terminal error
                if self.user_node_handler:
                    error_info = {
                        'failed_agent': agent_name,
                        'error_type': 'AgentConfigurationError',
                        'error_code': e.error_code,
                        'message': e.user_message,
                        'details': e.developer_message,
                        'suggestion': e.suggestion,
                        'config_field': getattr(e, 'config_field', 'unknown')
                    }
                    branch = context.get('branch')
                    if branch:
                        return await self.user_node_handler._handle_terminal_error(
                            branch, error_info, context
                        )

                # Otherwise just raise
                raise

            except VisionAgentNotConfiguredError as e:
                # Vision agent configuration errors are terminal - route to user display
                logger.error(f"Vision agent configuration error in {agent_name}: {str(e)}")

                # If user node handler available, display terminal error
                if self.user_node_handler:
                    error_info = {
                        'failed_agent': agent_name,
                        'error_type': 'VisionAgentNotConfiguredError',
                        'error_code': e.error_code,
                        'message': e.user_message,
                        'details': e.developer_message,
                        'suggestion': e.suggestion,
                        'operation': getattr(e, 'operation', 'unknown'),
                        'auto_screenshot': getattr(e, 'auto_screenshot', None)
                    }
                    branch = context.get('branch')
                    if branch:
                        return await self.user_node_handler._handle_terminal_error(
                            branch, error_info, context
                        )

                # Otherwise just raise
                raise

            except (TopologyError, RoutingError, AgentPermissionError) as e:
                # Configuration errors are terminal
                logger.error(f"Configuration error in {agent_name}: {str(e)}")
                raise

            except ToolCallError as e:
                # Tool call format errors need steering, not blind retries
                # Re-raise immediately so BranchExecutor can set error context
                logger.warning(f"Tool call error in {agent_name}: {str(e)}")
                raise

            except AgentFrameworkError as e:
                # All other framework errors - retry a few times
                logger.error(f"Framework error in {agent_name}: {str(e)}")
                if retry_count < self.max_retries:
                    retry_count += 1
                    last_error = str(e)
                    await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                else:
                    raise

            except Exception as e:
                # Unknown errors - convert to framework error
                framework_error = create_error_from_exception(
                    e,
                    agent_name=agent_name,
                    task_id=step_context.session_id,
                    context={"branch_id": step_context.branch_id}
                )
                result = await self._handle_framework_error(framework_error, agent_name, step_context, retry_count)
                if result is not None:
                    return result
                retry_count += 1
                last_error = str(e)
        
        # All retries exhausted
        duration = time.time() - start_time
        failed_result = StepResult(
            agent_name=agent_name,
            success=False,
            error=f"Max retries exceeded. Last error: {last_error}",
            requires_retry=False  # Don't retry further
        )

        # Emit failure completion event
        if self.event_bus:
            from ..status.events import AgentCompleteEvent

            await self.event_bus.emit(AgentCompleteEvent(
                session_id=context.get("session_id", "unknown"),
                branch_id=context.get("branch_id"),
                agent_name=agent_name,
                success=False,
                duration=duration,
                next_action=None,
                error=f"Max retries exceeded. Last error: {last_error}"
            ))

        self._update_stats(agent_name, failed_result, retry_count, duration)
        return failed_result

    def _build_coordination_context(
        self,
        agent_name: str,
        topology_graph: Optional['TopologyGraph'],
    ) -> CoordinationContext:
        """
        Build CoordinationContext dataclass from topology.

        Args:
            agent_name: Name of the agent
            topology_graph: The topology graph for determining available actions

        Returns:
            CoordinationContext with next_agents and can_return_final_response
        """
        if not topology_graph:
            return CoordinationContext()

        next_agents = topology_graph.get_next_agents(agent_name)

        # Determine if agent can return final response
        can_final = topology_graph.has_user_access(agent_name)
        if hasattr(topology_graph, 'exit_points') and agent_name in topology_graph.exit_points:
            can_final = True

        return CoordinationContext(
            next_agents=next_agents,
            can_return_final_response=can_final,
        )

    def _get_json_guidelines_concise(self, actions_str: str) -> str:
        """
        Get concise JSON format reminder for steering/error messages.
        Avoids duplication with system prompt to prevent repetition.
        """
        return f"""Respond with a single JSON object in a markdown block: ```json {{"thought": "...", "next_action": "...", "action_input": {{...}}}} ```"""

    def _get_available_actions(self, agent: 'BaseAgent', step_context: 'StepContext') -> List[str]:
        """
        Extract available actions from topology for steering.

        Args:
            agent: The agent being executed
            step_context: Current step execution context

        Returns:
            List of available action names
        """
        available = []

        topology_graph = getattr(step_context, 'topology_graph', None)

        if topology_graph:
            current_agent = agent.name if hasattr(agent, 'name') else str(agent)

            # Check for next agents
            next_agents = topology_graph.get_next_agents(current_agent)
            if next_agents and [a for a in next_agents if a != "User"]:
                available.append("invoke_agent")

            # Check for user access (final_response capability)
            if topology_graph.has_user_access(current_agent):
                available.append("final_response")

        # Check for tools
        if hasattr(agent, 'tools_schema') and agent.tools_schema:
            available.append("tool_calls")

        # Fallback
        if not available:
            available.append("final_response")

        return available

    def _get_steering_prompt(self, agent: BaseAgent, context: StepContext,
                             is_retry: bool = False) -> str:
        """
        Generate steering prompt to guide agent behavior.
        
        Args:
            agent: The agent being executed
            context: Current execution context
            is_retry: Whether this is a retry attempt

        Returns:
            Steering prompt to inject as last user message
        """
        # Get available actions from topology
        topology_graph = context.topology_graph if hasattr(context, 'topology_graph') else None
        branch = context.branch if hasattr(context, 'branch') else None
        
        # Get available next agents
        if topology_graph and branch:
            # Get next agents from topology graph
            current_agent = agent.name if hasattr(agent, 'name') else str(agent)
            next_agents = topology_graph.get_next_agents(current_agent)
            
            # Filter out User and duplicates
            next_agents = [a for a in next_agents if a != "User"]
            next_agents = list(dict.fromkeys(next_agents))  # Remove duplicates while preserving order
        else:
            next_agents = []
        
        # Determine available actions
        available_actions = []
        action_descriptions = []
        
        # Add invoke_agent if there are next agents
        if next_agents:
            available_actions.append("'invoke_agent'")
            action_descriptions.append(
                f'- If `next_action` is `"invoke_agent"`:\n'
                f'     `{{"agent_name": "AgentName", "request": {{"task": "specific task details"}}}}`\n'
                f'     Available agents: {", ".join(next_agents)}'
            )
        
        # Add tool_calls if agent has tools
        if hasattr(agent, 'tools_schema') and agent.tools_schema:
            available_actions.append("'tool_calls'")
            action_descriptions.append(
                '- If you want to call tools:\n'
                '     Use the native tool_calls response format (NOT as next_action)\n'
                '     The model will automatically handle tool execution'
            )

        # CRITICAL: Only add final_response if agent has permission
        # This must match the logic in _get_dynamic_format_instructions (lines 614-631)
        can_return_final = False
        current_agent = agent.name if hasattr(agent, 'name') else str(agent)

        # Check if agent has user access in topology (the official way)
        if topology_graph and topology_graph.has_user_access(current_agent):
            can_return_final = True

        # Also check if agent is an exit point (for auto_run scenarios)
        if topology_graph and hasattr(topology_graph, 'exit_points') and current_agent in topology_graph.exit_points:
            can_return_final = True

        if can_return_final:
            available_actions.append("'final_response'")
            action_descriptions.append(
                '- If `next_action` is `"final_response"`:\n'
                '     `{"content": "Your final answer..."}`'
            )

        # Ensure agent has at least one available action
        if not available_actions:
            raise RuntimeError(
                f"Agent '{current_agent}' has no available actions in steering prompt. "
                f"The agent cannot invoke other agents, use tools, or return final response. "
                f"This suggests a topology configuration error."
            )

        actions_str = ", ".join(available_actions)

        # Use concise guidelines for steering to avoid duplication with system prompt
        json_guidelines_concise = self._get_json_guidelines_concise(actions_str)

        # Build action guidance from descriptions
        action_guidance = "\n\n".join(action_descriptions) if action_descriptions else ""

        # Build steering based on retry
        if is_retry:
            return f"""Your previous response had an incorrect format. Let's try again with the correct structure.
{json_guidelines_concise}

Available options:
{action_guidance}"""
        else:
            return f"""Make sure to format your response correctly:
{json_guidelines_concise}

Available options:
{action_guidance}"""
    
    def _get_agent_request_example(self, agent_name: str) -> str:
        """
        Get an example request format for an agent based on its input schema.
        Returns either a schema-based example or a generic placeholder.
        """
        try:
            # Try to import AgentRegistry to get schema
            from marsys.agents.registry import AgentRegistry
            
            # Try to get the agent from registry
            agent = AgentRegistry.get(agent_name)
            if agent and hasattr(agent, '_compiled_input_schema'):
                schema = agent._compiled_input_schema
                if schema:
                    # Generate example based on schema
                    return self._generate_example_from_schema(schema)
        except (ImportError, Exception):
            pass
        
        # Default to generic example
        return '{"task": "task details"}'
    
    def _generate_example_from_schema(self, schema: Dict[str, Any]) -> str:
        """
        Generate an example request based on a JSON schema.
        """
        if not schema or not isinstance(schema, dict):
            return '{"task": "task details"}'
        
        schema_type = schema.get('type', 'any')
        
        # Handle string type
        if schema_type == 'string':
            return '"<your-request-string>"'
        
        # Handle object type
        if schema_type == 'object':
            properties = schema.get('properties', {})
            required = schema.get('required', [])
            
            if not properties:
                return '{}'
            
            # Build example object
            example_obj = {}
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get('type', 'any')
                is_required = prop_name in required
                
                # Generate example based on type and requirement
                if prop_type == 'string':
                    placeholder = f'{prop_name}_value{" (required)" if is_required else ""}'
                elif prop_type == 'array':
                    placeholder = f'["{prop_name}_item1", "{prop_name}_item2"]'
                elif prop_type == 'object':
                    placeholder = f'{{"key": "{prop_name}_data"}}'
                elif prop_type == 'number' or prop_type == 'integer':
                    placeholder = 123
                elif prop_type == 'boolean':
                    placeholder = True
                else:
                    placeholder = f'{prop_name}_value'
                
                example_obj[prop_name] = placeholder
            
            # Convert to JSON string with proper formatting
            import json
            return json.dumps(example_obj, indent=2)
        
        # For any other type or unknown schema
        return '<request-data-specific-to-agent>'
    
    async def _process_tool_calls(
        self,
        raw_response: Any,
        agent: BaseAgent,
        context: StepContext
    ) -> StepResult:
        """
        Process ONLY native tool calls using ToolCallProcessor.
        NO content parsing, NO routing decisions.
        ONLY accepts Message responses.
        """
        from ...agents.memory import Message
        from ..formats.processors import ToolCallProcessor

        # CRITICAL: Only accept Message responses
        if not isinstance(raw_response, Message):
            raise TypeError(
                f"Agent {context.agent_name} must return Message object, "
                f"got {type(raw_response).__name__}"
            )

        result = StepResult(
            agent_name=context.agent_name,
            success=True,
            response=raw_response  # Store full Message object
        )

        # Extract native tool_calls using ToolCallProcessor
        tool_processor = ToolCallProcessor()
        if tool_processor.can_process(raw_response):
            parsed = tool_processor.process(raw_response)
            if parsed and "tool_calls" in parsed:
                result.tool_calls = parsed["tool_calls"]
                logger.debug(f"Extracted {len(result.tool_calls)} native tool_calls from {context.agent_name}")

        # Create memory update with all Message fields
        memory_update = {
            "role": raw_response.role,
            "content": raw_response.content,
            "name": raw_response.name or context.agent_name,
            "timestamp": time.time()
        }

        # Include tool_calls in memory if present
        if raw_response.tool_calls:
            memory_update["tool_calls"] = raw_response.tool_calls

        result.memory_updates = [memory_update]

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

    # ============================================================================
    # Enhanced Error Handling Methods
    # ============================================================================

    async def _notify_critical_error(
        self,
        error: AgentFrameworkError,
        context: Union[StepContext, Dict[str, Any]]
    ) -> None:
        """Notify user about critical errors that require immediate attention."""

        if not self.event_bus:
            return

        from ..status.events import CriticalErrorEvent

        # Extract context info
        if isinstance(context, StepContext):
            session_id = context.session_id
            branch_id = context.branch_id
            agent_name = context.agent_name
        else:
            session_id = context.get("session_id", "unknown")
            branch_id = context.get("branch_id")
            agent_name = context.get("agent_name")

        # Determine provider for API errors
        provider = None
        if isinstance(error, ModelAPIError):
            provider = error.provider

        # Create and emit critical error event
        event = CriticalErrorEvent(
            session_id=session_id,
            branch_id=branch_id,
            agent_name=agent_name,
            error_type=error.__class__.__name__,
            error_code=error.error_code,
            message=error.message,
            provider=provider,
            suggested_action=error.suggestion,
            requires_user_action=True,
            timestamp=time.time()
        )

        await self.event_bus.emit(event)

        logger.critical(
            f"CRITICAL ERROR: {error.message}\n"
            f"Action Required: {error.suggestion}"
        )

    async def _handle_model_api_error(
        self,
        error: ModelAPIError,
        agent_name: str,
        step_context: StepContext,
        retry_count: int
    ) -> Optional[StepResult]:
        """Handle ModelAPIError with intelligent retry logic."""

        logger.error(f"API error for agent '{agent_name}': {error.message}")
        logger.debug(f"Error classification: {error.classification}")

        # Check if critical (non-retryable)
        if error.is_critical():
            # Notify user immediately for critical errors
            await self._notify_critical_error(error, step_context)

            return StepResult(
                agent_name=agent_name,
                response=None,
                action_type="error",
                success=False,
                error=error.message,
                metadata={
                    "error_type": "critical_api_error",
                    "error_classification": error.classification,
                    "suggested_action": error.suggestion,
                    "provider": error.provider
                }
            )

        # Check retry logic for retryable errors
        if error.is_retryable and retry_count < self.max_retries:
            # Use error's retry_after if available
            wait_time = error.retry_after or (self.retry_delay * (retry_count + 1))

            logger.info(f"Retrying after {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
            await asyncio.sleep(wait_time)

            # Return None to trigger retry in main loop
            return None

        # Max retries exhausted
        return StepResult(
            agent_name=agent_name,
            response=None,
            action_type="error",
            success=False,
            error=error.message,
            requires_retry=False,
            metadata={
                "error_type": "api_error",
                "error_classification": error.classification,
                "max_retries_exhausted": True
            }
        )

    async def _handle_pool_exhausted_error(
        self,
        error: PoolExhaustedError,
        agent_name: str,
        step_context: StepContext,
        retry_count: int
    ) -> Optional[StepResult]:
        """Handle pool exhaustion with limited retries."""

        MAX_POOL_RETRIES = 2  # Limit retries for pool exhaustion

        if retry_count < MAX_POOL_RETRIES:
            wait_time = 5.0  # Fixed wait for pool availability
            logger.warning(f"Pool exhausted for '{agent_name}', waiting {wait_time}s...")

            # Emit resource limit event if available
            if self.event_bus:
                from ..status.events import ResourceLimitEvent
                await self.event_bus.emit(ResourceLimitEvent(
                    session_id=step_context.session_id,
                    branch_id=step_context.branch_id,
                    resource_type="agent_pool",
                    pool_name=error.pool_name,
                    limit_value=error.total_instances,
                    current_value=error.allocated_instances,
                    suggestion="Wait for instances to become available"
                ))

            await asyncio.sleep(wait_time)
            return None  # Trigger retry

        return StepResult(
            agent_name=agent_name,
            response=None,
            action_type="error",
            success=False,
            error=error.message,
            metadata={
                "error_type": "pool_exhausted",
                "pool_name": error.pool_name,
                "allocated": error.allocated_instances,
                "total": error.total_instances
            }
        )

    async def _handle_timeout_error(
        self,
        error: FrameworkTimeoutError,
        agent_name: str,
        step_context: StepContext
    ) -> StepResult:
        """Handle timeout errors (usually not retryable)."""

        logger.error(f"Timeout for agent '{agent_name}': {error.message}")

        return StepResult(
            agent_name=agent_name,
            response=None,
            action_type="error",
            success=False,
            error=error.message,
            requires_retry=False,
            metadata={
                "error_type": "timeout",
                "operation": error.operation,
                "timeout_seconds": error.timeout_seconds,
                "elapsed_seconds": error.elapsed_seconds
            }
        )

    async def _handle_configuration_error(
        self,
        error: AgentFrameworkError,
        agent_name: str,
        step_context: StepContext
    ) -> StepResult:
        """Handle configuration errors that should never be retried."""

        logger.error(f"Configuration error for agent '{agent_name}': {error.message}")

        # Notify about configuration issues
        if self.event_bus:
            from ..status.events import CriticalErrorEvent
            await self.event_bus.emit(CriticalErrorEvent(
                session_id=step_context.session_id,
                branch_id=step_context.branch_id,
                agent_name=agent_name,
                error_type="configuration_error",
                error_code=error.error_code,
                message=error.message,
                suggested_action=error.suggestion,
                requires_user_action=True
            ))

        return StepResult(
            agent_name=agent_name,
            response=None,
            action_type="error",
            success=False,
            error=error.message,
            requires_retry=False,
            metadata={
                "error_type": "configuration_error",
                "error_code": error.error_code,
                "suggestion": error.suggestion
            }
        )

    async def _handle_framework_error(
        self,
        error: AgentFrameworkError,
        agent_name: str,
        step_context: StepContext,
        retry_count: int
    ) -> Optional[StepResult]:
        """Handle generic framework errors with standard retry logic."""

        logger.error(f"Framework error for agent '{agent_name}': {error.message}")

        if retry_count < self.max_retries:
            wait_time = self.retry_delay * (retry_count + 1)
            logger.info(f"Retrying after {wait_time}s (attempt {retry_count + 1}/{self.max_retries})")
            await asyncio.sleep(wait_time)
            return None  # Trigger retry

        return StepResult(
            agent_name=agent_name,
            response=None,
            action_type="error",
            success=False,
            error=error.message,
            metadata={
                "error_type": "framework_error",
                "error_code": error.error_code,
                "context": error.context
            }
        )
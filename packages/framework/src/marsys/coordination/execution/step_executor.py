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
    
    async def _emit_error_event(
        self,
        *,
        agent_name: str,
        step_number: Optional[int],
        step_span_id: Optional[str],
        exception: BaseException,
        retry_count: int,
        context: Dict[str, Any],
    ) -> None:
        """Emit a structured ``ErrorEvent`` for an exception propagating into the executor.

        Used for framework errors (``AgentFrameworkError``,
        ``PoolExhaustedError``, ``FrameworkTimeoutError``) and
        ``REQUEST_TOO_LARGE`` retries that exceed compaction. Most agent-
        internal exceptions are caught inside ``Agent._run`` and emit via
        ``Agent._emit_step_error_event`` before being converted to error
        Messages — this executor-level path is the fallback when something
        actually crosses the boundary. Both sites share ``build_error_event``
        so trace consumers see identical event shape.
        """
        if not self.event_bus:
            return
        from ..status.events import build_error_event

        await self.event_bus.emit(build_error_event(
            exception,
            session_id=context.get("session_id", "unknown"),
            branch_id=context.get("branch_id"),
            agent_name=agent_name,
            step_number=step_number,
            step_span_id=step_span_id,
            retry_count=retry_count,
        ))

    def _apply_runtime_context(
        self,
        agent: Any,
        execution_config: Any,
    ) -> None:
        """Push orchestration-level retry/event config onto the agent's model adapter.

        The adapter retry loop (``models/adapters/base.py``) reads
        ``self.error_config`` at call time. Agents are constructed before
        Orchestra runs, so we inject the active ``ExecutionConfig.error_handling``
        and ``event_bus`` here, just before each ``run_step`` invocation. No-op
        when the agent has no model (e.g., deterministic agents) or the model
        has no adapter (e.g., custom in-process implementations).
        """
        model = getattr(agent, "model", None)
        if model is None:
            return
        error_config = getattr(execution_config, "error_handling", None)
        for attr in ("adapter", "async_adapter"):
            adapter = getattr(model, attr, None)
            if adapter is None:
                continue
            # ``error_config`` and ``event_bus`` are declared on
            # ``APIProviderAdapter.__init__`` — set them as plain attributes so
            # adapters that don't extend the base class (rare, custom users)
            # are unaffected.
            if hasattr(adapter, "error_config") or hasattr(adapter, "__dict__"):
                adapter.error_config = error_config
                adapter.event_bus = self.event_bus

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

        # User-node dispatch is handled at the orchestrator level via the
        # UserNode det-node's on_single_invoke (orchestrator.py:441-444),
        # which fires before RealRuntime.step ever sees a User branch. The
        # legacy string-based "agent='User'" path used to live here but is
        # unreachable in production; removed in step 7 cleanup.

        # Normal agent handling
        agent_name = agent.name if hasattr(agent, 'name') else str(agent)

        # Generate a unique span ID for this step execution (used for trace correlation)
        import uuid
        step_span_id = str(uuid.uuid4())
        step_number = context.get("step_number", 0)

        # Full-input capture lives in ``Agent._run`` (emits
        # ``AgentMessagesPreparedEvent`` at the model-dispatch site).
        # ``tracing_enabled`` is read here to decide whether to build a
        # ``TraceContext`` for the model-wrapper capture helper
        # (``emit_llm_call``).
        execution_config_obj = context.get("execution_config")
        tracing_cfg = getattr(execution_config_obj, "tracing", None) if execution_config_obj else None
        tracing_enabled = bool(tracing_cfg and getattr(tracing_cfg, "enabled", False))

        if self.event_bus:
            from ..status.events import AgentStartEvent

            request_summary = str(request)[:100] if request else None

            await self.event_bus.emit(AgentStartEvent(
                session_id=context.get("session_id", "unknown"),
                branch_id=context.get("branch_id"),
                agent_name=agent_name,
                request_summary=request_summary,
                step_number=step_number,
                step_span_id=step_span_id,
            ))

        # Store span ID in context for downstream use (tool executor, generation event)
        context["step_span_id"] = step_span_id

        # Build a TraceContext only when tracing is on. When disabled,
        # ``emit_llm_call`` bypasses on a None trace_ctx — same
        # code path as raw model usage outside Orchestra.
        if tracing_enabled:
            from ..tracing.trace_context import TraceContext
            trace_ctx = TraceContext(
                step_span_id=step_span_id,
                branch_id=context.get("branch_id"),
                agent_name=agent_name,
                session_id=context.get("session_id", "unknown"),
                event_bus=self.event_bus,
            )
        else:
            trace_ctx = None
        context["trace_ctx"] = trace_ctx

        # Create step context
        step_context = StepContext(
            session_id=context.get("session_id", ""),
            branch_id=context.get("branch_id", ""),
            step_number=step_number,
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
                run_context = {
                    "request_context": step_context.to_request_context(),
                    "event_bus": self.event_bus,  # Pass event_bus for planning/memory events
                    "step_number": step_context.step_number,  # Pass step_number for planning triggers
                    "session_id": step_context.session_id,  # Required for planning events emission
                    "branch_id": step_context.branch_id,  # For consistent event tracking
                    "step_span_id": step_span_id,  # so the agent can stamp AgentMessagesPreparedEvent with the correct span
                    "trace_ctx": trace_ctx,         # picked up by emit_llm_call at the model-wrapper layer
                }

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
                    topology_graph,
                    branch=branch
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

                    # Topology peers (peer agents only — det-nodes excluded)
                    topology_neighbors: list[str] = []
                    topology_graph = getattr(step_context, "topology_graph", None)
                    if topology_graph:
                        try:
                            raw_next = topology_graph.get_next_agents(agent_name) or []
                            topology_neighbors = [
                                a for a in raw_next
                                if a.lower() not in ("user", "start", "end")
                            ]
                        except Exception:
                            topology_neighbors = []

                    # Build steering context
                    steering_ctx = SteeringContext(
                        agent_name=agent_name,
                        available_actions=available_actions,
                        error_context=error_context,
                        is_retry=is_retry,
                        steering_mode=execution_config.steering_mode,
                        topology_neighbors=topology_neighbors,
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
                
                # Apply orchestration-level retry/event config to the model adapter
                # so adapter-level retries (5xx / 429 backoff) honour ExecutionConfig.
                self._apply_runtime_context(agent, execution_config)

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

                            # Determine continuation behavior based on coordination action
                            if not hasattr(step_result, 'metadata'):
                                step_result.metadata = {}

                            if step_result.coordination_action:
                                # Mixed call: agent called real tools AND a coordination tool.
                                # invoke_agent is a handoff - agent is done after tools execute.
                                # Add synthetic tool response for the coordination call so there
                                # are no orphaned tool_call_ids in memory.
                                if hasattr(agent, 'memory') and hasattr(agent.memory, 'add'):
                                    from ..formats.coordination_tools import parse_coordination_call
                                    # Find the coordination tool_call_id from the original message
                                    coord_call_id = None
                                    if hasattr(raw_response, 'tool_calls') and raw_response.tool_calls:
                                        from ..formats.coordination_tools import is_coordination_tool
                                        for tc in raw_response.tool_calls:
                                            if isinstance(tc, dict):
                                                tc_name = tc.get("function", {}).get("name", "")
                                                tc_id = tc.get("id", "")
                                            else:
                                                tc_name = getattr(tc, 'name', "")
                                                tc_id = getattr(tc, 'id', "")
                                            if is_coordination_tool(tc_name):
                                                coord_call_id = tc_id
                                                break

                                    if coord_call_id:
                                        # Determine target info for synthetic response
                                        target_info = ""
                                        if step_result.coordination_action == "invoke_agent":
                                            invocations = (step_result.coordination_data or {}).get("invocations", [])
                                            targets = [inv.get("agent_name", "?") for inv in invocations]
                                            target_info = f" to {', '.join(targets)}"
                                        elif step_result.coordination_action == "return_final_response":
                                            target_info = " - delivering final response"

                                        agent.memory.add(
                                            role='tool',
                                            content=f"Control delegated{target_info}. Current agent's turn complete.",
                                            tool_call_id=coord_call_id,
                                            name=step_result.coordination_action,
                                        )

                                step_result.metadata['has_tool_results'] = True
                                # No tool_continuation - coordination action means handoff
                                logger.info(
                                    f"Agent '{agent_name}' executed {len(tool_result.tool_results)} tools "
                                    f"+ coordination action '{step_result.coordination_action}' - handing off"
                                )
                            else:
                                # Regular tools only - agent continues to process results
                                step_result.metadata['tool_continuation'] = True
                                step_result.metadata['has_tool_results'] = True
                                logger.info(f"Agent '{agent_name}' executed {len(tool_result.tool_results)} tools - will continue in next step")
                
                # 5. Add synthetic tool response for coordination-only calls.
                # When the agent calls ONLY a coordination tool (no regular tools),
                # the tool execution block above is skipped entirely. The Message
                # in agent memory has tool_calls=[invoke_agent] but no matching tool
                # response. Without this, _clean_orphaned_tool_calls() strips the
                # tool_call on the next invocation, and the agent loses context of
                # having already delegated (causing double invocation after convergence).
                if (step_result.coordination_action
                        and not step_result.tool_calls
                        and not step_result.tool_results
                        and hasattr(agent, 'memory') and hasattr(agent.memory, 'add')):
                    from ..formats.coordination_tools import is_coordination_tool
                    coord_call_id = None
                    if hasattr(raw_response, 'tool_calls') and raw_response.tool_calls:
                        for tc in raw_response.tool_calls:
                            if isinstance(tc, dict):
                                tc_name = tc.get("function", {}).get("name", "")
                                tc_id = tc.get("id", "")
                            else:
                                tc_name = getattr(tc, 'name', "")
                                tc_id = getattr(tc, 'id', "")
                            if is_coordination_tool(tc_name):
                                coord_call_id = tc_id
                                break

                    if coord_call_id:
                        target_info = ""
                        if step_result.coordination_action == "invoke_agent":
                            invocations = (step_result.coordination_data or {}).get("invocations", [])
                            targets = [inv.get("agent_name", "?") for inv in invocations]
                            target_info = f" to {', '.join(targets)}"
                        elif step_result.coordination_action == "return_final_response":
                            target_info = " - delivering final response"

                        agent.memory.add(
                            role='tool',
                            content=f"Control delegated{target_info}. Current agent's turn complete.",
                            tool_call_id=coord_call_id,
                            name=step_result.coordination_action,
                        )
                        logger.debug(f"Added synthetic tool response for coordination-only call '{step_result.coordination_action}' (tool_call_id: {coord_call_id})")

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

                # Generation spans are emitted by the model-wrapper capture
                # helper (a single LLMCallEvent per call) which carries the
                # full payload. The legacy GenerationEvent dataclass is kept
                # only for out-of-tree emitters — this repo neither emits it
                # nor subscribes to it (see TraceCollector.IGNORED_EVENTS).

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
                        error=step_result.error,
                        step_number=step_number,
                        step_span_id=step_span_id,
                    ))

                return step_result

            # Enhanced error handling with unified handler
            except ModelAPIError as e:
                # Model API errors will be caught by Agent._run() and converted to error Messages
                # which will then be handled by ValidationProcessor
                logger.error(f"Model API error in {agent_name}: {str(e)}")
                await self._emit_error_event(
                    agent_name=agent_name,
                    step_number=step_number,
                    step_span_id=step_span_id,
                    exception=e,
                    retry_count=retry_count,
                    context=context,
                )
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
                    delay = execution_config.error_handling.compute_delay(
                        provider=None, attempt=retry_count - 1
                    )
                    await asyncio.sleep(delay)
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

        # Emit structured error event before the legacy completion event so
        # the tracing collector sees both on the same step span.
        if self.event_bus:
            from ..status.events import AgentCompleteEvent, ErrorEvent

            # Synthesize a generic exception for the structured event when we
            # only have a string description from the retry loop.
            synthetic_error = RuntimeError(last_error or "Max retries exceeded")
            await self._emit_error_event(
                agent_name=agent_name,
                step_number=step_number,
                step_span_id=step_span_id,
                exception=synthetic_error,
                retry_count=retry_count,
                context=context,
            )

            await self.event_bus.emit(AgentCompleteEvent(
                session_id=context.get("session_id", "unknown"),
                branch_id=context.get("branch_id"),
                agent_name=agent_name,
                success=False,
                duration=duration,
                next_action=None,
                error=f"Max retries exceeded. Last error: {last_error}",
                step_number=step_number,
                step_span_id=step_span_id,
            ))

        self._update_stats(agent_name, failed_result, retry_count, duration)
        return failed_result

    def _build_coordination_context(
        self,
        agent_name: str,
        topology_graph: Optional['TopologyGraph'],
        branch: Optional['ExecutionBranch'] = None,
    ) -> CoordinationContext:
        """
        Build CoordinationContext from topology.

        Topology-driven gating, single source of truth:
            - can_terminate_workflow: agent has direct edge to End det-node.
            - can_ask_user: agent has direct edge to User det-node.
            - is_conversation_branch: branch is in a conversation loop.

        Legacy entry_point / exit_points / User-edge patterns are translated
        into explicit det-node edges by Orchestra._apply_legacy_topology_shim
        before this function runs; that shim is the only legacy support layer.
        """
        if not topology_graph:
            return CoordinationContext()

        next_agents = topology_graph.get_next_agents(agent_name)

        can_terminate = (
            hasattr(topology_graph, 'has_edge_to_endnode')
            and topology_graph.has_edge_to_endnode(agent_name)
        )
        can_ask_user = (
            hasattr(topology_graph, 'has_edge_to_usernode')
            and topology_graph.has_edge_to_usernode(agent_name)
        )

        is_conversation = False
        if branch is not None:
            is_conversation = branch.is_conversation_branch() if hasattr(branch, 'is_conversation_branch') else False

        return CoordinationContext(
            next_agents=next_agents,
            can_terminate_workflow=can_terminate,
            can_ask_user=can_ask_user,
            is_conversation_branch=is_conversation,
        )

    def _get_available_actions(self, agent: 'BaseAgent', step_context: 'StepContext') -> List[str]:
        """
        Extract available coordination tools and regular tools for steering.

        Returns:
            List of available action/tool names
        """
        available = []

        topology_graph = getattr(step_context, 'topology_graph', None)

        if topology_graph:
            current_agent = agent.name if hasattr(agent, 'name') else str(agent)

            next_agents = topology_graph.get_next_agents(current_agent) or []
            invocable = [a for a in next_agents if a.lower() not in ("user", "start", "end")]
            if invocable:
                available.append("invoke_agent")

            if hasattr(topology_graph, 'has_edge_to_endnode') and \
                    topology_graph.has_edge_to_endnode(current_agent):
                available.append("terminate_workflow")

            if hasattr(topology_graph, 'has_edge_to_usernode') and \
                    topology_graph.has_edge_to_usernode(current_agent):
                available.append("ask_user")

        if hasattr(agent, 'tools_schema') and agent.tools_schema:
            available.append("tool_calls")

        if not available:
            available.append("terminate_workflow")

        return available

    async def _process_tool_calls(
        self,
        raw_response: Any,
        agent: BaseAgent,
        context: StepContext
    ) -> StepResult:
        """
        Process native tool calls: partition into regular tools vs coordination tools.

        Regular tool calls (web_search, file_read, etc.) go to result.tool_calls
        for execution by ToolExecutor.

        Coordination tool calls (invoke_agent, return_final_response, end_conversation)
        go to result.coordination_action/coordination_data for routing by BranchExecutor.
        """
        from ...agents.memory import Message
        from ..formats.processors import ToolCallProcessor
        from ..formats.coordination_tools import is_coordination_tool, parse_coordination_call

        # CRITICAL: Only accept Message responses
        if not isinstance(raw_response, Message):
            raise TypeError(
                f"Agent {context.agent_name} must return Message object, "
                f"got {type(raw_response).__name__}"
            )

        # Item 5: a role=error response means Agent._run caught an exception
        # and converted it to an error Message. The agent already emitted a
        # structured ErrorEvent before the conversion (see
        # Agent._emit_step_error_event). Mark the step failed so the span
        # gets status="error" and RealRuntime._translate returns kind="FAIL"
        # immediately — no point letting the content-only-loop heuristic
        # halt the branch ten retries later.
        if raw_response.role == "error":
            error_text = (
                raw_response.content
                if isinstance(raw_response.content, str)
                else "agent error"
            )
            failed = StepResult(
                agent_name=context.agent_name,
                success=False,
                error=error_text,
                response=raw_response,
            )
            # Still write to memory so prior error stays in the conversation
            # (steering / validation may consult it on a follow-up step).
            failed.memory_updates = [{
                "role": raw_response.role,
                "content": raw_response.content,
                "name": raw_response.name or context.agent_name,
                "timestamp": time.time(),
            }]
            return failed

        result = StepResult(
            agent_name=context.agent_name,
            success=True,
            response=raw_response  # Store full Message object
        )

        # Extract all native tool_calls using ToolCallProcessor
        tool_processor = ToolCallProcessor()
        all_tool_calls = []
        if tool_processor.can_process(raw_response):
            parsed = tool_processor.process(raw_response)
            if parsed and "tool_calls" in parsed:
                all_tool_calls = parsed["tool_calls"]

        # Partition into regular tools and coordination tools
        regular_calls = []
        coordination_call = None  # Only the first coordination call is used

        for tc in all_tool_calls:
            # Handle both dict format and ToolCallMsg objects
            if isinstance(tc, dict):
                tool_name = tc.get("function", {}).get("name", "")
            else:
                # ToolCallMsg has .name directly (not nested under .function)
                tool_name = getattr(tc, 'name', "")

            if is_coordination_tool(tool_name):
                if coordination_call is None:
                    coordination_call = tc
                    # Convert ToolCallMsg to dict for parse_coordination_call
                    tc_dict = tc.to_dict() if hasattr(tc, 'to_dict') else tc
                    action_name, action_data = parse_coordination_call(tc_dict)
                    result.coordination_action = action_name
                    result.coordination_data = action_data
                    logger.debug(
                        f"Agent '{context.agent_name}' coordination action: {action_name}"
                    )
                else:
                    logger.warning(
                        f"Agent '{context.agent_name}' made multiple coordination calls - "
                        f"ignoring extra: {tool_name}"
                    )
            else:
                regular_calls.append(tc)

        if regular_calls:
            result.tool_calls = regular_calls
            logger.debug(f"Extracted {len(regular_calls)} regular tool_calls from {context.agent_name}")

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
                    "agent_name": context.agent_name,
                    "step_number": context.step_number,
                    "step_span_id": context.metadata.get("step_span_id"),
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
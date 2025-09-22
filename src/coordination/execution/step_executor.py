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

# Enhanced error handling imports
from ...agents.exceptions import (
    # Base error
    AgentFrameworkError,

    # Model/API errors
    ModelAPIError,

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
    AgentPermissionError,
    AgentLimitError,

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
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        # Pass event_bus to tool executor
        self.tool_executor = tool_executor or RealToolExecutor(event_bus=event_bus)
        self.user_node_handler = user_node_handler
        self.event_bus = event_bus
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
                
                # Add format instructions to context if available
                if topology_graph and branch:
                    format_instructions = self._get_dynamic_format_instructions(
                        agent, 
                        branch,
                        topology_graph
                    )
                    if format_instructions:
                        run_context["format_instructions"] = format_instructions
                        logger.debug(f"Passing dynamic format instructions for {agent_name} via context")
                
                # NEW: Add steering based on config
                # Get agent retry count from context (if available)
                agent_retry_count = context.get('metadata', {}).get('agent_retry_count', 0)

                # Convert to boolean: is this ANY kind of retry?
                is_retry = (retry_count > 0) or (agent_retry_count > 0)

                if execution_config.should_apply_steering(is_retry):
                    steering_prompt = self._get_steering_prompt(
                        agent,
                        step_context,
                        is_retry=is_retry
                    )
                    run_context["steering_prompt"] = steering_prompt
                    logger.debug(f"Added steering for {agent_name} (exception_retry: {retry_count}, agent_retry: {agent_retry_count}, is_retry: {is_retry}, mode: {execution_config.steering_mode})")
                
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
                            # Add tool results to agent's memory (source of truth)
                            if hasattr(agent, 'memory') and hasattr(agent.memory, 'add'):
                                for tr in tool_result.tool_results:
                                    origin = tr.get('_origin', 'response')
                                    
                                    if origin == 'response':
                                        # Response-origin: Standard OpenAI format with role="tool"
                                        agent.memory.add(
                                            role="tool",
                                            content=json.dumps(tr['result']) if not isinstance(tr['result'], str) else tr['result'],
                                            tool_call_id=tr['tool_call_id'],
                                            name=tr['tool_name']
                                        )
                                    else:  # origin == 'content'
                                        # Content-origin: Must use role="user" (OpenAI API restriction)
                                        agent.memory.add(
                                            role="user",
                                            content=json.dumps(tr['result']) if not isinstance(tr['result'], str) else tr['result']
                                            # No tool_call_id or name fields when role != "tool"
                                        )
                            
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
                
                # FIX: Extract saved context for passing to next agent
                if hasattr(agent, '_context_selector'):
                    messages = agent.memory.retrieve_all() if hasattr(agent, 'memory') else []
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
                        next_action=step_result.action_type,
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

            except (TopologyError, RoutingError, AgentPermissionError) as e:
                # Configuration errors are terminal
                logger.error(f"Configuration error in {agent_name}: {str(e)}")
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
    
    def _get_environmental_context(self) -> str:
        """
        Generate environmental context information to inject into system prompts.

        Returns:
            Formatted string with contextual information
        """
        from datetime import datetime

        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")

        context_lines = []
        context_lines.append("--- ENVIRONMENTAL CONTEXT ---")
        context_lines.append(f"Today's date: {date_str}")

        # Future context items can be added here:
        # - Time zone information
        # - Session/branch IDs for debugging
        # - Execution metrics (step count, elapsed time)
        # - Resource limits and quotas
        # - Environment mode (production/staging/dev)
        # - User preferences
        # - Working directory context
        # - Active feature flags

        context_lines.append("--- END ENVIRONMENTAL CONTEXT ---")
        return "\n".join(context_lines)

    def _get_dynamic_format_instructions(
        self,
        agent: 'BaseAgent',
        branch: Optional['ExecutionBranch'] = None,
        topology_graph: Optional['TopologyGraph'] = None
    ) -> str:
        """
        Generate COMPLETE system prompt with dynamic action determination.
        This builds a full system prompt that replaces the agent's default one,
        with actions determined dynamically from topology and branch metadata.
        """
        if not topology_graph:
            # Fall back to agent's own system prompt if no topology
            return ""

        agent_name = agent.name if hasattr(agent, 'name') else str(agent)

        # Build complete system prompt components
        system_prompt_parts = []

        # 1. Start with agent's description (cleaned of schema hints)
        if not hasattr(agent, 'description'):
            raise RuntimeError(
                f"Agent '{agent_name}' does not have a description attribute. "
                "All agents must have a description for proper system prompt generation."
            )

        cleaned_description = self._strip_schema_hints(agent.description)
        system_prompt_parts.append(cleaned_description)

        # 2. Add environmental context
        environmental_context = self._get_environmental_context()
        if environmental_context:
            system_prompt_parts.append(environmental_context)
        
        # 2. Add tool instructions if agent has tools
        tool_instructions = self._get_tool_instructions(agent)
        if tool_instructions:
            system_prompt_parts.append(tool_instructions)
        
        # 3. Add peer agent instructions based on topology
        next_agents = topology_graph.get_next_agents(agent_name)
        if next_agents:
            peer_instructions = self._get_peer_agent_instructions(agent_name, next_agents)
            if peer_instructions:
                system_prompt_parts.append(peer_instructions)
        
        # 4. Add context handling instructions
        context_instructions = self._get_context_instructions(agent)
        if context_instructions:
            system_prompt_parts.append(context_instructions)
        
        # 5. Now build the dynamic format instructions
        guidelines = (
            "When responding, ensure your output adheres to the requested format. "
            "Be concise and stick to the persona and task given."
        )
        
        # Add schema instructions if agent has them
        if hasattr(agent, '_get_schema_instructions'):
            schema_instructions = agent._get_schema_instructions()
            if schema_instructions:
                guidelines += schema_instructions
        
        # Determine available actions dynamically
        available_actions = []
        action_descriptions = []
        
        # # 1. Check if agent has tools
        # if hasattr(agent, 'tools') and agent.tools:
        #     available_actions.append("'call_tool'")
        #     action_descriptions.append(
        #         '- If `next_action` is `"call_tool"`:\n'
        #         '     `{"tool_calls": [{"id": "...", "type": "function", "function": {"name": "tool_name", "arguments": "{\\"param\\": ...}"}}]}`\n'
        #         '     • The list **must** stay inside `action_input`.'
        #     )
        
        # 2. Check if agent can invoke other agents (from topology)
        next_agents = topology_graph.get_next_agents(agent_name)
        if next_agents:
            available_actions.append("'invoke_agent'")
            # Include the actual agent names in the description
            agents_list = ", ".join(next_agents)
            action_descriptions.append(
                '- If `next_action` is `"invoke_agent"`:\n'
                f'     An ARRAY of agent invocation objects for agents: {agents_list}\n'
                '     `[{"agent_name": "example_agent_name", "request": {...}}, ...]`\n'
                '     • Single agent: array with one object\n'
                '     • Multiple agents: array with multiple objects (parallel execution)\n'
                '     • IMPORTANT: Only invoke agents together if you do NOT need one\'s response before invoking another\n'
                '     • Control flow after invocation depends on the system topology configuration'
            )
        
        # 3. CRITICAL: Check if agent can return final response
        can_return_final = False
        
        # Check static flag first (set at branch start)
        if hasattr(agent, '_can_return_final_response') and agent._can_return_final_response:
            can_return_final = True
        
        # Check if agent has user access in topology
        if topology_graph.has_user_access(agent_name):
            can_return_final = True
        
        # Check if agent is an exit point
        if hasattr(topology_graph, 'exit_points') and agent_name in topology_graph.exit_points:
            can_return_final = True
        
        if can_return_final:
            available_actions.append("'final_response'")
            action_descriptions.append(
                '- If `next_action` is `"final_response"`:\n'
                '     `{"response": "Your final textual answer..."}`\n'
                '     **USE THIS when your assigned task is fully complete!**'
            )
        
        # Build actions string
        if not available_actions:
            # This indicates a configuration error - agent cannot do anything
            # This should never happen in a properly configured topology
            raise RuntimeError(
                f"Agent '{agent_name}' has no available actions. "
                f"The agent cannot invoke other agents (no outgoing edges in topology) "
                f"and cannot return final response (no user access or exit point designation). "
                f"This suggests a topology configuration error. "
                f"Please check that the agent has either: "
                f"1) Outgoing edges to other agents, or "
                f"2) Is designated as an exit point or has user access."
            )
        else:
            actions_str = ", ".join(available_actions)
        
        # Build the complete guidelines (matching agent's format)
        guidelines += self._get_json_format_guidelines(actions_str, action_descriptions)
        
        # Add examples based on available actions
        examples = []
        
        if "'invoke_agent'" in actions_str:
            # Use actual agent names in examples
            # example_agent = next_agents[0] if next_agents else "example_agent"
            
            # # Try to get schema-based example, otherwise use generic placeholder
            # request_example = self._get_agent_request_example(example_agent)
            
            examples.append("""
Example for single agent invocation:
```json
{{
  "thought": "I need to delegate this task to a specialist.",
  "next_action": "invoke_agent",
  "action_input": [
    {{
      "agent_name": "<example_agent>",
      "request": <request_example>
    }}
  ]
}}
```

Example for parallel invocations (when responses are independent):
```json
{{
  "thought": "I need to process multiple items that don't depend on each other.",
  "next_action": "invoke_agent",
  "action_input": [
    {{
      "agent_name": "<example_agent_1>",
      "request": <request_example_1>
    }},
    {{
      "agent_name": "<example_agent_2>", 
      "request": <request_example_2>
    }}
  ]
}}
```

REMEMBER: Only invoke agents together if you don't need one's response to invoke another!""")

#         if "'call_tool'" in actions_str:
#             # Get an example tool name if available
#             example_tool = "example_tool"
#             if hasattr(agent, 'tools') and agent.tools:
#                 tool_names = list(agent.tools.keys()) if isinstance(agent.tools, dict) else []
#                 if tool_names:
#                     example_tool = tool_names[0]
            
#             examples.append(f"""
# Example for `call_tool`:
# ```json
# {{
#   "thought": "I need to use a tool.",
#   "next_action": "call_tool",
#   "action_input": {{
#     "tool_calls": [{{
#       "id": "call_123",
#       "type": "function",
#       "function": {{
#         "name": "{example_tool}",
#         "arguments": "{{\\"param\\": \\"value\\"}}"
#       }}
#     }}]
#   }}
# }}
# ```""")

        if "'final_response'" in actions_str:
            examples.append("""
Example for `final_response`:
```json
{
  "thought": "I have completed the task.",
  "next_action": "final_response",
  "action_input": {
    "response": "Here is my final answer based on the analysis..."
  }
}
```""")
        
        if examples:
            guidelines += "\n" + "\n".join(examples)
        
        # Add the complete format guidelines to the system prompt parts
        system_prompt_parts.append(guidelines)
        
        # Combine all parts into the complete system prompt
        complete_system_prompt = "\n\n".join(filter(None, system_prompt_parts))
        
        return complete_system_prompt
    
    def _strip_schema_hints(self, text: str) -> str:
        """
        Remove lines that try to re-explain the standard JSON output contract.
        The official schema will be appended later.
        """
        import re
        
        # Pattern to match schema hint lines
        schema_hint_pattern = re.compile(
            r"(next_action|action_input|tool_calls|JSON\s*object|Response Structure)",
            re.IGNORECASE,
        )
        
        lines = [
            ln
            for ln in text.splitlines()
            if not schema_hint_pattern.search(ln)
        ]
        # Collapse consecutive blank lines that can appear after stripping
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return cleaned
    
    def _get_tool_instructions(self, agent: 'BaseAgent') -> str:
        """Generate tool usage instructions for the system prompt."""
        # Check if agent has tools
        if not hasattr(agent, 'tools') or not agent.tools:
            return ""
        if not hasattr(agent, 'tools_schema') or not agent.tools_schema:
            return ""
        
        prompt_lines = ["\n\n--- AVAILABLE TOOLS ---"]
        prompt_lines.append(
            "When you need to use a tool, your response should include a `tool_calls` field. This field should be a list of JSON objects, where each object represents a tool call."
        )
        prompt_lines.append(
            'Each tool call object must have an `id` (a unique identifier for the call), a `type` field set to "function", and a `function` field.'
        )
        prompt_lines.append(
            "The `function` field must be an object with a `name` (the tool name) and `arguments` (a JSON string of the arguments)."
        )
#         prompt_lines.append("Example of a tool call structure:")
#         prompt_lines.append(
#             """
# ```json
# {
#   "tool_calls": [
#     {
#       "id": "call_abc123",
#       "type": "function",
#       "function": {
#         "name": "tool_name_here",
#         "arguments": "{\\"param1\\": \\"value1\\", \\"param2\\": value2}"
#       }
#     }
#   ]
# }
# ```"""
#         )
        prompt_lines.append("Available tools are:")
        
        for tool_def in agent.tools_schema:
            func_spec = tool_def.get("function", {})
            name = func_spec.get("name", "Unknown tool")
            description = func_spec.get("description", "No description.")
            parameters = func_spec.get("parameters", {})
            prompt_lines.append(f"\nTool: `{name}`")
            prompt_lines.append(f"  Description: {description}")
            if parameters and parameters.get("properties"):
                prompt_lines.append("  Parameters:")
                param_lines = self._format_parameters(
                    parameters.get("properties", {}),
                    parameters.get("required", [])
                )
                prompt_lines.extend(param_lines)
            else:
                prompt_lines.append("  Parameters: None")
        
        prompt_lines.append("--- END AVAILABLE TOOLS ---")
        return "\n".join(prompt_lines)
    
    def _format_parameters(self, properties: Dict, required: List[str], indent: int = 2) -> List[str]:
        """Recursively format parameters including nested structures."""
        lines = []
        for p_name, p_spec in properties.items():
            p_type = p_spec.get("type", "any")
            p_desc = p_spec.get("description", "")
            is_required = p_name in required
            
            # Base parameter line
            lines.append(f"{'  ' * indent}- `{p_name}` ({p_type}): {p_desc} {'(required)' if is_required else ''}")
            
            # If it's an object with properties, show nested structure
            if p_type == "object" and "properties" in p_spec:
                nested_props = p_spec["properties"]
                nested_required = p_spec.get("required", [])
                lines.append(f"{'  ' * (indent + 1)}Nested parameters:")
                lines.extend(self._format_parameters(nested_props, nested_required, indent + 2))
        
        return lines
    
    def _get_peer_agent_instructions(self, agent_name: str, allowed_agents: List[str]) -> str:
        """Generate peer agent invocation instructions for the system prompt."""
        if not allowed_agents:
            return ""
        
        # Import here to avoid circular dependency
        from src.agents.registry import AgentRegistry
        
        prompt_lines = ["\n\n--- AVAILABLE PEER AGENTS ---"]
        prompt_lines.append(
            "You can invoke other agents to assist you. If you choose this path, your JSON response (as described in the general response guidelines) "
            'should set `next_action` to `"invoke_agent"`. The `action_input` field for this action must be an array containing agent invocation objects.'
        )
        
        prompt_lines.append("\n**CRITICAL DECISION PRINCIPLE:**")
        prompt_lines.append(
            "Before invoking any agent(s), ask yourself: 'Do I need the response from these agent(s) to complete my task or make my next decision?'"
        )
        prompt_lines.append(
            "- If YES → Invoke those agents first, wait for their responses, then proceed with your next action"
        )
        prompt_lines.append(
            "- If NO → You may invoke them alongside other agents or pass control directly"
        )
        
        prompt_lines.append("\n**EXECUTION SEMANTICS:**")
        prompt_lines.append(
            "- **Multiple agents in array**: They execute in parallel. Depending on the system topology, "
            "control may return to you with their responses OR flow directly to another designated agent."
        )
        prompt_lines.append(
            "- **Single agent in array**: Standard invocation. Depending on the system topology, "
            "you may receive its response OR it may continue the workflow to another agent."
        )
        prompt_lines.append(
            "- **Key Rule**: NEVER invoke agents together if you need one's response before invoking another. "
            "Invoke them in separate steps based on your information dependencies."
        )
        
        prompt_lines.append("\nEach invocation object must contain:")
        prompt_lines.append(
            "- `agent_name`: (String) The name of the agent to invoke from the list below (must be an exact match)."
        )
        prompt_lines.append(
            "- `request`: (String or Object) The specific task, question, or data payload for the target agent."
        )
        
        prompt_lines.append("\n**EXAMPLES OF CORRECT INVOCATION PATTERNS:**")
        prompt_lines.append("\n✅ CORRECT - When you need responses before proceeding:")
        prompt_lines.append(
            """```
Step 1: Invoke data collection agents (need their data first)
{
  "thought": "I need to collect data from multiple sources before I can proceed",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "DataAgent1", "request": {"query": "..."}},
    {"agent_name": "DataAgent2", "request": {"query": "..."}}
  ]
}

Step 2: After receiving data, invoke processing agent
{
  "thought": "Now that I have the data from both agents, I can send it for processing",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "ProcessingAgent", "request": {"data": "..."}}
  ]
}
```"""
        )
        
        prompt_lines.append("\n❌ INCORRECT - Invoking dependent agents together:")
        prompt_lines.append(
            """```
{
  "thought": "I'll invoke data collectors and processor together",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "DataAgent1", "request": {"query": "..."}},
    {"agent_name": "DataAgent2", "request": {"query": "..."}},
    {"agent_name": "ProcessingAgent", "request": {"data": "???"}}  // ERROR: What data? You don't have it yet!
  ]
}
```"""
        )
        
        prompt_lines.append("\n✅ CORRECT - When agents don't depend on each other:")
        prompt_lines.append(
            """```
{
  "thought": "These analysis tasks are independent and can run in parallel",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "AnalysisAgent1", "request": {"analyze": "dataset_A"}},
    {"agent_name": "AnalysisAgent2", "request": {"analyze": "dataset_B"}},
    {"agent_name": "AnalysisAgent3", "request": {"analyze": "dataset_C"}}
  ]
}
```"""
        )
        
        prompt_lines.append("\nYou are allowed to invoke the following agents:")
        
        for peer_name in allowed_agents:
            # Get instance count information
            try:
                total_instances = AgentRegistry.get_instance_count(peer_name)
                available_instances = AgentRegistry.get_available_count(peer_name)
                
                # Format agent name with instance info
                if total_instances > 1:
                    # It's a pool - show instance availability
                    instance_info = f" (Pool: {available_instances}/{total_instances} instances available)"
                    prompt_lines.append(f"- `{peer_name}`{instance_info}")
                    if available_instances < total_instances:
                        prompt_lines.append(f"  Note: Some instances are currently in use. You can invoke up to {available_instances} in parallel.")
                else:
                    # Single instance agent
                    prompt_lines.append(f"- `{peer_name}` (Single instance)")
            except:
                # If registry lookup fails, just list the agent name
                prompt_lines.append(f"- `{peer_name}`")
            
            # Add simple format note
            prompt_lines.append("  Expected input format: Any string or object")
        
        prompt_lines.append("--- END AVAILABLE PEER AGENTS ---")
        return "\n".join(prompt_lines)
    
    def _get_context_instructions(self, agent: 'BaseAgent') -> str:
        """Generate instructions for handling passed context."""
        instructions = []
        
        # Instructions for receiving context
        instructions.append("\n\n--- CONTEXT HANDLING ---")
        instructions.append(
            "You may receive saved context from other agents in your request. "
            "This context contains important information they've preserved for you, "
            "such as search results, analysis data, or other relevant content."
        )
        instructions.append(
            "Context will appear as '[Saved Context from AgentName]' followed by "
            "organized sections of saved messages."
        )
        
        # Instructions for saving context (if agent has the tool)
        if hasattr(agent, 'tools') and agent.tools and "save_to_context" in agent.tools:
            instructions.append(
                "\nTo pass important information to the next agent or user:"
            )
            instructions.append(
                "1. Use the 'save_to_context' tool to preserve key messages"
            )
            instructions.append(
                "2. Save context BEFORE invoking the next agent or returning final response"
            )
            instructions.append(
                "3. Use descriptive context keys like 'search_results', 'analysis_data', etc."
            )
            instructions.append(
                "4. The saved context will automatically be passed to the next recipient"
            )
            
            instructions.append(
                "\nExample: After receiving tool results, save them before proceeding:"
            )
            instructions.append(
                '{"next_action": "call_tool", "action_input": {"tool_calls": '
                '[{"id": "save_1", "type": "function", "function": '
                '{"name": "save_to_context", "arguments": '
                '{"selection_criteria": {"role_filter": ["tool"]}, '
                '"context_key": "search_results"}}}]}}'
            )
        
        return "\n".join(instructions) if instructions else ""
    
    def _get_json_format_guidelines(self, actions_str: str, action_descriptions: List[str]) -> str:
        """
        Get the strict JSON output format guidelines.
        
        Args:
            actions_str: Comma-separated list of available actions
            action_descriptions: Detailed descriptions for each action
        
        Returns:
            Formatted JSON guidelines string
        """
        return f"""

--- STRICT JSON OUTPUT FORMAT ---
Your *entire* response MUST be a single, valid JSON object.  This JSON object
must be enclosed in a JSON markdown code block, e.g.:
```json
{{ ... }}
```
No other text or additional JSON objects should appear outside this single
markdown block.

Example:
```json
{{
  "thought": "Your reasoning for the chosen action (optional).",
  "next_action": "Must be one of: {actions_str}.",
  "action_input": {{ /* parameters specific to next_action */ }}
}}
```

Detailed structure for the JSON object:
1. `thought` (String, optional) – your internal reasoning.
2. `next_action` (String, required) – {actions_str}.
3. `action_input` (Array/Object, required) – parameters specific to `next_action`.
{chr(10).join(action_descriptions)}
"""
    
    def _get_json_guidelines_concise(self, actions_str: str) -> str:
        """
        Get concise JSON format reminder for steering/error messages.
        Avoids duplication with system prompt to prevent repetition.
        """
        return f"""Respond with a single JSON object in a markdown block: ```json {{"thought": "...", "next_action": "...", "action_input": {{...}}}} ```"""

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
        
        # Add call_tool if agent has tools
        if hasattr(agent, 'tools_schema') and agent.tools_schema:
            available_actions.append("'call_tool'")
            action_descriptions.append(
                '- If `next_action` is `"call_tool"`:\n'
                '     `{"tool_calls": [{"id": "call_123", "type": "function", '
                '"function": {"name": "tool_name", "arguments": "{\\"param\\": \\"value\\"}"}}]}`'
            )
        
        # Always add final_response
        available_actions.append("'final_response'")
        action_descriptions.append(
            '- If `next_action` is `"final_response"`:\n'
            '     `{"content": "Your final answer..."}`'
        )
        
        actions_str = ", ".join(available_actions)

        # Use concise guidelines for steering to avoid duplication with system prompt
        json_guidelines_concise = self._get_json_guidelines_concise(actions_str)

        # Build steering based on retry
        if is_retry:
            return f"""Your previous response had an incorrect format. Let's try again with the correct structure.
{json_guidelines_concise}"""

# You must provide your response as a single JSON object in a markdown code block."""
        else:
            return f"""Make sure to format your response correctly:
{json_guidelines_concise}"""

# You must provide your response as a single JSON object in a markdown code block."""
    
    def _get_agent_request_example(self, agent_name: str) -> str:
        """
        Get an example request format for an agent based on its input schema.
        Returns either a schema-based example or a generic placeholder.
        """
        try:
            # Try to import AgentRegistry to get schema
            from src.agents.registry import AgentRegistry
            
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
            # Check if this is an error Message BEFORE storing content
            if raw_response.role == "error":
                # Mark this as an error response for ValidationProcessor
                result.response = raw_response  # Keep the full Message
                result.is_error_message = True  # Add flag for detection
            else:
                # Normal Message handling
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
        # Special case: error Messages are valid and should be processed by ValidationProcessor
        elif hasattr(result, 'is_error_message') and result.is_error_message:
            # Error messages will be handled by ErrorMessageProcessor in ValidationProcessor
            # Don't mark as invalid or require retry
            pass
        # If no action type, no tools, and no next agent specified, this is an invalid response
        elif not result.action_type and not result.tool_calls and not result.next_agent:
            result.next_agent = context.agent_name  # Stay with same agent
            result.requires_retry = True
            if not hasattr(result, 'metadata'):
                result.metadata = {}
            result.metadata['invalid_response'] = True
            result.metadata['retry_reason'] = 'format_error'
            
            # FIX 3: Generate dynamic format instructions based on allowed actions
            # Get topology graph from context if available
            topology_graph = None
            if hasattr(context, 'topology_graph'):
                topology_graph = context.topology_graph
            elif hasattr(self, 'topology_graph'):
                topology_graph = self.topology_graph
            
            # Generate dynamic instructions
            format_instructions = self._get_dynamic_format_instructions(
                agent, 
                context.branch if hasattr(context, 'branch') else None,
                topology_graph
            )
            
            # Try to get agent-specific format instructions if available (override dynamic)
            if hasattr(agent, '_get_response_guidelines'):
                try:
                    format_instructions = agent._get_response_guidelines()
                except:
                    pass  # Use dynamic format
            elif hasattr(agent, '_get_unified_response_format'):
                try:
                    format_instructions = agent._get_unified_response_format()
                except:
                    pass  # Use dynamic format
            
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
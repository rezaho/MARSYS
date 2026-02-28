"""
Real tool executor that replaces the placeholder implementation.
Handles execution of both environment tools and agent-specific tools.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable
import inspect
from difflib import get_close_matches

if TYPE_CHECKING:
    from ...agents import BaseAgent
    from ..event_bus import EventBus

logger = logging.getLogger(__name__)


def find_similar_tool_names(tool_name: str, available_tools: List[str], cutoff: float = 0.6) -> List[str]:
    """Find similar tool names using fuzzy matching."""
    # Strip common prefixes for matching
    clean_name = tool_name.replace("functions.", "").replace("tools.", "")
    matches = get_close_matches(clean_name, available_tools, n=3, cutoff=cutoff)
    return matches


class RealToolExecutor:
    """Executes actual tools instead of returning mock results."""

    def __init__(self, event_bus: Optional['EventBus'] = None):
        """
        Initialize tool executor with optional event bus.

        Args:
            event_bus: Optional EventBus for emitting tool execution events
        """
        self.event_bus = event_bus
        self.tool_registry = {}
        self.agent_tool_cache = {}  # Cache agent tools to avoid repeated introspection
        self._build_tool_registry()
        
    def _build_tool_registry(self):
        """Build registry of all available tools."""
        # Import standard tools
        try:
            from ...environment.tools import AVAILABLE_TOOLS
            self.tool_registry.update(AVAILABLE_TOOLS)
            
            # Add common aliases for tools
            if 'tool_google_search_api' in AVAILABLE_TOOLS:
                self.tool_registry['google_search'] = AVAILABLE_TOOLS['tool_google_search_api']
            if 'tool_google_search_community' in AVAILABLE_TOOLS:
                self.tool_registry['google_search_community'] = AVAILABLE_TOOLS['tool_google_search_community']
                
            logger.info(f"Loaded {len(AVAILABLE_TOOLS)} tools from environment.tools")
            logger.debug(f"Available tools: {list(self.tool_registry.keys())}")
        except ImportError as e:
            logger.error(f"Failed to import environment tools: {e}")
            
    def _get_agent_tools(self, agent: 'BaseAgent') -> Dict[str, Callable]:
        """Extract tools from agent if it has any."""
        agent_id = id(agent)

        # Get agent's tools version for cache invalidation
        current_version = getattr(agent, '_tools_version', 0)

        # Check cache first - include version check
        if agent_id in self.agent_tool_cache:
            cached_tools, cached_version = self.agent_tool_cache[agent_id]
            if cached_version == current_version:
                return cached_tools
            else:
                logger.debug(
                    f"Cache invalidated for {agent.__class__.__name__} "
                    f"(version {cached_version} -> {current_version})"
                )

        agent_tools = {}
        
        # Check if agent has tools attribute
        if hasattr(agent, 'tools') and isinstance(agent.tools, dict):
            agent_tools.update(agent.tools)
            logger.debug(f"Found {len(agent.tools)} tools in {agent.__class__.__name__}.tools")
            
        # Special handling for BrowserAgent
        if agent.__class__.__name__ == 'BrowserAgent':
            # Get browser-specific methods that are tools
            browser_tool_methods = [
                'extract_content_from_url',
                'navigate_to_url',
                'take_screenshot',
                'click_element',
                'fill_form_field',
                'wait_for_element',
                'execute_javascript',
                'get_page_content',
                'get_element_info',
                'scroll_page'
            ]
            
            for method_name in browser_tool_methods:
                if hasattr(agent, method_name):
                    method = getattr(agent, method_name)
                    if callable(method):
                        agent_tools[method_name] = method
                        logger.debug(f"Added BrowserAgent method: {method_name}")
        
        # Cache the result with version
        self.agent_tool_cache[agent_id] = (agent_tools, current_version)

        return agent_tools
    
    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        agent: 'BaseAgent',
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute tool calls and return real results."""
        results = []
        
        # Get agent-specific tools
        agent_tools = self._get_agent_tools(agent)
        
        logger.info(f"Executing {len(tool_calls)} tool calls for agent {agent.name}")
        
        for tool_call in tool_calls:
            # Initialize variables at the start to ensure they're always defined
            tool_id = ""
            tool_name = "unknown"
            
            try:
                # Handle both dict and ToolCallMsg objects
                if hasattr(tool_call, 'to_dict'):
                    # Convert ToolCallMsg to dict format
                    tool_call_dict = tool_call.to_dict()
                    tool_id = tool_call_dict.get("id", "")
                    function_data = tool_call_dict.get("function", {})
                    tool_name = function_data.get("name", "unknown")
                    tool_args_str = function_data.get("arguments", "{}")
                elif isinstance(tool_call, dict):
                    # Already a dict
                    tool_id = tool_call.get("id", "")
                    function_data = tool_call.get("function", {})
                    tool_name = function_data.get("name", "unknown")
                    tool_args_str = function_data.get("arguments", "{}")
                else:
                    # Try to access attributes directly (for ToolCallMsg objects)
                    tool_id = getattr(tool_call, 'id', "")
                    tool_name = getattr(tool_call, 'name', "unknown")
                    tool_args_str = getattr(tool_call, 'arguments', "{}")
                
                # Parse arguments
                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool arguments for {tool_name}: {e}")
                    tool_args = {}

                # Separate reasoning from tool args for event emission
                event_args = {**tool_args} if isinstance(tool_args, dict) else tool_args
                reasoning = event_args.pop("reasoning", None) if isinstance(event_args, dict) else None

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Emit tool start event if event_bus is available
                start_time = 0
                if self.event_bus:
                    from ..status.events import ToolCallEvent
                    import time

                    start_time = time.time()

                    await self.event_bus.emit(ToolCallEvent(
                        session_id=context.get("session_id", "unknown"),
                        branch_id=context.get("branch_id"),
                        agent_name=agent.name if hasattr(agent, 'name') else str(agent),
                        tool_name=tool_name,
                        status="started",
                        arguments=event_args,
                        reasoning=reasoning
                    ))

                # Find and execute the tool
                tool_func = None
                tool_source = None
                
                # Check agent tools first (higher priority)
                if tool_name in agent_tools:
                    tool_func = agent_tools[tool_name]
                    tool_source = "agent"
                # Check registry tools
                elif tool_name in self.tool_registry:
                    tool_func = self.tool_registry[tool_name]
                    tool_source = "registry"
                # Handle namespaced tools (e.g., functions.google_search)
                elif '.' in tool_name:
                    base_name = tool_name.split('.')[-1]
                    if base_name in self.tool_registry:
                        tool_func = self.tool_registry[base_name]
                        tool_source = "registry"
                    elif base_name in agent_tools:
                        tool_func = agent_tools[base_name]
                        tool_source = "agent"
                
                # Extract base name for special handling checks
                base_tool_name = tool_name.split('.')[-1] if '.' in tool_name else tool_name
                
                # Special handling for context tools (handle both with and without "functions." prefix)
                if base_tool_name == "save_to_context":
                    # Import the implementation function
                    from ...coordination.context_manager import execute_save_to_context
                    
                    # Call with agent injected as first parameter
                    result = await self._execute_single_tool(
                        lambda **kwargs: execute_save_to_context(agent, **kwargs),
                        tool_args,
                        tool_name
                    )
                    
                    results.append({
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": result,  # String result from context tool
                        "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                    })
                    continue

                elif base_tool_name == "preview_saved_context":
                    # Import the implementation function
                    from ...coordination.context_manager import execute_preview_saved_context

                    # Call with agent injected as first parameter
                    result = await self._execute_single_tool(
                        lambda **kwargs: execute_preview_saved_context(agent, **kwargs),
                        tool_args,
                        tool_name
                    )

                    results.append({
                        "tool_call_id": tool_id,
                        "tool_name": tool_name,
                        "result": result,  # String result from context tool
                        "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                    })
                    continue
                
                # # Try variations of google search tool names
                # if not tool_func and 'google' in tool_name.lower() and 'search' in tool_name.lower():
                #     # Try different variations
                #     variations = ['google_search', 'tool_google_search_api', 'tool_google_search_community']
                #     for variant in variations:
                #         if variant in self.tool_registry:
                #             tool_func = self.tool_registry[variant]
                #             tool_source = "registry"
                #             logger.info(f"Found tool {variant} for requested {tool_name}")
                #             break
                
                if tool_func:
                    logger.debug(f"Found tool {tool_name} from {tool_source}")
                    # Execute the tool
                    raw_result = await self._execute_single_tool(tool_func, tool_args, tool_name)

                    # Emit tool complete event
                    if self.event_bus:
                        import time
                        from ..status.events import ToolCallEvent

                        await self.event_bus.emit(ToolCallEvent(
                            session_id=context.get("session_id", "unknown"),
                            branch_id=context.get("branch_id"),
                            agent_name=agent.name if hasattr(agent, 'name') else str(agent),
                            tool_name=tool_name,
                            status="completed",
                            duration=time.time() - start_time if start_time else None
                        ))

                    # Store raw result - either ToolResponse object or string
                    # Step executor will determine message pattern based on result type
                    result = raw_result
                else:
                    # Check if the "tool" name is actually a peer agent name
                    next_agents = set()
                    if hasattr(agent, '_topology_graph_ref') and agent._topology_graph_ref:
                        next_agents = agent._topology_graph_ref.get_next_agents(agent.name)

                    if tool_name in next_agents:
                        full_error = (
                            f"'{tool_name}' is an agent, not a tool. "
                            f"You cannot invoke agents via tool calls.\n\n"
                            f"To invoke the '{tool_name}' agent, respond with JSON in your message content:\n"
                            f"```json\n"
                            f'{{\n'
                            f'  "thought": "your reasoning",\n'
                            f'  "next_action": "invoke_agent",\n'
                            f'  "action_input": [\n'
                            f'    {{\n'
                            f'      "agent_name": "{tool_name}",\n'
                            f'      "request": "your task description"\n'
                            f'    }}\n'
                            f'  ]\n'
                            f'}}\n'
                            f"```\n"
                            f"Do NOT use tool_calls for agent invocation."
                        )
                        logger.warning(
                            f"Agent '{agent.name}' tried to invoke peer '{tool_name}' "
                            f"via tool_calls instead of JSON invoke_agent"
                        )
                    else:
                        # Standard tool-not-found error with fuzzy matching
                        all_tools = list(self.tool_registry.keys()) + list(agent_tools.keys())
                        similar_tools = find_similar_tool_names(tool_name, all_tools)

                        if similar_tools:
                            error_msg = f"Tool '{tool_name}' not found. Did you mean: {similar_tools[0]}?"
                        else:
                            error_msg = f"Tool '{tool_name}' not found."
                        suggestion = f"Available tools: {', '.join(all_tools[:10])}"
                        if len(all_tools) > 10:
                            suggestion += f"... and {len(all_tools) - 10} more"

                        full_error = f"{error_msg} {suggestion}"
                        logger.error(full_error)

                    # Return error as string (will use single-message pattern)
                    result = full_error

                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": result,  # ToolResponse object or string
                    "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)

                # Emit tool failed event
                if self.event_bus:
                    import time
                    from ..status.events import ToolCallEvent

                    await self.event_bus.emit(ToolCallEvent(
                        session_id=context.get("session_id", "unknown"),
                        branch_id=context.get("branch_id"),
                        agent_name=agent.name if hasattr(agent, 'name') else str(agent),
                        tool_name=tool_name,
                        status="failed",
                        duration=time.time() - start_time if 'start_time' in locals() and start_time else None
                    ))

                # Create clear error message for the agent (as string)
                error_msg = f"Tool '{tool_name}' failed: {str(e)}"

                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": error_msg,  # Error as string (single-message pattern)
                    "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                })

        return results
    
    async def _execute_single_tool(self, tool_func: Callable, tool_args: Dict[str, Any], tool_name: str) -> Any:
        """Execute a single tool, handling async/sync differences."""
        try:
            # Log the actual function being called
            logger.debug(f"Executing {tool_func.__module__}.{tool_func.__name__} with args: {tool_args}")
            
            # Check if the tool is async
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_args)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool_func(**tool_args))
            
            # Log successful execution
            logger.info(f"Tool {tool_name} executed successfully")

            # Process result based on type
            # Two-message pattern: ONLY for ToolResponse objects
            # Single-message pattern: All other results (converted to string)
            from marsys.environment.tool_response import ToolResponse

            if isinstance(result, ToolResponse):
                # ToolResponse: use as-is (will trigger two-message pattern in step_executor)
                # ToolResponse.to_content_array() handles proper typed array conversion
                return result
            else:
                # Everything else: stringify for single-message pattern
                # This includes: strings, dicts, lists, numbers, etc.
                if isinstance(result, (dict, list)):
                    # Convert to compact JSON string
                    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
                else:
                    # Convert to string
                    return str(result)
                
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}", exc_info=True)

            # Provide helpful error message for parameter structure issues
            error_str = str(e)
            if "unexpected keyword argument" in error_str:
                # Import re for pattern matching
                import re
                match = re.search(r"unexpected keyword argument '(\w+)'", error_str)
                param_name = match.group(1) if match else "unknown"
                
                # Check if this is a known nested parameter issue
                if tool_name.endswith("save_to_context") and param_name in ["tool_names", "role_filter", "last_n_tools", "message_ids", "content_pattern"]:
                    return {
                        "error": f"Parameter '{param_name}' should be nested inside 'selection_criteria'",
                        "type": "ParameterStructureError",
                        "hint": f"The '{param_name}' parameter must be inside the 'selection_criteria' object",
                        "correct_format": {
                            "description": "Use this structure for save_to_context:",
                            "example": {
                                "selection_criteria": {
                                    param_name: "... your value here ...",
                                    "# other criteria": "..."
                                },
                                "context_key": "your_key_here"
                            }
                        },
                        "full_example": {
                            "selection_criteria": {
                                "tool_names": ["google_search"],
                                "role_filter": ["tool"],
                                "last_n_tools": 1
                            },
                            "context_key": "search_results"
                        }
                    }
                
                # Generic message for other tools with nested parameters
                return {
                    "error": f"Parameter '{param_name}' is not recognized at the top level",
                    "type": "ParameterStructureError", 
                    "hint": f"Check if '{param_name}' should be nested inside another parameter object",
                    "suggestion": "Review the tool's parameter structure in the schema. Parameters marked as 'object' type contain nested parameters."
                }
            
            # Default error for other cases
            return {"error": str(e), "type": type(e).__name__}
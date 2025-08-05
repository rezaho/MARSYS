"""
Real tool executor that replaces the placeholder implementation.
Handles execution of both environment tools and agent-specific tools.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable
import inspect

if TYPE_CHECKING:
    from ...agents import BaseAgent

logger = logging.getLogger(__name__)


class RealToolExecutor:
    """Executes actual tools instead of returning mock results."""
    
    def __init__(self):
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
        
        # Check cache first
        if agent_id in self.agent_tool_cache:
            return self.agent_tool_cache[agent_id]
        
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
        
        # Cache the result
        self.agent_tool_cache[agent_id] = agent_tools
        
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
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
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
                
                # Special handling for context tools
                if tool_name == "save_to_context":
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
                        "result": result,
                        "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                    })
                    continue

                elif tool_name == "preview_saved_context":
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
                        "result": result,
                        "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                    })
                    continue
                
                # Try variations of google search tool names
                if not tool_func and 'google' in tool_name.lower() and 'search' in tool_name.lower():
                    # Try different variations
                    variations = ['google_search', 'tool_google_search_api', 'tool_google_search_community']
                    for variant in variations:
                        if variant in self.tool_registry:
                            tool_func = self.tool_registry[variant]
                            tool_source = "registry"
                            logger.info(f"Found tool {variant} for requested {tool_name}")
                            break
                
                if tool_func:
                    logger.debug(f"Found tool {tool_name} from {tool_source}")
                    # Execute the tool
                    result = await self._execute_single_tool(tool_func, tool_args, tool_name)
                else:
                    error_msg = f"Tool not found: {tool_name}"
                    logger.error(f"{error_msg}. Available tools: {list(self.tool_registry.keys())} + {list(agent_tools.keys())}")
                    result = {"error": error_msg}
                
                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": result,
                    "_origin": tool_call.get('_origin', 'response') if isinstance(tool_call, dict) else 'response'
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
                results.append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "result": {"error": str(e), "type": type(e).__name__},
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
            
            # Ensure result is JSON serializable
            if isinstance(result, str):
                try:
                    # Try to parse if it's JSON string
                    parsed = json.loads(result)
                    return parsed
                except json.JSONDecodeError:
                    # Return as content if not JSON
                    return {"content": result}
            elif isinstance(result, dict):
                return result
            elif isinstance(result, list):
                return {"results": result}
            else:
                return {"result": str(result)}
                
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}", exc_info=True)
            return {"error": str(e), "type": type(e).__name__}
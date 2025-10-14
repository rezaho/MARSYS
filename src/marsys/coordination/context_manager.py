"""
Context management system for agent message passing.

This module provides functionality for agents to selectively pass context
(messages from their memory) to other agents or back to the user. It treats
context selection as a special tool that agents can use strategically.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

if TYPE_CHECKING:
    from ..agents import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ContextTemplate:
    """
    Represents a template for context insertion in agent responses.
    
    This allows agents to define how selected context should be
    incorporated into their final output.
    """
    template_string: str
    context_keys: List[str] = field(default_factory=list)
    placeholders: Dict[str, str] = field(default_factory=dict)
    
    def render(self, context_data: Dict[str, List[Dict]]) -> str:
        """
        Render the template with actual context data.
        
        Args:
            context_data: Dictionary mapping context keys to message lists
            
        Returns:
            Rendered template string with context inserted
        """
        rendered = self.template_string
        
        for key, placeholder in self.placeholders.items():
            if key in context_data:
                # Replace placeholder with actual content
                messages = context_data[key]
                content_parts = []
                for msg in messages:
                    content_parts.append(str(msg.get('content', '')))
                
                full_content = '\n\n'.join(content_parts)
                rendered = rendered.replace(placeholder, full_content)
        
        return rendered


class ContextSelector:
    """
    Manages context selection for an agent during a single run.
    
    This class handles saving message selections and preparing them
    for passing to the next agent or returning to the user. Context
    is reset at the beginning of each new run.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the context selector.
        
        Args:
            agent_name: Name of the agent this selector belongs to
        """
        self.agent_name = agent_name
        self.saved_selections: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"ContextSelector.{agent_name}")
        
    def reset(self) -> None:
        """Reset all saved selections for a new run."""
        self.saved_selections.clear()
        self.logger.debug("Context selections reset")
        
    def save_selection(self, criteria: Dict[str, Any], key: str) -> None:
        """
        Save a selection criteria with a key.
        
        Args:
            criteria: Selection criteria dictionary containing:
                     - message_ids: List of specific message IDs
                     - role_filter: Filter by message role(s)
                     - tool_names: Filter by tool name(s)
                     - last_n_tools: Include last N tool responses
                     - content_pattern: Regex pattern to match
            key: Identifier for this selection (e.g., 'search_results')
        """
        self.saved_selections[key] = {
            'criteria': criteria,
            'timestamp': time.time()
        }
        self.logger.info(f"Saved context selection with key '{key}': {criteria}")
        self.logger.info(f"Current saved_selections keys: {list(self.saved_selections.keys())}")
        
    def get_saved_context(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, List[Dict]]]:
        """
        Get all saved context based on stored selection criteria.
        
        Args:
            messages: List of all messages from agent's memory
            
        Returns:
            Dictionary mapping keys to lists of selected messages,
            or None if no selections saved
        """
        self.logger.info(f"Getting saved context. saved_selections keys: {list(self.saved_selections.keys())}")
        
        if not self.saved_selections:
            self.logger.info("No saved selections found")
            return None
            
        context_data = {}
        for key, selection_info in self.saved_selections.items():
            criteria = selection_info['criteria']
            selected_messages = self._select_messages(messages, criteria)
            if selected_messages:
                context_data[key] = selected_messages
                self.logger.info(f"Selected {len(selected_messages)} messages for key '{key}'")
                
        return context_data if context_data else None
    
    def _select_messages(self, messages: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Select messages based on the given criteria.
        
        Args:
            messages: All available messages
            criteria: Selection criteria
            
        Returns:
            List of selected messages
        """
        selected = []
        
        # Apply various selection criteria
        for i, msg in enumerate(messages):
            # Message ID selection (using index as ID for now)
            if 'message_ids' in criteria:
                # Check if message has an ID field
                msg_id = msg.get('message_id', str(i))
                if msg_id in criteria['message_ids']:
                    selected.append(msg)
                    continue
                    
            # Role filter
            if 'role_filter' in criteria:
                roles = criteria['role_filter'] if isinstance(criteria['role_filter'], list) else [criteria['role_filter']]
                if msg.get('role') in roles:
                    selected.append(msg)
                    continue
                    
            # Tool name filter
            if 'tool_names' in criteria:
                tool_names = criteria['tool_names'] if isinstance(criteria['tool_names'], list) else [criteria['tool_names']]
                if msg.get('role') == 'tool' and msg.get('name') in tool_names:
                    selected.append(msg)
                    continue
                    
            # Content pattern matching
            if 'content_pattern' in criteria:
                try:
                    pattern = re.compile(criteria['content_pattern'], re.IGNORECASE | re.MULTILINE)
                    content = str(msg.get('content', ''))
                    if pattern.search(content):
                        selected.append(msg)
                except re.error as e:
                    self.logger.warning(f"Invalid regex pattern: {e}")
        
        # Last N tools special case
        if 'last_n_tools' in criteria:
            n = criteria['last_n_tools']
            tool_msgs = [msg for msg in messages if msg.get('role') == 'tool']
            # Add the last N tool messages (might overlap with already selected)
            selected.extend(tool_msgs[-n:])
            
        # Remove duplicates while preserving order
        seen = set()
        unique_selected = []
        for msg in selected:
            # Create a unique identifier for the message
            msg_key = (
                msg.get('message_id', ''),
                msg.get('role', ''),
                str(msg.get('content', ''))[:100]  # First 100 chars of content
            )
            if msg_key not in seen:
                seen.add(msg_key)
                unique_selected.append(msg)
                
        return unique_selected
    
    def get_preview(self, messages: List[Dict[str, Any]], max_length: int = 150) -> Dict[str, List[Dict[str, str]]]:
        """
        Get a preview of saved context without full content.
        
        This is useful for showing the agent what context they have saved
        without saturating their context window.
        
        Args:
            messages: All available messages
            max_length: Maximum length for content preview
            
        Returns:
            Dictionary with previews of saved messages
        """
        preview_data = {}
        context_data = self.get_saved_context(messages)
        
        if not context_data:
            return {}
            
        for key, selected_messages in context_data.items():
            previews = []
            for msg in selected_messages:
                preview = {
                    'role': msg.get('role', 'unknown'),
                    'preview': self._get_message_preview(msg, max_length),
                }
                if msg.get('name'):
                    preview['name'] = msg['name']
                if msg.get('message_id'):
                    preview['id'] = msg['message_id']
                previews.append(preview)
            preview_data[key] = previews
            
        return preview_data
    
    def _get_message_preview(self, msg: Dict[str, Any], max_length: int) -> str:
        """Create a preview of a message's content."""
        content = msg.get('content', '')
        
        # Handle different content types
        if isinstance(content, dict):
            # For structured content, show a JSON preview
            content_str = json.dumps(content, indent=2)
        else:
            content_str = str(content)
        
        # Truncate if necessary
        if len(content_str) > max_length:
            return content_str[:max_length] + "..."
        return content_str
    
    def create_template(self, template_string: str) -> ContextTemplate:
        """
        Create a context template for rendering final output.
        
        Args:
            template_string: Template with placeholders like {{search_results}}
            
        Returns:
            ContextTemplate object
        """
        # Extract placeholders from template
        placeholder_pattern = re.compile(r'\{\{(\w+)\}\}')
        matches = placeholder_pattern.findall(template_string)
        
        context_keys = []
        placeholders = {}
        
        for match in matches:
            if match in self.saved_selections:
                context_keys.append(match)
                placeholders[match] = f"{{{{{match}}}}}"
        
        return ContextTemplate(
            template_string=template_string,
            context_keys=context_keys,
            placeholders=placeholders
        )
    
    def format_for_llm(self, messages: List[Dict[str, Any]], include_stats: bool = True) -> str:
        """
        Format saved context information for LLM understanding.
        
        This helps the LLM understand how to use context templates
        in their responses.
        
        Args:
            messages: All available messages
            include_stats: Whether to include statistics
            
        Returns:
            Formatted string explaining saved context
        """
        preview = self.get_preview(messages)
        
        if not preview:
            return "No context saved yet. Use save_to_context tool to save important messages."
        
        lines = ["Saved Context:"]
        
        for key, previews in preview.items():
            lines.append(f"\n[{key}] ({len(previews)} messages)")
            if include_stats:
                # Show role distribution
                role_counts = {}
                for p in previews:
                    role = p['role']
                    role_counts[role] = role_counts.get(role, 0) + 1
                stats = ", ".join([f"{role}: {count}" for role, count in role_counts.items()])
                lines.append(f"  Types: {stats}")
            
            # Show first few previews
            for i, p in enumerate(previews[:3]):
                role = p['role']
                name = f" ({p['name']})" if p.get('name') else ""
                preview_text = p['preview'].replace('\n', ' ')[:100]
                lines.append(f"  [{i+1}] {role}{name}: {preview_text}")
            
            if len(previews) > 3:
                lines.append(f"  ... and {len(previews) - 3} more messages")
        
        lines.append("\nTo use in response: reference with {{key}} in your template")
        
        return '\n'.join(lines)


def get_context_selection_tools() -> List[Dict[str, Any]]:
    """
    Get the tool schemas for context selection functionality.
    
    Returns:
        List of tool schemas in OpenAI format
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "save_to_context",
                "description": "Save EXISTING messages from your conversation history to context for passing to next agent. Use this AFTER receiving tool results (like search results) to preserve them. DO NOT reference message IDs that don't exist yet - use tool_names instead to save outputs from specific tools.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selection_criteria": {
                            "type": "object",
                            "description": "Criteria for selecting messages to save",
                            "properties": {
                                "message_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "IDs of EXISTING messages in your conversation history. DO NOT use IDs you're about to create - those don't exist yet!"
                                },
                                "role_filter": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": ["user", "assistant", "tool", "system"]},
                                    "description": "Include messages with these roles"
                                },
                                "tool_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Save all outputs from these tool names (e.g., ['google_search'] to save all search results). Use this instead of message_ids for saving tool outputs."
                                },
                                "last_n_tools": {
                                    "type": "integer",
                                    "description": "Include last N tool responses",
                                    "minimum": 1
                                },
                                "content_pattern": {
                                    "type": "string",
                                    "description": "Regex pattern to match in message content"
                                }
                            },
                            "required": []
                        },
                        "context_key": {
                            "type": "string",
                            "description": "Key to identify this context selection (e.g., 'search_results', 'analysis_data', 'sources')"
                        }
                    },
                    "required": ["selection_criteria", "context_key"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "preview_saved_context",
                "description": "Preview the messages currently saved in context. Shows a summary without full content to help you format your final response appropriately.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_stats": {
                            "type": "boolean",
                            "description": "Include statistics about saved messages",
                            "default": True
                        }
                    },
                    "required": []
                }
            }
        }
    ]
    
    return tools


# LLM-friendly functions for schema generation and agent's tools dict

def save_to_context(
    selection_criteria: Dict[str, Any],
    context_key: str = "default"
) -> Dict[str, Any]:
    """
    Save EXISTING messages from conversation history to context for passing to next agent.
    
    Use this AFTER receiving tool results to preserve them for the next agent.
    DO NOT try to save messages that don't exist yet.
    
    Args:
        selection_criteria: Criteria for selecting EXISTING messages to save
            - message_ids: List of EXISTING message IDs (don't use IDs you're about to create!)
            - role_filter: Include messages with these roles ["user", "assistant", "tool", "system"]
            - tool_names: Include outputs from these specific tools (recommended for saving tool results)
            - last_n_tools: Include last N tool responses
            - content_pattern: Regex pattern to match in message content
        context_key: Key to identify this context selection (e.g., 'search_results', 'analysis_data')
        
    Returns:
        Status of the save operation
    """
    # This function is never actually called - tool_executor intercepts it
    raise RuntimeError("This tool should be executed by the tool executor with special handling")


def preview_saved_context(
    include_stats: bool = True
) -> Dict[str, Any]:
    """
    Preview the messages currently saved in context.
    
    Shows a summary without full content to help you format your final response appropriately.
    
    Args:
        include_stats: Include statistics about saved messages
        
    Returns:
        Preview of saved context with message summaries
    """
    # This function is never actually called - tool_executor intercepts it
    raise RuntimeError("This tool should be executed by the tool executor with special handling")


# Implementation functions for tool executor (with agent parameter)

def execute_save_to_context(
    agent: 'BaseAgent',
    selection_criteria: Dict[str, Any],
    context_key: str = "default"
) -> Dict[str, Any]:
    """
    Execute save_to_context tool with agent context.
    
    This is the actual implementation called by tool_executor.
    The agent parameter is injected by the executor and not exposed to LLM.
    
    Args:
        agent: The agent instance (injected by tool executor)
        selection_criteria: Criteria for selecting messages
        context_key: Key to identify this context selection
        
    Returns:
        Status of the save operation
    """
    if not hasattr(agent, '_context_selector'):
        return {
            "status": "error",
            "message": "Agent does not support context selection"
        }
    
    try:
        agent._context_selector.save_selection(
            selection_criteria,
            context_key
        )
        logger.debug(f"Context saved for agent {agent.name} with key '{context_key}'")
        return {
            "status": "success",
            "message": f"Context saved with key: {context_key}"
        }
    except Exception as e:
        logger.error(f"Error saving context for agent {agent.name}: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def execute_preview_saved_context(
    agent: 'BaseAgent',
    include_stats: bool = True
) -> Dict[str, Any]:
    """
    Execute preview_saved_context tool with agent context.
    
    This is the actual implementation called by tool_executor.
    The agent parameter is injected by the executor and not exposed to LLM.
    
    Args:
        agent: The agent instance (injected by tool executor)
        include_stats: Include statistics about saved messages
        
    Returns:
        Preview of saved context or error
    """
    if not hasattr(agent, '_context_selector') or not hasattr(agent, 'memory'):
        return {"error": "Context preview not available"}
    
    try:
        messages = agent.memory.retrieve_all()
        preview = agent._context_selector.get_preview(messages)
        
        result = {"preview": preview}
        
        if include_stats and preview:
            stats = {
                "total_contexts": len(preview),
                "total_messages": sum(len(msgs) for msgs in preview.values())
            }
            result["stats"] = stats
        
        logger.debug(f"Context preview generated for agent {agent.name}")
        return result
    except Exception as e:
        logger.error(f"Error previewing context for agent {agent.name}: {e}")
        return {"error": str(e)}
"""
Robust JSON parsing utilities for model responses.
Handles various edge cases in LLM-generated JSON content.
"""

import json
import re
from typing import Any, Dict, List


def close_json_braces(src: str) -> str:
    """
    Appends missing closing braces/brackets so that a truncation at the end of
    a model response does not break json.loads.
    
    Args:
        src: JSON string that may have missing closing braces
        
    Returns:
        JSON string with properly closed braces/brackets
    """
    stack: list[str] = []
    pairs = {"{": "}", "[": "]"}
    for ch in src:
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values() and stack and stack[-1] == ch:
            stack.pop()
    return src + "".join(reversed(stack))


def extract_json_from_markdown(content: str) -> str:
    """Extract JSON content from markdown code blocks."""
    # Pattern to match ```json ... ``` with optional whitespace
    json_block_pattern = r'```json\s*\n?(.*?)\n?```'
    match = re.search(json_block_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Check for ``` ... ``` without json specifier (fallback)
    generic_block_pattern = r'```\s*\n?(.*?)\n?```'
    match = re.search(generic_block_pattern, content, re.DOTALL)
    
    if match:
        extracted = match.group(1).strip()
        # Only use if it looks like JSON (starts with { or [)
        if extracted.startswith(('{', '[')):
            return extracted
    
    # Return original content if no code block found
    return content


def parse_multiple_json_objects(content: str) -> List[Dict[str, Any]]:
    """Parse multiple concatenated JSON objects into a list."""
    json_str_clean = content.strip()
    
    # Only check if it looks like JSON
    if not json_str_clean.startswith('{'):
        return []
    
    json_objects = []
    current_pos = 0
    
    while current_pos < len(json_str_clean):
        # Skip whitespace
        while current_pos < len(json_str_clean) and json_str_clean[current_pos].isspace():
            current_pos += 1
        
        if current_pos >= len(json_str_clean):
            break
        
        # Find the end of the current JSON object
        brace_count = 0
        object_start = current_pos
        object_end = -1
        
        for i in range(current_pos, len(json_str_clean)):
            char = json_str_clean[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    object_end = i + 1
                    break
        
        if object_end == -1:
            # Incomplete JSON object, try to close it
            remaining_content = json_str_clean[object_start:]
            closed_content = close_json_braces(remaining_content)
            try:
                parsed_obj = json.loads(closed_content)
                if isinstance(parsed_obj, dict):
                    json_objects.append(parsed_obj)
            except json.JSONDecodeError:
                pass  # Skip invalid JSON
            break
        
        # Extract and parse the JSON object
        json_str = json_str_clean[object_start:object_end]
        try:
            parsed_obj = json.loads(json_str)
            if isinstance(parsed_obj, dict):
                json_objects.append(parsed_obj)
        except json.JSONDecodeError:
            pass  # Skip invalid JSON
        
        current_pos = object_end
    
    return json_objects


def merge_multiple_json_objects(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple JSON objects by extracting their actions into appropriate fields.
    
    Args:
        objects: List of parsed JSON objects
        
    Returns:
        A single merged object with combined tool_calls and agent_calls
    """
    merged_tool_calls = []
    merged_agent_calls = []
    thoughts = []
    final_response = None
    
    for obj in objects:
        next_action = obj.get("next_action")
        action_input = obj.get("action_input", {})
        thought = obj.get("thought")
        
        if thought:
            thoughts.append(thought)
        
        # Handle standard call_tool action
        if next_action == "call_tool" and isinstance(action_input, dict):
            tool_calls = action_input.get("tool_calls", [])
            if isinstance(tool_calls, list):
                merged_tool_calls.extend(tool_calls)

        elif next_action == "invoke_agent" and isinstance(action_input, dict):
            agent_name = action_input.get("agent_name")
            request = action_input.get("request")
            if agent_name:
                merged_agent_calls.append({
                    "agent_name": agent_name,
                    "request": request
                })
        
        elif next_action == "final_response" and isinstance(action_input, dict):
            if final_response is None:  # Use the first final_response found
                final_response = action_input.get("response")
    
    # Build the merged result
    if final_response is not None:
        # If there's a final response, prioritize that
        return {
            "thought": " | ".join(thoughts) if thoughts else None,
            "next_action": "final_response",
            "action_input": {"response": final_response}
        }
    elif merged_tool_calls:
        # If there are tool calls, return them
        return {
            "thought": " | ".join(thoughts) if thoughts else None,
            "next_action": "call_tool",
            "action_input": {"tool_calls": merged_tool_calls}
        }
    elif merged_agent_calls:
        # If there are agent calls, return the first one (agents typically handle one at a time)
        return {
            "thought": " | ".join(thoughts) if thoughts else None,
            "next_action": "invoke_agent",
            "action_input": merged_agent_calls[0]
        }
    else:
        # No valid actions found, return an error structure
        raise ValueError(
            f"Multiple JSON objects detected but no valid actions found. "
            f"Objects: {objects}"
        )


def robust_json_loads(src: str, max_depth: int = 3) -> Dict[str, Any]:
    """
    Attempts to load JSON with support for recursive/nested JSON strings.
    
    This method handles cases where:
    1. JSON wrapped in markdown code blocks (```json...```)
    2. JSON might have missing closing braces (auto-closes them)
    3. JSON content might be nested/double-encoded as strings
    4. Multiple levels of JSON encoding exist
    5. Multiple concatenated JSON objects (invalid format)
    
    Args:
        src: The source string to parse
        max_depth: Maximum recursion depth to prevent infinite loops
        
    Returns:
        Parsed dictionary
        
    Raises:
        json.JSONDecodeError: If parsing fails after all attempts
        ValueError: If multiple concatenated JSON objects are detected
    """
    
    def try_parse_recursive(content: str, depth: int = 0) -> Dict[str, Any]:
        if depth >= max_depth:
            raise json.JSONDecodeError("Maximum recursion depth reached", content, 0)
        
        # Extract JSON from markdown code blocks on first attempt
        if depth == 0:
            content = extract_json_from_markdown(content)
            
            # Check for multiple JSON objects
            multiple_objects = parse_multiple_json_objects(content)
            if len(multiple_objects) > 1:
                # Multiple objects - try to merge them if they contain actions
                try:
                    return merge_multiple_json_objects(multiple_objects)
                except ValueError:
                    # If merging fails (no valid actions), just return the first object
                    parsed = multiple_objects[0]
            elif len(multiple_objects) == 1:
                # Single object from multiple parser - continue with normal processing
                parsed = multiple_objects[0]
            else:
                # No objects found by multiple parser, try normal JSON parsing
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as e:
                    # Try auto-closing braces
                    closed_content = close_json_braces(content)
                    parsed = json.loads(closed_content)
        else:
            # For recursive calls, just try normal JSON parsing
            parsed = json.loads(content)
        
        # If we got a dictionary, check if any values are JSON strings that need parsing
        if isinstance(parsed, dict):
            def parse_nested_json(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, str) and value.strip().startswith(('{', '[')):
                            try:
                                # Try to recursively parse this value
                                obj[key] = try_parse_recursive(value, depth + 1)
                            except json.JSONDecodeError:
                                # If parsing fails, keep the original string value
                                pass
                        elif isinstance(value, (dict, list)):
                            parse_nested_json(value)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, str) and item.strip().startswith(('{', '[')):
                            try:
                                obj[i] = try_parse_recursive(item, depth + 1)
                            except json.JSONDecodeError:
                                pass
                        elif isinstance(item, (dict, list)):
                            parse_nested_json(item)
            
            parse_nested_json(parsed)
        
        return parsed if isinstance(parsed, dict) else {"content": parsed}
    
    return try_parse_recursive(src) 
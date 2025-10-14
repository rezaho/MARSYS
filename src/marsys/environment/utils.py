import inspect
import json
from typing import Callable, Dict, Any, List, get_type_hints, Union, Literal
import re
import logging

logger = logging.getLogger(__name__)

def _parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    Parses a docstring to extract the main description and parameter descriptions.
    Supports Google-style docstrings for parameter descriptions.
    E.g.:
    Args:
        param_name (param_type): Description of the parameter.
                                 Can span multiple lines.
    """
    if not docstring:
        return {"description": "", "params": {}}

    # Normalize line endings and remove leading/trailing whitespace from the whole docstring
    docstring = docstring.replace('\r\n', '\n').strip()
    lines = docstring.splitlines()
    
    main_description_lines = []
    param_descriptions = {}
    
    # Phase 1: Extract main description (everything before "Args:", "Parameters:", etc.)
    # or the first paragraph if no args section.
    args_section_index = -1
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if re.match(r"^(Args|Arguments|Parameters):$", stripped_line, re.IGNORECASE):
            args_section_index = i
            break
        if not stripped_line and main_description_lines: # End of first paragraph
            # Check if the next non-empty line starts an Args section
            is_next_line_args = False
            for next_line_idx in range(i + 1, len(lines)):
                if lines[next_line_idx].strip():
                    if re.match(r"^(Args|Arguments|Parameters):$", lines[next_line_idx].strip(), re.IGNORECASE):
                        is_next_line_args = True
                    break
            if is_next_line_args: # If it does, this paragraph break is just before Args
                 args_section_index = i # Mark this point to stop main description
                 break 
            # Otherwise, it's a real paragraph break in the main description
            main_description_lines.append("") # Keep paragraph break
            continue
        main_description_lines.append(stripped_line)

    if args_section_index == -1: # No "Args:" section found
        main_description = " ".join(l for l in main_description_lines if l).strip()
    else:
        main_description = " ".join(l for l in main_description_lines[:args_section_index] if l).strip()

    # Phase 2: Parse parameter descriptions if "Args:" section exists
    if args_section_index != -1:
        # Regex to capture "param_name (param_type): description" or "param_name: description"
        # It handles optional type hints in parentheses.
        param_pattern = re.compile(r"^\s*(\w+)\s*(?:\(([^)]+)\))?:\s*(.*)")
        
        current_param_name = None
        current_param_desc_lines = []

        for i in range(args_section_index + 1, len(lines)):
            line = lines[i]
            # Skip empty lines in the args section
            if not line.strip():
                continue

            match = param_pattern.match(line)
            
            # Check if the line is indented (continuation of previous parameter's description)
            # A common indentation is 4 spaces, but could be more.
            is_indented_continuation = line.startswith("    ") and not match

            if match: # New parameter starts
                if current_param_name and current_param_desc_lines: # Save previous param
                    param_descriptions[current_param_name] = " ".join(current_param_desc_lines).strip()
                
                current_param_name = match.group(1)
                # Description starts after the colon, group 3
                current_param_desc_lines = [match.group(3).strip()]
            elif current_param_name and is_indented_continuation: # Continuation of previous param description
                 current_param_desc_lines.append(line.strip())
            elif current_param_name: # Line doesn't match new param or continuation, end current param
                if current_param_desc_lines: # Save if there's anything
                     param_descriptions[current_param_name] = " ".join(current_param_desc_lines).strip()
                current_param_name = None # Reset
                current_param_desc_lines = []
                # This line might be something unexpected, or start of a new section (e.g. Returns:)
                # For simplicity, we stop parsing params here if structure is broken.
                # A more robust parser might look for "Returns:", "Yields:", etc. to delimit sections.
                if re.match(r"^(Returns|Yields|Raises):$", line.strip(), re.IGNORECASE):
                    break


        if current_param_name and current_param_desc_lines: # Save the last parameter
            param_descriptions[current_param_name] = " ".join(current_param_desc_lines).strip()
    
    if not main_description and lines: # Fallback if parsing was difficult
        main_description = lines[0].strip()
        logger.warning(f"Could not parse main description from docstring, using first line: '{docstring}'")


    return {"description": main_description, "params": param_descriptions}


def _map_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Maps Python types to JSON schema type definitions."""
    if py_type == str:
        return {"type": "string"}
    if py_type == int:
        return {"type": "integer"}
    if py_type == float:
        return {"type": "number"}
    if py_type == bool:
        return {"type": "boolean"}
    if py_type == list or getattr(py_type, "__origin__", None) == list:
        args = getattr(py_type, "__args__", ())
        if args and len(args) == 1: # For List[T]
            item_schema = _map_type_to_json_schema(args[0])
        else: # For plain list or List without specific type
            item_schema = {"type": "string"} # OpenAI default or make it "any" if possible
        return {"type": "array", "items": item_schema}
    if py_type == dict or getattr(py_type, "__origin__", None) == dict:
        # For Dict[K, V], OpenAI schema doesn't directly support typed key/value in properties
        # It expects "object" and you can describe typical properties if known,
        # or use additionalProperties. For generic dict, just "object".
        return {"type": "object", "additionalProperties": True} # Allows any properties
    
    # Handle typing.Literal
    if getattr(py_type, "__origin__", None) == Literal:
        args = getattr(py_type, "__args__", ())
        # Assuming Literal values are strings for enum, could be other types
        enum_type = "string"
        if args and isinstance(args[0], int):
            enum_type = "integer"
        elif args and isinstance(args[0], float):
            enum_type = "number"
        elif args and isinstance(args[0], bool):
            enum_type = "boolean"
        return {"type": enum_type, "enum": list(args)}

    # Handle typing.Union for Optional types (e.g., Union[str, NoneType])
    if getattr(py_type, "__origin__", None) == Union:
        args = getattr(py_type, "__args__", ())
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            # This is an Optional type, schema is for the non-None part
            return _map_type_to_json_schema(non_none_types[0])
        else:
            # For Union of multiple types (e.g., Union[str, int]), OpenAI schema doesn't directly support 'anyOf'.
            # Default to string or the first type.
            logger.warning(f"Complex Union type {py_type} encountered. Defaulting to schema of first type or string.")
            if non_none_types:
                return _map_type_to_json_schema(non_none_types[0])
            
    logger.warning(f"Unsupported type {py_type} for JSON schema mapping. Defaulting to 'string'.")
    return {"type": "string"} 


def generate_openai_tool_schema(func: Callable, func_name: str) -> Dict[str, Any]:
    """
    Generates an OpenAI-compatible tool schema from a Python function.

    Args:
        func: The function to generate a schema for.
        func_name: The name to use for the tool in the schema.

    Returns:
        A dictionary representing the tool schema.
    """
    sig = inspect.signature(func)
    # Use get_type_hints to resolve forward references and get actual types
    try:
        type_hints = get_type_hints(func)
    except Exception as e:
        logger.error(f"Could not get type hints for function {func_name}: {e}. Using annotations directly.")
        type_hints = {name: param.annotation for name, param in sig.parameters.items()}
        if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
            type_hints['return'] = func.__annotations__['return']


    docstring_info = _parse_docstring(inspect.getdoc(func))

    properties = {}
    required_params = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"): # Skip self/cls for methods
            continue

        param_type_hint = type_hints.get(name, param.annotation)
        
        # If no type hint is available (param.annotation is inspect.Parameter.empty)
        # and not found in type_hints (e.g. due to parsing error or it's truly missing)
        if param_type_hint == inspect.Parameter.empty:
            logger.warning(f"No type hint found for parameter '{name}' in function '{func_name}'. Defaulting to 'string'.")
            param_type_hint = str 

        json_type_def = _map_type_to_json_schema(param_type_hint)
        
        properties[name] = {
            **json_type_def,
            "description": docstring_info["params"].get(name, f"Parameter '{name}'")
        }
        
        if param.default == inspect.Parameter.empty:
            # Check if it's Optional by looking at Union with NoneType in original hint
            is_optional = False
            original_annotation = param.annotation # Use original annotation for Optional check
            if hasattr(original_annotation, "__origin__") and original_annotation.__origin__ == Union:
                 if type(None) in getattr(original_annotation, "__args__", ()):
                     is_optional = True
            
            if not is_optional:
                required_params.append(name)


    schema = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring_info["description"] or f"Executes the {func_name} tool.",
            "parameters": {
                "type": "object",
                "properties": properties,
            },
        },
    }
    if required_params: # Only add 'required' if there are any
        schema["function"]["parameters"]["required"] = required_params
        
    return schema

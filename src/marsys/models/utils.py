import json
from typing import Any, Dict, List


def detect_model_family(model_name: str) -> str:
    """
    Detect the model family from the model name to determine which reasoning
    parameters are supported.

    Args:
        model_name: Full model name (e.g., "openai/gpt-5", "google/gemini-2.5-flash")

    Returns:
        Model family identifier:
        - "openai_reasoning": OpenAI o1/o3/GPT-5 series (support reasoning.effort)
        - "grok": xAI Grok models (support reasoning.effort)
        - "anthropic": Anthropic Claude (support reasoning.max_tokens)
        - "google": Google Gemini (support reasoning.max_tokens)
        - "alibaba": Alibaba Qwen (support reasoning.max_tokens)
        - "other": Unknown/standard models

    Examples:
        >>> detect_model_family("openai/gpt-5")
        'openai_reasoning'
        >>> detect_model_family("google/gemini-2.5-flash")
        'google'
        >>> detect_model_family("anthropic/claude-3.5-sonnet")
        'anthropic'
    """
    model_lower = model_name.lower()

    # OpenAI reasoning models (support reasoning.effort)
    # Pattern: gpt-5*, o1-*, o3-*, o1 (space), o3 (space)
    if any(pattern in model_lower for pattern in ['gpt-5', 'o1-', 'o3-', 'o1 ', 'o3 ']):
        return "openai_reasoning"

    # Grok models (support reasoning.effort)
    if 'grok' in model_lower:
        return "grok"

    # Anthropic models (support reasoning.max_tokens)
    if 'claude' in model_lower or 'anthropic' in model_lower:
        return "anthropic"

    # Google models (support reasoning.max_tokens)
    if 'gemini' in model_lower or 'google' in model_lower:
        return "google"

    # Alibaba Qwen models (support reasoning.max_tokens)
    if 'qwen' in model_lower or 'alibaba' in model_lower:
        return "alibaba"

    return "other"


def apply_tools_template(messages: List[Dict[str, Any]], tools: List[str | Dict]):
    if not isinstance(tools, list):
        raise ValueError("tools must be a list of strings or dictionaries")

    tools_str = ""
    for tool in tools:
        if isinstance(tool, str):
            tools_str += f"{tool}\n"
        elif isinstance(tool, dict):
            # first serialize the dictionary into a string
            tools_str += f"{json.dumps(tool)}\n"
        else:
            raise ValueError(
                f"Elements inside tools must be strings or dictionaries. Found {type(tool)}"
            )

    # Then concatenate the beginning and ending tool message template to the tool_str
    tool_template_start = "\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    tool_template_ending = """</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "parameters": <args-json-object>}\n</tool_call>"""
    tool_message = f"{tool_template_start}{tools_str}{tool_template_ending}"
    # We need to add the tool message to the system message
    # first find the system message
    system_prompt_idx = next(
        (i for i, item in enumerate(messages) if item.get("role") == "system"),
        None,
    )
    if system_prompt_idx is not None:
        system_prompt = messages[system_prompt_idx]
        # then append the tool message to the system prompt
        system_prompt["content"] += tool_message
        messages[system_prompt_idx] = system_prompt
    else:
        # otherwise append the system prompt to the beginning of the messages
        messages.insert(
            0,
            {
                "role": "system",
                "content": "You are a intelligent assistant. You should help use to achieve the goal."
                + tool_message,
            },
        )
    if len(messages) >= 3:
        messages.append({"role": "user", "content": tool_message})

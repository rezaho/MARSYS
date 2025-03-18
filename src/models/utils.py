import json
from typing import Dict, List

from src.agents.memory import MessageMemory


def apply_tools_template(messages: MessageMemory, tools: List[str | Dict]):
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

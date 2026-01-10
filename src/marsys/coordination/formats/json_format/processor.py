"""
JSON response processor for parsing structured JSON responses.

This module handles JSON responses with next_action/action_input structure.
"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from ..processors import ResponseProcessor

logger = logging.getLogger(__name__)


class StructuredJSONProcessor(ResponseProcessor):
    """
    Handles JSON responses with next_action/action_input structure.

    This processor parses JSON responses that follow the MARSYS coordination
    format with next_action and action_input fields.
    """

    def _extract_json_from_code_block(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks or plain JSON."""
        # Greedy matching to handle nested code blocks
        code_block_pattern = r"```(?:json)?\s*\n?(.*)```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Check if entire text is valid JSON
        text_stripped = text.strip()
        if text_stripped.startswith("{") and text_stripped.endswith("}"):
            try:
                json.loads(text_stripped)
                return text_stripped
            except json.JSONDecodeError:
                pass

        return None

    def _extract_first_json(self, text: str) -> Optional[str]:
        """Extract the first valid JSON object from text."""
        brace_count = 0
        start_idx = None

        for i, char in enumerate(text):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = text[start_idx : i + 1]
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        continue

        return None

    def can_process(self, response: Any) -> bool:
        """Check for expected JSON structure."""
        if isinstance(response, dict):
            return "next_action" in response

        if isinstance(response, str):
            json_str = self._extract_json_from_code_block(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "next_action" in data:
                        return True
                except json.JSONDecodeError:
                    pass

            json_str = self._extract_first_json(response)
            if json_str:
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and "next_action" in data:
                        return True
                except json.JSONDecodeError:
                    pass

            try:
                data = json.loads(response)
                return isinstance(data, dict) and "next_action" in data
            except json.JSONDecodeError:
                return False

        return False

    def process(self, response: Any) -> Optional[Dict[str, Any]]:
        """Parse and validate JSON structure."""
        try:
            if isinstance(response, str):
                json_str = self._extract_json_from_code_block(response)
                if json_str:
                    data = json.loads(json_str)
                else:
                    json_str = self._extract_first_json(response)
                    if json_str:
                        data = json.loads(json_str)
                    else:
                        data = json.loads(response)
            else:
                data = response

            result = {"next_action": data.get("next_action"), "raw_response": data}

            # Handle different action types
            if data.get("next_action") == "invoke_agent":
                result = self._process_invoke_agent(data, result)
            elif data.get("next_action") == "final_response":
                result = self._process_final_response(data, result)
            elif data.get("next_action") == "call_tool":
                # Tool calls should use native format
                return {
                    "next_action": "validation_error",
                    "error": "Tool calls must use native tool_calls format",
                    "raw_response": data,
                }

            # Include content and message fields
            result["content"] = data.get("content", "")
            result["message"] = data.get("message", "")

            # Preserve additional fields
            for key, value in data.items():
                if key not in result:
                    result[key] = value

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing response: {e}")
            return None

    def _process_invoke_agent(
        self, data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process invoke_agent action."""
        from ...validation.types import AgentInvocation

        action_input = data.get("action_input", {})

        if isinstance(action_input, list):
            # Array format (new standard)
            invocations = []
            for idx, item in enumerate(action_input):
                if isinstance(item, dict) and "agent_name" in item:
                    invocation = AgentInvocation(
                        agent_name=item["agent_name"],
                        request=item.get("request", {}),
                        instance_id=f"{item['agent_name']}_{idx}_{uuid.uuid4().hex[:8]}",
                    )
                    invocations.append(invocation)
                else:
                    raise ValueError(f"Invalid invocation format at index {idx}")

            result["invocations"] = invocations
            result["agent_requests"] = {
                inv.instance_id: inv.request for inv in invocations
            }

            if len(invocations) == 1:
                result["target_agent"] = invocations[0].agent_name
                result["action_input"] = invocations[0].request

        elif isinstance(action_input, dict) and "agent_name" in action_input:
            # Single dict format - convert to list
            invocation = AgentInvocation(
                agent_name=action_input["agent_name"],
                request=action_input.get("request", {}),
                instance_id=f"{action_input['agent_name']}_0_{uuid.uuid4().hex[:8]}",
            )
            result["invocations"] = [invocation]
            result["target_agent"] = invocation.agent_name
            result["action_input"] = invocation.request
            result["agent_requests"] = {invocation.instance_id: invocation.request}

        else:
            return {
                "next_action": "validation_error",
                "error": "Invalid action_input format for invoke_agent",
                "raw_response": data,
            }

        return result

    def _process_final_response(
        self, data: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process final_response action."""
        action_input = data.get("action_input", {})

        if isinstance(action_input, dict):
            if "response" in action_input:
                result["final_response"] = action_input["response"]
            elif "report" in action_input:
                result["final_response"] = action_input["report"]
            else:
                result["final_response"] = data.get(
                    "final_response", data.get("content", "")
                )
        else:
            result["final_response"] = data.get(
                "final_response", data.get("content", "")
            )

        return result

    def priority(self) -> int:
        return 80  # Lower than error and tool call processors

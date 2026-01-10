"""
JSON response format implementation for MARSYS.

This module provides the JSON-based response format that is the default
format used by MARSYS coordination system.
"""

from typing import List, TYPE_CHECKING

from ..base import BaseResponseFormat
from ..context import SystemPromptContext

if TYPE_CHECKING:
    from ..processors import ResponseProcessor


class JSONResponseFormat(BaseResponseFormat):
    """
    JSON response format implementation.

    This is the default format used by MARSYS. It produces responses in the form:
    ```json
    {
      "thought": "reasoning...",
      "next_action": "action_name",
      "action_input": { ... }
    }
    ```
    """

    def get_format_name(self) -> str:
        return "json"

    def build_format_instructions(
        self, available_actions: List[str], action_descriptions: List[str]
    ) -> str:
        """Build JSON format instructions."""
        actions_str = ", ".join(f"'{a}'" for a in available_actions)

        return f"""
--- STRICT JSON OUTPUT FORMAT ---
Your *entire* response MUST be a single, valid JSON object. This JSON object
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
2. `next_action` (String, required) – one of: {actions_str}.
3. `action_input` (Array/Object, required) – parameters specific to `next_action`.
{chr(10).join(action_descriptions)}
--- END STRICT JSON OUTPUT FORMAT ---"""

    def build_action_descriptions(
        self, available_actions: List[str], context: SystemPromptContext
    ) -> List[str]:
        """Build JSON-specific action descriptions."""
        descriptions = []

        if "invoke_agent" in available_actions:
            agents_list = ", ".join(context.coordination.next_agents)
            descriptions.append(
                '- If `next_action` is `"invoke_agent"`:\n'
                f"     An ARRAY of agent invocation objects for agents: {agents_list}\n"
                '     `[{"agent_name": "example_agent_name", "request": {...}}, ...]`\n'
                "     • Single agent: array with one object\n"
                "     • Multiple agents: array with multiple objects (parallel execution)\n"
                "     • IMPORTANT: Only invoke agents together if you do NOT need one's response before invoking another"
            )

        if "final_response" in available_actions:
            descriptions.append(
                '- If `next_action` is `"final_response"`:\n'
                '     `{"response": "Your final textual answer..."}`\n'
                "     **USE THIS when your assigned task is fully complete!**"
            )

        return descriptions

    def get_parallel_invocation_examples(self, context: SystemPromptContext) -> str:
        """Generate JSON-specific examples for parallel agent invocation patterns."""
        lines = []

        lines.append("\n**EXAMPLES OF CORRECT INVOCATION PATTERNS:**")

        lines.append("\n✅ CORRECT - When you need responses before proceeding:")
        lines.append(
            """```json
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

        lines.append("\n❌ INCORRECT - Invoking dependent agents together:")
        lines.append(
            """```json
{
  "thought": "I'll invoke data collectors and processor together",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "DataAgent1", "request": {"query": "..."}},
    {"agent_name": "DataAgent2", "request": {"query": "..."}},
    {"agent_name": "ProcessingAgent", "request": {"data": "???"}}
  ]
}
// ERROR: What data? You don't have it yet!
```"""
        )

        lines.append("\n✅ CORRECT - When agents don't depend on each other:")
        lines.append(
            """```json
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

        return "\n".join(lines)

    def get_examples(
        self, available_actions: List[str], context: SystemPromptContext
    ) -> str:
        """Generate JSON examples for available actions."""
        examples = []

        if "invoke_agent" in available_actions:
            examples.append(
                """
Example for single agent invocation:
```json
{
  "thought": "I need to delegate this task to a specialist.",
  "next_action": "invoke_agent",
  "action_input": [
    {
      "agent_name": "<example_agent>",
      "request": {"task": "specific task details"}
    }
  ]
}
```

Example for parallel invocations (when responses are independent):
```json
{
  "thought": "I need to process multiple items that don't depend on each other.",
  "next_action": "invoke_agent",
  "action_input": [
    {"agent_name": "<agent_1>", "request": {"query": "..."}},
    {"agent_name": "<agent_2>", "request": {"query": "..."}}
  ]
}
```

REMEMBER: Only invoke agents together if you don't need one's response to invoke another!"""
            )

        if "final_response" in available_actions:
            examples.append(
                """
Example for `final_response`:
```json
{
  "thought": "I have completed the task.",
  "next_action": "final_response",
  "action_input": {
    "response": "Here is my final answer based on the analysis..."
  }
}
```"""
            )

        return "\n".join(examples)

    def create_processor(self) -> "ResponseProcessor":
        """Create JSON processor for validation."""
        from .processor import StructuredJSONProcessor

        return StructuredJSONProcessor()

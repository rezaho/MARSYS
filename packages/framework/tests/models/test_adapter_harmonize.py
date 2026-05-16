"""Adapter wire-contract tests — request construction + response
harmonization for the standard Anthropic adapter (no network).

Two contracts, both regressed on the standard `anthropic` adapter and both
masked by the OAuth adapter (which constructs messages correctly):

1. Response: a tool call's ``arguments`` at the ``ToolCallMsg`` boundary
   must be a JSON **string** (memory.py:191-192 rejects a non-string).
   ``harmonize_response`` must emit it that way.
2. Request: ``format_request_payload`` must emit Anthropic messages with
   ONLY the wire-legal keys (``role``, ``content``). The OpenAI-shaped
   message carries a message-level ``name`` (agent identity); Anthropic's
   Messages API rejects it with
   ``messages.N.name: Extra inputs are not permitted`` -> HTTP 400.
"""

import json

from marsys.models.adapters.anthropic import AnthropicAdapter


def _adapter() -> AnthropicAdapter:
    return AnthropicAdapter(
        model_name="claude-haiku-4-5-20251001",
        api_key="not-a-real-key",
        base_url="https://api.anthropic.com/v1",
    )


def test_anthropic_harmonize_tool_call_arguments_is_json_string():
    """A tool_use block's object input is harmonized to a JSON string,
    faithfully (json.loads round-trips to the original object)."""
    raw_input = {"invocations": [{"agent_name": "Researcher", "request": "speed of light"}]}
    raw_response = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Dispatching."},
            {"type": "tool_use", "id": "toolu_1", "name": "invoke_agent", "input": raw_input},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    harmonized = _adapter().harmonize_response(raw_response, request_start_time=0.0)

    assert len(harmonized.tool_calls) == 1
    arguments = harmonized.tool_calls[0].function["arguments"]
    assert isinstance(arguments, str), (
        f"tool-call arguments must be a JSON string at the harmonized "
        f"boundary, got {type(arguments).__name__}"
    )
    assert json.loads(arguments) == raw_input


def test_anthropic_harmonize_empty_tool_input_is_json_string():
    """A tool_use block with no input still yields a JSON string ("{}"),
    never a dict or None."""
    raw_response = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "noop", "input": {}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    harmonized = _adapter().harmonize_response(raw_response, request_start_time=0.0)

    arguments = harmonized.tool_calls[0].function["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {}


def test_anthropic_format_request_payload_drops_message_level_name():
    """A plain assistant/user message carrying an OpenAI-family `name`
    (agent identity from memory.to_llm_dict) must NOT leak into the
    Anthropic payload — Anthropic rejects any message key beyond
    role/content with `messages.N.name: Extra inputs are not permitted`."""
    messages = [
        {"role": "system", "content": "You are a coordinator."},
        {"role": "user", "content": "Verify the speed of light.", "name": "Coordinator"},
        {"role": "assistant", "content": "Here are the verified facts ...", "name": "FactChecker"},
    ]

    payload = _adapter().format_request_payload(messages)

    for m in payload["messages"]:
        assert set(m.keys()) <= {"role", "content"}, (
            f"Anthropic message must carry only role/content, got {sorted(m.keys())}"
        )
    # content is preserved on the rebuilt messages
    assert payload["messages"][-1]["content"] == "Here are the verified facts ..."
    assert payload["messages"][-1]["role"] == "assistant"


def test_anthropic_format_request_payload_tool_branches_unaffected():
    """The tool / assistant-with-tool_calls branches still emit wire-legal
    shapes (regression guard: the Root-C fix only rewrote the plain-message
    else branch)."""
    messages = [
        {"role": "assistant", "content": "calling", "name": "W",
         "tool_calls": [{"id": "t1", "type": "function",
                         "function": {"name": "plan_update", "arguments": '{"x": 1}'}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"},
    ]

    payload = _adapter().format_request_payload(messages)

    asst = payload["messages"][0]
    assert set(asst.keys()) <= {"role", "content"}
    assert asst["content"][-1]["type"] == "tool_use"
    assert asst["content"][-1]["input"] == {"x": 1}
    tool_turn = payload["messages"][1]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0]["type"] == "tool_result"
    assert tool_turn["content"][0]["tool_use_id"] == "t1"

"""Tests for AgentInput — the canonical wrapper for agent step input.

Covers the three factory paths (from_text, from_message, from_messages),
the coerce path (str/dict/Message/list/AgentInput round-tripping), the
aggregate path (single vs multi arrival), and the memory-write semantics.
"""
from __future__ import annotations

from marsys.agents.agent_input import AgentInput
from marsys.agents.memory import ConversationMemory, Message


class TestFromText:

    def test_default_role(self):
        ai = AgentInput.from_text("hello")
        assert len(ai.messages) == 1
        assert ai.messages[0].role == "user"
        assert ai.messages[0].content == "hello"

    def test_custom_role_and_name(self):
        ai = AgentInput.from_text("ack", role="system", name="src")
        assert ai.messages[0].role == "system"
        assert ai.messages[0].name == "src"

    def test_primary_content(self):
        assert AgentInput.from_text("hi").primary_content() == "hi"


class TestFromMessage:

    def test_wraps_existing_message(self):
        m = Message(role="tool", content="result", tool_call_id="tc_1")
        ai = AgentInput.from_message(m)
        assert ai.messages == [m]


class TestCoerce:

    def test_passthrough_for_agent_input(self):
        original = AgentInput.from_text("x")
        assert AgentInput.coerce(original) is original

    def test_string_becomes_user_message(self):
        ai = AgentInput.coerce("research light")
        assert ai.messages[0].role == "user"
        assert ai.messages[0].content == "research light"

    def test_message_passthrough(self):
        m = Message(role="assistant", content="ok")
        ai = AgentInput.coerce(m)
        assert ai.messages == [m]

    def test_list_of_messages_passthrough(self):
        m1 = Message(role="user", content="a")
        m2 = Message(role="assistant", content="b")
        ai = AgentInput.coerce([m1, m2])
        assert ai.messages == [m1, m2]

    def test_dict_with_tool_result(self):
        ai = AgentInput.coerce({"tool_result": "42", "tool_name": "calc"})
        assert ai.messages[0].role == "tool"
        assert ai.messages[0].content == "42"
        assert ai.messages[0].name == "calc"

    def test_dict_with_error_feedback(self):
        ai = AgentInput.coerce({"error_feedback": "bad action"})
        assert ai.messages[0].role == "system"
        assert ai.messages[0].content == "bad action"

    def test_dict_with_prompt(self):
        ai = AgentInput.coerce({"prompt": "do it"})
        assert ai.messages[0].role == "user"
        assert ai.messages[0].content == "do it"

    def test_unknown_dict_kept_as_content(self):
        d = {"x": 1, "y": 2}
        ai = AgentInput.coerce(d)
        assert ai.messages[0].content == d

    def test_list_of_strings_aggregates_with_markers(self):
        ai = AgentInput.coerce(["alpha", "beta"])
        assert len(ai.messages) == 1
        content = ai.messages[0].content
        assert isinstance(content, list)
        assert all(isinstance(b, dict) and b.get("type") == "text" for b in content)
        joined = " ".join(b["text"] for b in content)
        assert "alpha" in joined and "beta" in joined
        assert "[from " in joined

    def test_arbitrary_value_stringified(self):
        ai = AgentInput.coerce(42)
        assert ai.messages[0].content == "42"


class TestAggregate:

    def test_empty_arrived(self):
        ai = AgentInput.aggregate({})
        assert ai.is_empty()

    def test_single_arrival_short_circuits(self):
        ai = AgentInput.aggregate({"Researcher": "findings"})
        assert len(ai.messages) == 1
        assert ai.messages[0].content == "findings"

    def test_multi_arrival_combines_with_source_markers(self):
        ai = AgentInput.aggregate({
            "Researcher": "speed of light is 299,792,458 m/s",
            "FactChecker": "Ole Roemer measured it in 1676",
        })
        assert len(ai.messages) == 1
        content = ai.messages[0].content
        # Multi-block typed-text-blocks list (Anthropic-native, OpenAI-compat)
        assert isinstance(content, list)
        assert all(b.get("type") == "text" for b in content)
        joined = " ".join(b["text"] for b in content)
        assert "[from Researcher]" in joined
        assert "[from FactChecker]" in joined
        assert "299,792,458" in joined
        assert "Roemer" in joined

    def test_multi_arrival_with_agent_input_values(self):
        sub_a = AgentInput.from_text("alpha findings")
        sub_b = AgentInput.from_text("beta findings")
        ai = AgentInput.aggregate({"A": sub_a, "B": sub_b})
        assert len(ai.messages) == 1
        content = ai.messages[0].content
        joined = " ".join(b["text"] for b in content)
        assert "alpha findings" in joined and "beta findings" in joined

    def test_multi_arrival_with_message_values(self):
        m_a = Message(role="user", content="alpha")
        m_b = Message(role="user", content="beta")
        ai = AgentInput.aggregate({"A": m_a, "B": m_b})
        content = ai.messages[0].content
        joined = " ".join(b["text"] for b in content)
        assert "alpha" in joined and "beta" in joined

    def test_multi_arrival_preserves_typed_blocks(self):
        """A vision agent's multimodal output (text + image) survives
        aggregation as inline typed blocks, not as stringified text."""
        vision_message = Message(role="user", content=[
            {"type": "text", "text": "I see a chart"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
        ])
        ai = AgentInput.aggregate({
            "VisionAgent": vision_message,
            "Reporter": "summary text",
        })
        content = ai.messages[0].content
        assert isinstance(content, list)
        types = [b.get("type") for b in content]
        assert "image_url" in types
        # source headers and payload blocks both present
        text_blocks = [b for b in content if b.get("type") == "text"]
        joined = " ".join(b["text"] for b in text_blocks)
        assert "[from VisionAgent]" in joined and "[from Reporter]" in joined


class TestStr:

    def test_str_of_text_message(self):
        ai = AgentInput.from_text("hello")
        assert str(ai) == "hello"

    def test_str_of_aggregated_typed_blocks(self):
        ai = AgentInput.aggregate({"A": "x", "B": "y"})
        s = str(ai)
        assert "x" in s and "y" in s
        assert "[from A]" in s and "[from B]" in s

    def test_str_of_empty(self):
        ai = AgentInput(messages=[])
        assert str(ai) == ""

    def test_str_of_multimodal_renders_non_text_block_marker(self):
        ai = AgentInput.from_message(Message(role="user", content=[
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]))
        s = str(ai)
        assert "describe this" in s
        assert "[image_url]" in s


class TestAddToMemory:

    def test_appends_each_message(self):
        memory = ConversationMemory()
        ai = AgentInput.from_messages([
            Message(role="user", content="q"),
            Message(role="user", content="follow-up"),
        ])
        ids = ai.add_to_memory(memory)
        assert len(ids) == 2
        msgs = memory.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "q"
        assert msgs[1]["content"] == "follow-up"


class TestAnthropicSafety:
    """Verify aggregated output is in a shape Anthropic's API accepts.

    Bug we are fixing: a content list of bare strings (e.g.,
    ``content=[\"a\", \"b\"]``) was rejected by Anthropic with
    ``messages.X.content.0: Input should be an object``. Aggregation must
    produce a list of typed-block objects (or a single string)."""

    def test_aggregate_produces_typed_blocks_list(self):
        ai = AgentInput.aggregate({"A": "x", "B": "y"})
        content = ai.messages[0].content
        assert isinstance(content, list)
        # Every element is a dict with "type" key — never a bare string
        for block in content:
            assert isinstance(block, dict)
            assert "type" in block

    def test_coerce_list_of_strings_produces_typed_blocks_list(self):
        ai = AgentInput.coerce(["x", "y"])
        content = ai.messages[0].content
        assert isinstance(content, list)
        for block in content:
            assert isinstance(block, dict)
            assert "type" in block

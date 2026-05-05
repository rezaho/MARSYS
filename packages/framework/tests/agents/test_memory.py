"""
Tests for the marsys.agents.memory module.

This module tests:
- Message dataclass: creation, conversion, validation
- ToolCallMsg and AgentCallMsg dataclasses
- ConversationMemory: CRUD operations
- MemoryManager: factory and delegation
"""

import pytest
import uuid
import json
from typing import Dict, Any

from marsys.agents.memory import (
    Message,
    ToolCallMsg,
    AgentCallMsg,
    MessageContent,
    ConversationMemory,
    MemoryManager,
    BaseMemory,
)
from marsys.agents.exceptions import MessageError


# =============================================================================
# ToolCallMsg Tests
# =============================================================================

class TestToolCallMsg:
    """Tests for ToolCallMsg dataclass."""

    def test_creation_with_valid_data(self):
        """Test creating a ToolCallMsg with valid data."""
        tool_call = ToolCallMsg(
            id="call_123",
            call_id="call_123",
            type="function",
            name="search",
            arguments='{"query": "test"}'
        )

        assert tool_call.id == "call_123"
        assert tool_call.call_id == "call_123"
        assert tool_call.type == "function"
        assert tool_call.name == "search"
        assert tool_call.arguments == '{"query": "test"}'

    def test_to_dict_format(self):
        """Test conversion to OpenAI API dict format."""
        tool_call = ToolCallMsg(
            id="call_456",
            call_id="call_456",
            type="function",
            name="fetch_url",
            arguments='{"url": "https://example.com"}'
        )

        result = tool_call.to_dict()

        assert result["id"] == "call_456"
        assert result["type"] == "function"
        assert result["function"]["name"] == "fetch_url"
        assert result["function"]["arguments"] == '{"url": "https://example.com"}'

    def test_from_dict_creation(self):
        """Test creating ToolCallMsg from dict."""
        data = {
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": '{"x": 1, "y": 2}'
            }
        }

        tool_call = ToolCallMsg.from_dict(data)

        assert tool_call.id == "call_789"
        assert tool_call.name == "calculate"
        assert tool_call.arguments == '{"x": 1, "y": 2}'

    def test_validation_empty_id(self):
        """Test validation rejects empty id."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            ToolCallMsg(
                id="",
                call_id="call_123",
                type="function",
                name="test",
                arguments="{}"
            )

    def test_validation_empty_name(self):
        """Test validation rejects empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ToolCallMsg(
                id="call_123",
                call_id="call_123",
                type="function",
                name="",
                arguments="{}"
            )

    def test_validation_arguments_not_string(self):
        """Test validation rejects non-string arguments."""
        with pytest.raises(ValueError, match="arguments must be a string"):
            ToolCallMsg(
                id="call_123",
                call_id="call_123",
                type="function",
                name="test",
                arguments={"invalid": "dict"}  # Should be string
            )


# =============================================================================
# AgentCallMsg Tests
# =============================================================================

class TestAgentCallMsg:
    """Tests for AgentCallMsg dataclass."""

    def test_creation_with_valid_data(self):
        """Test creating an AgentCallMsg with valid data."""
        agent_call = AgentCallMsg(
            agent_name="ResearchAgent",
            request={"task": "analyze data"}
        )

        assert agent_call.agent_name == "ResearchAgent"
        assert agent_call.request == {"task": "analyze data"}

    def test_to_dict_format(self):
        """Test conversion to dict format."""
        agent_call = AgentCallMsg(
            agent_name="SummaryAgent",
            request="Summarize the document"
        )

        result = agent_call.to_dict()

        assert result["agent_name"] == "SummaryAgent"
        assert result["request"] == "Summarize the document"

    def test_from_dict_creation(self):
        """Test creating AgentCallMsg from dict."""
        data = {
            "agent_name": "AnalysisAgent",
            "request": {"data": [1, 2, 3]}
        }

        agent_call = AgentCallMsg.from_dict(data)

        assert agent_call.agent_name == "AnalysisAgent"
        assert agent_call.request == {"data": [1, 2, 3]}

    def test_validation_empty_agent_name(self):
        """Test validation rejects empty agent_name."""
        with pytest.raises(ValueError, match="agent_name cannot be empty"):
            AgentCallMsg(agent_name="", request="test")

    def test_validation_none_request(self):
        """Test validation rejects None request."""
        with pytest.raises(ValueError, match="request cannot be None"):
            AgentCallMsg(agent_name="TestAgent", request=None)


# =============================================================================
# MessageContent Tests
# =============================================================================

class TestMessageContent:
    """Tests for MessageContent dataclass."""

    def test_creation_with_valid_action(self):
        """Test creating MessageContent with valid next_action."""
        content = MessageContent(
            thought="I need to search",
            next_action="call_tool",
            action_input={"tool_calls": []}
        )

        assert content.thought == "I need to search"
        assert content.next_action == "call_tool"

    def test_to_dict_format(self):
        """Test conversion to dict."""
        content = MessageContent(
            thought="Analysis complete",
            next_action="final_response",
            action_input={"response": "Done"}
        )

        result = content.to_dict()

        assert result["thought"] == "Analysis complete"
        assert result["next_action"] == "final_response"
        assert result["action_input"] == {"response": "Done"}

    def test_from_dict_creation(self):
        """Test creating MessageContent from dict."""
        data = {
            "thought": "Delegating task",
            "next_action": "invoke_agent",
            "action_input": {"agent_name": "Worker"}
        }

        content = MessageContent.from_dict(data)

        assert content.thought == "Delegating task"
        assert content.next_action == "invoke_agent"

    def test_validation_invalid_action(self):
        """Test validation rejects invalid next_action."""
        with pytest.raises(ValueError, match="next_action must be one of"):
            MessageContent(
                thought="test",
                next_action="invalid_action",
                action_input={}
            )

    def test_valid_actions_accepted(self):
        """Test all valid next_action values are accepted."""
        valid_actions = ["call_tool", "invoke_agent", "final_response"]

        for action in valid_actions:
            content = MessageContent(next_action=action)
            assert content.next_action == action


# =============================================================================
# Message Tests
# =============================================================================

class TestMessage:
    """Tests for Message dataclass."""

    def test_basic_creation(self):
        """Test creating a basic Message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.message_id is not None
        assert len(msg.message_id) == 36  # UUID format

    def test_creation_with_all_fields(self):
        """Test creating Message with all optional fields."""
        msg_id = str(uuid.uuid4())
        msg = Message(
            role="assistant",
            content="Response",
            message_id=msg_id,
            name="TestModel",
            images=["image1.png", "image2.png"]
        )

        assert msg.message_id == msg_id
        assert msg.name == "TestModel"
        assert msg.images == ["image1.png", "image2.png"]

    def test_tool_calls_auto_conversion(self):
        """Test that dict tool_calls are auto-converted to ToolCallMsg."""
        tool_calls_dict = [{
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"}
        }]

        msg = Message(role="assistant", content=None, tool_calls=tool_calls_dict)

        assert len(msg.tool_calls) == 1
        assert isinstance(msg.tool_calls[0], ToolCallMsg)
        assert msg.tool_calls[0].name == "search"

    def test_agent_calls_auto_conversion(self):
        """Test that dict agent_calls are auto-converted to AgentCallMsg."""
        agent_calls_dict = [{
            "agent_name": "Worker",
            "request": {"task": "process"}
        }]

        msg = Message(role="assistant", content=None, agent_calls=agent_calls_dict)

        assert len(msg.agent_calls) == 1
        assert isinstance(msg.agent_calls[0], AgentCallMsg)
        assert msg.agent_calls[0].agent_name == "Worker"

    def test_to_llm_dict_basic(self):
        """Test basic to_llm_dict conversion."""
        msg = Message(role="user", content="Test message")

        result = msg.to_llm_dict()

        assert result["role"] == "user"
        assert result["content"] == "Test message"

    def test_to_llm_dict_tool_role(self):
        """Test to_llm_dict for tool role (content must be string)."""
        msg = Message(
            role="tool",
            content={"result": "success"},
            tool_call_id="call_123",
            name="search"
        )

        result = msg.to_llm_dict()

        assert result["role"] == "tool"
        assert isinstance(result["content"], str)
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "search"

    def test_to_llm_dict_with_tool_calls(self):
        """Test to_llm_dict includes tool_calls."""
        tool_call = ToolCallMsg(
            id="call_123",
            call_id="call_123",
            type="function",
            name="test",
            arguments="{}"
        )
        msg = Message(role="assistant", content="Using tool", tool_calls=[tool_call])

        result = msg.to_llm_dict()

        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1

    def test_to_llm_dict_dict_content_serialization(self):
        """Test that dict content is JSON serialized."""
        msg = Message(role="assistant", content={"key": "value"})

        result = msg.to_llm_dict()

        assert result["content"] == '{"key":"value"}'

    def test_from_response_dict(self):
        """Test creating Message from response dict."""
        response = {
            "role": "assistant",
            "content": "Hello there",
            "name": "GPT-4"
        }

        msg = Message.from_response_dict(response)

        assert msg.role == "assistant"
        assert msg.content == "Hello there"
        assert msg.name == "GPT-4"

    def test_images_validation_valid(self):
        """Test valid images list is accepted."""
        msg = Message(
            role="user",
            content="Check this image",
            images=["path/to/image.png"]
        )

        assert msg.images == ["path/to/image.png"]

    def test_images_validation_invalid_type(self):
        """Test images validation rejects non-list."""
        # MessageError is raised when images is not a list
        with pytest.raises((MessageError, TypeError)):
            Message(role="user", content="test", images="not_a_list")

    def test_images_validation_invalid_item(self):
        """Test images validation rejects non-string items."""
        # MessageError is raised when images item is not a string
        with pytest.raises((MessageError, TypeError)):
            Message(role="user", content="test", images=[123])


# =============================================================================
# ConversationMemory Tests
# =============================================================================

class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_initialization_empty(self):
        """Test empty initialization."""
        memory = ConversationMemory()

        assert memory.memory_type == "conversation_history"
        assert len(memory.memory) == 0

    def test_initialization_with_description(self):
        """Test initialization with system description."""
        memory = ConversationMemory(description="You are a helpful assistant.")

        assert len(memory.memory) == 1
        assert memory.memory[0].role == "system"
        assert memory.memory[0].content == "You are a helpful assistant."

    def test_add_message_object(self):
        """Test adding a Message object."""
        memory = ConversationMemory()
        msg = Message(role="user", content="Hello")

        msg_id = memory.add(message=msg)

        assert msg_id == msg.message_id
        assert len(memory.memory) == 1
        assert memory.memory[0].content == "Hello"

    def test_add_with_parameters(self):
        """Test adding with role/content parameters."""
        memory = ConversationMemory()

        msg_id = memory.add(role="user", content="Test message")

        assert msg_id is not None
        assert len(memory.memory) == 1
        assert memory.memory[0].role == "user"
        assert memory.memory[0].content == "Test message"

    def test_add_requires_message_or_role(self):
        """Test that add requires either message or role."""
        memory = ConversationMemory()

        # Either MessageError or TypeError is raised
        with pytest.raises((MessageError, TypeError)):
            memory.add()

    def test_retrieve_all(self):
        """Test retrieve_all returns all messages as dicts."""
        memory = ConversationMemory()
        memory.add(role="user", content="Message 1")
        memory.add(role="assistant", content="Message 2")

        result = memory.retrieve_all()

        assert len(result) == 2
        assert all(isinstance(m, dict) for m in result)
        assert result[0]["content"] == "Message 1"
        assert result[1]["content"] == "Message 2"

    def test_retrieve_recent(self):
        """Test retrieve_recent returns N most recent messages."""
        memory = ConversationMemory()
        for i in range(5):
            memory.add(role="user", content=f"Message {i}")

        result = memory.retrieve_recent(n=2)

        assert len(result) == 2
        assert result[0]["content"] == "Message 3"
        assert result[1]["content"] == "Message 4"

    def test_retrieve_by_id(self):
        """Test retrieve_by_id finds correct message."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="Target message")
        memory.add(role="assistant", content="Other message")

        result = memory.retrieve_by_id(msg_id)

        assert result is not None
        assert result["content"] == "Target message"

    def test_retrieve_by_id_not_found(self):
        """Test retrieve_by_id returns None for non-existent ID."""
        memory = ConversationMemory()

        result = memory.retrieve_by_id("non_existent_id")

        assert result is None

    def test_retrieve_by_role(self):
        """Test retrieve_by_role filters correctly."""
        memory = ConversationMemory()
        memory.add(role="user", content="User 1")
        memory.add(role="assistant", content="Assistant 1")
        memory.add(role="user", content="User 2")

        result = memory.retrieve_by_role("user")

        assert len(result) == 2
        assert all(m["role"] == "user" for m in result)

    def test_update_message(self):
        """Test updating an existing message."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="Original")

        memory.update(msg_id, content="Updated")

        result = memory.retrieve_by_id(msg_id)
        assert result["content"] == "Updated"

    def test_update_message_not_found(self):
        """Test update raises error for non-existent message."""
        memory = ConversationMemory()

        # Either MessageError or KeyError is raised when message not found
        with pytest.raises((MessageError, KeyError, TypeError)):
            memory.update("non_existent_id", content="test")

    def test_remove_by_id(self):
        """Test removing message by ID."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="To be removed")

        result = memory.remove_by_id(msg_id)

        assert result is True
        assert len(memory.memory) == 0

    def test_remove_by_id_not_found(self):
        """Test remove_by_id returns False for non-existent ID."""
        memory = ConversationMemory()

        result = memory.remove_by_id("non_existent_id")

        assert result is False

    def test_delete_memory_by_index(self):
        """Test deleting message by index."""
        memory = ConversationMemory()
        memory.add(role="user", content="Message 0")
        memory.add(role="user", content="Message 1")

        memory.delete_memory(0)

        assert len(memory.memory) == 1
        assert memory.memory[0].content == "Message 1"

    def test_delete_memory_invalid_index(self):
        """Test delete_memory raises error for invalid index."""
        memory = ConversationMemory()

        with pytest.raises(IndexError):
            memory.delete_memory(0)

    def test_replace_memory_with_message(self):
        """Test replacing memory at index with Message object."""
        memory = ConversationMemory()
        memory.add(role="user", content="Original")

        new_msg = Message(role="user", content="Replacement")
        memory.replace_memory(0, message=new_msg)

        assert memory.memory[0].content == "Replacement"

    def test_replace_memory_with_params(self):
        """Test replacing memory at index with parameters."""
        memory = ConversationMemory()
        memory.add(role="user", content="Original")

        memory.replace_memory(0, role="assistant", content="New content")

        assert memory.memory[0].role == "assistant"
        assert memory.memory[0].content == "New content"

    def test_reset_memory_keeps_system(self):
        """Test reset_memory keeps system message."""
        memory = ConversationMemory(description="System prompt")
        memory.add(role="user", content="User message")
        memory.add(role="assistant", content="Assistant message")

        memory.reset_memory()

        assert len(memory.memory) == 1
        assert memory.memory[0].role == "system"

    def test_reset_memory_clears_all_when_no_system(self):
        """Test reset_memory clears all when no system message."""
        memory = ConversationMemory()
        memory.add(role="user", content="User message")

        memory.reset_memory()

        assert len(memory.memory) == 0


# =============================================================================
# MemoryManager Tests
# =============================================================================

class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_initialization_conversation_history(self):
        """Test initialization with conversation_history type."""
        manager = MemoryManager(memory_type="conversation_history")

        assert manager.memory_type == "conversation_history"
        assert isinstance(manager.memory_module, ConversationMemory)

    def test_initialization_with_description(self):
        """Test initialization with description."""
        manager = MemoryManager(
            memory_type="conversation_history",
            description="Test description"
        )

        messages = manager.retrieve_all()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_initialization_invalid_type(self):
        """Test initialization with invalid memory type raises error."""
        from marsys.agents.exceptions import AgentConfigurationError

        with pytest.raises(AgentConfigurationError):
            MemoryManager(memory_type="invalid_type")

    def test_add_delegation(self):
        """Test add method delegates to underlying module."""
        manager = MemoryManager()

        msg_id = manager.add(role="user", content="Test")

        assert msg_id is not None
        messages = manager.retrieve_all()
        assert len(messages) == 1

    def test_retrieve_all_delegation(self):
        """Test retrieve_all delegates to underlying module."""
        manager = MemoryManager()
        manager.add(role="user", content="Test 1")
        manager.add(role="assistant", content="Test 2")

        result = manager.retrieve_all()

        assert len(result) == 2

    def test_get_messages_delegation(self):
        """Test get_messages delegates to underlying module."""
        manager = MemoryManager()
        manager.add(role="user", content="Test")

        result = manager.get_messages()

        assert len(result) == 1

    def test_update_delegation(self):
        """Test update delegates to underlying module."""
        manager = MemoryManager()
        msg_id = manager.add(role="user", content="Original")

        manager.update(msg_id, content="Updated")

        result = manager.retrieve_by_id(msg_id)
        assert result["content"] == "Updated"

    def test_reset_memory_delegation(self):
        """Test reset_memory delegates to underlying module."""
        manager = MemoryManager()
        manager.add(role="user", content="Test")

        manager.reset_memory()

        assert len(manager.retrieve_all()) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestMemoryIntegration:
    """Integration tests for memory components."""

    def test_message_to_memory_round_trip(self):
        """Test Message -> ConversationMemory -> dict round trip."""
        memory = ConversationMemory()

        # Create message with tool calls
        tool_call = ToolCallMsg(
            id="call_1",
            call_id="call_1",
            type="function",
            name="search",
            arguments='{"q": "test"}'
        )
        msg = Message(
            role="assistant",
            content="Searching...",
            tool_calls=[tool_call]
        )

        # Add to memory
        memory.add(message=msg)

        # Retrieve and verify
        result = memory.retrieve_all()
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]

    def test_memory_manager_with_conversation_flow(self):
        """Test MemoryManager with realistic conversation flow."""
        manager = MemoryManager(description="You are a helpful assistant.")

        # Simulate conversation
        manager.add(role="user", content="What is 2+2?")
        manager.add(role="assistant", content="2+2 equals 4")
        manager.add(role="user", content="And 3+3?")
        manager.add(role="assistant", content="3+3 equals 6")

        # Verify
        messages = manager.get_messages()
        assert len(messages) == 5  # 1 system + 4 conversation

        # Check roles
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

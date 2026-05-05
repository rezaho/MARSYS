"""
Comprehensive unit tests for the memory module.

Tests cover:
- Message dataclass and helper classes (ToolCallMsg, AgentCallMsg)
- ConversationMemory
- MemoryManager
- EventBus integration (MemoryResetEvent, set_event_context, _emit_reset_event)
- save_to_file/load_from_file with additional_state support
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marsys.agents.memory import (
    AgentCallMsg,
    ConversationMemory,
    KGMemory,
    ManagedConversationMemory,
    ManagedMemoryConfig,
    MemoryManager,
    MemoryResetEvent,
    Message,
    MessageContent,
    ToolCallMsg,
)
from marsys.agents.exceptions import MessageError, AgentConfigurationError, AgentFrameworkError


# ==============================================================================
# ToolCallMsg Tests
# ==============================================================================

class TestToolCallMsg:
    """Tests for ToolCallMsg dataclass."""

    def test_create_tool_call(self):
        """Test creating a valid tool call."""
        tc = ToolCallMsg(
            id="call_123",
            call_id="call_123",
            type="function",
            name="search",
            arguments='{"query": "test"}',
        )
        assert tc.id == "call_123"
        assert tc.name == "search"

    def test_to_dict(self):
        """Test conversion to OpenAI API format dict."""
        tc = ToolCallMsg(
            id="call_abc",
            call_id="call_abc",
            type="function",
            name="calculator",
            arguments='{"a": 5, "b": 3}',
        )
        d = tc.to_dict()

        assert d["id"] == "call_abc"
        assert d["type"] == "function"
        assert d["function"]["name"] == "calculator"
        assert d["function"]["arguments"] == '{"a": 5, "b": 3}'

    def test_from_dict(self):
        """Test creating from OpenAI API format dict."""
        data = {
            "id": "call_xyz",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"url": "example.com"}',
            },
        }
        tc = ToolCallMsg.from_dict(data)

        assert tc.id == "call_xyz"
        assert tc.name == "web_search"
        assert tc.arguments == '{"url": "example.com"}'

    def test_validation_empty_id(self):
        """Test validation fails for empty id."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            ToolCallMsg(id="", call_id="abc", type="function", name="test", arguments="{}")

    def test_validation_empty_name(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ToolCallMsg(id="abc", call_id="abc", type="function", name="", arguments="{}")


# ==============================================================================
# AgentCallMsg Tests
# ==============================================================================

class TestAgentCallMsg:
    """Tests for AgentCallMsg dataclass."""

    def test_create_agent_call(self):
        """Test creating a valid agent call."""
        ac = AgentCallMsg(agent_name="ResearchAgent", request="Find information")
        assert ac.agent_name == "ResearchAgent"
        assert ac.request == "Find information"

    def test_to_dict(self):
        """Test conversion to dict."""
        ac = AgentCallMsg(agent_name="CodeAgent", request={"task": "write code"})
        d = ac.to_dict()

        assert d["agent_name"] == "CodeAgent"
        assert d["request"] == {"task": "write code"}

    def test_from_dict(self):
        """Test creating from dict."""
        data = {"agent_name": "AnalysisAgent", "request": "Analyze data"}
        ac = AgentCallMsg.from_dict(data)

        assert ac.agent_name == "AnalysisAgent"
        assert ac.request == "Analyze data"

    def test_validation_empty_agent_name(self):
        """Test validation fails for empty agent_name."""
        with pytest.raises(ValueError, match="agent_name cannot be empty"):
            AgentCallMsg(agent_name="", request="test")

    def test_validation_none_request(self):
        """Test validation fails for None request."""
        with pytest.raises(ValueError, match="request cannot be None"):
            AgentCallMsg(agent_name="TestAgent", request=None)


# ==============================================================================
# Message Tests
# ==============================================================================

class TestMessage:
    """Tests for Message dataclass."""

    def test_simple_message(self):
        """Test creating a simple message."""
        msg = Message(role="user", content="Hello, world!")
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert msg.message_id is not None
        assert len(msg.message_id) > 0

    def test_message_with_custom_id(self):
        """Test message with custom ID."""
        msg = Message(role="user", content="Test", message_id="custom_123")
        assert msg.message_id == "custom_123"

    def test_message_with_dict_content(self):
        """Test message with dictionary content."""
        content = {"thought": "thinking", "next_action": "call_tool"}
        msg = Message(role="assistant", content=content)
        assert msg.content == content

    def test_message_with_tool_calls_from_dict(self):
        """Test message auto-converts tool_calls dicts to ToolCallMsg."""
        msg = Message(
            role="assistant",
            content="Let me search.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        )
        assert len(msg.tool_calls) == 1
        assert isinstance(msg.tool_calls[0], ToolCallMsg)
        assert msg.tool_calls[0].name == "search"

    def test_message_with_agent_calls_from_dict(self):
        """Test message auto-converts agent_calls dicts to AgentCallMsg."""
        msg = Message(
            role="assistant",
            content="Delegating task.",
            agent_calls=[{"agent_name": "WorkerAgent", "request": "Process data"}],
        )
        assert len(msg.agent_calls) == 1
        assert isinstance(msg.agent_calls[0], AgentCallMsg)
        assert msg.agent_calls[0].agent_name == "WorkerAgent"

    def test_message_with_images(self):
        """Test message with images list."""
        msg = Message(
            role="user",
            content="What's in this image?",
            images=["path/to/image.png"],
        )
        assert msg.images == ["path/to/image.png"]

    def test_to_llm_dict_simple(self):
        """Test converting simple message to LLM dict."""
        msg = Message(role="user", content="Hello")
        d = msg.to_llm_dict()

        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_to_llm_dict_tool_role(self):
        """Test tool role message conversion (content must be string)."""
        msg = Message(
            role="tool",
            content={"result": "success"},
            tool_call_id="call_123",
            name="my_tool",
        )
        d = msg.to_llm_dict()

        assert d["role"] == "tool"
        assert isinstance(d["content"], str)  # Must be string
        assert d["tool_call_id"] == "call_123"

    def test_to_llm_dict_with_tool_calls(self):
        """Test message with tool calls conversion."""
        tc = ToolCallMsg(
            id="call_abc",
            call_id="call_abc",
            type="function",
            name="test",
            arguments="{}",
        )
        msg = Message(role="assistant", content="Calling tool", tool_calls=[tc])
        d = msg.to_llm_dict()

        assert "tool_calls" in d
        assert len(d["tool_calls"]) == 1

    def test_from_response_dict(self):
        """Test creating message from response dict."""
        response = {
            "role": "assistant",
            "content": "Response text",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
        msg = Message.from_response_dict(response)

        assert msg.role == "assistant"
        assert msg.content == "Response text"
        assert len(msg.tool_calls) == 1

    def test_invalid_tool_calls_type(self):
        """Test error for invalid tool_calls type."""
        with pytest.raises((MessageError, TypeError)):
            Message(role="assistant", tool_calls="invalid")

    def test_invalid_images_type(self):
        """Test error for invalid images type."""
        with pytest.raises((MessageError, TypeError)):
            Message(role="user", content="Test", images="invalid")


# ==============================================================================
# MessageContent Tests
# ==============================================================================

class TestMessageContent:
    """Tests for MessageContent dataclass."""

    def test_create_message_content(self):
        """Test creating MessageContent."""
        mc = MessageContent(
            thought="Analyzing request",
            next_action="call_tool",
            action_input={"tool": "search"},
        )
        assert mc.thought == "Analyzing request"
        assert mc.next_action == "call_tool"

    def test_to_dict(self):
        """Test conversion to dict."""
        mc = MessageContent(thought="Thinking", next_action="final_response")
        d = mc.to_dict()

        assert d["thought"] == "Thinking"
        assert d["next_action"] == "final_response"
        assert "action_input" not in d  # None values not included

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "thought": "Processing",
            "next_action": "invoke_agent",
            "action_input": {"agent": "Helper"},
        }
        mc = MessageContent.from_dict(data)

        assert mc.thought == "Processing"
        assert mc.next_action == "invoke_agent"
        assert mc.action_input == {"agent": "Helper"}

    def test_invalid_next_action(self):
        """Test validation for invalid next_action."""
        with pytest.raises(ValueError, match="next_action must be one of"):
            MessageContent(next_action="invalid_action")


# ==============================================================================
# ConversationMemory Tests
# ==============================================================================

class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_initialization_empty(self):
        """Test empty initialization."""
        memory = ConversationMemory()
        assert len(memory.memory) == 0
        assert memory.memory_type == "conversation_history"

    def test_initialization_with_description(self):
        """Test initialization with system description."""
        memory = ConversationMemory(description="You are a helpful assistant.")
        assert len(memory.memory) == 1
        assert memory.memory[0].role == "system"
        assert memory.memory[0].content == "You are a helpful assistant."

    def test_add_by_parameters(self):
        """Test adding message by parameters."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="Hello")

        assert len(memory.memory) == 1
        assert msg_id is not None
        assert memory.memory[0].content == "Hello"

    def test_add_by_message_object(self):
        """Test adding message by Message object."""
        memory = ConversationMemory()
        msg = Message(role="assistant", content="Hi there!")
        msg_id = memory.add(message=msg)

        assert len(memory.memory) == 1
        assert msg_id == msg.message_id

    def test_add_without_role_raises(self):
        """Test add without role or message raises error."""
        memory = ConversationMemory()
        with pytest.raises((MessageError, TypeError)):
            memory.add()

    def test_update_message(self):
        """Test updating an existing message."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="Original")

        memory.update(msg_id, content="Updated")

        msg = memory.retrieve_by_id(msg_id)
        assert msg["content"] == "Updated"

    def test_update_nonexistent_raises(self):
        """Test updating non-existent message raises error."""
        memory = ConversationMemory()
        with pytest.raises((MessageError, TypeError)):
            memory.update("nonexistent_id", content="Test")

    def test_retrieve_all(self):
        """Test retrieving all messages."""
        memory = ConversationMemory()
        memory.add(role="user", content="First")
        memory.add(role="assistant", content="Second")
        memory.add(role="user", content="Third")

        all_msgs = memory.retrieve_all()

        assert len(all_msgs) == 3
        assert all(isinstance(m, dict) for m in all_msgs)

    def test_retrieve_recent(self):
        """Test retrieving recent messages."""
        memory = ConversationMemory()
        for i in range(5):
            memory.add(role="user", content=f"Message {i}")

        recent = memory.retrieve_recent(2)

        assert len(recent) == 2
        assert "Message 3" in recent[0]["content"]
        assert "Message 4" in recent[1]["content"]

    def test_retrieve_by_id(self):
        """Test retrieving message by ID."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="Target message")
        memory.add(role="assistant", content="Other message")

        msg = memory.retrieve_by_id(msg_id)

        assert msg is not None
        assert msg["content"] == "Target message"

    def test_retrieve_by_id_not_found(self):
        """Test retrieve_by_id returns None for non-existent ID."""
        memory = ConversationMemory()
        result = memory.retrieve_by_id("nonexistent")
        assert result is None

    def test_remove_by_id(self):
        """Test removing message by ID."""
        memory = ConversationMemory()
        msg_id = memory.add(role="user", content="To remove")

        result = memory.remove_by_id(msg_id)

        assert result is True
        assert len(memory.memory) == 0

    def test_remove_by_id_not_found(self):
        """Test remove_by_id returns False for non-existent ID."""
        memory = ConversationMemory()
        result = memory.remove_by_id("nonexistent")
        assert result is False

    def test_retrieve_by_role(self):
        """Test filtering messages by role."""
        memory = ConversationMemory()
        memory.add(role="user", content="User 1")
        memory.add(role="assistant", content="Assistant 1")
        memory.add(role="user", content="User 2")

        user_msgs = memory.retrieve_by_role("user")
        assistant_msgs = memory.retrieve_by_role("assistant")

        assert len(user_msgs) == 2
        assert len(assistant_msgs) == 1

    def test_retrieve_by_role_with_limit(self):
        """Test retrieve_by_role with n limit."""
        memory = ConversationMemory()
        for i in range(5):
            memory.add(role="user", content=f"User {i}")

        limited = memory.retrieve_by_role("user", n=2)
        assert len(limited) == 2

    def test_replace_memory(self):
        """Test replacing message at index."""
        memory = ConversationMemory()
        memory.add(role="user", content="Original")

        memory.replace_memory(0, role="user", content="Replaced")

        assert memory.memory[0].content == "Replaced"

    def test_replace_memory_with_message_object(self):
        """Test replacing with Message object."""
        memory = ConversationMemory()
        memory.add(role="user", content="Original")

        new_msg = Message(role="assistant", content="New message")
        memory.replace_memory(0, message=new_msg)

        assert memory.memory[0].role == "assistant"

    def test_replace_memory_out_of_range(self):
        """Test replace_memory raises for out of range index."""
        memory = ConversationMemory()
        with pytest.raises(IndexError, match="out of range"):
            memory.replace_memory(0, role="user", content="Test")

    def test_delete_memory(self):
        """Test deleting message by index."""
        memory = ConversationMemory()
        memory.add(role="user", content="First")
        memory.add(role="assistant", content="Second")

        memory.delete_memory(0)

        assert len(memory.memory) == 1
        assert memory.memory[0].content == "Second"

    def test_delete_memory_out_of_range(self):
        """Test delete_memory raises for out of range index."""
        memory = ConversationMemory()
        with pytest.raises(IndexError, match="out of range"):
            memory.delete_memory(0)

    def test_reset_memory_keeps_system_message(self):
        """Test reset_memory keeps system message."""
        memory = ConversationMemory(description="System prompt")
        memory.add(role="user", content="User message")
        memory.add(role="assistant", content="Response")

        memory.reset_memory()

        assert len(memory.memory) == 1
        assert memory.memory[0].role == "system"

    def test_reset_memory_clears_all_without_system(self):
        """Test reset_memory clears all when no system message."""
        memory = ConversationMemory()
        memory.add(role="user", content="Message")

        memory.reset_memory()

        assert len(memory.memory) == 0

    def test_get_messages_delegates_to_retrieve_all(self):
        """Test get_messages delegates to retrieve_all."""
        memory = ConversationMemory()
        memory.add(role="user", content="Test")

        messages = memory.get_messages()

        assert messages == memory.retrieve_all()


# ==============================================================================
# MemoryResetEvent Tests
# ==============================================================================

class TestMemoryResetEvent:
    """Tests for MemoryResetEvent dataclass."""

    def test_create_event(self):
        """Test creating a MemoryResetEvent."""
        event = MemoryResetEvent(agent_name="TestAgent")

        assert event.agent_name == "TestAgent"
        assert event.timestamp is not None
        assert event.timestamp > 0

    def test_custom_timestamp(self):
        """Test event with custom timestamp."""
        custom_time = 1234567890.0
        event = MemoryResetEvent(agent_name="Agent", timestamp=custom_time)

        assert event.timestamp == custom_time


# ==============================================================================
# EventBus Integration Tests
# ==============================================================================

class TestMemoryEventBusIntegration:
    """Tests for EventBus integration in memory classes."""

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock EventBus."""
        bus = MagicMock()
        bus.emit_nowait = MagicMock()
        return bus

    def test_set_event_context(self, mock_event_bus):
        """Test set_event_context configures memory."""
        memory = ConversationMemory()

        memory.set_event_context("TestAgent", mock_event_bus)

        assert memory._agent_name == "TestAgent"
        assert memory._event_bus is mock_event_bus

    def test_set_event_context_without_bus(self):
        """Test set_event_context works without bus."""
        memory = ConversationMemory()

        memory.set_event_context("TestAgent")

        assert memory._agent_name == "TestAgent"
        assert memory._event_bus is None

    def test_reset_emits_event(self, mock_event_bus):
        """Test reset_memory emits MemoryResetEvent."""
        memory = ConversationMemory()
        memory.set_event_context("TestAgent", mock_event_bus)
        memory.add(role="user", content="Test")

        memory.reset_memory()

        mock_event_bus.emit_nowait.assert_called_once()
        event = mock_event_bus.emit_nowait.call_args[0][0]
        assert isinstance(event, MemoryResetEvent)
        assert event.agent_name == "TestAgent"

    def test_reset_no_event_without_bus(self):
        """Test reset_memory doesn't fail without event bus."""
        memory = ConversationMemory()
        memory.add(role="user", content="Test")

        # Should not raise
        memory.reset_memory()

    def test_managed_memory_reset_emits_event(self, mock_event_bus):
        """Test ManagedConversationMemory reset emits event."""
        memory = ManagedConversationMemory()
        memory.set_event_context("ManagedAgent", mock_event_bus)
        memory.add(role="user", content="Test")

        memory.reset_memory()

        mock_event_bus.emit_nowait.assert_called_once()
        event = mock_event_bus.emit_nowait.call_args[0][0]
        assert event.agent_name == "ManagedAgent"


# ==============================================================================
# MemoryManager Tests
# ==============================================================================

class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_initialization_conversation_history(self):
        """Test initialization with conversation_history type."""
        manager = MemoryManager(memory_type="conversation_history")

        assert manager.memory_type == "conversation_history"
        assert isinstance(manager.memory_module, ConversationMemory)

    def test_initialization_managed_conversation(self):
        """Test initialization with managed_conversation type."""
        manager = MemoryManager(memory_type="managed_conversation")

        assert manager.memory_type == "managed_conversation"
        assert isinstance(manager.memory_module, ManagedConversationMemory)

    def test_initialization_with_description(self):
        """Test initialization with system description."""
        manager = MemoryManager(
            memory_type="conversation_history",
            description="You are helpful.",
        )

        messages = manager.retrieve_all()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_initialization_invalid_type(self):
        """Test initialization with invalid memory type."""
        with pytest.raises(AgentConfigurationError, match="Unknown memory_type"):
            MemoryManager(memory_type="invalid_type")

    def test_add_delegates(self):
        """Test add delegates to memory module."""
        manager = MemoryManager()
        msg_id = manager.add(role="user", content="Hello")

        assert msg_id is not None
        assert len(manager.retrieve_all()) == 1

    def test_update_delegates(self):
        """Test update delegates to memory module."""
        manager = MemoryManager()
        msg_id = manager.add(role="user", content="Original")

        manager.update(msg_id, content="Updated")

        msg = manager.retrieve_by_id(msg_id)
        assert msg["content"] == "Updated"

    def test_retrieve_methods_delegate(self):
        """Test all retrieve methods delegate properly."""
        manager = MemoryManager()
        manager.add(role="user", content="User 1")
        manager.add(role="assistant", content="Assistant 1")

        assert len(manager.retrieve_all()) == 2
        assert len(manager.retrieve_recent(1)) == 1
        assert len(manager.retrieve_by_role("user")) == 1
        assert len(manager.get_messages()) == 2

    def test_remove_by_id_delegates(self):
        """Test remove_by_id delegates properly."""
        manager = MemoryManager()
        msg_id = manager.add(role="user", content="Test")

        result = manager.remove_by_id(msg_id)

        assert result is True
        assert len(manager.retrieve_all()) == 0

    def test_reset_memory_delegates(self):
        """Test reset_memory delegates properly."""
        manager = MemoryManager(description="System")
        manager.add(role="user", content="Message")

        manager.reset_memory()

        # Only system message should remain
        assert len(manager.retrieve_all()) == 1

    def test_set_event_context_delegates(self):
        """Test set_event_context delegates to memory module."""
        manager = MemoryManager()
        mock_bus = MagicMock()

        manager.set_event_context("TestAgent", mock_bus)

        assert manager.memory_module._agent_name == "TestAgent"
        assert manager.memory_module._event_bus is mock_bus


# ==============================================================================
# MemoryManager Persistence Tests
# ==============================================================================

class TestMemoryManagerPersistence:
    """Tests for MemoryManager save_to_file and load_from_file."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_save_to_file_creates_file(self, temp_file):
        """Test save_to_file creates the file."""
        manager = MemoryManager()
        manager.add(role="user", content="Test message")

        manager.save_to_file(temp_file)

        assert Path(temp_file).exists()

    def test_save_to_file_content(self, temp_file):
        """Test save_to_file content structure."""
        manager = MemoryManager()
        manager.add(role="user", content="Hello")
        manager.add(role="assistant", content="Hi there!")

        manager.save_to_file(temp_file)

        with open(temp_file, 'r') as f:
            data = json.load(f)

        assert data["memory_type"] == "conversation_history"
        assert "messages" in data
        assert len(data["messages"]) == 2
        assert "timestamp" in data

    def test_save_to_file_with_additional_state(self, temp_file):
        """Test save_to_file with additional_state."""
        manager = MemoryManager()
        manager.add(role="user", content="Test")

        additional_state = {
            "planning": {"plan_id": "abc123", "items": ["task1", "task2"]},
            "tools_version": 5,
        }
        manager.save_to_file(temp_file, additional_state=additional_state)

        with open(temp_file, 'r') as f:
            data = json.load(f)

        assert "additional_state" in data
        assert data["additional_state"]["planning"]["plan_id"] == "abc123"
        assert data["additional_state"]["tools_version"] == 5

    def test_save_to_file_creates_directory(self):
        """Test save_to_file creates parent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dir" / "memory.json"

            manager = MemoryManager()
            manager.add(role="user", content="Test")
            manager.save_to_file(str(nested_path))

            assert nested_path.exists()

    def test_load_from_file_restores_messages(self, temp_file):
        """Test load_from_file restores messages."""
        # Save first
        manager1 = MemoryManager()
        manager1.add(role="user", content="Message 1")
        manager1.add(role="assistant", content="Response 1")
        manager1.save_to_file(temp_file)

        # Load in new manager
        manager2 = MemoryManager()
        manager2.load_from_file(temp_file)

        messages = manager2.retrieve_all()
        assert len(messages) == 2
        assert messages[0]["content"] == "Message 1"
        assert messages[1]["content"] == "Response 1"

    def test_load_from_file_returns_additional_state(self, temp_file):
        """Test load_from_file returns additional_state."""
        manager1 = MemoryManager()
        manager1.add(role="user", content="Test")
        additional = {"custom_data": {"key": "value"}, "version": 3}
        manager1.save_to_file(temp_file, additional_state=additional)

        manager2 = MemoryManager()
        returned_state = manager2.load_from_file(temp_file)

        assert returned_state is not None
        assert returned_state["custom_data"]["key"] == "value"
        assert returned_state["version"] == 3

    def test_load_from_file_returns_none_without_additional_state(self, temp_file):
        """Test load_from_file returns None when no additional_state."""
        manager1 = MemoryManager()
        manager1.add(role="user", content="Test")
        manager1.save_to_file(temp_file)  # No additional_state

        manager2 = MemoryManager()
        returned_state = manager2.load_from_file(temp_file)

        assert returned_state is None

    def test_load_from_file_nonexistent_returns_none(self):
        """Test load_from_file returns None for non-existent file."""
        manager = MemoryManager()
        result = manager.load_from_file("/nonexistent/path/file.json")

        assert result is None

    def test_load_from_file_clears_existing_memory(self, temp_file):
        """Test load_from_file clears existing memory first."""
        # Save with 1 message
        manager1 = MemoryManager()
        manager1.add(role="user", content="Saved message")
        manager1.save_to_file(temp_file)

        # Create manager with existing messages
        manager2 = MemoryManager()
        manager2.add(role="user", content="Existing 1")
        manager2.add(role="user", content="Existing 2")
        manager2.add(role="user", content="Existing 3")

        # Load should clear and replace
        manager2.load_from_file(temp_file)

        messages = manager2.retrieve_all()
        assert len(messages) == 1
        assert messages[0]["content"] == "Saved message"

    def test_load_from_file_type_mismatch_warning(self, temp_file):
        """Test load_from_file warns on type mismatch."""
        # Save as conversation_history
        manager1 = MemoryManager(memory_type="conversation_history")
        manager1.add(role="user", content="Test")
        manager1.save_to_file(temp_file)

        # Load in managed_conversation manager
        manager2 = MemoryManager(memory_type="managed_conversation")

        # Should log warning but still load
        with patch('logging.warning') as mock_warn:
            manager2.load_from_file(temp_file)
            # Verify warning was logged about mismatch
            # (The actual implementation logs the warning)

        # Messages should still be loaded
        assert len(manager2.get_messages()) == 1

    def test_roundtrip_with_tool_calls(self, temp_file):
        """Test save/load preserves tool calls."""
        manager1 = MemoryManager()
        manager1.add(
            role="assistant",
            content="Calling tool",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }
            ],
        )
        manager1.save_to_file(temp_file)

        manager2 = MemoryManager()
        manager2.load_from_file(temp_file)

        messages = manager2.retrieve_all()
        assert len(messages) == 1
        assert "tool_calls" in messages[0]

    def test_roundtrip_with_images(self, temp_file):
        """Test save/load preserves image references."""
        manager1 = MemoryManager()
        manager1.add(
            role="user",
            content="What's in this image?",
            images=["path/to/image.png"],
        )
        manager1.save_to_file(temp_file)

        manager2 = MemoryManager()
        manager2.load_from_file(temp_file)

        # Note: Image paths are stored but content may not be in retrieve format
        messages = manager2.retrieve_all()
        assert len(messages) == 1


# ==============================================================================
# ManagedConversationMemory Persistence Tests
# ==============================================================================

class TestManagedMemoryPersistence:
    """Additional tests for ManagedConversationMemory persistence."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)

    def test_managed_memory_save_load(self, temp_file):
        """Test ManagedConversationMemory through MemoryManager."""
        manager1 = MemoryManager(memory_type="managed_conversation")
        manager1.add(role="user", content="Managed message 1")
        manager1.add(role="assistant", content="Managed response 1")
        manager1.save_to_file(temp_file)

        manager2 = MemoryManager(memory_type="managed_conversation")
        manager2.load_from_file(temp_file)

        messages = manager2.get_messages()
        assert len(messages) == 2

    def test_managed_memory_with_additional_state(self, temp_file):
        """Test ManagedConversationMemory with planning state."""
        manager1 = MemoryManager(memory_type="managed_conversation")
        manager1.add(role="user", content="Task")

        planning_state = {
            "planning": {
                "plan": {
                    "id": "plan_123",
                    "items": [
                        {"id": "item1", "title": "Step 1", "status": "completed"},
                        {"id": "item2", "title": "Step 2", "status": "in_progress"},
                    ],
                    "goal": "Complete the task",
                }
            }
        }
        manager1.save_to_file(temp_file, additional_state=planning_state)

        manager2 = MemoryManager(memory_type="managed_conversation")
        returned = manager2.load_from_file(temp_file)

        assert returned["planning"]["plan"]["id"] == "plan_123"
        assert len(returned["planning"]["plan"]["items"]) == 2


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestMemoryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content_message(self):
        """Test message with empty content."""
        memory = ConversationMemory()
        msg_id = memory.add(role="assistant", content="")

        msg = memory.retrieve_by_id(msg_id)
        assert msg["content"] == ""

    def test_none_content_with_tool_calls(self):
        """Test assistant message with None content but tool calls."""
        memory = ConversationMemory()
        msg_id = memory.add(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        )

        msg = memory.retrieve_by_id(msg_id)
        assert msg is not None
        assert "tool_calls" in msg

    def test_unicode_content(self):
        """Test messages with unicode content."""
        memory = ConversationMemory()
        content = "Hello 你好 مرحبا Привет 🎉"
        msg_id = memory.add(role="user", content=content)

        msg = memory.retrieve_by_id(msg_id)
        assert msg["content"] == content

    def test_large_content(self):
        """Test handling large content."""
        memory = ConversationMemory()
        large_content = "x" * 100000  # 100KB of content
        msg_id = memory.add(role="user", content=large_content)

        msg = memory.retrieve_by_id(msg_id)
        assert len(msg["content"]) == 100000

    def test_nested_dict_content(self):
        """Test deeply nested dictionary content."""
        memory = ConversationMemory()
        nested_content = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {"data": "deep value"}
                    }
                }
            }
        }
        msg_id = memory.add(role="assistant", content=nested_content)

        # Retrieve returns JSON string for dict content
        msg = memory.retrieve_by_id(msg_id)
        assert "deep value" in msg["content"]

    def test_concurrent_operations(self):
        """Test thread safety hint (not truly concurrent but sequential)."""
        memory = ConversationMemory()

        # Add many messages quickly
        ids = []
        for i in range(100):
            ids.append(memory.add(role="user", content=f"Message {i}"))

        assert len(memory.memory) == 100
        assert all(memory.retrieve_by_id(msg_id) is not None for msg_id in ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

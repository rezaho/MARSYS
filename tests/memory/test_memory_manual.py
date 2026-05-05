"""
Manual tests for Memory and Message objects.
Run this file directly to test memory functionality and log results.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from marsys.agents.memory import (
    AgentCallMsg,
    ConversationMemory,
    MemoryManager,
    Message,
    MessageContent,
    ToolCallMsg,
)

# Setup logging
log_dir = Path(__file__).parent
log_file = log_dir / f"test_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Helper class to run tests and log results."""

    def __init__(self):
        self.test_count = 0
        self.passed = 0
        self.failed = 0

    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a test and log the results."""
        self.test_count += 1
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TEST {self.test_count}: {test_name}")
        logger.info(f"{'=' * 60}")

        try:
            # Log inputs
            if args:
                logger.info(f"Input args: {args}")
            if kwargs:
                logger.info(f"Input kwargs: {kwargs}")

            # Run test
            result = test_func(*args, **kwargs)

            # Log output
            logger.info(
                f"Output: {json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result}"
            )
            logger.info(f"✅ TEST PASSED: {test_name}")
            self.passed += 1

            return result

        except Exception as e:
            logger.error(f"❌ TEST FAILED: {test_name}")
            logger.error(f"Error: {type(e).__name__}: {str(e)}")
            self.failed += 1
            raise

    def print_summary(self):
        """Print test summary."""
        logger.info(f"\n{'=' * 60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total tests: {self.test_count}")
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Success rate: {(self.passed / self.test_count) * 100:.1f}%")


# Test functions
def test_message_creation():
    """Test creating Message objects with various configurations."""

    # Simple message
    msg1 = Message(role="user", content="Hello, world!")

    # Message with dictionary content
    dict_content = {
        "thought": "User is asking about weather",
        "analysis": {"location": "unknown", "time": "current"},
        "response": "I need location information"
    }
    msg2 = Message(role="assistant", content=dict_content)

    # Message with tool calls
    tool_call = ToolCallMsg(
        id="call_123",
        call_id="call_123",
        type="function",
        name="calculator",
        arguments='{"operation": "add", "a": 5, "b": 3}',
    )
    msg3 = Message(
        role="assistant", content="I'll calculate that for you.", tool_calls=[tool_call]
    )

    # Message with agent calls
    agent_call = AgentCallMsg(
        agent_name="ResearchAgent", request="Find information about Python"
    )
    msg4 = Message(
        role="assistant",
        content="I'll ask the research agent.",
        agent_calls=[agent_call],
    )

    # Message with images
    msg5 = Message(
        role="user",
        content="What's in this image?",
        images=["path/to/image1.png", "path/to/image2.jpg"],
    )

    # Test auto-generated IDs
    msg6 = Message(role="user", content="Test auto ID")
    msg7 = Message(role="user", content="Another test", message_id="custom_id_123")

    return {
        "simple_message": {"role": msg1.role, "content": msg1.content, "id": msg1.message_id},
        "dict_content_message": {
            "role": msg2.role,
            "content": msg2.content,
            "id": msg2.message_id,
            "content_type": type(msg2.content).__name__
        },
        "tool_message": {
            "role": msg3.role,
            "content": msg3.content,
            "tool_calls": len(msg3.tool_calls),
        },
        "agent_message": {
            "role": msg4.role,
            "content": msg4.content,
            "agent_calls": len(msg4.agent_calls),
        },
        "image_message": {
            "role": msg5.role,
            "content": msg5.content,
            "images": msg5.images,
        },
        "auto_id_message": {
            "id": msg6.message_id,
            "id_length": len(msg6.message_id),
            "is_uuid": "-" in msg6.message_id  # UUIDs have dashes
        },
        "custom_id_message": {
            "id": msg7.message_id,
            "custom_id_preserved": msg7.message_id == "custom_id_123"
        }
    }


def test_conversation_memory_add():
    """Test adding messages to ConversationMemory."""
    memory = ConversationMemory(description="Test conversation system")

    # Add messages using different methods
    memory.add(role="user", content="What's the weather?")
    memory.add(role="assistant", content="I'll check the weather for you.")

    # Add with Message object
    msg = Message(role="user", content="Thanks!")
    memory.add(message=msg)

    # Retrieve all messages
    all_msgs = memory.retrieve_all()

    return {"total_messages": len(all_msgs), "messages": all_msgs}


def test_conversation_memory_update():
    """Test updating messages in ConversationMemory."""
    memory = ConversationMemory()

    # Add a message and get auto-generated ID
    msg_id = memory.add(role="user", content="Original content")

    # Update the message
    memory.update(message_id=msg_id, content="Updated content")

    # Retrieve and check
    msg = memory.retrieve_by_id(msg_id)

    return {"message_id": msg_id, "updated_content": msg["content"] if msg else None}


def test_conversation_memory_retrieve():
    """Test various retrieval methods."""
    memory = ConversationMemory()

    # Add multiple messages
    memory.add(role="system", content="You are a helpful assistant.")
    memory.add(role="user", content="Hello")
    memory.add(role="assistant", content="Hi there!")
    memory.add(role="user", content="How are you?")
    memory.add(role="assistant", content="I'm doing well, thanks!")

    # Test different retrieval methods
    recent = memory.retrieve_recent(2)
    by_role = memory.retrieve_by_role("user")
    all_msgs = memory.retrieve_all()

    return {
        "total_messages": len(all_msgs),
        "recent_2": recent,
        "user_messages": by_role,
        "all_messages_count": len(all_msgs),
    }


def test_message_with_images():
    """Test message dict conversion with images."""
    memory = ConversationMemory()

    # Create a test image file
    test_image = (
        Path(__file__).parent.parent.parent
        / "tmp"
        / "screenshots"
        / "vision_analysis_screenshot.png"
    )
    test_image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    )

    # Add message with image
    memory.add(role="user", content="What's in this image?", images=[str(test_image)])

    # Retrieve and check format
    msgs = memory.retrieve_all()

    # Clean up
    test_image.unlink(missing_ok=True)

    return {
        "message_count": len(msgs),
        "has_multimodal_content": (
            isinstance(msgs[0]["content"], list) if msgs else False
        ),
        "first_message": msgs[0] if msgs else None,
    }


def test_memory_manager():
    """Test MemoryManager functionality."""
    # Test ConversationMemory through manager
    manager = MemoryManager(
        memory_type="conversation_history", description="Test system"
    )

    # Add messages
    manager.add(role="user", content="Hello, manager!")
    manager.add(role="assistant", content="Hello! How can I help?")

    # Retrieve
    all_msgs = manager.retrieve_all()
    recent = manager.retrieve_recent(1)

    return {
        "memory_type": manager.memory_type,
        "total_messages": len(all_msgs),
        "recent_message": recent[0] if recent else None,
    }


def test_message_validation():
    """Test Message validation and error handling."""
    results = {}

    # Valid message
    try:
        msg = Message(role="user", content="Valid message")
        results["valid_message"] = "Created successfully"
    except Exception as e:
        results["valid_message"] = f"Failed: {e}"

    # Invalid tool call
    try:
        msg = Message(
            role="assistant",
            tool_calls=[{"invalid": "structure"}],  # Missing required fields
        )
        results["invalid_tool_call"] = "Created (unexpected)"
    except Exception as e:
        results["invalid_tool_call"] = f"Failed as expected: {type(e).__name__}"

    # Valid tool call from dict
    try:
        msg = Message(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        )
        results["valid_tool_call_dict"] = "Created successfully"
    except Exception as e:
        results["valid_tool_call_dict"] = f"Failed: {e}"

    return results


def test_tool_and_agent_calls():
    """Test handling of tool calls and agent calls."""
    memory = ConversationMemory()

    # Add message with tool calls
    memory.add(
        role="assistant",
        content=None,  # Can be None when tool_calls present
        tool_calls=[
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "Python documentation"}',
                },
            }
        ],
    )

    # Add tool result
    memory.add(
        role="tool",
        content='{"tool_call_id": "call_abc", "output": "Found Python docs at python.org"}',
        name="search",
    )

    # Add message with agent call
    memory.add(
        role="assistant",
        agent_calls=[
            {"agent_name": "CodeAgent", "request": "Write a hello world program"}
        ],
    )

    msgs = memory.retrieve_all()

    return {"total_messages": len(msgs), "messages": msgs}


def test_memory_persistence():
    """Test memory state management."""
    memory = ConversationMemory(description="Persistent system")

    # Add messages
    memory.add(role="user", content="Message 1")
    memory.add(role="assistant", content="Response 1")
    memory.add(role="user", content="Message 2")

    # Get state before reset
    before_reset = len(memory.retrieve_all())

    # Reset memory (should keep system message)
    memory.reset_memory()

    # Get state after reset
    after_reset = memory.retrieve_all()

    return {
        "messages_before_reset": before_reset,
        "messages_after_reset": len(after_reset),
        "system_message_preserved": (
            after_reset[0]["role"] == "system" if after_reset else False
        ),
        "after_reset_messages": after_reset,
    }


def test_dict_content_storage_retrieval():
    """Test storing and retrieving messages with dictionary content."""
    memory = ConversationMemory()
    
    # Test various dictionary content types
    simple_dict = {"key": "value", "number": 42}
    nested_dict = {
        "user_info": {"name": "John", "age": 30},
        "preferences": {"theme": "dark", "language": "en"},
        "metadata": {"timestamp": "2024-01-01", "session_id": "abc123"}
    }
    action_dict = {
        "thought": "Analyzing user request",
        "next_action": "call_tool",
        "action_input": {"tool_name": "calculator", "args": {"a": 5, "b": 3}}
    }
    
    # Add messages with dict content and get IDs
    dict_1_id = memory.add(role="user", content=simple_dict)
    dict_2_id = memory.add(role="assistant", content=nested_dict)
    dict_3_id = memory.add(role="assistant", content=action_dict)
    
    # Retrieve and check
    all_msgs = memory.retrieve_all()
    dict_1 = memory.retrieve_by_id(dict_1_id)
    dict_2 = memory.retrieve_by_id(dict_2_id)
    dict_3 = memory.retrieve_by_id(dict_3_id)
    
    # Parse back the JSON strings
    import json
    dict_1_parsed = json.loads(dict_1["content"]) if dict_1 else None
    dict_2_parsed = json.loads(dict_2["content"]) if dict_2 else None
    dict_3_parsed = json.loads(dict_3["content"]) if dict_3 else None
    
    return {
        "total_messages": len(all_msgs),
        "simple_dict_preserved": dict_1_parsed == simple_dict,
        "nested_dict_preserved": dict_2_parsed == nested_dict,
        "action_dict_preserved": dict_3_parsed == action_dict,
        "retrieved_contents": {
            "simple": dict_1_parsed,
            "nested": dict_2_parsed,
            "action": dict_3_parsed
        }
    }


def main():
    """Run all tests."""
    runner = TestRunner()

    logger.info(f"Starting Memory Module Tests - {datetime.now()}")
    logger.info(f"Log file: {log_file}")

    # Run tests
    try:
        runner.run_test("Message Creation", test_message_creation)
        runner.run_test("ConversationMemory Add", test_conversation_memory_add)
        runner.run_test("ConversationMemory Update", test_conversation_memory_update)
        runner.run_test(
            "ConversationMemory Retrieve", test_conversation_memory_retrieve
        )
        runner.run_test("Message with Images", test_message_with_images)
        runner.run_test("MemoryManager", test_memory_manager)
        runner.run_test("Message Validation", test_message_validation)
        runner.run_test("Tool and Agent Calls", test_tool_and_agent_calls)
        runner.run_test("Memory Persistence", test_memory_persistence)
        runner.run_test("Dict Content Storage/Retrieval", test_dict_content_storage_retrieval)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")

    # Print summary
    runner.print_summary()
    logger.info(f"\nTest log saved to: {log_file}")


if __name__ == "__main__":
    main()

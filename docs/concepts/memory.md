# Memory

Memory management enables agents to maintain context across multi-turn conversations.

!!! info "See Also"
    For detailed API signatures, see [Memory API Reference](../api/memory.md).


## Overview

MARSYS provides a memory system that:

- **Maintains Context**: Preserves conversation history
- **Supports Multiple Types**: ConversationMemory, ManagedConversationMemory, KGMemory
- **Handles Token Limits**: ManagedConversationMemory automatically manages context size


## Core Components

### Message Structure

The fundamental unit of memory:

```python
from marsys.agents.memory import Message

@dataclass
class Message:
    role: str  # user, assistant, system, tool
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None
    message_id: str  # Auto-generated UUID
    name: Optional[str] = None  # Tool name or model name
    tool_calls: Optional[List[ToolCallMsg]] = None
    agent_calls: Optional[List[AgentCallMsg]] = None
    structured_data: Optional[Dict[str, Any]] = None
    images: Optional[List[str]] = None  # For vision models
    tool_call_id: Optional[str] = None  # For tool response messages
```

### ConversationMemory

Standard memory implementation for storing conversation history:

```python
from marsys.agents.memory import ConversationMemory

# Create memory with optional system prompt
memory = ConversationMemory(description="You are a helpful assistant")

# Add a message
message_id = memory.add(role="user", content="Hello")

# Or add a Message object directly
from marsys.agents.memory import Message
msg = Message(role="assistant", content="Hi there!")
memory.add(message=msg)

# Retrieve messages (returns List[Dict])
all_messages = memory.retrieve_all()
recent_messages = memory.retrieve_recent(n=5)

# Get messages for LLM (same as retrieve_all for ConversationMemory)
llm_messages = memory.get_messages()

# Other operations
memory.retrieve_by_id("message-uuid-here")
memory.retrieve_by_role("user", n=3)
memory.remove_by_id("message-uuid-here")
memory.reset_memory()  # Clears all except system message
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `add()` | Add a message, returns message_id |
| `update()` | Update existing message by ID |
| `retrieve_all()` | Get all messages as dicts |
| `retrieve_recent(n)` | Get last n messages as dicts |
| `get_messages()` | Get messages for LLM consumption |
| `retrieve_by_id()` | Get message by ID |
| `retrieve_by_role()` | Filter by role |
| `remove_by_id()` | Delete message by ID |
| `reset_memory()` | Clear all messages (keeps system prompt) |

### ManagedConversationMemory

Advanced memory with automatic token management:

```python
from marsys.agents.memory import ManagedConversationMemory, ManagedMemoryConfig

config = ManagedMemoryConfig(
    max_total_tokens_trigger=150000,  # When to engage context management
    target_total_tokens=100000,        # Target after pruning
    image_token_estimate=800
)

memory = ManagedConversationMemory(config=config)

# Usage is identical to ConversationMemory
memory.add(role="user", content="Hello")
messages = memory.get_messages()  # Returns curated context within token budget
```

### KGMemory

Knowledge graph memory that extracts facts from text:

```python
from marsys.agents.memory import KGMemory

# Requires a model for fact extraction
memory = KGMemory(model=your_model, description="Initial context")

# Add facts directly
memory.add_fact(role="user", subject="Paris", predicate="is capital of", obj="France")

# Or add text and extract facts automatically
memory.add(role="user", content="The Eiffel Tower is in Paris.")
# Facts are extracted asynchronously using the model
```

### MemoryManager

Factory class that creates the appropriate memory type:

```python
from marsys.agents.memory import MemoryManager

# Create ConversationMemory
manager = MemoryManager(
    memory_type="conversation_history",
    description="System prompt"
)

# Create ManagedConversationMemory
manager = MemoryManager(
    memory_type="managed_conversation",
    description="System prompt",
    memory_config=ManagedMemoryConfig(...)
)

# Create KGMemory
manager = MemoryManager(
    memory_type="kg",
    description="System prompt",
    model=your_model  # Required for KG
)

# Use like the underlying memory type
manager.add(role="user", content="Hello")
messages = manager.get_messages()

# Save/load for persistence
manager.save_to_file("memory.json")
manager.load_from_file("memory.json")
```


## Message Addition Examples

```python
from marsys.agents.memory import ConversationMemory

memory = ConversationMemory()

# User input
memory.add(role="user", content="Analyze the quarterly sales data")

# Agent response
memory.add(role="assistant", content="I'll analyze the data for you.")

# Tool call (assistant requesting a tool)
memory.add(
    role="assistant",
    content=None,
    tool_calls=[{
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "analyze_sales",
            "arguments": '{"quarter": "Q4"}'
        }
    }]
)

# Tool result
memory.add(
    role="tool",
    content='{"total_sales": 1500000, "growth": "15%"}',
    tool_call_id="call_123",
    name="analyze_sales"
)
```


## Best Practices

### 1. Use Correct Methods

```python
# CORRECT
memory.add(role="user", content="Hello")
messages = memory.retrieve_all()
memory.reset_memory()

# WRONG - these methods don't exist
# memory.add_message(...)
# memory.get_recent(...)
# memory.clear()
```

### 2. Handle Tool Results Properly

```python
# CORRECT - Link tool result to tool call
memory.add(
    role="tool",
    content=result,
    tool_call_id="call_123",
    name="tool_name"
)

# WRONG - No association
memory.add(role="tool", content=result)
```

### 3. Use ManagedConversationMemory for Long Conversations

For conversations that may exceed token limits:

```python
from marsys.agents.memory import MemoryManager, ManagedMemoryConfig

manager = MemoryManager(
    memory_type="managed_conversation",
    memory_config=ManagedMemoryConfig(
        max_total_tokens_trigger=100000,
        target_total_tokens=80000
    )
)
```


## Next Steps

<div class="grid cards" markdown="1">

- :material-message:{ .lg .middle } **[Messages](messages.md)**

    ---

    Understand message types and formats

- :material-robot:{ .lg .middle } **[Agents](agents.md)**

    ---

    How agents use memory systems

- :material-tools:{ .lg .middle } **[Tools](tools.md)**

    ---

    Tool results in memory

</div>

---

!!! success "Memory System"
    MARSYS provides ConversationMemory for basic needs, ManagedConversationMemory for automatic token management, and KGMemory for knowledge graphs.

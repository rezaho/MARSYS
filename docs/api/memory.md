# Memory API

API reference for the memory system that manages agent conversation history.

## Core Classes

### ConversationMemory

Stores conversation history as a list of Message objects.

**Import:**
```python
from marsys.agents.memory import ConversationMemory
```

**Constructor:**
```python
ConversationMemory(description: Optional[str] = None)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `description` | `str` | Initial system message content | `None` |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `add(message=None, *, role=None, content=None, name=None, tool_calls=None, agent_calls=None, images=None, tool_call_id=None) -> str` | Add message, returns message_id |
| `update` | `update(message_id, *, role=None, content=None, ...) -> None` | Update existing message by ID |
| `retrieve_all` | `retrieve_all() -> List[Dict[str, Any]]` | Get all messages as dicts |
| `retrieve_recent` | `retrieve_recent(n: int = 1) -> List[Dict[str, Any]]` | Get last n messages as dicts |
| `get_messages` | `get_messages() -> List[Dict[str, Any]]` | Get messages for LLM (same as retrieve_all) |
| `retrieve_by_id` | `retrieve_by_id(message_id: str) -> Optional[Dict]` | Get message by ID |
| `retrieve_by_role` | `retrieve_by_role(role: str, n: Optional[int] = None) -> List[Dict]` | Filter by role |
| `remove_by_id` | `remove_by_id(message_id: str) -> bool` | Delete message by ID |
| `replace_memory` | `replace_memory(idx: int, ...) -> None` | Replace message at index |
| `delete_memory` | `delete_memory(idx: int) -> None` | Delete message at index |
| `reset_memory` | `reset_memory() -> None` | Clear all (keeps system prompt) |

**Example:**
```python
memory = ConversationMemory(description="You are helpful")

# Add messages
msg_id = memory.add(role="user", content="Hello")
memory.add(role="assistant", content="Hi there!")

# Retrieve messages
all_msgs = memory.retrieve_all()  # List[Dict]
recent = memory.retrieve_recent(5)

# Clear
memory.reset_memory()  # Keeps system prompt
```

---

### ManagedConversationMemory

Conversation memory with automatic token management.

**Import:**
```python
from marsys.agents.memory import ManagedConversationMemory, ManagedMemoryConfig
```

**Constructor:**
```python
ManagedConversationMemory(
    config: Optional[ManagedMemoryConfig] = None,
    trigger_strategy: Optional[TriggerStrategy] = None,
    process_strategy: Optional[ProcessStrategy] = None,
    retrieval_strategy: Optional[RetrievalStrategy] = None,
    description: Optional[str] = None
)
```

**Methods:** Same as ConversationMemory, plus:

| Method | Description |
|--------|-------------|
| `get_raw_messages()` | Get full raw message history |
| `get_cache_stats()` | Get cache statistics |

**ManagedMemoryConfig:**
```python
@dataclass
class ManagedMemoryConfig:
    max_total_tokens_trigger: int = 150_000
    target_total_tokens: int = 100_000
    image_token_estimate: int = 800
    min_retrieval_gap_steps: int = 2
    min_retrieval_gap_tokens: int = 5000
    trigger_events: List[str] = ["add", "get_messages"]
    cache_invalidation_events: List[str] = ["add", "remove_by_id", "delete_memory"]
    token_counter: Optional[Callable] = None
    enable_headroom_percent: float = 0.1
    processing_strategy: str = "none"
```

---

### KGMemory

Knowledge graph memory storing facts as (Subject, Predicate, Object) triplets.

**Import:**
```python
from marsys.agents.memory import KGMemory
```

**Constructor:**
```python
KGMemory(
    model: Union[BaseLocalModel, BaseAPIModel, LocalProviderAdapter],
    description: Optional[str] = None
)
```

**Methods:** Same as BaseMemory, plus:

| Method | Description |
|--------|-------------|
| `add_fact(role, subject, predicate, obj)` | Add fact directly |
| `extract_and_update_from_text(input_text, role)` | Extract facts from text using model |

---

### MemoryManager

Factory that creates appropriate memory type.

**Import:**
```python
from marsys.agents.memory import MemoryManager
```

**Constructor:**
```python
MemoryManager(
    memory_type: str = "conversation_history",
    description: Optional[str] = None,
    model: Optional[Union[BaseLocalModel, LocalProviderAdapter]] = None,
    memory_config: Optional[ManagedMemoryConfig] = None,
    token_counter: Optional[Callable] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `memory_type` | `str` | `"conversation_history"`, `"managed_conversation"`, or `"kg"` | `"conversation_history"` |
| `description` | `str` | Initial system prompt | `None` |
| `model` | `BaseLocalModel/LocalProviderAdapter` | Required for `"kg"` type | `None` |
| `memory_config` | `ManagedMemoryConfig` | Config for managed memory | `None` |
| `token_counter` | `Callable` | Custom token counter | `None` |

**Methods:** Delegates to underlying memory type, plus:

| Method | Description |
|--------|-------------|
| `save_to_file(filepath: str)` | Save memory state to JSON |
| `load_from_file(filepath: str)` | Load memory state from JSON |

**Example:**
```python
# Standard memory
manager = MemoryManager(
    memory_type="conversation_history",
    description="System prompt"
)

# With token management
manager = MemoryManager(
    memory_type="managed_conversation",
    memory_config=ManagedMemoryConfig(
        max_total_tokens_trigger=100000
    )
)

# Knowledge graph
manager = MemoryManager(
    memory_type="kg",
    model=your_model
)

# Use it
manager.add(role="user", content="Hello")
msgs = manager.get_messages()
manager.save_to_file("memory.json")
```

---

### Message

Single message in conversation.

**Import:**
```python
from marsys.agents.memory import Message
```

**Constructor:**
```python
Message(
    role: str,
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None,
    message_id: str = auto_generated,
    name: Optional[str] = None,
    tool_calls: Optional[List[ToolCallMsg]] = None,
    agent_calls: Optional[List[AgentCallMsg]] = None,
    structured_data: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    tool_call_id: Optional[str] = None
)
```

**Role Values:**
- `"system"` - System instructions
- `"user"` - User input
- `"assistant"` - Agent response
- `"tool"` - Tool response

**Methods:**

| Method | Description |
|--------|-------------|
| `to_llm_dict()` | Convert to LLM API format dict |
| `to_api_format()` | Convert to OpenAI API format |
| `from_response_dict(response_dict, ...)` | Create from model response (classmethod) |
| `from_harmonized_response(response, ...)` | Create from HarmonizedResponse (classmethod) |

**Example:**
```python
# Simple message
msg = Message(role="user", content="Hello")

# With tool calls
msg = Message(
    role="assistant",
    content=None,
    tool_calls=[
        ToolCallMsg(
            id="call_123",
            call_id="call_123",
            type="function",
            name="search",
            arguments='{"query": "AI"}'
        )
    ]
)

# Tool response
msg = Message(
    role="tool",
    content='{"result": "found"}',
    tool_call_id="call_123",
    name="search"
)

# With images
msg = Message(
    role="user",
    content="Describe this",
    images=["path/to/image.jpg"]
)
```

---

### MessageContent

Structured content for agent action responses.

**Import:**
```python
from marsys.agents.memory import MessageContent
```

**Constructor:**
```python
MessageContent(
    thought: Optional[str] = None,
    next_action: Optional[str] = None,
    action_input: Optional[Dict[str, Any]] = None
)
```

**Valid next_action values:**
- `"call_tool"`
- `"invoke_agent"`
- `"final_response"`

**Methods:**

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to dictionary |
| `from_dict(data)` | Create from dictionary (classmethod) |

---

### ToolCallMsg

Tool call request in a message.

**Import:**
```python
from marsys.agents.memory import ToolCallMsg
```

**Constructor:**
```python
ToolCallMsg(
    id: str,
    call_id: str,
    type: str,
    name: str,
    arguments: str
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to OpenAI API format |
| `from_dict(data)` | Create from dict (classmethod) |

---

### AgentCallMsg

Agent invocation request.

**Import:**
```python
from marsys.agents.memory import AgentCallMsg
```

**Constructor:**
```python
AgentCallMsg(
    agent_name: str,
    request: Any
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `to_dict()` | Convert to dictionary |
| `from_dict(data)` | Create from dict (classmethod) |

**Example:**
```python
agent_call = AgentCallMsg(
    agent_name="DataProcessor",
    request="Process the sales data"
)
```

---

## Related Documentation

- [Agent API](agent-class.md) - Agent memory integration
- [Memory Concepts](../concepts/memory.md) - Memory usage patterns

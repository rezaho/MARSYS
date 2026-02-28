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
| `compact_for_payload_error(runtime: Optional[Dict[str, Any]] = None) -> bool` | Run payload-too-large recovery compaction. Returns `True` only when serialized payload bytes are reduced |

**Payload Error Recovery (`compact_for_payload_error`)**

When an LLM/API call fails with payload-too-large classification, `Agent.run_step()` can call this method to compact memory and retry.

Behavior:
- Splits messages into `prefix` and protected `tail` using assistant-round boundary semantics (`grace_recent_messages = n` protects from the n-th most recent assistant message onward)
- Runs **summarization** on the prefix using the same summarization config as normal compaction
- Re-appends the protected tail unchanged
- Preserves conversation structure (including leading system message) instead of flattening into a single user message
- Retains only selected images with a provider-aware byte budget (50% of payload limit) and optional `max_retained_images` cap
- Applies safety guards (minimum viable output and at least one `user` message)
- Treats compaction as successful only if estimated serialized payload bytes decrease
- Emits `CompactionEvent` lifecycle events (`started`, `completed`, `failed`)

**ManagedMemoryConfig:**
```python
@dataclass
class ManagedMemoryConfig:
    threshold_tokens: int = 150_000
    image_token_estimate: int = 800
    trigger_events: List[str] = ["add", "get_messages"]
    cache_invalidation_events: List[str] = ["add", "remove_by_id", "delete_memory"]
    token_counter: Optional[Callable] = None
    active_context: ActiveContextPolicyConfig = ...

    @property
    def compaction_target_tokens(self) -> int:
        """Derived: threshold_tokens * (1 - min_reduction_ratio)."""
```

**ActiveContextPolicyConfig (key fields):**
```python
@dataclass
class ActiveContextPolicyConfig:
    enabled: bool = True
    mode: str = "compaction"  # "compaction" | "sliding_window"
    processor_order: List[str] = ["tool_truncation", "summarization", "backward_packing"]
    excluded_processors: List[str] = []
    min_reduction_ratio: float = 0.4
    tool_truncation: ToolTruncationConfig = ...
    summarization: SummarizationConfig = ...
    backward_packing: BackwardPackingConfig = ...
```

**SummarizationConfig (key fields):**
```python
@dataclass
class SummarizationConfig:
    enabled: bool = True
    grace_recent_messages: int = 1
    output_max_tokens: int = 6000
    include_original_instruction: bool = True
    include_image_payload_bytes: bool = False
    max_retained_images: Optional[int] = None  # Payload recovery defaults to 5 when unset
```

Notes:
- `grace_recent_messages` uses assistant-round semantics (protect from the N-th most recent assistant message onward).
- `max_retained_images` limits retained image count in summarization output.
- Payload recovery compaction additionally enforces a provider-aware byte budget for retained images.

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
| `set_event_context(agent_name: str, event_bus: Optional[EventBus], session_id: Optional[str] = None)` | Enable memory event emission (`MemoryResetEvent`, `CompactionEvent`) |
| `compact_if_needed(runtime: Optional[Dict[str, Any]] = None)` | Trigger token-threshold compaction when supported by the underlying memory module |
| `compact_for_payload_error(runtime: Optional[Dict[str, Any]] = None) -> bool` | Trigger payload-too-large recovery compaction when supported by the underlying memory module |
| `save_to_file(filepath: str, additional_state: Optional[Dict])` | Save memory state to JSON (optionally with extra state) |
| `load_from_file(filepath: str) -> Optional[Dict]` | Load memory state and return additional_state if present |

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
        threshold_tokens=100000
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

### MemoryResetEvent

`MemoryResetEvent` is emitted when memory is cleared (e.g., `reset_memory()`), enabling planning state to auto-clear.

**Import:**
```python
from marsys.agents.memory import MemoryResetEvent
```

**Usage:**
```python
from marsys.coordination.event_bus import EventBus

bus = EventBus()
manager.set_event_context(agent_name="Researcher", event_bus=bus, session_id="run_123")
```

---

### CompactionEvent

`CompactionEvent` is emitted by managed memory compaction to report lifecycle progress to status channels.

**Import:**
```python
from marsys.coordination.status.events import CompactionEvent
```

**Fields:**
```python
CompactionEvent(
    session_id: str,
    agent_name: str,
    status: str,              # "started" | "completed" | "failed"
    pre_tokens: int = 0,
    post_tokens: int = 0,
    pre_messages: int = 0,
    post_messages: int = 0,
    duration: Optional[float] = None,
    stages_run: Optional[List[str]] = None,
)
```

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
    tool_call_id: Optional[str] = None,
    reasoning_details: Optional[List[Dict[str, Any]]] = None
)
```

!!! note "Reasoning Details"
    The `reasoning_details` field preserves model thinking/reasoning traces (e.g., Gemini 3 thought signatures). This is critical for multi-turn tool calling with models that use extended thinking.

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

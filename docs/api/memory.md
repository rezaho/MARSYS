# Memory API

Complete API reference for the conversation memory system that manages agent memory and message history.

## 🎯 Overview

The Memory API provides comprehensive memory management for agents, including conversation history, message formatting, and memory retention policies.

## 📦 Core Classes

### ConversationMemory

Manages conversation history for an agent.

**Import:**
```python
from src.agents.memory import ConversationMemory
```

**Constructor:**
```python
ConversationMemory(
    max_messages: Optional[int] = None,
    retention_policy: str = "session"
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_messages` | `int` | Maximum messages to retain | `None` (unlimited) |
| `retention_policy` | `str` | Memory retention policy | `"session"` |

**Methods:**

#### add_message
```python
def add_message(message: Message) -> None
```
Add a message to memory.

#### get_messages
```python
def get_messages() -> List[Message]
```
Get all messages in memory.

#### get_recent
```python
def get_recent(n: int = 10) -> List[Message]
```
Get n most recent messages.

#### clear
```python
def clear() -> None
```
Clear all messages.

#### to_dict
```python
def to_dict() -> List[Dict[str, Any]]
```
Convert messages to dictionary format.

#### save_to_file
```python
def save_to_file(filepath: Path) -> None
```
Save memory to JSON file.

#### load_from_file
```python
def load_from_file(filepath: Path) -> None
```
Load memory from JSON file.

**Example:**
```python
memory = ConversationMemory(max_messages=100)

# Add messages
memory.add_message(Message(
    role="user",
    content="What is the weather?"
))

memory.add_message(Message(
    role="assistant",
    content="I'll help you check the weather."
))

# Get recent messages
recent = memory.get_recent(5)

# Save to file
memory.save_to_file(Path("conversation.json"))
```

---

### Message

Represents a single message in conversation.

**Import:**
```python
from src.agents.memory import Message
```

**Constructor:**
```python
Message(
    role: str,
    content: Optional[Union[str, Dict[str, Any]]] = None,
    message_id: str = auto_generated,
    name: Optional[str] = None,
    tool_calls: Optional[List[ToolCallMsg]] = None,
    agent_calls: Optional[List[AgentCallMsg]] = None,
    structured_data: Optional[Dict[str, Any]] = None,
    images: Optional[List[str]] = None,
    tool_call_id: Optional[str] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `role` | `str` | Message role | Required |
| `content` | `Union[str, Dict]` | Message content | `None` |
| `message_id` | `str` | Unique identifier | Auto-generated |
| `name` | `str` | Tool/model name | `None` |
| `tool_calls` | `List[ToolCallMsg]` | Tool call requests | `None` |
| `agent_calls` | `List[AgentCallMsg]` | Agent invocations | `None` |
| `structured_data` | `Dict` | Structured response data | `None` |
| `images` | `List[str]` | Image paths/URLs | `None` |
| `tool_call_id` | `str` | Link to tool call | `None` |

**Role Values:**
- `"system"` - System instructions
- `"user"` - User input
- `"assistant"` - Agent response
- `"tool"` - Tool response

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `to_dict()` | Convert to dictionary | `Dict[str, Any]` |
| `from_dict(data)` | Create from dictionary | `Message` |
| `is_valid_role(role)` | Check if role is valid | `bool` |

**Example:**
```python
# Simple text message
msg = Message(
    role="user",
    content="Analyze this data"
)

# Assistant with tool calls
msg = Message(
    role="assistant",
    content="I'll search for that information.",
    tool_calls=[
        ToolCallMsg(
            id="call_123",
            call_id="call_123",
            type="function",
            name="search",
            arguments='{"query": "AI trends"}'
        )
    ]
)

# Structured data response
msg = Message(
    role="assistant",
    structured_data={
        "analysis": "Complete",
        "results": [1, 2, 3]
    }
)

# Vision message with images
msg = Message(
    role="user",
    content="What's in this image?",
    images=["path/to/image.jpg"]
)
```

---

### MemoryManager

Manages memory for multiple agents in a session.

**Import:**
```python
from src.agents.memory import MemoryManager
```

**Constructor:**
```python
MemoryManager(
    retention_policy: str = "session",
    max_messages_per_agent: int = 1000
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `retention_policy` | `str` | Default retention policy | `"session"` |
| `max_messages_per_agent` | `int` | Max messages per agent | `1000` |

**Retention Policies:**
- `"single_run"` - Clear after each run
- `"session"` - Keep for session duration
- `"persistent"` - Persist across sessions

**Methods:**

#### get_memory
```python
def get_memory(agent_name: str) -> ConversationMemory
```
Get or create memory for agent.

#### clear_agent_memory
```python
def clear_agent_memory(agent_name: str) -> None
```
Clear memory for specific agent.

#### clear_all
```python
def clear_all() -> None
```
Clear all agent memories.

#### set_retention_policy
```python
def set_retention_policy(
    agent_name: str,
    policy: str
) -> None
```
Set retention policy for agent.

**Example:**
```python
manager = MemoryManager(retention_policy="session")

# Get memory for agent
agent_memory = manager.get_memory("Analyzer")
agent_memory.add_message(msg)

# Set custom retention
manager.set_retention_policy("Analyzer", "persistent")

# Clear specific agent
manager.clear_agent_memory("Analyzer")

# Clear all
manager.clear_all()
```

---

### MessageContent

Structured content for agent responses.

**Import:**
```python
from src.agents.memory import MessageContent
```

**Constructor:**
```python
MessageContent(
    thought: Optional[str] = None,
    next_action: Optional[str] = None,
    action_input: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `thought` | `str` | Agent's reasoning | `None` |
| `next_action` | `str` | Next action to take | `None` |
| `action_input` | `Dict` | Action parameters | `None` |

**Valid Actions:**
- `"call_tool"` - Execute tool
- `"invoke_agent"` - Call another agent
- `"final_response"` - Return final result

**Example:**
```python
content = MessageContent(
    thought="I need to analyze this data",
    next_action="invoke_agent",
    action_input={"agent_name": "DataAnalyzer"}
)

# Convert to dict
data = content.to_dict()

# Create from dict
content = MessageContent.from_dict(data)
```

---

### ToolCallMsg

Represents a tool call in a message.

**Import:**
```python
from src.agents.memory import ToolCallMsg
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

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `id` | `str` | Unique identifier | Required |
| `call_id` | `str` | Call identifier | Required |
| `type` | `str` | Call type | Required |
| `name` | `str` | Tool name | Required |
| `arguments` | `str` | JSON arguments | Required |

**Example:**
```python
tool_call = ToolCallMsg(
    id="call_123",
    call_id="call_123",
    type="function",
    name="calculate",
    arguments='{"a": 5, "b": 3}'
)

# Convert to OpenAI format
openai_format = tool_call.to_dict()
```

---

### AgentCallMsg

Represents an agent invocation call.

**Import:**
```python
from src.agents.memory import AgentCallMsg
```

**Constructor:**
```python
AgentCallMsg(
    agent_name: str,
    request: Any
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agent_name` | `str` | Target agent name | Required |
| `request` | `Any` | Request data | Required |

**Example:**
```python
agent_call = AgentCallMsg(
    agent_name="DataProcessor",
    request="Process the quarterly sales data"
)
```

---

## 🎨 Memory Patterns

### Conversation Tracking

```python
class ConversationTracker:
    """Track multi-turn conversations."""

    def __init__(self):
        self.memory = ConversationMemory()
        self.turn_count = 0

    def add_turn(self, user_msg: str, assistant_msg: str):
        """Add a conversation turn."""
        self.memory.add_message(Message(
            role="user",
            content=user_msg
        ))
        self.memory.add_message(Message(
            role="assistant",
            content=assistant_msg
        ))
        self.turn_count += 1

    def get_context(self, max_turns: int = 5) -> List[Message]:
        """Get recent conversation context."""
        return self.memory.get_recent(max_turns * 2)
```

### Memory Summarization

```python
class SummarizingMemory(ConversationMemory):
    """Memory that summarizes old messages."""

    def __init__(self, max_messages: int = 50):
        super().__init__(max_messages)
        self.summaries = []

    def add_message(self, message: Message):
        """Add message and summarize if needed."""
        super().add_message(message)

        if len(self.messages) >= self.max_messages:
            # Summarize oldest 25 messages
            old_messages = self.messages[:25]
            summary = self._summarize(old_messages)
            self.summaries.append(summary)
            # Remove summarized messages
            self.messages = self.messages[25:]

    def _summarize(self, messages: List[Message]) -> str:
        """Create summary of messages."""
        # Implement summarization logic
        pass
```

### Memory Persistence

```python
class PersistentMemory:
    """Memory with automatic persistence."""

    def __init__(self, agent_name: str, base_path: Path):
        self.agent_name = agent_name
        self.filepath = base_path / f"{agent_name}_memory.json"
        self.memory = ConversationMemory()
        self._load()

    def _load(self):
        """Load from disk if exists."""
        if self.filepath.exists():
            self.memory.load_from_file(self.filepath)

    def save(self):
        """Save to disk."""
        self.memory.save_to_file(self.filepath)

    def add_message(self, message: Message):
        """Add message and auto-save."""
        self.memory.add_message(message)
        self.save()
```

---

## 🔧 Message Formatting

### For Different Models

```python
def format_for_openai(messages: List[Message]) -> List[Dict]:
    """Format messages for OpenAI API."""
    formatted = []
    for msg in messages:
        entry = {"role": msg.role}

        if msg.content:
            entry["content"] = msg.content
        if msg.name:
            entry["name"] = msg.name
        if msg.tool_calls:
            entry["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id

        formatted.append(entry)
    return formatted

def format_for_anthropic(messages: List[Message]) -> List[Dict]:
    """Format messages for Anthropic API."""
    # Anthropic has different format
    formatted = []
    for msg in messages:
        if msg.role == "system":
            # Handle system message differently
            pass
        else:
            formatted.append({
                "role": msg.role,
                "content": msg.content
            })
    return formatted
```

---

## 📋 Best Practices

### ✅ DO:
- Set appropriate max_messages limits
- Use correct retention policies
- Save important conversations
- Clear memory when appropriate
- Validate message roles

### ❌ DON'T:
- Keep unlimited message history
- Mix retention policies inappropriately
- Store sensitive data without encryption
- Ignore memory limits in long sessions
- Create circular references in messages

---

## 🚦 Memory Lifecycle

### Complete Flow

```python
# 1. Initialize memory manager
manager = MemoryManager(retention_policy="session")

# 2. Agent gets memory
agent_memory = manager.get_memory("Agent1")

# 3. Add messages during execution
agent_memory.add_message(Message(
    role="user",
    content="Start analysis"
))

# 4. Process with context
context = agent_memory.get_recent(10)
# ... agent processing ...

# 5. Save results
agent_memory.add_message(Message(
    role="assistant",
    content="Analysis complete"
))

# 6. Persist if needed
if important_session:
    agent_memory.save_to_file(Path("important.json"))

# 7. Clear after session
manager.clear_all()
```

---

## 🚦 Related Documentation

- [Agent API](agent-class.md) - Agent memory integration
- [Execution API](execution.md) - Memory in execution
- [State API](state.md) - Memory persistence
- [Memory Patterns](../concepts/memory-patterns.md) - Memory strategies

---

!!! tip "Pro Tip"
    Use `get_recent()` instead of `get_messages()` for large conversations to limit token usage when sending to LLMs.

!!! warning "Memory Limits"
    Always set reasonable `max_messages` limits to prevent memory issues in long-running sessions. Consider implementing summarization for very long conversations.
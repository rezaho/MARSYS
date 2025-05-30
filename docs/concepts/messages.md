# Messages

Messages are the fundamental communication unit in the Multi-Agent Reasoning Systems (MARSYS) framework. They follow the OpenAI message format while extending it for multi-agent scenarios.

## Message Structure

```python
@dataclass
class Message:
    role: str                    # Role of the message sender
    content: str                 # Main message content
    name: Optional[str]          # Agent/tool name
    message_id: str              # Unique identifier
    tool_calls: Optional[List]   # Tool invocations
    tool_call_id: Optional[str]  # For tool responses
    metadata: Optional[Dict]     # Additional data
    timestamp: datetime          # Creation time
```

## Message Roles

The framework supports standard and extended roles:

| Role | Description | Example |
|------|-------------|---------|
| `system` | System instructions | Initial agent instructions |
| `user` | User input | Human queries or commands |
| `assistant` | AI response | Agent responses |
| `tool` | Tool result | Function execution results |
| `error` | Error message | Exception or failure info |
| `agent_call` | Agent invocation | One agent calling another |
| `agent_response` | Agent reply | Response from invoked agent |

## Creating Messages

### Basic Messages

```python
from src.agents.memory import Message

# User message
user_msg = Message(
    role="user",
    content="What's the weather like?"
)

# Assistant response
assistant_msg = Message(
    role="assistant",
    content="I'll check the weather for you.",
    name="weather_agent"
)

# System message
system_msg = Message(
    role="system",
    content="You are a helpful weather assistant."
)
```

### Tool Messages

```python
# Assistant calling a tool
tool_call_msg = Message(
    role="assistant",
    content="",  # Can be empty for tool calls
    tool_calls=[{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "New York"}'
        }
    }]
)

# Tool response
tool_response = Message(
    role="tool",
    content='{"temperature": 72, "condition": "sunny"}',
    name="get_weather",
    tool_call_id="call_abc123"
)
```

### Agent Communication Messages

```python
# Agent A invoking Agent B
agent_call = Message(
    role="agent_call",
    content="Research quantum computing basics",
    name="researcher_agent",
    metadata={
        "caller": "orchestrator_agent",
        "task_id": "task_123"
    }
)

# Agent B's response
agent_response = Message(
    role="agent_response",
    content="Quantum computing uses quantum bits...",
    name="researcher_agent",
    metadata={
        "task_id": "task_123",
        "completion_time": 2.5
    }
)
```

## Message Conversion

### To LLM Format

```python
# Convert for OpenAI-compatible APIs
llm_dict = message.to_llm_dict()

# Example output:
{
    "role": "assistant",
    "content": "I'll help you with that.",
    "name": "helper_agent"
}

# Tool calls are preserved
{
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {"name": "search", "arguments": "..."}
    }]
}
```

### From LLM Response

```python
# Convert LLM response to Message
llm_response = {
    "role": "assistant",
    "content": "Here's what I found...",
    "tool_calls": None
}

message = Message.from_response_dict(
    llm_response,
    processor=agent._input_message_processor()
)
```

## Message Processing

### Input Processing

Transform LLM responses to Messages:

```python
def process_llm_response(response_dict: Dict) -> Message:
    """Process LLM response into Message object."""
    if "tool_calls" in response_dict:
        return Message(
            role="assistant",
            content=response_dict.get("content", ""),
            tool_calls=response_dict["tool_calls"]
        )
    
    return Message(
        role="assistant",
        content=response_dict["content"]
    )
```

### Output Processing

Transform Messages for LLM consumption:

```python
def process_for_llm(message: Message) -> Dict:
    """Transform Message for LLM."""
    # Convert agent_call to user message
    if message.role == "agent_call":
        return {
            "role": "user",
            "content": f"[Request from {message.metadata.get('caller')}]: {message.content}"
        }
    
    # Standard conversion
    return message.to_llm_dict()
```

## Message Validation

### Role Validation

```python
VALID_ROLES = {"system", "user", "assistant", "tool", "error", "agent_call", "agent_response"}

def validate_message(message: Message) -> bool:
    """Validate message format."""
    if message.role not in VALID_ROLES:
        return False
    
    if message.role == "tool" and not message.tool_call_id:
        return False
    
    if message.role == "tool" and not message.name:
        return False
    
    return True
```

### Content Validation

```python
def validate_tool_response(message: Message) -> bool:
    """Validate tool response format."""
    if message.role != "tool":
        return False
    
    try:
        # Tool responses should be valid JSON
        import json
        json.loads(message.content)
        return True
    except:
        return False
```

## Message Patterns

### Conversation Flow

```python
# Standard conversation pattern
messages = [
    Message(role="system", content="You are a helpful assistant"),
    Message(role="user", content="What's 2+2?"),
    Message(role="assistant", content="2+2 equals 4"),
    Message(role="user", content="What about 3+3?"),
    Message(role="assistant", content="3+3 equals 6")
]
```

### Tool Usage Pattern

```python
# Tool invocation flow
flow = [
    Message(role="user", content="What's the weather in Paris?"),
    Message(
        role="assistant",
        content="I'll check the weather in Paris for you.",
        tool_calls=[{
            "id": "call_1",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}
        }]
    ),
    Message(
        role="tool",
        name="get_weather",
        content='{"temp": 18, "conditions": "partly cloudy"}',
        tool_call_id="call_1"
    ),
    Message(
        role="assistant",
        content="The weather in Paris is 18°C and partly cloudy."
    )
]
```

### Error Handling Pattern

```python
# Error message pattern
try:
    result = await some_operation()
except Exception as e:
    error_msg = Message(
        role="error",
        content=str(e),
        name=agent.name,
        metadata={
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }
    )
    return error_msg
```

## Best Practices

1. **Immutability**: Never modify messages after creation
2. **Unique IDs**: Always preserve message_id when forwarding
3. **Role Consistency**: Use appropriate roles for each message type
4. **Tool Naming**: Sanitize tool names (remove "functions." prefix)
5. **Error Messages**: Return errors as Message objects, not exceptions
6. **Metadata**: Use metadata for additional context, not core data

## Advanced Usage

### Message Chaining

```python
class MessageChain:
    """Track related messages."""
    def __init__(self, initial_message: Message):
        self.chain_id = initial_message.message_id
        self.messages = [initial_message]
    
    def add_response(self, message: Message):
        message.metadata = message.metadata or {}
        message.metadata["chain_id"] = self.chain_id
        self.messages.append(message)
    
    def get_chain(self) -> List[Message]:
        return self.messages.copy()
```

### Message Filtering

```python
def filter_messages(messages: List[Message], **criteria) -> List[Message]:
    """Filter messages by criteria."""
    filtered = messages
    
    if "role" in criteria:
        filtered = [m for m in filtered if m.role == criteria["role"]]
    
    if "name" in criteria:
        filtered = [m for m in filtered if m.name == criteria["name"]]
    
    if "after" in criteria:
        filtered = [m for m in filtered if m.timestamp > criteria["after"]]
    
    return filtered
```

## Next Steps

- Learn about [Memory](memory.md) - How messages are stored
- Understand [Tools](tools.md) - Tool message patterns
- Explore [Agent Communication](communication.md) - Multi-agent messaging

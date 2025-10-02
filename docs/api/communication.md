# Communication API

Complete API reference for the user interaction and communication system in multi-agent workflows.

## 🎯 Overview

The Communication API provides bi-directional user interaction capabilities, supporting synchronous terminal interactions, asynchronous web interfaces, and event-driven communication patterns.

## 📦 Core Classes

### CommunicationManager

Central manager for user communication across different channels.

**Import:**
```python
from src.coordination.communication import CommunicationManager
from src.coordination.config import CommunicationConfig
```

**Constructor:**
```python
CommunicationManager(
    config: Optional[CommunicationConfig] = None
)
```

**Key Methods:**

#### register_channel
```python
def register_channel(channel: CommunicationChannel) -> None
```
Register a communication channel.

#### request_user_input
```python
async def request_user_input(
    prompt: str,
    session_id: str,
    channel_id: Optional[str] = None,
    timeout: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Request input from user.

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | `str` | Prompt to display | Required |
| `session_id` | `str` | Session identifier | Required |
| `channel_id` | `str` | Target channel | Auto-select |
| `timeout` | `float` | Input timeout | From config |
| `metadata` | `Dict` | Additional metadata | `None` |

#### send_message
```python
async def send_message(
    message: str,
    session_id: str,
    channel_id: Optional[str] = None,
    message_type: str = "info"
) -> None
```
Send message to user.

#### subscribe
```python
def subscribe(
    topic: str,
    callback: Callable[[Any], None]
) -> str
```
Subscribe to communication events.

**Example:**
```python
# Initialize manager
config = CommunicationConfig(
    use_rich_formatting=True,
    theme_name="modern"
)
manager = CommunicationManager(config)

# Request user input
response = await manager.request_user_input(
    prompt="What would you like to analyze?",
    session_id="session_123"
)

# Send message
await manager.send_message(
    message="Analysis complete!",
    session_id="session_123",
    message_type="success"
)
```

---

### UserNodeHandler

Handles execution when control reaches a User node in topology.

**Import:**
```python
from src.coordination.communication import UserNodeHandler
```

**Constructor:**
```python
UserNodeHandler(
    communication_manager: CommunicationManager,
    event_bus: Optional[EventBus] = None
)
```

**Key Methods:**

#### handle_user_node
```python
async def handle_user_node(
    branch: ExecutionBranch,
    incoming_message: Any,
    context: Dict[str, Any]
) -> StepResult
```
Handle User node execution.

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `branch` | `ExecutionBranch` | Current execution branch | Required |
| `incoming_message` | `Any` | Message from calling agent | Required |
| `context` | `Dict` | Execution context | Required |

**Returns:** `StepResult` with user response and routing decision

**Example:**
```python
handler = UserNodeHandler(communication_manager)

# Handle user interaction
result = await handler.handle_user_node(
    branch=current_branch,
    incoming_message="Please approve the analysis results",
    context={"session_id": "session_123"}
)

# Process result
if result.success:
    user_response = result.data["response"]
    next_agent = result.data.get("next_agent")
```

---

### CommunicationChannel (Abstract)

Base class for communication channels.

**Import:**
```python
from src.coordination.communication import CommunicationChannel
```

**Abstract Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `send(message, metadata)` | Send message to user | `None` |
| `receive(timeout)` | Receive user input | `str` |
| `is_available()` | Check if channel available | `bool` |
| `close()` | Close channel | `None` |

---

### TerminalChannel

Basic terminal I/O channel.

**Import:**
```python
from src.coordination.communication.channels import TerminalChannel
```

**Constructor:**
```python
TerminalChannel(
    channel_id: str = "terminal",
    use_colors: bool = True
)
```

**Example:**
```python
channel = TerminalChannel()

# Send message
await channel.send("Enter your choice:")

# Receive input
response = await channel.receive(timeout=30.0)
```

---

### EnhancedTerminalChannel

Rich terminal with advanced formatting.

**Import:**
```python
from src.coordination.communication.channels import EnhancedTerminalChannel
```

**Constructor:**
```python
EnhancedTerminalChannel(
    channel_id: str = "terminal",
    use_rich: bool = True,
    theme_name: str = "modern",
    prefix_width: int = 20,
    show_timestamps: bool = True
)
```

**Features:**
- Rich text formatting
- Color themes
- Progress indicators
- Tables and panels
- Markdown rendering

**Example:**
```python
channel = EnhancedTerminalChannel(
    theme_name="modern",
    use_rich=True
)

# Send formatted message
await channel.send({
    "content": "## Analysis Results",
    "format": "markdown",
    "style": "success"
})
```

---

### PrefixedCLIChannel

CLI channel with agent name prefixes.

**Import:**
```python
from src.coordination.communication.channels import PrefixedCLIChannel
```

**Constructor:**
```python
PrefixedCLIChannel(
    channel_id: str = "cli",
    prefix_width: int = 20,
    show_timestamps: bool = False,
    prefix_alignment: str = "left"
)
```

**Example:**
```python
channel = PrefixedCLIChannel(
    prefix_width=25,
    show_timestamps=True
)

# Send with prefix
await channel.send(
    message="Processing data...",
    metadata={"agent_name": "DataProcessor"}
)
# Output: [DataProcessor]     Processing data...
```

---

## 🎨 Communication Modes

### CommunicationMode

Enumeration of communication patterns.

**Import:**
```python
from src.coordination.communication import CommunicationMode
```

**Values:**
| Mode | Description | Use Case |
|------|-------------|----------|
| `SYNC` | Synchronous blocking | Terminal input |
| `ASYNC_PUBSUB` | Event-driven async | Web interfaces |
| `ASYNC_QUEUE` | Queue-based async | Message systems |

### Mode Selection

```python
# Sync mode (terminal)
manager = CommunicationManager(
    config=CommunicationConfig(
        mode=CommunicationMode.SYNC
    )
)

# Async pub/sub (web)
manager = CommunicationManager(
    config=CommunicationConfig(
        mode=CommunicationMode.ASYNC_PUBSUB
    )
)
```

---

## 🔧 User Interaction Patterns

### Simple User Input

```python
# In topology
topology = {
    "nodes": ["Agent1", "User", "Agent2"],
    "edges": [
        "Agent1 -> User",
        "User -> Agent2"
    ]
}

# Agent1 response triggers user interaction
response = {
    "next_action": "invoke_agent",
    "action_input": "User",
    "message": "Please review the results"
}

# User sees message and provides input
# System routes response to Agent2
```

### Error Recovery

```python
# Agent triggers error recovery
response = {
    "next_action": "error_recovery",
    "error_details": {
        "type": "api_quota_exceeded",
        "message": "OpenAI API quota exceeded"
    },
    "suggested_action": "retry_with_different_model"
}

# Routes to User node for intervention
# User can:
# - Retry with same settings
# - Change model
# - Skip operation
# - Abort workflow
```

### Approval Workflow

```python
# Configure approval interaction
interaction = UserInteraction(
    id="approval_123",
    prompt="Approve deployment to production?",
    options=["approve", "reject", "review"],
    metadata={
        "changes": ["Update API", "Database migration"],
        "risk_level": "medium"
    }
)

# Request approval
response = await manager.request_structured_input(interaction)

if response == "approve":
    # Continue with deployment
    pass
elif response == "review":
    # Show detailed changes
    pass
```

---

## 🔄 Event System

### EventBus

Event bus for communication events.

**Import:**
```python
from src.coordination.communication import EventBus
```

**Methods:**

#### publish
```python
async def publish(
    event: str,
    data: Any,
    session_id: Optional[str] = None
) -> None
```
Publish event to subscribers.

#### subscribe
```python
def subscribe(
    event: str,
    callback: Callable[[Any], None]
) -> str
```
Subscribe to events.

**Events:**
- `user.input.requested` - Input requested
- `user.input.received` - Input received
- `user.message.sent` - Message sent
- `channel.connected` - Channel connected
- `channel.disconnected` - Channel disconnected

**Example:**
```python
bus = EventBus()

# Subscribe to events
def on_input(data):
    print(f"User input: {data['response']}")

bus.subscribe("user.input.received", on_input)

# Publish event
await bus.publish(
    "user.input.received",
    {"response": "yes", "session_id": "123"}
)
```

---

## 🎨 Custom Channels

### Creating Custom Channel

```python
from src.coordination.communication import CommunicationChannel

class WebSocketChannel(CommunicationChannel):
    """WebSocket-based communication channel."""

    def __init__(self, ws_url: str):
        super().__init__(channel_id="websocket")
        self.ws_url = ws_url
        self.ws = None

    async def connect(self):
        self.ws = await websocket.connect(self.ws_url)

    async def send(self, message: str, metadata: Dict = None):
        await self.ws.send(json.dumps({
            "message": message,
            "metadata": metadata
        }))

    async def receive(self, timeout: float = None):
        try:
            data = await asyncio.wait_for(
                self.ws.receive(),
                timeout=timeout
            )
            return json.loads(data)["response"]
        except asyncio.TimeoutError:
            return None

    def is_available(self) -> bool:
        return self.ws and not self.ws.closed

    async def close(self):
        if self.ws:
            await self.ws.close()
```

---

## 📋 Best Practices

### ✅ DO:
- Set appropriate timeouts for user input
- Provide clear prompts and options
- Handle timeout gracefully
- Store interaction history
- Use structured interactions for complex inputs

### ❌ DON'T:
- Block indefinitely on user input
- Mix communication channels in same session
- Ignore channel availability
- Send sensitive data unencrypted
- Assume user always provides valid input

---

## 🚦 Themes and Formatting

### Available Themes

```python
# Modern theme (default)
config = CommunicationConfig(theme_name="modern")

# Classic terminal
config = CommunicationConfig(theme_name="classic")

# Minimal
config = CommunicationConfig(theme_name="minimal")

# Custom theme
config = CommunicationConfig(
    theme_name="custom",
    custom_theme={
        "primary": "#007ACC",
        "success": "#4CAF50",
        "error": "#F44336",
        "warning": "#FF9800"
    }
)
```

### Message Formatting

```python
# Rich formatting
await channel.send({
    "content": "# Results\n- Item 1\n- Item 2",
    "format": "markdown",
    "style": "panel",
    "title": "Analysis"
})

# Table display
await channel.send({
    "content": [
        ["Metric", "Value"],
        ["Accuracy", "95%"],
        ["Speed", "1.2s"]
    ],
    "format": "table"
})
```

---

## 🚦 Related Documentation

- [User Node Guide](../concepts/user-node.md) - User node patterns
- [Configuration API](configuration.md) - Communication configuration
- [Orchestra API](orchestra.md) - Integration with Orchestra
- [Interactive Workflows](../usage/interactive-workflows.md) - Interactive patterns

---

!!! tip "Pro Tip"
    Use `EnhancedTerminalChannel` for better user experience with colors, formatting, and progress indicators. It automatically falls back to basic terminal if Rich is unavailable.

!!! warning "Timeout Handling"
    Always set reasonable timeouts for user input to prevent indefinite blocking. Consider offering a retry option if timeout occurs.
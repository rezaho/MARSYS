# Core Concepts

Understanding these core concepts is essential for using the framework effectively.

## Messages

Messages are the fundamental communication unit:

```python
from src.models.message import Message

message = Message(
    role="user",
    content="Hello, agent!",
    name="user_1"
)
```

## Memory Management

Agents maintain conversation history:

```python
# Access agent memory
history = agent.memory.retrieve_all()

# Search memory
results = agent.memory.search("previous task")
```

## Request Context

Tracks the lifecycle of requests:

```python
from src.utils.types import RequestContext

context = RequestContext(
    request_id="req_123",
    agent_name="my_agent",
    depth=0
)
```

## Tools and Functions

Extend agent capabilities:

```python
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

agent = Agent(
    name="calculator",
    tools={"calculate": calculate}
)
```

## Next Steps

- Deep dive into [Agents](agents.md)
- Understand [Memory](memory.md)
- Learn about [Communication](communication.md)

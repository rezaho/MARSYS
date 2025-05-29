# API Reference

Comprehensive API documentation for the Multi-Agent Reasoning Systems (MARSYS) Framework.

## Overview

The framework provides a clean, modular API organized into several key components:

- **Agents** - Base classes and implementations for various agent types
- **Models** - Abstraction layer for different AI model providers
- **Memory** - Memory management and message handling
- **Tools** - Function tools that agents can use
- **Registry** - Agent discovery and communication
- **Utils** - Monitoring, configuration, and utilities

## Core Components

### Agents Module

The agents module provides the foundation for all agent implementations.

#### BaseAgent

```python
from src.agents.base_agent import BaseAgent

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        name: str,
        register: bool = True,
        tools: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            name: Unique agent identifier
            register: Whether to register with AgentRegistry
            tools: Dictionary of tool functions
        """
```

**Key Methods:**
- `register()` - Register agent with the registry
- `unregister()` - Remove agent from registry
- `_log_progress()` - Log agent activity

#### Agent

```python
from src.agents.agent import Agent

class Agent(BaseAgent):
    """Standard agent implementation with memory and model integration."""
    
    def __init__(
        self,
        name: str,
        model_config: ModelConfig,
        instructions: str = "",
        tools: Optional[Dict[str, Callable]] = None,
        register: bool = True
    ):
        """
        Initialize agent with model and memory.
        
        Args:
            name: Unique agent identifier
            model_config: Model configuration
            instructions: System instructions for the agent
            tools: Available tools
            register: Auto-register with registry
        """
```

**Key Methods:**
- `auto_run(task: str, max_steps: int) -> Message` - Execute task autonomously
- `invoke_agent(agent_name: str, task: str) -> Message` - Call another agent
- `_run(messages: List[Message], context: RequestContext) -> Message` - Core execution

#### BrowserAgent

```python
from src.agents.browser_agent import BrowserAgent

class BrowserAgent(Agent):
    """Agent with browser automation capabilities."""
    
    def __init__(
        self,
        name: str,
        model_config: ModelConfig,
        headless: bool = True,
        **kwargs
    ):
        """
        Initialize browser agent.
        
        Args:
            name: Agent identifier
            model_config: Model configuration
            headless: Run browser in headless mode
            **kwargs: Additional agent parameters
        """
```

**Browser-Specific Methods:**
- `navigate_to(url: str)` - Navigate to URL
- `click(selector: str)` - Click element
- `fill(selector: str, value: str)` - Fill form field
- `get_text(selector: str)` - Extract text content
- `screenshot(path: str)` - Take screenshot

#### LearnableAgent

```python
from src.agents.learnable_agent import LearnableAgent

class LearnableAgent(Agent):
    """Agent capable of learning from feedback."""
    
    def __init__(
        self,
        name: str,
        model_config: ModelConfig,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize learning agent.
        
        Args:
            name: Agent identifier
            model_config: Model configuration
            learning_rate: Learning rate for updates
            **kwargs: Additional parameters
        """
```

**Learning Methods:**
- `learn_from_feedback(task: str, response: str, feedback: float)` - Process feedback
- `get_performance_metrics() -> Dict` - Get learning statistics

### Models Module

#### ModelConfig

```python
from src.utils.config import ModelConfig

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    
    provider: str  # "openai", "anthropic", "google"
    model_name: str  # e.g., "gpt-4", "claude-3"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: Optional[str] = None
    timeout: int = 120
```

#### BaseLLM

```python
from src.models.base_models import BaseLLM

class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    @abstractmethod
    async def run(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        output_json: bool = False
    ) -> Dict[str, Any]:
        """Run the model with messages."""
```

### Memory Module

#### Message

```python
from src.models.message import Message

@dataclass
class Message:
    """OpenAI-compatible message format."""
    
    role: str  # "system", "user", "assistant", "tool", "error"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_llm_dict(self) -> Dict[str, Any]:
        """Convert to LLM-compatible format."""
    
    @classmethod
    def from_response_dict(
        cls,
        response_dict: Dict[str, Any],
        name: Optional[str] = None,
        processor: Optional[Callable] = None
    ) -> "Message":
        """Create Message from LLM response."""
```

#### MemoryManager

```python
from src.models.memory import MemoryManager

class MemoryManager:
    """Manages agent conversation memory."""
    
    def __init__(
        self,
        input_processor: Optional[Callable] = None,
        output_processor: Optional[Callable] = None
    ):
        """
        Initialize memory manager.
        
        Args:
            input_processor: Transform LLM responses to Messages
            output_processor: Transform Messages for LLM
        """
    
    def update_memory(self, message: Message) -> None:
        """Add message to memory."""
    
    def retrieve_all(self) -> List[Message]:
        """Get all messages."""
    
    def to_llm_format(self) -> List[Dict[str, Any]]:
        """Convert messages to LLM format."""
```

### Registry Module

#### AgentRegistry

```python
from src.agents.registry import AgentRegistry

class AgentRegistry:
    """Central registry for agent discovery."""
    
    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """Register an agent."""
    
    @classmethod
    def unregister(cls, agent_name: str) -> None:
        """Remove agent from registry."""
    
    @classmethod
    def get_agent(cls, agent_name: str) -> Optional[BaseAgent]:
        """Retrieve agent by name."""
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
```

### Tools Module

#### Tool Functions

Tools must follow this pattern:

```python
def tool_name(param1: str, param2: int = 10) -> str:
    """
    Tool description for the agent.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Implementation
    return result
```

#### Available Tools

```python
from src.environment.tools import AVAILABLE_TOOLS

# Built-in tools
AVAILABLE_TOOLS = {
    "calculate": calculate_function,
    # Add more tools here
}
```

### Utils Module

#### RequestContext

```python
from src.utils.types import RequestContext

@dataclass
class RequestContext:
    """Context for tracking request lifecycle."""
    
    request_id: str
    agent_name: str
    depth: int = 0
    interaction_count: int = 0
    interaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_context: Optional["RequestContext"] = None
    progress_queue: Optional[asyncio.Queue] = None
```

#### LogLevel

```python
from src.utils.types import LogLevel

class LogLevel(IntEnum):
    """Logging levels for agent operations."""
    
    MINIMAL = 0
    SUMMARY = 1
    DETAILED = 2
    DEBUG = 3
```

## Usage Examples

### Creating an Agent

```python
from src.agents.agent import Agent
from src.utils.config import ModelConfig

# Configure model
config = ModelConfig(
    provider="openai",
    model_name="gpt-4",
    temperature=0.7
)

# Create agent
agent = Agent(
    name="my_assistant",
    model_config=config,
    instructions="You are a helpful assistant.",
    tools={"calculate": calculate_tool}
)

# Run task
response = await agent.auto_run(
    task="Calculate the sum of 15 and 27",
    max_steps=3
)
```

### Agent Communication

```python
# Agent A invokes Agent B
response = await agent_a.invoke_agent(
    agent_name="agent_b",
    task="Analyze this data: ..."
)
```

### Custom Tools

```python
async def fetch_data(url: str) -> str:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Add to agent
agent = Agent(
    name="fetcher",
    tools={"fetch_data": fetch_data},
    ...
)
```

### Memory Access

```python
# Get all messages
messages = agent.memory.retrieve_all()

# Search messages
results = agent.memory.search("keyword")

# Get recent messages
recent = agent.memory.retrieve_last_n(5)
```

## Best Practices

1. **Always use async/await** for I/O operations
2. **Return Message objects** from all agent methods
3. **Register agents** for inter-agent communication
4. **Handle errors gracefully** with try-except blocks
5. **Use appropriate log levels** for monitoring
6. **Follow the framework's invariants** (see Contributing Guide)

## Type Reference

### Common Types

```python
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import asyncio
```

### Framework Types

- `Message` - Standard message format
- `RequestContext` - Request tracking
- `ModelConfig` - Model configuration
- `LogLevel` - Logging levels
- `BaseAgent` - Agent base class

## Error Handling

All errors are returned as Message objects:

```python
try:
    result = await operation()
except Exception as e:
    return Message(
        role="error",
        content=str(e),
        name=self.name
    )
```

## Threading and Async

The framework is built on asyncio:

```python
# For CPU-bound operations
result = await asyncio.to_thread(cpu_intensive_function, data)

# For parallel operations
results = await asyncio.gather(
    agent1.auto_run(task1),
    agent2.auto_run(task2)
)
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GOOGLE_API_KEY` - Google API key
- `LOG_LEVEL` - Logging level (MINIMAL, SUMMARY, DETAILED, DEBUG)

### Model Configuration

```python
config = ModelConfig(
    provider="openai",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000,
    timeout=120
)
```

## Next Steps

- Explore [Examples](../use-cases/ for practical usage
- Read [Contributing Guide](../contributing/guidelines.md) to extend the framework
- Check [FAQ](../project/faq.md) for common questions

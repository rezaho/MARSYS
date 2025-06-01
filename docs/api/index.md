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
from src.agents import BaseAgent

class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        description: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None
    ):
        """
        Initialize base agent.
        
        Args:
            model: The language model instance (local or API)
            description: Agent's role and purpose description
            tools: Dictionary of tool functions
            max_tokens: Default maximum tokens for generation
            agent_name: Optional specific name for registration
            allowed_peers: List of agent names this agent can invoke
        """
```

**Key Methods:**
- `auto_run(initial_request, max_steps=10) -> str` - Execute multi-step autonomous task
- `invoke_agent(target_agent_name, request, request_context) -> Message` - Call another agent
- `_log_progress()` - Log agent activity

#### Agent

```python
from src.agents import Agent

class Agent(BaseAgent):
    """Standard agent implementation with memory and model integration."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        description: str,
        tools: Optional[Dict[str, Callable]] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None
    ):
        """
        Initialize agent with model and memory.
        
        Args:
            model_config: Configuration for the language model
            description: The base description of the agent's role and purpose
            tools: Optional dictionary of tools
            memory_type: Type of memory module to use
            max_tokens: Default maximum tokens for generation (overrides model_config default)
            agent_name: Optional specific name for registration
            allowed_peers: List of agent names this agent can call
        """
```

**Key Methods:**
- `auto_run(task: str, max_steps: int) -> Message` - Execute task autonomously
- `invoke_agent(agent_name: str, task: str) -> Message` - Call another agent
- `_run(messages: List[Message], context: RequestContext) -> Message` - Core execution

#### BrowserAgent

```python
from src.agents import BrowserAgent

class BrowserAgent(Agent):
    """Agent with browser automation capabilities."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        generation_description: Optional[str] = None,
        critic_description: Optional[str] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None
    ):
        """
        Initialize browser agent.
        
        Args:
            model_config: Configuration for the language model
            generation_description: Base description for the agent's generation/thinking mode
            critic_description: Base description for the agent's critic mode
            memory_type: Type of memory to use
            max_tokens: Max tokens for model generation
            agent_name: Optional specific name for registration
            allowed_peers: List of allowed peer agents
        """
```

**Browser-Specific Methods:**
- `navigate_to(url: str)` - Navigate to URL
- `click(selector: str)` - Click element
- `fill(selector: str, value: str)` - Fill form field
- `get_text(selector: str)` - Extract text content
- `screenshot(path: str)` - Take screenshot

**Note:** BrowserAgent is typically created using the `create()` class method which handles browser tool initialization:
```python
browser_agent = await BrowserAgent.create(
    model_config=config,
    temp_dir="./tmp/screenshots",
    headless_browser=True
)
```

#### LearnableAgent

```python
from src.agents import LearnableAgent

class LearnableAgent(BaseLearnableAgent):
    """Agent capable of learning from feedback."""
    
    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM],
        description: str,
        tools: Optional[Dict[str, Callable]] = None,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize learning agent.
        
        Args:
            model: The local language model instance (BaseLLM or BaseVLM)
            description: The base description of the agent's role and purpose
            tools: Optional dictionary of tools
            learning_head: Optional type of learning head ('peft')
            learning_head_config: Optional configuration for the learning head
            max_tokens: Default maximum tokens for model generation
            agent_name: Optional specific name for registration
            allowed_peers: List of agent names this agent can call
            **kwargs: Additional arguments passed to BaseAgent.__init__
        """
```

**Learning Methods:**
- `learn_from_feedback(task: str, response: str, feedback: float)` - Process feedback
- `get_performance_metrics() -> Dict` - Get learning statistics

### Models Module

#### ModelConfig

```python
from src.models.models import ModelConfig

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    
    provider: str  # "openai", "anthropic", "google"
    model_name: str  # e.g., "gpt-4.1-mini, "claude-3"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: Optional[str] = None
    timeout: int = 120
```

#### BaseLLM

```python
from src.models.models import BaseLLM

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
from src.agents.memory import Message

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
from src.agents.memory import MemoryManager

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
from src.agents.utils import RequestContext

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
from src.agents.utils import LogLevel

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
from src.agents import Agent
from src.models.models import ModelConfig

# Configure model
config = ModelConfig(
    provider="openai",
    model_name="gpt-4.1-mini,
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
    initial_request="Calculate the sum of 15 and 27",
    max_steps=3
)
```

### Agent Communication

```python
# Agent A invokes Agent B
response = await agent_a.invoke_agent(
    agent_name="agent_b",
    initial_request="Analyze this data: ..."
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
    model_name="gpt-4.1-mini,
    temperature=0.7,
    max_tokens=2000,
    timeout=120
)
```

## Next Steps

- Explore [Examples](../use-cases/) for practical usage
- Read [Contributing Guide](../contributing/guidelines.md) to extend the framework
- Check [FAQ](../project/faq.md) for common questions

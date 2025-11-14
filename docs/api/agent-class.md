# Agent Class API Reference

Complete API documentation for the Agent classes in MARSYS, including BaseAgent, Agent, BrowserAgent, and AgentPool.

## 📦 BaseAgent

Abstract base class that all agents must inherit from.

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union, Callable
from marsys.models.base import BaseModel
from marsys.agents.memory import ConversationMemory, Message

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        model: Union[BaseVLM, BaseLLM, BaseAPIModel],
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_tokens: int = 512,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        memory_retention: str = "session",
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None
    ):
        """
        Initialize base agent.

        Args:
            model: Language model instance
            goal: 1-2 sentence summary of what the agent accomplishes
            instruction: Detailed instructions on how the agent should behave
            tools: Dictionary of tool functions
            max_tokens: Maximum response tokens
            name: Unique agent identifier
            allowed_peers: List of agents this can invoke
            memory_retention: Memory policy (single_run, session, persistent)
            input_schema: Pydantic schema for input validation
            output_schema: Pydantic schema for output validation
        """
```

### Abstract Methods

```python
@abstractmethod
async def _run(
    self,
    prompt: Any,
    context: Dict[str, Any],
    **kwargs
) -> Message:
    """
    Pure execution logic - must be implemented by subclasses.

    Args:
        prompt: Input prompt or message
        context: Execution context
        **kwargs: Additional parameters

    Returns:
        Message object with response

    Note:
        This method must be pure - no side effects allowed!
    """
    pass
```

### Public Methods

#### `run(prompt, context=None, **kwargs) -> Message`

Public interface for agent execution.

```python
async def run(
    self,
    prompt: Union[str, Message, Dict],
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Message:
    """
    Execute agent with automatic context management.

    Args:
        prompt: Input prompt, message, or structured data
        context: Optional execution context
        **kwargs: Additional parameters

    Returns:
        Message with agent response
    """
```

#### `cleanup() -> None`

Clean up agent resources (model sessions, tools, browser handles, etc.).

```python
async def cleanup(self) -> None:
    """
    Clean up agent resources.

    Called automatically by Orchestra at end of run if auto_cleanup_agents=True.
    Can be overridden by subclasses for custom cleanup logic.

    Default implementation:
    1. Closes model async resources (aiohttp sessions, etc.)
    2. Calls agent-specific close() if available (e.g., BrowserAgent.close())

    Example override:
        async def cleanup(self):
            # Custom cleanup
            await self.custom_resource.close()
            # Call parent cleanup
            await super().cleanup()
    """
```

**Automatic Cleanup:**

The framework automatically calls `cleanup()` on all topology agents after `Orchestra.run()` completes (unless `auto_cleanup_agents=False`).

**Manual Cleanup:**

```python
# Create agent
agent = Agent(agent_name="my_agent", model_config=config)

# Use agent
result = await agent.run("Process data")

# Manual cleanup when done
await agent.cleanup()

# Unregister from registry (identity-safe)
from marsys.agents.registry import AgentRegistry
AgentRegistry.unregister_if_same("my_agent", agent)
```

## 🤖 Agent

Standard agent implementation with built-in capabilities.

### Class Definition

```python
from marsys.agents import Agent
from marsys.models import ModelConfig

class Agent(BaseAgent):
    """Standard agent with full framework capabilities."""

    def __init__(
        self,
        model_config: ModelConfig,
        agent_name: Optional[str] = None,
        description: str = "A helpful AI assistant",
        system_prompt: Optional[str] = None,
        tools: Optional[Union[List[Callable], Dict[str, Callable]]] = None,
        max_tokens: int = 2048,
        allowed_peers: Optional[List[str]] = None,
        memory_retention: str = "session",
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize agent with model configuration.

        Args:
            model_config: ModelConfig instance
            agent_name: Unique identifier
            description: Agent role description
            system_prompt: System instruction override
            tools: Available tool functions
            max_tokens: Maximum response tokens
            allowed_peers: Invokable agent names
            memory_retention: Memory persistence policy
            input_schema: Input validation schema
            output_schema: Output validation schema
            **kwargs: Additional model parameters
        """
```

### Key Methods

#### `auto_run(initial_prompt, context=None, max_steps=10, **kwargs)`

Execute agent autonomously with tool and agent invocation.

```python
async def auto_run(
    self,
    initial_prompt: str,
    context: Optional[RequestContext] = None,
    max_steps: int = 10,
    max_re_prompts: int = 3,
    run_async: bool = False,
    **kwargs
) -> Union[Message, str]:
    """
    Run agent autonomously with automatic tool/agent invocation.

    Args:
        initial_prompt: Starting prompt
        context: Request context for tracking
        max_steps: Maximum execution steps
        max_re_prompts: Maximum reprompt attempts
        run_async: Enable async execution
        **kwargs: Additional parameters

    Returns:
        Final response as Message or string

    Example:
        response = await agent.auto_run(
            "Research and summarize recent AI breakthroughs",
            max_steps=5
        )
    """
```

#### `invoke_agent(agent_name, task, context=None)`

Invoke another agent for task delegation.

```python
async def invoke_agent(
    self,
    agent_name: str,
    task: str,
    context: Optional[RequestContext] = None
) -> Message:
    """
    Invoke another agent to perform a task.

    Args:
        agent_name: Name of agent to invoke
        task: Task description
        context: Request context

    Returns:
        Response from invoked agent

    Raises:
        AgentPermissionError: If not allowed to invoke agent
        AgentNotFoundError: If agent doesn't exist

    Example:
        result = await coordinator.invoke_agent(
            "researcher",
            "Find information about quantum computing"
        )
    """
```

### Usage Examples

```python
from marsys.agents import Agent
from marsys.models import ModelConfig

# Basic agent
assistant = Agent(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-haiku-4.5",
        temperature=0.7,
        max_tokens=12000
    ),
    name="assistant",
    goal="Provide helpful assistance to users",
    instruction="A helpful AI assistant that responds thoughtfully to queries"
)

# Agent with tools
def calculate(expression: str) -> float:
    """Evaluate mathematical expression."""
    return eval(expression)

calculator = Agent(
    model_config=config,
    name="calculator",
    goal="Perform mathematical calculations accurately",
    instruction="Mathematical calculation specialist who uses tools for precise computations",
    tools=[calculate],
    system_prompt="You are a precise calculator. Always use the calculate tool for math."
)

# Multi-agent coordinator
coordinator = Agent(
    model_config=config,
    name="coordinator",
    goal="Coordinate tasks between specialized agents",
    instruction="I coordinate tasks between specialized agents and synthesize results",
    allowed_peers=["researcher", "writer", "calculator"]
)
```

## 🌐 BrowserAgent

Specialized agent for web automation and scraping.

### Class Definition

```python
from marsys.agents import BrowserAgent
from marsys.environment.browser_tool import BrowserTool

class BrowserAgent(Agent):
    """Agent with browser automation capabilities."""

    def __init__(
        self,
        model_config: ModelConfig,
        agent_name: Optional[str] = None,
        description: str = "Web automation specialist",
        headless_browser: bool = True,
        temp_dir: str = "./tmp/screenshots",
        viewport_size: Tuple[int, int] = (1920, 1080),
        **kwargs
    ):
        """
        Initialize browser agent.

        Args:
            model_config: Model configuration
            agent_name: Unique identifier
            description: Agent role
            headless_browser: Run without UI
            temp_dir: Screenshot directory
            viewport_size: Browser viewport dimensions
            **kwargs: Additional parameters
        """
```

### Creation Methods

```python
@classmethod
async def create_safe(
    cls,
    model_config: ModelConfig,
    agent_name: Optional[str] = None,
    headless_browser: bool = True,
    **kwargs
) -> "BrowserAgent":
    """
    Safely create browser agent with initialization.

    Returns:
        Initialized BrowserAgent instance

    Example:
        browser = await BrowserAgent.create_safe(
            model_config=config,
            name="web_scraper",
            headless_browser=True
        )
    """
```

### Browser-Specific Methods

```python
# Navigation
await browser.browser_tool.navigate_to_url("https://example.com")
await browser.browser_tool.go_back_in_history()
await browser.browser_tool.refresh_page()

# Interaction
await browser.browser_tool.click_element("#submit-button")
await browser.browser_tool.fill_form_field("input[name='email']", "test@example.com")
await browser.browser_tool.select_option_in_dropdown("select#country", "US")

# Extraction
text = await browser.browser_tool.get_text_content("h1.title")
data = await browser.browser_tool.extract_multiple_elements_data(
    main_selector="div.product",
    properties={
        "name": "h2.name",
        "price": "span.price"
    }
)

# Screenshots
await browser.browser_tool.take_screenshot_of_page("screenshot.png", full_page=True)

# Cleanup
await browser.browser_tool.close_browser()
```

### Usage Example

```python
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="scraper",
    headless_browser=True
)

try:
    # Use browser agent
    result = await browser_agent.auto_run(
        "Go to example.com and extract all product names and prices",
        max_steps=3
    )
finally:
    # Always cleanup
    await browser_agent.browser_tool.close_browser()
```

## 🏊 AgentPool

Manages multiple agent instances for parallel execution.

### Class Definition

```python
from marsys.agents import AgentPool
from typing import Type, Optional, Any
import asyncio

class AgentPool:
    """Pool of agent instances for parallel execution."""

    def __init__(
        self,
        agent_class: Type[BaseAgent],
        num_instances: int,
        model_config: ModelConfig,
        agent_name: str,
        **agent_kwargs
    ):
        """
        Initialize agent pool.

        Args:
            agent_class: Agent class to instantiate
            num_instances: Number of pool instances
            model_config: Model configuration
            agent_name: Base name for agents
            **agent_kwargs: Additional agent parameters
        """
```

### Key Methods

#### `acquire(branch_id=None)`

Acquire agent instance from pool.

```python
async def acquire(
    self,
    branch_id: Optional[str] = None,
    timeout: float = 30.0
) -> AsyncContextManager[BaseAgent]:
    """
    Acquire agent from pool with context manager.

    Args:
        branch_id: Optional branch identifier
        timeout: Acquisition timeout in seconds

    Returns:
        Context manager yielding agent instance

    Example:
        async with pool.acquire("branch_1") as agent:
            result = await agent.run("Task description")
    """
```

#### `get_statistics()`

Get pool usage statistics.

```python
def get_statistics(self) -> Dict[str, Any]:
    """
    Get pool statistics.

    Returns:
        Dictionary with:
        - total_instances: Pool size
        - available: Available instances
        - in_use: Currently allocated
        - total_allocations: Historical count
        - average_wait_time: Avg acquisition wait
    """
```

### Usage Example

```python
from marsys.agents import AgentPool, BrowserAgent

# Create pool of browser agents
browser_pool = AgentPool(
    agent_class=BrowserAgent,
    num_instances=3,
    model_config=config,
    name="BrowserPool",
    headless_browser=True
)

# Parallel scraping
async def scrape_url(url: str, branch_id: str):
    async with browser_pool.acquire(branch_id) as agent:
        return await agent.run(f"Scrape content from {url}")

# Execute in parallel
urls = ["http://site1.com", "http://site2.com", "http://site3.com"]
tasks = [
    scrape_url(url, f"branch_{i}")
    for i, url in enumerate(urls)
]
results = await asyncio.gather(*tasks)

# Cleanup pool
await browser_pool.cleanup()

# Check statistics
stats = browser_pool.get_statistics()
print(f"Total allocations: {stats['total_allocations']}")
```

## 🧩 Custom Agents

Create specialized agents by subclassing BaseAgent.

### Example Custom Agent

```python
from marsys.agents import BaseAgent
from marsys.agents.memory import Message
from typing import Dict, Any

class AnalysisAgent(BaseAgent):
    """Custom agent for data analysis."""

    def __init__(self, model, **kwargs):
        super().__init__(
            model=model,
            goal="Analyze data and provide insights",
            instruction="Data analysis specialist with multiple analysis methods",
            **kwargs
        )
        self.analysis_methods = ["statistical", "trend", "anomaly"]

    async def _run(
        self,
        prompt: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> Message:
        """
        Pure execution logic for analysis.

        Note: Must be pure - no side effects!
        """
        # Prepare messages
        messages = self._prepare_messages(prompt)

        # Add analysis context
        analysis_type = context.get("analysis_type", "statistical")
        if analysis_type in self.analysis_methods:
            messages.append({
                "role": "system",
                "content": f"Perform {analysis_type} analysis"
            })

        # Execute model
        response = await self.model.run(messages)

        # Return pure Message
        return Message(
            role="assistant",
            content=response["content"],
            metadata={"analysis_type": analysis_type}
        )

    async def analyze_dataset(
        self,
        data: List[float],
        method: str = "statistical"
    ) -> Dict[str, Any]:
        """
        High-level analysis method.
        """
        prompt = f"Analyze this dataset: {data}"
        context = {"analysis_type": method}

        result = await self.run(prompt, context)

        # Parse and structure results
        return {
            "method": method,
            "results": result.content,
            "timestamp": result.timestamp
        }
```

## 📊 Agent Configuration

### Memory Retention Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `single_run` | Clear after each run | Stateless operations |
| `session` | Maintain for workflow | Multi-step tasks |
| `persistent` | Save to disk | Long-term memory |

### Tool Registration

```python
# Function with proper docstring
def search(query: str, limit: int = 5) -> List[str]:
    """
    Search for information.

    Args:
        query: Search query
        limit: Maximum results

    Returns:
        List of results
    """
    return ["result1", "result2"]

# Auto-schema generation
agent = Agent(
    model_config=config,
    tools=[search]  # Schema auto-generated
)

# Or with custom names
agent = Agent(
    model_config=config,
    tools={"web_search": search}
)
```

## 🛡️ Error Handling

```python
from marsys.agents.exceptions import (
    AgentError,
    AgentNotFoundError,
    AgentPermissionError,
    AgentExecutionError
)

try:
    result = await agent.run(prompt)

except AgentPermissionError as e:
    print(f"Permission denied: {e.target_agent}")

except AgentExecutionError as e:
    print(f"Execution failed: {e.message}")

except AgentError as e:
    print(f"Agent error: {e}")
```

## 🔗 Related Documentation

- [Concepts: Agents](../concepts/agents.md) - Agent architecture
- [Models API](models.md) - Model configuration
- [Memory](../concepts/memory.md) - Memory management
- [Tools](../concepts/tools.md) - Tool integration
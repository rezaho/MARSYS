# Agent Class API Reference

Complete API documentation for the Agent classes in MARSYS, including BaseAgent, Agent, BrowserAgent, CodeExecutionAgent, DataAnalysisAgent, and AgentPool.

!!! tip "Architecture & Patterns"
    For architectural overview, design patterns, and best practices, see [Agents Concept Guide](../concepts/agents.md).

## 📦 BaseAgent

Abstract base class that all agents must inherit from.

### Class Definition

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Union, Callable
from marsys.agents.memory import ConversationMemory, Message
from marsys.agents.planning import PlanningConfig

class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        model: Union[BaseLocalModel, BaseAPIModel],
        name: str,
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable]] = None,
        max_tokens: Optional[int] = 10000,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,
        is_convergence_point: Optional[bool] = None,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        plan_config: Optional[Union[PlanningConfig, Dict, bool]] = None
    ):
        """
        Initialize base agent.

        Args:
            model: Language model instance (BaseLocalModel or BaseAPIModel)
            goal: 1-2 sentence summary of what the agent accomplishes
            instruction: Detailed instructions on how the agent should behave
            tools: Dictionary of tool functions
            max_tokens: Maximum response tokens
            name: Unique agent identifier
            allowed_peers: List of agents this can invoke
            memory_retention: Memory policy (single_run, session, persistent)
            input_schema: Pydantic schema for input validation
            output_schema: Pydantic schema for output validation
            plan_config: Planning configuration. PlanningConfig instance, dict of
                config values, True/None (enabled with defaults), or False (disabled).
                Planning is enabled by default. See [Task Planning](../concepts/planning.md).
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
agent = Agent(
    name="my_agent",
    model_config=config,
    goal="Process data efficiently",
    instruction="You are a data processing agent. Handle data operations as requested."
)

# Use agent
result = await agent.run("Process data")

# Manual cleanup when done
await agent.cleanup()

# Unregister from registry (identity-safe)
from marsys.agents.registry import AgentRegistry
AgentRegistry.unregister_if_same("my_agent", agent)
```

#### `save_state(filepath: str) -> None`

Save agent state to disk, including memory and planning state.

```python
agent.save_state("./state/agent_state.json")
```

What gets saved:
- Memory contents
- Planning state (current plan + planning config)
- Tool schema version for cache invalidation

#### `load_state(filepath: str) -> None`

Load agent state from disk. This restores memory and planning state.

```python
agent.load_state("./state/agent_state.json")
```

Notes:
- Planning state temporarily unsubscribes from `MemoryResetEvent` during load to avoid race conditions.
- If planning is disabled (`plan_config=False`), only memory is restored.

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
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        memory_type: Optional[str] = "managed_conversation",
        memory_config: Optional[ManagedMemoryConfig] = None,
        compaction_model_config: Optional[ModelConfig] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        plan_config: Optional[Union[PlanningConfig, Dict, bool]] = None
    ):
        """
        Initialize agent with model configuration.

        Args:
            model_config: ModelConfig instance
            goal: 1-2 sentence summary of what this agent accomplishes
            instruction: Detailed instructions on how the agent should behave
            tools: Dictionary of tool functions
            memory_type: Type of memory module to use (default: managed_conversation)
            memory_config: Optional configuration for managed memory behavior
            compaction_model_config: Optional ModelConfig for a separate compaction model
            max_tokens: Maximum response tokens (None uses ModelConfig default)
            name: Unique identifier for registration
            allowed_peers: List of agent names this agent can invoke
            bidirectional_peers: If True, creates bidirectional edges with allowed_peers
            input_schema: Input validation schema
            output_schema: Output validation schema
            memory_retention: Memory persistence policy (single_run, session, persistent)
            memory_storage_path: Path for persistent memory storage
            plan_config: Planning configuration. PlanningConfig instance, dict of
                config values, True/None (enabled with defaults), or False (disabled).
                Planning is enabled by default. See [Task Planning](../concepts/planning.md).
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

### Usage Examples

```python
from marsys.agents import Agent
from marsys.models import ModelConfig

# Basic agent
assistant = Agent(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",
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
    goal="Perform mathematical calculations accurately",
    instruction="You are a precise calculator. Always use the calculate tool for math.",
    tools={"calculate": calculate},
    name="calculator"
)

# Multi-agent coordinator
coordinator = Agent(
    model_config=config,
    name="coordinator",
    goal="Coordinate tasks between specialized agents",
    instruction="I coordinate tasks between specialized agents and synthesize results",
    allowed_peers=["researcher", "writer", "calculator"]
)

# Agent with separate compaction model
agent_with_compaction_model = Agent(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",
    ),
    compaction_model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-haiku-4.5",
    ),
    goal="Long-running research and synthesis",
    instruction="You are a research assistant."
)
```

## 🌐 BrowserAgent

Specialized agent for web automation and scraping.

### Class Definition

```python
from marsys.agents import BrowserAgent
from marsys.models import ModelConfig

class BrowserAgent(Agent):
    """Agent with browser automation capabilities."""

    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        mode: str = "advanced",
        headless: bool = True,
        viewport_width: Optional[int] = None,
        viewport_height: Optional[int] = None,
        tmp_dir: Optional[str] = None,
        vision_model_config: Optional[ModelConfig] = None,
        auto_screenshot: bool = False,
        element_detection_mode: str = "auto",
        timeout: int = 5000,
        show_mouse_helper: bool = True,
        session_path: Optional[str] = None,
        filesystem: Optional["RunFileSystem"] = None,
        downloads_subdir: str = "downloads",
        downloads_virtual_dir: str = "./downloads",
        fetch_file_tool_name: str = "download_file",
        plan_config: Optional[Union[PlanningConfig, Dict, bool]] = None,
        **kwargs
    ):
        """
        Initialize browser agent.

        Args:
            model_config: Model configuration
            name: Unique identifier
            goal: Agent's purpose (defaults to web automation)
            instruction: Detailed behavior instructions
            mode: Browser mode ("primitive" for extraction, "advanced" for visual interaction)
            headless: Run without UI
            viewport_width: Optional viewport width (auto-detected if omitted)
            viewport_height: Optional viewport height (auto-detected if omitted)
            tmp_dir: Run root directory for downloads/screenshots/artifacts
            vision_model_config: Optional dedicated vision model config
            auto_screenshot: Auto-screenshot after actions (advanced mode only)
            element_detection_mode: "auto", "rule_based", "vision", or "both"
            timeout: Browser operation timeout in milliseconds
            show_mouse_helper: Visual cursor helper in advanced mode
            session_path: Optional storage-state JSON for restoring browser session
            filesystem: Optional shared RunFileSystem for unified path resolution
            downloads_subdir: Host downloads subdirectory under tmp_dir
            downloads_virtual_dir: Virtual path returned to the agent for downloads
            fetch_file_tool_name: Tool name alias for file-download action
            plan_config: Planning configuration. PlanningConfig instance, dict of
                config values, True/None (enabled with defaults), or False (disabled).
                Planning is enabled by default. See [Task Planning](../concepts/planning.md).
            **kwargs: Additional parameters
        """
```

### Creation Methods

```python
@classmethod
async def create_safe(
    cls,
    model_config: ModelConfig,
    name: str,
    goal: Optional[str] = None,
    instruction: Optional[str] = None,
    mode: str = "advanced",
    headless: bool = True,
    viewport_width: Optional[int] = None,
    viewport_height: Optional[int] = None,
    tmp_dir: Optional[str] = None,
    vision_model_config: Optional[ModelConfig] = None,
    auto_screenshot: bool = False,
    element_detection_mode: str = "auto",
    timeout: int = 5000,
    session_path: Optional[str] = None,
    filesystem: Optional["RunFileSystem"] = None,
    show_mouse_helper: bool = True,
    downloads_subdir: str = "downloads",
    downloads_virtual_dir: str = "./downloads",
    fetch_file_tool_name: str = "download_file",
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
            headless=True
        )
    """
```

Artifact path notes:
- `tmp_dir` acts as the run root for browser artifacts.
- `downloads_subdir` controls host folder placement under `tmp_dir`.
- `downloads_virtual_dir` controls the virtual path shown to agents.
- `fetch_file_tool_name` lets you expose `download_file` under a custom tool name.

Viewport auto-detection (when width/height are omitted):
- Google/Gemini models: `1000x1000`
- Anthropic/Claude models: `1344x896`
- OpenAI/GPT models: `1024x768`
- Fallback: `1536x1536`

### Browser-Specific Methods

```python
# Navigation
await browser.browser_tool.goto("https://example.com")
await browser.browser_tool.go_back()
await browser.browser_tool.reload()

# Interaction
await browser.browser_tool.mouse_click(450, 300)
await browser.browser_tool.keyboard_input("search term")
await browser.browser_tool.keyboard_press("Enter")

# Extraction
metadata = await browser.browser_tool.get_page_metadata()
elements = await browser.browser_tool.get_page_elements()

# Downloads and screenshots
downloads = await browser.browser_tool.list_downloads()
shot = await browser.browser_tool.screenshot()

# Cleanup
await browser.cleanup()
```

### Usage Example

```python
browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="scraper",
    mode="primitive",
    headless=True,
    tmp_dir="./runs/run-20260206",
    downloads_virtual_dir="./downloads",
    fetch_file_tool_name="fetch_file",
)

try:
    result = await browser_agent.auto_run("Go to example.com and summarize key content", max_steps=3)
finally:
    await browser_agent.cleanup()
```

## 💻 CodeExecutionAgent

Specialized agent that combines file operations with controlled Python/shell execution.

### Class Definition

```python
from marsys.agents import CodeExecutionAgent
from marsys.environment.code import CodeExecutionConfig

class CodeExecutionAgent(Agent):
    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        working_directory: Optional[str] = None,  # Deprecated
        base_directory: Optional[Path] = None,
        code_config: Optional[CodeExecutionConfig] = None,
        filesystem: Optional["RunFileSystem"] = None,
        **kwargs
    ):
        ...
```

Behavior notes:
- Combines `FileOperationTools` with `CodeExecutionTools`.
- Exposes `python_execute` and `shell_execute`.
- Uses shared `RunFileSystem` when `filesystem` is provided.

### Usage Example

```python
agent = CodeExecutionAgent(
    model_config=config,
    name="CodeRunner",
    code_config=CodeExecutionConfig(
        timeout_default=30,
        max_memory_mb=1024,
        allow_network=False,
    ),
)

result = await agent.auto_run("Run unit tests and summarize failures", max_steps=6)
await agent.cleanup()
```

## 📊 DataAnalysisAgent

Specialized agent for iterative data-science style workflows.

### Class Definition

```python
from marsys.agents import DataAnalysisAgent
from marsys.environment.code import CodeExecutionConfig

class DataAnalysisAgent(Agent):
    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        working_directory: Optional[str] = None,  # Deprecated
        base_directory: Optional[Path] = None,
        code_config: Optional[CodeExecutionConfig] = None,
        filesystem: Optional["RunFileSystem"] = None,
        **kwargs
    ):
        ...
```

Behavior notes:
- Forces persistent Python session (`session_persistent_python=True`).
- Keeps notebook-like state across `python_execute` calls.
- Includes file and shell tools alongside persistent Python execution.

### Usage Example

```python
agent = DataAnalysisAgent(
    model_config=config,
    name="Analyst",
)

result = await agent.auto_run(
    "Load sales.csv, find trends, and save a chart to ./outputs",
    max_steps=8
)
await agent.cleanup()
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
        agent_class: Type,
        num_instances: int,
        *args,
        **kwargs
    ):
        """
        Initialize agent pool.

        Args:
            agent_class: Agent class to instantiate
            num_instances: Number of pool instances
            *args: Positional constructor arguments passed to each agent instance
            **kwargs: Keyword constructor arguments passed to each agent instance
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
    headless=True
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
            content=response["content"]
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
            "results": result.content
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
    goal="Search for relevant information",
    instruction="Use search for retrieval requests.",
    tools={"search": search}  # Schema auto-generated
)

# Or with custom names
agent = Agent(
    model_config=config,
    goal="Search the web for evidence",
    instruction="Use web_search for web retrieval requests.",
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

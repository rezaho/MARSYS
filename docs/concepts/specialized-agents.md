# Specialized Agents

MARSYS provides specialized agents that extend the base `Agent` class with domain-specific tools and instructions. These agents are production-ready and optimized for common tasks.

## Overview

Specialized agents combine:

- **Domain-specific tools**: Curated toolsets for specific tasks
- **Scenario-based instructions**: Adaptive guidance rather than rigid workflows
- **Configurable capabilities**: Enable/disable features based on requirements
- **Security features**: Built-in validation and safety mechanisms

## Available Specialized Agents

### [BrowserAgent](browser-automation.md)

Autonomous browser automation with vision-based interaction and screenshot analysis.

**Best for**: Web scraping, UI testing, form filling, web research, dynamic content extraction

**Key Features**:
- Vision-based element interaction (no selectors needed)
- Multi-mode operation (primitive, advanced)
- Screenshot analysis with multimodal models
- JavaScript execution and console monitoring
- Auto-screenshot management with sliding window
- **Session persistence**: Save and load browser sessions (cookies, localStorage) for persistent authentication
- **Tab management**: List, switch, and close browser tabs programmatically

**Tools**: Browser navigation, interaction, JavaScript execution, screenshot analysis, session management, tab management

```python
from marsys.agents import BrowserAgent

# BrowserAgent requires async creation via create_safe()
agent = await BrowserAgent.create_safe(
    model_config=config,
    name="WebAutomation",
    mode="advanced",  # "primitive" or "advanced"
    headless=True,
    session_path="./sessions/my_session.json"  # Optional: load existing session
)
```

[**Read Full Documentation →**](browser-automation.md)

---

### [FileOperationAgent](../guides/file-operations.md)

Intelligent file and directory operations with optional shell command execution.

**Best for**: Code analysis, configuration management, log processing, documentation generation

**Key Features**:
- Type-aware file handling (Python, JSON, PDF, Markdown, images)
- Unified diff editing with high success rate
- Content and structure search (ripgrep-based)
- Optional shell tools for complex operations
- Security: Command validation, blocked dangerous patterns, timeouts

**Tools**: 6 file operation tools + 10 optional shell tools

```python
from marsys.agents import FileOperationAgent

agent = FileOperationAgent(
    model_config=config,
    name="FileHelper",
    enable_shell=True,  # Enable shell commands
    allowed_shell_commands=["grep", "find", "wc"]  # Whitelist
)
```

To share files across agents, pass a shared `RunFileSystem` via `filesystem=...`.
See [Run Filesystem](run-filesystem.md).

[**Read Full Documentation →**](../guides/file-operations.md)

---

### CodeExecutionAgent

Code-first specialist that combines file operations with safe Python and shell execution.

**Best for**: Running tests, scripts, build/lint workflows, automation tasks, reproducible code-driven debugging.

**Key Features**:
- Unified file + execution toolset (`read_file`/`edit_file` + `python_execute`/`shell_execute`)
- Shared `RunFileSystem` support for consistent virtual paths across agents
- Configurable `CodeExecutionConfig` for resource and security limits
- Explicit cleanup support for persistent execution resources

**Tools**: file operation tools + `python_execute` + `shell_execute`

```python
from marsys.agents import CodeExecutionAgent

agent = CodeExecutionAgent(
    model_config=config,
    name="CodeRunner",
)
```

---

### DataAnalysisAgent

Data-science oriented agent with persistent Python session behavior (Jupyter-like workflow).

**Best for**: Iterative analysis, data exploration, statistical modeling, plotting, and experiment-style workflows.

**Key Features**:
- Persistent `python_execute` session by default (`session_persistent_python=True`)
- File operations + execution tools in one agent
- Shared `RunFileSystem` support for multi-agent handoffs
- Designed for incremental analysis loops instead of one-shot commands

**Tools**: file operation tools + persistent `python_execute` + `shell_execute`

```python
from marsys.agents import DataAnalysisAgent

agent = DataAnalysisAgent(
    model_config=config,
    name="DataScientist",
)
```

---

### [WebSearchAgent](web-search-agent.md)

Multi-source information gathering across web and scholarly databases.

**Best for**: Research, fact-checking, literature reviews, current events

**Key Features**:
- Multi-source search (DuckDuckGo, Google, arXiv, Semantic Scholar, PubMed)
- Configurable search modes (web, scholarly, or all)
- API key validation at initialization
- Query formulation strategies
- Iterative refinement guidance

**Tools**: Up to 8 search sources (configurable)

```python
from marsys.agents import WebSearchAgent

agent = WebSearchAgent(
    model_config=config,
    name="Researcher",
    search_mode="all",  # "web", "scholarly", or "all"
    include_google=False  # Avoid Google API key requirement
)
```

[**Read Full Documentation →**](web-search-agent.md)

---

## Comparison

| Agent | Primary Use Case | Tools | API Keys Required | Security Features |
|-------|------------------|-------|-------------------|-------------------|
| **BrowserAgent** | Web automation | Browser control | None | Timeout enforcement, mode-based restrictions |
| **FileOperationAgent** | File system operations | 6-16 tools | None | Command validation, blocked patterns |
| **CodeExecutionAgent** | Code + automation tasks | File ops + Python + shell | None | Resource limits, blocked patterns, optional network disable |
| **DataAnalysisAgent** | Iterative data science workflows | Persistent Python + file ops + shell | None | Resource limits, blocked patterns, persistent session controls |
| **WebSearchAgent** | Information gathering | 1-8 sources | Google (optional for Google tools)<br>None (DuckDuckGo/scholarly basics) | API key validation |

## Common Patterns

### Pattern 1: Multi-Agent Workflow

Combine specialized agents in a topology:

```python
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig
from marsys.agents import BrowserAgent, FileOperationAgent, DataAnalysisAgent, WebSearchAgent

browser_agent = await BrowserAgent.create_safe(
    model_config=config,
    name="BrowserAgent",
    mode="primitive"
)
file_agent = FileOperationAgent(model_config=config, name="FileHelper", enable_shell=True)
analysis_agent = DataAnalysisAgent(model_config=config, name="Analyzer")
search_agent = WebSearchAgent(model_config=config, name="Researcher", search_mode="scholarly")

topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["BrowserAgent", "FileHelper", "Researcher"]
)

result = await Orchestra.run(
    task="Research topic, scrape related websites, and analyze findings",
    topology=topology
)
```

### Pattern 2: Sequential Pipeline

Chain specialized agents for complex tasks:

```python
topology = PatternConfig.pipeline(
    stages=[
        {"name": "search", "agents": ["Researcher"]},
        {"name": "scrape", "agents": ["BrowserAgent"]},
        {"name": "analyze", "agents": ["FileHelper"]},
        {"name": "report", "agents": ["ReportWriter"]}
    ]
)
```

### Pattern 3: Parallel Data Gathering

Use agent pools for concurrent operations:

```python
from marsys.agents import create_browser_agent_pool

# Create pool of 3 browser instances
pool = await create_browser_agent_pool(
    num_instances=3,
    model_config=config,
    mode="primitive",
    headless=True
)

# Pool is registered by default (register=True)
# and automatically manages concurrent requests
```

### Pattern 4: Conditional Tool Enabling

Enable features based on environment:

```python
# Production: restricted shell commands
file_agent = FileOperationAgent(
    model_config=config,
    name="FileHelper",
    enable_shell=True,
    allowed_shell_commands=["grep", "find", "wc", "ls"]
)

# Development: more permissive
file_agent = FileOperationAgent(
    model_config=config,
    name="FileHelperDev",
    enable_shell=True  # Uses default blocked list only
)
```

## Creating Custom Specialized Agents

To create your own specialized agent:

1. **Extend Agent class**:
   ```python
   from marsys.agents import Agent

   class MySpecializedAgent(Agent):
       def __init__(self, model_config, **kwargs):
           # Initialize tools
           tools = self._build_tools()

           # Build instruction
           instruction = self._build_instruction()

           super().__init__(
               model_config=model_config,
               goal="Your goal here",
               instruction=instruction,
               tools=tools,
               **kwargs
           )
   ```

2. **Create domain-specific tools**: Use classes like `FileOperationTools` or `SearchTools` as templates

3. **Write scenario-based instructions**: Guide the agent on **how to choose** rather than prescribing steps

4. **Add validation**: Validate configuration at initialization (API keys, paths, etc.)

See [Custom Agents Guide](custom-agents.md) for detailed instructions.

## Tool Integration

Specialized agents use tool classes that provide:

- **Structured output**: Consistent Dict/JSON responses
- **Error handling**: Comprehensive error classification
- **Configuration**: Environment variables or explicit parameters
- **Security**: Validation, timeouts, output size limits

Available tool classes:

- [BrowserAgent Tools](browser-automation.md): Browser automation and control
- [FileOperationTools](../guides/file-operations.md): File system operations
- [ShellTools](../guides/specialized-tools.md): Shell command execution
- [SearchTools](../guides/specialized-tools.md): Multi-source search

## Best Practices

### 1. Choose the Right Agent

**Use BrowserAgent when**:
- Interacting with dynamic web content
- Need to fill forms or click buttons
- Scraping JavaScript-rendered pages
- Visual verification of web elements

**Use FileOperationAgent when**:
- Working with files on the local file system
- Need shell commands for system integration
- Analyzing codebases or processing logs

**Use CodeExecutionAgent when**:
- You need to execute Python/shell while editing files in the same workflow
- Build/test/lint/automation loops are central to the task
- You want strict execution controls via `CodeExecutionConfig`

**Use DataAnalysisAgent when**:
- You need notebook-like iterative analysis with persistent Python state
- You are exploring datasets across multiple execution steps
- You need plotting/analysis artifacts while keeping shared virtual paths

**Use WebSearchAgent when**:
- Gathering information from online sources
- Conducting research or fact-checking
- Need both current web content and academic papers

### 2. Configure Security Appropriately

```python
# BrowserAgent: production mode
agent = await BrowserAgent.create_safe(
    model_config=config,
    name="WebAutomation",
    mode="advanced",
    headless=True,
    auto_screenshot=False,
    timeout=8000
)

# FileOperationAgent: strict whitelist
agent = FileOperationAgent(
    model_config=config,
    name="FileHelper",
    enable_shell=True,
    allowed_shell_commands=["grep", "find", "wc"]
)
```

### 3. Handle Missing API Keys Gracefully

```python
try:
    search_agent = WebSearchAgent(
        model_config=config,
        name="Researcher",
        search_mode="web",
        include_google=True,
        google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID_GENERIC"),
    )
except ValueError as e:
    print(f"Missing API key: {e}")
    # Fall back to scholarly-only mode
    search_agent = WebSearchAgent(model_config=config, name="Researcher", search_mode="scholarly")
```

### 4. Use Scenario-Based Instructions

Specialized agents use scenario-based instructions that adapt to context:

```
**When you don't have complete context**:
- List directories to understand structure before assuming paths
- Take screenshots to see page state before interacting
- Read file headers or samples before processing large files

**When operations don't work as expected**:
- File not found: List the directory to see what actually exists
- Element not found: Take screenshot to verify page loaded correctly
- No search results: Try broader terms, different file types
```

This approach is more flexible than step-by-step workflows.

## Agent Pools for Parallel Execution

Some specialized agents support pooling for true parallel execution:

```python
from marsys.agents import create_browser_agent_pool

# Create pool
pool = await create_browser_agent_pool(
    num_instances=3,
    model_config=config,
    mode="advanced"
)

# Acquire instance for task
async with pool.acquire(branch_id="task_1") as agent:
    result = await agent.run("Navigate to example.com")

# Pool handles instance allocation and release
```

**Benefits**:
- True parallelism (separate instances, no shared state)
- Automatic instance management
- Fair allocation with queuing
- Resource cleanup

## Related Documentation

- [Base Agent API](../api/agent-class.md)
- [Tool Development Guide](../guides/built-in-tools.md)
- [Multi-Agent Coordination](orchestration.md)
- [Custom Agent Development](custom-agents.md)
- [Browser Automation Concepts](browser-automation.md)

## Support

For issues or questions:
- GitHub Issues: [Report bugs or request features](https://github.com/rezaho/MARS/issues)
- Examples: Check `examples/agents/` for usage patterns
- Tests: Check `tests/agents/` for integration examples

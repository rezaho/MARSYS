# Specialized Tool Classes

MARSYS provides specialized tool classes that encapsulate domain-specific functionality for agents. These tool classes provide structured APIs, error handling, and configuration management.

## Overview

Tool classes provide:

- **Structured Output**: Consistent Dict/JSON responses for easy parsing
- **Error Handling**: Comprehensive error classification and recovery
- **Configuration**: Environment variables or explicit parameters
- **Security**: Validation, timeouts, output size limits
- **Agent Integration**: Methods to generate wrapped functions for agents

## Available Tool Classes

### [FileOperationTools](file-operations.md)

High-level file system operations with type-aware handling.

**Capabilities**: Read, write, edit (unified diff), search content, find files, list directories

**Key Features**:
- Type-aware file reading (Python, JSON, PDF, Markdown, images)
- Unified diff editing for precise changes
- Ripgrep-based content search
- Glob/regex file finding
- Run filesystem boundaries and mounts

See [Run Filesystem](../concepts/run-filesystem.md) for virtual path semantics.

```python
from pathlib import Path
from marsys.environment.file_operations import FileOperationTools, FileOperationConfig
from marsys.environment.filesystem import RunFileSystem

fs = RunFileSystem.local(run_root=Path("/project"))
config = FileOperationConfig(run_filesystem=fs)
file_tools = FileOperationTools(config)

# Get tools dict for agent
tools = file_tools.get_tools()
```

**Use with**: FileOperationAgent, or custom agents needing file operations

[**Read Full Documentation →**](file-operations.md)

---

### ShellTools

Safe shell command execution with validation and specialized helpers.

**Capabilities**: Execute commands, grep, find, sed, awk, tail, head, wc, diff, streaming execution

**Key Features**:
- Command validation with blocked dangerous patterns
- Whitelisting support for production
- Timeout enforcement
- Output size limits (prevents memory exhaustion)
- Specialized helper methods for common operations

```python
from marsys.environment.shell_tools import ShellTools

shell_tools = ShellTools(
    working_directory="/project",
    allowed_commands=["grep", "find", "wc"],  # Whitelist
    timeout_default=30
)

# Get tools dict for agent
tools = shell_tools.get_tools()
```

**Use with**: FileOperationAgent (optional), or custom agents needing shell access

---

### CodeExecutionTools

Safe Python and shell execution toolkit for agent workflows.

**Capabilities**: `python_execute`, `shell_execute`

**Key Features**:
- Resource limits (timeout, output size, memory, CPU)
- Optional persistent Python sessions (`session_persistent_python=True`)
- Security controls for shell patterns and Python modules
- Network disabled by default (`allow_network=False`)
- Virtual output path support via run filesystem (`output_virtual_dir`, default `./outputs`)

```python
from pathlib import Path
from marsys.environment.code import CodeExecutionConfig, CodeExecutionTools
from marsys.environment.filesystem import RunFileSystem

fs = RunFileSystem.local(run_root=Path("./runs/run-20260206"))

config = CodeExecutionConfig(
    run_filesystem=fs,
    timeout_default=30,
    max_memory_mb=1024,
    allow_network=False,
    session_persistent_python=True,
)

code_tools = CodeExecutionTools(config)
tools = code_tools.get_tools()  # {"python_execute": ..., "shell_execute": ...}
```

**Use with**: CodeExecutionAgent, DataAnalysisAgent, or custom agents requiring controlled execution

---

### SearchTools

Multi-source search across web and scholarly databases.

**Capabilities**: DuckDuckGo, Google, arXiv, Semantic Scholar, PubMed search

**Key Features**:
- API key validation at initialization
- Only exposes tools with valid credentials
- Optional API keys for higher rate limits (Semantic Scholar, PubMed)
- Configurable result limits (max 20 per query)

```python
from marsys.environment.search_tools import SearchTools

search_tools = SearchTools(
    google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID_GENERIC")
)

# Get tools (validates API keys, raises ValueError if missing)
try:
    tools = search_tools.get_tools(tools_subset=["duckduckgo", "arxiv"])
except ValueError as e:
    print(f"Missing required API keys: {e}")
```

**Use with**: WebSearchAgent, or custom agents needing search capabilities

---

### BrowserAgent

Browser automation through the BrowserAgent class with Playwright integration.

**Capabilities**: Navigate, click, type, screenshot, JavaScript execution, element extraction

**Key Features**:
- Two modes: PRIMITIVE (content extraction) and ADVANCED (visual interaction)
- Vision-based interaction with auto-screenshot
- Built-in browser tools
- Download discovery via `list_downloads` (see BrowserAgent docs)

```python
from marsys.agents import BrowserAgent
from marsys.models import ModelConfig

# Create BrowserAgent with built-in browser tools
browser_agent = await BrowserAgent.create_safe(
    model_config=ModelConfig(
        type="api",
        provider="openrouter",
        name="anthropic/claude-opus-4.6",
        temperature=0.3
    ),
    name="web_scraper",
    mode="primitive",  # or "advanced" for visual interaction
    goal="Web automation agent",
    instruction="You are a web automation specialist. Navigate websites, extract content, and interact with web pages as instructed.",
    headless=True
)

# Browser tools are automatically available to the agent
result = await browser_agent.run("Navigate to example.com and extract content")

# Always cleanup
await browser_agent.cleanup()
```

**Note**: Browser tools are accessed through BrowserAgent, not a separate BrowserTools class.

[**See BrowserAgent Documentation →**](../concepts/browser-automation.md)

---

## Comparison

| Tool Class | Operations | API Keys | Security Features | Output Format |
|------------|-----------|----------|-------------------|---------------|
| **FileOperationTools** | 6 file ops | None | Run filesystem boundaries, path validation | Dict with success/content |
| **ShellTools** | 10 commands | None | Blocked patterns, whitelist, timeout | Dict with success/output |
| **CodeExecutionTools** | Python + shell execution | None | Resource limits, module/pattern blocking, network controls | ToolResponse (text + optional images) |
| **SearchTools** | 5 sources | None required<br>Google optional | API key validation, rate limits | JSON string |
| **BrowserAgent built-ins** | Browser control | None | Timeout, mode restrictions | Dict/JSON depending on operation |

## Integration Patterns

### Pattern 1: Using with Agents

All tool classes provide `get_tools()` method that returns a dict of wrapped functions:

```python
from marsys.agents import Agent
from marsys.environment.code import CodeExecutionTools
from marsys.environment.file_operations import FileOperationTools
from marsys.environment.shell_tools import ShellTools

file_tools = FileOperationTools()
shell_tools = ShellTools()
code_tools = CodeExecutionTools()

# Combine tools from multiple classes
tools = {}
tools.update(file_tools.get_tools())
tools.update(shell_tools.get_tools())
tools.update(code_tools.get_tools())

agent = Agent(
    model_config=config,
    goal="File operations specialist",
    instruction="...",
    tools=tools
)
```

### Pattern 2: Conditional Tool Loading

Enable tools based on configuration:

```python
from marsys.environment.code import CodeExecutionConfig, CodeExecutionTools
from marsys.environment.search_tools import SearchTools
from marsys.environment.shell_tools import ShellTools

tools = {}

# Always include file operations
tools.update(file_tools.get_tools())

# Conditionally add shell tools
if enable_shell:
    shell_tools = ShellTools(
        allowed_commands=allowed_commands if production else None
    )
    tools.update(shell_tools.get_tools())

# Conditionally add code execution tools
if enable_code_execution:
    code_tools = CodeExecutionTools(
        CodeExecutionConfig(session_persistent_python=enable_persistent_python)
    )
    tools.update(code_tools.get_tools())

# Conditionally add search tools (validates API keys)
if has_search_api_keys:
    try:
        search_tools = SearchTools(google_api_key=api_key, google_cse_id=cse_id)
        tools.update(search_tools.get_tools(tools_subset=["google", "arxiv"]))
    except ValueError:
        logger.warning("Search tools not available - missing API keys")
```

### Pattern 3: Custom Tool Subset

Select specific tools from a class:

```python
from marsys.environment.code import CodeExecutionTools
from marsys.environment.search_tools import SearchTools

# FileOperationTools: get only read/write tools
all_file_tools = file_tools.get_tools()
read_write_only = {
    k: v for k, v in all_file_tools.items()
    if k in ["read_file", "write_file"]
}

# SearchTools: get only scholarly sources
search_tools = SearchTools()
scholarly_only = search_tools.get_tools(tools_subset=["arxiv", "semantic_scholar", "pubmed"])

# CodeExecutionTools: keep only Python execution
code_tools = CodeExecutionTools()
python_only = {
    k: v for k, v in code_tools.get_tools().items()
    if k == "python_execute"
}
```

### Pattern 4: Tool Configuration

Configure tool behavior through initialization:

```python
from pathlib import Path
from marsys.environment.code import CodeExecutionConfig, CodeExecutionTools
from marsys.environment.file_operations import FileOperationTools, FileOperationConfig
from marsys.environment.filesystem import RunFileSystem
from marsys.environment.search_tools import SearchTools
from marsys.environment.shell_tools import ShellTools

# FileOperationTools: restrict to specific run root
fs = RunFileSystem.local(run_root=Path("/app/data"))
config = FileOperationConfig(run_filesystem=fs)
file_tools = FileOperationTools(config)

# ShellTools: production whitelist
shell_tools = ShellTools(
    allowed_commands=["grep", "find", "wc", "ls"],
    blocked_patterns=["rm -rf /", "sudo", "mv /"],  # Additional blocks
    timeout_default=10  # Shorter timeout
)

# CodeExecutionTools: persistent Python + stricter limits
code_tools = CodeExecutionTools(
    CodeExecutionConfig(
        timeout_default=20,
        max_output_bytes=500_000,
        max_memory_mb=1024,
        allow_network=False,
        session_persistent_python=True,
    )
)

# SearchTools: explicit API keys (override env vars)
search_tools = SearchTools(
    google_api_key="explicit_key_here",
    google_cse_id="explicit_cse_id_here",
    semantic_scholar_api_key="explicit_key_here"
)
```

## Creating Custom Tool Classes

To create your own tool class, follow these patterns:

### 1. Class Structure

```python
from typing import Dict, Callable, Any
from pathlib import Path

class MyToolClass:
    """Custom tool class for domain-specific operations."""

    def __init__(
        self,
        config_param: str,
        optional_param: Optional[int] = None
    ):
        """Initialize with configuration."""
        self.config_param = config_param
        self.optional_param = optional_param or 10

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate configuration at initialization."""
        if not self.config_param:
            raise ValueError("config_param is required")

    def get_tools(self) -> Dict[str, Callable]:
        """Return wrapped tool functions for agent integration."""
        return {
            "my_tool": self.my_tool_operation,
            "another_tool": self.another_operation
        }

    async def my_tool_operation(
        self,
        param: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Tool operation with structured output.

        Returns:
            Dict with success, result, and optional error fields
        """
        try:
            # Perform operation
            result = self._do_work(param)

            return {
                "success": True,
                "result": result,
                "metadata": {"param": param}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
```

### 2. Structured Output

Always return Dict with consistent structure:

```python
# Success response
{
    "success": True,
    "result": <data>,
    "metadata": {...}  # Optional
}

# Error response
{
    "success": False,
    "error": <error_message>,
    "error_type": <exception_type>,
    "details": {...}  # Optional
}
```

### 3. Validation

Validate inputs and configuration:

```python
def _validate_params(self, param: str) -> None:
    """Validate parameters before execution."""
    if not param:
        raise ValueError("param cannot be empty")

    if len(param) > 1000:
        raise ValueError("param too long (max 1000 chars)")
```

### 4. Error Handling

Catch and classify errors:

```python
try:
    result = await self._perform_operation(param)
    return {"success": True, "result": result}
except FileNotFoundError as e:
    return {
        "success": False,
        "error": f"File not found: {e}",
        "error_type": "FileNotFoundError",
        "recoverable": True
    }
except PermissionError as e:
    return {
        "success": False,
        "error": f"Permission denied: {e}",
        "error_type": "PermissionError",
        "recoverable": False
    }
except Exception as e:
    return {
        "success": False,
        "error": f"Unexpected error: {e}",
        "error_type": type(e).__name__,
        "recoverable": False
    }
```

## Best Practices

### 1. Use Tool Classes for Consistency

**✅ DO**: Use tool classes for structured, tested functionality
```python
file_tools = FileOperationTools()
tools = file_tools.get_tools()
```

**❌ DON'T**: Create ad-hoc tool functions without structure
```python
def my_read_file(path):
    with open(path) as f:
        return f.read()  # No error handling, no structured output
```

### 2. Validate Configuration Early

Validate at initialization, not at tool invocation:

```python
class MyTools:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key required")  # Fail fast
        self.api_key = api_key
```

### 3. Provide Helpful Error Messages

```python
# ✅ Helpful
raise ValueError(
    "Google Search requires GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID_GENERIC. "
    "Get from: https://developers.google.com/custom-search/v1/overview"
)

# ❌ Vague
raise ValueError("Missing API key")
```

### 4. Use Type Hints

```python
async def search(
    self,
    query: str,
    max_results: int = 10
) -> Dict[str, Any]:
    """Type hints improve IDE support and clarity."""
    ...
```

### 5. Document Return Formats

```python
async def my_tool(self, param: str) -> Dict[str, Any]:
    """
    Perform operation.

    Args:
        param: Input parameter

    Returns:
        Dict with:
            - success (bool): Operation success
            - result (Any): Operation result if successful
            - error (str): Error message if failed
    """
```

## Environment Variables

Tool classes commonly use environment variables for API keys:

| Tool Class | Environment Variables | Required |
|------------|----------------------|----------|
| **SearchTools** | `GOOGLE_SEARCH_API_KEY` (optional)<br>`GOOGLE_CSE_ID_GENERIC` (optional)<br>`SEMANTIC_SCHOLAR_API_KEY` (optional)<br>`NCBI_API_KEY` (optional) | None (DuckDuckGo is free) |
| **FileOperationTools** | None | No |
| **ShellTools** | None | No |
| **CodeExecutionTools** | None | No |
| **BrowserAgent built-ins** | None | No |

Setup guides:
- [Google Custom Search](https://developers.google.com/custom-search/v1/overview)
- [Semantic Scholar API](https://www.semanticscholar.org/product/api#api-key-form)

## Related Documentation

- [Built-in Tools Guide](built-in-tools.md) - Simple function-based tools
- [Specialized Agents](../concepts/specialized-agents.md) - Agents using these tool classes
- [Custom Tool Development](../tutorials/custom-tools.md) - Creating your own tools
- [Agent Development](../api/agent-class.md) - Building agents with tools

## Support

For issues or questions:
- GitHub Issues: [Report bugs or request features](https://github.com/rezaho/MARS/issues)
- Examples: Check `examples/tools/` for usage patterns
- Tests: Check `tests/environment/` for integration examples

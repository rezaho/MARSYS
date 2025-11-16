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

### [FileOperationTools](file-operation-tools.md)

High-level file system operations with type-aware handling.

**Capabilities**: Read, write, edit (unified diff), search content, find files, list directories

**Key Features**:
- Type-aware file reading (Python, JSON, PDF, Markdown, images)
- Unified diff editing for precise changes
- R ipgrep-based content search
- Glob/regex file finding
- Base directory restrictions for security

```python
from marsys.environment.file_operations import FileOperationTools

file_tools = FileOperationTools(
    base_directory=Path("/project"),
    force_base_directory=True  # Jail to base directory
)

# Get tools dict for agent
tools = file_tools.get_tools()
```

**Use with**: FileOperationAgent, or custom agents needing file operations

[**Read Full Documentation →**](file-operation-tools.md)

---

### [BashTools](bash-tools.md)

Safe bash command execution with validation and specialized helpers.

**Capabilities**: Execute commands, grep, find, sed, awk, tail, head, wc, diff, streaming execution

**Key Features**:
- Command validation with blocked dangerous patterns
- Whitelisting support for production
- Timeout enforcement
- Output size limits (prevents memory exhaustion)
- Specialized helper methods for common operations

```python
from marsys.environment.bash_tools import BashTools

bash_tools = BashTools(
    working_directory="/project",
    allowed_commands=["grep", "find", "wc"],  # Whitelist
    timeout_default=30
)

# Get tools dict for agent
tools = bash_tools.get_tools()
```

**Use with**: FileOperationAgent (optional), or custom agents needing shell access

[**Read Full Documentation →**](bash-tools.md)

---

### [SearchTools](search-tools.md)

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

[**Read Full Documentation →**](search-tools.md)

---

### [BrowserTools](browser-tools.md)

Browser automation and control with Playwright integration.

**Capabilities**: Navigate, click, type, screenshot, JavaScript execution, element extraction

**Key Features**:
- Multi-mode operation (basic, CDP, stealth, vision)
- Vision-based interaction (no selectors)
- Auto-screenshot management
- Console monitoring
- Cookie/storage management

```python
from marsys.environment.browser_tools import BrowserTools

browser_tools = BrowserTools(
    mode="vision",
    headless=True,
    timeout=30
)

# Get tools dict for agent
tools = await browser_tools.get_tools()
```

**Use with**: BrowserAgent, or custom agents needing web automation

[**Read Full Documentation →**](browser-tools.md)

---

## Comparison

| Tool Class | Operations | API Keys | Security Features | Output Format |
|------------|-----------|----------|-------------------|---------------|
| **FileOperationTools** | 6 file ops | None | Base dir restriction, path validation | Dict with success/content |
| **BashTools** | 10 commands | None | Blocked patterns, whitelist, timeout | Dict with success/output |
| **SearchTools** | 5 sources | None required<br>Google optional | API key validation, rate limits | JSON string |
| **BrowserTools** | Browser control | None | Timeout, mode restrictions | Dict/JSON depending on operation |

## Integration Patterns

### Pattern 1: Using with Agents

All tool classes provide `get_tools()` method that returns a dict of wrapped functions:

```python
from marsys.agents import Agent

file_tools = FileOperationTools()
bash_tools = BashTools()

# Combine tools from multiple classes
tools = {}
tools.update(file_tools.get_tools())
tools.update(bash_tools.get_tools())

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
tools = {}

# Always include file operations
tools.update(file_tools.get_tools())

# Conditionally add bash tools
if enable_bash:
    bash_tools = BashTools(
        allowed_commands=allowed_commands if production else None
    )
    tools.update(bash_tools.get_tools())

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
# FileOperationTools: get only read/write tools
all_file_tools = file_tools.get_tools()
read_write_only = {
    k: v for k, v in all_file_tools.items()
    if k in ["read_file", "write_file"]
}

# SearchTools: get only scholarly sources
search_tools = SearchTools()
scholarly_only = search_tools.get_tools(tools_subset=["arxiv", "semantic_scholar", "pubmed"])
```

### Pattern 4: Tool Configuration

Configure tool behavior through initialization:

```python
# FileOperationTools: restrict to specific directory
file_tools = FileOperationTools(
    base_directory=Path("/app/data"),
    force_base_directory=True  # Cannot escape /app/data
)

# BashTools: production whitelist
bash_tools = BashTools(
    allowed_commands=["grep", "find", "wc", "ls"],
    blocked_commands=["rm", "mv"],  # Additional blocks
    timeout_default=10  # Shorter timeout
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
| **BashTools** | None | No |
| **BrowserTools** | None | No |

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

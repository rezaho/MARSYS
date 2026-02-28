# Tools API

Complete API reference for the tool system that enables agents to execute functions and interact with external services.

## 🎯 Overview

The Tools API provides automatic schema generation, tool execution, and integration with agent capabilities, supporting OpenAI-compatible function calling.

## 📦 Core Functions

### generate_openai_tool_schema

Generates OpenAI-compatible tool schema from Python functions.

**Import:**
```python
from marsys.environment.utils import generate_openai_tool_schema
```

**Signature:**
```python
def generate_openai_tool_schema(
    func: Callable,
    func_name: str
) -> Dict[str, Any]
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `func` | `Callable` | Function to generate schema for | Required |
| `func_name` | `str` | Name for the tool in schema | Required |

**Returns:** Dictionary with OpenAI tool schema format

**Example:**
```python
def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results
    """
    # Implementation
    pass

# Generate schema
schema = generate_openai_tool_schema(search_web, "search_web")

# Result:
{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
```

---

### Tool Schema Structure

OpenAI-compatible tool schema format.

```python
{
    "type": "function",
    "function": {
        "name": str,           # Function name
        "description": str,     # Function description
        "parameters": {
            "type": "object",
            "properties": {
                # Parameter definitions
                "param_name": {
                    "type": str,        # JSON schema type
                    "description": str,  # Parameter description
                    "default": Any,     # Default value (optional)
                    "enum": List[Any]   # Allowed values (optional)
                }
            },
            "required": List[str]  # Required parameter names
        }
    }
}
```

---

## 🎨 Tool Creation Patterns

### Basic Tool Function

```python
def calculate_statistics(
    data: List[float],
    include_std: bool = True
) -> Dict[str, float]:
    """
    Calculate statistics for numerical data.

    Args:
        data: List of numerical values
        include_std: Whether to include standard deviation

    Returns:
        Dictionary with statistical measures
    """
    import statistics

    result = {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data)
    }

    if include_std and len(data) > 1:
        result["std"] = statistics.stdev(data)

    return result
```

### Tool with Complex Types

```python
from typing import List, Dict, Optional, Literal

def process_data(
    input_data: Dict[str, Any],
    operation: Literal["transform", "filter", "aggregate"],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process data with specified operation.

    Args:
        input_data: Input data dictionary
        operation: Type of operation to perform
        options: Additional operation options

    Returns:
        Processed data result
    """
    # Implementation
    pass

# Schema includes enum for operation
schema = generate_openai_tool_schema(process_data, "process_data")
```

### Async Tool Function

```python
async def fetch_api_data(
    endpoint: str,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Fetch data from API endpoint.

    Args:
        endpoint: API endpoint URL
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        API response data
    """
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(
            endpoint,
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            return await response.json()
```

---

## 🔧 Tool Integration with Agents

### Adding Tools to Agent

```python
from marsys.agents import Agent

# Define tools
def search_tool(query: str) -> List[str]:
    """Search for information."""
    # Implementation
    return ["result1", "result2"]

def calculate_tool(expression: str) -> float:
    """Calculate mathematical expression."""
    # Implementation
    return eval(expression)  # Simplified example

# Create agent with tools
agent = Agent(
    model_config=model_config,
    name="Assistant",
    goal="Assistant with search and calculation capabilities",
    instruction="Use search_tool for lookup tasks and calculate_tool for math.",
    tools={"search_tool": search_tool, "calculate_tool": calculate_tool},
)

# Tools are automatically available to the agent
```

### Custom Tool Name Mapping

```python
from typing import Literal

# Define tool function
def custom_tool_func(input_text: str, mode: Literal["fast", "accurate"] = "fast") -> str:
    """Custom tool with explicit mode selection."""
    return f"{mode}: {input_text}"

# Map to a custom public tool name
agent = Agent(
    model_config=model_config,
    name="CustomAgent",
    goal="Run custom processing tasks",
    instruction="Use custom_tool when the request needs this specialized processing.",
    tools={"custom_tool": custom_tool_func}
)
```

---

## 🔄 Tool Execution

### ToolExecutor

Executes tool calls within the coordination system.

**Import:**
```python
from marsys.coordination.execution import ToolExecutor
```

**Key Methods:**

#### execute_tool_call
```python
async def execute_tool_call(
    tool_call: Dict[str, Any],
    available_tools: Dict[str, Callable],
    context: Dict[str, Any]
) -> Any
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tool_call` | `Dict` | Tool call specification | Required |
| `available_tools` | `Dict[str, Callable]` | Available tool functions | Required |
| `context` | `Dict` | Execution context | Required |

**Tool Call Format:**
```python
tool_call = {
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "search_web",
        "arguments": '{"query": "AI trends"}'
    }
}
```

**Example:**
```python
executor = ToolExecutor()

# Execute tool call
result = await executor.execute_tool_call(
    tool_call=tool_call,
    available_tools={"search_web": search_web},
    context={"session_id": "123"}
)

# Result is the tool's return value
```

---

## 🎨 Advanced Tool Patterns

### Tool with File I/O

```python
def read_csv_file(
    filepath: str,
    encoding: str = "utf-8",
    delimiter: str = ","
) -> List[Dict[str, str]]:
    """
    Read CSV file and return as list of dictionaries.

    Args:
        filepath: Path to CSV file
        encoding: File encoding
        delimiter: CSV delimiter

    Returns:
        List of row dictionaries
    """
    import csv

    with open(filepath, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return list(reader)
```

### Tool with External API

```python
import os
import requests

def get_weather(
    city: str,
    units: Literal["metric", "imperial"] = "metric"
) -> Dict[str, Any]:
    """
    Get current weather for a city.

    Args:
        city: City name
        units: Temperature units

    Returns:
        Weather data dictionary
    """
    api_key = os.getenv("WEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"

    response = requests.get(url, params={
        "q": city,
        "units": units,
        "appid": api_key
    })

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Weather API error: {response.status_code}")
```

### Tool with State Management

```python
class StatefulTool:
    """Tool that maintains state between calls."""

    def __init__(self):
        self.history = []
        self.cache = {}

    def process_with_memory(
        self,
        input_data: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process data with memory of previous calls.

        Args:
            input_data: Input to process
            use_cache: Whether to use cached results

        Returns:
            Processing result with context
        """
        # Check cache
        if use_cache and input_data in self.cache:
            return {
                "result": self.cache[input_data],
                "cached": True,
                "history_length": len(self.history)
            }

        # Process
        result = self._process(input_data)

        # Update state
        self.history.append(input_data)
        self.cache[input_data] = result

        return {
            "result": result,
            "cached": False,
            "history_length": len(self.history)
        }

    def _process(self, data: str) -> str:
        # Actual processing logic
        return data.upper()

# Create instance and use as tool
stateful_tool = StatefulTool()
agent = Agent(
    model_config=config,
    name="StatefulProcessor",
    goal="Process text with cached state",
    instruction="Use stateful_process for repeated inputs and report cache usage.",
    tools={"stateful_process": stateful_tool.process_with_memory},
)
```

---

## 📋 Type Mapping

### Python to JSON Schema Type Mapping

| Python Type | JSON Schema Type | Example |
|-------------|------------------|---------|
| `str` | `"string"` | `name: str` |
| `int` | `"integer"` | `age: int` |
| `float` | `"number"` | `price: float` |
| `bool` | `"boolean"` | `active: bool` |
| `List[T]` | `"array"` | `items: List[str]` |
| `Dict[K, V]` | `"object"` | `data: Dict[str, Any]` |
| `Optional[T]` | `T` with nullable | `value: Optional[int]` |
| `Literal[...]` | enum | `mode: Literal["a", "b"]` |
| `Any` | No type constraint | `data: Any` |

---

## 🚦 Error Handling

### Tool Execution Errors

```python
def safe_tool_wrapper(func: Callable) -> Callable:
    """Wrap tool function with error handling."""

    async def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "tool_name": func.__name__
            }

    # Preserve function metadata for schema generation
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__annotations__ = func.__annotations__

    return wrapper

# Use wrapper
@safe_tool_wrapper
def risky_tool(data: str) -> str:
    """Tool that might fail."""
    if not data:
        raise ValueError("Empty data")
    return data.upper()
```

---

## 📋 Best Practices

### ✅ DO:
- Add clear docstrings with parameter descriptions
- Use type hints for all parameters
- Provide default values where appropriate
- Handle errors gracefully
- Validate input parameters
- Return structured data (dicts/lists)

### ❌ DON'T:
- Use `*args` or `**kwargs` (breaks schema generation)
- Return complex objects (use dicts instead)
- Perform long-running operations without timeout
- Modify global state without careful design
- Expose sensitive operations without validation

---

## 🚦 Related Documentation

- [Agent API](agent-class.md) - Agent tool integration
- [Memory API](memory.md) - Tool call messages
- [Execution API](execution.md) - Tool execution
- [Tool Patterns](../concepts/tool-patterns.md) - Common patterns

---

!!! tip "Pro Tip"
    Always include comprehensive docstrings with Google-style parameter descriptions. The schema generator extracts descriptions from docstrings to create helpful tool descriptions for the LLM.

!!! warning "Security"
    Be careful with tools that execute code, access files, or make network requests. Always validate inputs and implement appropriate access controls.

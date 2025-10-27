# FileOperationAgent

**FileOperationAgent** is a specialized agent for intelligent file and directory operations with optional bash command execution.

## Overview

FileOperationAgent combines [FileOperationTools](../guides/file-operation-tools.md) with optional [BashTools](../guides/bash-tools.md) to provide a complete file system automation solution. It uses scenario-based instructions that adapt to different situations rather than following rigid step-by-step workflows.

**Best for**: Code analysis, configuration management, log processing, documentation generation, data file operations

## Quick Start

### Basic Usage (File Operations Only)

```python
from marsys.agents import FileOperationAgent
from marsys.models import ModelConfig
import os

agent = FileOperationAgent(
    model_config=ModelConfig(
        type="api",
        name="anthropic/claude-haiku-4.5",
        provider="openrouter",
        api_key=os.getenv("OPENROUTER_API_KEY")
    ),
    agent_name="FileHelper",
    enable_bash=False  # Default: bash disabled
)

# Use the agent
result = await agent.run(
    "Read config.json and extract database settings"
)
```

### With Bash Tools Enabled

```python
agent = FileOperationAgent(
    model_config=model_config,
    agent_name="FileHelper",
    enable_bash=True,  # Enable bash commands
    working_directory="/path/to/project",
    bash_timeout_default=30
)

result = await agent.run(
    "Find all Python files modified in last week and count their lines"
)
```

### With Restricted Bash Commands (Production)

```python
agent = FileOperationAgent(
    model_config=model_config,
    agent_name="RestrictedFileHelper",
    enable_bash=True,
    allowed_bash_commands=["grep", "find", "wc", "ls"],  # Whitelist
    blocked_bash_commands=["rm", "mv"]  # Additional blocks
)
```

## Features

### File Operation Tools (Always Available)

**6 core tools** for file system operations:

1. **read_file**: Type-aware reading (text, PDF, images, JSON, YAML, Markdown)
2. **write_file**: Create/overwrite files with automatic directory creation
3. **edit_file**: Unified diff editing for precise changes
4. **search_content**: Ripgrep-based content search
5. **search_files**: Find files by name patterns (glob/regex)
6. **list_directory**: List directory contents with metadata

### Bash Tools (Optional - 10 Tools)

When `enable_bash=True`, adds **10 bash helper tools**:

1. **bash_execute**: General command execution with timeout
2. **bash_grep**: Pattern matching with context lines
3. **bash_find**: File finding with filters
4. **bash_sed**: Stream editing
5. **bash_awk**: Text processing and field extraction
6. **bash_tail**: View end of files
7. **bash_head**: View start of files
8. **bash_wc**: Count lines, words, characters
9. **bash_diff**: Compare files
10. **bash_execute_streaming**: Long-running commands with streaming output

## Configuration

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_config` | `ModelConfig` | Required | Model configuration |
| `agent_name` | `str` | `"FileOperationAgent"` | Agent identifier |
| `enable_bash` | `bool` | `False` | Enable bash tools |

### File Operation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_directory` | `Path` | `Path.cwd()` | Base directory for operations |
| `force_base_directory` | `bool` | `False` | Restrict to base directory |

### Bash Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_directory` | `str` | Current dir | Working directory for bash |
| `allowed_bash_commands` | `List[str]` | `None` | Whitelist (None = all allowed) |
| `blocked_bash_commands` | `List[str]` | Default blocks | Additional commands to block |
| `bash_timeout_default` | `int` | `30` | Default timeout (seconds) |

## Security Features

### Bash Command Validation

**Default blocked patterns** (when bash enabled):
- Deletion: `rm`, `rmdir`, `rm -rf`
- Dangerous moves: `mv`, `cp /dev/null`
- Permission changes: `chmod`, `chown`
- Disk operations: `mkfs`, `dd`
- Privilege escalation: `sudo`, `su`
- Process termination: `kill`, `killall`
- System control: `reboot`, `shutdown`

### Path Security

- **Base directory restriction**: `force_base_directory=True` jails operations
- **Path traversal prevention**: Automatic `../` attack detection
- **Symbolic link handling**: Safe symlink resolution

## Common Use Cases

### Code Analysis

```python
agent = FileOperationAgent(config, enable_bash=True)

await agent.run("""
Analyze Python files in src/:
- Find all files with TODO comments
- Count total lines of code
- List files larger than 500 lines
""")
```

### Configuration Management

```python
agent = FileOperationAgent(
    config,
    enable_bash=False,
    base_directory=Path("/etc/myapp"),
    force_base_directory=True  # Restricted to /etc/myapp
)

await agent.run(
    "Read database.yaml and update connection pool size to 20"
)
```

### Log Analysis

```python
agent = FileOperationAgent(
    config,
    enable_bash=True,
    working_directory="/var/log"
)

await agent.run("""
Analyze application logs from last hour:
- Count ERROR and WARNING occurrences
- Extract unique error messages
- Show last 50 lines of most recent log
""")
```

### Documentation Generation

```python
agent = FileOperationAgent(config, enable_bash=False)

await agent.run("""
Generate API documentation:
1. Read all Python files in api/
2. Extract docstrings from public functions
3. Create markdown file with API reference
""")
```

## Adaptive Behavior

The agent uses scenario-based instructions that adapt to context:

### When File Not Found
- Lists directory to see actual contents
- Suggests alternatives based on similar names

### When Search Returns No Results
- Tries broader search patterns
- Checks different file types
- Suggests different search locations

### When Edit Fails
- Re-reads exact section to get current state
- Retries with precise line matching
- Falls back to `write_file` if necessary

### When Working with Different File Types

**Python files**: Preserves imports, uses unified diff

**JSON/YAML**: Validates syntax after edits

**PDF files**: Extracts images by default (diagrams often crucial)

**Markdown**: Preserves formatting and structure

**Images**: Reads as base64 for vision models

## Multi-Agent Integration

### Sequential Pipeline

```python
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig

topology = PatternConfig.pipeline(
    stages=[
        {"name": "fetch", "agents": ["DataFetcher"]},
        {"name": "process", "agents": ["FileManager"]},
        {"name": "analyze", "agents": ["Analyzer"]}
    ]
)

await Orchestra.run(
    task="Process and analyze downloaded datasets",
    topology=topology
)
```

### Hub-and-Spoke

```python
file_agent = FileOperationAgent(config, enable_bash=True)

topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["FileManager", "CodeReviewer", "TestRunner"]
)

await Orchestra.run(
    task="Refactor authentication module",
    topology=topology
)
```

## Best Practices

### 1. Choose Bash Enablement Wisely

**Enable bash when you need**:
- Complex text processing (awk, sed)
- File system operations beyond basic CRUD
- Integration with system tools
- Performance-critical operations

**Use file operations only when**:
- Simple CRUD operations
- Working in untrusted environments
- Maximum portability (Windows compatibility)

### 2. Use Command Whitelisting in Production

```python
# Development: permissive
agent = FileOperationAgent(enable_bash=True)

# Production: strict whitelist
agent = FileOperationAgent(
    enable_bash=True,
    allowed_bash_commands=["grep", "find", "wc", "ls", "cat"]
)
```

### 3. Set Appropriate Timeouts

```python
# Quick operations
agent = FileOperationAgent(enable_bash=True, bash_timeout_default=10)

# Long-running analysis
agent = FileOperationAgent(enable_bash=True, bash_timeout_default=300)
```

### 4. Use Base Directory Restriction

```python
agent = FileOperationAgent(
    base_directory=Path("/app/data"),
    force_base_directory=True  # Cannot escape /app/data
)
```

### 5. Provide Clear Task Descriptions

```python
# ❌ Vague
await agent.run("Fix the config")

# ✅ Clear
await agent.run("""
Update config/database.yaml:
- Change max_connections from 50 to 100
- Add timeout: 30 under connection section
- Preserve all other settings
""")
```

## Troubleshooting

### Bash commands not working
- Check `enable_bash=True` set
- Verify command not in blocked list
- Check whitelist if using `allowed_bash_commands`
- Ensure sufficient timeout

### File edit failures
- File exists and readable?
- Exact line matching (diff requires precision)
- File unchanged since last read?
- No special character encoding issues?

### Permission denied
- Process has read/write permissions?
- Not trying to write to system directories?
- `base_directory` accessible?
- SELinux/AppArmor not blocking?

## API Reference

```python
FileOperationAgent(
    model_config: ModelConfig,
    agent_name: str = "FileOperationAgent",
    goal: Optional[str] = None,
    instruction: Optional[str] = None,
    enable_bash: bool = False,
    working_directory: Optional[str] = None,
    base_directory: Optional[Path] = None,
    force_base_directory: bool = False,
    allowed_bash_commands: Optional[list] = None,
    blocked_bash_commands: Optional[list] = None,
    bash_timeout_default: int = 30,
    **kwargs
)
```

**Methods**:
- `run(prompt, context=None, **kwargs)`: Execute agent with prompt
- `get_capabilities()`: Returns dict with available tools and features
- `cleanup()`: Cleanup resources

## Related Documentation

- [Specialized Agents Overview](specialized-agents.md)
- [FileOperationTools Guide](../guides/file-operation-tools.md)
- [BashTools Guide](../guides/bash-tools.md)
- [Multi-Agent Coordination](../api/orchestra.md)

## Examples

See `examples/agents/` for complete working examples:
- `file_operations_basic.py`
- `file_operations_with_bash.py`
- `code_analysis.py`
- `config_management.py`

# Configuration API

Complete API reference for the MARSYS configuration system, including execution, status, communication, and error handling configurations.

## 🎯 Overview

The configuration system provides hierarchical settings for all aspects of multi-agent orchestration, with sensible defaults and granular override capabilities.

## 📦 Core Classes

### ExecutionConfig

Main configuration for orchestration execution.

**Import:**
```python
from marsys.coordination.config import ExecutionConfig, StatusConfig, ErrorHandlingConfig
```

**Constructor:**
```python
ExecutionConfig(
    # Timeouts
    convergence_timeout: float = 300.0,
    branch_timeout: float = 600.0,
    agent_acquisition_timeout: float = 240.0,
    step_timeout: float = 600.0,
    tool_execution_timeout: float = 120.0,
    user_interaction_timeout: float = 300.0,

    # Convergence behavior
    dynamic_convergence_enabled: bool = True,
    parent_completes_on_spawn: bool = True,
    auto_detect_convergence: bool = True,

    # Steering
    steering_mode: Literal["auto", "always", "error"] = "error",

    # User interaction
    user_first: bool = False,
    initial_user_msg: Optional[str] = None,
    user_interaction: str = "terminal",

    # Agent lifecycle management
    auto_cleanup_agents: bool = True,
    cleanup_scope: Literal["topology_nodes", "used_agents"] = "topology_nodes",

    # Sub-configurations
    status: StatusConfig = field(default_factory=StatusConfig),
    error_handling: Optional[ErrorHandlingConfig] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `convergence_timeout` | `float` | Max wait at convergence points (seconds) | `300.0` |
| `branch_timeout` | `float` | Overall branch execution timeout | `600.0` |
| `agent_acquisition_timeout` | `float` | Timeout for acquiring from pool | `240.0` |
| `step_timeout` | `float` | Individual step timeout | `600.0` |
| `tool_execution_timeout` | `float` | Tool call timeout | `120.0` |
| `user_interaction_timeout` | `float` | User input timeout | `300.0` |
| `dynamic_convergence_enabled` | `bool` | Enable runtime convergence detection | `True` |
| `parent_completes_on_spawn` | `bool` | Parent completes when spawning children | `True` |
| `auto_detect_convergence` | `bool` | Automatic convergence point detection | `True` |
| `steering_mode` | `str` | Retry guidance mode (`auto`, `always`, `error`) | `"error"` |
| `user_first` | `bool` | Show message to user before task | `False` |
| `initial_user_msg` | `str` | Custom initial message for user | `None` |
| `user_interaction` | `str` | User interaction type | `"terminal"` |
| `auto_cleanup_agents` | `bool` | Automatically cleanup agents after run | `True` |
| `cleanup_scope` | `str` | Which agents to cleanup | `"topology_nodes"` |
| `status` | `StatusConfig` | Status configuration | `StatusConfig()` |

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `should_apply_steering(is_retry)` | Determine if steering should be applied | `bool` |

**Example:**
```python
# Production configuration
config = ExecutionConfig(
    convergence_timeout=600.0,
    step_timeout=300.0,
    steering_mode="auto",
    status=StatusConfig.from_verbosity(1)
)

# Development configuration
dev_config = ExecutionConfig(
    step_timeout=30.0,
    steering_mode="always",
    status=StatusConfig.from_verbosity(2)
)

# Interactive configuration
interactive_config = ExecutionConfig(
    user_first=True,
    initial_user_msg="Hello! How can I help?",
    user_interaction_timeout=300.0
)

# Long-running workflow (disable auto-cleanup)
long_running_config = ExecutionConfig(
    auto_cleanup_agents=False,  # Keep agents alive between runs
    convergence_timeout=3600.0
)
```

---

### StatusConfig

Configuration for monitoring and output display.

**Import:**
```python
from marsys.coordination.config import StatusConfig, VerbosityLevel
```

**Constructor:**
```python
StatusConfig(
    enabled: bool = False,
    verbosity: Optional[VerbosityLevel] = None,
    cli_output: bool = True,
    cli_colors: bool = True,
    show_thoughts: bool = True,
    show_tool_calls: bool = True,
    show_timings: bool = True,
    aggregation_window_ms: int = 500,
    aggregate_parallel: bool = True,
    max_events_per_session: int = 10000,
    session_cleanup_after_s: int = 3600,
    channels: List[str] = field(default_factory=lambda: ["cli"]),
    show_agent_prefixes: bool = True,
    prefix_width: int = 20,
    prefix_alignment: str = "left",
    follow_up_timeout: float = 30.0
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `enabled` | `bool` | Enable status output | `False` |
| `verbosity` | `VerbosityLevel` | Output verbosity level | `None` |
| `cli_output` | `bool` | Show CLI output | `True` |
| `cli_colors` | `bool` | Use colored output | `True` |
| `show_thoughts` | `bool` | Show agent reasoning | `True` |
| `show_tool_calls` | `bool` | Show tool usage | `True` |
| `show_timings` | `bool` | Show execution times | `True` |
| `aggregation_window_ms` | `int` | Group updates within window | `500` |
| `aggregate_parallel` | `bool` | Aggregate parallel updates | `True` |
| `show_agent_prefixes` | `bool` | Show agent names in output | `True` |
| `prefix_width` | `int` | Width of agent name prefix | `20` |
| `prefix_alignment` | `str` | Prefix alignment | `"left"` |

**Class Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `from_verbosity(level)` | Create from verbosity level | `StatusConfig` |

**Instance Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `should_show_event(event_type)` | Check if event should be shown | `bool` |

**VerbosityLevel Enum:**
```python
class VerbosityLevel(IntEnum):
    QUIET = 0     # Minimal output
    NORMAL = 1    # Standard output
    VERBOSE = 2   # Detailed output
```

**Example:**
```python
# Create from verbosity level
status = StatusConfig.from_verbosity(1)

# Custom configuration
custom_status = StatusConfig(
    enabled=True,
    verbosity=VerbosityLevel.NORMAL,
    show_thoughts=True,
    show_tool_calls=True,
    cli_colors=False  # Disable colors for logs
)

# Quiet mode with timing
timing_only = StatusConfig(
    enabled=True,
    verbosity=VerbosityLevel.QUIET,
    show_timings=True
)
```

---

### CommunicationConfig

Configuration for user interaction and communication channels.

**Import:**
```python
from marsys.coordination.config import CommunicationConfig
```

**Constructor:**
```python
CommunicationConfig(
    use_rich_formatting: bool = True,
    theme_name: str = "modern",
    prefix_width: int = 20,
    show_timestamps: bool = True,
    enable_history: bool = True,
    history_size: int = 1000,
    enable_tab_completion: bool = True,
    use_colors: bool = True,
    color_depth: str = "truecolor",
    input_timeout: Optional[float] = None,
    show_typing_indicator: bool = False,
    use_enhanced_terminal: bool = True,
    fallback_on_error: bool = True
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_rich_formatting` | `bool` | Use rich text formatting | `True` |
| `theme_name` | `str` | Theme name | `"modern"` |
| `prefix_width` | `int` | Width for agent name prefixes | `20` |
| `show_timestamps` | `bool` | Show timestamps in interactions | `True` |
| `enable_history` | `bool` | Keep interaction history | `True` |
| `history_size` | `int` | Maximum history entries | `1000` |
| `enable_tab_completion` | `bool` | Tab completion for inputs | `True` |
| `use_colors` | `bool` | Enable colored output | `True` |
| `color_depth` | `str` | Color depth setting | `"truecolor"` |
| `input_timeout` | `float` | Timeout for user input | `None` |
| `use_enhanced_terminal` | `bool` | Use enhanced terminal features | `True` |
| `fallback_on_error` | `bool` | Fallback to basic mode on error | `True` |

**Theme Options:**
- `"modern"` - Modern theme with gradients
- `"classic"` - Traditional terminal colors
- `"minimal"` - Minimal styling

**Color Depth Options:**
- `"truecolor"` - 24-bit true color
- `"256"` - 256-color palette
- `"16"` - 16-color basic palette
- `"none"` - No colors

**Example:**
```python
# Rich terminal configuration
rich_config = CommunicationConfig(
    use_rich_formatting=True,
    theme_name="modern",
    use_colors=True,
    color_depth="truecolor"
)

# Basic terminal configuration
basic_config = CommunicationConfig(
    use_rich_formatting=False,
    use_colors=False,
    use_enhanced_terminal=False
)

# Web interface configuration
web_config = CommunicationConfig(
    use_rich_formatting=False,  # Web handles formatting
    use_colors=False,            # Web handles colors
    input_timeout=0              # Non-blocking
)
```

---

### ErrorHandlingConfig

Advanced error handling and recovery configuration.

**Import:**
```python
from marsys.coordination.config import ErrorHandlingConfig
```

**Constructor:**
```python
ErrorHandlingConfig(
    use_error_classification: bool = True,
    notify_on_critical_errors: bool = True,
    enable_error_routing: bool = True,
    preserve_error_context: bool = True,
    auto_retry_on_rate_limits: bool = True,
    max_rate_limit_retries: int = 3,
    pool_retry_attempts: int = 2,
    pool_retry_delay: float = 5.0,
    timeout_seconds: float = 300.0,
    timeout_retry_enabled: bool = False,
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_error_classification` | `bool` | Enable intelligent error classification | `True` |
| `notify_on_critical_errors` | `bool` | Send notifications for critical errors | `True` |
| `enable_error_routing` | `bool` | Route errors to User node for intervention | `True` |
| `preserve_error_context` | `bool` | Include full error context in responses | `True` |
| `auto_retry_on_rate_limits` | `bool` | Automatically retry rate-limited requests | `True` |
| `max_rate_limit_retries` | `int` | Maximum retries for rate limit errors | `3` |
| `pool_retry_attempts` | `int` | Number of retries for pool exhaustion | `2` |
| `pool_retry_delay` | `float` | Delay between pool retry attempts | `5.0` |
| `timeout_seconds` | `float` | Default timeout for operations | `300.0` |
| `timeout_retry_enabled` | `bool` | Retry after timeout | `False` |
| `provider_settings` | `Dict` | Provider-specific settings | `{}` |

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `get_provider_setting(provider, setting, default)` | Get provider-specific setting | `Any` |

**Provider Settings Structure:**
```python
provider_settings = {
    "openai": {
        "max_retries": 3,
        "base_retry_delay": 60,
        "insufficient_quota_action": "raise"  # "raise", "notify", "fallback"
    },
    "anthropic": {
        "max_retries": 3,
        "base_retry_delay": 30,
        "insufficient_quota_action": "raise"
    }
}
```

**Example:**
```python
# Aggressive retry strategy
aggressive = ErrorHandlingConfig(
    auto_retry_on_rate_limits=True,
    max_rate_limit_retries=10,
    timeout_retry_enabled=True,
    pool_retry_attempts=5
)

# Conservative strategy
conservative = ErrorHandlingConfig(
    auto_retry_on_rate_limits=False,
    max_rate_limit_retries=0,
    timeout_retry_enabled=False,
    enable_error_routing=True  # Route to user
)

# Custom provider settings
custom = ErrorHandlingConfig(
    provider_settings={
        "openai": {"max_retries": 5, "base_retry_delay": 30},
        "anthropic": {"max_retries": 3, "base_retry_delay": 60}
    }
)
```

---

## 🎨 Configuration Patterns

### Development Configuration

```python
def create_dev_config() -> ExecutionConfig:
    """Development configuration with verbose output."""
    return ExecutionConfig(
        status=StatusConfig(
            enabled=True,
            verbosity=VerbosityLevel.VERBOSE,
            show_thoughts=True,
            show_tool_calls=True
        ),
        step_timeout=30.0,
        steering_mode="always"
    )
```

### Production Configuration

```python
def create_prod_config() -> ExecutionConfig:
    """Production configuration with minimal output."""
    return ExecutionConfig(
        status=StatusConfig.from_verbosity(0),
        step_timeout=600.0,
        convergence_timeout=1200.0,
        steering_mode="auto",
        error_handling=ErrorHandlingConfig(
            auto_retry_on_rate_limits=True,
            max_rate_limit_retries=5
        )
    )
```

### Testing Configuration

```python
def create_test_config() -> ExecutionConfig:
    """Testing configuration with fast failure."""
    return ExecutionConfig(
        status=StatusConfig.from_verbosity(1),
        step_timeout=5.0,
        convergence_timeout=10.0,
        steering_mode="error"
    )
```

### Interactive Configuration

```python
def create_interactive_config() -> ExecutionConfig:
    """User-interactive configuration."""
    return ExecutionConfig(
        user_first=True,
        initial_user_msg="Hello! How can I help?",
        user_interaction_timeout=300.0,
        status=StatusConfig(
            enabled=True,
            cli_colors=True,
            show_agent_prefixes=True
        )
    )
```

---

## 🔧 Advanced Features

### Agent Lifecycle Management

The framework provides automatic agent cleanup to prevent resource leaks and registry collisions.

**How It Works:**

```python
# Default behavior - auto-cleanup enabled
result = await Orchestra.run(
    task="Analyze data",
    topology=topology,
    execution_config=ExecutionConfig(
        auto_cleanup_agents=True  # Default
    )
)
# After run completes:
# 1. Closes model resources (aiohttp sessions, etc.)
# 2. Closes agent-specific resources (browsers, playwright, etc.)
# 3. Unregisters agents from registry (frees names for reuse)
```

**Cleanup Scopes:**

- `topology_nodes` (default): Cleanup all agents defined in the topology
- `used_agents`: Cleanup only agents that were actually invoked during execution

**When to Disable:**

```python
# Scenario 1: Multiple runs with same agents
config = ExecutionConfig(auto_cleanup_agents=False)

for task in tasks:
    result = await Orchestra.run(task, topology, execution_config=config)
    # Agents stay alive, registry intact

# Scenario 2: Long-lived agent pools
pool = await create_browser_agent_pool(num_instances=3)
AgentRegistry.register_instance(pool, "BrowserPool")

# Pool manages its own lifecycle
result = await Orchestra.run(
    task="Scrape websites",
    topology=topology,
    execution_config=ExecutionConfig(auto_cleanup_agents=False)
)

await pool.cleanup()  # Manual cleanup when done
```

**Benefits:**

- **No Resource Leaks**: aiohttp sessions, browser instances closed deterministically
- **No Registry Collisions**: Agent names freed for reuse in subsequent runs
- **No Manual Cleanup**: Works "out of the box" without user intervention
- **Identity-Safe**: Uses instance identity checks to prevent race conditions

**Low-Level Control:**

```python
from marsys.agents.registry import AgentRegistry

# Manual cleanup for specific agent
agent = AgentRegistry.get("my_agent")
if agent:
    await agent.cleanup()  # Close resources
    AgentRegistry.unregister_if_same("my_agent", agent)  # Identity-safe unregister
```

---

### Configuration Merging

```python
from dataclasses import replace

# Base configuration
base_config = ExecutionConfig(
    step_timeout=300.0,
    status=StatusConfig.from_verbosity(1)
)

# Override specific settings
custom_config = replace(
    base_config,
    step_timeout=60.0,
    steering_mode="always"
)
```

### Dynamic Configuration

```python
def get_config_for_task(task_type: str) -> ExecutionConfig:
    """Get configuration based on task type."""
    configs = {
        "research": ExecutionConfig(
            convergence_timeout=600.0,
            max_steps=200
        ),
        "quick_query": ExecutionConfig(
            step_timeout=30.0,
            max_steps=10
        )
    }
    return configs.get(task_type, ExecutionConfig())
```

### Environment Variable Integration

```python
import os
from dotenv import load_dotenv

load_dotenv()

config = ExecutionConfig(
    step_timeout=float(os.getenv("MARSYS_TIMEOUT", "300")),
    status=StatusConfig(
        verbosity=int(os.getenv("MARSYS_VERBOSITY", "1"))
    )
)
```

---

## 🔄 Configuration Lifecycle

### Configuration Priority

1. **Explicit parameters** (highest priority)
2. **Configuration objects**
3. **Environment variables**
4. **Default values** (lowest priority)

### Runtime Updates

```python
# Start with base config
orchestra = Orchestra(execution_config=base_config)

# Override for specific run
result = await orchestra.execute(
    task=task,
    topology=topology,
    execution_config=ExecutionConfig(
        step_timeout=60.0  # Override just this setting
    )
)
```

---

## 📋 Best Practices

### ✅ DO:
- Start with defaults and override as needed
- Use verbosity levels appropriately
- Configure timeouts based on expected task duration
- Enable error routing for critical workflows
- Use provider-specific settings for API optimization

### ❌ DON'T:
- Set timeouts too short (< 10 seconds)
- Disable all retries in production
- Use VERBOSE mode in production
- Ignore error handling configuration
- Mix incompatible settings (e.g., retries with "never" steering)

---

## 🚦 Related Documentation

- [Orchestra API](orchestra.md) - Main orchestration interface
- [Execution API](execution.md) - Execution system using configs
- [Status System](../concepts/monitoring.md) - Status and monitoring details
- [Error Handling](../concepts/error-handling.md) - Error recovery patterns

---

!!! tip "Pro Tip"
    Use `StatusConfig.from_verbosity()` for quick setup. It automatically configures all related settings appropriately for the chosen verbosity level.

!!! warning "Configuration Validation"
    Always validate custom configurations, especially when merging multiple sources. Invalid configurations can cause runtime failures.
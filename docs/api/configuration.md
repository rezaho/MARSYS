!!! warning "Updated for v0.3.0 — code examples below may use legacy formats"
    The JSON `{"next_action": "..."}` response format shown in some examples on
    this page was **removed** in commit `bc19b98` (no shim). Coordination now
    uses native tool calls: `invoke_agent`, `terminate_workflow`, `ask_user`,
    `end_conversation`. See [Coordination Tools](../concepts/coordination-tools.md)
    for the canonical reference, and [ADR-006](../architecture/framework/decisions/ADR-006-deprecation-timeline.md)
    for the full v0.2.x → v0.3.0 migration table. The conceptual content on
    this page is otherwise still accurate.

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

    # Response format
    response_format: str = "json",

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
| `response_format` | `str` | Response format for agent outputs | `"json"` |
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
    QUIET = 0     # Minimal output (no thinking, no tool calls)
    NORMAL = 1    # Standard output (agent thinking, tool calls with reasoning)
    VERBOSE = 2   # Detailed output (all of above + tool arguments, completion timings)
```

**What Shows at Each Level:**

| Event Type | QUIET | NORMAL | VERBOSE |
|------------|-------|--------|---------|
| Agent start/complete | ✓ | ✓ | ✓ |
| Agent thinking | ✗ | ✓ | ✓ |
| Tool call name | ✗ | ✓ | ✓ |
| Tool reasoning | ✗ | ✓ | ✓ |
| Tool arguments | ✗ | ✗ | ✓ |
| Tool completion | ✗ | ✗ | ✓ |
| Action type details | ✗ | ✗ | ✓ |

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

Retry, backoff, and error-handling configuration. Consumed by:

- the model-adapter retry loop in [`models/adapters/base.py`](https://github.com/rezaho/MARS/blob/main/packages/framework/src/marsys/models/adapters/base.py) — wraps every API call with exponential-backoff-with-jitter and records per-attempt history on the response;
- the framework-level retry loop in `StepExecutor` — retries `AgentFrameworkError` failures with the same backoff math.

`ExecutionConfig` carries an `error_handling: ErrorHandlingConfig` field by default; override it to tune retry behaviour.

**Import:**
```python
from marsys.coordination.config import ErrorHandlingConfig
```

**Constructor:**
```python
ErrorHandlingConfig(
    # Retry / backoff (top-level)
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.1,
    max_delay: float = 60.0,

    # Classification / routing flags
    use_error_classification: bool = True,
    notify_on_critical_errors: bool = True,
    enable_error_routing: bool = True,
    preserve_error_context: bool = True,

    # Steering-related retry knobs (kept for back-compat)
    auto_retry_on_rate_limits: bool = True,
    max_rate_limit_retries: int = 3,
    pool_retry_attempts: int = 2,
    pool_retry_delay: float = 5.0,

    # Timeouts
    timeout_seconds: float = 600.0,
    timeout_retry_enabled: bool = False,

    # Per-provider overrides
    provider_settings: Dict[str, Dict[str, Any]] = ...,
)
```

**Top-level retry parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_retries` | `int` | Maximum retry attempts after the initial call | `3` |
| `base_delay` | `float` | Starting delay (seconds) for exponential backoff | `1.0` |
| `jitter` | `float` | Symmetric multiplicative jitter, range `[0, 1]` (e.g. `0.1` = ±10%) | `0.1` |
| `max_delay` | `float` | Cap on per-attempt sleep (seconds) | `60.0` |

**Backoff formula:**
```
delay = min(
    base_delay * (2 ** attempt) * (1 + uniform(-jitter, jitter)),
    max_delay,
)
```

A server-supplied `retry-after` / `x-ratelimit-reset-after` header always wins over the computed delay (capped at `max_delay`).

**Other parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `use_error_classification` | `bool` | Enable intelligent error classification | `True` |
| `notify_on_critical_errors` | `bool` | Send notifications for critical errors | `True` |
| `enable_error_routing` | `bool` | Route errors to User node for intervention | `True` |
| `preserve_error_context` | `bool` | Include full error context in responses | `True` |
| `auto_retry_on_rate_limits` | `bool` | (Steering) automatically retry rate-limited requests | `True` |
| `max_rate_limit_retries` | `int` | (Steering) maximum retries for rate limit errors | `3` |
| `pool_retry_attempts` | `int` | (Steering) retries for pool exhaustion | `2` |
| `pool_retry_delay` | `float` | (Steering) delay between pool retry attempts | `5.0` |
| `timeout_seconds` | `float` | Default timeout for operations | `600.0` |
| `timeout_retry_enabled` | `bool` | Retry after timeout | `False` |
| `provider_settings` | `Dict[str, Dict[str, Any]]` | Per-provider overrides | populated for `openai`, `anthropic`, `google`, `openrouter`, `xai` |

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `resolve_max_retries(provider)` | `max_retries` for ``provider`` (override-aware) | `int` |
| `resolve_base_delay(provider)` | `base_delay` for ``provider`` (override-aware) | `float` |
| `compute_delay(provider, attempt)` | Compute delay for `attempt` (0-indexed); applies jitter, caps at `max_delay` | `float` |
| `get_provider_setting(provider, setting, default)` | Raw provider-setting lookup | `Any` |

**Provider settings structure:**
```python
provider_settings = {
    "openai":     {"max_retries": 3, "base_delay": 1.0, "insufficient_quota_action": "raise"},
    "anthropic":  {"max_retries": 3, "base_delay": 1.0, "insufficient_quota_action": "raise"},
    "google":     {"max_retries": 3, "base_delay": 1.0, "insufficient_quota_action": "notify"},
    "openrouter": {"max_retries": 2, "base_delay": 1.0, "insufficient_quota_action": "raise"},
    "xai":        {"max_retries": 2, "base_delay": 2.0, "insufficient_quota_action": "notify"},
}
```

The provider key is derived from the adapter class name (e.g. `OpenAIAdapter` → `"openai"`, `AsyncAnthropicAdapter` → `"anthropic"`). When a provider isn't in the dict, the top-level fields apply.

**Examples:**

```python
# Default behaviour: 3 retries, 1s base, ±10% jitter, 60s cap.
default = ErrorHandlingConfig()

# Aggressive retries for flaky free tiers.
aggressive = ErrorHandlingConfig(
    max_retries=8,
    base_delay=2.0,
    jitter=0.2,
    max_delay=120.0,
)

# Conservative — fail fast.
conservative = ErrorHandlingConfig(
    max_retries=1,
    base_delay=0.5,
    enable_error_routing=True,  # Route to user instead of retrying further
)

# Per-provider override: very long backoff on xai free tier.
custom = ErrorHandlingConfig()
custom.provider_settings["xai"]["base_delay"] = 10.0
custom.provider_settings["xai"]["max_retries"] = 1

# Wired into ExecutionConfig.
ec = ExecutionConfig(error_handling=aggressive)
```

**Per-attempt retry history:** when more than one attempt is made, the harmonized response carries a `retry_attempts` list on `ResponseMetadata` (extra-allowed Pydantic field). Each entry records `attempt`, `success`, `status_code`, `delay_used`, `retry_after_used`, `error_class`, `error_message`, `response_time_ms`. Used today by tracing for fine-grained replay and reserved for future per-attempt span emission.

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
pool = await create_browser_agent_pool(num_instances=3, register=False)
AgentRegistry.register_pool(pool)

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

### Response Format Configuration

The `response_format` parameter controls how agents structure their outputs and how the system builds agent system prompts.

**Available Formats:**

| Format | Description |
|--------|-------------|
| `json` | Default JSON format with `next_action`/`action_input` structure |

**How It Works:**

1. **System Prompt Building**: The format determines how coordination instructions are included in agent system prompts
2. **Response Parsing**: Each format has an associated processor for parsing agent responses
3. **Parallel Invocation Examples**: Format-specific examples guide agents on correct parallel invocation patterns

**Example:**
```python
# Default JSON format (recommended)
config = ExecutionConfig(response_format="json")

# The JSON format produces system prompts with instructions like:
# --- STRICT JSON OUTPUT FORMAT ---
# Your response MUST be a single, valid JSON object...
# ```json
# {
#   "thought": "reasoning...",
#   "next_action": "invoke_agent",
#   "action_input": [...]
# }
# ```
```

**Custom Formats:**

To implement a custom format (e.g., XML):

```python
from marsys.coordination.formats import BaseResponseFormat, register_format

class XMLResponseFormat(BaseResponseFormat):
    def get_format_name(self) -> str:
        return "xml"

    def build_format_instructions(self, actions, descriptions) -> str:
        # Build XML-specific instructions
        ...

    def get_parallel_invocation_examples(self, context) -> str:
        # Provide XML examples for parallel invocation
        ...

    def create_processor(self):
        return XMLProcessor()

# Register and use
register_format("xml", XMLResponseFormat)
config = ExecutionConfig(response_format="xml")
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
- [Validation API](validation.md) - Response format system and processors
- [Status System](../concepts/monitoring.md) - Status and monitoring details
- [Error Handling](../concepts/error-handling.md) - Error recovery patterns

---

!!! tip "Pro Tip"
    Use `StatusConfig.from_verbosity()` for quick setup. It automatically configures all related settings appropriately for the chosen verbosity level.

!!! warning "Configuration Validation"
    Always validate custom configurations, especially when merging multiple sources. Invalid configurations can cause runtime failures.

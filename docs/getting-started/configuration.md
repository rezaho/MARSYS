# Configuration

Master MARSYS configuration to fine-tune execution behavior, timeouts, status management, and more.

## 🎯 Overview

MARSYS provides comprehensive configuration at multiple levels:

- **Model Configuration**: Provider settings, API keys, parameters
- **Agent Configuration**: Tools, memory, response formats
- **Execution Configuration**: Timeouts, retries, convergence behavior
- **Status Configuration**: Verbosity, output formatting, channels
- **Communication Configuration**: User interaction, rich formatting

## 🔑 Environment Variables

### Basic Setup

Create a `.env` file in your project root:

```bash
# .env

# API Keys (at least one required)
OPENAI_API_KEY="sk-..."
ANTHROPIC_API_KEY="sk-ant-..."
GOOGLE_API_KEY="AIza..."
GROQ_API_KEY="gsk_..."

# Optional Configuration
HEADLESS=true              # Browser automation mode
LOG_LEVEL=INFO            # Logging verbosity
MAX_RETRIES=3             # API retry attempts
TIMEOUT=300               # Default timeout in seconds
```

### Advanced Environment Variables

```bash
# Model-specific settings
OPENAI_ORG_ID="org-..."
OPENAI_BASE_URL="https://api.openai.com/v1"
ANTHROPIC_VERSION="2023-06-01"

# Browser automation
PLAYWRIGHT_BROWSERS_PATH="/path/to/browsers"
PLAYWRIGHT_TIMEOUT=30000

# System resources
MAX_WORKERS=4
MEMORY_LIMIT_MB=2048
DISK_CACHE_PATH="/tmp/marsys_cache"

# Monitoring
ENABLE_TELEMETRY=false
METRICS_PORT=9090
TRACE_LEVEL=ERROR
```

## 🤖 Model Configuration

### ModelConfig Class

The core configuration class for all models:

```python
from marsys.models import ModelConfig

config = ModelConfig(
    type="api",                    # "api" or "local"
    provider="openrouter",         # Provider name
    name="anthropic/claude-haiku-4.5",  # Model name
    api_key=None,                  # Auto-loads from env
    base_url=None,                 # Custom endpoint
    parameters={                   # Model parameters
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5
    }
)
```

### Provider-Specific Configurations

Common provider examples:

```python
# OpenRouter (recommended - access to all models)
config = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-haiku-4.5",  # or gpt-5, gemini-2.5-pro, etc.
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7,
    max_tokens=12000
)

# Local Ollama
config = ModelConfig(
    type="local",
    provider="ollama",
    name="llama2:13b",
    base_url="http://localhost:11434"
)
```

!!! info "Provider-Specific Parameters"
    Each provider supports unique parameters (reasoning_effort, thinking_budget, safety_settings, etc.).
    See [Models API Reference](../api/models.md) for complete provider-specific documentation.

## ⚙️ Execution Configuration

### ExecutionConfig

Fine-tune how Orchestra executes workflows:

```python
from marsys.coordination.config import ExecutionConfig, StatusConfig, VerbosityLevel

config = ExecutionConfig(
    # Timeout settings (seconds)
    convergence_timeout=300.0,         # Max wait for parallel branches
    branch_timeout=600.0,              # Max time per branch
    agent_acquisition_timeout=240.0,   # Max wait to acquire from pool
    step_timeout=300.0,                # Max time per step
    tool_execution_timeout=60.0,       # Max time for tool calls
    user_interaction_timeout=300.0,    # Max wait for user input

    # Convergence behavior
    dynamic_convergence_enabled=True,  # Auto-detect convergence points
    parent_completes_on_spawn=True,    # Parent waits for children
    auto_detect_convergence=True,      # Automatic convergence detection

    # Retry and steering
    steering_mode="auto",              # "auto", "always", "never"
    max_retries=3,                     # Retry attempts per step
    retry_delay=1.0,                   # Delay between retries
    exponential_backoff=True,          # Exponential retry delay

    # Status and output
    status=StatusConfig.from_verbosity(VerbosityLevel.NORMAL),

    # User interaction
    user_interaction="terminal",       # "terminal", "none", "async"
    user_first=False,                  # Show initial message to user
    initial_user_msg=None,            # Custom initial message
)
```

### Timeout Configuration

Different timeout levels for different scenarios:

```python
# Quick tasks - tight timeouts
quick_config = ExecutionConfig(
    step_timeout=30.0,
    convergence_timeout=60.0,
    branch_timeout=120.0
)

# Long-running research - relaxed timeouts
research_config = ExecutionConfig(
    step_timeout=300.0,
    convergence_timeout=600.0,
    branch_timeout=1800.0,
    user_interaction_timeout=600.0
)

# Real-time systems - strict timeouts
realtime_config = ExecutionConfig(
    step_timeout=10.0,
    convergence_timeout=30.0,
    branch_timeout=60.0,
    steering_mode="never"  # No retries for speed
)
```

## 📊 Status Configuration

### StatusConfig

Control output verbosity and formatting:

```python
from marsys.coordination.config import StatusConfig, VerbosityLevel

# Quick setup with verbosity levels
status = StatusConfig.from_verbosity(VerbosityLevel.QUIET)    # Minimal output
status = StatusConfig.from_verbosity(VerbosityLevel.NORMAL)   # Standard output
status = StatusConfig.from_verbosity(VerbosityLevel.VERBOSE)  # Detailed output

# Detailed configuration
status = StatusConfig(
    enabled=True,
    verbosity=VerbosityLevel.NORMAL,

    # Output control
    cli_output=True,                  # Show CLI output
    cli_colors=True,                  # Use colors
    show_thoughts=False,              # Show agent thoughts
    show_tool_calls=True,             # Show tool invocations
    show_timings=True,                # Show execution times

    # Aggregation
    aggregation_window_ms=500,        # Group updates within window
    aggregate_parallel=True,          # Aggregate parallel branches

    # Display formatting
    show_agent_prefixes=True,         # Show agent names
    prefix_width=20,                  # Width for agent names
    prefix_alignment="left",          # "left", "right", "center"

    # Output channels
    channels=["cli", "file"],         # Output destinations
    file_path="execution.log",        # Log file path

    # Progress indicators
    show_progress_bar=True,           # Show progress
    show_step_counter=True,           # Show step numbers
    show_branch_indicators=True,      # Show branch status
)
```

### Verbosity Levels Explained

| Level | Description | Use Case |
|-------|-------------|----------|
| `QUIET` (0) | Minimal output, errors only | Production, CI/CD |
| `NORMAL` (1) | Standard output with key events | Development |
| `VERBOSE` (2) | Detailed output with all events | Debugging |

## 💬 Communication Configuration

### CommunicationConfig

Configure user interaction and formatting:

```python
from marsys.coordination.config import CommunicationConfig

comm_config = CommunicationConfig(
    # Rich formatting
    use_rich_formatting=True,          # Use rich terminal features
    theme_name="modern",               # "modern", "classic", "minimal"

    # Display settings
    prefix_width=20,                  # Agent name column width
    show_timestamps=True,             # Show message timestamps
    timestamp_format="%H:%M:%S",      # Time format

    # History
    enable_history=True,              # Keep conversation history
    history_size=1000,                # Max history entries

    # Interactive features
    enable_tab_completion=True,       # Tab completion for commands
    use_colors=True,                  # Terminal colors
    color_depth="truecolor",          # "truecolor", "256", "16", "none"

    # Input handling
    input_timeout=None,               # No timeout by default
    multiline_input=True,             # Support multi-line input

    # Terminal enhancement
    use_enhanced_terminal=True,       # Use enhanced terminal features
    fallback_on_error=True,          # Fallback to basic on error
)
```

## 🔧 Error Handling Configuration

### ErrorHandlingConfig

Configure error recovery strategies:

```python
from marsys.coordination.config import ErrorHandlingConfig

error_config = ErrorHandlingConfig(
    # Classification and routing
    use_error_classification=True,    # Classify error types
    enable_error_routing=True,        # Route errors to User node
    preserve_error_context=True,      # Keep error context

    # Notifications
    notify_on_critical_errors=True,   # Alert on critical errors
    notification_channels=["cli", "log"],

    # Retry behavior
    auto_retry_on_rate_limits=True,   # Auto-retry rate limits
    max_rate_limit_retries=3,
    rate_limit_backoff=60,            # Seconds

    # Pool-specific
    pool_retry_attempts=2,            # Retries for pool acquisition
    pool_retry_delay=5.0,             # Delay between pool retries

    # Timeout handling
    timeout_seconds=300.0,            # Global timeout
    timeout_retry_enabled=False,      # Retry on timeout

    # Provider-specific settings
    provider_settings={
        "openai": {
            "max_retries": 3,
            "base_retry_delay": 60,
            "insufficient_quota_action": "raise",  # or "fallback"
            "fallback_model": "gpt-5-mini"
        },
        "anthropic": {
            "max_retries": 2,
            "base_retry_delay": 30,
            "insufficient_quota_action": "fallback",
            "fallback_model": "anthropic/claude-haiku-4.5"
        },
        "google": {
            "max_retries": 3,
            "base_retry_delay": 45,
            "insufficient_quota_action": "raise"
        }
    }
)
```

## 🎛️ Complete Configuration Example

Here's a comprehensive configuration for a production system:

```python
import os
from marsys.coordination import Orchestra
from marsys.coordination.config import (
    ExecutionConfig,
    StatusConfig,
    CommunicationConfig,
    ErrorHandlingConfig,
    VerbosityLevel
)
from marsys.coordination.state import StateManager, FileStorageBackend
from pathlib import Path

# Create comprehensive configuration
def create_production_config():
    """Create production-ready configuration."""

    # Execution configuration
    exec_config = ExecutionConfig(
        # Balanced timeouts
        convergence_timeout=300.0,
        branch_timeout=600.0,
        step_timeout=120.0,
        tool_execution_timeout=30.0,
        user_interaction_timeout=300.0,

        # Enable smart features
        dynamic_convergence_enabled=True,
        auto_detect_convergence=True,
        steering_mode="auto",

        # Retry with exponential backoff
        max_retries=3,
        retry_delay=2.0,
        exponential_backoff=True,

        # Status for production
        status=StatusConfig(
            enabled=True,
            verbosity=VerbosityLevel.NORMAL,
            cli_output=True,
            cli_colors=True,
            show_timings=True,
            show_tool_calls=False,  # Reduce noise
            show_thoughts=False,    # Reduce noise
            aggregate_parallel=True,
            channels=["cli", "file"],
            file_path="logs/execution.log"
        ),

        # User interaction
        user_interaction="terminal",
        user_first=False
    )

    # Communication configuration
    comm_config = CommunicationConfig(
        use_rich_formatting=True,
        theme_name="modern",
        show_timestamps=True,
        enable_history=True,
        history_size=1000,
        use_enhanced_terminal=True,
        fallback_on_error=True
    )

    # Error handling
    error_config = ErrorHandlingConfig(
        use_error_classification=True,
        enable_error_routing=True,
        notify_on_critical_errors=True,
        auto_retry_on_rate_limits=True,
        max_rate_limit_retries=5,
        rate_limit_backoff=60,
        provider_settings={
            "openai": {
                "max_retries": 3,
                "base_retry_delay": 60,
                "insufficient_quota_action": "fallback",
                "fallback_model": "gpt-5-mini"
            }
        }
    )

    return exec_config, comm_config, error_config

# Use configuration
async def run_with_config():
    exec_config, comm_config, error_config = create_production_config()

    # State persistence
    storage = FileStorageBackend(Path("./state"))
    state_manager = StateManager(storage)

    # Run with full configuration
    result = await Orchestra.run(
        task="Complex multi-agent task",
        topology=topology,
        execution_config=exec_config,
        communication_config=comm_config,
        error_config=error_config,
        state_manager=state_manager,
        max_steps=50
    )

    return result
```

## 🎯 Configuration Patterns

### Pattern 1: Development Configuration
```python
# Maximum visibility for debugging
dev_config = ExecutionConfig(
    status=StatusConfig.from_verbosity(VerbosityLevel.VERBOSE),
    steering_mode="always",  # Always retry
    user_interaction="terminal"
)
```

### Pattern 2: Production Configuration
```python
# Balanced for reliability
prod_config = ExecutionConfig(
    status=StatusConfig.from_verbosity(VerbosityLevel.QUIET),
    steering_mode="auto",
    max_retries=5,
    exponential_backoff=True
)
```

### Pattern 3: Real-time Configuration
```python
# Optimized for speed
realtime_config = ExecutionConfig(
    step_timeout=10.0,
    convergence_timeout=30.0,
    steering_mode="never",  # No retries
    status=StatusConfig(enabled=False)  # No output overhead
)
```

### Pattern 4: Long-running Configuration
```python
# For multi-hour workflows
long_config = ExecutionConfig(
    branch_timeout=3600.0,  # 1 hour
    convergence_timeout=1800.0,  # 30 minutes
    user_interaction_timeout=900.0,  # 15 minutes
    dynamic_convergence_enabled=True
)
```

!!! tip "Advanced Configuration Topics"
    For production deployment, monitoring, security, and advanced features, see:

    - [Configuration API Reference](../api/configuration.md) - Complete parameter documentation
    - [Guides](../guides/overview.md) - Production deployment patterns

## 🚦 Next Steps

With configuration mastered:

<div class="grid cards" markdown="1">

- :material-graph:{ .lg .middle } **[Learn Topologies](../concepts/advanced/topology/)**

    ---

    Design agent interaction patterns

- :material-book-open:{ .lg .middle } **[Understand Concepts](../concepts/)**

    ---

    Explore framework architecture

- :material-code-tags:{ .lg .middle } **[See Examples](../use-cases/)**

    ---

    Learn from real implementations

- :material-api:{ .lg .middle } **[API Reference](../api/)**

    ---

    Detailed API documentation

</div>

---

!!! success "Configuration Complete!"
    You now understand MARSYS configuration! Explore [Core Concepts](../concepts/) to understand the framework architecture.
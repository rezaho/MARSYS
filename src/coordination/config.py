"""
Configuration classes for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any
from enum import IntEnum


class VerbosityLevel(IntEnum):
    """Verbosity levels for status output."""
    QUIET = 0     # Minimal output
    NORMAL = 1    # Standard output
    VERBOSE = 2   # Detailed output


@dataclass
class StatusConfig:
    """Configuration for status updates."""
    enabled: bool = False  # Opt-in by default
    verbosity: Optional[VerbosityLevel] = None  # None means use default

    # Output configuration
    cli_output: bool = True
    cli_colors: bool = True
    show_thoughts: bool = True
    show_tool_calls: bool = True
    show_timings: bool = True

    # Aggregation settings
    aggregation_window_ms: int = 500
    aggregate_parallel: bool = True

    # Memory management
    max_events_per_session: int = 10000
    session_cleanup_after_s: int = 3600  # 1 hour

    # Channel configuration
    channels: List[str] = field(default_factory=lambda: ["cli"])

    # Prefixed display configuration
    show_agent_prefixes: bool = True     # Enable agent name prefixes
    prefix_width: int = 20                # Width of agent name field
    prefix_alignment: str = "left"        # left, center, or right

    # Note: User interaction configuration moved to AutoRunConfig
    # follow_up_timeout kept for backward compatibility
    follow_up_timeout: float = 30.0  # seconds to wait for follow-up

    @classmethod
    def from_verbosity(cls, level: int) -> 'StatusConfig':
        """Create StatusConfig from verbosity level."""
        if level == VerbosityLevel.QUIET:
            return cls(
                enabled=True,
                verbosity=VerbosityLevel.QUIET,
                cli_output=True,
                cli_colors=False,
                show_thoughts=False,
                show_tool_calls=False,
                show_timings=False,
                aggregate_parallel=True
            )
        elif level == VerbosityLevel.NORMAL:
            return cls(
                enabled=True,
                verbosity=VerbosityLevel.NORMAL,
                cli_output=True,
                cli_colors=True,
                show_thoughts=False,
                show_tool_calls=False,
                show_timings=True,
                aggregate_parallel=True
            )
        elif level == VerbosityLevel.VERBOSE:
            return cls(
                enabled=True,
                verbosity=VerbosityLevel.VERBOSE,
                cli_output=True,
                cli_colors=True,
                show_thoughts=True,
                show_tool_calls=True,
                show_timings=True,
                aggregate_parallel=False  # Show all details
            )
        else:
            # Default to NORMAL for unknown levels
            return cls.from_verbosity(VerbosityLevel.NORMAL)

    def should_show_event(self, event_type: str) -> bool:
        """Determine if an event type should be shown based on verbosity."""
        if self.verbosity == VerbosityLevel.QUIET:
            return event_type in ["completion", "error", "final_response"]
        elif self.verbosity == VerbosityLevel.NORMAL:
            return event_type not in ["thought", "tool_call"]
        else:  # VERBOSE
            return True


@dataclass
class CommunicationConfig:
    """Configuration for enhanced communication channels."""

    # Visual settings
    use_rich_formatting: bool = True
    theme_name: str = "modern"  # modern, classic, minimal
    prefix_width: int = 20
    show_timestamps: bool = True

    # Input settings
    enable_history: bool = True
    history_size: int = 1000
    enable_tab_completion: bool = True

    # Color settings
    use_colors: bool = True
    color_depth: str = "truecolor"  # truecolor, 256, 16, none

    # Behavior
    input_timeout: Optional[float] = None
    show_typing_indicator: bool = False  # Future enhancement

    # Channel selection
    use_enhanced_terminal: bool = True  # Use EnhancedTerminalChannel instead of TerminalChannel
    fallback_on_error: bool = True  # Fall back to basic TerminalChannel if Rich fails


@dataclass
class ExecutionConfig:
    """Configuration for multi-agent system execution."""

    # Steering configuration
    steering_mode: Literal["auto", "always", "never"] = "auto"

    # Convergence behavior
    dynamic_convergence_enabled: bool = True
    parent_completes_on_spawn: bool = True
    auto_detect_convergence: bool = True  # Automatically mark exit nodes and parents as convergence

    # Timeouts
    convergence_timeout: float = 300.0
    branch_timeout: float = 600.0

    # Status configuration
    status: StatusConfig = field(default_factory=StatusConfig)
    
    def should_apply_steering(self, is_retry: bool = False) -> bool:
        """
        Determine if steering should be applied.
        
        Args:
            is_retry: Whether this is a retry attempt

        Returns:
            True if steering should be applied, False otherwise
        """
        if self.steering_mode == "never":
            return False
        elif self.steering_mode == "always":
            return True
        else:  # auto mode - only on retries
            return is_retry


@dataclass
class ErrorHandlingConfig:
    """
    Configuration for enhanced error handling system.

    Attributes:
        use_error_classification: Enable intelligent error classification
        notify_on_critical_errors: Send notifications for critical errors
        auto_retry_on_rate_limits: Automatically retry rate-limited requests
        max_rate_limit_retries: Maximum retries for rate limit errors
        pool_retry_attempts: Number of retries for pool exhaustion
        pool_retry_delay: Delay between pool retry attempts (seconds)
        timeout_seconds: Default timeout for operations
        enable_error_routing: Route errors to User node for intervention
        preserve_error_context: Include full error context in responses
    """

    # Core features
    use_error_classification: bool = True
    notify_on_critical_errors: bool = True
    enable_error_routing: bool = True
    preserve_error_context: bool = True

    # Retry configuration
    auto_retry_on_rate_limits: bool = True
    max_rate_limit_retries: int = 3
    pool_retry_attempts: int = 2
    pool_retry_delay: float = 5.0

    # Timeout configuration
    timeout_seconds: float = 300.0  # 5 minutes default
    timeout_retry_enabled: bool = False

    # Provider-specific settings
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "openai": {
            "max_retries": 3,
            "base_retry_delay": 60,
            "insufficient_quota_action": "raise"  # "raise", "notify", "fallback"
        },
        "anthropic": {
            "max_retries": 3,
            "base_retry_delay": 30,
            "insufficient_quota_action": "raise"
        },
        "google": {
            "max_retries": 3,
            "base_retry_delay": 60,
            "insufficient_quota_action": "notify"
        },
        "openrouter": {
            "max_retries": 2,
            "base_retry_delay": 120,
            "insufficient_quota_action": "raise"
        },
        "xai": {
            "max_retries": 2,
            "base_retry_delay": 120,  # xAI has strict rate limits
            "insufficient_quota_action": "notify"
        }
    })

    def get_provider_setting(
        self,
        provider: str,
        setting: str,
        default: Any = None
    ) -> Any:
        """Get a specific setting for a provider."""
        provider_config = self.provider_settings.get(provider, {})
        return provider_config.get(setting, default)
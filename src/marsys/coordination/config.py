"""
Configuration classes for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Union
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
class ConvergencePolicyConfig:
    """
    Policy for handling convergence timeout in parallel branch execution.

    When a parallel group times out (exceeds convergence_timeout), this policy
    determines whether to proceed with partial results or fail the workflow.

    Attributes:
        min_ratio: Minimum fraction of branches that must reach convergence (0.0-1.0)
                   Example: 0.67 means at least 2/3 branches must converge
        on_insufficient: Action when min_ratio not met
                        - "proceed": Continue anyway (risky, may have incomplete data)
                        - "fail": Raise WorkflowTimeoutError
                        - "user": Ask user for confirmation (interactive mode only)
        terminate_orphans: If True, cancel branches that didn't reach convergence
                          Prevents wasted computation on orphaned work
        log_level: Logging level for timeout events ("info", "warning", "error")
    """

    min_ratio: float = 0.67  # Require 2/3 of branches by default
    on_insufficient: Literal["proceed", "fail", "user"] = "fail"
    terminate_orphans: bool = True
    log_level: Literal["info", "warning", "error"] = "warning"

    @classmethod
    def from_value(cls, value: Union[float, str, 'ConvergencePolicyConfig']) -> 'ConvergencePolicyConfig':
        """
        Create ConvergencePolicyConfig from flexible input.

        Args:
            value: Configuration value in one of three forms:
                   - float (0.0-1.0): Minimum convergence ratio, fail if not met
                   - str: Named policy ("strict", "majority", "fail", "user", "any")
                   - ConvergencePolicyConfig: Return as-is

        Returns:
            ConvergencePolicyConfig object

        Examples:
            >>> ConvergencePolicyConfig.from_value(0.67)
            ConvergencePolicyConfig(min_ratio=0.67, on_insufficient="fail", ...)

            >>> ConvergencePolicyConfig.from_value("strict")
            ConvergencePolicyConfig(min_ratio=1.0, on_insufficient="fail", ...)

            >>> ConvergencePolicyConfig.from_value("user")
            ConvergencePolicyConfig(min_ratio=0.67, on_insufficient="user", ...)
        """
        # Already a config object
        if isinstance(value, cls):
            return value

        # Float: minimum ratio with fail on insufficient
        if isinstance(value, (int, float)):
            ratio = float(value)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"Convergence ratio must be between 0.0 and 1.0, got {ratio}")
            return cls(
                min_ratio=ratio,
                on_insufficient="fail",
                terminate_orphans=True
            )

        # String: named policy
        if isinstance(value, str):
            if value == "strict":
                return cls(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)
            elif value == "majority":
                return cls(min_ratio=0.51, on_insufficient="fail", terminate_orphans=True)
            elif value == "fail":
                return cls(min_ratio=1.0, on_insufficient="fail", terminate_orphans=True)
            elif value == "user":
                return cls(min_ratio=0.67, on_insufficient="user", terminate_orphans=True)
            elif value == "any":
                return cls(min_ratio=0.0, on_insufficient="proceed", terminate_orphans=True)
            else:
                raise ValueError(
                    f"Unknown convergence policy: '{value}'. "
                    f"Valid options: 'strict', 'majority', 'fail', 'user', 'any', or float 0.0-1.0"
                )

        raise TypeError(f"Invalid convergence_policy type: {type(value)}")

    def describe(self) -> str:
        """Human-readable description of this policy."""
        threshold = f"{self.min_ratio:.0%}"
        action = {
            "proceed": "continue anyway",
            "fail": "fail workflow",
            "user": "ask user"
        }[self.on_insufficient]
        termination = "terminate orphans" if self.terminate_orphans else "let orphans complete"

        return (
            f"Require {threshold} of branches at convergence. "
            f"If insufficient: {action}. "
            f"On timeout: {termination}."
        )


@dataclass
class ExecutionConfig:
    """Configuration for multi-agent system execution."""

    # Steering configuration
    # "error" = inject only when error occurred (minimum interference)
    # "auto" = inject on any retry
    # "always" = inject on every step
    steering_mode: Literal["auto", "always", "error"] = "error"

    # Convergence behavior
    dynamic_convergence_enabled: bool = True
    parent_completes_on_spawn: bool = True
    auto_detect_convergence: bool = True  # Automatically mark exit nodes and parents as convergence

    # Timeouts (in seconds)
    convergence_timeout: float = 300.0  # Waiting for children branches to complete
    convergence_policy: Union[float, str, ConvergencePolicyConfig] = 0.67  # What to do on timeout
    branch_timeout: float = 600.0  # Overall branch execution timeout
    agent_acquisition_timeout: float = 240.0  # Acquiring agent from pool
    step_timeout: float = 600.0  # Individual step execution timeout (10 minutes)
    tool_execution_timeout: float = 120.0  # Tool call execution timeout (2 minutes)
    user_interaction_timeout: float = 300.0  # Waiting for user input

    # Status configuration
    status: StatusConfig = field(default_factory=StatusConfig)

    # NEW: User interaction control fields (Orchestra only)
    user_first: bool = False  # Enable user-first execution mode
    initial_user_msg: Optional[str] = None  # Message shown to user in user-first mode
    user_interaction: str = "terminal"  # Type of user interaction: "terminal", "none"

    # Agent lifecycle management
    auto_cleanup_agents: bool = True  # Automatically cleanup agents after run (closes resources, unregisters)
    cleanup_scope: Literal["topology_nodes", "used_agents"] = "topology_nodes"  # Which agents to cleanup

    def should_apply_steering(self, is_retry: bool = False, has_error: bool = False) -> bool:
        """
        Determine if steering should be applied.

        Args:
            is_retry: Whether this is a retry attempt
            has_error: Whether an error occurred in previous attempt

        Returns:
            True if steering should be applied, False otherwise
        """
        if self.steering_mode == "error":
            return has_error
        elif self.steering_mode == "auto":
            return is_retry
        elif self.steering_mode == "always":
            return True

        return False


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
    timeout_seconds: float = 600.0  # 10 minutes default
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
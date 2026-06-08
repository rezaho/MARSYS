"""
Configuration classes for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Any, Union
from enum import IntEnum

from .tracing.config import TracingConfig
from .aggui.config import AGGUIConfig


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
                   Example: 1.0 means every branch must converge (default)
        on_insufficient: Action when min_ratio not met
                        - "proceed": Continue anyway (risky, may have incomplete data)
                        - "fail": Raise WorkflowTimeoutError
                        - "user": Ask user for confirmation (interactive mode only)
        terminate_orphans: If True, cancel branches that didn't reach convergence
                          Prevents wasted computation on orphaned work
        log_level: Logging level for timeout events ("info", "warning", "error")
    """

    min_ratio: float = 1.0  # Require all branches by default
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
            >>> ConvergencePolicyConfig.from_value(1.0)
            ConvergencePolicyConfig(min_ratio=1.0, on_insufficient="fail", ...)

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
    # REMOVE-IN-V0.4: auto-detection of convergence points; users will mark
    # convergence explicitly in their topology after v0.4.
    auto_detect_convergence: bool = True  # Automatically mark exit nodes and parents as convergence

    # Timeouts (in seconds)
    convergence_timeout: float = 300.0  # Waiting for children branches to complete
    convergence_policy: Union[float, str, ConvergencePolicyConfig] = 1.0  # Require full convergence by default
    branch_timeout: float = 600.0  # Overall branch execution timeout
    agent_acquisition_timeout: float = 240.0  # Acquiring agent from pool
    step_timeout: float = 600.0  # Individual step execution timeout (10 minutes)
    tool_execution_timeout: float = 120.0  # Tool call execution timeout (2 minutes)
    user_interaction_timeout: float = 300.0  # Waiting for user input

    # Status configuration
    status: StatusConfig = field(default_factory=StatusConfig)

    # Tracing configuration
    tracing: 'TracingConfig' = field(default_factory=lambda: TracingConfig())

    # AG-UI translator configuration. Off by default. When enabled, the framework
    # wires an EventBus → AG-UI translator alongside the trace collector so
    # consumers (SSE UIs, hosted control planes, third-party AG-UI clients) can
    # iterate live events via ``orchestra.aggui_translator``. Requires the
    # optional ``aggui`` extras (``pip install 'marsys[aggui]'``).
    aggui: 'AGGUIConfig' = field(default_factory=lambda: AGGUIConfig())

    # Retry/backoff and error-handling configuration. Consumed by the
    # model-adapter retry loop (``models/adapters/base.py``) and the
    # framework-level retry loop in ``StepExecutor``.
    error_handling: 'ErrorHandlingConfig' = field(
        default_factory=lambda: ErrorHandlingConfig()
    )

    # NEW: User interaction control fields (Orchestra only)
    user_first: bool = False  # Enable user-first execution mode
    initial_user_msg: Optional[str] = None  # Message shown to user in user-first mode
    user_interaction: str = "terminal"  # Type of user interaction: "terminal", "none"

    # Agent lifecycle management
    auto_cleanup_agents: bool = True  # Automatically cleanup agents after run (closes resources, unregisters)
    cleanup_scope: Literal["topology_nodes", "used_agents"] = "topology_nodes"  # Which agents to cleanup

    # Response format for agent outputs
    response_format: str = "json"  # Format name (e.g., "json", "xml")

    # Content-only loop detection (RealRuntime). When an agent emits N
    # consecutive content-only responses (no coord tool call, no regular tool
    # call), steering kicks in at content_only_steering_threshold and the
    # branch is FAILED with a structured diagnostic at content_only_hard_limit.
    # Threshold MUST be strictly less than hard_limit (enforced in __post_init__).
    content_only_steering_threshold: int = 2
    content_only_hard_limit: int = 10

    def __post_init__(self) -> None:
        if self.content_only_steering_threshold >= self.content_only_hard_limit:
            raise ValueError(
                f"content_only_steering_threshold ({self.content_only_steering_threshold}) "
                f"must be strictly less than content_only_hard_limit "
                f"({self.content_only_hard_limit})."
            )

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
    """Configuration for retry, backoff, and error-handling behaviour.

    The retry loop in model adapters (``models/adapters/base.py``) and the
    framework-level retry loop in ``StepExecutor`` both consume these
    settings. Per-provider overrides in ``provider_settings`` take precedence
    over the top-level defaults.

    Backoff formula::

        delay = min(
            base_delay * (2 ** attempt) * (1 + uniform(-jitter, jitter)),
            max_delay,
        )

    The ``retry-after`` / ``x-ratelimit-reset-after`` headers always win over
    the computed delay when the server signals a specific retry time.
    """

    # ── Top-level retry/backoff defaults (Phase 1) ──────────────────────
    max_retries: int = 3
    base_delay: float = 1.0
    jitter: float = 0.1  # multiplicative ±10% to break thundering-herd
    max_delay: float = 60.0  # cap on per-attempt sleep

    # ── Existing core features (kept for back-compat) ───────────────────
    use_error_classification: bool = True
    notify_on_critical_errors: bool = True
    enable_error_routing: bool = True
    preserve_error_context: bool = True

    # ── Existing retry knobs (kept for back-compat with steering) ───────
    auto_retry_on_rate_limits: bool = True
    max_rate_limit_retries: int = 3
    pool_retry_attempts: int = 2
    pool_retry_delay: float = 5.0

    # ── Existing timeout knobs ──────────────────────────────────────────
    timeout_seconds: float = 600.0
    timeout_retry_enabled: bool = False

    # ── Per-provider overrides (override top-level defaults when set) ───
    provider_settings: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "openai": {
            "max_retries": 3,
            "base_delay": 1.0,
            "insufficient_quota_action": "raise",  # "raise", "notify", "fallback"
        },
        "anthropic": {
            "max_retries": 3,
            "base_delay": 1.0,
            "insufficient_quota_action": "raise",
        },
        "google": {
            "max_retries": 3,
            "base_delay": 1.0,
            "insufficient_quota_action": "notify",
        },
        "openrouter": {
            "max_retries": 2,
            "base_delay": 1.0,
            "insufficient_quota_action": "raise",
        },
        "xai": {
            "max_retries": 2,
            "base_delay": 2.0,  # xAI has strict free-tier rate limits
            "insufficient_quota_action": "notify",
        },
    })

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay <= 0:
            raise ValueError(f"base_delay must be > 0, got {self.base_delay}")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError(f"jitter must be in [0, 1], got {self.jitter}")
        if self.max_delay < self.base_delay:
            raise ValueError(
                f"max_delay ({self.max_delay}) must be >= base_delay ({self.base_delay})"
            )

    # ── Resolved-setting helpers (provider-override-aware) ──────────────

    def resolve_max_retries(self, provider: Optional[str]) -> int:
        """Return ``max_retries`` for ``provider``, falling back to the global default."""
        return int(self.get_provider_setting(provider, "max_retries", self.max_retries))

    def resolve_base_delay(self, provider: Optional[str]) -> float:
        """Return ``base_delay`` for ``provider``, falling back to the global default."""
        return float(self.get_provider_setting(provider, "base_delay", self.base_delay))

    def compute_delay(self, provider: Optional[str], attempt: int) -> float:
        """Return the backoff delay for ``attempt`` (0-indexed) on ``provider``.

        Applies symmetric multiplicative jitter and caps at ``max_delay``.
        Callers should still prefer a server-supplied ``retry-after`` value
        over this computed delay.
        """
        import random
        base = self.resolve_base_delay(provider)
        delay = base * (2 ** attempt)
        if self.jitter > 0:
            delay = delay * (1 + random.uniform(-self.jitter, self.jitter))
        return min(max(delay, 0.0), self.max_delay)

    def get_provider_setting(
        self,
        provider: Optional[str],
        setting: str,
        default: Any = None,
    ) -> Any:
        """Get a specific setting for a provider, or ``default`` if unset."""
        if not provider:
            return default
        provider_config = self.provider_settings.get(provider, {})
        return provider_config.get(setting, default)

"""
Configuration classes for the coordination system.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, List
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
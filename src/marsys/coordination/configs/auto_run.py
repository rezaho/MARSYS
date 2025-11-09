"""
Configuration for auto_run behavior.

This module provides configuration for the agent auto_run functionality,
including user interaction settings, execution configuration, and status updates.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from ..config import ExecutionConfig, StatusConfig


@dataclass
class UserInteractionConfig:
    """Configuration for user interaction behavior."""
    mode: Optional[str] = None  # "terminal", "web", or None
    auto_detect: bool = True  # Auto-detect based on allowed_peers
    warn_on_missing_handler: bool = True
    timeout: float = 30.0


@dataclass
class AutoRunConfig:
    """Unified configuration for auto_run behavior."""

    # User interaction configuration
    user_interaction: UserInteractionConfig = field(default_factory=UserInteractionConfig)

    # Execution configuration (reuse existing)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Status configuration (reuse existing)
    status: StatusConfig = field(default_factory=StatusConfig)

    # Auto-run specific defaults
    default_max_steps: int = 30
    default_max_re_prompts: int = 3
    default_timeout: Optional[int] = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'AutoRunConfig':
        """Create config from auto_run keyword arguments.

        Args:
            **kwargs: Keyword arguments from auto_run method

        Returns:
            AutoRunConfig instance with settings from kwargs
        """
        config = cls()

        # Map user_interaction parameter
        if 'user_interaction' in kwargs:
            value = kwargs['user_interaction']
            if isinstance(value, bool):
                config.user_interaction.mode = "terminal" if value else None
            elif isinstance(value, str):
                config.user_interaction.mode = value
            # CommunicationManager instance handled separately in auto_run

        # Map execution settings
        if 'steering_mode' in kwargs:
            steering_mode = kwargs['steering_mode']
            # Validate steering mode
            valid_modes = ["auto", "always", "error"]
            if steering_mode not in valid_modes:
                logger.warning(f"Invalid steering_mode '{steering_mode}', defaulting to 'error'")
                steering_mode = "error"
            config.execution.steering_mode = steering_mode
        if 'auto_detect_convergence' in kwargs:
            config.execution.auto_detect_convergence = kwargs['auto_detect_convergence']
        if 'parent_completes_on_spawn' in kwargs:
            config.execution.parent_completes_on_spawn = kwargs['parent_completes_on_spawn']
        if 'dynamic_convergence_enabled' in kwargs:
            config.execution.dynamic_convergence_enabled = kwargs['dynamic_convergence_enabled']
        # Map user interaction settings to execution config
        if 'user_first' in kwargs and kwargs['user_first'] is not None:
            config.execution.user_first = kwargs['user_first']
        if 'initial_user_msg' in kwargs and kwargs['initial_user_msg'] is not None:
            config.execution.initial_user_msg = kwargs['initial_user_msg']

        # Map status settings
        if 'verbosity' in kwargs:
            config.status = StatusConfig.from_verbosity(kwargs['verbosity'])
            # Also set execution.status since Orchestra uses that
            config.execution.status = config.status

        # Map timeout settings to ExecutionConfig
        if 'convergence_timeout' in kwargs:
            config.execution.convergence_timeout = kwargs['convergence_timeout']
        if 'agent_acquisition_timeout' in kwargs:
            config.execution.agent_acquisition_timeout = kwargs['agent_acquisition_timeout']
        if 'step_timeout' in kwargs:
            config.execution.step_timeout = kwargs['step_timeout']
        if 'tool_execution_timeout' in kwargs:
            config.execution.tool_execution_timeout = kwargs['tool_execution_timeout']
        if 'user_interaction_timeout' in kwargs:
            config.execution.user_interaction_timeout = kwargs['user_interaction_timeout']
        if 'branch_timeout' in kwargs:
            config.execution.branch_timeout = kwargs['branch_timeout']

        # Set auto_run specific defaults
        config.default_max_steps = kwargs.get('max_steps', 30)
        config.default_max_re_prompts = kwargs.get('max_re_prompts', 3)
        config.default_timeout = kwargs.get('timeout')

        return config
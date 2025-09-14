"""
Configuration classes for the coordination system.
"""

from dataclasses import dataclass
from typing import Literal


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
            return retry_count > 0
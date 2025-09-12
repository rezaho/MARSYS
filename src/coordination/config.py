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
    
    def should_apply_steering(self, retry_count: int = 0) -> bool:
        """
        Determine if steering should be applied.
        
        Args:
            retry_count: Current retry attempt number
            
        Returns:
            True if steering should be applied, False otherwise
        """
        if self.steering_mode == "never":
            return False
        elif self.steering_mode == "always":
            return True
        else:  # auto mode - only on retries
            return retry_count > 0
"""Configuration for agent planning behavior."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set, Union


class InjectionTrigger(Enum):
    """Triggers for injecting plan context into messages."""
    SESSION_START = "session_start"
    STEP_START = "step_start"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class PlanningConfig:
    """Configuration for agent planning behavior."""

    enabled: bool = True  # Enabled by default

    # Minimum number of items required when creating a plan
    # Enforced in create_plan() - prevents trivial single-item plans
    min_plan_items: int = 2

    # Inject plan context after this many execution steps (0-based)
    # Value of 0 means inject from the first step
    # Only applies when STEP_START trigger is enabled
    inject_after_step: int = 0

    # Which triggers activate plan injection
    inject_triggers: Set[InjectionTrigger] = field(
        default_factory=lambda: {
            InjectionTrigger.SESSION_START,
            InjectionTrigger.STEP_START,
            InjectionTrigger.ERROR_RECOVERY,
        }
    )

    # Display settings
    compact_mode: bool = True
    max_items_in_compact: int = 3

    # Size limits
    max_plan_items: int = 20
    max_item_content_length: int = 500

    # Custom instruction override
    custom_instruction: Optional[str] = None

    @classmethod
    def from_value(cls, value: Union['PlanningConfig', Dict, bool, None]) -> 'PlanningConfig':
        """
        Create config from flexible input types.

        Args:
            value: One of:
                - PlanningConfig: returned as-is
                - True/None: returns enabled config with defaults
                - False: returns disabled config
                - Dict: creates config from dict values
        """
        if isinstance(value, cls):
            return value
        elif isinstance(value, dict):
            return cls.from_dict(value)
        elif value is False:
            return cls(enabled=False)
        else:
            # None or True -> enabled with defaults
            return cls(enabled=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanningConfig':
        """Create config from dictionary."""
        if not data:
            return cls()

        # Convert string triggers to enum
        if 'inject_triggers' in data:
            triggers = data['inject_triggers']
            if isinstance(triggers, (list, set)):
                data = data.copy()
                data['inject_triggers'] = {
                    InjectionTrigger(t) if isinstance(t, str) else t
                    for t in triggers
                }

        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

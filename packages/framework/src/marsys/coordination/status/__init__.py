"""
Status update system for multi-agent coordination.
"""

from .events import (
    StatusEvent,
    AgentStartEvent,
    AgentThinkingEvent,
    AgentCompleteEvent,
    ToolCallEvent,
    BranchEvent,
    ParallelGroupEvent,
    UserInteractionEvent,
    FinalResponseEvent,
    CompactionEvent
)
from .manager import StatusManager
from .channels import ChannelAdapter, CLIChannel

__all__ = [
    'StatusEvent',
    'AgentStartEvent',
    'AgentThinkingEvent',
    'AgentCompleteEvent',
    'ToolCallEvent',
    'BranchEvent',
    'ParallelGroupEvent',
    'UserInteractionEvent',
    'FinalResponseEvent',
    'CompactionEvent',
    'StatusManager',
    'ChannelAdapter',
    'CLIChannel'
]
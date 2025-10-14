"""
Communication infrastructure for User node interactions.

This module provides the foundation for bidirectional communication
between the multi-agent system and external interfaces (terminal, web, API).
"""

from .core import (
    CommunicationMode,
    UserInteraction,
    CommunicationChannel,
    SyncChannel,
    AsyncChannel
)
from .manager import CommunicationManager
from .user_node_handler import UserNodeHandler

__all__ = [
    "CommunicationMode",
    "UserInteraction", 
    "CommunicationChannel",
    "SyncChannel",
    "AsyncChannel",
    "CommunicationManager",
    "UserNodeHandler"
]
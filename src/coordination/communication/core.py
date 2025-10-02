"""
Core types and interfaces for user communication.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid


class CommunicationMode(Enum):
    """Communication mode for user interactions."""
    SYNC = "sync"                    # Blocking call/response
    ASYNC_PUBSUB = "async_pubsub"    # Pub/sub pattern
    ASYNC_QUEUE = "async_queue"      # Queue-based


@dataclass
class UserInteraction:
    """Represents an interaction between system and user."""
    
    # Core identifiers
    interaction_id: str
    branch_id: str
    session_id: str
    
    # Message content
    incoming_message: Any
    interaction_type: str = "question"  # question, notification, choice, confirmation
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    
    # Communication details
    communication_mode: CommunicationMode = CommunicationMode.SYNC
    channel_preferences: List[str] = field(default_factory=list)
    
    # Agent tracing
    calling_agent: Optional[str] = None      # Who initiated the interaction
    resume_agent: Optional[str] = None       # Where to resume after response
    execution_trace: List[Any] = field(default_factory=list)
    
    # Branch context for resumption
    branch_context: Dict[str, Any] = field(default_factory=dict)
    memory_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    
    # For async patterns
    topic: Optional[str] = None              # For pub/sub
    queue_name: Optional[str] = None         # For queue-based
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_display_dict(self) -> Dict[str, Any]:
        """Format for display to user/API."""
        return {
            "id": self.interaction_id,
            "type": self.interaction_type,
            "message": self.incoming_message,
            "from_agent": self.calling_agent,
            "context": {
                "session": self.session_id,
                "branch": self.branch_id,
                "step": len(self.execution_trace)
            },
            "metadata": {
                "will_resume_at": self.resume_agent,
                "timestamp": self.timestamp,
                "mode": self.communication_mode.value
            }
        }


class CommunicationChannel(ABC):
    """Base class for all communication channels."""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.active = False
        
    @abstractmethod
    async def start(self) -> None:
        """Start the communication channel."""
        pass
        
    @abstractmethod
    async def stop(self) -> None:
        """Stop the communication channel."""
        pass
        
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if channel is available for communication."""
        pass


class SyncChannel(CommunicationChannel):
    """Base class for synchronous channels (blocking I/O)."""
    
    @abstractmethod
    async def send_interaction(self, interaction: UserInteraction) -> None:
        """Send interaction and expect blocking response."""
        pass
    
    @abstractmethod
    async def get_response(self, interaction_id: str) -> Tuple[str, Any]:
        """Get response (blocking) - returns (interaction_id, response)."""
        pass


class AsyncChannel(CommunicationChannel):
    """Base class for asynchronous channels (non-blocking)."""
    
    @abstractmethod
    async def publish_interaction(self, interaction: UserInteraction) -> None:
        """Publish interaction (non-blocking)."""
        pass
    
    @abstractmethod
    def subscribe_responses(self, callback: Callable[[str, Any], None]) -> None:
        """Subscribe to responses with callback(interaction_id, response)."""
        pass
    
    @abstractmethod
    async def unsubscribe_responses(self) -> None:
        """Unsubscribe from responses."""
        pass
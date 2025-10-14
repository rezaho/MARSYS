"""
StatusManager for handling and distributing status events.
"""

import asyncio
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..event_bus import EventBus
    from .events import StatusEvent, ParallelGroupEvent
    from .channels import ChannelAdapter
    from ..config import StatusConfig

logger = logging.getLogger(__name__)


class StatusManager:
    """
    Manages status events and distributes to channels.

    Subscribes to EventBus for status events and forwards
    formatted updates to configured output channels.
    """

    def __init__(self, event_bus: 'EventBus', config: 'StatusConfig'):
        self.event_bus = event_bus
        self.config = config
        self.channels: List['ChannelAdapter'] = []

        # Session-based storage
        self.session_events: Dict[str, deque] = {}
        self._session_last_activity: Dict[str, float] = {}

        # Parallel group tracking
        self.parallel_groups: Dict[str, List] = {}

        # Async safety
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

        # Subscribe to all status events
        self._subscribe_to_events()

        # Start cleanup if configured
        if config.session_cleanup_after_s > 0:
            self._start_cleanup_task()

    def _subscribe_to_events(self):
        """Subscribe to all status event types."""
        event_types = [
            # New status events from this system
            'AgentStartEvent',
            'AgentThinkingEvent',
            'AgentCompleteEvent',
            'ToolCallEvent',
            'BranchEvent',
            'ParallelGroupEvent',
            'UserInteractionEvent',
            'FinalResponseEvent',

            # Existing events from BranchSpawner
            'BranchCreatedEvent',    # Already emitted by DynamicBranchSpawner
            'BranchCompletedEvent'   # Already emitted by DynamicBranchSpawner
        ]

        for event_type in event_types:
            self.event_bus.subscribe(event_type, self.handle_event)

    async def handle_event(self, event: 'StatusEvent') -> None:
        """Process incoming status event."""
        # Check if event should be shown
        if not self.config.should_show_event(event.event_type):
            return

        async with self._lock:
            # Store event
            session_id = event.session_id
            if session_id not in self.session_events:
                self.session_events[session_id] = deque(
                    maxlen=self.config.max_events_per_session
                )
            self.session_events[session_id].append(event)
            self._session_last_activity[session_id] = time.time()

            # Track parallel groups if applicable
            if hasattr(event, 'group_id') and event.group_id:
                if event.group_id not in self.parallel_groups:
                    self.parallel_groups[event.group_id] = []
                self.parallel_groups[event.group_id].append(event)

        # Forward to channels (outside lock)
        await self._forward_to_channels(event)

    async def _forward_to_channels(self, event: 'StatusEvent') -> None:
        """Forward event to all active channels."""
        from .events import ParallelGroupEvent

        # Apply aggregation for parallel events if configured
        if (self.config.aggregate_parallel and
            isinstance(event, ParallelGroupEvent) and
            event.status == "executing"):
            # Buffer for aggregation window
            await asyncio.sleep(self.config.aggregation_window_ms / 1000.0)
            # Send aggregated update
            event = self._create_aggregated_event(event.group_id)

        # Send to all channels
        for channel in self.channels:
            if channel.is_enabled():
                try:
                    await channel.send(event)
                except Exception as e:
                    logger.debug(f"Channel {channel.name} failed: {e}")

    def _create_aggregated_event(self, group_id: str) -> 'ParallelGroupEvent':
        """Create aggregated event for parallel group."""
        events = self.parallel_groups.get(group_id, [])
        if not events:
            return None

        # Find latest status for each agent
        agent_statuses = {}
        for event in events:
            if hasattr(event, 'agent_name'):
                agent_statuses[event.agent_name] = event

        # Count completions
        completed = sum(1 for e in agent_statuses.values()
                       if hasattr(e, 'success') and e.success)

        # Return aggregated event
        from .events import ParallelGroupEvent
        return ParallelGroupEvent(
            session_id=events[0].session_id,
            group_id=group_id,
            agent_names=list(agent_statuses.keys()),
            status="executing",
            completed_count=completed,
            total_count=len(agent_statuses)
        )

    def add_channel(self, channel: 'ChannelAdapter') -> None:
        """Add output channel."""
        self.channels.append(channel)
        logger.info(f"Added channel: {channel.name}")

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_old_sessions()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_old_sessions(self):
        """Remove old session data."""
        async with self._lock:
            current_time = time.time()
            cutoff = current_time - self.config.session_cleanup_after_s

            to_remove = [
                sid for sid, last_activity in self._session_last_activity.items()
                if last_activity < cutoff
            ]

            for sid in to_remove:
                del self.session_events[sid]
                del self._session_last_activity[sid]
                logger.debug(f"Cleaned up session {sid}")

    async def shutdown(self):
        """Clean shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        for channel in self.channels:
            await channel.close()
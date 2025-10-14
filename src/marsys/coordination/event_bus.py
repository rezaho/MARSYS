"""
Event bus for coordination system.

This module contains the EventBus class extracted from orchestra.py
and enhanced with additional functionality for the status system.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple event bus for coordination events.

    Enhanced version of the original EventBus from orchestra.py with
    additional error handling and subscription management features.
    """

    def __init__(self):
        """Initialize the event bus."""
        self.events: List[Any] = []
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._listener_errors: Dict[str, int] = defaultdict(int)
        self._max_listener_errors = 5  # Prevent runaway error listeners

    async def emit(self, event: Any) -> None:
        """
        Emit an event to all listeners.

        Args:
            event: The event object to emit
        """
        # Store event in history
        self.events.append(event)

        # Get event type name for routing
        event_type = type(event).__name__

        # Notify all listeners for this event type
        if event_type in self.listeners:
            for listener in self.listeners[event_type]:
                try:
                    # Call listener with event
                    await listener(event)
                except Exception as e:
                    # Track errors per listener
                    listener_id = f"{event_type}:{id(listener)}"
                    self._listener_errors[listener_id] += 1

                    logger.error(f"Error in event listener for {event_type}: {e}")

                    # Remove listener if too many errors
                    if self._listener_errors[listener_id] >= self._max_listener_errors:
                        logger.warning(f"Removing failing listener for {event_type} after {self._max_listener_errors} errors")
                        self.listeners[event_type].remove(listener)

    def subscribe(self, event_type: str, listener: Callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Name of the event class to subscribe to
            listener: Async callable to handle events
        """
        if event_type not in self.listeners:
            self.listeners[event_type] = []

        # Avoid duplicate subscriptions
        if listener not in self.listeners[event_type]:
            self.listeners[event_type].append(listener)
            logger.debug(f"Subscribed listener to {event_type}")

    def unsubscribe(self, event_type: str, listener: Callable) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Name of the event class to unsubscribe from
            listener: The listener to remove
        """
        if event_type in self.listeners and listener in self.listeners[event_type]:
            self.listeners[event_type].remove(listener)
            logger.debug(f"Unsubscribed listener from {event_type}")

    def clear_listeners(self, event_type: Optional[str] = None) -> None:
        """
        Clear listeners for a specific event type or all listeners.

        Args:
            event_type: Optional event type to clear. If None, clears all.
        """
        if event_type:
            if event_type in self.listeners:
                self.listeners[event_type].clear()
                logger.debug(f"Cleared all listeners for {event_type}")
        else:
            self.listeners.clear()
            self._listener_errors.clear()
            logger.debug("Cleared all event listeners")

    def clear_events(self) -> None:
        """Clear the event history."""
        self.events.clear()

    def get_event_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of events in history.

        Args:
            event_type: Optional event type to count. If None, counts all.

        Returns:
            Number of events
        """
        if event_type:
            return sum(1 for e in self.events if type(e).__name__ == event_type)
        return len(self.events)

    def get_listener_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of registered listeners.

        Args:
            event_type: Optional event type to count. If None, counts all.

        Returns:
            Number of listeners
        """
        if event_type:
            return len(self.listeners.get(event_type, []))
        return sum(len(listeners) for listeners in self.listeners.values())
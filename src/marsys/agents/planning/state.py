"""Planning state management for agents."""

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .config import InjectionTrigger, PlanningConfig
from .types import Plan, PlanItem

if TYPE_CHECKING:
    from ...coordination.event_bus import EventBus

logger = logging.getLogger(__name__)


def _import_planning_events():
    """Import planning events lazily to avoid circular imports."""
    from ...coordination.status.events import (
        PlanCreatedEvent,
        PlanUpdatedEvent,
        PlanItemAddedEvent,
        PlanItemRemovedEvent,
        PlanClearedEvent,
    )
    return (
        PlanCreatedEvent,
        PlanUpdatedEvent,
        PlanItemAddedEvent,
        PlanItemRemovedEvent,
        PlanClearedEvent,
    )


class PlanningState:
    """
    Manages planning state for an agent.

    Responsibilities:
    - Store and manipulate the current plan
    - Subscribe to MemoryResetEvent for automatic clearing
    - Format plan for injection into messages
    - Handle persistence (serialization/deserialization)
    """

    def __init__(self, config: PlanningConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self._plan: Optional[Plan] = None
        self._event_bus: Optional['EventBus'] = None
        self._subscribed = False
        self._session_id: Optional[str] = None

    # ==================== EventBus Integration ====================

    def set_event_bus(self, event_bus: 'EventBus') -> None:
        """
        Set EventBus and subscribe to memory reset events.

        Handles rebinding: if called with a different EventBus, unsubscribes
        from the old one before subscribing to the new one.
        """
        # Same bus, already subscribed - nothing to do
        if event_bus is self._event_bus and self._subscribed:
            return

        # Different bus - unsubscribe from old first
        if self._event_bus and self._subscribed:
            try:
                self._event_bus.unsubscribe("MemoryResetEvent", self._on_memory_reset)
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from old EventBus: {e}")

        # Subscribe to new bus
        self._event_bus = event_bus
        event_bus.subscribe("MemoryResetEvent", self._on_memory_reset)
        self._subscribed = True
        logger.debug(f"PlanningState for '{self.agent_name}' subscribed to MemoryResetEvent")

    def unsubscribe(self) -> None:
        """
        Unsubscribe from MemoryResetEvent.

        Used during state loading to prevent race conditions with emit_nowait().
        The caller should resubscribe after loading is complete.
        """
        if self._event_bus and self._subscribed:
            try:
                self._event_bus.unsubscribe("MemoryResetEvent", self._on_memory_reset)
                self._subscribed = False
                logger.debug(f"PlanningState for '{self.agent_name}' unsubscribed from MemoryResetEvent")
            except Exception as e:
                logger.warning(f"Failed to unsubscribe: {e}")

    def resubscribe(self) -> None:
        """
        Resubscribe to MemoryResetEvent after unsubscribing.

        Called after state loading is complete.
        """
        if self._event_bus and not self._subscribed:
            self._event_bus.subscribe("MemoryResetEvent", self._on_memory_reset)
            self._subscribed = True
            logger.debug(f"PlanningState for '{self.agent_name}' resubscribed to MemoryResetEvent")

    def set_session_id(self, session_id: str) -> None:
        """
        Set the session ID for event emission.

        Args:
            session_id: The current execution session ID
        """
        self._session_id = session_id

    def _emit_event(self, event) -> None:
        """
        Emit a planning event via EventBus if available.

        Args:
            event: The event to emit
        """
        if self._event_bus:
            try:
                self._event_bus.emit_nowait(event)
            except Exception as e:
                logger.warning(f"Failed to emit planning event: {e}")

    async def _on_memory_reset(self, event) -> None:
        """Handle MemoryResetEvent by clearing the plan."""
        # Only respond to events for this agent
        if event.agent_name == self.agent_name:
            logger.info(f"Clearing plan for '{self.agent_name}' due to memory reset")
            self.clear_plan(reason="reset")

    # ==================== Plan CRUD Operations ====================

    def create_plan(
        self,
        items: List[Dict[str, Any]],
        goal: Optional[str] = None
    ) -> Plan:
        """
        Create a new plan, replacing any existing plan.

        Args:
            items: List of item dicts with keys: title, content, active_form, priority
            goal: High-level goal description

        Returns:
            The created Plan

        Raises:
            ValueError: If items count is outside allowed range or content exceeds max length
        """
        # Enforce minimum item count
        if len(items) < self.config.min_plan_items:
            raise ValueError(
                f"Plan has {len(items)} items but minimum is {self.config.min_plan_items}. "
                f"For simple tasks, consider not using a plan."
            )

        # Enforce maximum item count
        if len(items) > self.config.max_plan_items:
            raise ValueError(
                f"Plan has {len(items)} items but maximum is {self.config.max_plan_items}"
            )

        plan_items = []
        for item_dict in items:
            # Enforce content length limit
            content = item_dict.get("content", "")
            title = item_dict.get("title", "")

            if len(content) > self.config.max_item_content_length:
                raise ValueError(
                    f"Item content ({len(content)} chars) exceeds maximum "
                    f"({self.config.max_item_content_length} chars)"
                )
            if len(title) > self.config.max_item_content_length:
                raise ValueError(
                    f"Item title ({len(title)} chars) exceeds maximum "
                    f"({self.config.max_item_content_length} chars)"
                )

            plan_items.append(PlanItem(
                title=title,
                content=content,
                active_form=item_dict.get("active_form", ""),
                priority=item_dict.get("priority", "medium"),
            ))

        self._plan = Plan(items=plan_items, goal=goal)
        logger.info(f"Created plan with {len(plan_items)} items for '{self.agent_name}'")

        # Emit PlanCreatedEvent
        if self._session_id and self._event_bus:
            PlanCreatedEvent, _, _, _, _ = _import_planning_events()
            self._emit_event(PlanCreatedEvent(
                session_id=self._session_id,
                agent_name=self.agent_name,
                goal=goal,
                item_count=len(plan_items),
                item_titles=[item.title for item in plan_items],
            ))

        return self._plan

    def update_item(
        self,
        item_id: str,
        status: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        active_form: Optional[str] = None,
        priority: Optional[str] = None,
        blocked_reason: Optional[str] = None
    ) -> Optional[PlanItem]:
        """
        Update a plan item.

        Args:
            item_id: ID of the item to update
            status: New status (pending, in_progress, completed, blocked)
            title: New title
            content: New content
            active_form: New active form text
            priority: New priority (high, medium, low)
            blocked_reason: Reason for blocking (required if status is blocked)

        Returns:
            Updated PlanItem or None if not found

        Raises:
            ValueError: If attempting invalid state transitions or exceeding limits
        """
        if not self._plan:
            return None

        item = self._plan.get_item(item_id)
        if not item:
            return None

        # Capture old status for event emission
        old_status = item.status if status else None

        # Validate content length
        if content and len(content) > self.config.max_item_content_length:
            raise ValueError(f"Content exceeds {self.config.max_item_content_length} chars")
        if title and len(title) > self.config.max_item_content_length:
            raise ValueError(f"Title exceeds {self.config.max_item_content_length} chars")

        # Track which fields were updated (for event metadata)
        updated_fields = []

        # Handle status changes
        if status:
            if status == "in_progress":
                # Enforce single in_progress constraint
                current = self._plan.current_item
                if current and current.id != item_id:
                    raise ValueError(
                        f"Cannot start '{item.title}' - '{current.title}' is already in progress"
                    )
                item.start()
            elif status == "completed":
                item.complete()
            elif status == "blocked":
                item.block(blocked_reason or "No reason provided")
            elif status == "pending":
                item.status = "pending"
                item.started_at = None
            updated_fields.append("status")

        # Handle field updates
        if title:
            item.title = title
            updated_fields.append("title")
        if content:
            item.content = content
            updated_fields.append("content")
        if active_form:
            item.active_form = active_form
            updated_fields.append("active_form")
        if priority:
            if priority not in ("high", "medium", "low"):
                raise ValueError(f"Invalid priority: {priority}")
            item.priority = priority
            updated_fields.append("priority")
        if blocked_reason and status == "blocked":
            updated_fields.append("blocked_reason")

        self._plan.version += 1

        # Emit PlanUpdatedEvent for ANY update (not just status changes)
        if self._session_id and self._event_bus and updated_fields:
            _, PlanUpdatedEvent, _, _, _ = _import_planning_events()
            self._emit_event(PlanUpdatedEvent(
                session_id=self._session_id,
                agent_name=self.agent_name,
                item_id=item_id,
                item_title=item.title,
                old_status=old_status,
                new_status=status,
                active_form=item.active_form,
                metadata={"updated_fields": updated_fields},
            ))

        return item

    def add_item(
        self,
        title: str,
        content: str,
        active_form: str,
        priority: str = "medium",
        after_item_id: Optional[str] = None
    ) -> PlanItem:
        """
        Add a new item to the existing plan.

        Args:
            title: Item title
            content: Item content/description
            active_form: Present continuous form for display
            priority: Priority level
            after_item_id: Insert after this item (default: append to end)

        Returns:
            The created PlanItem

        Raises:
            ValueError: If no plan exists or limits exceeded
        """
        if not self._plan:
            raise ValueError("No plan exists. Create one first with create_plan().")

        if len(self._plan.items) >= self.config.max_plan_items:
            raise ValueError(f"Plan already has {self.config.max_plan_items} items (maximum)")

        if len(content) > self.config.max_item_content_length:
            raise ValueError(f"Content exceeds {self.config.max_item_content_length} chars")

        item = PlanItem(
            title=title,
            content=content,
            active_form=active_form,
            priority=priority,
        )
        self._plan.add_item(item, after_id=after_item_id)

        # Emit PlanItemAddedEvent
        if self._session_id and self._event_bus:
            _, _, PlanItemAddedEvent, _, _ = _import_planning_events()
            # Calculate position (1-based)
            position = self._plan.items.index(item) + 1
            self._emit_event(PlanItemAddedEvent(
                session_id=self._session_id,
                agent_name=self.agent_name,
                item_id=item.id,
                item_title=item.title,
                position=position,
            ))

        return item

    def remove_item(self, item_id: str) -> bool:
        """
        Remove an item from the plan.

        Args:
            item_id: ID of item to remove

        Returns:
            True if removed, False if not found

        Raises:
            ValueError: If item is in_progress
        """
        if not self._plan:
            return False

        item = self._plan.get_item(item_id)
        if not item:
            return False

        if item.status == "in_progress":
            raise ValueError(f"Cannot remove '{item.title}' while in progress")

        # Capture title before removal for event
        item_title = item.title

        result = self._plan.remove_item(item_id)

        # Emit PlanItemRemovedEvent
        if result and self._session_id and self._event_bus:
            _, _, _, PlanItemRemovedEvent, _ = _import_planning_events()
            self._emit_event(PlanItemRemovedEvent(
                session_id=self._session_id,
                agent_name=self.agent_name,
                item_id=item_id,
                item_title=item_title,
            ))

        return result

    def clear_plan(self, reason: Optional[str] = None) -> None:
        """
        Clear the current plan.

        Args:
            reason: Optional reason for clearing (e.g., "completed", "abandoned", "reset")
        """
        had_plan = self._plan is not None
        self._plan = None

        # Emit PlanClearedEvent
        if had_plan and self._session_id and self._event_bus:
            _, _, _, _, PlanClearedEvent = _import_planning_events()
            self._emit_event(PlanClearedEvent(
                session_id=self._session_id,
                agent_name=self.agent_name,
                reason=reason,
            ))

    def get_plan(self) -> Optional[Plan]:
        """Get the current plan."""
        return self._plan

    def is_empty(self) -> bool:
        """Check if there's no active plan or plan is empty."""
        return self._plan is None or self._plan.is_empty

    # ==================== Context Formatting ====================

    def format_for_injection(self, verbose: bool = False) -> Optional[str]:
        """
        Format the plan for injection into messages.

        Args:
            verbose: Use full format (True) or compact format (False)

        Returns:
            Formatted string or None if no plan
        """
        if not self._plan or self._plan.is_empty:
            return None

        if verbose or not self.config.compact_mode:
            return self._format_full()
        else:
            return self._format_compact()

    def _format_full(self) -> str:
        """Full plan format with all details."""
        plan = self._plan
        lines = []

        if plan.goal:
            lines.append(f"Goal: {plan.goal}")

        completed = len(plan.completed_items)
        total = len(plan.items)
        lines.append(f"Progress: {completed}/{total} ({plan.progress_ratio:.0%})")
        lines.append("")

        for i, item in enumerate(plan.items, 1):
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
                "blocked": "[!]"
            }.get(item.status, "[ ]")

            priority_tag = f" [{item.priority.upper()}]" if item.priority != "medium" else ""
            lines.append(f"{i}. {status_icon} {item.title}{priority_tag} (id: {item.id})")

            if item.content and item.content != item.title:
                lines.append(f"   {item.content}")

            if item.status == "in_progress" and item.active_form:
                lines.append(f"   Currently: {item.active_form}")
            elif item.status == "blocked" and item.blocked_reason:
                lines.append(f"   Blocked: {item.blocked_reason}")

        return "\n".join(lines)

    def _format_compact(self) -> str:
        """Compact plan format showing only current and next items."""
        plan = self._plan
        lines = []

        current = plan.current_item
        if current:
            lines.append(f"Current: {current.active_form or current.title} (id: {current.id})")

        pending = plan.pending_items[:self.config.max_items_in_compact]
        if pending:
            next_items = ", ".join(f"{p.title} (id: {p.id})" for p in pending)
            lines.append(f"Next: {next_items}")

        lines.append(f"Progress: {plan.progress_ratio:.0%}")

        return "\n".join(lines)

    def get_all_items_summary(self) -> str:
        """Get summary of all items with IDs (for error messages)."""
        if not self._plan:
            return "No plan active."

        lines = ["Available plan items:"]
        for item in self._plan.items:
            lines.append(f"  - {item.title} (id: {item.id}, status: {item.status})")
        return "\n".join(lines)

    # ==================== Persistence ====================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "agent_name": self.agent_name,
            "plan": self._plan.to_dict() if self._plan else None,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        if data.get("plan"):
            self._plan = Plan.from_dict(data["plan"])
        else:
            self._plan = None

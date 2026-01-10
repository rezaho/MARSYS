"""Planning tools for agents to create and manage task plans."""

from typing import Any, Dict, List, Optional

from .state import PlanningState


def create_planning_tools(planning_state: PlanningState) -> Dict[str, callable]:
    """
    Create planning tools bound to a PlanningState instance.

    Args:
        planning_state: The PlanningState to operate on

    Returns:
        Dict mapping tool names to functions
    """

    async def plan_create(
        items: List[Dict[str, str]],
        goal: Optional[str] = None
    ) -> str:
        """
        Create a new task plan, replacing any existing plan.

        Use this tool when starting a complex task that requires multiple steps.
        This creates a fresh plan - any existing plan will be replaced.

        Args:
            items: List of plan items. Each item must have:
                - title (str): Short title for the step
                - content (str): Detailed description in imperative form
                - active_form (str): Present continuous form (e.g., "Running tests")
                - priority (str, optional): "high", "medium", or "low"
            goal: High-level goal description (recommended)

        Returns:
            Confirmation with plan summary and item IDs

        Note:
            Plan must have at least min_plan_items items (default: 2).
            For single-step tasks, don't use a plan.
        """
        try:
            plan = planning_state.create_plan(items=items, goal=goal)
            items_list = "\n".join(
                f"  {i+1}. {item.title} (id: {item.id})"
                for i, item in enumerate(plan.items)
            )
            return (
                f"Plan created with {len(plan.items)} items.\n"
                f"Goal: {goal or 'Not specified'}\n"
                f"Items:\n{items_list}\n\n"
                f"Use plan_update with item id to change status."
            )
        except ValueError as e:
            return f"Error creating plan: {e}"

    async def plan_read() -> str:
        """
        Read the current plan state.

        Returns:
            Formatted plan with all items, statuses, and IDs
        """
        if planning_state.is_empty():
            return "No plan currently active. Use plan_create to create one."
        return planning_state._format_full()

    async def plan_update(
        item_id: str,
        status: Optional[str] = None,
        title: Optional[str] = None,
        content: Optional[str] = None,
        active_form: Optional[str] = None,
        priority: Optional[str] = None,
        blocked_reason: Optional[str] = None
    ) -> str:
        """
        Update a plan item.

        Args:
            item_id: ID of the item to update (from plan_read output)
            status: New status: "pending", "in_progress", "completed", "blocked"
            title: New title (optional)
            content: New description (optional)
            active_form: New active form text (optional)
            priority: New priority: "high", "medium", "low" (optional)
            blocked_reason: Required if status is "blocked"

        Returns:
            Confirmation or error message

        Rules:
            - Only one item can be "in_progress" at a time
            - Complete the current item before starting another
        """
        if planning_state.is_empty():
            return "No plan active. Use plan_create first."

        if not any([status, title, content, active_form, priority]):
            return "No updates specified. Provide at least one field to update."

        try:
            item = planning_state.update_item(
                item_id=item_id,
                status=status,
                title=title,
                content=content,
                active_form=active_form,
                priority=priority,
                blocked_reason=blocked_reason
            )
            if item:
                changes = []
                if status:
                    changes.append(f"status -> {status}")
                if title:
                    changes.append("title updated")
                if content:
                    changes.append("content updated")
                if active_form:
                    changes.append("active_form updated")
                if priority:
                    changes.append(f"priority -> {priority}")
                return f"Updated '{item.title}': {', '.join(changes)}"
            else:
                return (
                    f"Item '{item_id}' not found.\n\n"
                    f"{planning_state.get_all_items_summary()}"
                )
        except ValueError as e:
            return f"Error: {e}"

    async def plan_add_item(
        title: str,
        content: str,
        active_form: str,
        priority: str = "medium",
        after_item_id: Optional[str] = None
    ) -> str:
        """
        Add a new item to the existing plan.

        Args:
            title: Short title for the step
            content: Detailed description
            active_form: Present continuous form for status display
            priority: "high", "medium", or "low"
            after_item_id: Insert after this item ID (default: append to end)

        Returns:
            Confirmation with new item ID
        """
        if planning_state.is_empty():
            return "No plan active. Use plan_create to create a plan first."

        try:
            item = planning_state.add_item(
                title=title,
                content=content,
                active_form=active_form,
                priority=priority,
                after_item_id=after_item_id
            )
            return f"Added item: '{title}' (id: {item.id})"
        except ValueError as e:
            return f"Error: {e}"

    async def plan_remove_item(item_id: str) -> str:
        """
        Remove an item from the plan.

        Args:
            item_id: ID of the item to remove

        Returns:
            Confirmation or error message

        Note: Cannot remove an item that is "in_progress"
        """
        if planning_state.is_empty():
            return "No plan active."

        try:
            if planning_state.remove_item(item_id):
                return f"Removed item {item_id}"
            else:
                return (
                    f"Item '{item_id}' not found.\n\n"
                    f"{planning_state.get_all_items_summary()}"
                )
        except ValueError as e:
            return f"Error: {e}"

    async def plan_clear() -> str:
        """
        Clear the current plan entirely.

        Use when:
        - Task is complete and plan no longer needed
        - Starting a completely new task
        - Abandoning current approach

        Returns:
            Confirmation message
        """
        planning_state.clear_plan()
        return "Plan cleared."

    return {
        "plan_create": plan_create,
        "plan_read": plan_read,
        "plan_update": plan_update,
        "plan_add_item": plan_add_item,
        "plan_remove_item": plan_remove_item,
        "plan_clear": plan_clear,
    }

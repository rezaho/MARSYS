"""Default planning instructions for agents."""

from typing import Optional


DEFAULT_PLANNING_INSTRUCTION = """
--- TASK PLANNING ---

You have planning tools to organize complex tasks.

WHEN TO USE PLANNING:
- Tasks with 2+ distinct steps
- Tasks requiring progress tracking
- Tasks needing careful coordination

WORKFLOW:
1. Analyze the task and identify required steps
2. Call `plan_create` to create your plan BEFORE starting work
3. Call `plan_update` to mark the first item as "in_progress"
4. Work on the current item
5. Mark it "completed" IMMEDIATELY when done
6. Continue with next pending item

RULES:
- Only ONE item can be "in_progress" at a time
- Mark items "completed" IMMEDIATELY after finishing
- Never mark "completed" if there are errors or partial work
- Use "blocked" status with a reason if you cannot proceed
- Use `plan_read` to check your progress
- Use `plan_clear` when completely done

--- END TASK PLANNING ---
"""


def get_planning_instruction(custom_instruction: Optional[str] = None) -> str:
    """Get planning instruction for system prompt."""
    return custom_instruction or DEFAULT_PLANNING_INSTRUCTION

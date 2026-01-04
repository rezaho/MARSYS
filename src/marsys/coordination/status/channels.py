"""
Output channels for status events.
"""

from abc import ABC, abstractmethod
import sys
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Dict, Set
import asyncio

if TYPE_CHECKING:
    from .events import StatusEvent
    from ..config import StatusConfig, VerbosityLevel


class ChannelAdapter(ABC):
    """Base class for output channels."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def is_enabled(self) -> bool:
        return self.enabled

    @abstractmethod
    async def send(self, event: 'StatusEvent') -> None:
        """Send event to channel."""
        pass

    async def close(self) -> None:
        """Close channel resources."""
        pass


class CLIChannel(ChannelAdapter):
    """
    Terminal output channel with verbosity-aware formatting.
    """

    def __init__(self, config: 'StatusConfig'):
        super().__init__("cli", config.cli_output)
        self.config = config
        self.use_colors = config.cli_colors and sys.stdout.isatty()

        # Track state for cleaner output
        self.last_agent_name: Optional[str] = None
        self.start_time: float = time.time()

        # Track parallel execution groups
        self.parallel_groups: Dict[str, Set[str]] = {}  # group_id -> set of agent_names
        self.agent_to_group: Dict[str, str] = {}  # agent_name -> group_id
        self.active_parallel_groups: Set[str] = set()  # Currently active parallel groups

        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'red': '\033[31m',
            'gray': '\033[90m',
            'cyan': '\033[36m'
        } if self.use_colors else {k: '' for k in ['reset', 'bold', 'green', 'yellow', 'blue', 'red', 'gray', 'cyan']}

    async def send(self, event: 'StatusEvent') -> None:
        """Format and print status event based on verbosity."""
        from .events import (
            AgentStartEvent, AgentThinkingEvent, AgentCompleteEvent,
            ToolCallEvent, BranchEvent, ParallelGroupEvent,
            UserInteractionEvent, FinalResponseEvent,
            PlanCreatedEvent, PlanUpdatedEvent, PlanItemAddedEvent,
            PlanItemRemovedEvent, PlanClearedEvent
        )
        from ..config import VerbosityLevel

        # Get timestamp
        if self.config.show_timings:
            elapsed = time.time() - self.start_time
            timestamp = f"[{elapsed:6.2f}s]"
        else:
            timestamp = ""

        # Format based on event type and verbosity
        verbosity = self.config.verbosity or VerbosityLevel.NORMAL

        if isinstance(event, AgentStartEvent):
            await self._print_agent_start(event, timestamp, verbosity)
        elif isinstance(event, AgentThinkingEvent):
            await self._print_agent_thinking(event, timestamp, verbosity)
        elif isinstance(event, AgentCompleteEvent):
            await self._print_agent_complete(event, timestamp, verbosity)
        elif isinstance(event, ToolCallEvent):
            await self._print_tool_call(event, timestamp, verbosity)
        elif isinstance(event, BranchEvent):
            await self._print_branch_event(event, timestamp, verbosity)
        elif isinstance(event, ParallelGroupEvent):
            await self._print_parallel_group(event, timestamp, verbosity)
        elif isinstance(event, UserInteractionEvent):
            await self._print_user_interaction(event, timestamp, verbosity)
        elif isinstance(event, FinalResponseEvent):
            await self._print_final_response(event, timestamp, verbosity)
        # Planning events
        elif isinstance(event, PlanCreatedEvent):
            await self._print_plan_created(event, timestamp, verbosity)
        elif isinstance(event, PlanUpdatedEvent):
            await self._print_plan_updated(event, timestamp, verbosity)
        elif isinstance(event, PlanItemAddedEvent):
            await self._print_plan_item_added(event, timestamp, verbosity)
        elif isinstance(event, PlanItemRemovedEvent):
            await self._print_plan_item_removed(event, timestamp, verbosity)
        elif isinstance(event, PlanClearedEvent):
            await self._print_plan_cleared(event, timestamp, verbosity)

    async def _print_agent_start(self, event: 'AgentStartEvent', ts: str, verbosity: int):
        """Print agent start event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        # Check if this agent is in a parallel group
        in_parallel = event.agent_name in self.agent_to_group

        # Print section header if agent changed
        if event.agent_name != self.last_agent_name:
            if in_parallel:
                print(f"\n  {c['bold']}{c['blue']}┌─ {event.agent_name} ─┐{c['reset']}")
            else:
                print(f"\n{c['bold']}{c['blue']}━━━ {event.agent_name} ━━━{c['reset']}")
            self.last_agent_name = event.agent_name

        if verbosity >= VerbosityLevel.NORMAL:
            indent = "    " if in_parallel else "  "
            status = f"{c['green']}● Starting{c['reset']}"
            print(f"{indent}{ts} {status}")

        if verbosity == VerbosityLevel.VERBOSE and event.request_summary:
            indent = "    " if in_parallel else "  "
            print(f"{indent}  {c['gray']}Request: {event.request_summary[:100]}...{c['reset']}")

    async def _print_agent_thinking(self, event: 'AgentThinkingEvent', ts: str, verbosity: int):
        """Print agent thinking event."""
        from ..config import VerbosityLevel
        if verbosity < VerbosityLevel.NORMAL:
            return

        c = self.colors

        # Check if this agent is in a parallel group
        in_parallel = event.agent_name in self.agent_to_group

        # Print section header if agent changed
        if event.agent_name != self.last_agent_name:
            if in_parallel:
                print(f"\n  {c['bold']}{c['blue']}┌─ {event.agent_name} ─┐{c['reset']}")
            else:
                print(f"\n{c['bold']}{c['blue']}━━━ {event.agent_name} ━━━{c['reset']}")
            self.last_agent_name = event.agent_name

        indent = "    " if in_parallel else "  "
        thought = event.thought[:200] + "..." if len(event.thought) > 200 else event.thought
        print(f"{indent}{c['dim']}💭 Thinking:{c['reset']} {c['gray']}{thought}{c['reset']}")

        if verbosity >= VerbosityLevel.VERBOSE and event.action_type:
            print(f"{indent}  {c['gray']}→ Action: {event.action_type}{c['reset']}")

    async def _print_agent_complete(self, event: 'AgentCompleteEvent', ts: str, verbosity: int):
        """Print agent completion."""
        from ..config import VerbosityLevel
        c = self.colors

        # Check if this agent is in a parallel group
        in_parallel = event.agent_name in self.agent_to_group

        # Print section header if agent changed
        if event.agent_name != self.last_agent_name:
            if in_parallel:
                print(f"\n  {c['bold']}{c['blue']}┌─ {event.agent_name} ─┐{c['reset']}")
            else:
                print(f"\n{c['bold']}{c['blue']}━━━ {event.agent_name} ━━━{c['reset']}")
            self.last_agent_name = event.agent_name

        if event.success:
            status = f"{c['green']}✓ Completed{c['reset']}"
        else:
            status = f"{c['red']}✗ Failed{c['reset']}"

        if verbosity == VerbosityLevel.QUIET:
            if not event.success:
                indent = "    " if in_parallel else "  "
                print(f"{indent}{event.agent_name}: {status}")
        else:
            indent = "    " if in_parallel else "  "
            duration = f" ({event.duration:.2f}s)" if self.config.show_timings and event.duration else ""
            print(f"{indent}{ts} {status}{duration}")

            if verbosity >= VerbosityLevel.NORMAL and event.next_action:
                print(f"{indent}  {c['cyan']}→ Next: {event.next_action}{c['reset']}")

            if not event.success and event.error:
                print(f"{indent}  {c['red']}Error: {event.error}{c['reset']}")

    async def _print_tool_call(self, event: 'ToolCallEvent', ts: str, verbosity: int):
        """Print tool call event."""
        from ..config import VerbosityLevel
        if verbosity < VerbosityLevel.NORMAL:
            return

        c = self.colors

        # Check if this agent is in a parallel group
        in_parallel = event.agent_name in self.agent_to_group

        # Print section header if agent changed
        if event.agent_name != self.last_agent_name:
            if in_parallel:
                print(f"\n  {c['bold']}{c['blue']}┌─ {event.agent_name} ─┐{c['reset']}")
            else:
                print(f"\n{c['bold']}{c['blue']}━━━ {event.agent_name} ━━━{c['reset']}")
            self.last_agent_name = event.agent_name

        indent = "    " if in_parallel else "  "

        if event.status == "started":
            if event.reasoning:
                reasoning_text = event.reasoning[:200] + "..." if len(event.reasoning) > 200 else event.reasoning
                print(f"{indent}{c['dim']}💭 Thinking:{c['reset']} {c['gray']}{reasoning_text}{c['reset']}")

            print(f"{indent}{ts} {c['cyan']}🔧 Tool:{c['reset']} {event.tool_name}")

            if verbosity >= VerbosityLevel.VERBOSE and event.arguments:
                args = str(event.arguments)[:100]
                print(f"{indent}  {c['gray']}Args: {args}{c['reset']}")
        elif event.status == "completed":
            if verbosity >= VerbosityLevel.VERBOSE:
                duration = f" ({event.duration:.2f}s)" if event.duration else ""
                print(f"{indent}{ts} {c['green']}✓ Tool completed{c['reset']}{duration}")
        else:  # failed
            print(f"{indent}{ts} {c['red']}✗ Tool failed{c['reset']}")

    async def _print_branch_event(self, event: 'BranchEvent', ts: str, verbosity: int):
        """Print branch event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return  # Don't show branch events in quiet mode

        # Format branch info
        branch_info = f"{event.branch_name} ({event.branch_type})"
        if event.is_parallel:
            branch_info = f"⚡ {branch_info}"

        print(f"{ts} {c['gray']}Branch: {branch_info} → {event.status}{c['reset']}")

    async def _print_parallel_group(self, event: 'ParallelGroupEvent', ts: str, verbosity: int):
        """Print parallel execution group and track parallel agents."""
        from ..config import VerbosityLevel
        c = self.colors

        if event.status == "started":
            # Track this parallel group
            self.parallel_groups[event.group_id] = set(event.agent_names)
            self.active_parallel_groups.add(event.group_id)

            # Map each agent to this group
            for agent_name in event.agent_names:
                self.agent_to_group[agent_name] = event.group_id

            agents_str = ", ".join(event.agent_names[:3])
            if len(event.agent_names) > 3:
                agents_str += f" +{len(event.agent_names) - 3} more"

            print(f"\n{c['bold']}{c['yellow']}⚡ Parallel Execution{c['reset']}")
            print(f"{ts} Agents: {agents_str}")

        elif event.status == "executing" and verbosity >= VerbosityLevel.NORMAL:
            # Progress bar
            progress = event.completed_count / event.total_count if event.total_count > 0 else 0
            bar_length = 20
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)

            print(f"{ts} Progress: [{bar}] {event.completed_count}/{event.total_count}")

        elif event.status == "completed":
            # Clean up parallel group tracking
            if event.group_id in self.active_parallel_groups:
                self.active_parallel_groups.remove(event.group_id)

                # Remove agent mappings for this group
                if event.group_id in self.parallel_groups:
                    for agent_name in self.parallel_groups[event.group_id]:
                        if agent_name in self.agent_to_group and self.agent_to_group[agent_name] == event.group_id:
                            del self.agent_to_group[agent_name]

                # Clean up group tracking
                if event.group_id in self.parallel_groups:
                    del self.parallel_groups[event.group_id]

            print(f"{ts} {c['green']}✓ All parallel branches completed{c['reset']}")

    async def _print_user_interaction(self, event: 'UserInteractionEvent', ts: str, verbosity: int):
        """Print user interaction request."""
        c = self.colors

        print(f"\n{c['bold']}{c['yellow']}⚠ User Input Required{c['reset']}")
        print(f"{ts} {event.prompt}")

        if event.options:
            print("Options:")
            for i, option in enumerate(event.options, 1):
                print(f"  {i}. {option}")

    async def _print_final_response(self, event: 'FinalResponseEvent', ts: str, verbosity: int):
        """Print final response."""
        from ..config import VerbosityLevel
        c = self.colors

        # Show different header based on success/failure
        if event.success:
            print(f"\n{c['bold']}{c['green']}═══ Workflow Complete ═══{c['reset']}")
        else:
            print(f"\n{c['bold']}{c['red']}═══ Workflow Failed ═══{c['reset']}")

        if verbosity >= VerbosityLevel.NORMAL:
            status = "Success" if event.success else "Failed"
            print(f"Status: {status}")
            print(f"Total Steps: {event.total_steps}")

            if self.config.show_timings:
                print(f"Duration: {event.total_duration:.2f}s")

        if verbosity == VerbosityLevel.VERBOSE:
            print(f"\n{c['bold']}Summary:{c['reset']}")
            # Handle both old and new field names
            summary = getattr(event, 'final_response', getattr(event, 'response_summary', ''))
            print(summary)

    # ==================== Planning Events ====================

    async def _print_plan_created(self, event, ts: str, verbosity: int):
        """Print plan created event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        print(f"\n{c['bold']}{c['cyan']}📋 Plan Created{c['reset']}")
        print(f"  {ts} Agent: {event.agent_name}")
        if event.goal:
            print(f"  {c['gray']}Goal: {event.goal}{c['reset']}")
        print(f"  {c['gray']}Items: {event.item_count}{c['reset']}")

        if verbosity >= VerbosityLevel.VERBOSE and event.item_titles:
            for i, title in enumerate(event.item_titles[:5], 1):
                print(f"    {i}. {title}")
            if len(event.item_titles) > 5:
                print(f"    ... and {len(event.item_titles) - 5} more")

    async def _print_plan_updated(self, event, ts: str, verbosity: int):
        """Print plan item updated event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        # At NORMAL verbosity, skip non-status updates (show them only at VERBOSE)
        if event.new_status is None and verbosity < VerbosityLevel.VERBOSE:
            return

        # Determine icon and color based on status change
        if event.new_status == "completed":
            icon = "✓"
            color = c['green']
            status_text = "Completed"
        elif event.new_status == "in_progress":
            icon = "▶"
            color = c['cyan']
            status_text = event.active_form or "In Progress"
        elif event.new_status == "blocked":
            icon = "⚠"
            color = c['yellow']
            status_text = "Blocked"
        elif event.new_status is None:
            # Non-status update (only shown at VERBOSE)
            icon = "✎"
            color = c['gray']
            updated_fields = event.metadata.get("updated_fields", []) if event.metadata else []
            status_text = f"Updated: {', '.join(updated_fields)}" if updated_fields else "Updated"
        else:
            icon = "○"
            color = c['gray']
            status_text = event.new_status or "Updated"

        print(f"  {ts} {color}{icon} {event.item_title}: {status_text}{c['reset']}")

    async def _print_plan_item_added(self, event, ts: str, verbosity: int):
        """Print plan item added event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        print(f"  {ts} {c['cyan']}+ Added to plan: {event.item_title} (#{event.position}){c['reset']}")

    async def _print_plan_item_removed(self, event, ts: str, verbosity: int):
        """Print plan item removed event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        print(f"  {ts} {c['gray']}- Removed from plan: {event.item_title}{c['reset']}")

    async def _print_plan_cleared(self, event, ts: str, verbosity: int):
        """Print plan cleared event."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        reason_text = f" ({event.reason})" if event.reason else ""
        print(f"  {ts} {c['gray']}📋 Plan cleared{reason_text}{c['reset']}")


class PrefixedCLIChannel(ChannelAdapter):
    """
    CLI channel with dynamic agent name prefixes for clear identification.
    Simple, elegant, and works with terminal's streaming nature.
    Completely generic - no hardcoded agent names or types.
    """

    def __init__(self, config: 'StatusConfig'):
        super().__init__("prefixed_cli", config.cli_output)
        self.config = config
        self.use_colors = config.cli_colors and sys.stdout.isatty()

        # Dynamic configuration
        self.agent_width = getattr(config, 'prefix_width', 20)
        self.show_prefixes = getattr(config, 'show_agent_prefixes', True)

        # Dynamic agent tracking
        self.agent_colors: Dict[str, str] = {}  # Assigned dynamically
        # Expanded palette with research-based selection for maximum visibility
        self.color_palette = [
            'cyan',           # Light blue, high visibility
            'magenta',        # Purple/pink, very distinct
            'yellow',         # Bright and visible
            'blue',           # Standard blue
            'red',            # High contrast (for agents, not errors)
            'bright_cyan',    # Brighter variant
            'bright_magenta', # Brighter purple
            'bright_yellow',  # Extra bright
            'bright_blue',    # Bright blue
            'bright_red',     # Bright red
            'bright_green',   # Different from User's green
            'bright_white'    # Very bright, good contrast
        ]
        self.color_index = 0

        # State tracking
        self.start_time = time.time()
        self.last_agent_name: Optional[str] = None

        # Parallel group tracking (for special formatting)
        self.parallel_groups: Dict[str, Set[str]] = {}
        self.agent_to_group: Dict[str, str] = {}

        # User interaction manager reference
        self.interaction_manager = None  # Will be set by Orchestra

        # ANSI color codes
        self.colors = self._get_colors() if self.use_colors else self._get_no_colors()

    def _get_colors(self) -> Dict[str, str]:
        """Get ANSI color codes."""
        return {
            # Standard colors
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'red': '\033[31m',
            'gray': '\033[90m',
            'cyan': '\033[36m',
            'magenta': '\033[35m',
            'white': '\033[37m',
            # Bright colors for more distinction
            'bright_red': '\033[91m',
            'bright_green': '\033[92m',
            'bright_yellow': '\033[93m',
            'bright_blue': '\033[94m',
            'bright_magenta': '\033[95m',
            'bright_cyan': '\033[96m',
            'bright_white': '\033[97m'
        }

    def _get_no_colors(self) -> Dict[str, str]:
        """Return empty strings when colors are disabled."""
        return {k: '' for k in ['reset', 'bold', 'dim', 'green', 'yellow',
                                'blue', 'red', 'gray', 'cyan', 'magenta', 'white',
                                'bright_red', 'bright_green', 'bright_yellow',
                                'bright_blue', 'bright_magenta', 'bright_cyan',
                                'bright_white']}

    def _get_agent_color(self, agent_name: str) -> str:
        """
        Dynamically assign colors to agents as they appear.
        No hardcoding based on agent type.
        """
        if agent_name not in self.agent_colors:
            # Assign next color from palette
            color_name = self.color_palette[self.color_index % len(self.color_palette)]
            self.agent_colors[agent_name] = self.colors[color_name]
            self.color_index += 1

        return self.agent_colors[agent_name]

    def _format_prefix(self, source: str) -> str:
        """
        Create prefix for any source (agent, system, user).
        Completely generic - works with any name.
        """
        if not self.show_prefixes:
            return ""

        # Handle special sources
        if source == "System":
            color = self.colors['bold'] + self.colors['gray']
        elif source == "User":
            color = self.colors['bold'] + self.colors['green']
        else:
            # Regular agent - use dynamic color
            color = self._get_agent_color(source)

        reset = self.colors['reset']

        # Dynamic truncation and alignment
        truncated = source[:self.agent_width]
        alignment = getattr(self.config, 'prefix_alignment', 'left')

        if alignment == 'left':
            formatted = truncated.ljust(self.agent_width)
        elif alignment == 'right':
            formatted = truncated.rjust(self.agent_width)
        else:  # center
            formatted = truncated.center(self.agent_width)

        return f"{color}[{formatted}]{reset} "

    async def send(self, event: 'StatusEvent') -> None:
        """Process and display status event with agent prefix."""
        from .events import (
            AgentStartEvent, AgentThinkingEvent, AgentCompleteEvent,
            ToolCallEvent, BranchEvent, ParallelGroupEvent,
            UserInteractionEvent, FinalResponseEvent,
            PlanCreatedEvent, PlanUpdatedEvent, PlanItemAddedEvent,
            PlanItemRemovedEvent, PlanClearedEvent
        )
        from ..config import VerbosityLevel

        # Get timestamp if configured
        if self.config.show_timings:
            elapsed = time.time() - self.start_time
            timestamp = f"[{elapsed:6.2f}s]"
        else:
            timestamp = ""

        # Get verbosity level
        verbosity = self.config.verbosity or VerbosityLevel.NORMAL

        # Handle different event types
        if isinstance(event, AgentStartEvent):
            await self._print_agent_start(event, timestamp, verbosity)
        elif isinstance(event, AgentThinkingEvent):
            await self._print_agent_thinking(event, timestamp, verbosity)
        elif isinstance(event, AgentCompleteEvent):
            await self._print_agent_complete(event, timestamp, verbosity)
        elif isinstance(event, ToolCallEvent):
            await self._print_tool_call(event, timestamp, verbosity)
        elif isinstance(event, ParallelGroupEvent):
            await self._print_parallel_group(event, timestamp, verbosity)
        # Removed handling for UserInteractionRequestEvent, UserInteractionResponseEvent, FollowUpRequestEvent
        # These are now handled by CommunicationManager per separation of concerns
        elif isinstance(event, FinalResponseEvent):
            await self._print_final_response(event, timestamp, verbosity)
        elif isinstance(event, BranchEvent):
            await self._print_branch_event(event, timestamp, verbosity)
        elif isinstance(event, UserInteractionEvent):
            await self._print_user_interaction(event, timestamp, verbosity)
        # Planning events
        elif isinstance(event, PlanCreatedEvent):
            await self._print_plan_created(event, timestamp, verbosity)
        elif isinstance(event, PlanUpdatedEvent):
            await self._print_plan_updated(event, timestamp, verbosity)
        elif isinstance(event, PlanItemAddedEvent):
            await self._print_plan_item_added(event, timestamp, verbosity)
        elif isinstance(event, PlanItemRemovedEvent):
            await self._print_plan_item_removed(event, timestamp, verbosity)
        elif isinstance(event, PlanClearedEvent):
            await self._print_plan_cleared(event, timestamp, verbosity)

    async def _print_agent_start(self, event: 'AgentStartEvent', ts: str, verbosity: int):
        """Print agent start event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        prefix = self._format_prefix(event.agent_name)

        # Add section separator if agent changed (optional)
        if self.last_agent_name != event.agent_name and self.last_agent_name is not None:
            if not self.show_prefixes:  # Only show separator if not using prefixes
                print(f"\n{c['dim']}{'─' * 40}{c['reset']}")

        self.last_agent_name = event.agent_name

        # Format and print the start message
        status = f"{c['green']}● Starting{c['reset']}"
        print(f"{prefix}{ts} {status}")

        # Show request summary in verbose mode
        if verbosity == VerbosityLevel.VERBOSE and event.request_summary:
            request = event.request_summary[:80] + "..." if len(event.request_summary) > 80 else event.request_summary
            print(f"{prefix}    {c['gray']}Request: {request}{c['reset']}")

    async def _print_agent_thinking(self, event: 'AgentThinkingEvent', ts: str, verbosity: int):
        """Print agent thinking event with prefix."""
        from ..config import VerbosityLevel
        if verbosity < VerbosityLevel.NORMAL:
            return

        c = self.colors
        prefix = self._format_prefix(event.agent_name)

        thought = event.thought[:200] + "..." if len(event.thought) > 200 else event.thought
        print(f"{prefix}{c['dim']}💭 Thinking:{c['reset']} {c['gray']}{thought}{c['reset']}")

        if verbosity >= VerbosityLevel.VERBOSE and event.action_type:
            print(f"{prefix}    {c['gray']}→ Action: {event.action_type}{c['reset']}")

    async def _print_agent_complete(self, event: 'AgentCompleteEvent', ts: str, verbosity: int):
        """Print agent completion with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        prefix = self._format_prefix(event.agent_name)

        if event.success:
            status = f"{c['green']}✓ Completed{c['reset']}"
        else:
            status = f"{c['red']}✗ Failed{c['reset']}"

        if verbosity == VerbosityLevel.QUIET:
            if not event.success:
                print(f"{prefix}{status}")
        else:
            duration = f" ({event.duration:.2f}s)" if self.config.show_timings and event.duration else ""
            print(f"{prefix}{ts} {status}{duration}")

            if verbosity >= VerbosityLevel.NORMAL and event.next_action:
                print(f"{prefix}    {c['cyan']}→ Next: {event.next_action}{c['reset']}")

            if not event.success and event.error:
                print(f"{prefix}    {c['red']}Error: {event.error}{c['reset']}")

    async def _print_tool_call(self, event: 'ToolCallEvent', ts: str, verbosity: int):
        """Print tool call event with prefix."""
        from ..config import VerbosityLevel
        if verbosity < VerbosityLevel.NORMAL:
            return

        c = self.colors
        prefix = self._format_prefix(event.agent_name)

        if event.status == "started":
            if event.reasoning:
                reasoning_text = event.reasoning[:200] + "..." if len(event.reasoning) > 200 else event.reasoning
                print(f"{prefix}{c['dim']}💭 Thinking:{c['reset']} {c['gray']}{reasoning_text}{c['reset']}")

            print(f"{prefix}{ts} {c['cyan']}🔧 Tool:{c['reset']} {event.tool_name}")

            if verbosity >= VerbosityLevel.VERBOSE and event.arguments:
                args = str(event.arguments)[:100]
                print(f"{prefix}    {c['gray']}Args: {args}{c['reset']}")
        elif event.status == "completed":
            if verbosity >= VerbosityLevel.VERBOSE:
                duration = f" ({event.duration:.2f}s)" if event.duration else ""
                print(f"{prefix}{ts} {c['green']}✓ Tool completed{c['reset']}{duration}")
        else:  # failed
            print(f"{prefix}{ts} {c['red']}✗ Tool failed{c['reset']}")

    async def _print_parallel_group(self, event: 'ParallelGroupEvent', ts: str, verbosity: int):
        """Print parallel execution group."""
        from ..config import VerbosityLevel
        c = self.colors

        if event.status == "started":
            # Track parallel group
            self.parallel_groups[event.group_id] = set(event.agent_names)
            for agent_name in event.agent_names:
                self.agent_to_group[agent_name] = event.group_id

            agents_str = ", ".join(event.agent_names[:3])
            if len(event.agent_names) > 3:
                agents_str += f" +{len(event.agent_names) - 3} more"

            # Use system prefix for group events
            prefix = self._format_prefix("System")
            print(f"\n{prefix}{c['bold']}{c['yellow']}⚡ Parallel Execution{c['reset']}")
            print(f"{prefix}{ts} Agents: {agents_str}")

        elif event.status == "completed":
            # Clean up tracking
            if event.group_id in self.parallel_groups:
                for agent_name in self.parallel_groups[event.group_id]:
                    if agent_name in self.agent_to_group:
                        del self.agent_to_group[agent_name]
                del self.parallel_groups[event.group_id]

            prefix = self._format_prefix("System")
            print(f"{prefix}{ts} {c['green']}✓ All parallel branches completed{c['reset']}")

    async def _print_branch_event(self, event: 'BranchEvent', ts: str, verbosity: int):
        """Print branch event with prefix."""
        from ..config import VerbosityLevel
        if verbosity == VerbosityLevel.QUIET:
            return

        c = self.colors
        prefix = self._format_prefix("System")

        branch_info = f"{event.branch_name} ({event.branch_type})"
        if event.is_parallel:
            branch_info = f"⚡ {branch_info}"

        print(f"{prefix}{ts} {c['gray']}Branch: {branch_info} → {event.status}{c['reset']}")

    async def _print_user_interaction(self, event: 'UserInteractionEvent', ts: str, verbosity: int):
        """Print user interaction with prefix."""
        c = self.colors
        prefix = self._format_prefix(event.agent_name)

        print(f"\n{prefix}{c['bold']}{c['yellow']}⚠ User Input Required{c['reset']}")
        print(f"{prefix}{ts} {event.prompt}")

        if event.options:
            print(f"{prefix}Options:")
            for i, option in enumerate(event.options, 1):
                print(f"{prefix}  {i}. {option}")

    # Methods removed due to separation of concerns:
    # - _print_user_interaction_request: Full content display belongs in CommunicationManager
    # - _print_user_interaction_response: User input handling is CommunicationManager's responsibility
    # - _print_follow_up_request: Follow-up workflow is CommunicationManager's domain

    async def _print_final_response(self, event: 'FinalResponseEvent', ts: str, verbosity: int):
        """Print final response."""
        from ..config import VerbosityLevel
        c = self.colors

        # Final response doesn't belong to a specific agent
        prefix = self._format_prefix("System")

        # Show different header based on success/failure
        if event.success:
            print(f"\n{prefix}{c['bold']}{c['green']}═══ Workflow Complete ═══{c['reset']}")
        else:
            print(f"\n{prefix}{c['bold']}{c['red']}═══ Workflow Failed ═══{c['reset']}")

        if verbosity >= VerbosityLevel.NORMAL:
            status = "Success" if event.success else "Failed"
            print(f"{prefix}Status: {status}")
            print(f"{prefix}Total Steps: {event.total_steps}")

            if self.config.show_timings:
                print(f"{prefix}Duration: {event.total_duration:.2f}s")

    # ==================== Planning Events ====================

    async def _print_plan_created(self, event, ts: str, verbosity: int):
        """Print plan created event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        prefix = self._format_prefix(event.agent_name)
        print(f"\n{prefix}{c['bold']}{c['cyan']}📋 Plan Created{c['reset']}")
        if event.goal:
            print(f"{prefix}  {c['gray']}Goal: {event.goal}{c['reset']}")
        print(f"{prefix}  {c['gray']}Items: {event.item_count}{c['reset']}")

        if verbosity >= VerbosityLevel.VERBOSE and event.item_titles:
            for i, title in enumerate(event.item_titles[:5], 1):
                print(f"{prefix}    {i}. {title}")
            if len(event.item_titles) > 5:
                print(f"{prefix}    ... and {len(event.item_titles) - 5} more")

    async def _print_plan_updated(self, event, ts: str, verbosity: int):
        """Print plan item updated event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        # At NORMAL verbosity, skip non-status updates (show them only at VERBOSE)
        if event.new_status is None and verbosity < VerbosityLevel.VERBOSE:
            return

        prefix = self._format_prefix(event.agent_name)

        # Determine icon and color based on status change
        if event.new_status == "completed":
            icon = "✓"
            color = c['green']
            status_text = "Completed"
        elif event.new_status == "in_progress":
            icon = "▶"
            color = c['cyan']
            status_text = event.active_form or "In Progress"
        elif event.new_status == "blocked":
            icon = "⚠"
            color = c['yellow']
            status_text = "Blocked"
        elif event.new_status is None:
            # Non-status update (only shown at VERBOSE)
            icon = "✎"
            color = c['gray']
            updated_fields = event.metadata.get("updated_fields", []) if event.metadata else []
            status_text = f"Updated: {', '.join(updated_fields)}" if updated_fields else "Updated"
        else:
            icon = "○"
            color = c['gray']
            status_text = event.new_status or "Updated"

        print(f"{prefix}{ts} {color}{icon} {event.item_title}: {status_text}{c['reset']}")

    async def _print_plan_item_added(self, event, ts: str, verbosity: int):
        """Print plan item added event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        prefix = self._format_prefix(event.agent_name)
        print(f"{prefix}{ts} {c['cyan']}+ Added to plan: {event.item_title} (#{event.position}){c['reset']}")

    async def _print_plan_item_removed(self, event, ts: str, verbosity: int):
        """Print plan item removed event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        prefix = self._format_prefix(event.agent_name)
        print(f"{prefix}{ts} {c['gray']}- Removed from plan: {event.item_title}{c['reset']}")

    async def _print_plan_cleared(self, event, ts: str, verbosity: int):
        """Print plan cleared event with prefix."""
        from ..config import VerbosityLevel
        c = self.colors

        if verbosity == VerbosityLevel.QUIET:
            return

        prefix = self._format_prefix(event.agent_name)
        reason_text = f" ({event.reason})" if event.reason else ""
        print(f"{prefix}{ts} {c['gray']}📋 Plan cleared{reason_text}{c['reset']}")


class StatusWebChannel(ChannelAdapter):
    """
    Channel adapter that bridges StatusManager to WebChannel for web client support.

    This adapter receives status events (including planning events) and forwards them
    to the WebChannel for WebSocket push and API polling.

    Usage:
        from marsys.coordination.communication.channels import WebChannel
        from marsys.coordination.status.channels import StatusWebChannel

        web_channel = WebChannel()
        status_web = StatusWebChannel(web_channel)
        status_manager.add_channel(status_web)
    """

    def __init__(self, web_channel, enabled: bool = True):
        """
        Initialize StatusWebChannel.

        Args:
            web_channel: WebChannel instance to forward events to
            enabled: Whether the channel is enabled
        """
        super().__init__("status_web", enabled)
        self.web_channel = web_channel

    async def send(self, event: 'StatusEvent') -> None:
        """Convert and forward status event to WebChannel."""
        if not self.web_channel:
            return

        # Convert event to serializable dict
        event_data = self._event_to_dict(event)

        # Forward to WebChannel
        await self.web_channel.push_status_event(event_data)

    def _event_to_dict(self, event: 'StatusEvent') -> dict:
        """Convert a StatusEvent to a serializable dictionary."""
        from dataclasses import asdict, is_dataclass

        if is_dataclass(event):
            data = asdict(event)
        else:
            # Fallback for non-dataclass events
            data = {
                "session_id": getattr(event, "session_id", None),
                "event_id": getattr(event, "event_id", None),
                "timestamp": getattr(event, "timestamp", None),
            }

        # Add event type
        data["event_type"] = event.__class__.__name__

        return data
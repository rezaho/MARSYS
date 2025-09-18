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
            UserInteractionEvent, FinalResponseEvent
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
        if verbosity < VerbosityLevel.VERBOSE:
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
        print(f"{indent}{ts} {c['yellow']}💭 Thinking:{c['reset']} {thought}")

        if event.action_type:
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
        if verbosity < VerbosityLevel.VERBOSE:
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
            print(f"{indent}{ts} {c['cyan']}🔧 Tool:{c['reset']} {event.tool_name}")
            if event.arguments:
                args = str(event.arguments)[:100]
                print(f"{indent}  {c['gray']}Args: {args}{c['reset']}")
        elif event.status == "completed":
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

        print(f"\n{c['bold']}{c['green']}═══ Workflow Complete ═══{c['reset']}")

        if verbosity >= VerbosityLevel.NORMAL:
            status = "Success" if event.success else "Failed"
            print(f"Status: {status}")
            print(f"Total Steps: {event.total_steps}")

            if self.config.show_timings:
                print(f"Duration: {event.total_duration:.2f}s")

        if verbosity == VerbosityLevel.VERBOSE:
            print(f"\n{c['bold']}Summary:{c['reset']}")
            print(event.response_summary)
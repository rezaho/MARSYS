"""
Enhanced terminal channel with Rich formatting and advanced input capabilities.
"""

import asyncio
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import readline  # For input history

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text
from rich.theme import Theme
from rich.table import Table
from rich.layout import Layout

from .terminal import TerminalChannel
from ..core import UserInteraction

logger = logging.getLogger(__name__)


class EnhancedTerminalChannel(TerminalChannel):
    """
    Enhanced terminal channel with Rich formatting and advanced input.

    Features:
    - Beautiful bordered panels for interactions
    - Prefix-based output like StatusManager
    - Advanced input with history and editing
    - Modern color theming
    - Graceful fallback for non-TTY environments
    """

    def __init__(
        self,
        channel_id: str = "terminal",
        use_rich: bool = True,
        theme_name: str = "modern",
        prefix_width: int = 20,
        show_timestamps: bool = True
    ):
        super().__init__(channel_id)

        # Configuration
        self.use_rich = use_rich and sys.stdout.isatty()  # Auto-detect TTY
        self.prefix_width = prefix_width
        self.show_timestamps = show_timestamps
        self.start_time = time.time()

        # Initialize Rich console if available
        if self.use_rich:
            try:
                self.console = Console(theme=self._create_theme(theme_name))

                # Configure readline for input history
                readline.parse_and_bind('tab: complete')
                readline.parse_and_bind('set editing-mode emacs')  # Default editing mode

                # Dynamic agent color tracking (like PrefixedCLIChannel)
                self.agent_colors = {}
                self.color_palette = [
                    'cyan', 'magenta', 'yellow', 'blue', 'bright_cyan',
                    'bright_magenta', 'bright_yellow', 'bright_blue',
                    'bright_green', 'bright_white'
                ]
                self.color_index = 0
            except Exception as e:
                logger.warning(f"Failed to initialize Rich console: {e}")
                self.use_rich = False
                self.console = None
        else:
            self.console = None

    def _create_theme(self, theme_name: str) -> Theme:
        """Create Rich theme based on theme name."""
        themes = {
            "modern": {
                "user_input": "bright_cyan bold",
                "agent_message": "bright_blue",
                "system_message": "bright_yellow",
                "border": "bright_magenta",
                "prompt": "bright_green bold",
                "error": "bright_red bold",
                "success": "bright_green",
                "info": "bright_cyan",
                "warning": "bright_yellow",
                "timestamp": "dim white",
                "prefix": "bold"
            },
            "classic": {
                "user_input": "cyan",
                "agent_message": "blue",
                "system_message": "yellow",
                "border": "white",
                "prompt": "green",
                "error": "red",
                "success": "green",
                "info": "cyan",
                "warning": "yellow",
                "timestamp": "dim white",
                "prefix": "bold"
            },
            "minimal": {
                "user_input": "white bold",
                "agent_message": "white",
                "system_message": "white dim",
                "border": "white dim",
                "prompt": "white bold",
                "error": "white bold",
                "success": "white",
                "info": "white",
                "warning": "white",
                "timestamp": "white dim",
                "prefix": "white bold"
            }
        }

        theme_dict = themes.get(theme_name, themes["modern"])
        return Theme(theme_dict)

    def _get_agent_color(self, agent_name: str) -> str:
        """Get or assign color for an agent dynamically."""
        if agent_name not in self.agent_colors:
            if agent_name == "System":
                return "gray"
            elif agent_name == "User":
                return "green"
            else:
                # Assign next color from palette
                color = self.color_palette[self.color_index % len(self.color_palette)]
                self.agent_colors[agent_name] = color
                self.color_index += 1
                return color
        return self.agent_colors[agent_name]

    def _format_prefix(self, source: str, timestamp: Optional[float] = None) -> str:
        """Create formatted prefix for output."""
        if not self.use_rich:
            # Fallback to simple prefix
            prefix = f"[{source:<{self.prefix_width}}]"
            if self.show_timestamps and timestamp is not None:
                elapsed = timestamp - self.start_time
                prefix += f" [{elapsed:6.2f}s]"
            return prefix + " "

        # Rich formatted prefix
        color = self._get_agent_color(source)
        prefix_text = Text()

        # Agent name
        truncated = source[:self.prefix_width]
        formatted = truncated.ljust(self.prefix_width)
        prefix_text.append(f"[{formatted}]", style=f"bold {color}")

        # Timestamp
        if self.show_timestamps and timestamp is not None:
            elapsed = timestamp - self.start_time
            prefix_text.append(f" [{elapsed:6.2f}s]", style="dim white")

        return prefix_text

    async def send_interaction(self, interaction: UserInteraction) -> None:
        """Display interaction with Rich formatting."""
        self.current_interaction = interaction
        current_time = time.time()

        # Handle special error types with dedicated methods
        if interaction.interaction_type == "error_recovery":
            await self._send_fixable_error(interaction, current_time)
        elif interaction.interaction_type == "terminal_error":
            await self._send_terminal_error(interaction, current_time)
        elif self.use_rich:
            await self._send_rich_interaction(interaction, current_time)
        else:
            # Fallback to original formatting
            await super().send_interaction(interaction)

    async def _send_rich_interaction(self, interaction: UserInteraction, timestamp: float) -> None:
        """Display interaction using Rich panels and formatting."""
        # Determine interaction header
        if interaction.calling_agent == "System":
            if interaction.interaction_type == "question":
                header = "📝 USER INPUT REQUIRED"
                border_style = "bright_yellow"
            elif interaction.interaction_type == "task":
                header = "🚀 NEW TASK"
                border_style = "bright_green"
            elif interaction.interaction_type == "notification":
                header = "📢 SYSTEM NOTIFICATION"
                border_style = "bright_cyan"
            else:
                header = f"💬 {interaction.interaction_type.upper()}"
                border_style = "bright_blue"
        else:
            if interaction.interaction_type == "question":
                header = "🤔 QUESTION FROM AGENT"
                border_style = "bright_magenta"
            elif interaction.interaction_type == "choice":
                header = "📋 PLEASE CHOOSE AN OPTION"
                border_style = "bright_cyan"
            elif interaction.interaction_type == "confirmation":
                header = "✅ CONFIRMATION REQUIRED"
                border_style = "bright_green"
            else:
                header = f"💬 {interaction.interaction_type.upper()}"
                border_style = "bright_blue"

        # Create content for the panel
        content_parts = []

        # Add calling agent with prefix style
        if interaction.calling_agent:
            prefix = self._format_prefix(interaction.calling_agent, timestamp)
            if isinstance(prefix, Text):
                content_parts.append(prefix)
            else:
                content_parts.append(Text(prefix))

        # Process message content
        message = interaction.incoming_message
        if isinstance(message, dict):
            if "content" in message:
                content_parts.append(Text(message["content"]))

            # Show context if provided
            if "context" in message and message["context"]:
                content_parts.append(Text("\n"))
                context_table = Table(show_header=False, box=None, padding=(0, 1))
                context_table.add_column("Key", style="dim")
                context_table.add_column("Value")

                context = message["context"]
                if isinstance(context, dict):
                    for key, value in context.items():
                        context_table.add_row(str(key), str(value))
                else:
                    context_table.add_row("Context", str(context))

                content_parts.append(context_table)

            # Show options if this is a choice
            if "options" in message and message["options"]:
                content_parts.append(Text("\n"))
                options_text = Text("Options:\n", style="bold")
                for i, option in enumerate(message["options"], 1):
                    options_text.append(f"  {i}. {option}\n", style="cyan")
                content_parts.append(options_text)
        else:
            content_parts.append(Text(str(message)))

        # Combine all content
        if len(content_parts) == 1:
            panel_content = content_parts[0]
        else:
            # Create a group of renderables
            from rich.console import Group
            panel_content = Group(*content_parts)

        # Create and display the panel
        panel = Panel(
            panel_content,
            title=f"[bold]{header}[/]",
            border_style=border_style,
            padding=(1, 2),
            expand=False
        )

        if self.console:
            self.console.print()
            self.console.print(panel)
        else:
            # Fallback to parent's display method
            await super().send_interaction(interaction)

    async def get_response(self, interaction_id: str) -> Tuple[str, Any]:
        """Get user response with Rich input capabilities."""
        if not self.current_interaction or self.current_interaction.interaction_id != interaction_id:
            raise ValueError(f"No current interaction matching {interaction_id}")

        interaction = self.current_interaction

        try:
            # Handle error recovery and terminal errors specially
            if interaction.interaction_type == "error_recovery":
                return await self.get_error_recovery_choice(interaction_id)
            elif interaction.interaction_type == "terminal_error":
                return await self.get_terminal_acknowledgment(interaction_id)
            elif self.use_rich:
                return await self._get_rich_response(interaction_id)
            else:
                # Fallback to original implementation
                return await super().get_response(interaction_id)
        finally:
            self.current_interaction = None

    async def _get_rich_response(self, interaction_id: str) -> Tuple[str, Any]:
        """Get response using Rich prompts."""
        interaction = self.current_interaction

        # Handle different interaction types
        if interaction.interaction_type == "notification":
            # Use simple confirmation for notifications
            if self.console:
                self.console.print("[dim]Press Enter to continue...[/]")
            else:
                print("Press Enter to continue...")
            await self._async_input("")
            return (interaction_id, {"acknowledged": True})

        elif interaction.interaction_type == "choice":
            # Handle multiple choice with Rich
            options = self._get_options_from_message(interaction.incoming_message)
            if options:
                return await self._get_rich_choice(interaction_id, options)

        elif interaction.interaction_type == "confirmation":
            # Handle yes/no with Rich Confirm
            return await self._get_rich_confirmation(interaction_id)

        # Default: free-form text input with Rich Prompt
        return await self._get_rich_text_input(interaction_id)

    async def _get_rich_text_input(self, interaction_id: str) -> Tuple[str, Any]:
        """Get text input using Rich Prompt."""
        loop = asyncio.get_event_loop()

        # Create prompt string
        prompt_str = "[bright_green]💬[/] Your response"

        # Run Rich Prompt in executor to avoid blocking
        # Prompt.ask uses the default console or the one specified at creation
        response = await loop.run_in_executor(
            None,
            lambda: Prompt.ask(prompt_str, console=self.console)
        )

        # Validate response
        if not response.strip():
            if self.console:
                self.console.print("[error]❌ Please provide a response to continue.[/]")
            else:
                print("❌ Please provide a response to continue.")
            return await self._get_rich_text_input(interaction_id)

        return (interaction_id, response.strip())

    async def _get_rich_choice(self, interaction_id: str, options: list) -> Tuple[str, Any]:
        """Get choice using Rich IntPrompt."""
        loop = asyncio.get_event_loop()

        # Create choice prompt string
        prompt_str = f"[bright_cyan]📝[/] Enter your choice (1-{len(options)})"

        # Get choice with validation
        while True:
            choice = await loop.run_in_executor(
                None,
                lambda: IntPrompt.ask(
                    prompt_str,
                    console=self.console,
                    default=None,
                    show_default=False,
                    choices=list(range(1, len(options) + 1))
                )
            )

            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                return (interaction_id, {
                    "choice_index": choice - 1,
                    "choice_value": selected
                })

            if self.console:
                self.console.print(f"[error]❌ Please enter a number between 1 and {len(options)}[/]")
            else:
                print(f"❌ Please enter a number between 1 and {len(options)}")

    async def _get_rich_confirmation(self, interaction_id: str) -> Tuple[str, Any]:
        """Get confirmation using Rich Confirm."""
        loop = asyncio.get_event_loop()

        # Create confirmation prompt string
        prompt_str = "[bright_green]✅[/] Please confirm"

        # Get confirmation
        confirmed = await loop.run_in_executor(
            None,
            lambda: Confirm.ask(
                prompt_str,
                console=self.console,
                default=False  # default to No
            )
        )

        return (interaction_id, {"confirmed": confirmed})

    async def send_and_wait_for_response(self, interaction: UserInteraction) -> Tuple[str, Any]:
        """
        Send interaction and wait for response atomically.
        Maintains compatibility with original implementation.
        """
        async with self._interaction_lock:
            await self.send_interaction(interaction)
            return await self.get_response(interaction.interaction_id)

    async def _send_fixable_error(
        self,
        interaction: UserInteraction,
        timestamp: float
    ) -> None:
        """Display fixable error with retry options using Rich panels."""
        from rich.panel import Panel
        from rich.text import Text
        from rich.console import Group

        message = interaction.incoming_message
        if not isinstance(message, dict):
            # Fallback to regular interaction if message is not properly formatted
            await self._send_rich_interaction(interaction, timestamp)
            return

        title = message.get('title', '⚠️  Error - Action Required')
        content = message.get('content', '')

        # Build content parts
        content_parts = []

        # Add main content
        content_parts.append(Text(content))

        # Add options with colored styling
        if message.get('options'):
            content_parts.append(Text("\n"))
            options_text = Text()
            for i, option in enumerate(message['options'], 1):
                if option == 'Retry':
                    options_text.append(f"  [{i}] {option}\n", style="bold green")
                elif option == 'Skip':
                    options_text.append(f"  [{i}] {option}\n", style="bold yellow")
                elif option == 'Abort':
                    options_text.append(f"  [{i}] {option}\n", style="bold red")
                else:
                    options_text.append(f"  [{i}] {option}\n", style="bold cyan")
            content_parts.append(options_text)

        # Create panel with yellow border for fixable errors
        panel = Panel(
            Group(*content_parts) if len(content_parts) > 1 else content_parts[0],
            title=f"[bold yellow]{title}[/]",
            border_style="bold yellow",
            padding=(1, 2),
            expand=False
        )

        if self.console:
            self.console.print("\n")
            self.console.print(panel)
        else:
            # Fallback to basic terminal output if Rich is not available
            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
            print(content)
            if message.get('options'):
                print("\nOptions:")
                for i, option in enumerate(message['options'], 1):
                    print(f"  [{i}] {option}")
            print("=" * 80)

    async def _send_terminal_error(
        self,
        interaction: UserInteraction,
        timestamp: float
    ) -> None:
        """Display terminal error (no retry options) using Rich panels."""
        from rich.panel import Panel
        from rich.text import Text

        message = interaction.incoming_message
        if not isinstance(message, dict):
            # Fallback to regular interaction if message is not properly formatted
            await self._send_rich_interaction(interaction, timestamp)
            return

        title = message.get('title', '❌ Fatal Error')
        content = message.get('content', 'A fatal error occurred')

        # Build content with error icon and details
        content_text = Text(content, style="bold white")

        # Create panel with red border for terminal errors
        panel = Panel(
            content_text,
            title=f"[bold red]{title}[/]",
            border_style="bold red",
            padding=(1, 2),
            expand=False
        )

        if self.console:
            self.console.print("\n")
            self.console.print(panel)
        else:
            # Fallback to basic terminal output if Rich is not available
            print("\n" + "=" * 80)
            print(title)
            print("=" * 80)
            print(content)
            print("=" * 80)

    async def get_error_recovery_choice(self, interaction_id: str) -> Tuple[str, Any]:
        """Get user choice for error recovery (retry/skip/abort)."""
        # Use simple prompt for now (Rich prompt doesn't work well in async context)
        prompt_text = "\n💬 Enter your choice (1=Retry, 2=Skip, 3=Abort): "

        while True:
            choice = await self._async_input(prompt_text)
            choice = choice.strip().lower()

            # Accept various forms of input
            if choice in ["1", "retry", "r"]:
                choice = "retry"
                break
            elif choice in ["2", "skip", "s"]:
                choice = "skip"
                break
            elif choice in ["3", "abort", "a", ""]:
                choice = "abort"
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

        # Choice is already mapped to the standard action
        action = choice

        # Show confirmation
        if self.console:
            if action == "retry":
                self.console.print("[bold green]✓ Retrying after fixing the issue...[/]\n")
            elif action == "skip":
                self.console.print("[bold yellow]⚠ Skipping failed step...[/]\n")
            else:
                self.console.print("[bold red]✗ Aborting execution...[/]\n")
        else:
            # Fallback to basic terminal output
            if action == "retry":
                print("✓ Retrying after fixing the issue...\n")
            elif action == "skip":
                print("⚠ Skipping failed step...\n")
            else:
                print("✗ Aborting execution...\n")

        return (interaction_id, {"choice": action})

    async def get_terminal_acknowledgment(self, interaction_id: str) -> Tuple[str, Any]:
        """Get acknowledgment for terminal error (no choices)."""
        # Just wait for Enter key
        await self._async_input("\n[dim]Press Enter to exit...[/]")

        if self.console:
            self.console.print("[bold red]✗ Execution terminated due to fatal error[/]\n")
        else:
            print("✗ Execution terminated due to fatal error\n")

        return (interaction_id, {"acknowledged": True})
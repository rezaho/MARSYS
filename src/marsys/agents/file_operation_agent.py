"""
FileOperationAgent - Specialized agent for file and system operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from marsys.agents.agents import Agent
from marsys.environment.file_operations import FileOperationTools
from marsys.environment.file_operations.config import FileOperationConfig
from marsys.environment.shell_tools import ShellTools
from marsys.models.models import ModelConfig

if TYPE_CHECKING:
    from marsys.environment.filesystem import RunFileSystem

logger = logging.getLogger(__name__)


class FileOperationAgent(Agent):
    """Specialized agent for file operations and optional shell commands."""

    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        enable_shell: bool = False,
        working_directory: Optional[str] = None,
        base_directory: Optional[Path] = None,
        allowed_shell_commands: Optional[list] = None,
        blocked_shell_patterns: Optional[list] = None,
        shell_timeout_default: int = 30,
        filesystem: Optional["RunFileSystem"] = None,
        memory_config=None,
        compaction_model_config=None,
        **kwargs
    ):
        """
        Initialize FileOperationAgent.

        Args:
            model_config: Model configuration
            name: Agent name
            goal: Agent goal (default: auto-generated)
            instruction: Agent instruction (default: auto-generated)
            enable_shell: Whether to enable shell commands
            working_directory: Working directory for shell commands
            base_directory: Base directory for file operations
            allowed_shell_commands: Whitelist of allowed shell commands
            blocked_shell_patterns: Patterns to block in shell commands
            shell_timeout_default: Default timeout for shell commands
            filesystem: Optional shared RunFileSystem for unified path resolution
            memory_config: Optional ManagedMemoryConfig for compaction settings.
            compaction_model_config: Optional ModelConfig for a separate compaction model.
            **kwargs: Additional Agent arguments
        """
        # Build default goal
        if goal is None:
            shell_status = "with shell execution" if enable_shell else "file operations only"
            goal = f"Perform file operations and system tasks ({shell_status})"

        # Build conditional instruction
        if instruction is None:
            instruction = self._build_instruction(enable_shell, allowed_shell_commands)

        # Create shared filesystem if not provided
        if filesystem is None:
            from marsys.environment.filesystem import RunFileSystem
            filesystem = RunFileSystem.local(
                run_root=base_directory or Path.cwd(),
                cwd="/",
            )
        self.filesystem = filesystem

        # Initialize FileOperationTools with shared filesystem
        file_config = FileOperationConfig(
            base_directory=base_directory or Path.cwd(),
            run_filesystem=filesystem,
        )
        self.file_tools = FileOperationTools(config=file_config)
        tools = self.file_tools.get_tools()

        # Initialize ShellTools if enabled
        self.shell_enabled = enable_shell
        self.shell_tools = None

        if enable_shell:
            default_blocked = ShellTools._default_blocked_patterns()
            merged_blocked = list(set(default_blocked + (blocked_shell_patterns or [])))

            self.shell_tools = ShellTools(
                working_directory=working_directory,
                timeout_default=shell_timeout_default,
                allowed_commands=allowed_shell_commands,
                blocked_patterns=merged_blocked
            )

            tools.update({
                "shell_execute": self.shell_tools.execute,
                "shell_grep": self.shell_tools.grep,
                "shell_find": self.shell_tools.find,
                "shell_sed": self.shell_tools.sed,
                "shell_awk": self.shell_tools.awk,
                "shell_tail": self.shell_tools.tail,
                "shell_head": self.shell_tools.head,
                "shell_wc": self.shell_tools.wc,
                "shell_diff": self.shell_tools.diff,
                "shell_execute_streaming": self.shell_tools.execute_streaming
            })

        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            tools=tools,
            name=name,
            memory_config=memory_config,
            compaction_model_config=compaction_model_config,
            **kwargs
        )

    def _build_instruction(self, enable_shell: bool, allowed_shell_commands: Optional[list]) -> str:
        """Build conditional instruction."""

        instruction = """You are a file operations specialist. Your role is to help users work with files, directories, and data efficiently.

## Tool Selection Strategy

You have access to intelligent file operation tools (read, write, edit, search files and directories)"""

        if enable_shell:
            instruction += """ and shell command execution tools (grep, find, sed, awk, etc.)"""

        instruction += """.

**Choosing the right tool depends on the situation**:
- When you need to understand file content or structure, use read_file (supports text, PDF, images, JSON)
- When you need to modify existing code or text precisely, use edit_file with unified diff format
- When you need to create new files or completely replace content, use write_file
- When searching for content patterns, use search_content"""

        if enable_shell:
            instruction += """ or shell_grep for flexible pattern matching"""

        instruction += """
- When finding files by name, use search_files"""

        if enable_shell:
            instruction += """ or shell_find based on needed filters"""

        if enable_shell:
            shell_guidance = "\n\n**Shell tools give you flexibility**:"
            if allowed_shell_commands:
                shell_guidance += f"\n- You are restricted to: {', '.join(allowed_shell_commands)}"
            else:
                shell_guidance += "\n- Most standard commands available (dangerous operations blocked)"
                shell_guidance += "\n- Use specialized helpers (shell_grep, shell_find) for structured output"
                shell_guidance += "\n- Use shell_execute for complex pipelines when specialized tools don't fit"

            shell_guidance += "\n- Use shell_execute_streaming for long-running operations"
            instruction += shell_guidance

        instruction += """

## Working with Incomplete Information

When you don't have complete context:
- List directories to understand structure before assuming paths
- Read file headers or samples before processing large files
- Search for patterns to locate content when unsure where it is
- Check file types and sizes before deciding how to process them

If you encounter ambiguity, make reasonable assumptions based on common patterns, then proceed. Mention your assumptions.

## Handling Different Request Types

**For reading/analysis**: Consider file size - use line/page ranges for large files. For PDFs, extract images by default (extract_images=True) as they often contain important diagrams or charts. For JSON, use include_overview for large files. Return relevant excerpts rather than full content when appropriate.

**For editing**: Read the target section first. Use edit_file with precise diff format for code changes. Fall back to write_file only if edit_file fails and you're certain of full content. Verify changes by reading the modified section.

**For search**: Start with specific patterns; broaden if no results. Consider case sensitivity and regex vs literal. Search likely locations first, expand scope if needed.

**For creation/deletion**: Verify paths before writing. Create parent directories automatically (tools handle this). Confirm deletions or overwrites if they seem destructive.

## Recovering from Failures

When operations don't work as expected, adapt your approach:

**File not found**: Don't guess - list the directory to see what actually exists, then adjust your path or suggest alternatives.

**No search results**: Your pattern might be too specific. Try broader terms, check different file types, or verify you're searching the right directories. Content may exist but use different terminology.

**Edit failures**: The diff format might not match actual file content. Re-read the specific section to get exact current state, then try again with precise line matching. If still failing, explain the issue and consider alternatives.
"""

        if enable_shell:
            instruction += """
**Shell command errors**: Parse the error message to understand what went wrong. Often it's syntax or missing file. Break complex commands into simpler steps, or fall back to file tools which have better error handling.
"""

        instruction += """
**Permission issues**: These usually can't be fixed by you. Explain the problem clearly and suggest the user check permissions or use different paths.

## Response Format

Always return structured results with clear information about what you accomplished. Your response should contain:
- **Source attribution**: When returning file contents or search results, always note which files the information came from
- **Status**: Whether operations succeeded or failed
- **Relevant data**: Search results, file contents, or summaries as appropriate
- **Issues encountered**: Any problems or limitations, with context about why they occurred

Don't narrate your actions as steps. Present results naturally as if answering a question.

## Key Principles

**Be accurate**: Use exact paths and content. Don't paraphrase code or data.
**Be efficient**: Don't read entire files when a sample will do. Use ranges intelligently.
**Be adaptive**: If one approach doesn't work, try another based on the failure mode.
**Be safe**: Validate paths, avoid destructive operations without good reason, respect security boundaries.
**Be helpful**: When something fails, explain why and suggest what the user can do."""

        return instruction

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "file_operations": True,
            "shell_commands": self.shell_enabled,
            "total_tools": len(self.tools)
        }

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.file_tools, 'cleanup'):
            await self.file_tools.cleanup()
        if hasattr(super(), 'cleanup'):
            await super().cleanup()

"""
CodeExecutionAgent - Specialized agent for code execution tasks.

Combines file operations with Python and shell code execution for
development, scripting, and automation tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from marsys.agents.agents import Agent
from marsys.environment.code import CodeExecutionConfig, CodeExecutionTools
from marsys.environment.file_operations import FileOperationTools
from marsys.environment.file_operations.config import FileOperationConfig
from marsys.models.models import ModelConfig

if TYPE_CHECKING:
    from marsys.environment.filesystem import RunFileSystem

logger = logging.getLogger(__name__)


class CodeExecutionAgent(Agent):
    """
    Specialized agent for code execution and file operations.

    Combines:
    - FileOperationTools: Read, write, edit, search files
    - CodeExecutionTools: Execute Python and shell code safely

    Use cases:
    - Development tasks (run tests, build, lint)
    - Scripting and automation
    - Data processing pipelines
    - System administration tasks
    """

    def __init__(
        self,
        model_config: ModelConfig,
        name: str,
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
        working_directory: Optional[str] = None,
        base_directory: Optional[Path] = None,
        code_config: Optional[CodeExecutionConfig] = None,
        filesystem: Optional["RunFileSystem"] = None,
        memory_config=None,
        compaction_model_config=None,
        **kwargs,
    ):
        """
        Initialize CodeExecutionAgent.

        Args:
            model_config: Model configuration
            name: Agent name
            goal: Agent goal (default: auto-generated)
            instruction: Agent instruction (default: auto-generated)
            working_directory: Working directory for code execution (deprecated, use filesystem)
            base_directory: Base directory for file operations
            code_config: Optional pre-configured CodeExecutionConfig
            filesystem: Optional shared RunFileSystem for unified path resolution
            memory_config: Optional ManagedMemoryConfig for compaction settings.
            compaction_model_config: Optional ModelConfig for a separate compaction model.
            **kwargs: Additional Agent arguments
        """
        if goal is None:
            goal = "Run Python and shell code safely and perform file operations"

        if instruction is None:
            instruction = self._build_instruction()

        # Create shared filesystem if not provided
        root_dir = base_directory or Path.cwd()
        if filesystem is None:
            from marsys.environment.filesystem import RunFileSystem
            filesystem = RunFileSystem.local(run_root=root_dir, cwd="/")
        self.filesystem = filesystem

        # Initialize FileOperationTools with shared filesystem
        file_config = FileOperationConfig(
            base_directory=root_dir,
            run_filesystem=filesystem,
        )
        self.file_tools = FileOperationTools(config=file_config)
        tools = self.file_tools.get_tools()

        # Initialize CodeExecutionTools with shared filesystem
        code_config = code_config or CodeExecutionConfig()
        code_config.base_directory = root_dir
        code_config.run_filesystem = filesystem
        self.code_tools = CodeExecutionTools(code_config)
        tools.update(self.code_tools.get_tools())

        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            tools=tools,
            name=name,
            memory_config=memory_config,
            compaction_model_config=compaction_model_config,
            **kwargs,
        )

    def _build_instruction(self) -> str:
        """Build agent instruction."""
        return """You are a code execution specialist.

Use `python_execute` for code and analysis. Use `shell_execute` only when you need shell utilities or OS-level commands.
Prefer file tools for structured file changes (read_file, edit_file, write_file).
Keep commands minimal, deterministic, and verify outputs before proceeding."""

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "file_operations": True,
            "python_execution": True,
            "shell_execution": True,
            "total_tools": len(self.tools),
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.code_tools.cleanup()
        if hasattr(self.file_tools, "cleanup"):
            await self.file_tools.cleanup()
        if hasattr(super(), "cleanup"):
            await super().cleanup()

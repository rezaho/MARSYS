"""
DataAnalysisAgent - General-purpose agent for data science tasks.

Optimized for iterative, stateful work using a persistent Python session,
similar to a Jupyter notebook workflow.
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


class DataAnalysisAgent(Agent):
    """
    General-purpose data science agent with persistent Python session.

    Works like a Jupyter notebook - executes code cells incrementally,
    maintains state between executions, and allows iterative experimentation.

    Use cases:
    - Exploratory data analysis
    - Machine learning experiments
    - Data processing pipelines
    - Statistical modeling
    - Visualization and reporting
    - Any task requiring iterative Python experimentation
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
        **kwargs,
    ):
        """
        Initialize DataAnalysisAgent.

        Args:
            model_config: Model configuration
            name: Agent name
            goal: Agent goal (default: auto-generated)
            instruction: Agent instruction (default: auto-generated)
            working_directory: Working directory for code execution (deprecated, use filesystem)
            base_directory: Base directory for file operations
            code_config: Optional pre-configured CodeExecutionConfig
            filesystem: Optional shared RunFileSystem for unified path resolution
            **kwargs: Additional Agent arguments
        """
        if goal is None:
            goal = "Act as a data scientist - explore, analyze, model, and solve problems using Python and available tools"

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

        # Initialize CodeExecutionTools with persistent session and shared filesystem
        if code_config is None:
            code_config = CodeExecutionConfig(session_persistent_python=True)
        else:
            code_config.session_persistent_python = True

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
            **kwargs,
        )

    def _build_instruction(self) -> str:
        """Build agent instruction."""
        return """You are a data scientist with access to a persistent Python environment and file tools.

## Jupyter-like Python Session

Your `python_execute` tool works like a Jupyter notebook:
- **State persists between cells**: Variables, imports, and objects remain available across executions
- **Iterative workflow**: Run code, observe results, refine your approach
- **Experimentation**: Test hypotheses, inspect intermediate outputs, build incrementally

This allows you to work naturally - load data once, explore it across multiple steps, and develop solutions iteratively based on what you discover.

## Shell Commands (Non-Persistent)

The `shell_execute` tool runs in a separate subprocess each time - it does NOT share state with your Python session or between shell calls. Use it for system tasks when needed, but prefer Python for stateful work.

## Available Tools

- `python_execute`: Persistent Python session (like Jupyter cells)
- `shell_execute`: One-off shell commands (no persistence)
- File tools: `read_file`, `write_file`, `edit_file`, `search_files`, `list_directory`

## Constraints

Network access is disabled for security."""

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "file_operations": True,
            "python_execution": True,
            "persistent_python_session": True,
            "shell_execution": True,
            "network_access": False,
            "total_tools": len(self.tools),
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.code_tools.cleanup()
        if hasattr(self.file_tools, "cleanup"):
            await self.file_tools.cleanup()
        if hasattr(super(), "cleanup"):
            await super().cleanup()

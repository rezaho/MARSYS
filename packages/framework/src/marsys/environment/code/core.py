"""
Main CodeExecutionTools interface.

This module provides the primary API for code execution in MARSYS agents.
"""

import logging
from typing import Any, Callable, Dict, Optional

from marsys.environment.tool_response import ToolResponse, ToolResponseContent

from .config import CodeExecutionConfig
from .python_executor import PythonExecutor
from .shell_executor import ShellExecutor

logger = logging.getLogger(__name__)


class CodeExecutionTools:
    """
    Main interface for code execution in MARSYS agents.

    Provides safe Python and shell code execution with:
    - Configurable security policies
    - Resource limits (timeout, memory, CPU)
    - Persistent Python sessions for stateful analysis
    - Output truncation and structured results
    - Image capture via display() and display_image() hooks

    Example:
        ```python
        config = CodeExecutionConfig(
            timeout_default=60,
            allow_network=False,
            session_persistent_python=True,
        )
        tools = CodeExecutionTools(config)

        # Get tools for agent integration
        agent_tools = tools.get_tools()

        # Or use directly
        result = await tools.python_execute("print('Hello')")
        ```
    """

    def __init__(self, config: Optional[CodeExecutionConfig] = None):
        """
        Initialize code execution tools.

        Args:
            config: Configuration (if None, uses safe defaults)
        """
        self.config = config or CodeExecutionConfig()
        self.python_executor = PythonExecutor(self.config)
        self.shell_executor = ShellExecutor(self.config)

        logger.info(
            f"CodeExecutionTools initialized: "
            f"base_dir={self.config.base_directory}, "
            f"persistent_python={self.config.session_persistent_python}"
        )

    async def python_execute(
        self,
        code: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,  # Reserved for future multi-session support
    ) -> ToolResponse:
        """
        Execute Python code safely.

        Args:
            code: Python code to execute
            stdin: Standard input text (optional)
            timeout: Timeout in seconds (default: config.timeout_default)
            env: Additional environment variables (must be in allowed list)
            session_id: Reserved for future multi-session support

        Returns:
            ToolResponse with execution result and images

        Example:
            ```python
            # Basic execution
            result = await tools.python_execute("print('Hello')")

            # With image capture (persistent session)
            result = await tools.python_execute('''
                import matplotlib.pyplot as plt
                plt.plot([1, 2, 3], [1, 4, 9])
                display()  # Captures the figure
            ''')
            ```
        """
        result = await self.python_executor.execute(
            code=code,
            stdin=stdin,
            timeout=timeout,
            env=env,
        )

        # Build content blocks: text result + any images
        content_blocks = [ToolResponseContent(text=result.to_dict())]

        # Use host paths (image_host_paths) for actual image loading,
        # but the result.images contains virtual paths for agent visibility
        host_images = result.image_host_paths or []
        for image_path in host_images:
            content_blocks.append(ToolResponseContent(image_path=image_path))

        return ToolResponse(content=content_blocks)

    async def shell_execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
    ) -> ToolResponse:
        """
        Execute shell command safely.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default: config.timeout_default)
            env: Additional environment variables (must be in allowed list)
            shell: Shell to use (default: config.shell_path)

        Returns:
            ToolResponse with execution result

        Example:
            ```python
            result = await tools.shell_execute("ls -la")
            ```
        """
        result = await self.shell_executor.execute(
            command=command,
            timeout=timeout,
            env=env,
            shell=shell,
        )

        return ToolResponse(content=[ToolResponseContent(text=result.to_dict())])

    def get_tools(self) -> Dict[str, Callable]:
        """
        Get dictionary of tools for agent integration.

        Returns:
            Dict mapping tool name to callable function

        Example:
            ```python
            tools = code_exec_tools.get_tools()
            agent = Agent(tools=tools, ...)
            ```
        """
        return {
            "python_execute": self.python_execute,
            "shell_execute": self.shell_execute,
        }

    async def cleanup(self) -> None:
        """
        Cleanup resources.

        Shuts down persistent Python session if running.
        Should be called when the agent is done using code execution.
        """
        await self.python_executor.shutdown()
        logger.debug("CodeExecutionTools cleanup completed")

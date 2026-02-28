"""
Code Execution Toolkit for MARSYS Agents

Provides safe Python and shell code execution with:
- Configurable security policies (blocked modules, patterns, network access)
- Resource limits (timeout, memory, CPU)
- Persistent Python sessions for stateful analysis
- Image capture via display() and display_image() hooks
- Output truncation and structured results
"""

from .config import CodeExecutionConfig
from .core import CodeExecutionTools
from .data_models import ExecutionResult
from .python_executor import PythonExecutor
from .shell_executor import ShellExecutor

__all__ = [
    "CodeExecutionConfig",
    "CodeExecutionTools",
    "ExecutionResult",
    "PythonExecutor",
    "ShellExecutor",
]

__version__ = "0.2"

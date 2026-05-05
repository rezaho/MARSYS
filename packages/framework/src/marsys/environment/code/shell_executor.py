"""
Shell executor with policy enforcement and resource limits.

This module wraps ShellTools with additional security policies
and resource limit enforcement via preexec_fn.
"""

import platform
import time
from typing import Dict, Optional, Tuple

from marsys.agents.exceptions import ToolExecutionError
from marsys.environment.shell_tools import ShellTools

from .config import CodeExecutionConfig
from .data_models import ExecutionResult
from .validators import (
    build_base_env,
    validate_env,
    validate_shell_command,
)


def _build_preexec_fn(config: CodeExecutionConfig):
    """
    Build preexec function for subprocess resource limits.

    Only applies limits on Linux where resource module is available.

    Args:
        config: Code execution configuration

    Returns:
        Function to call in child process before exec, or None if not supported
    """
    # Only apply resource limits on Linux
    if platform.system() != "Linux":
        return None

    def _apply_limits():
        try:
            import resource

            # Set CPU time limit
            if config.max_cpu_seconds:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (config.max_cpu_seconds, config.max_cpu_seconds)
                )

            # Set memory limit (address space)
            if config.max_memory_mb:
                mem_bytes = config.max_memory_mb * 1024 * 1024
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (mem_bytes, mem_bytes)
                )
        except (ImportError, ValueError, OSError):
            # resource module not available or limits not supported
            pass

    return _apply_limits


def _truncate(text: str, max_bytes: int) -> Tuple[str, bool]:
    """
    Truncate text to max_bytes with indicator.

    Args:
        text: Text to truncate
        max_bytes: Maximum bytes allowed

    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if len(text) <= max_bytes:
        return text, False
    return text[:max_bytes] + f"\n... (truncated, limit: {max_bytes} bytes)", True


class ShellExecutor:
    """
    Shell executor with policy enforcement and resource limits.

    Wraps ShellTools with:
    - Working directory validation and resolution
    - Environment variable validation
    - Command validation against security policies
    - Resource limits via preexec_fn (Linux only)
    - Output truncation
    """

    def __init__(self, config: CodeExecutionConfig):
        """
        Initialize shell executor.

        Args:
            config: Code execution configuration
        """
        self.config = config
        self.shell_tools = ShellTools(
            working_directory=str(config.base_directory),
            timeout_default=config.timeout_default,
            allowed_commands=config.allowed_shell_commands,
            blocked_patterns=config.blocked_shell_patterns,
            max_output_size=config.max_output_bytes,
            shell_path=config.shell_path,
        )

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        shell: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute shell command with policy enforcement.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (default: config.timeout_default)
            env: Additional environment variables
            shell: Shell to use (default: config.shell_path)

        Returns:
            ExecutionResult with execution details (virtual paths)
        """
        start = time.time()

        # Resolve working directory via RunFileSystem
        fs = self.config.run_filesystem
        resolved = fs.resolve("/")  # Always use root as cwd
        host_cwd = resolved.host_path
        virtual_cwd = resolved.virtual_path

        # Validate command against config policies
        allowed, reason = validate_shell_command(command, self.config)
        if not allowed:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=False,
                duration_ms=0,
                cwd=virtual_cwd,
                truncated=False,
                language="shell",
                command=command,
                error=reason,
            )

        # Validate environment variables
        env_ok, env_reason = validate_env(env, self.config)
        if not env_ok:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=False,
                duration_ms=0,
                cwd=virtual_cwd,
                truncated=False,
                language="shell",
                command=command,
                error=env_reason,
            )

        # Build execution environment
        exec_env = build_base_env(self.config)
        if env:
            exec_env.update(env)

        # Use custom shell if specified
        tools = self.shell_tools
        if shell and shell != tools.shell_path:
            tools = ShellTools(
                working_directory=str(self.config.base_directory),
                timeout_default=self.config.timeout_default,
                allowed_commands=self.config.allowed_shell_commands,
                blocked_patterns=self.config.blocked_shell_patterns,
                max_output_size=self.config.max_output_bytes,
                shell_path=shell,
            )

        # Execute command (using host path internally)
        try:
            raw = await tools.execute(
                command,
                timeout=timeout or self.config.timeout_default,
                working_dir=str(host_cwd),
                env=exec_env,
                preexec_fn=_build_preexec_fn(self.config),
            )

            # Truncate output if needed
            stdout, trunc_out = _truncate(raw["stdout"], self.config.max_output_bytes)
            stderr, trunc_err = _truncate(raw["stderr"], self.config.max_output_bytes)
            duration_ms = int((time.time() - start) * 1000)

            return ExecutionResult(
                success=raw["success"],
                stdout=stdout,
                stderr=stderr,
                exit_code=raw["return_code"],
                timed_out=False,
                duration_ms=duration_ms,
                cwd=virtual_cwd,  # Return virtual path
                truncated=trunc_out or trunc_err,
                language="shell",
                command=command,
            )

        except ToolExecutionError as e:
            duration_ms = int((time.time() - start) * 1000)
            timed_out = "timed out" in str(e).lower()

            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                exit_code=-1,
                timed_out=timed_out,
                duration_ms=duration_ms,
                cwd=virtual_cwd,  # Return virtual path
                truncated=False,
                language="shell",
                command=command,
                error=str(e),
            )

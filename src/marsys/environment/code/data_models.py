"""
Data models for code execution results.

This module defines the structured result format for code execution operations.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """
    Structured result from code execution.

    Contains execution output, status, and metadata for both
    Python and shell executions.

    Attributes:
        success: Whether execution completed without errors (exit_code == 0)
        stdout: Standard output from execution (may be truncated)
        stderr: Standard error from execution (may be truncated)
        exit_code: Process exit code (0 = success, non-zero = error)
        timed_out: Whether execution was terminated due to timeout
        duration_ms: Execution duration in milliseconds
        cwd: Virtual working directory where code was executed (e.g., '/')
        truncated: Whether output was truncated due to size limits
        language: Execution language ("python" or "shell")
        command: Shell command (only for shell execution)
        artifacts: List of virtual file paths created during execution (optional)
        images: List of virtual image paths generated during execution
        error: Error message if validation or execution failed
        image_host_paths: Internal-only host paths for images (for base64 encoding)
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int
    cwd: str
    truncated: bool
    language: str
    command: Optional[str] = None
    artifacts: Optional[List[str]] = None
    images: Optional[List[str]] = None
    error: Optional[str] = None
    image_host_paths: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Excludes internal-only fields like image_host_paths.

        Returns:
            Dictionary representation of execution result
        """
        result = asdict(self)
        # Remove internal-only fields from agent-visible output
        result.pop('image_host_paths', None)
        return result

    @property
    def output(self) -> str:
        """
        Get combined output (stdout preferred, fallback to stderr or error).

        Returns:
            Primary output string for display
        """
        if self.stdout:
            return self.stdout
        if self.stderr:
            return self.stderr
        if self.error:
            return f"Error: {self.error}"
        return ""

    def __repr__(self) -> str:
        """String representation."""
        status = "success" if self.success else "failed"
        if self.timed_out:
            status = "timed_out"
        img_count = len(self.images) if self.images else 0
        return (
            f"ExecutionResult({status}, "
            f"exit_code={self.exit_code}, "
            f"duration={self.duration_ms}ms, "
            f"images={img_count}, "
            f"language={self.language})"
        )

"""
Bash command execution toolkit for MARSYS agents.

Provides safe bash command execution with:
- General execution for any command
- Specialized helpers for common operations (grep, find, sed, etc.)
- Safety validation and blocked commands
- Structured output for easy parsing
- Timeout enforcement and output size limits

TODO: Add user approval workflow for potentially dangerous commands
"""

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from marsys.agents.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class BashTools:
    """
    Safe bash command execution toolkit with specialized helpers.

    Provides both general command execution and specialized methods
    for common operations like grep, find, sed, etc.

    Example:
        ```python
        bash = BashTools(working_directory="./project")

        # Use specialized helper
        result = await bash.grep("TODO", ".", recursive=True)

        # Or general execution
        result = await bash.execute("find . -name '*.py' | wc -l")
        ```
    """

    def __init__(
        self,
        working_directory: Optional[str] = None,
        timeout_default: int = 30,
        allowed_commands: Optional[List[str]] = None,
        blocked_commands: Optional[List[str]] = None,
        max_output_size: int = 1_000_000  # 1MB
    ):
        """
        Initialize BashTools.

        Args:
            working_directory: Default working directory for commands
            timeout_default: Default timeout in seconds
            allowed_commands: Whitelist of allowed commands (empty = all allowed except blocked)
            blocked_commands: Blacklist of blocked commands
            max_output_size: Maximum output size in bytes
        """
        self.working_directory = Path(working_directory) if working_directory else Path.cwd()
        self.timeout_default = timeout_default
        self.allowed_commands = allowed_commands or []
        self.blocked_commands = blocked_commands or self._default_blocked_commands()
        self.max_output_size = max_output_size
        self.history: List[Dict] = []

    async def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute any bash command safely.

        Args:
            command: Bash command to execute
            timeout: Timeout in seconds (default: self.timeout_default)
            working_dir: Working directory (default: self.working_directory)
            reasoning: Optional reasoning for this command

        Returns:
            Dict with:
                - success: bool
                - stdout: str
                - stderr: str
                - return_code: int
                - command: str

        Raises:
            ToolExecutionError: If command is blocked or fails validation
        """
        if reasoning:
            logger.info(f"BashTools execute: {reasoning}")

        # Validate command
        is_valid, error_msg = self.validate_command(command)
        if not is_valid:
            raise ToolExecutionError(
                f"Command validation failed: {error_msg}",
                tool_name="bash_execute",
                context={"command": command}
            )

        # Setup execution
        timeout_val = timeout or self.timeout_default
        work_dir = Path(working_dir) if working_dir else self.working_directory

        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(work_dir),
                shell=True
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_val
                )

                stdout = stdout_bytes.decode('utf-8', errors='replace')
                stderr = stderr_bytes.decode('utf-8', errors='replace')

                # Enforce output size limit
                if len(stdout) > self.max_output_size:
                    stdout = stdout[:self.max_output_size] + f"\n... (truncated, limit: {self.max_output_size} bytes)"

                result = {
                    "success": process.returncode == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": process.returncode,
                    "command": command
                }

                # Log to history
                self.history.append({
                    "action": "execute",
                    "command": command,
                    "return_code": process.returncode,
                    "success": result["success"]
                })

                return result

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ToolExecutionError(
                    f"Command timed out after {timeout_val} seconds",
                    tool_name="bash_execute",
                    context={"command": command, "timeout": timeout_val}
                )

        except Exception as e:
            logger.error(f"BashTools execute failed: {e}")
            raise ToolExecutionError(
                f"Command execution failed: {str(e)}",
                tool_name="bash_execute",
                context={"command": command}
            )

    async def grep(
        self,
        pattern: str,
        file_or_dir: str,
        recursive: bool = False,
        case_insensitive: bool = False,
        line_numbers: bool = True,
        context_lines: int = 0,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for pattern in files with structured output.

        Args:
            pattern: Search pattern (regex)
            file_or_dir: File or directory to search
            recursive: Search recursively in directories
            case_insensitive: Case-insensitive search
            line_numbers: Include line numbers
            context_lines: Number of context lines before/after match
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - matches: List[Dict] with file, line_number, line
                - total_matches: int
                - files_searched: int
        """
        if reasoning:
            logger.info(f"BashTools grep: {reasoning}")

        # Build grep command
        cmd_parts = ["grep"]
        if recursive:
            cmd_parts.append("-r")
        if case_insensitive:
            cmd_parts.append("-i")
        if line_numbers:
            cmd_parts.append("-n")
        if context_lines > 0:
            cmd_parts.append(f"-C {context_lines}")

        # Add pattern and file (with quotes for safety)
        cmd_parts.append(f"'{pattern}'")
        cmd_parts.append(f"'{file_or_dir}'")

        command = " ".join(cmd_parts)

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"] and result["return_code"] == 1:
                # grep returns 1 when no matches found (not an error)
                return {
                    "success": True,
                    "matches": [],
                    "total_matches": 0,
                    "files_searched": 0
                }

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "matches": [],
                    "total_matches": 0,
                    "files_searched": 0
                }

            # Parse output
            matches = []
            files_seen = set()

            for line in result["stdout"].strip().split("\n"):
                if not line:
                    continue

                # Parse format: file:line_number:content or just content if single file
                if ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        content = parts[2]

                        files_seen.add(file_path)
                        matches.append({
                            "file": file_path,
                            "line_number": int(line_num) if line_num.isdigit() else None,
                            "line": content
                        })
                    elif len(parts) == 2:
                        # Single file search: line_number:content
                        line_num = parts[0]
                        content = parts[1]
                        matches.append({
                            "file": file_or_dir,
                            "line_number": int(line_num) if line_num.isdigit() else None,
                            "line": content
                        })

            return {
                "success": True,
                "matches": matches,
                "total_matches": len(matches),
                "files_searched": len(files_seen) if files_seen else 1
            }

        except Exception as e:
            logger.error(f"BashTools grep failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "matches": [],
                "total_matches": 0,
                "files_searched": 0
            }

    async def find(
        self,
        path: str,
        name_pattern: Optional[str] = None,
        type: Optional[str] = None,  # "f" (file), "d" (dir)
        max_depth: Optional[int] = None,
        modified_within: Optional[str] = None,  # e.g., "7" for 7 days
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find files matching criteria with structured output.

        Args:
            path: Starting path
            name_pattern: Filename pattern (e.g., "*.py")
            type: File type: "f" (file), "d" (directory)
            max_depth: Maximum depth to search
            modified_within: Modified within N days
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - files: List[Dict] with path
                - total_found: int
        """
        if reasoning:
            logger.info(f"BashTools find: {reasoning}")

        # Build find command
        cmd_parts = ["find", f"'{path}'"]

        if max_depth is not None:
            cmd_parts.append(f"-maxdepth {max_depth}")

        if type:
            cmd_parts.append(f"-type {type}")

        if name_pattern:
            cmd_parts.append(f"-name '{name_pattern}'")

        if modified_within:
            cmd_parts.append(f"-mtime -{modified_within}")

        command = " ".join(cmd_parts)

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "files": [],
                    "total_found": 0
                }

            # Parse output
            files = []
            for line in result["stdout"].strip().split("\n"):
                if line:
                    files.append({"path": line})

            return {
                "success": True,
                "files": files,
                "total_found": len(files)
            }

        except Exception as e:
            logger.error(f"BashTools find failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "files": [],
                "total_found": 0
            }

    async def wc(
        self,
        file_path: str,
        count_type: str = "all",  # "lines", "words", "chars", "all"
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Word/line/character count with structured output.

        Args:
            file_path: Path to file
            count_type: What to count: "lines", "words", "chars", "all"
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - lines: int
                - words: int
                - characters: int
        """
        if reasoning:
            logger.info(f"BashTools wc: {reasoning}")

        command = f"wc '{file_path}'"

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"]
                }

            # Parse output: "lines words chars filename"
            parts = result["stdout"].strip().split()
            if len(parts) >= 3:
                return {
                    "success": True,
                    "lines": int(parts[0]),
                    "words": int(parts[1]),
                    "characters": int(parts[2])
                }

            return {
                "success": False,
                "error": "Could not parse wc output"
            }

        except Exception as e:
            logger.error(f"BashTools wc failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def sed(
        self,
        pattern: str,
        replacement: str,
        file_path: str,
        in_place: bool = False,
        global_replace: bool = True,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Text substitution using sed with structured output.

        Args:
            pattern: Pattern to match (regex)
            replacement: Replacement text
            file_path: File to process
            in_place: Modify file in place
            global_replace: Replace all occurrences (vs. first only)
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - output: str (modified content if not in_place)
                - lines_modified: int
        """
        if reasoning:
            logger.info(f"BashTools sed: {reasoning}")

        # Build sed command
        flags = "g" if global_replace else ""
        sed_expr = f"s/{pattern}/{replacement}/{flags}"

        cmd_parts = ["sed"]
        if in_place:
            cmd_parts.append("-i")
        cmd_parts.append(f"'{sed_expr}'")
        cmd_parts.append(f"'{file_path}'")

        command = " ".join(cmd_parts)

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "output": "",
                    "lines_modified": 0
                }

            # Count modified lines (approximate)
            output_lines = result["stdout"].strip().split("\n") if result["stdout"] else []

            return {
                "success": True,
                "output": result["stdout"],
                "lines_modified": len(output_lines) if not in_place else None
            }

        except Exception as e:
            logger.error(f"BashTools sed failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "lines_modified": 0
            }

    async def awk(
        self,
        script: str,
        file_path: str,
        field_separator: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pattern processing using awk with structured output.

        Args:
            script: AWK script/pattern
            file_path: File to process
            field_separator: Field separator (default: whitespace)
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - output: str
                - lines_processed: int
        """
        if reasoning:
            logger.info(f"BashTools awk: {reasoning}")

        # Build awk command
        cmd_parts = ["awk"]
        if field_separator:
            cmd_parts.append(f"-F '{field_separator}'")
        cmd_parts.append(f"'{script}'")
        cmd_parts.append(f"'{file_path}'")

        command = " ".join(cmd_parts)

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "output": "",
                    "lines_processed": 0
                }

            output_lines = result["stdout"].strip().split("\n") if result["stdout"] else []

            return {
                "success": True,
                "output": result["stdout"],
                "lines_processed": len(output_lines)
            }

        except Exception as e:
            logger.error(f"BashTools awk failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "lines_processed": 0
            }

    async def tail(
        self,
        file_path: str,
        num_lines: int = 10,
        follow: bool = False,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get last N lines of file with structured output.

        Args:
            file_path: Path to file
            num_lines: Number of lines to return
            follow: Follow file for new lines (not recommended, use execute_streaming)
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - lines: List[str]
                - total_lines: int
        """
        if reasoning:
            logger.info(f"BashTools tail: {reasoning}")

        command = f"tail -n {num_lines} '{file_path}'"
        if follow:
            logger.warning("tail -f not recommended, use execute_streaming instead")
            command = f"tail -f -n {num_lines} '{file_path}'"

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "lines": [],
                    "total_lines": 0
                }

            lines = result["stdout"].strip().split("\n") if result["stdout"] else []

            return {
                "success": True,
                "lines": lines,
                "total_lines": len(lines)
            }

        except Exception as e:
            logger.error(f"BashTools tail failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "lines": [],
                "total_lines": 0
            }

    async def head(
        self,
        file_path: str,
        num_lines: int = 10,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get first N lines of file with structured output.

        Args:
            file_path: Path to file
            num_lines: Number of lines to return
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - lines: List[str]
                - total_lines: int
        """
        if reasoning:
            logger.info(f"BashTools head: {reasoning}")

        command = f"head -n {num_lines} '{file_path}'"

        try:
            result = await self.execute(command, reasoning=reasoning)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["stderr"],
                    "lines": [],
                    "total_lines": 0
                }

            lines = result["stdout"].strip().split("\n") if result["stdout"] else []

            return {
                "success": True,
                "lines": lines,
                "total_lines": len(lines)
            }

        except Exception as e:
            logger.error(f"BashTools head failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "lines": [],
                "total_lines": 0
            }

    async def diff(
        self,
        file1: str,
        file2: str,
        unified: bool = True,
        context_lines: int = 3,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare files with structured output.

        Args:
            file1: First file path
            file2: Second file path
            unified: Use unified diff format
            context_lines: Number of context lines
            reasoning: Optional reasoning

        Returns:
            Dict with:
                - success: bool
                - identical: bool
                - diff_output: str
                - changes: int
        """
        if reasoning:
            logger.info(f"BashTools diff: {reasoning}")

        # Build diff command
        cmd_parts = ["diff"]
        if unified:
            cmd_parts.append(f"-u{context_lines}")
        cmd_parts.append(f"'{file1}'")
        cmd_parts.append(f"'{file2}'")

        command = " ".join(cmd_parts)

        try:
            result = await self.execute(command, reasoning=reasoning)

            # diff returns 0 if identical, 1 if different, 2 if error
            if result["return_code"] == 0:
                return {
                    "success": True,
                    "identical": True,
                    "diff_output": "",
                    "changes": 0
                }
            elif result["return_code"] == 1:
                # Files differ
                diff_lines = result["stdout"].strip().split("\n")
                # Count actual change lines (start with +/-)
                changes = sum(1 for line in diff_lines if line and line[0] in ['+', '-'])

                return {
                    "success": True,
                    "identical": False,
                    "diff_output": result["stdout"],
                    "changes": changes
                }
            else:
                # Error case
                return {
                    "success": False,
                    "error": result["stderr"],
                    "identical": None,
                    "diff_output": "",
                    "changes": 0
                }

        except Exception as e:
            logger.error(f"BashTools diff failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "identical": None,
                "diff_output": "",
                "changes": 0
            }

    async def execute_streaming(
        self,
        command: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Execute command with streaming output for long-running commands.

        Args:
            command: Bash command
            timeout: Timeout in seconds
            working_dir: Working directory
            reasoning: Optional reasoning

        Yields:
            Lines of output as they become available
        """
        if reasoning:
            logger.info(f"BashTools execute_streaming: {reasoning}")

        # Validate command
        is_valid, error_msg = self.validate_command(command)
        if not is_valid:
            raise ToolExecutionError(
                f"Command validation failed: {error_msg}",
                tool_name="bash_execute_streaming",
                context={"command": command}
            )

        timeout_val = timeout or self.timeout_default
        work_dir = Path(working_dir) if working_dir else self.working_directory

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            shell=True
        )

        try:
            while True:
                line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=timeout_val
                )

                if not line:
                    break

                yield line.decode('utf-8', errors='replace').rstrip()

            await process.wait()

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ToolExecutionError(
                f"Command timed out after {timeout_val} seconds",
                tool_name="bash_execute_streaming",
                context={"command": command}
            )

    def validate_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """
        Validate command for safety.

        Args:
            command: Command to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if command is in allowed list (if whitelist exists)
        if self.allowed_commands:
            cmd_start = command.strip().split()[0] if command.strip() else ""
            if cmd_start not in self.allowed_commands:
                return False, f"Command '{cmd_start}' not in allowed commands list"

        # Check blocked commands
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Command contains blocked pattern: '{blocked}'"

        # Check for dangerous patterns
        dangerous_patterns = [
            r"rm\s+-rf\s+/",  # Delete root
            r">\s*/dev/",  # Write to devices
            r"dd\s+if=",  # Disk operations
            r"mkfs",  # Format filesystem
            r":\(\)\{",  # Fork bomb start
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return False, f"Command matches dangerous pattern: {pattern}"

        return True, None

    @staticmethod
    def _default_blocked_commands() -> List[str]:
        """Default dangerous commands to block."""
        return [
            "rm -rf /",
            "sudo",
            "dd if=",
            "mkfs",
            "format",
            "> /dev/",
            "chmod -R 777",
            ":(){ :|:& };:",  # Fork bomb
        ]

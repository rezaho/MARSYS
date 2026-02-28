"""
Security framework for file operations.

This module implements path validation, permission checking, and approval workflows
to ensure safe file operations within configured boundaries.
"""

import asyncio
import logging
from pathlib import Path, PurePath
from typing import Optional, Dict, Any
from datetime import datetime

from .config import FileOperationConfig
from .data_models import ValidationResult

logger = logging.getLogger(__name__)


class PathValidator:
    """Validates file paths against security rules."""

    def __init__(self, config: FileOperationConfig):
        """
        Initialize path validator.

        Args:
            config: File operation configuration
        """
        self.config = config

    def validate(self, path: Path, operation: str = "access") -> ValidationResult:
        """
        Validate a path against security rules.

        Args:
            path: Path to validate
            operation: Type of operation (access, read, write, delete, etc.)

        Returns:
            ValidationResult with allowed status and details
        """
        # Convert to absolute path
        try:
            abs_path = path.resolve()
        except (OSError, RuntimeError) as e:
            return ValidationResult(
                allowed=False,
                reason=f"Invalid path: {e}"
            )

        # Check for symlinks
        if abs_path.is_symlink() and not self.config.follow_symlinks:
            return ValidationResult(
                allowed=False,
                reason="Symbolic links are not allowed"
            )

        # Check allowed patterns first (they override blocked)
        if self.config.allowed_patterns:
            for pattern in self.config.allowed_patterns:
                if PurePath(str(abs_path)).match(pattern):
                    logger.debug(f"Path {abs_path} explicitly allowed by pattern: {pattern}")
                    return ValidationResult(
                        allowed=True,
                        auto_approved=True
                    )

        # Check blocked patterns
        for pattern in self.config.blocked_patterns:
            if PurePath(str(abs_path)).match(pattern):
                return ValidationResult(
                    allowed=False,
                    reason=f"Path matches blocked pattern: {pattern}"
                )

        # Path is allowed
        return ValidationResult(allowed=True)

    def is_safe_path(self, path: Path) -> bool:
        """
        Quick check if a path is safe (doesn't contain dangerous patterns).

        Args:
            path: Path to check

        Returns:
            True if path appears safe, False otherwise
        """
        path_str = str(path).lower()

        # Check for path traversal attempts
        dangerous_patterns = [
            "..",
            "~",
            "${",  # Variable expansion
            "$(",  # Command substitution
        ]

        for pattern in dangerous_patterns:
            if pattern in path_str:
                return False

        return True


class PermissionChecker:
    """Checks permissions for file operations."""

    def __init__(self, config: FileOperationConfig):
        """
        Initialize permission checker.

        Args:
            config: File operation configuration
        """
        self.config = config

    def check_operation(
        self,
        path: Path,
        operation: str
    ) -> ValidationResult:
        """
        Check if an operation is permitted on a path.

        Args:
            path: File path
            operation: Operation type (read, write, edit, delete)

        Returns:
            ValidationResult indicating permission status
        """
        # Check if delete is enabled
        if operation == "delete" and not self.config.enable_delete:
            return ValidationResult(
                allowed=False,
                reason="Delete operations are disabled in configuration"
            )

        # Check file size limits for read operations
        if operation in ["read", "edit"] and path.exists() and path.is_file():
            file_size_bytes = path.stat().st_size
            if file_size_bytes > self.config.max_file_size_bytes:
                file_size_mb = file_size_bytes / (1024 * 1024)
                max_size_mb = self.config.max_file_size_bytes / (1024 * 1024)
                return ValidationResult(
                    allowed=False,
                    reason=f"File size ({file_size_mb:.2f} MB) exceeds limit ({max_size_mb:.2f} MB)"
                )

        # Check auto-approve patterns
        if self.config.should_auto_approve(path):
            return ValidationResult(
                allowed=True,
                auto_approved=True
            )

        # Check if approval required
        if self.config.requires_approval(path):
            return ValidationResult(
                allowed=True,
                needs_approval=True
            )

        # Default: allowed but not auto-approved
        return ValidationResult(allowed=True)


class ApprovalWorkflow:
    """Handles user approval requests for file operations."""

    def __init__(self, config: FileOperationConfig):
        """
        Initialize approval workflow.

        Args:
            config: File operation configuration
        """
        self.config = config
        self.approval_history: Dict[str, bool] = {}

    async def request_approval(
        self,
        operation: str,
        path: Path,
        preview: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Request user approval for a file operation.

        Args:
            operation: Type of operation (read, write, edit, delete)
            path: File path
            preview: Optional preview of changes (for edits)
            metadata: Optional additional metadata

        Returns:
            True if approved, False if rejected
        """
        # Create approval request key for tracking
        request_key = f"{operation}:{path}:{datetime.now().isoformat()}"

        # Format approval message
        message = self._format_approval_message(operation, path, preview, metadata)

        logger.info(f"Requesting approval for {operation} on {path}")

        try:
            # Get approval from user
            # In a real implementation, this would integrate with CommunicationManager
            # For now, we'll use a simple input approach
            approved = await self._get_user_approval(message)

            # Record in history
            self.approval_history[request_key] = approved

            logger.info(f"Approval {'granted' if approved else 'denied'} for {operation} on {path}")

            return approved

        except asyncio.TimeoutError:
            logger.warning(f"Approval request timed out for {operation} on {path}")
            # Use configured timeout behavior
            return self.config.auto_approve_on_timeout

    def _format_approval_message(
        self,
        operation: str,
        path: Path,
        preview: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Format approval request message."""
        lines = [
            "=" * 60,
            "FILE OPERATION APPROVAL REQUEST",
            "=" * 60,
            f"Operation: {operation.upper()}",
            f"Path: {path}",
        ]

        if metadata:
            lines.append("\nMetadata:")
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")

        if preview:
            lines.append("\nPreview:")
            lines.append("-" * 60)
            # Limit preview length
            preview_lines = preview.split('\n')[:50]
            if len(preview.split('\n')) > 50:
                preview_lines.append("... (preview truncated)")
            lines.extend(preview_lines)
            lines.append("-" * 60)

        lines.extend([
            "",
            "Approve this operation? (yes/no): "
        ])

        return "\n".join(lines)

    async def _get_user_approval(self, message: str) -> bool:
        """
        Get user approval response.

        This is a placeholder that should be replaced with actual
        CommunicationManager integration.

        Args:
            message: Approval message to display

        Returns:
            True if approved, False otherwise
        """
        # TODO: Integrate with MARSYS CommunicationManager for actual user input
        # For now, we'll simulate approval based on configuration

        # In production, this would be something like:
        # response = await self.comm_manager.request_input(message, timeout=self.config.approval_timeout_seconds)
        # return response.lower() in ['yes', 'y']

        # For testing/development: auto-approve if timeout is None or very long
        if self.config.approval_timeout_seconds is None or self.config.approval_timeout_seconds > 300:
            logger.warning("Auto-approving (placeholder implementation)")
            return True

        # Otherwise use timeout behavior
        return self.config.auto_approve_on_timeout


class AuditLogger:
    """Logs file operations for audit purposes."""

    def __init__(self, config: FileOperationConfig):
        """
        Initialize audit logger.

        Args:
            config: File operation configuration
        """
        self.config = config
        self.log_file = config.log_file_path

        # Set up file logging if configured
        if self.log_file:
            self._setup_file_logging()

    def _setup_file_logging(self):
        """Set up file-based audit logging."""
        if not self.log_file:
            return

        # Create log directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        # Add to logger
        audit_logger = logging.getLogger('marsys.environment.file_operations.audit')
        audit_logger.addHandler(file_handler)
        audit_logger.setLevel(logging.INFO)

    def log_operation(
        self,
        operation: str,
        path: Path,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a file operation.

        Args:
            operation: Type of operation
            path: File path
            success: Whether operation succeeded
            details: Optional additional details
        """
        if not self.config.enable_audit_logging:
            return

        audit_logger = logging.getLogger('marsys.environment.file_operations.audit')

        status = "SUCCESS" if success else "FAILURE"
        message = f"{status} - {operation} - {path}"

        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            message += f" - {detail_str}"

        audit_logger.info(message)


class SecurityManager:
    """
    Main security manager coordinating all security components.

    This class provides a unified interface for path validation,
    permission checking, approval workflows, and audit logging.
    """

    def __init__(self, config: FileOperationConfig):
        """
        Initialize security manager.

        Args:
            config: File operation configuration
        """
        self.config = config
        self.path_validator = PathValidator(config)
        self.permission_checker = PermissionChecker(config)
        self.approval_workflow = ApprovalWorkflow(config)
        self.audit_logger = AuditLogger(config)

    def validate_path(self, path: Path, operation: str = "access") -> ValidationResult:
        """
        Validate a path for an operation.

        Args:
            path: Path to validate
            operation: Type of operation

        Returns:
            ValidationResult
        """
        # First validate path structure
        path_result = self.path_validator.validate(path, operation)
        if not path_result.allowed:
            return path_result

        # Then check permissions
        perm_result = self.permission_checker.check_operation(path, operation)
        if not perm_result.allowed:
            return perm_result

        # Combine results
        return ValidationResult(
            allowed=True,
            needs_approval=perm_result.needs_approval,
            auto_approved=perm_result.auto_approved,
            warnings=path_result.warnings + perm_result.warnings
        )

    async def authorize_operation(
        self,
        operation: str,
        path: Path,
        preview: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Authorize a file operation (including approval if needed).

        Args:
            operation: Type of operation
            path: File path
            preview: Optional preview of changes
            metadata: Optional metadata

        Returns:
            ValidationResult with final authorization decision
        """
        # Validate path first
        validation = self.validate_path(path, operation)
        if not validation.allowed:
            self.audit_logger.log_operation(operation, path, False, {"reason": validation.reason})
            return validation

        # If auto-approved, we're done
        if validation.auto_approved:
            self.audit_logger.log_operation(operation, path, True, {"auto_approved": True})
            return validation

        # If approval needed, request it
        if validation.needs_approval:
            approved = await self.approval_workflow.request_approval(
                operation, path, preview, metadata
            )

            result = ValidationResult(
                allowed=approved,
                needs_approval=True,
                auto_approved=False,
                reason="User denied approval" if not approved else None
            )

            self.audit_logger.log_operation(
                operation, path, approved,
                {"approval_requested": True, "approved": approved}
            )

            return result

        # Otherwise, operation is allowed
        self.audit_logger.log_operation(operation, path, True)
        return validation

    def log_operation(
        self,
        operation: str,
        path: Path,
        success: bool,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a file operation to audit trail.

        Args:
            operation: Type of operation
            path: File path
            success: Whether operation succeeded
            details: Optional additional details
        """
        self.audit_logger.log_operation(operation, path, success, details)

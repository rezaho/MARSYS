"""
Configuration for file operations toolkit.

This module defines the configuration class for controlling security,
permissions, operation limits, and feature flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from marsys.environment.filesystem import RunFileSystem


# Default security patterns
DEFAULT_BLOCKED_PATTERNS = [
    "**/.env",           # Environment variables
    "**/.env.*",         # Environment variants (.env.local, .env.production, etc.)
    "**/*.key",          # Private keys
    "**/*.pem",          # Certificates
    "**/*.p12",          # PKCS#12 files
    "**/*.pfx",          # Personal Information Exchange files
    "**/credentials*",   # Credential files
    "**/*secret*",       # Files with 'secret' in name
    "**/.git/**",        # Git internals
    "**/.svn/**",        # SVN internals
    "**/node_modules/**", # Node dependencies
    "**/__pycache__/**", # Python cache
    "**/venv/**",        # Python virtual environments
    "**/.venv/**",       # Python virtual environments (alternative)
    "**/env/**",         # Python virtual environments (alternative)
    "**/*.pyc",          # Python compiled files
    "**/.DS_Store",      # macOS system files
    "**/Thumbs.db",      # Windows system files
    "**/.idea/**",       # IDE files
    "**/.vscode/**",     # IDE files (VSCode)
    "**/id_rsa*",        # SSH keys
    "**/.ssh/**",        # SSH directory
    "**/.aws/credentials", # AWS credentials
    "**/.azure/credentials", # Azure credentials
    "**/.gcp/credentials.json", # GCP credentials
]

DEFAULT_AUTO_APPROVE_PATTERNS = [
    "**/*.md",           # Markdown files
    "**/*.txt",          # Text files
    "**/tests/**",       # Test files
    "**/test_*.py",      # Python test files
    "**/*_test.py",      # Python test files (alternative naming)
    "**/README*",        # README files
    "**/LICENSE*",       # License files
    "**/CHANGELOG*",     # Changelog files
    "**/*.log",          # Log files (read-only typically)
]

DEFAULT_REQUIRE_APPROVAL_PATTERNS = [
    "**/config*",        # Configuration files
    "**/*.yaml",         # YAML files (may contain sensitive config)
    "**/*.yml",          # YAML files (alternative extension)
    "**/*.toml",         # TOML files (configuration)
    "**/*.ini",          # INI files (configuration)
    "**/*.conf",         # Configuration files
    "**/.*rc",           # RC files (.bashrc, .npmrc, etc.)
    "**/Dockerfile*",    # Docker configuration
    "**/docker-compose*", # Docker Compose files
    "**/requirements*.txt", # Python dependencies
    "**/package*.json",  # Node dependencies
    "**/Pipfile*",       # Python dependencies (Pipenv)
    "**/poetry.lock",    # Python dependencies (Poetry)
    "**/Cargo.toml",     # Rust dependencies
    "**/go.mod",         # Go dependencies
]


@dataclass
class FileOperationConfig:
    """
    Configuration for file operations toolkit.

    This class controls security settings, permissions, operation limits,
    and feature flags for the file operations system.
    """

    # === Security Settings ===

    base_directory: Optional[Path] = None
    """Base directory for file operations. All operations are relative to this directory."""

    allowed_patterns: List[str] = field(default_factory=list)
    """
    Glob patterns for explicitly allowed files (overrides blocked patterns).
    Example: ["**/safe/**", "**/*.py"]
    """

    blocked_patterns: List[str] = field(default_factory=lambda: DEFAULT_BLOCKED_PATTERNS.copy())
    """
    Glob patterns for blocked files (cannot be accessed at all).
    Default includes sensitive files like .env, credentials, keys, etc.
    """

    auto_approve_patterns: List[str] = field(default_factory=lambda: DEFAULT_AUTO_APPROVE_PATTERNS.copy())
    """
    Glob patterns for files that are automatically approved for operations.
    Default includes markdown, text files, test files, etc.
    """

    require_approval_patterns: List[str] = field(default_factory=lambda: DEFAULT_REQUIRE_APPROVAL_PATTERNS.copy())
    """
    Glob patterns for files that require user approval before modification.
    Default includes config files, dependency manifests, etc.
    """

    respect_gitignore: bool = True
    """Whether to respect .gitignore files when listing/searching files."""

    respect_ignore_files: List[str] = field(default_factory=lambda: [".gitignore", ".marsysignore"])
    """List of ignore file names to respect (like .gitignore, .dockerignore, etc.)."""

    # === Operation Limits ===

    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MB absolute hard limit for safety
    """Maximum file size in bytes (hard limit for safety). Default: 100 MB."""

    max_pages_per_read: int = 5
    """
    Maximum PDF pages to return in a single read operation.
    Default: 5 pages (~10k-25k characters typically).
    Agents can request specific page ranges with start_page/end_page parameters.
    """

    max_lines_per_read: int = 250
    """
    Maximum lines to return for text files in a single read operation.
    Default: 250 lines (~10k-20k characters typically, depending on line length).
    Applies to: .py, .txt, .md, .yaml, .csv, and other line-based text files.
    Agents can request specific line ranges with start_line/end_line parameters.
    """

    max_json_content_chars: int = 40000
    """
    Maximum characters for JSON file content before triggering truncation/overview.
    Default: 40,000 characters (~10k tokens).

    When a JSON file exceeds this limit:
    - Dictionary values are truncated to 200 chars each
    - Arrays show only first 20 items
    - Result marked as 'partial' with metadata indicating truncation

    This is a JSON-specific limit. Other file types use:
    - Text files: max_lines_per_read (line-based limiting)
    - PDF files: max_pages_per_read (page-based limiting)
    - All files: max_characters_absolute (hard safety limit)
    """

    max_image_pixels: int = 2 * 1024 * 1024  # 2 megapixels (e.g., 2048x1024 or 1448x1448)
    """
    Maximum pixels for single image files (JPG, PNG, etc.).
    Based on VLM tokenization: ~2MP = 512-2048 tokens depending on model.
    Examples:
    - GPT-4V: 170 tokens per 512x512 tile, so ~680 tokens for 2MP
    - Claude 3: ~1600 tokens per 1568x1568 image (2.46MP)
    - Gemini: Variable based on resolution
    Default: 2 megapixels (good balance for most models)

    Note: This limit applies ONLY to single image files.
    PDFs are limited by max_pages_per_read, not pixel count.
    """

    max_images_per_read: int = 4
    """
    Maximum number of images to extract from PDFs in a single read operation.
    This works together with max_pages_per_read to limit PDF image extraction.
    Note: Single image files are limited by max_image_pixels, not this parameter.
    """

    max_characters_absolute: int = 120_000
    """
    ABSOLUTE maximum characters allowed in any single read operation (safety limit).

    This is a hard limit enforced regardless of strategy (full or partial).
    Prevents accidental context window overflow from excessively large file reads.

    Default: 120,000 characters (~30K tokens)

    ⚠️  WARNING: Setting this too high can cause serious issues:
    - Context window overflow (most models: 128K-200K tokens total)
    - Out of memory errors
    - API timeout failures
    - Workflow crashes

    Recommended: Keep at or below 120K characters (30K tokens) to leave room for:
    - System prompts (~2-5K tokens)
    - Agent memory/history (~10-20K tokens)
    - Images (if any) (~500-2000 tokens per image)
    - Response generation (~2-10K tokens)

    Only increase if:
    - You're using models with large context windows (200K+)
    - You understand the token budget implications
    - You've tested with your specific use case
    """

    default_read_strategy: str = "partial"
    """
    Default reading strategy for file operations (config-level setting).
    This controls the default behavior when agents request file reads without specifying ranges.

    Options:
    - "partial" (default): Read first N pages/lines (controlled by max_pages_per_read / max_lines_per_read)
      - For PDFs: First 5 pages by default
      - For text: First 1000 lines by default
      - For JSON: Full content if under max_characters_per_read, otherwise overview
      - For images: Full image if under max_image_pixels

    - "auto": Smart selection based on file size
      - Small files (< limits): Full content
      - Large files (> limits): Partial content

    - "full": Always read complete file (up to configured limits)
      - Use with caution: May exceed token limits for large files
      - Still respects max_pages_per_read, max_lines_per_read, etc.

    Note: This is a config-level setting, NOT exposed to agents.
    Agents cannot override this setting via read_file parameters.
    When agents specify page/line ranges, PARTIAL strategy is always used.
    """

    max_files_per_operation: int = 20
    """Maximum number of files that can be processed in a batch operation."""

    max_search_results: int = 100
    """Maximum number of search results to return."""

    max_context_lines: int = 10
    """Maximum number of context lines before/after search matches."""

    # === Feature Flags ===

    enable_delete: bool = False
    """
    Whether to enable file deletion operations.
    Disabled by default for safety. Set to True to allow delete operations.
    """

    enable_tree_sitter: bool = True
    """Whether to enable tree-sitter for code structure extraction."""

    enable_semantic_search: bool = False
    """
    Whether to enable semantic search (requires indexing).
    Disabled by default as it requires additional setup and resources.
    """

    enable_caching: bool = True
    """Whether to enable caching of file structures and contents."""

    cache_ttl_seconds: int = 300
    """Time-to-live for cached data in seconds (5 minutes default)."""

    # === Read Strategy Settings ===

    small_file_threshold: int = 10000  # 10k characters
    """
    Files with content smaller than this (in characters) use FULL strategy in AUTO mode.
    Default: 10k characters ≈ 2.5k tokens.
    """

    medium_file_threshold: int = 100000  # 100k characters
    """
    Files between small and medium threshold use PARTIAL strategy in AUTO mode.
    Default: 100k characters ≈ 25k tokens.
    """

    large_file_threshold: int = 500000  # 500k characters
    """
    Files larger than this (in characters) start with OVERVIEW strategy in AUTO mode.
    Default: 500k characters ≈ 125k tokens.
    """

    # === Edit Settings ===

    default_edit_format: str = "unified_diff"
    """Default format for edits: 'unified_diff' or 'search_replace'."""

    enable_flexible_patching: bool = True
    """Whether to enable flexible patching algorithm for unified diffs."""

    flexible_patch_max_attempts: int = 5
    """Maximum number of patching strategies to try."""

    # === Approval Workflow Settings ===

    approval_timeout_seconds: Optional[float] = 60.0
    """
    Timeout for approval requests in seconds.
    None means no timeout (wait indefinitely).
    """

    auto_approve_on_timeout: bool = False
    """
    If True, auto-approve operations that time out.
    If False, reject operations that time out.
    """

    # === Logging and Monitoring ===

    enable_audit_logging: bool = True
    """Whether to log all file operations for audit purposes."""

    log_file_path: Optional[Path] = None
    """Path to audit log file. If None, logs to standard logging."""

    enable_metrics: bool = True
    """Whether to collect performance metrics."""

    # === Advanced Settings ===

    follow_symlinks: bool = False
    """Whether to follow symbolic links. False by default for security."""

    encoding_fallback: str = "utf-8"
    """Fallback encoding if detection fails."""

    max_tree_depth: int = 10
    """Maximum depth for recursive directory operations."""

    # === Virtual Filesystem Settings ===

    run_filesystem: Optional["RunFileSystem"] = None
    """
    Shared run filesystem for unified path resolution across tools.
    If provided, tool path resolution uses this object.
    If None, one is auto-created from base_directory.
    """

    extra_mounts: Optional[Dict[str, Path]] = None
    """
    Additional mount points mapping virtual prefixes to host directories.
    Each entry creates a mount point accessible via the virtual prefix.
    Example: {"/datasets": Path("/shared/data"), "/memory": Path("/persistent/memory")}
    Only used when auto-creating the RunFileSystem (i.e., run_filesystem is None).
    """

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert base_directory to Path if it's a string
        if self.base_directory is not None and not isinstance(self.base_directory, Path):
            self.base_directory = Path(self.base_directory)

        # Resolve to absolute path
        if self.base_directory is not None:
            self.base_directory = self.base_directory.resolve()

        # Convert log_file_path to Path if it's a string
        if self.log_file_path is not None and not isinstance(self.log_file_path, Path):
            self.log_file_path = Path(self.log_file_path)

        # Validate thresholds
        if self.max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")

        if self.max_json_content_chars <= 0:
            raise ValueError("max_json_content_chars must be positive")

        if self.small_file_threshold >= self.medium_file_threshold:
            raise ValueError("small_file_threshold must be less than medium_file_threshold")

        if self.medium_file_threshold >= self.large_file_threshold:
            raise ValueError("medium_file_threshold must be less than large_file_threshold")

        if self.max_image_pixels <= 0:
            raise ValueError("max_image_pixels must be positive")

        if self.max_images_per_read <= 0:
            raise ValueError("max_images_per_read must be positive")

        # Auto-create RunFileSystem if not provided
        if self.run_filesystem is None:
            from marsys.environment.filesystem import RunFileSystem

            # Use base_directory or cwd as the root
            root_dir = self.base_directory if self.base_directory is not None else Path.cwd()
            self.run_filesystem = RunFileSystem.local(
                run_root=root_dir,
                extra_mounts=self.extra_mounts,
                allow_symlink_escape=self.follow_symlinks,
            )

    def _match_pattern(self, path: Path, pattern: str) -> bool:
        """
        Helper to match a path against a glob pattern.
        Handles ** patterns correctly for both root and nested paths.

        Args:
            path: Path to match
            pattern: Glob pattern

        Returns:
            True if path matches pattern
        """
        from pathlib import PurePath

        pure_path = PurePath(path)
        path_str = str(pure_path).replace('\\', '/')

        # Try direct match
        if pure_path.match(pattern):
            return True

        # For patterns starting with **/,  also try without the **/ prefix
        # This allows matching files at root level
        if pattern.startswith('**/'):
            pattern_without_prefix = pattern[3:]
            if pure_path.match(pattern_without_prefix):
                return True

        # Handle patterns like **/node_modules/** (middle directory with **)
        if pattern.startswith('**/') and pattern.endswith('/**'):
            middle_part = pattern[3:-3]  # Remove **/ prefix and /** suffix
            # Check if this directory is in the path parts
            path_parts = pure_path.parts
            if middle_part in path_parts:
                return True

        # Handle patterns with **/something (filename at root or nested)
        if pattern.startswith('**/') and '/**' not in pattern[3:]:
            # Pattern like **/requirements*.txt - match filename
            import fnmatch
            path_parts = pure_path.parts
            filename = path_parts[-1] if path_parts else ''
            pattern_filename = pattern[3:]  # Remove **/
            if fnmatch.fnmatch(filename, pattern_filename):
                return True

        return False

    def is_file_blocked(self, path: Path) -> bool:
        """
        Quick check if a file matches blocked patterns.

        Args:
            path: File path to check

        Returns:
            True if file is blocked, False otherwise
        """
        # Check allowed patterns first (they override blocked)
        for pattern in self.allowed_patterns:
            if self._match_pattern(path, pattern):
                return False

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if self._match_pattern(path, pattern):
                return True

        return False

    def should_auto_approve(self, path: Path) -> bool:
        """
        Check if a file should be auto-approved for operations.

        Args:
            path: File path to check

        Returns:
            True if file should be auto-approved, False otherwise
        """
        for pattern in self.auto_approve_patterns:
            if self._match_pattern(path, pattern):
                return True

        return False

    def requires_approval(self, path: Path) -> bool:
        """
        Check if a file requires user approval before modification.

        Args:
            path: File path to check

        Returns:
            True if file requires approval, False otherwise
        """
        # Check require_approval patterns first (more specific patterns take precedence)
        for pattern in self.require_approval_patterns:
            if self._match_pattern(path, pattern):
                return True

        # If explicitly auto-approved and not in require_approval, doesn't require approval
        if self.should_auto_approve(path):
            return False

        return False

    @classmethod
    def create_permissive(cls, base_directory: Optional[Path] = None) -> 'FileOperationConfig':
        """
        Create a permissive configuration with minimal restrictions.

        Useful for testing or trusted environments.

        Args:
            base_directory: Optional base directory

        Returns:
            Permissive FileOperationConfig
        """
        return cls(
            base_directory=base_directory,
            blocked_patterns=[],
            auto_approve_patterns=["**/*"],
            require_approval_patterns=[],
            enable_delete=True,
            max_file_size_bytes=100 * 1024 * 1024,  # 100 MB
        )

    @classmethod
    def create_restrictive(cls, base_directory: Path) -> 'FileOperationConfig':
        """
        Create a restrictive configuration with maximum security.

        Useful for production or untrusted environments.

        Args:
            base_directory: Required base directory

        Returns:
            Restrictive FileOperationConfig
        """
        return cls(
            base_directory=base_directory,
            blocked_patterns=DEFAULT_BLOCKED_PATTERNS + [
                "**/*.exe",
                "**/*.dll",
                "**/*.so",
                "**/*.dylib",
                "**/bin/**",
                "**/obj/**",
            ],
            auto_approve_patterns=[],  # Nothing auto-approved
            require_approval_patterns=["**/*"],  # Everything requires approval
            enable_delete=False,
            max_file_size_bytes=5 * 1024 * 1024,  # 5 MB
            approval_timeout_seconds=30.0,
            auto_approve_on_timeout=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FileOperationConfig(base_dir={self.base_directory}, "
            f"blocked={len(self.blocked_patterns)}, "
            f"auto_approve={len(self.auto_approve_patterns)}, "
            f"require_approval={len(self.require_approval_patterns)})"
        )

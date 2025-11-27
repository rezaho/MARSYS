"""
Main FileOperationTools interface.

This module provides the primary API for file operations in MARSYS agents.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union

from ..tool_response import ToolResponse
from .config import FileOperationConfig
from .data_models import (
    FileContent,
    DocumentStructure,
    EditResult,
    SearchResults,
    DirectoryResult,
    DeleteResult,
    FileInfo,
    ReadStrategy,
    SearchType,
    EditFormat,
)
from .security import SecurityManager
from .readers import IntelligentFileReader
from .editors import UnifiedDiffEditor, SearchReplaceEditor
from .search import FileSearchEngine
from .handlers.text import TextFileHandler
from .handlers.pdf_handler import PDFHandler
from .handlers.image_handler import ImageFileHandler

logger = logging.getLogger(__name__)


class FileOperationTools:
    """
    Main interface for file operations in MARSYS agents.

    Provides intelligent, secure, and type-aware file operations including:
    - Smart reading strategies (full, partial, overview, progressive)
    - Hierarchical structure extraction
    - Unified diff editing with high success rate
    - Search capabilities (content, filename, structure)
    - Security controls and approval workflows
    """

    def __init__(self, config: Optional[FileOperationConfig] = None):
        """
        Initialize file operation tools.

        Args:
            config: Configuration (if None, uses defaults)
        """
        # Use provided config or create default
        if config is None:
            config = FileOperationConfig(
                base_directory=Path.cwd(),
                force_base_directory=False
            )

        self.config = config

        # Initialize security manager
        self.security = SecurityManager(config)

        # Initialize handler registry
        self.handler_registry = self._build_handler_registry()

        # Initialize core components
        self.reader = IntelligentFileReader(config, self.handler_registry)
        self.diff_editor = UnifiedDiffEditor(config)
        self.replace_editor = SearchReplaceEditor()
        self.search_engine = FileSearchEngine(config, self.handler_registry)

        logger.info(f"FileOperationTools initialized with {len(self.handler_registry)} handlers")

    def _build_handler_registry(self) -> Dict[str, Any]:
        """Build registry of file type handlers."""
        registry = {}

        # Text handler
        registry['text'] = TextFileHandler()

        # Image handler (if Pillow available)
        try:
            registry['image'] = ImageFileHandler()
            logger.debug("Image handler registered")
        except ImportError as e:
            logger.warning(f"Image handler not available (Pillow not installed): {e}")

        # PDF handler (if available)
        try:
            registry['pdf'] = PDFHandler()
            logger.debug("PDF handler registered")
        except ImportError as e:
            logger.warning(f"PDF handler not available: {e}")

        # TODO: Add more handlers as they're implemented
        # registry['python'] = PythonHandler()
        # registry['markdown'] = MarkdownHandler()
        # registry['json'] = JSONHandler()
        # registry['yaml'] = YAMLHandler()

        return registry

    # ========== Core File Operations ==========

    async def read(
        self,
        path: Union[str, Path],
        strategy: Union[str, ReadStrategy] = ReadStrategy.AUTO,
        **kwargs
    ) -> FileContent:
        """
        Read file with intelligent strategy selection.

        Args:
            path: File path to read
            strategy: Reading strategy (AUTO, FULL, PARTIAL, OVERVIEW, PROGRESSIVE)
            **kwargs: Additional arguments:
                - start_line: Starting line number (for PARTIAL)
                - end_line: Ending line number (for PARTIAL)
                - section: Section ID to read
                - max_depth: Maximum depth for structure (for OVERVIEW/PROGRESSIVE)
                - encoding: File encoding (optional)

        Returns:
            FileContent with file contents and metadata

        Raises:
            ValueError: If path is invalid or unauthorized
            FileNotFoundError: If file doesn't exist
        """
        # Convert to Path and resolve relative to base_directory
        path = Path(path)
        if not path.is_absolute() and self.config.base_directory:
            path = self.config.base_directory / path

        # Security check
        validation = await self.security.authorize_operation("read", path)
        if not validation.allowed:
            raise ValueError(f"Read not allowed: {validation.reason}")

        # Convert strategy string to enum
        if isinstance(strategy, str):
            strategy = ReadStrategy(strategy.lower())

        # Pass max_pixels config to kwargs for single image files only
        # Note: PDFs use page limits, not pixel limits
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif']:
            if 'max_pixels' not in kwargs:
                kwargs['max_pixels'] = self.config.max_image_pixels

        # Read file
        content = await self.reader.read(path, strategy, **kwargs)

        # Validate image pixel limits ONLY for single image files (not PDFs)
        # PDFs are limited by page count (max_pages_per_read), not total pixels
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif']:
            max_pixels_limit = kwargs.get('max_pixels', self.config.max_image_pixels)
            if content.total_image_pixels is not None and content.total_image_pixels > 0:
                if content.total_image_pixels > max_pixels_limit:
                    raise ValueError(
                        f"Image exceeds pixel limit: {content.total_image_pixels} > {max_pixels_limit}"
                    )

        # Log operation
        self.security.log_operation("read", path, True, {
            'strategy': strategy.value,
            'partial': content.partial
        })

        return content

    async def write(
        self,
        path: Union[str, Path],
        content: str,
        mode: str = "write",
        create_parents: bool = True,
        **kwargs
    ) -> EditResult:
        """
        Write or append content to a file.

        Args:
            path: File path to write
            content: Content to write
            mode: Write mode - "write" (overwrite) or "append" (add to end)
            create_parents: Whether to create parent directories
            **kwargs: Additional arguments (encoding, etc.)

        Returns:
            EditResult indicating success/failure
        """
        path = Path(path)

        # Validate mode
        mode_lower = mode.lower()
        if mode_lower not in ("write", "append"):
            return EditResult(
                success=False,
                path=path,
                error=f"Invalid mode '{mode}'. Must be 'write' or 'append'."
            )

        # Map to Python file modes
        file_mode = 'w' if mode_lower == "write" else 'a'

        # Security check
        validation = await self.security.authorize_operation("write", path)
        if not validation.allowed:
            raise PermissionError(f"Write not allowed: {validation.reason}")

        try:
            # Create parent directories if needed
            if create_parents and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

            # Write/append to file
            with open(path, file_mode, encoding=kwargs.get('encoding', 'utf-8')) as f:
                f.write(content)

            # Count lines
            lines_written = content.count('\n') + 1

            # Log operation
            self.security.log_operation("write", path, True, {
                'mode': mode_lower,
                'lines': lines_written,
                'size': len(content)
            })

            return EditResult(
                success=True,
                path=path,
                lines_changed=lines_written
            )

        except Exception as e:
            logger.error(f"Error writing {path}: {e}", exc_info=True)
            self.security.log_operation("write", path, False, {'error': str(e)})
            return EditResult(
                success=False,
                path=path,
                error=str(e)
            )

    async def edit(
        self,
        path: Union[str, Path],
        changes: Union[str, Dict[str, str]],
        edit_format: Union[str, EditFormat] = None,
        dry_run: bool = False,
        **kwargs
    ) -> EditResult:
        """
        Edit file using unified diff or search/replace.

        Args:
            path: File path to edit
            changes: Either unified diff string or dict with 'search' and 'replace' keys
            edit_format: Format of changes (unified_diff or search_replace)
            dry_run: If True, preview changes without applying
            **kwargs: Additional arguments

        Returns:
            EditResult with operation results
        """
        path = Path(path)

        # Auto-detect format if not specified
        if edit_format is None:
            if isinstance(changes, str) and ('---' in changes or '+++' in changes or '@@' in changes):
                edit_format = EditFormat.UNIFIED_DIFF
            elif isinstance(changes, dict):
                edit_format = EditFormat.SEARCH_REPLACE
            else:
                edit_format = EditFormat(self.config.default_edit_format)

        # Convert to enum
        if isinstance(edit_format, str):
            edit_format = EditFormat(edit_format.lower())

        # Security check (get approval if needed, but don't apply if dry_run)
        if not dry_run:
            # Generate preview for approval
            if edit_format == EditFormat.UNIFIED_DIFF:
                preview = changes if isinstance(changes, str) else str(changes)
            else:
                preview = f"Search: {changes.get('search', '')}\nReplace: {changes.get('replace', '')}"

            validation = await self.security.authorize_operation(
                "edit",
                path,
                preview=preview[:500]  # Limit preview size
            )

            if not validation.allowed:
                raise PermissionError(f"Edit not allowed: {validation.reason}")

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Apply edit
        try:
            if edit_format == EditFormat.UNIFIED_DIFF:
                result = await self.diff_editor.apply_diff(
                    path,
                    changes if isinstance(changes, str) else str(changes),
                    dry_run=dry_run,
                    **kwargs
                )
            else:  # SEARCH_REPLACE
                if not isinstance(changes, dict):
                    return EditResult(
                        success=False,
                        path=path,
                        error="Search/replace format requires dict with 'search' and 'replace' keys"
                    )

                if dry_run:
                    # For dry run, just validate search exists
                    with open(path, 'r') as f:
                        content = f.read()
                    if changes['search'] not in content:
                        return EditResult(
                            success=False,
                            path=path,
                            dry_run=True,
                            error=f"Search text not found: {changes['search'][:50]}"
                        )
                    return EditResult(
                        success=True,
                        path=path,
                        dry_run=True,
                        preview=f"Would replace: {changes['search']}\nWith: {changes['replace']}"
                    )

                result = await self.replace_editor.edit(
                    path,
                    changes['search'],
                    changes['replace'],
                    kwargs.get('replace_all', False)
                )

            # Log operation if not dry run
            if not dry_run:
                self.security.log_operation("edit", path, result.success, {
                    'format': edit_format.value,
                    'lines_changed': result.lines_changed
                })

            return result

        except Exception as e:
            logger.error(f"Error editing {path}: {e}", exc_info=True)
            return EditResult(
                success=False,
                path=path,
                error=str(e)
            )

    async def search(
        self,
        query: str,
        search_type: Union[str, SearchType] = SearchType.CONTENT,
        path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> SearchResults:
        """
        Search for files or content.

        Args:
            query: Search query
            search_type: Type of search (content, filename, structure)
            path: Root path to search (default: base_directory)
            **kwargs: Additional search parameters

        Returns:
            SearchResults with matches
        """
        # Convert to Path
        if path:
            path = Path(path)

        # Convert search type
        if isinstance(search_type, str):
            search_type = SearchType(search_type.lower())

        # Use base directory if no path specified
        if path is None:
            path = self.config.base_directory or Path.cwd()

        # Security check
        validation = self.security.validate_path(path, "search")
        if not validation.allowed:
            return SearchResults(
                query=query,
                search_type=search_type,
                matches=[],
                total_matches=0,
                metadata={'error': validation.reason}
            )

        # Execute search
        results = await self.search_engine.search(query, search_type, path, **kwargs)

        return results

    async def get_structure(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> DocumentStructure:
        """
        Get hierarchical structure of a file.

        Args:
            path: File path
            **kwargs: Additional arguments (max_depth, etc.)

        Returns:
            DocumentStructure with file hierarchy
        """
        path = Path(path)

        # Security check
        validation = await self.security.authorize_operation("read", path)
        if not validation.allowed:
            raise ValueError(f"Read not allowed: {validation.reason}")

        # Get handler
        handler = self._get_handler(path)
        if not handler:
            raise ValueError(f"No handler found for file: {path}")

        # Get structure
        structure = await handler.get_structure(path, **kwargs)

        return structure

    async def read_section(
        self,
        path: Union[str, Path],
        section_id: str,
        **kwargs
    ) -> str:
        """
        Read a specific section from a file.

        Args:
            path: File path
            section_id: Section identifier
            **kwargs: Additional arguments

        Returns:
            Content of the section
        """
        path = Path(path)

        # Security check
        validation = await self.security.authorize_operation("read", path)
        if not validation.allowed:
            raise ValueError(f"Read not allowed: {validation.reason}")

        # Get handler
        handler = self._get_handler(path)
        if not handler:
            raise ValueError(f"No handler found for file: {path}")

        # Read section
        content = await handler.read_section(path, section_id, **kwargs)

        return content

    # ========== Directory Operations ==========

    async def list_files(
        self,
        path: Union[str, Path],
        pattern: Optional[str] = None,
        recursive: bool = False,
        **kwargs
    ) -> List[FileInfo]:
        """
        List files in a directory.

        Args:
            path: Directory path
            pattern: Optional glob pattern
            recursive: Whether to search recursively
            **kwargs: Additional arguments

        Returns:
            List of FileInfo objects
        """
        path = Path(path)

        # Security check
        validation = self.security.validate_path(path, "list")
        if not validation.allowed:
            raise ValueError(f"List not allowed: {validation.reason}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        # Get file list
        if pattern:
            glob_pattern = f"**/{pattern}" if recursive else pattern
            files = list(path.glob(glob_pattern))
        else:
            if recursive:
                files = list(path.rglob("*"))
            else:
                files = list(path.iterdir())

        # Filter out blocked files
        files = [f for f in files if not self.config.is_file_blocked(f)]

        # Convert to FileInfo
        file_infos = []
        for file_path in files:
            try:
                stat = file_path.stat()
                size = stat.st_size
                size_human = self._format_size(size)

                file_infos.append(FileInfo(
                    path=file_path,
                    size=size,
                    size_human=size_human,
                    modified=stat.st_mtime,
                    is_file=file_path.is_file(),
                    is_dir=file_path.is_dir(),
                    is_symlink=file_path.is_symlink(),
                    extension=file_path.suffix.lstrip('.')
                ))
            except Exception as e:
                logger.warning(f"Error getting info for {file_path}: {e}")

        return file_infos

    async def create_directory(
        self,
        path: Union[str, Path],
        parents: bool = True,
        **kwargs
    ) -> DirectoryResult:
        """
        Create a directory.

        Args:
            path: Directory path
            parents: Whether to create parent directories
            **kwargs: Additional arguments

        Returns:
            DirectoryResult with success status and details
        """
        path = Path(path)

        # Security check
        validation = await self.security.authorize_operation("write", path)
        if not validation.allowed:
            return DirectoryResult(
                success=False,
                path=path,
                error=f"Create directory not allowed: {validation.reason}"
            )

        try:
            already_existed = path.exists()
            path.mkdir(parents=parents, exist_ok=True)
            self.security.log_operation("create_directory", path, True)
            return DirectoryResult(
                success=True,
                path=path,
                already_existed=already_existed
            )
        except Exception as e:
            logger.error(f"Error creating directory {path}: {e}")
            self.security.log_operation("create_directory", path, False, {'error': str(e)})
            return DirectoryResult(
                success=False,
                path=path,
                error=str(e)
            )

    async def delete(self, path: Union[str, Path], **kwargs) -> DeleteResult:
        """Delete a file or directory."""
        path = Path(path)
        if not self.config.enable_delete:
            raise PermissionError("Delete operations are disabled in configuration")
        validation = await self.security.authorize_operation("delete", path)
        if not validation.allowed:
            raise ValueError(f"Delete not allowed: {validation.reason}")
        try:
            if path.is_dir():
                path.rmdir()
            else:
                path.unlink()
            self.security.log_operation("delete", path, True)
            return DeleteResult(success=True, path=path)
        except Exception as e:
            logger.error(f"Error deleting {path}: {e}")
            self.security.log_operation("delete", path, False, {'error': str(e)})
            return DeleteResult(success=False, path=path, error=str(e))

    # ========== Utility Methods ==========

    def _get_handler(self, path: Path):
        """Get appropriate handler for file type."""
        for handler in self.handler_registry.values():
            if handler.can_handle(path):
                return handler
        # Fallback to text handler
        return self.handler_registry.get('text')

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    # ========== Tool Interface for Agents ==========

    def get_tools(self) -> Dict[str, Callable]:
        """
        Get dictionary of tools for agent integration.

        These wrapper functions convert result objects to dictionaries
        for easier consumption by agents.

        Returns:
            Dict mapping tool name to callable function
        """

        async def read_file_wrapper(
            path: Union[str, Path],
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
            start_page: Optional[int] = None,
            end_page: Optional[int] = None,
            extract_images: bool = True,
            include_overview: bool = False,
            **kwargs
        ) -> ToolResponse:
            """
            Read file with optional line/page ranges, image extraction, and overview.

            Args:
                path: Path to file to read
                start_line: Starting line number for text files (1-indexed)
                end_line: Ending line number for text files (1-indexed, inclusive)
                start_page: Starting page number for PDFs (1-indexed)
                end_page: Ending page number for PDFs (1-indexed, inclusive)
                extract_images: Whether to extract images from PDFs (default: True)
                include_overview: Request document overview/table of contents (default: False)

            Default behavior (no ranges specified):
              - PDFs: First 5 pages with images + auto-overview (configurable)
              - Text files: First 1000 lines (configurable)
              - JSON files: Full content if small, overview if large
              - Images: Full image if under pixel limit

            Auto-overview on first read:
              - When reading a PDF without specifying page ranges, overview is included automatically
              - Overview provides document structure, TOC, and page count
              - Helps agents understand document layout before requesting specific pages

            Using include_overview:
              - Set to True to explicitly request document overview
              - Works for PDFs and structured documents
              - Provides hierarchical structure without reading full content
              - Example: read_file("document.pdf", include_overview=True)

            Reading specific ranges:
              - When you specify start_page/end_page or start_line/end_line, partial reading is used
              - For PDFs: read_file("doc.pdf", start_page=10, end_page=15) → Pages 10-15
              - For text: read_file("file.txt", start_line=100, end_line=200) → Lines 100-200
              - Ranges always use partial reading strategy regardless of config

            Image extraction (PDFs only):
              - Images are extracted by default (extract_images=True)
              - Images returned as base64-encoded data in ordered content blocks
              - Set extract_images=False to skip images if not needed

            Examples:
                # First read - gets overview automatically
                read_file("document.pdf")
                → First 5 pages with images + document overview

                # Request specific pages (no auto-overview)
                read_file("document.pdf", start_page=6, end_page=10)
                → Pages 6-10 with images

                # Get only overview without content
                read_file("document.pdf", include_overview=True, start_page=1, end_page=1)
                → Page 1 with overview structure

                # Skip image extraction for faster reads
                read_file("document.pdf", extract_images=False)
                → First 5 pages without images + overview

                # Text file partial read
                read_file("code.py", start_line=50, end_line=100)
                → Lines 50-100

            Returns:
                ToolResponse with content and metadata
            """
            path = Path(path)

            # Validate page/line range requests
            if start_page is not None and end_page is not None:
                pages_requested = end_page - start_page + 1
                max_pages = self.config.max_pages_per_read

                if pages_requested > max_pages:
                    return {
                        "error": True,
                        "message": f"Request exceeds maximum pages per read",
                        "details": {
                            "requested_pages": pages_requested,
                            "maximum_pages": max_pages,
                            "suggestion": f"Request fewer pages (e.g., start_page={start_page}, end_page={start_page + max_pages - 1})"
                        },
                        "path": str(path)
                    }

            if start_line is not None and end_line is not None:
                lines_requested = end_line - start_line + 1
                max_lines = self.config.max_lines_per_read

                if lines_requested > max_lines:
                    return {
                        "error": True,
                        "message": f"Request exceeds maximum lines per read",
                        "details": {
                            "requested_lines": lines_requested,
                            "maximum_lines": max_lines,
                            "suggestion": f"Request fewer lines (e.g., start_line={start_line}, end_line={start_line + max_lines - 1})"
                        },
                        "path": str(path)
                    }

            # Pass extract_images and include_overview
            kwargs['extract_images'] = extract_images
            kwargs['include_overview'] = include_overview

            # If page/line range specified, use PARTIAL strategy
            if start_page is not None or start_line is not None:
                if start_page is not None:
                    kwargs['start_page'] = start_page
                    kwargs['end_page'] = end_page if end_page is not None else start_page
                if start_line is not None:
                    kwargs['start_line'] = start_line
                    kwargs['end_line'] = end_line if end_line is not None else (start_line + 100)

                result = await self.read(path, ReadStrategy.PARTIAL, **kwargs)
            else:
                # Use strategy from config (not exposed to agent)
                strategy = ReadStrategy[self.config.default_read_strategy.upper()]
                result = await self.read(path, strategy, **kwargs)

            return result.to_tool_response()

        async def write_file_wrapper(
            path: Union[str, Path],
            content: str,
            mode: str = "write",
            **kwargs
        ) -> Dict[str, Any]:
            """
            Write or append content to a file.

            Args:
                path: Path to the file to write
                content: Content to write to the file
                mode: Write mode - "write" (overwrite existing content) or "append" (add to end of file).
                      Use "append" when adding entries to log files, JSONL files, or any file where
                      you want to preserve existing content. Default is "write".

            Returns:
                Dict with success status and details

            Examples:
                # Overwrite a file (default)
                write_file("report.md", "# Report\\nContent here")

                # Append to a JSONL file
                write_file("data.jsonl", '{"key": "value"}\\n', mode="append")
            """
            result = await self.write(path, content, mode=mode, **kwargs)
            return result.to_dict()

        async def edit_file_wrapper(
            path: Union[str, Path],
            changes: Union[str, Dict[str, str]],
            edit_format: Union[str, EditFormat] = None,
            dry_run: bool = False,
            **kwargs
        ) -> Dict[str, Any]:
            """Edit file and return dict with edit results."""
            result = await self.edit(path, changes, edit_format, dry_run, **kwargs)
            return result.to_dict()

        async def search_files_wrapper(
            query: str,
            search_type: Union[str, SearchType] = SearchType.CONTENT,
            path: Optional[Path] = None,
            **kwargs
        ) -> Dict[str, Any]:
            """Search files and return dict with formatted results."""
            result = await self.search(query, search_type, path, **kwargs)
            return result.to_dict()

        async def get_file_structure_wrapper(
            path: Union[str, Path],
            **kwargs
        ) -> Dict[str, Any]:
            """Get file structure and return dict with outline."""
            result = await self.get_structure(path, **kwargs)
            structure_dict = result.to_dict()
            structure_dict["outline"] = result.to_outline()
            return structure_dict

        async def read_section_wrapper(
            path: Union[str, Path],
            section_id: str,
            **kwargs
        ) -> str:
            """Read specific section and return content string."""
            # read_section already returns a string
            return await self.read_section(path, section_id, **kwargs)

        async def list_files_wrapper(
            path: Union[str, Path],
            pattern: Optional[str] = None,
            recursive: bool = False,
            **kwargs
        ) -> Dict[str, Any]:
            """List files and return dict with formatted directory info."""
            file_infos = await self.list_files(path, pattern, recursive, **kwargs)

            return {
                "path": str(path),
                "files": [f.to_dict() for f in file_infos],
                "file_count": sum(1 for f in file_infos if f.is_file),
                "directory_count": sum(1 for f in file_infos if f.is_dir),
                "total_size": sum(f.size for f in file_infos if f.is_file),
                "pattern": pattern,
                "recursive": recursive,
            }

        async def create_directory_wrapper(
            path: Union[str, Path],
            parents: bool = True,
            **kwargs
        ) -> Dict[str, Any]:
            """Create directory and return dict with success status."""
            result = await self.create_directory(path, parents, **kwargs)
            return result.to_dict()

        return {
            'read_file': read_file_wrapper,
            'write_file': write_file_wrapper,
            'edit_file': edit_file_wrapper,
            'search_files': search_files_wrapper,
            'get_file_structure': get_file_structure_wrapper,
            'read_section': read_section_wrapper,
            'list_files': list_files_wrapper,
            'create_directory': create_directory_wrapper,
        }


# ========== Helper Functions ==========

def create_file_operation_tools(
    config: Optional[FileOperationConfig] = None
) -> Dict[str, Callable]:
    """
    Create file operation tools for agent integration.

    This is a convenience function that creates a FileOperationTools instance
    and returns its tool dictionary.

    Args:
        config: Optional configuration (if None, uses defaults)

    Returns:
        Dictionary of tool_name -> callable function

    Example:
        >>> tools = create_file_operation_tools()
        >>> content = await tools['read_file']('document.pdf')
    """
    file_ops = FileOperationTools(config)
    return file_ops.get_tools()

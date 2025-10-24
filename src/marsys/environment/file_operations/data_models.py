"""
Data models for file operations toolkit.

This module defines all data structures used throughout the file operations system,
including content representation, structure models, and result types.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tool_response import ToolResponse


class ReadStrategy(str, Enum):
    """Strategy for reading files."""
    FULL = "full"  # Read entire file
    PARTIAL = "partial"  # Read specific line ranges
    OVERVIEW = "overview"  # Get structure only, no content
    PROGRESSIVE = "progressive"  # Structure first, allow drill-down
    AUTO = "auto"  # Automatically select based on file size


class SearchType(str, Enum):
    """Type of search operation."""
    CONTENT = "content"  # Search within file contents (grep-like)
    FILENAME = "filename"  # Search by filename patterns (glob)
    GLOB = "filename"  # Alias for FILENAME (backward compatibility)
    STRUCTURE = "structure"  # Search within code/document structure


class EditFormat(str, Enum):
    """Format for edit operations."""
    UNIFIED_DIFF = "unified_diff"  # Unified diff format (recommended)
    DIFF = "unified_diff"  # Alias for UNIFIED_DIFF (backward compatibility)
    SEARCH_REPLACE = "search_replace"  # Search/replace blocks
    LINE_BASED = "line_based"  # Line-based editing (replace specific lines)


@dataclass
class Section:
    """
    Represents a section in a hierarchical document structure.

    Sections can be:
    - Functions or classes in code files
    - Chapters or headings in documents
    - Top-level keys in data files
    """
    id: str  # Unique identifier (e.g., "function:process_payment", "heading:introduction")
    title: str  # Display title
    level: int  # Hierarchical level (1 = top level, 2 = subsection, etc.)
    start_line: Optional[int] = None  # Starting line number (1-indexed, None for PDFs)
    end_line: Optional[int] = None  # Ending line number (inclusive)
    start_page: Optional[int] = None  # For PDFs: starting page
    end_page: Optional[int] = None  # For PDFs: ending page
    docstring: Optional[str] = None  # For code: docstring/description
    parameters: Optional[List[str]] = None  # For functions: parameter list
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    children: List['Section'] = field(default_factory=list)  # Nested sections
    parent: Optional['Section'] = None  # Parent section reference

    def __str__(self) -> str:
        """String representation."""
        return f"Section(id='{self.id}', title='{self.title}', level={self.level}, lines={self.start_line}-{self.end_line})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children] if self.children else [],
        }


@dataclass
class DocumentStructure:
    """
    Represents the hierarchical structure of a document or file.
    """
    type: str  # File type (python, markdown, pdf, json, etc.)
    sections: List[Section]  # Top-level sections
    metadata: Dict[str, Any] = field(default_factory=dict)  # File-level metadata

    def find_section(self, section_id: str) -> Optional[Section]:
        """Find a section by its ID (recursive search)."""
        def search(sections: List[Section]) -> Optional[Section]:
            for section in sections:
                if section.id == section_id:
                    return section
                if section.children:
                    result = search(section.children)
                    if result:
                        return result
            return None

        return search(self.sections)

    def to_summary(self, max_depth: int = 2) -> str:
        """
        Generate a text summary of the structure.

        Args:
            max_depth: Maximum depth to display

        Returns:
            Multi-line string summary
        """
        lines = [f"File Type: {self.type}"]
        if self.metadata:
            lines.append(f"Metadata: {self.metadata}")
        lines.append("\nStructure:")

        def format_section(section: Section, depth: int, indent: str = "  ") -> List[str]:
            result = []
            if depth > max_depth:
                return result

            prefix = indent * depth
            line_info = f"lines {section.start_line}-{section.end_line}" if section.end_line else f"line {section.start_line}"
            result.append(f"{prefix}- {section.title} ({line_info})")

            if section.docstring and depth < max_depth:
                # Show first line of docstring
                first_line = section.docstring.split('\n')[0].strip()
                if first_line:
                    result.append(f"{prefix}  \"{first_line}\"")

            for child in section.children:
                result.extend(format_section(child, depth + 1, indent))

            return result

        for section in self.sections:
            lines.extend(format_section(section, 0))

        return "\n".join(lines)

    def to_outline(self) -> str:
        """Generate a concise outline of the structure."""
        lines = [f"=== {self.type.upper()} FILE STRUCTURE ===\n"]

        def format_outline(section: Section, depth: int) -> List[str]:
            result = []
            indent = "  " * depth
            marker = "•" if depth == 0 else "○"
            result.append(f"{indent}{marker} {section.title}")

            for child in section.children:
                result.extend(format_outline(child, depth + 1))

            return result

        for section in self.sections:
            lines.extend(format_outline(section, 0))

        lines.append(f"\nTotal sections: {self._count_sections()}")
        lines.append("Use read_section(path, section_id) to read specific sections")

        return "\n".join(lines)

    def _count_sections(self) -> int:
        """Count total number of sections recursively."""
        def count(sections: List[Section]) -> int:
            total = len(sections)
            for section in sections:
                total += count(section.children)
            return total

        return count(self.sections)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "type": self.type,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
        }


@dataclass
class ImageData:
    """Represents an image within a file."""
    data: bytes  # Raw image bytes
    format: str  # Image format (png, jpg, webp, etc.)
    width: int  # Width in pixels
    height: int  # Height in pixels
    page_number: Optional[int] = None  # Page number if from PDF
    estimated_tokens: Optional[int] = None  # Estimated token count for VLMs
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata (DPI, color space, etc.)

    @property
    def total_pixels(self) -> int:
        """Total number of pixels in the image."""
        return self.width * self.height

    @property
    def megapixels(self) -> float:
        """Total megapixels (for easy comparison)."""
        return self.total_pixels / (1024 * 1024)

    def __str__(self) -> str:
        """String representation."""
        return f"ImageData({self.format}, {self.width}x{self.height}, {self.megapixels:.2f}MP, ~{self.estimated_tokens}tokens)"

    def to_base64(self) -> str:
        """
        Convert image data to base64 data URL for vision APIs.

        Returns:
            Data URL string in format: data:image/{format};base64,{encoded_data}
        """
        import base64
        encoded = base64.b64encode(self.data).decode('ascii')
        return f"data:image/{self.format};base64,{encoded}"


@dataclass
class FileContent:
    """Represents file content with metadata."""
    path: Path  # File path
    content: str  # File content (or summary if partial)
    partial: bool = False  # Whether content is partial
    encoding: str = "utf-8"  # File encoding
    line_range: Optional[tuple[int, int]] = None  # (start_line, end_line) if partial
    total_lines: Optional[int] = None  # Total lines in file
    file_size: Optional[int] = None  # File size in bytes
    character_count: Optional[int] = None  # Total characters in content
    estimated_tokens: Optional[int] = None  # Estimated text tokens (~chars/4)
    structure: Optional[DocumentStructure] = None  # Hierarchical structure if available
    sections_available: bool = False  # Whether sections can be read individually
    images: List[ImageData] = field(default_factory=list)  # Images extracted from file
    total_image_pixels: int = 0  # Total pixels across all images
    total_estimated_image_tokens: int = 0  # Total estimated tokens for all images
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def __str__(self) -> str:
        """String representation."""
        partial_str = " (partial)" if self.partial else ""
        lines_str = f", lines {self.line_range[0]}-{self.line_range[1]}" if self.line_range else ""
        img_str = f", {len(self.images)} images" if self.images else ""
        return f"FileContent({self.path}{partial_str}{lines_str}{img_str})"

    def get_total_estimated_tokens(self) -> int:
        """Get total estimated tokens (text + images)."""
        text_tokens = self.estimated_tokens or (len(self.content) // 4)  # Rough estimate: 1 token ≈ 4 chars
        return text_tokens + self.total_estimated_image_tokens

    def generate_usage_guide(self) -> str:
        """
        Generate detailed usage instructions for agents when content is partial.
        This builds the explanation message that goes in the tool response.
        """
        if not self.partial:
            return ""

        explanation_parts = ["⚠️  PARTIAL CONTENT - This is not the complete file."]

        # PDF-specific guidance
        if self.path.suffix.lower() == '.pdf' and self.metadata:
            pages_read = self.metadata.get('pages_read') or self.metadata.get('pages_shown')
            total_pages = self.metadata.get('total_pages', '?')
            max_pages = self.metadata.get('max_pages_per_read')

            if pages_read:
                explanation_parts.append(f"📄 Showing: Pages {pages_read} of {total_pages} total")

            if max_pages:
                explanation_parts.append(f"⚠️  Maximum pages per request: {max_pages}")
                explanation_parts.append(f"📊 To read more pages: read_file(path, start_page=X, end_page=Y) where (end_page - start_page + 1) ≤ {max_pages}")
            else:
                explanation_parts.append("📊 To read more pages: read_file(path, start_page=X, end_page=Y)")

            explanation_parts.append("📊 To get overview/table of contents: read_file(path, include_overview=True)")
            explanation_parts.append("🔍 To search content: search_files(query, path)")

        # Text file guidance
        elif self.line_range:
            lines_shown = f"{self.line_range[0]}-{self.line_range[1]}"
            total_lines = self.total_lines or '?'
            max_lines = self.metadata.get('max_lines_per_read') if self.metadata else None

            explanation_parts.append(f"📝 Showing: Lines {lines_shown} of {total_lines} total")

            if max_lines:
                explanation_parts.append(f"⚠️  Maximum lines per request: {max_lines}")
                explanation_parts.append(f"📊 To read more lines: read_file(path, start_line=X, end_line=Y) where (end_line - start_line + 1) ≤ {max_lines}")
            else:
                explanation_parts.append("📊 To read more lines: read_file(path, start_line=X, end_line=Y)")

            explanation_parts.append("🔍 To search content: search_files(query, path)")

        return "\n".join(explanation_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        result = {
            "content": self.content,
            "path": str(self.path),
            "partial": self.partial,
            "encoding": self.encoding,
        }

        # Add optional fields
        if self.line_range:
            result["line_range"] = self.line_range
        if self.total_lines:
            result["total_lines"] = self.total_lines
        if self.file_size:
            result["file_size"] = self.file_size
        if self.character_count:
            result["character_count"] = self.character_count
        if self.estimated_tokens:
            result["estimated_tokens"] = self.estimated_tokens

        # Add structure if available
        if self.structure:
            result["structure"] = self.structure.to_dict()
            result["sections_available"] = self.sections_available

        # Add image information
        if self.images:
            # Include base64 encoded images for vision APIs
            result["images"] = [img.to_base64() for img in self.images]
            result["image_metadata"] = [
                {
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "total_pixels": img.total_pixels,
                    "estimated_tokens": img.estimated_tokens,
                    "page_number": img.page_number,
                }
                for img in self.images
            ]
            result["total_image_pixels"] = self.total_image_pixels
            result["total_estimated_image_tokens"] = self.total_estimated_image_tokens

        # Add metadata if present
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_tool_response(self) -> "ToolResponse":
        """
        Convert FileContent to ToolResponse with ordered content blocks.

        This method creates a ToolResponse with:
        - For files with images: ordered list of ToolResponseContent blocks (text + images)
        - For files without images: simple dict (backward compatible)
        - metadata: Summary for tool message (path, partial status, image count)

        The ordered content preserves the reading order of text and images as they
        appear in the file (e.g., for PDFs with diagrams and captions).

        Returns:
            ToolResponse ready to return from a tool function

        Example:
            ```python
            def read_pdf(path: str) -> ToolResponse:
                file_content = read_file(path)
                return file_content.to_tool_response()  # Ordered text + images
            ```
        """
        from marsys.environment.tool_response import ToolResponse, ToolResponseContent

        # Check if we have images
        has_images = self.images and len(self.images) > 0

        if has_images:
            # Create ordered content blocks: text + images
            content_blocks = []
            ordered_chunks = self.metadata.get('ordered_chunks') if self.metadata else None

            if ordered_chunks:
                # PDF with ordered chunks - preserve reading order
                for chunk in ordered_chunks:
                    if chunk["type"] == "text":
                        content_blocks.append(
                            ToolResponseContent(
                                type="text",
                                text=chunk["text"]
                            )
                        )
                    elif chunk["type"] == "image":
                        # Get image from list
                        img_idx = chunk["image_index"]
                        if img_idx < len(self.images):
                            image_data = self.images[img_idx]
                            content_blocks.append(
                                ToolResponseContent(
                                    type="image",
                                    image_data=image_data.to_base64()
                                )
                            )
            else:
                # Simple case: text content then all images (image files, etc.)
                content_blocks.append(
                    ToolResponseContent(
                        type="text",
                        text=self.content
                    )
                )

                # Add images
                for image_data in self.images:
                    content_blocks.append(
                        ToolResponseContent(
                            type="image",
                            image_data=image_data.to_base64()
                        )
                    )

            # Return with ordered content blocks
            return ToolResponse(
                content=content_blocks,
                metadata=self._build_tool_response_metadata()
            )
        else:
            # No images: backward compatible - return dict
            return ToolResponse(
                content=self.to_dict(),
                metadata=self._build_tool_response_metadata()
            )

    def _build_tool_response_metadata(self) -> str:
        """
        Build metadata string for ToolResponse.
        This creates the explanation message that appears in the tool response.
        """
        if self.partial:
            # For partial content: return usage guide (detailed explanation)
            return self.generate_usage_guide()
        else:
            # For complete content: simple success message
            return "Tool execution completed successfully."


@dataclass
class ValidationResult:
    """Result of security validation."""
    allowed: bool  # Whether operation is allowed
    needs_approval: bool = False  # Whether user approval is required
    auto_approved: bool = False  # Whether operation was auto-approved
    reason: Optional[str] = None  # Reason if not allowed
    warnings: List[str] = field(default_factory=list)  # Non-fatal warnings

    def __bool__(self) -> bool:
        """Boolean conversion."""
        return self.allowed


@dataclass
class EditResult:
    """Result of edit operation."""
    success: bool  # Whether edit succeeded
    path: Optional[Path] = None  # File path edited
    hunks_applied: int = 0  # Number of diff hunks applied
    hunks_total: int = 0  # Total number of diff hunks
    lines_changed: int = 0  # Number of lines changed
    dry_run: bool = False  # Whether this was a dry run
    preview: Optional[str] = None  # Preview of changes (for dry run)
    warnings: List[str] = field(default_factory=list)  # Warnings (e.g., whitespace normalized)
    error: Optional[str] = None  # Error message if failed
    strategy_used: Optional[str] = None  # Which patching strategy succeeded
    diff_applied: Optional[str] = None  # The actual diff that was applied

    def __str__(self) -> str:
        """String representation."""
        if not self.success:
            return f"EditResult(success=False, error='{self.error}')"
        if self.dry_run:
            return f"EditResult(dry_run=True, changes={self.lines_changed} lines)"
        return f"EditResult(success=True, {self.hunks_applied}/{self.hunks_total} hunks, {self.lines_changed} lines changed)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        return {
            "success": self.success,
            "path": str(self.path) if self.path else None,
            "hunks_applied": self.hunks_applied,
            "hunks_total": self.hunks_total,
            "lines_changed": self.lines_changed,
            "dry_run": self.dry_run,
            "preview": self.preview,
            "warnings": self.warnings,
            "error": self.error,
            "strategy_used": self.strategy_used,
            "diff_applied": self.diff_applied,
        }


@dataclass
class SearchMatch:
    """A single search match."""
    file: Path  # File path
    line: int  # Line number (1-indexed)
    match: str  # Matching line
    column: Optional[int] = None  # Column number (0-indexed)
    context_before: List[str] = field(default_factory=list)  # Lines before match
    context_after: List[str] = field(default_factory=list)  # Lines after match
    relevance_score: Optional[float] = None  # For semantic search
    section_id: Optional[str] = None  # If match is in a specific section
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata (page number, file type, etc.)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.file}:{self.line}:{self.match[:80]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        result = {
            "file": str(self.file),
            "line": self.line,
            "match": self.match,
        }

        # Add human-readable location string
        if self.metadata.get('page'):
            result["location"] = f"page {self.metadata['page']}, line {self.line}"
            result["page"] = self.metadata['page']
        else:
            result["location"] = f"line {self.line}"

        if self.column is not None:
            result["column"] = self.column
        if self.context_before:
            result["context_before"] = self.context_before
        if self.context_after:
            result["context_after"] = self.context_after
        if self.relevance_score is not None:
            result["relevance_score"] = self.relevance_score
        if self.section_id:
            result["section_id"] = self.section_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class SearchResults:
    """Results from search operation."""
    query: str  # Search query
    search_type: SearchType  # Type of search performed
    matches: List[SearchMatch]  # List of matches
    total_matches: int  # Total number of matches (may be > len(matches) if truncated)
    truncated: bool = False  # Whether results were truncated
    files_searched: int = 0  # Number of files searched
    search_duration_ms: Optional[float] = None  # Search duration in milliseconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    @property
    def num_files(self) -> int:
        """Alias for files_searched (for backward compatibility)."""
        return self.files_searched

    def __str__(self) -> str:
        """String representation."""
        truncated_str = " (truncated)" if self.truncated else ""
        return f"SearchResults(query='{self.query}', matches={len(self.matches)}/{self.total_matches}{truncated_str})"

    def format_results(self, max_results: int = 50, show_context: bool = True) -> str:
        """
        Format search results as text.

        Args:
            max_results: Maximum number of results to show
            show_context: Whether to show context lines

        Returns:
            Formatted string
        """
        lines = [
            f"Search Results for '{self.query}' ({self.search_type.value})",
            f"Found {self.total_matches} matches in {self.files_searched} files",
            ""
        ]

        for i, match in enumerate(self.matches[:max_results], 1):
            lines.append(f"{i}. {match.file}:{match.line}")

            if show_context and match.context_before:
                for ctx_line in match.context_before:
                    lines.append(f"  {ctx_line}")

            lines.append(f"  > {match.match}")

            if show_context and match.context_after:
                for ctx_line in match.context_after:
                    lines.append(f"  {ctx_line}")

            lines.append("")  # Blank line between matches

        if self.truncated or len(self.matches) > max_results:
            lines.append(f"... and {self.total_matches - max_results} more matches")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        return {
            "query": self.query,
            "search_type": self.search_type.value,
            "matches": [match.to_dict() for match in self.matches],
            "total_matches": self.total_matches,
            "num_files": self.num_files,
            "truncated": self.truncated,
            "files_searched": self.files_searched,
            "search_duration_ms": self.search_duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class DirectoryResult:
    """Result of directory creation operation."""
    success: bool  # Whether operation succeeded
    path: Optional[Path] = None  # Directory path
    already_existed: bool = False  # Whether directory already existed
    error: Optional[str] = None  # Error message if failed

    def __bool__(self) -> bool:
        """Boolean conversion."""
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        return {
            "success": self.success,
            "path": str(self.path) if self.path else None,
            "already_existed": self.already_existed,
            "error": self.error,
        }


@dataclass
class DeleteResult:
    """Result of delete operation."""
    success: bool  # Whether operation succeeded
    path: Optional[Path] = None  # Path that was deleted
    error: Optional[str] = None  # Error message if failed

    def __bool__(self) -> bool:
        """Boolean conversion."""
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        return {
            "success": self.success,
            "path": str(self.path) if self.path else None,
            "error": self.error,
        }


@dataclass
class FileInfo:
    """File metadata information."""
    path: Path  # File path
    size: int  # File size in bytes
    size_human: str  # Human-readable size (e.g., "4.5 MB")
    modified: str  # Last modified timestamp (ISO format)
    created: Optional[str] = None  # Created timestamp (ISO format)
    is_file: bool = True  # Whether this is a file (vs directory)
    is_dir: bool = False  # Whether this is a directory
    is_symlink: bool = False  # Whether this is a symbolic link
    permissions: Optional[str] = None  # File permissions (e.g., "rw-r--r--")
    owner: Optional[str] = None  # File owner
    encoding: Optional[str] = None  # Detected encoding for text files
    line_count: Optional[int] = None  # Total lines for text files
    extension: str = ""  # File extension
    mime_type: Optional[str] = None  # MIME type

    def __str__(self) -> str:
        """String representation."""
        type_str = "dir" if self.is_dir else "file"
        return f"FileInfo({type_str}: {self.path}, {self.size_human})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for agent consumption."""
        result = {
            "path": str(self.path),
            "name": self.path.name,
            "size": self.size,
            "size_human": self.size_human,
            "modified": self.modified,
            "is_file": self.is_file,
            "is_directory": self.is_dir,
            "is_symlink": self.is_symlink,
            "extension": self.extension,
        }
        if self.created:
            result["created"] = self.created
        if self.permissions:
            result["permissions"] = self.permissions
        if self.owner:
            result["owner"] = self.owner
        if self.encoding:
            result["encoding"] = self.encoding
        if self.line_count is not None:
            result["line_count"] = self.line_count
        if self.mime_type:
            result["mime_type"] = self.mime_type
        return result


# Type aliases for convenience
StructureDict = Dict[str, Any]  # Dictionary representation of DocumentStructure
SectionDict = Dict[str, Any]  # Dictionary representation of Section

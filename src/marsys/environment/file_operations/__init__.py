"""
File Operations Toolkit for MARSYS Agents

Provides intelligent, secure, and type-aware file operations with:
- Hierarchical content extraction
- Multiple reading strategies (full, partial, overview, progressive)
- Security framework with configurable boundaries
- Type-specific handlers for Python, PDF, Markdown, JSON, YAML, etc.
- Unified diff editing with high success rate
- Search capabilities (content, filename, structure)
"""

from .config import FileOperationConfig
from .core import FileOperationTools, create_file_operation_tools
from .data_models import (
    FileContent,
    ImageData,
    DocumentStructure,
    Section,
    ValidationResult,
    EditResult,
    SearchResults,
    DirectoryResult,
    ReadStrategy,
    SearchType,
    EditFormat,
)

__all__ = [
    # Main interface
    "FileOperationTools",
    "create_file_operation_tools",
    # Configuration
    "FileOperationConfig",
    # Data models
    "FileContent",
    "ImageData",
    "DocumentStructure",
    "Section",
    "ValidationResult",
    "EditResult",
    "SearchResults",
    "DirectoryResult",
    # Enums
    "ReadStrategy",
    "SearchType",
    "EditFormat",
]

__version__ = "0.1.0"

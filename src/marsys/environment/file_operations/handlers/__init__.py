"""File type handlers for different file formats."""

from .base import FileHandler
from .text import TextFileHandler
from .pdf_handler import PDFHandler

__all__ = [
    "FileHandler",
    "TextFileHandler",
    "PDFHandler",
]

"""Web tools for file reading and content extraction."""

import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pdfminer.high_level

    PDF_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import for pdfminer.six
        from pdfminer.high_level import extract_text

        pdfminer.high_level = type("module", (), {"extract_text": extract_text})()
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

try:
    from bs4 import BeautifulSoup, Comment

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger(__name__)


async def read_file(file_path: str) -> str:
    """
    Read a file and return its contents.

    Args:
        file_path: Path to the file to read

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine file type and handle accordingly
    file_ext = Path(file_path).suffix.lower()

    if file_ext in [".txt", ".md", ".py", ".js", ".html", ".css", ".json"]:
        # Text files
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # For HTML files, optionally extract text content
        if file_ext == ".html":
            try:
                soup = BeautifulSoup(content, "html.parser")
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                # Get text content
                text_content = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                return "\n".join(chunk for chunk in chunks if chunk)
            except ImportError:
                # BeautifulSoup not available, return raw HTML
                pass

        return content

    elif file_ext == ".pdf":
        return extract_text_from_pdf(file_path)

    else:
        # For other file types, read as binary and return repr
        with open(file_path, "rb") as f:
            content = f.read()
        return f"Binary file ({len(content)} bytes): {file_path}"


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text content
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "PDF support not available. Install with: pip install pdfminer.six"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return pdfminer.high_level.extract_text(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


async def clean_and_extract_html(
    raw_html: str, goal: str, context: str = "", model: Optional[Any] = None
) -> str:
    """
    Extract relevant HTML segment based on goal.

    Args:
        raw_html: The raw HTML content to process
        goal: What we're trying to extract from the HTML
        context: Additional context about the extraction task
        model: Optional model instance to help identify relevant sections

    Returns:
        Cleaned and relevant HTML fragment
    """
    # Clean HTML with BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style", "noscript"]):
        script.decompose()

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove unnecessary attributes
    for tag in soup.find_all():
        # Keep only essential attributes
        allowed_attrs = [
            "id",
            "class",
            "href",
            "src",
            "alt",
            "title",
            "role",
            "aria-label",
        ]
        attrs = dict(tag.attrs)
        for attr in attrs:
            if attr not in allowed_attrs:
                del tag.attrs[attr]

    if model:
        # Use model to identify relevant sections
        cleaned_html = str(soup)

        # Prepare prompt for model
        prompt = f"""Given this HTML content and the goal of "{goal}", identify and return only the most relevant section.
Context: {context}

HTML Content (truncated for brevity):
{cleaned_html[:3000]}...

Return only the relevant HTML section that contains the information we're looking for."""

        try:
            # Call model to identify relevant section
            response = await model.arun(prompt)

            # Extract HTML from response (assuming model returns the HTML wrapped in code blocks)
            html_match = re.search(r"```html?\s*(.*?)\s*```", response, re.DOTALL)
            if html_match:
                return html_match.group(1).strip()
            else:
                # If no code block, try to extract any HTML-like content
                return response.strip()
        except Exception as e:
            logger.warning(
                f"Model extraction failed: {e}. Falling back to heuristic extraction."
            )

    # Fallback: Heuristic extraction based on common patterns
    # Look for main content areas
    main_content = soup.find(
        ["main", "article", '[role="main"]', "#content", ".content"]
    )
    if main_content:
        return str(main_content)

    # Look for body content, excluding headers and footers
    body = soup.find("body")
    if body:
        # Remove headers and footers
        for elem in body.find_all(["header", "footer", "nav"]):
            elem.decompose()
        return str(body)

    # Return cleaned full HTML if no specific section found
    return str(soup)


# Additional utility functions for web content processing


def extract_links(html: str) -> List[Dict[str, str]]:
    """Extract all links from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        links.append(
            {
                "url": link["href"],
                "text": link.get_text(strip=True),
                "title": link.get("title", ""),
            }
        )

    return links


def extract_images(html: str) -> List[Dict[str, str]]:
    """Extract all images from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    images = []

    for img in soup.find_all("img", src=True):
        images.append(
            {
                "src": img["src"],
                "alt": img.get("alt", ""),
                "title": img.get("title", ""),
            }
        )

    return images


def extract_text_content(html: str, preserve_structure: bool = False) -> str:
    """
    Extract text content from HTML.

    Args:
        html: HTML content
        preserve_structure: If True, preserves some structure with newlines

    Returns:
        Extracted text
    """
    soup = BeautifulSoup(html, "html.parser")

    if preserve_structure:
        # Add newlines for block elements
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for p in soup.find_all("p"):
            p.append("\n")
        for div in soup.find_all("div"):
            div.append("\n")
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            h.insert(0, "\n")
            h.append("\n")

    return soup.get_text(separator=" " if not preserve_structure else "", strip=True)

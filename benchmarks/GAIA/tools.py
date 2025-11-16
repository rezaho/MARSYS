"""Tools for GAIA benchmark file handling.

This module provides tools for reading various file types that appear in GAIA
benchmark questions, including images, PDFs, spreadsheets, and text files.

The tools return a consistent format with optional image references for vision models.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def tool_read_file(file_path: str) -> Dict[str, Any]:
    """
    Read GAIA benchmark file attachments (images, PDFs, spreadsheets, text).

    This tool handles various file types and returns a consistent format
    with optional image references for vision models.

    Args:
        file_path: Path to the file to read (absolute or relative)

    Returns:
        Dictionary with:
        - result: Text summary or extracted content
        - images: List of image paths for vision models
        - file_type: Type of file processed
        - metadata: Additional file information

    Examples:
        >>> tool_read_file("chart.png")
        {
            "result": "Loaded image: chart.png",
            "images": ["/path/to/chart.png"],
            "file_type": "image"
        }

        >>> tool_read_file("data.csv")
        {
            "result": "CSV with 10 rows, 5 columns...",
            "images": [],
            "file_type": "spreadsheet"
        }
    """
    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        return {"result": f"Error: File not found: {file_path}", "images": [], "file_type": "error", "error": "file_not_found"}

    file_extension = file_path.suffix.lower()

    try:
        # ============================================================
        # IMAGE FILES - Return path for vision model
        # ============================================================
        if file_extension in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            return {
                "result": f"Loaded image file: {file_path.name}",
                "images": [str(file_path.absolute())],
                "file_type": "image",
                "metadata": {"filename": file_path.name, "size_bytes": file_path.stat().st_size, "extension": file_extension},
            }

        # ============================================================
        # PDF FILES - Extract text and images
        # ============================================================
        elif file_extension == ".pdf":
            try:
                from .pdf_utils import extract_pdf_content

                text, images = extract_pdf_content(str(file_path))

                return {"result": f"PDF content from {file_path.name}:\n\n{text}", "images": images, "file_type": "pdf", "metadata": {"filename": file_path.name, "num_images_extracted": len(images), "text_length": len(text)}}
            except ImportError:
                return {"result": f"Error: PDF support not available. Install PyPDF2 and pdf2image: pip install PyPDF2 pdf2image", "images": [], "file_type": "error", "error": "pdf_support_missing"}

        # ============================================================
        # TEXT FILES - Return content directly
        # ============================================================
        elif file_extension in [".txt", ".md", ".json", ".log", ".csv", ".tsv"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # For JSON, try to pretty-print it
            if file_extension == ".json":
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    pass  # Keep as-is if not valid JSON

            return {"result": content, "images": [], "file_type": "text", "metadata": {"filename": file_path.name, "size_bytes": len(content), "lines": content.count("\n") + 1}}

        # ============================================================
        # CSV FILES - Format as table (if pandas available)
        # ============================================================
        elif file_extension == ".csv" and PANDAS_AVAILABLE:
            df = pd.read_csv(file_path)

            # Create readable table representation
            table_str = f"CSV File: {file_path.name}\n"
            table_str += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            table_str += f"Columns: {', '.join(df.columns)}\n\n"

            # Show first 20 rows
            num_rows_to_show = min(20, len(df))
            table_str += f"First {num_rows_to_show} rows:\n"
            table_str += df.head(num_rows_to_show).to_string(index=False)

            if len(df) > num_rows_to_show:
                table_str += f"\n\n... and {len(df) - num_rows_to_show} more rows"

            return {"result": table_str, "images": [], "file_type": "spreadsheet", "metadata": {"filename": file_path.name, "rows": len(df), "columns": len(df.columns), "column_names": list(df.columns)}}

        # ============================================================
        # EXCEL FILES - Format as table
        # ============================================================
        elif file_extension in [".xlsx", ".xls"] and PANDAS_AVAILABLE:
            df = pd.read_excel(file_path)

            # Create readable table representation
            table_str = f"Excel File: {file_path.name}\n"
            table_str += f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            table_str += f"Columns: {', '.join(df.columns)}\n\n"

            # Show first 20 rows
            num_rows_to_show = min(20, len(df))
            table_str += f"First {num_rows_to_show} rows:\n"
            table_str += df.head(num_rows_to_show).to_string(index=False)

            if len(df) > num_rows_to_show:
                table_str += f"\n\n... and {len(df) - num_rows_to_show} more rows"

            return {"result": table_str, "images": [], "file_type": "spreadsheet", "metadata": {"filename": file_path.name, "rows": len(df), "columns": len(df.columns), "column_names": list(df.columns)}}

        # ============================================================
        # UNSUPPORTED FORMAT
        # ============================================================
        else:
            return {
                "result": f"Unsupported file type: {file_extension}\nSupported types: images (.png, .jpg, .jpeg, .gif, .webp), PDFs (.pdf), text files (.txt, .md, .json), spreadsheets (.csv, .xlsx, .xls)",
                "images": [],
                "file_type": "unknown",
                "error": "unsupported_format",
                "metadata": {"filename": file_path.name, "extension": file_extension},
            }

    except Exception as e:
        return {"result": f"Error reading file {file_path.name}: {str(e)}", "images": [], "file_type": "error", "error": str(e), "metadata": {"filename": file_path.name, "exception_type": type(e).__name__}}


def tool_list_files(directory: str = ".") -> Dict[str, Any]:
    """
    List files in a directory (helper for GAIA questions with multiple files).

    Args:
        directory: Directory path to list (default: current directory)

    Returns:
        Dictionary with list of files and their metadata
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            return {"result": f"Error: Directory not found: {directory}", "files": [], "error": "directory_not_found"}

        if not dir_path.is_dir():
            return {"result": f"Error: Not a directory: {directory}", "files": [], "error": "not_a_directory"}

        # List all files (not directories)
        files = []
        for item in dir_path.iterdir():
            if item.is_file():
                files.append({"name": item.name, "path": str(item.absolute()), "size_bytes": item.stat().st_size, "extension": item.suffix.lower()})

        # Sort by name
        files.sort(key=lambda x: x["name"])

        # Create readable summary
        result_str = f"Directory: {dir_path.absolute()}\n"
        result_str += f"Total files: {len(files)}\n\n"

        if files:
            result_str += "Files:\n"
            for f in files:
                size_kb = f["size_bytes"] / 1024
                result_str += f"  - {f['name']} ({size_kb:.1f} KB)\n"
        else:
            result_str += "No files found in directory."

        return {"result": result_str, "files": files, "metadata": {"directory": str(dir_path.absolute()), "total_files": len(files)}}

    except Exception as e:
        return {"result": f"Error listing directory: {str(e)}", "files": [], "error": str(e)}


def tool_get_youtube_transcript(youtube_url: str, format: str = "text") -> Dict[str, Any]:
    """
    Extract transcript from a YouTube video using Tactiq API.

    This tool fetches the full transcript/captions from a YouTube video by sending
    a request to the Tactiq transcript service. It returns the video title and
    timestamped captions that can be used to answer questions about video content.

    Use this tool when you need to:
    - Extract text content from YouTube videos
    - Read what was said in a video without watching it
    - Find specific information mentioned at certain timestamps
    - Analyze video content that requires text-based understanding

    Args:
        youtube_url: Full YouTube URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID")
                     or just the video ID (e.g., "VIDEO_ID")
        format: Output format - "text" for formatted transcript string (default),
                "json" for raw list of caption objects

    Returns:
        Dictionary with:
        - result: Either formatted text string (if format="text") or list of caption
                  objects (if format="json") with 'start', 'dur', and 'text' fields
        - title: Video title
        - video_id: Extracted YouTube video ID
        - total_captions: Number of caption segments
        - duration_seconds: Approximate video duration based on last caption

        On error:
        - result: Error message
        - error: Error type or description
        - video_id: The video ID that was attempted (if extracted)

    """
    # Extract video ID from URL or use as-is if already just an ID
    video_id = youtube_url

    # Try to extract video ID from various YouTube URL formats
    youtube_patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})",
    ]

    for pattern in youtube_patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break

    # If no pattern matched and input looks like a URL, return error
    if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
        if video_id == youtube_url:
            return {"result": f"Error: Could not extract video ID from URL: {youtube_url}", "error": "invalid_url_format", "video_id": None}

    # Validate video ID format (11 characters, alphanumeric with - and _)
    if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        return {"result": f"Error: Invalid YouTube video ID format: {video_id}. Expected 11 characters.", "error": "invalid_video_id", "video_id": video_id}

    try:
        # Prepare request to Tactiq API
        api_url = "https://tactiq-apps-prod.tactiq.io/transcript"
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Origin": "https://tactiq.io",
            "Referer": "https://tactiq.io/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        payload = {"langCode": "en", "videoUrl": f"https://www.youtube.com/watch?v={video_id}"}

        # Send POST request
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)

        # Accept both 200 and 201 as success
        if response.status_code not in [200, 201]:
            return {
                "result": f"Error: Tactiq API returned status {response.status_code}. Response: {response.text}",
                "error": f"api_error_status_{response.status_code}",
                "video_id": video_id
            }

        data = response.json()

        # Extract title and captions
        title = data.get("title", "No title found")
        captions = data.get("captions", [])

        if not captions:
            return {"result": f"No captions available for video: {title} (ID: {video_id})", "title": title, "captions": [], "video_id": video_id, "total_captions": 0, "error": "no_captions_available"}

        # Calculate approximate duration
        last_caption = captions[-1]
        duration_seconds = float(last_caption.get("start", 0)) + float(last_caption.get("dur", 0))

        # Format result based on format parameter
        if format.lower() == "json":
            # Return raw list of caption objects
            result_data = captions
        else:
            # Format as text with timestamps (default)
            transcript_lines = []
            for caption in captions:
                start_time = float(caption.get("start", 0))
                text = caption.get("text", "")
                if text and text != "No text":
                    transcript_lines.append(f"[{start_time:.2f}s] {text}")
            result_data = "\n".join(transcript_lines)

        return {"result": result_data, "title": title, "video_id": video_id, "total_captions": len(captions), "duration_seconds": duration_seconds, "metadata": {"video_url": f"https://www.youtube.com/watch?v={video_id}", "format": format}}

    except requests.exceptions.RequestException as e:
        return {"result": f"Error: Network error while fetching transcript: {str(e)}", "error": "network_error", "video_id": video_id, "exception_type": type(e).__name__}
    except Exception as e:
        return {"result": f"Error: Failed to fetch transcript: {str(e)}", "error": str(e), "video_id": video_id, "exception_type": type(e).__name__}


# Export tools for easy import
__all__ = ["tool_read_file", "tool_list_files", "tool_get_youtube_transcript"]

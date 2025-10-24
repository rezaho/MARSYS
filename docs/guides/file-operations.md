# File Operations Toolkit

Advanced file management system with intelligent reading strategies, hierarchical content extraction, and secure editing capabilities for MARSYS agents.

## 🎯 Overview

The File Operations Toolkit provides type-aware file handling with advanced features designed for AI agents:

- **Intelligent Reading Strategies**: Optimize token usage with AUTO, FULL, PARTIAL, OVERVIEW, and PROGRESSIVE strategies
- **Character-Based Token Management**: Uses character count (not file size) as proxy for text tokens
- **Image Token Estimation**: Provider-specific token estimation for images (OpenAI, Anthropic, Google, xAI)
- **Hierarchical Structure Extraction**: AST-based parsing for code, font-analysis for PDFs
- **Image Support**: Extract and process images from PDFs and read image files directly
- **Unified Diff Editing**: High-success-rate patching with multiple fallback strategies
- **Security Framework**: Base directory enforcement, pattern-based permissions, approval workflows
- **Search Capabilities**: Content search (grep), filename search (glob), and structure search
- **Type-Specific Handlers**: Specialized handlers for images, PDFs, JSON, YAML, Markdown, and code files

This toolkit was designed to address the limitations of simple file reading tools, particularly for handling complex documents like PDFs and source code files, with intelligent token management for vision-language models.

---

## 📦 Prerequisites

**Core Dependencies:**

PDF support and image support are included in the core marsys installation:
```bash
pip install marsys  # Includes PyMuPDF and Pillow
```

For advanced code parsing (future feature):
```bash
pip install tree-sitter
```

!!! info "Core Dependencies"
    - **PDF support** is provided by `PyMuPDF` (included in core installation).
    - **Image support** is provided by `Pillow` (PIL) (included in core installation).
    - Both are automatically installed with marsys for full functionality with vision-language models.

---

## 🚀 Quick Start

### Basic Usage

```python
import os
from pathlib import Path
from marsys import Agent
from marsys.models import ModelConfig
from marsys.environment import create_file_operation_tools

# Create model configuration (example using OpenRouter)
# Note: API key is only required if you use API-based models
model_config = ModelConfig(
    model_type="api",
    model_name="anthropic/claude-haiku-4.5",
    provider="openrouter",
    api_key=os.getenv("OPENROUTER_API_KEY")  # Set if using OpenRouter
)

# Create file operation tools
file_tools = create_file_operation_tools()

# Create agent with file capabilities
file_agent = Agent(
    model=model_config,
    name="FileAssistant",
    goal="Manage and analyze files efficiently",
    instruction="""You are a file management assistant. Use your file operation tools to:
    - Read files intelligently based on size and type
    - Extract structured information from documents
    - Edit files using unified diff format for reliability
    - Search for content across multiple files
    - Maintain security by respecting base directory restrictions

    Always use the most appropriate reading strategy to optimize token usage.
    When editing, prefer unified diff format for complex changes.""",
    tools=file_tools
)

# Use the agent
result = await file_agent.run(
    prompt="Read the README.md file and summarize its contents",
    context={"working_dir": Path.cwd()}
)
```

!!! tip "API Keys"
    If you're using API-based models (like OpenRouter, OpenAI, etc.), ensure the appropriate API key environment variable is set (e.g., `OPENROUTER_API_KEY`, `OPENAI_API_KEY`).

---

## 🔧 Configuration

### Default Configuration

```python
from pathlib import Path
from marsys.environment import create_file_operation_tools, FileOperationConfig

# Use defaults (permissive mode)
file_tools = create_file_operation_tools()
```

### Custom Configuration

```python
from pathlib import Path
from marsys.environment import FileOperationConfig, create_file_operation_tools

# Create custom configuration
config = FileOperationConfig(
    # Base directory enforcement
    base_directory=Path("/home/user/projects"),
    force_base_directory=True,  # Require all operations within base_directory

    # File size limits (hard limit for safety)
    max_file_size_bytes=100 * 1024 * 1024,  # 100 MB absolute limit

    # Character-based reading thresholds (token proxy for text)
    small_file_threshold=10000,      # < 10k chars (~2.5k tokens): FULL read
    medium_file_threshold=100000,    # 10-100k chars (~25k tokens): PARTIAL read
    large_file_threshold=500000,     # > 500k chars (~125k tokens): OVERVIEW first

    # File type-specific limits
    max_json_content_chars=40000,     # JSON truncation threshold (~10k tokens)
    max_lines_per_read=250,           # Max lines for text files
    max_pages_per_read=5,             # Max pages for PDF files

    # Absolute safety limit (applies to ALL file types)
    max_characters_absolute=120000,   # Hard limit: 120K chars (~30k tokens)

    # Image token limits (for vision models)
    max_image_pixels=1024 * 1024,    # 1 megapixel (1024x1024)
    max_images_per_read=4,           # Maximum images per operation

    # Security patterns (glob-style)
    blocked_patterns=[
        "*.key", "*.pem", "*.p12",  # Private keys
        ".env", ".env.*",             # Environment files
        "*.sqlite", "*.db",           # Databases
        ".git/**",                    # Git internals
    ],

    # Auto-approve patterns (no user confirmation needed)
    auto_approve_patterns=[
        "*.md", "*.txt",              # Documentation
        "*.py", "*.js", "*.java",     # Source code
        "*.json", "*.yaml", "*.yml",  # Configuration
    ],

    # Require approval patterns
    require_approval_patterns=[
        "*.sh", "*.bash",             # Shell scripts
        "Makefile", "Dockerfile",     # Build files
        "*.sql",                      # SQL files
    ],

    # Editing
    enable_editing=True,
    enable_dry_run=True,  # Allow preview before applying edits

    # Search
    enable_content_search=True,
    enable_filename_search=True,
    enable_structure_search=True,
    max_search_results=100,

    # Audit logging
    enable_audit_log=True,
    audit_log_path=Path("./file_operations_audit.log"),
)

# Create tools with custom config
file_tools = create_file_operation_tools(config)
```

### Character Limit Configuration

The toolkit uses different character limits for different purposes:

#### 1. **File Type-Specific Limits**

These limits control how different file types are read and processed:

- **`max_json_content_chars`** (default: 40,000 characters):
  - **Applies to**: JSON files only
  - **Behavior**: Triggers truncation/overview when JSON exceeds this limit
  - **What happens**: Dictionary values truncated to 200 chars, arrays show first 20 items
  - **Use case**: Prevent huge JSON files from consuming entire context window

- **`max_lines_per_read`** (default: 250 lines):
  - **Applies to**: Text files (.txt, .py, .md, .yaml, etc.)
  - **Behavior**: Controls line-based partial reading
  - **What happens**: Used with `start_line`/`end_line` parameters
  - **Use case**: Read specific sections of code or text files

- **`max_pages_per_read`** (default: 5 pages):
  - **Applies to**: PDF files
  - **Behavior**: Controls page-based partial reading
  - **What happens**: Used with `start_page`/`end_page` parameters
  - **Use case**: Read specific sections of PDFs incrementally

#### 2. **Absolute Safety Limit**

- **`max_characters_absolute`** (default: 120,000 characters):
  - **Applies to**: ALL file types (text, PDF, JSON, code)
  - **Behavior**: Hard limit that raises an error when exceeded
  - **What happens**: Returns detailed error telling agent to request fewer lines/pages
  - **Use case**: Prevent context window overflow from excessively large reads
  - **⚠️ WARNING**: Setting this too high (>120K) can cause:
    - Context window overflow (most models: 128K-200K tokens total)
    - Out of memory errors
    - API timeout failures
    - Workflow crashes

**Recommended values**:
```python
config = FileOperationConfig(
    # JSON-specific truncation
    max_json_content_chars=40000,      # ~10K tokens for JSON

    # Text file line-based limiting
    max_lines_per_read=250,            # ~10-20K chars typically

    # PDF page-based limiting
    max_pages_per_read=5,              # ~2.5K chars/page typical

    # Hard safety limit for ALL files
    max_characters_absolute=120000,    # ~30K tokens maximum
)
```

!!! warning "Context Window Management"
    The absolute character limit (120K) leaves room for:
    - System prompts (~2-5K tokens)
    - Agent memory/history (~10-20K tokens)
    - Images (if any) (~500-2000 tokens per image)
    - Response generation (~2-10K tokens)

    Total context budget: ~128K-200K tokens for most modern models

### Preset Configurations

```python
from marsys.environment import FileOperationConfig, create_file_operation_tools

# Permissive mode (default)
permissive_config = FileOperationConfig.create_permissive()
permissive_tools = create_file_operation_tools(permissive_config)

# Restrictive mode (tighter security)
restrictive_config = FileOperationConfig.create_restrictive()
restrictive_tools = create_file_operation_tools(restrictive_config)
```

---

## 📖 Reading Strategies

The toolkit provides five intelligent reading strategies to optimize token usage:

### AUTO Strategy (Default)

Automatically selects the best strategy based on file size:

```python
# Agent automatically uses AUTO strategy
result = await file_agent.run(
    prompt="Read data.json",
    context={}
)
```

**Selection Logic (based on character count for text files):**
- **< 10k characters (~2.5k tokens)**: FULL read (complete content)
- **10-100k characters (~2.5-25k tokens)**: PARTIAL read (sections with overview)
- **100-500k characters (~25-125k tokens)**: PROGRESSIVE (structure first, drill down)
- **> 500k characters (~125k+ tokens)**: OVERVIEW (structure + summary only)

### FULL Strategy

Read complete file contents:

```python
from marsys.environment.file_operations import ReadStrategy

result = await file_agent.run(
    prompt="Read config.yaml using FULL strategy",
    context={"read_strategy": ReadStrategy.FULL}
)
```

**Best For:**
- Small configuration files
- Complete data processing needed
- Files under 10 KB

### PARTIAL Strategy

Read with structure overview + selected sections:

```python
result = await file_agent.run(
    prompt="Read the 'Installation' section from README.md using PARTIAL strategy",
    context={"read_strategy": ReadStrategy.PARTIAL}
)
```

**Best For:**
- Medium-sized documents
- When specific sections needed
- Files 10-100 KB

### OVERVIEW Strategy

Extract structure and summary only:

```python
result = await file_agent.run(
    prompt="Get overview of large_report.pdf",
    context={"read_strategy": ReadStrategy.OVERVIEW}
)
```

**Best For:**
- Large documents
- Initial exploration
- Files > 100 KB

**Returns:**
- Table of contents
- Section headings
- Document summary
- Metadata

### PROGRESSIVE Strategy

Load sections incrementally on demand:

```python
# First: Get structure
result1 = await file_agent.run(
    prompt="Get structure of codebase/main.py",
    context={"read_strategy": ReadStrategy.PROGRESSIVE}
)

# Then: Load specific sections
result2 = await file_agent.run(
    prompt="Read section 'class:DatabaseManager' from main.py",
    context={"section_id": "class:DatabaseManager"}
)
```

**Best For:**
- Very large files
- Code exploration
- Files > 500 KB

---

## 📑 Incremental Reading

For large documents, the toolkit provides incremental reading capabilities that allow agents to request specific page or line ranges.

### Reading Specific PDF Pages

```python
# Read pages 5-10 of a PDF
result = await file_agent.run(
    prompt="Read pages 5 to 10 from research_paper.pdf",
    context={
        "start_page": 5,
        "end_page": 10
    }
)
```

**Features:**
- Automatic limit enforcement (prevents requesting too many pages at once)
- Clean response without usage guides (agent already knows what was requested)
- Returns pure content from requested range

**Response Format:**
When you explicitly specify a page range, you get **just the content**:

```
JOURNALOFLATEXCLASSFILES,VOL.14,NO.8,AUGUST2021 5
[Content from pages 5-10]
JOURNALOFLATEXCLASSFILES,VOL.14,NO.8,AUGUST2021 10
```

**Note:** Usage guides are only shown when the system automatically returns partial content (file too large, no range specified). When you explicitly request pages 5-10, the system knows you're aware of what you're requesting.

### Reading Specific Text Lines

```python
# Read lines 100-200 from a code file
result = await file_agent.run(
    prompt="Read lines 100 to 200 from main.py",
    context={
        "start_line": 100,
        "end_line": 200
    }
)
```

**Features:**
- Character-based limit enforcement (prevents requesting too many lines)
- Maximum characters enforced by `max_characters_absolute` (120K default)
- Clean response without usage guides (explicit request)

**Response Format:**
```
Line 100
Line 101
Line 102
...
Line 200
```

Pure content from the requested line range, no headers or footers.

### Automatic Overflow Handling

When a file exceeds `max_characters_absolute` (120K default) and **no explicit range is specified**, the system raises an error telling the agent to request specific page/line ranges instead.

**For PDFs:**
Returns first N pages (default: 5, configurable via `max_pages_per_read`) with:

```
=== PARTIAL CONTENT ===
Document: 18 pages total
Showing: Pages 1-5
Maximum pages per request: 7
To read more: use read_file(path, start_page=X, end_page=Y) or search_files(query)
==================================================

[PDF content from pages 1-5]

==================================================
END OF PAGES 1-5 (of 18 total)
To continue: read_file(path, start_page=6, end_page=...)
==================================================
```

**For Text Files:**
1. Truncates content at character limit
2. Prepends usage guide showing total lines
3. Provides guidance on reading more with line ranges

**Important:** Usage guides are **only shown for automatic overflow**, not for explicit range requests.

### Request Validation

The toolkit validates all page/line range requests:

```python
# Request too many pages (e.g., 200 pages when limit is 100)
result = await file_agent.run(
    prompt="Read pages 1 to 200 from large_manual.pdf",
    context={"start_page": 1, "end_page": 200}
)

# Returns error response:
# {
#   "error": true,
#   "message": "Request exceeds maximum pages per read",
#   "details": {
#     "requested_pages": 200,
#     "maximum_pages": 100,
#     "suggestion": "Request fewer pages (e.g., start_page=1, end_page=100)"
#   }
# }
```

**Limits enforced:**
- **PDF pages**: `max_pages_per_read` (default: 5 pages)
- **Text lines**: `max_lines_per_read` (default: 250 lines)
- **All file types**: `max_characters_absolute` (default: 120K characters)

### Search Within Large Documents

For finding specific information in large files:

```python
# Search for keywords in PDF with page numbers
result = await file_agent.run(
    prompt="Search for 'machine learning' in research_paper.pdf",
    context={
        "search_type": "content",
        "pattern": "machine learning",
        "include_context": True
    }
)

# Result includes page numbers and line numbers:
# {
#   "matches": [
#     {
#       "match": "Machine learning algorithms...",
#       "location": "page 3, line 45",
#       "page": 3,
#       "line": 45,
#       "context_before": [...],
#       "context_after": [...]
#     }
#   ]
# }
```

**Benefits:**
- Quickly locate information without reading entire document
- Page-level navigation for PDFs
- Context lines show surrounding content

---

## 🏗️ Structure Extraction

The toolkit extracts hierarchical structure from various file types:

### PDF Structure

Uses font-size analysis to detect headings:

```python
result = await file_agent.run(
    prompt="Extract structure from research_paper.pdf",
    context={}
)

# Returns DocumentStructure with sections like:
# Section(id="1", title="Introduction", level=1, ...)
#   ├── Section(id="1.1", title="Background", level=2, ...)
#   └── Section(id="1.2", title="Motivation", level=2, ...)
```

### Code Structure (Future)

AST-based parsing with tree-sitter:

```python
result = await file_agent.run(
    prompt="Show me the class structure of models.py",
    context={}
)

# Returns hierarchy like:
# Section(id="module", title="models.py", ...)
#   ├── Section(id="class:User", title="class User", ...)
#   │   ├── Section(id="method:__init__", ...)
#   │   └── Section(id="method:validate", ...)
#   └── Section(id="class:Database", ...)
```

### Accessing Sections

```python
# Read specific section by ID
result = await file_agent.run(
    prompt="Read section '1.2' from research_paper.pdf",
    context={"section_id": "1.2"}
)
```

---

## ✏️ Editing Files

### Unified Diff Format (Recommended)

High-success-rate editing using unified diff format:

```python
result = await file_agent.run(
    prompt="""Edit config.py using unified diff format:

    --- config.py
    +++ config.py
    @@ -10,3 +10,3 @@
     DEBUG = True
    -MAX_WORKERS = 4
    +MAX_WORKERS = 8
     LOG_LEVEL = "INFO"
    """,
    context={}
)
```

**Features:**
- Multiple fallback strategies (exact match → whitespace normalization → fuzzy matching)
- Dry-run preview before applying
- Detailed change reports
- ~98% success rate (targeting Aider-like performance)

### Search and Replace

Simple text replacement:

```python
result = await file_agent.run(
    prompt="""Replace 'old_function()' with 'new_function()' in utils.py""",
    context={}
)
```

### Dry Run Preview

Preview changes before applying:

```python
result = await file_agent.run(
    prompt="""Show me what would change if I apply this diff (dry run):

    --- app.py
    +++ app.py
    @@ -5,1 +5,1 @@
    -version = "1.0.0"
    +version = "1.1.0"
    """,
    context={"dry_run": True}
)
```

---

## 🖼️ Image Support & Token Estimation

The toolkit provides comprehensive image support with provider-specific token estimation for vision-language models.

### Reading Images Directly

```python
# Read an image file with token estimation
result = await file_agent.run(
    prompt="Read the diagram.png image",
    context={
        "provider": "anthropic",  # Options: openai, anthropic, google, xai, generic
        "detail": "high",          # Options: high, low (affects some providers)
        "max_pixels": 1024 * 1024  # Downsample if exceeds this limit
    }
)

# Result includes:
# - Image dimensions and format
# - Estimated token count for the provider
# - Base64-encoded image data (for sending to VLM)
# - Metadata (DPI, color mode, etc.)
```

### Token Estimation by Provider

Different vision-language model providers use different tokenization strategies:

**OpenAI (GPT-4V, GPT-4o):**
- Divides images into 512x512 pixel tiles
- Formula: `85 + (170 * num_tiles)`
- Example: 1024x1024 image = 85 + (170 × 4) = **765 tokens**

**Anthropic (Claude):**
- Formula: `(width * height) / 750`
- Scales down images > 1568px on long edge
- Example: 1024x1024 image = (1024 × 1024) / 750 = **1398 tokens**

**Google (Gemini):**
- Small images (≤384px both dimensions): **258 tokens**
- Larger images: 768x768 pixel tiles, 258 tokens/tile
- Example: 1024x1024 image = 2 × 2 tiles = **1032 tokens**

**xAI (Grok):**
- Divides into 448x448 pixel tiles
- Formula: `(num_tiles + 1) * 256`
- Example: 1024x1024 image = (6 + 1) × 256 = **1792 tokens**

### Image Configuration

```python
config = FileOperationConfig(
    # Image pixel limits
    max_image_pixels=1024 * 1024,  # 1 megapixel max per image
    max_images_per_read=4,          # Max images in single operation

    # Auto-downsample if needed
    # Images exceeding max_pixels will be resized maintaining aspect ratio
)
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- ICO (.ico)
- SVG (.svg)

### Image Extraction from PDFs

When reading PDFs, the toolkit can extract embedded images:

```python
# Read PDF with images (future feature)
result = await file_agent.run(
    prompt="Read report.pdf and include any charts or diagrams",
    context={
        "extract_images": True,
        "provider": "openai",  # For token estimation
        "max_images": 4        # Limit images per PDF
    }
)

# Result includes both text and ImageData objects
print(f"Text tokens: {result.estimated_tokens}")
print(f"Image tokens: {result.total_estimated_image_tokens}")
print(f"Total tokens: {result.get_total_estimated_tokens()}")
```

### Token Budget Management

```python
from marsys.environment.file_operations.token_estimation import (
    estimate_total_tokens,
    should_downsample_image
)

# Estimate total tokens before reading
estimation = estimate_total_tokens(
    text_content="Sample text...",
    images=[(1920, 1080), (1024, 768)],  # Image dimensions
    provider="anthropic"
)
print(f"Total estimated tokens: {estimation['total_tokens']}")
print(f"  Text: {estimation['text_tokens']} tokens")
print(f"  Images: {estimation['image_tokens']} tokens")

# Check if image needs downsampling
needs_downsample, target_dims = should_downsample_image(
    width=2048,
    height=2048,
    max_pixels=1024 * 1024,
    max_tokens=1000,
    provider="openai"
)
if needs_downsample:
    print(f"Downsample to: {target_dims}")
```

---

## 🔍 Search Capabilities

### Content Search (Grep)

Search file contents with regex patterns:

```python
result = await file_agent.run(
    prompt="Search for 'TODO' comments in all Python files",
    context={
        "search_type": "content",
        "pattern": r"#\s*TODO:",
        "file_pattern": "*.py"
    }
)
```

**Options:**
- Case-sensitive/insensitive search
- Regex patterns
- Context lines (before/after matches)
- File type filtering
- Pagination for large results

### PDF Content Search with Page Numbers

Search within PDF files with page-level location tracking:

```python
result = await file_agent.run(
    prompt="Search for 'neural network' in all PDF files",
    context={
        "search_type": "content",
        "pattern": "neural network",
        "file_pattern": "*.pdf",
        "include_context": True,
        "context_lines": 2
    }
)

# Returns matches with page numbers:
# {
#   "matches": [
#     {
#       "file": "paper.pdf",
#       "match": "Neural networks have revolutionized...",
#       "location": "page 5, line 23",
#       "page": 5,
#       "line": 23,
#       "context_before": ["...", "..."],
#       "context_after": ["...", "..."]
#     }
#   ]
# }
```

**Features:**
- Automatic page number tracking
- Line numbers within each page
- Context lines around matches
- Location string for easy reference ("page X, line Y")

### Filename Search (Glob)

Find files by name patterns:

```python
result = await file_agent.run(
    prompt="Find all test files",
    context={
        "search_type": "filename",
        "pattern": "**/test_*.py"
    }
)
```

**Supports:**
- Glob patterns (`*.py`, `**/*.md`, `test_*`)
- Recursive search
- File metadata (size, modified time, etc.)

### Structure Search

Search within document structures:

```python
result = await file_agent.run(
    prompt="Find all class definitions in the codebase",
    context={
        "search_type": "structure",
        "query": "class:*"
    }
)
```

---

## 🔒 Security Features

### Base Directory Enforcement

Restrict operations to a specific directory tree:

```python
from pathlib import Path
from marsys.environment import FileOperationConfig, create_file_operation_tools

config = FileOperationConfig(
    base_directory=Path("/home/user/safe_workspace"),
    force_base_directory=True  # Reject operations outside this directory
)

file_tools = create_file_operation_tools(config)

# Attempts to access /etc/passwd will be blocked
```

### Pattern-Based Permissions

Control access with glob patterns:

```python
config = FileOperationConfig(
    # Block these patterns entirely
    blocked_patterns=[
        "*.key", "*.pem",  # Private keys
        ".env*",           # Environment files
        ".git/**",         # Git internals
    ],

    # Auto-approve these patterns (no confirmation)
    auto_approve_patterns=[
        "*.md", "*.txt",   # Safe documents
        "*.json",          # Configuration
    ],

    # Require user approval for these
    require_approval_patterns=[
        "*.sh",            # Shell scripts
        "*.sql",           # Database queries
    ]
)
```

### File Size Limits

Prevent memory issues:

```python
config = FileOperationConfig(
    max_file_size_bytes=10 * 1024 * 1024,  # 10 MB limit
    max_tokens_per_read=8000  # Token limit per read
)
```

### Audit Logging

Track all file operations:

```python
config = FileOperationConfig(
    enable_audit_log=True,
    audit_log_path=Path("./file_ops_audit.log")
)

# All operations logged with:
# - Timestamp
# - Operation type
# - File path
# - Success/failure
# - Agent name
```

---

## 📚 Available Tools

When you call `create_file_operation_tools()`, you get these tools:

| Tool | Description | Parameters | Example Usage |
|------|-------------|------------|---------------|
| `read_file` | Read file with intelligent strategy | `path`, `strategy`, `start_page`, `end_page`, `start_line`, `end_line` | "Read pages 5-10 from paper.pdf" |
| `write_file` | Write content to file | `path`, `content` | "Write results to output.txt" |
| `edit_file` | Edit using unified diff or search/replace | `path`, `changes`, `edit_format`, `dry_run` | "Change version to 2.0.0 in setup.py" |
| `search_files` | Search content, filenames, or structure | `query`, `search_type`, `path`, `include_context`, `context_lines` | "Search for 'TODO' in all .py files" |
| `get_file_structure` | Extract hierarchical structure | `path` | "Show structure of report.pdf" |
| `read_file_section` | Read specific section by ID | `path`, `section_id` | "Read section 3.2 from report.pdf" |
| `list_files` | List directory contents | `path`, `pattern` | "List all Python files in src/" |
| `create_directory` | Create directories | `path` | "Create directory for logs" |
| `delete_file` | Delete files (with approval) | `path` | "Delete temp files" |

### read_file Parameters

- **path** (required): File path to read
- **strategy** (optional): Reading strategy (`auto`, `full`, `partial`, `overview`, `progressive`)
- **start_page** (optional): First page to read (PDFs only)
- **end_page** (optional): Last page to read (PDFs only)
- **start_line** (optional): First line to read (text files)
- **end_line** (optional): Last line to read (text files)

**Validation:**
- Page range requests validated against `max_pages_per_read` limit
- Line range requests validated against `max_lines_per_read` limit
- All requests validated against `max_characters_absolute` (120K) hard limit
- Returns error with suggestion if request exceeds limits

### search_files Parameters

- **query** (required): Search pattern or query
- **search_type** (optional): `content` (default), `filename`, or `structure`
- **path** (optional): Specific file or directory to search
- **include_context** (optional): Include surrounding lines (default: false)
- **context_lines** (optional): Number of context lines (default: 2)

**PDF Search Features:**
- Automatically tracks page numbers
- Returns location as "page X, line Y"
- Includes context lines from the same page

---

## 💡 Use Cases

### Use Case 1: Analyzing Large Documents with Incremental Reading

```python
# Agent extracts key information from large PDF using incremental reading
result = await file_agent.run(
    prompt="""Analyze research_paper.pdf (18 pages):
    1. First, read the file without specifying pages to get overview
    2. The system will return first 5 pages with guidance
    3. Search for 'methodology' to find the relevant section
    4. Read specific pages containing methodology
    5. Summarize key findings
    """,
    context={}
)

# Example of what happens:
# Step 1: Agent calls read_file("paper.pdf")
# System returns: Pages 1-5 WITH usage guide header/footer
#   Content starts with: "=== PARTIAL CONTENT ===\nDocument: 18 pages total..."
#
# Step 2: Agent calls search_files("methodology", path="paper.pdf")
# System returns: "Found at page 8, line 15"
#
# Step 3: Agent calls read_file("paper.pdf", start_page=7, end_page=10)
# System returns: Pages 7-10 WITHOUT usage guide (clean content only)
#   Content starts with: "JOURNALOFLATEXCLASSFILES,VOL.14,NO.8,AUGUST2021 7..."
```

### Use Case 2: Code Refactoring

```python
# Agent performs safe code changes
result = await file_agent.run(
    prompt="""Refactor database.py:
    1. Search for all deprecated function calls
    2. Preview changes with dry-run
    3. Apply unified diff to update functions
    4. Verify changes were successful
    """,
    context={"dry_run": True}
)
```

### Use Case 3: Multi-File Search and Update

```python
# Agent searches and updates across codebase
result = await file_agent.run(
    prompt="""Update version numbers:
    1. Search for 'version = ' in all Python files
    2. Update to version 2.0.0
    3. List all modified files
    """,
    context={}
)
```

### Use Case 4: Secure File Management

```python
# Agent with restricted permissions
from pathlib import Path
from marsys.environment import FileOperationConfig, create_file_operation_tools

config = FileOperationConfig.create_restrictive()
config.base_directory = Path("/workspace/project")
config.force_base_directory = True

secure_tools = create_file_operation_tools(config)

secure_agent = Agent(
    model=model_config,
    name="SecureFileAgent",
    goal="Manage project files securely",
    instruction="Only access files within /workspace/project. Never access system files.",
    tools=secure_tools
)
```

### Use Case 5: Image Analysis with Token Management

```python
# Agent analyzes images with token budget awareness
result = await file_agent.run(
    prompt="""Analyze the architecture diagram:
    1. Read architecture_diagram.png
    2. Identify main components
    3. Estimate token cost
    4. If needed, downsample to fit budget
    """,
    context={
        "provider": "anthropic",      # Claude for analysis
        "max_pixels": 1024 * 1024,    # 1MP budget
        "detail": "high"               # High detail analysis
    }
)

# Result includes token estimation
print(f"Image tokens used: {result.total_estimated_image_tokens}")
```

### Use Case 6: Multi-Modal Document Processing

```python
# Agent processes documents with text and images
result = await file_agent.run(
    prompt="""Process the quarterly report:
    1. Extract text content
    2. Find all embedded charts/graphs
    3. Analyze each chart
    4. Generate executive summary combining text and visual insights
    """,
    context={
        "extract_images": True,
        "provider": "openai",  # GPT-4V for vision
        "max_images": 4,       # Limit to key visuals
        "max_pixels": 512 * 512  # Reasonable size per image
    }
)
```

---

## 🚨 Common Issues

### Issue: "PyMuPDF not available"

**Solution:**
```bash
pip install --upgrade marsys  # PyMuPDF is included in core
# Or install directly:
pip install PyMuPDF
```

PyMuPDF is included in core marsys installation for PDF text/image extraction and layout analysis.

### Issue: "Pillow (PIL) not available"

**Solution:**
```bash
pip install --upgrade marsys  # Pillow is included in core
# Or install directly:
pip install Pillow
```

Pillow is included in core marsys installation for:
- Direct image file reading (.jpg, .png, .webp, etc.)
- Image extraction from PDFs
- Image token estimation and processing

### Issue: "Path outside base directory"

**Solution:**
```python
# Either expand base directory
config = FileOperationConfig(
    base_directory=Path("/broader/path")
)

# Or disable enforcement
config = FileOperationConfig(
    force_base_directory=False
)
```

### Issue: "File too large to read"

**Solution:**
```python
# Use OVERVIEW or PROGRESSIVE strategy
result = await file_agent.run(
    prompt="Get overview of large_file.pdf",
    context={"read_strategy": ReadStrategy.OVERVIEW}
)

# Or increase limits (use cautiously)
config = FileOperationConfig(
    max_file_size_bytes=100 * 1024 * 1024,  # 100 MB hard limit
    max_characters_absolute=150000,          # Increase absolute limit (use cautiously!)
    max_pages_per_read=10,                   # More pages per request
    max_lines_per_read=500,                  # More lines per request
    large_file_threshold=1000000             # 1M chars threshold
)
```

### Issue: "Request exceeds maximum pages per read"

When an agent requests too many pages at once:

**Error:**
```json
{
  "error": true,
  "message": "Request exceeds maximum pages per read",
  "details": {
    "requested_pages": 50,
    "maximum_pages": 7,
    "suggestion": "Request fewer pages (e.g., start_page=1, end_page=7)"
  }
}
```

**Solutions:**
```python
# Option 1: Request fewer pages (recommended)
result = await file_agent.run(
    prompt="Read pages 1-7 instead of 1-50",
    context={"start_page": 1, "end_page": 7}
)

# Option 2: Increase page limit via config
config = FileOperationConfig(
    max_pages_per_read=10,  # Increase pages per request
    max_characters_absolute=150000,  # May also need to increase absolute limit
)

# Option 3: Read in batches
for start in range(1, 50, 7):
    result = await file_agent.run(
        prompt=f"Read pages {start} to {start+6}",
        context={"start_page": start, "end_page": min(start+6, 50)}
    )
```

### Issue: "Request exceeds maximum lines per read"

Similar to pages, but for text files:

**Error:**
```json
{
  "error": true,
  "message": "Request exceeds maximum lines per read",
  "details": {
    "requested_lines": 1000,
    "maximum_lines": 625,
    "suggestion": "Request fewer lines (e.g., start_line=1, end_line=625)"
  }
}
```

**Solution:**
Read in batches or increase `max_lines_per_read` and `max_characters_absolute` configuration (use cautiously).

### Issue: "Image exceeds token budget"

**Solution:**
```python
# Images are automatically downsampled if they exceed max_pixels
config = FileOperationConfig(
    max_image_pixels=2 * 1024 * 1024,  # 2 megapixels
    max_images_per_read=6               # More images allowed
)

# Or specify max_pixels per operation
result = await file_agent.run(
    prompt="Read large_image.png",
    context={"max_pixels": 2 * 1024 * 1024}  # Will downsample if needed
)
```

### Issue: "Edit failed to apply"

**Solution:**
```python
# Use dry-run to preview
result = await file_agent.run(
    prompt="Apply diff with dry-run first",
    context={"dry_run": True}
)

# Check the preview before applying
# If still fails, use search-replace format instead
```

---

## 📋 Best Practices

### 1. **Use Appropriate Reading Strategies**

```python
# ✅ GOOD - Let AUTO choose for you
result = await file_agent.run("Read document.pdf", context={})

# ✅ GOOD - Use incremental reading for large files
result = await file_agent.run(
    "Read pages 10-15 from large_report.pdf",
    context={"start_page": 10, "end_page": 15}
)

# ✅ GOOD - Use search to find relevant sections first
result = await file_agent.run(
    "Search for 'conclusions' in report.pdf, then read those pages",
    context={}
)

# ❌ AVOID - Requesting entire large file at once
result = await file_agent.run(
    "Read all 100 pages of manual.pdf",  # Will hit limits or return partial
    context={}
)
```

### 2. **Leverage Search for Large Documents**

```python
# ✅ GOOD - Search before reading
result = await file_agent.run(
    prompt="""Find sections about 'machine learning' in research.pdf,
    then read the relevant pages in detail""",
    context={}
)

# ❌ LESS EFFICIENT - Reading everything hoping to find it
result = await file_agent.run(
    prompt="Read entire 200-page thesis and find ML sections",
    context={}
)
```

### 3. **Always Use Dry Run for Critical Edits**

```python
# ✅ GOOD - Preview before applying
result = await file_agent.run(
    "Update production config (dry run first)",
    context={"dry_run": True}
)

# ❌ RISKY - Direct edit without preview
result = await file_agent.run(
    "Update production config",  # No preview
    context={}
)
```

### 4. **Restrict Base Directory for Security**

```python
# ✅ GOOD - Enforce base directory
config = FileOperationConfig(
    base_directory=Path("/workspace"),
    force_base_directory=True
)

# ❌ RISKY - No restrictions
config = FileOperationConfig(
    force_base_directory=False  # Agent can access any file
)
```

### 5. **Use Pattern-Based Permissions**

```python
# ✅ GOOD - Block sensitive files
config = FileOperationConfig(
    blocked_patterns=["*.key", "*.pem", ".env*"]
)

# ❌ MISSING - No protection for secrets
config = FileOperationConfig(
    blocked_patterns=[]  # No restrictions
)
```

### 6. **Enable Audit Logging for Production**

```python
# ✅ GOOD - Track all operations
config = FileOperationConfig(
    enable_audit_log=True,
    audit_log_path=Path("./audit.log")
)

# ❌ MISSING - No audit trail
config = FileOperationConfig(
    enable_audit_log=False
)
```

### 7. **Handle Errors Gracefully**

```python
# ✅ GOOD - Provide fallback instructions
file_agent = Agent(
    model=model_config,
    name="FileAssistant",
    goal="Manage files",
    instruction="""Handle errors gracefully:
    - If file not found, suggest similar filenames
    - If access denied, explain permissions needed
    - If edit fails, try search-replace format
    - Always report errors clearly to user
    """,
    tools=file_tools
)
```

---

## 🎯 Advanced Examples

### Multi-Agent File Processing Pipeline

```python
from marsys import Agent
from marsys.models import ModelConfig
from marsys.environment import create_file_operation_tools
from marsys.coordination import Orchestra
from marsys.coordination.topology.patterns import PatternConfig
import os

# Model configuration (adjust based on your provider)
model_config = ModelConfig(
    model_type="api",
    model_name="anthropic/claude-haiku-4.5",
    provider="openrouter",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

file_tools = create_file_operation_tools()

# Agent 1: File Scanner
scanner = Agent(
    model=model_config,
    name="Scanner",
    goal="Scan directories and identify files for processing",
    instruction="Use list_files and search_files to find target files",
    tools=file_tools
)

# Agent 2: Content Analyzer
analyzer = Agent(
    model=model_config,
    name="Analyzer",
    goal="Analyze file contents and extract insights",
    instruction="Read files intelligently using appropriate strategies",
    tools=file_tools
)

# Agent 3: Report Generator
reporter = Agent(
    model=model_config,
    name="Reporter",
    goal="Generate summary reports from analysis",
    instruction="Create structured reports and write to output files",
    tools=file_tools
)

# Create pipeline topology
topology = PatternConfig.pipeline(
    stages=[
        {"name": "scan", "agents": ["Scanner"]},
        {"name": "analyze", "agents": ["Analyzer"]},
        {"name": "report", "agents": ["Reporter"]}
    ],
    parallel_within_stage=False
)

# Execute pipeline
result = await Orchestra.run(
    task="Analyze all Python files in src/ and generate report",
    topology=topology
)
```

### Progressive Document Reading

```python
# Agent uses progressive strategy for large documents
result = await file_agent.run(
    prompt="""Read technical_manual.pdf progressively:
    1. Get document structure first
    2. Identify the 'Installation' section
    3. Read only the Installation section in detail
    4. Summarize installation steps
    """,
    context={"read_strategy": ReadStrategy.PROGRESSIVE}
)
```

---

## 🎯 Next Steps

- [Tool API Reference](../api/tools.md) - Complete API documentation
- [Custom Tools](../concepts/tools.md) - Create your own file handlers
- [Agent Development](../getting-started/first-agent.md) - Build agents with file capabilities
- [Security Best Practices](../concepts/error-handling.md) - Secure file operations

---

!!! warning "Security"
    Always configure `blocked_patterns` to prevent access to sensitive files like private keys, environment variables, and credentials.

!!! info "Core Dependencies"
    - **PDF support**: Provided by `PyMuPDF` (included in core marsys installation) for PDF structure extraction, text reading, and image extraction
    - **Image support**: Provided by `Pillow` (included in core marsys installation) for image reading and processing
    - Both are automatically included for full VLM capabilities

!!! note "Token Management"
    - The toolkit uses **character count** (not file size) as a proxy for text tokens
    - Image tokens are estimated using **provider-specific formulas** (OpenAI, Anthropic, Google, xAI)
    - AUTO strategy intelligently selects reading approach based on character count
    - Images are automatically downsampled if they exceed pixel/token budgets

!!! tip "Incremental Reading (New in v0.2)"
    **Efficient Large Document Handling:**
    - **PDF pages**: Use `start_page`/`end_page` parameters to read specific page ranges
    - **Text lines**: Use `start_line`/`end_line` parameters for line-based reading
    - **Clean responses**: Explicit range requests return pure content without usage guides
    - **Automatic limits**: System enforces limits and returns helpful error messages
    - **PDF search**: Returns page numbers with each match ("page 5, line 23")
    - **Validation**: All requests validated against:
      - `max_pages_per_read` (default: 5 pages)
      - `max_lines_per_read` (default: 250 lines)
      - `max_characters_absolute` (default: 120K characters)

    **Response Types:**
    - **Explicit range** (`start_page=5, end_page=10`): Pure content only
    - **File too large** (exceeds 120K chars): Error with suggestion to use ranges
    - **Excessive range** (exceeds limit): Error with suggestion to request fewer lines/pages

    **Recommended workflow for large documents:**
    1. Try reading the file (if > 120K chars, error will suggest using ranges)
    2. Use search to find relevant sections (returns page numbers)
    3. Request specific page/line ranges based on search results
    4. Continue reading in batches if needed

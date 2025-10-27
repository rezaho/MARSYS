"""
MARSYS (Multi-Agent Reasoning Systems) - Setup Configuration

A powerful framework for orchestrating collaborative AI agents with
sophisticated reasoning, planning, and autonomous capabilities.

Author: Reza Hosseini
License: MIT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies (DEFAULT installation - comprehensive framework)
core_deps = [
    # Framework core
    "pydantic>=2.11.9",
    "requests>=2.32.3",
    "psutil>=7.1.0",
    "packaging>=25.0",
    "aiohttp>=3.12.15",
    "pyyaml>=6.0.2",
    # Browser automation
    "playwright>=1.55.0",
    "pillow>=12.0.0",  # Updated for file operations image handling
    "beautifulsoup4>=4.14.2",
    "lxml>=6.0.2",  # Fast HTML parsing for search tools
    "brotli>=1.1.0",  # Compression support for HTTP requests
    "markdownify>=1.2.0",
    # UI/Terminal
    "rich>=14.1.0",
    "textual>=6.2.0",
    "python-dateutil>=2.9.0",
    # Tools
    "pypdf>=3.0.0",  # Compatible with PyPDF2 API
    "googlesearch-python>=1.3.0",
    # File operations (moved from optional to core)
    "PyMuPDF>=1.26.0",     # PDF text/image extraction and layout analysis
    "chardet>=5.2.0",      # Character encoding detection
    # Logging
    "structlog>=25.4.0",
    "python-json-logger>=2.0.7",  # v2.x (v3 requires testing)
    # Validation
    "jsonschema>=4.23.0",
]

# Local models dependencies
local_models_deps = [
    "torch>=2.6.0",
    "torchvision>=0.23.0",
    "transformers>=4.54.1",
    "accelerate>=1.5.2",
    "peft>=0.17.1",
    "trl>=0.22.1",
    "datasets>=4.1.1",
    "decord>=0.6.0",
    "qwen-vl-utils>=0.0.8",
]

# Production inference dependencies
production_deps = [
    "vllm>=0.10.2",
    "flash-attn>=2.7.4.post1",
    "triton>=3.4.0",
    "ninja>=1.13.0",
]

# Development dependencies
dev_deps = [
    # Testing
    "pytest>=8.4.1",
    "pytest-asyncio>=1.0.0",
    "pytest-mock>=3.14.1",
    "pytest-cov>=6.2.1",
    # Code quality
    "black>=25.0.0",
    "flake8>=7.1.0",
    "mypy>=1.13.0",
    # Documentation
    "mkdocs>=1.6.1",
    "mkdocs-material>=9.6.21",
    "mkdocstrings>=0.30.1",
    "mkdocstrings-python>=1.18.2",
]

setup(
    name="marsys",
    version="0.1.0",

    # Author information
    author="Reza Hosseini",
    author_email="reza.hosseini@marsys.io",
    maintainer="Reza Hosseini",
    maintainer_email="reza.hosseini@marsys.io",

    # Package description
    description="A powerful framework for orchestrating collaborative AI agents with sophisticated reasoning, planning, and autonomous capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # URLs
    url="https://github.com/rezaho/MARSYS",
    project_urls={
        "Documentation": "https://marsys.io",
        "Source Code": "https://github.com/rezaho/MARSYS",
        "Bug Tracker": "https://github.com/rezaho/MARSYS/issues",
        "Changelog": "https://github.com/rezaho/MARSYS/blob/main/CHANGELOG.md",
        "Discussions": "https://github.com/rezaho/MARSYS/discussions",
    },

    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Python version requirement
    python_requires=">=3.12",

    # Dependencies
    install_requires=core_deps,  # DEFAULT installation gets full framework

    # Optional dependencies (extras)
    extras_require={
        # Core alias (same as default)
        "core": core_deps,

        # Local models: DEFAULT + PyTorch + Transformers + Learning + Qwen-VL
        "local-models": core_deps + local_models_deps,

        # Production inference: vLLM + Flash Attention
        "production": production_deps,

        # Development: ALL + testing + docs
        "dev": core_deps + local_models_deps + production_deps + dev_deps,
    },

    # PyPI classifiers
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",

        # Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",

        # License
        "License :: OSI Approved :: Apache Software License",

        # OS
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",

        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",

        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries :: Application Frameworks",

        # Framework
        "Framework :: AsyncIO",

        # Natural Language
        "Natural Language :: English",
    ],

    # Keywords for PyPI search
    keywords=[
        "ai", "agents", "multi-agent", "llm", "orchestration",
        "coordination", "reasoning", "planning", "autonomous",
        "openai", "anthropic", "claude", "gpt", "gemini",
        "multi-agent-systems", "agent-framework", "ai-agents",
        "workflow", "automation", "collaboration"
    ],

    # License
    license="Apache-2.0",

    # Package data
    include_package_data=True,
    zip_safe=False,

    # Entry points (if CLI tools are added in future)
    # entry_points={
    #     "console_scripts": [
    #         "marsys=marsys.cli:main",
    #     ],
    # },
)

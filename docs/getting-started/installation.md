# Installation

Get MARSYS up and running on your system in just a few minutes.

## 📋 Prerequisites

- **Python 3.12+** (required)
- **pip** package manager
- **Git** for cloning the repository (optional)
- **API Keys** from at least one provider (OpenAI, Anthropic, Google)

## 🚀 Quick Install

### Recommended Setup with uv (10-100x faster than pip)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for MARSYS. It's significantly faster than pip and handles dependency resolution more reliably.

**Step 1: Install uv**

=== "Unix/macOS"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

**Step 2: Create virtual environment**

```bash
# Create virtual environment
uv venv

# Activate (Unix/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Alternative: Use uv run without explicit activation
uv run python your_script.py
```

**Step 3: Install MARSYS**

=== "Basic (Recommended)"
    Everything you need for most use cases:
    ```bash
    uv pip install marsys
    ```

    Includes: API models, browser automation, UI, tools, logging

=== "With Local Models"
    Add PyTorch and Transformers for local LLM/VLM support:
    ```bash
    uv pip install marsys[local-models]
    ```

    Includes: Everything in Basic + PyTorch, Transformers, TRL, Datasets

=== "Production"
    High-performance inference with vLLM:
    ```bash
    uv pip install marsys[production]
    ```

    Includes: vLLM, Flash Attention, Triton, Ninja

=== "Development"
    Complete setup for contributors:
    ```bash
    uv pip install marsys[dev]
    ```

    Includes: Everything + testing, linting, documentation tools

### Alternative Installation Methods

=== "Using pip"
    Standard Python package manager:
    ```bash
    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # Unix/macOS
    # .venv\Scripts\activate  # Windows

    # Install MARSYS
    pip install marsys
    ```

=== "From Source"
    For development or latest changes:
    ```bash
    git clone https://github.com/rezaho/MARSYS.git
    cd MARSYS
    pip install -e .[dev]
    ```

## 🔧 Detailed Installation

### 1. Set Up Virtual Environment (Recommended: uv)

!!! tip "Best Practice"
    Always use a virtual environment to avoid dependency conflicts. We recommend **uv** for faster installation.

=== "uv (Recommended)"
    ```bash
    # Create virtual environment
    uv venv

    # Activate (Unix/macOS)
    source .venv/bin/activate

    # Activate (Windows)
    .venv\Scripts\activate

    # Or skip activation and use uv run
    uv run python your_script.py
    ```

=== "venv (Standard)"
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Unix/macOS
    # .venv\Scripts\activate  # Windows
    ```

=== "conda"
    ```bash
    conda create -n marsys python=3.12
    conda activate marsys
    ```

### 2. Choose Your Installation

| Installation | Size | Time | Use Case |
|-------------|------|------|----------|
| `marsys` | ~200 MB | 1-2 min | API models + browser + tools ✨ |
| `marsys[local-models]` | ~2-3 GB | 5-10 min | + Local LLMs/VLMs |
| `marsys[production]` | ~1 GB | 3-5 min | + High-performance inference |
| `marsys[dev]` | ~3+ GB | 10-15 min | + Testing & docs tools |

Install with uv (recommended):
```bash
uv pip install marsys  # or marsys[local-models], etc.
```

Or with pip:
```bash
pip install marsys  # or marsys[local-models], etc.
```

### 3. Configure API Keys (Required)

!!! warning "Required Step"
    **You must configure API keys before running any examples.** MARSYS needs at least one API provider configured.

**Method 1: `.env` file (Recommended for development)**

Create a `.env` file in your project root:

```bash
# .env
# Required: At least one API key
OPENAI_API_KEY="sk-..."           # OpenAI GPT models
ANTHROPIC_API_KEY="sk-ant-..."    # Claude models
GOOGLE_API_KEY="AIza..."          # Gemini models
OPENROUTER_API_KEY="sk-or-..."    # OpenRouter (access to many models)

# Optional: Additional configurations
HEADLESS=true                      # Browser automation mode
LOG_LEVEL=INFO                     # Logging verbosity
```

MARSYS automatically loads `.env` files using `python-dotenv`.

**Method 2: Environment variables (Recommended for production)**

=== "Unix/macOS/Linux"
    ```bash
    export OPENAI_API_KEY="your-key-here"
    export ANTHROPIC_API_KEY="your-key-here"
    export GOOGLE_API_KEY="your-key-here"
    export OPENROUTER_API_KEY="your-key-here"
    ```

=== "Windows (Command Prompt)"
    ```cmd
    set OPENAI_API_KEY=your-key-here
    set ANTHROPIC_API_KEY=your-key-here
    set GOOGLE_API_KEY=your-key-here
    ```

=== "Windows (PowerShell)"
    ```powershell
    $env:OPENAI_API_KEY="your-key-here"
    $env:ANTHROPIC_API_KEY="your-key-here"
    $env:GOOGLE_API_KEY="your-key-here"
    ```

!!! warning "Security"
    Never commit `.env` files to version control. Add `.env` to your `.gitignore` file.

### 4. Install Browser Automation (Optional)

!!! info "Only for BrowserAgent"
    **This step is optional** and only required if you plan to use `BrowserAgent` for web automation and scraping.

**Important**: After installing the `playwright` package (included in basic installation), you must separately install browser binaries:

```bash
# Install Chromium (recommended - smallest download)
playwright install chromium

# Or install all browsers (Chrome, Firefox, WebKit)
playwright install

# On Linux: Install system dependencies
playwright install --with-deps chromium
```

If you skip this step, BrowserAgent will fail with an error about missing browser binaries. All other MARSYS features will work normally.

## ✅ Verify Installation

Run this quick test to verify everything is working:

```python
# test_installation.py
import asyncio
from marsys import Orchestra, Agent
from marsys.models import ModelConfig

async def test():
    # Create a simple agent
    agent = Agent(
        model_config=ModelConfig(
            type="api",
            name="gpt-4",
            provider="openai"
        ),
        agent_name="TestAgent",
        description="Test agent for verification"
    )

    # Run a simple task
    result = await Orchestra.run(
        task="Say 'Hello, MARSYS is working!'",
        topology={"nodes": ["TestAgent"], "edges": []}
    )

    print("✅ Installation successful!")
    print(f"Response: {result.final_response}")

if __name__ == "__main__":
    asyncio.run(test())
```

Run the test:
```bash
python test_installation.py
```

## 🐳 Docker Installation

For containerized deployments:

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  marsys:
    image: marsys:latest
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./workspace:/app/workspace
      - ./logs:/app/logs
    ports:
      - "8000:8000"  # If running web interface
```

Build and run:
```bash
# Build the image
docker-compose build

# Run the container
docker-compose up
```

### Using Docker CLI

```bash
# Build image
docker build -t marsys .

# Run container
docker run -it \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd)/workspace:/app/workspace \
  marsys
```

## 📦 Package Structure

After installation, you'll have access to:

```
marsys/
├── src/
│   ├── coordination/     # Orchestra and topology system
│   ├── agents/           # Agent implementations
│   ├── models/           # Model configurations
│   ├── environment/      # Browser and OS tools
│   └── utils/            # Utility functions
├── examples/             # Example implementations
├── tests/                # Test suite
└── docs/                 # Documentation
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required* |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required* |
| `GOOGLE_API_KEY` | Google AI API key | Required* |
| `HEADLESS` | Run browsers headlessly | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_RETRIES` | API retry attempts | `3` |
| `TIMEOUT` | Default timeout (seconds) | `300` |

*At least one API key is required

### Model Providers Setup

=== "OpenAI"
    ```python
    ModelConfig(
        type="api",
        name="gpt-4",
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    ```

=== "Anthropic"
    ```python
    ModelConfig(
        type="api",
        name="claude-3-sonnet",
        provider="anthropic",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    ```

=== "Google"
    ```python
    ModelConfig(
        type="api",
        name="gemini-pro",
        provider="google",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    ```

=== "Local (Ollama)"
    ```python
    ModelConfig(
        type="local",
        name="llama2",
        provider="ollama",
        base_url="http://localhost:11434"
    )
    ```

## 🔍 Troubleshooting

### Common Issues and Solutions

??? error "ImportError: No module named 'src'"
    **Solution**: Ensure you're in the project root and have installed MARSYS:
    ```bash
    cd MARSYS
    pip install -e .
    ```

??? error "API Key not found"
    **Solution**: Check your `.env` file is in the project root:
    ```bash
    # Verify .env exists
    ls -la .env

    # Check environment variable
    echo $OPENAI_API_KEY
    ```

??? error "Playwright browser not found"
    **Solution**: Install browser binaries:
    ```bash
    playwright install chromium
    # For all browsers
    playwright install
    ```

??? error "Async syntax error"
    **Solution**: MARSYS requires Python 3.12+ for async support:
    ```bash
    python --version  # Should be 3.12 or higher
    ```

### Platform-Specific Issues

=== "macOS"
    - For M1/M2 Macs, use Python 3.10+ for best compatibility
    - Install Rosetta 2 if needed: `softwareupdate --install-rosetta`

=== "Windows"
    - Use PowerShell or WSL2 for best experience
    - Ensure long path support is enabled in Windows

=== "Linux"
    - Install system dependencies for Playwright:
      ```bash
      playwright install-deps
      ```
    - On Ubuntu/Debian, you may need: `sudo apt-get install python3-dev`

## 🚦 Next Steps

Installation complete! Now you're ready to:

<div class="grid cards" markdown="1">

- :material-rocket-launch:{ .lg .middle } **[Quick Start](quick-start/)**

    ---

    Build your first multi-agent system in 10 minutes

- :material-robot:{ .lg .middle } **[Create Your First Agent](first-agent/)**

    ---

    Learn how to create custom agents with tools

- :material-cog:{ .lg .middle } **[Configuration Guide](configuration/)**

    ---

    Explore advanced configuration options

</div>

## 🆘 Need Help?

- 📖 Check the [FAQ](../project/faq/)
- 🐛 Report issues on [GitHub](https://github.com/rezaho/MARSYS/issues)
- 💬 Join our [Discord Community](https://discord.gg/marsys)
- 📧 Email support: [support@marsys.io](mailto:support@marsys.io)

---

!!! success "Ready to build?"
    Head to the [Quick Start Guide](quick-start/) to create your first multi-agent system!
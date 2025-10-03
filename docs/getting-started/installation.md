# Installation

Get MARSYS up and running on your system in just a few minutes.

## 📋 Prerequisites

- **Python 3.12+** (required)
- **pip** package manager
- **Git** for cloning the repository (optional)
- **API Keys** from at least one provider (OpenAI, Anthropic, Google)

## 🚀 Quick Install

### Option 1: Install from PyPI (Recommended)

MARSYS offers flexible installation options based on your needs:

=== "Basic (Recommended)"
    Everything you need for most use cases:
    ```bash
    pip install marsys
    ```

    Includes: API models, browser automation, UI, tools, logging

=== "With Local Models"
    Add PyTorch and Transformers for local LLM/VLM support:
    ```bash
    pip install marsys[local-models]
    ```

    Includes: Everything in Basic + PyTorch, Transformers, TRL, Datasets

=== "Production"
    High-performance inference with vLLM:
    ```bash
    pip install marsys[production]
    ```

    Includes: vLLM, Flash Attention, Triton, Ninja

=== "Development"
    Complete setup for contributors:
    ```bash
    pip install marsys[dev]
    ```

    Includes: Everything + testing, linting, documentation tools

### Option 2: Using uv (Faster)

[uv](https://github.com/astral-sh/uv) is 10-100x faster than pip:

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install marsys (same syntax as pip)
uv pip install marsys[local-models]
```

### Option 3: Install from Source

```bash
# Clone the repository
git clone https://github.com/rezaho/MARSYS.git
cd MARSYS

# Install in development mode
pip install -e .[dev]
```

## 🔧 Detailed Installation

### 1. Set Up Virtual Environment

!!! tip "Best Practice"
    Always use a virtual environment to avoid dependency conflicts

=== "venv"
    ```bash
    python -m venv marsys-env
    source marsys-env/bin/activate  # On Windows: marsys-env\Scripts\activate
    ```

=== "conda"
    ```bash
    conda create -n marsys python=3.11
    conda activate marsys
    ```

=== "uv"
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

### 2. Choose Your Installation

| Installation | Size | Time | Use Case |
|-------------|------|------|----------|
| `pip install marsys` | ~200 MB | 1-2 min | API models + browser + tools ✨ |
| `pip install marsys[local-models]` | ~2-3 GB | 5-10 min | + Local LLMs/VLMs |
| `pip install marsys[production]` | ~1 GB | 3-5 min | + High-performance inference |
| `pip install marsys[dev]` | ~3+ GB | 10-15 min | + Testing & docs tools |

### 3. Configure API Keys

Create a `.env` file in your project root:

```bash
# .env
# Required: At least one API key
OPENAI_API_KEY="sk-..."           # OpenAI GPT models
ANTHROPIC_API_KEY="sk-ant-..."    # Claude models
GOOGLE_API_KEY="AIza..."          # Gemini models

# Optional: Additional configurations
HEADLESS=true                      # Browser automation mode
LOG_LEVEL=INFO                     # Logging verbosity
```

!!! warning "Security"
    Never commit `.env` files to version control. Add `.env` to your `.gitignore` file.

### 4. Install Browser Automation (Optional)

For web scraping and browser agents:

```bash
# Install Playwright browsers
playwright install chromium

# Install all browsers (Chrome, Firefox, Safari)
playwright install

# Install system dependencies (Linux only)
playwright install-deps
```

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
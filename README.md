# MARSYS - Multi-Agent Reasoning Systems

<div align="center">

![MARSYS Logo](https://img.shields.io/badge/MARSYS-v0.1--beta-blue?style=for-the-badge)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=for-the-badge)](LICENSE)
[![CLA assistant](https://cla-assistant.io/readme/badge/rezaho/MARSYS)](https://cla-assistant.io/rezaho/MARSYS)
[![Documentation](https://img.shields.io/badge/docs-marsys.ai-orange?style=for-the-badge)](https://marsys.ai/framework)

**A framework for orchestrating collaborative AI agents with sophisticated reasoning, planning, and autonomous capabilities**

[Documentation](https://marsys.ai/framework) | [Quick Start](#quick-start) | [Examples](examples/) | [Contributing](#contributing)

</div>

---

## What is MARSYS?

MARSYS (Multi-Agent Reasoning Systems) is a Python framework for building and coordinating multi-agent AI systems. It provides flexible topology definitions, parallel execution with agent pools, state persistence, and human-in-the-loop support.

### Core Features

- **Multi-Agent Orchestration**: Coordinate workflows with multiple specialized agents
- **Flexible Topologies**: Define agent relationships using strings, objects, or pre-defined patterns
- **Parallel Execution**: True concurrency with AgentPool and dynamic branch spawning
- **State Persistence**: Pause/resume with checkpointing for long-running tasks
- **Human-in-the-Loop**: Built-in user interaction for approval workflows and error recovery
- **Multi-Model Support**: OpenAI, Anthropic, Google, OpenRouter, xAI, and local models

---

## Quick Start

### Installation

**Create and activate a virtual environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):**
```bash
uv venv
source .venv/bin/activate  # Unix/macOS
# .venv\Scripts\activate   # Windows
```

**Install MARSYS:**
```bash
pip install marsys
```

With optional dependencies:
```bash
pip install marsys[local-models]  # Local model support (PyTorch, Transformers)
pip install marsys[production]    # Production inference (vLLM, Flash Attention)
pip install marsys[dev]           # Everything (local-models + production + testing + docs)
```

**Install from source (for development):**
```bash
git clone https://github.com/rezaho/MARSYS.git
cd MARSYS
pip install -e ".[dev]"
```

### API Key Configuration

Configure your API keys via environment variables or a `.env` file:

```bash
# Environment variables
export OPENROUTER_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

Or create a `.env` file in your project root:
```bash
OPENROUTER_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### Playwright Setup (BrowserAgent only)

Only required if you plan to use BrowserAgent for web automation:
```bash
playwright install chromium --with-deps
```

### Basic Usage

A simple two-agent collaboration using `allowed_peers`:

```python
from marsys.agents import Agent
from marsys.models import ModelConfig

model_config = ModelConfig(
    type="api",
    name="anthropic/claude-haiku-4.5",
    provider="openrouter"
)

researcher = Agent(
    model_config=model_config,
    name="Researcher",
    goal="Expert at finding and analyzing information",
    instruction="You are a research specialist. Find and analyze information thoroughly.",
    allowed_peers=["Writer"]
)

writer = Agent(
    model_config=model_config,
    name="Writer",
    goal="Skilled at creating clear, engaging content",
    instruction="You are a skilled writer. Create clear, engaging content based on research.",
    allowed_peers=[]
)

result = await researcher.auto_run(
    task="Research the latest AI breakthroughs and write a summary",
    max_steps=20,
    verbosity=1
)

print(result)
```

### Using Orchestra with Topology

For more control, define the topology explicitly using `Orchestra.run()`:

```python
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

model_config = ModelConfig(
    type="api",
    name="anthropic/claude-haiku-4.5",
    provider="openrouter"
)

researcher = Agent(
    model_config=model_config,
    name="Researcher",
    goal="Expert at finding and analyzing information",
    instruction="You are a research specialist. Find and analyze information thoroughly."
)

writer = Agent(
    model_config=model_config,
    name="Writer",
    goal="Skilled at creating clear, engaging content",
    instruction="You are a skilled writer. Create clear, engaging content based on research."
)

topology = {
    "agents": ["Researcher", "Writer"],
    "flows": ["Researcher -> Writer"]
}

result = await Orchestra.run(
    task="Research the latest AI breakthroughs and write a summary",
    topology=topology
)

print(result.final_response)
```

[More examples](examples/)

---

## Contributing

We welcome contributions from the community. MARSYS is an open-source project that thrives on collaboration.

### Contributor License Agreement (CLA)

Before your first contribution can be merged, you must sign our CLA. This is a one-time, automated process:

1. Open a pull request
2. CLA Assistant bot will comment with a link
3. Click the link and sign
4. Your PR will be automatically unblocked

The CLA ensures legal clarity and protects both contributors and the project. You retain ownership of your code and can use it elsewhere. See [docs/CLA.md](docs/CLA.md) for full details.

### How to Contribute

1. Fork the repository and create your branch from `main`
2. Make your changes and ensure tests pass
3. Write/update tests for your changes
4. Submit a pull request with a clear description

### Development Setup

```bash
git clone https://github.com/rezaho/MARSYS.git
cd MARSYS
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
```

[Contributing guide](CONTRIBUTING.md)

---

## Citation

If you use MARSYS in your research or projects, please cite:

```bibtex
@software{marsys2025,
  author = {Hosseini, Reza},
  title = {MARSYS: Multi-Agent Reasoning Systems Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/rezaho/MARSYS}
}
```

---

## License

MARSYS is released under the **Apache License 2.0**. See [LICENSE](LICENSE) for full terms.

Copyright 2025 Marsys Project. Original Author: [rezaho](https://github.com/rezaho)

---

<div align="center">

**Built by [Reza Hosseini](https://github.com/rezaho) and contributors**

[Documentation](https://marsys.ai/framework) | [Examples](examples/) | [GitHub](https://github.com/rezaho/MARSYS)

</div>

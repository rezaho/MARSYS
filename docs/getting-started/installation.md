# Installation

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/MARSYS.git
cd MARSYS
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (optional)
GOOGLE_API_KEY=your_google_api_key_here

# Browser automation (optional)
HEADLESS=true
```

## Optional Components

### Browser Automation

For browser-based agents, install Playwright:

```bash
pip install playwright
playwright install chromium
```

### Local Models

For running local models:

```bash
pip install torch transformers
```

## Verification

Verify your installation:

```python
from src.agents.agent import Agent
from src.utils.config import ModelConfig

# This should not raise any import errors
print("Installation successful!")
```

## Docker Installation

### Using Docker Compose

```yaml
version: '3.8'
services:
  multi-agent-ai:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./src:/app/src
    command: python -m src.main
```

### Build and Run

```bash
docker-compose build
docker-compose up
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're in the project root and have activated the virtual environment
2. **API Key Errors**: Check that your `.env` file is properly configured
3. **Playwright Issues**: Run `playwright install-deps` if you encounter browser automation errors

### Getting Help

- Check the [FAQ](../project/faq.md)
- Open an issue on [GitHub](https://github.com/yourusername/MARSYS/issues)
- Join our [Discord community](https://discord.gg/yourinvite)

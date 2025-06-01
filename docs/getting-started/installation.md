\
<!-- filepath: /home/rezaho/research_projects/Multi-agent_AI_Learning/docs/getting-started/installation.md -->
# Installation

## Prerequisites

- Python 3.12 (as specified in `pyproject.toml`)
- pip package manager
- Git

## Basic Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rezaho/MARSys.git 
cd Marsys 
```

### 2. Create Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

The project uses `pyproject.toml` to manage dependencies. Install them using pip:

```bash
pip install .
```
This command will install all necessary packages listed in the `pyproject.toml` file, including core libraries and those for optional features like local model support.

### 4. Set Up Environment Variables

Create a `.env` file in the project root by copying the example file if provided, or create it manually:

```env
# .env

# OpenAI API Key (Required for OpenAI models)
OPENAI_API_KEY="your_openai_api_key_here"

# Anthropic API Key (Optional, for Anthropic models)
# ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Google API Key (Optional, for Google models)
# GOOGLE_API_KEY="your_google_api_key_here"

# Configuration for browser automation tools (e.g., Playwright)
# HEADLESS=true # Set to false to see browser UI, true for headless mode
```
Fill in your actual API keys and adjust other settings as needed.

## Optional Components

The core dependencies installed via `pip install .` already include support for:
- **Browser Automation:** Using Playwright. Ensure browser binaries are installed if needed:
  ```bash
  playwright install # Installs default browsers like Chromium
  # playwright install chromium # To install a specific browser
  ```
- **Local Models:** Using libraries like `torch`, `transformers`, `vllm`, etc.

Refer to the `pyproject.toml` for a full list of dependencies.

## Verification

To verify your installation, you can try running a simple script or importing core modules. For example, create a Python file (e.g., `verify_install.py`) in the project root:

```python
# verify_install.py
try:
    from src.agents import Agent
    from src.models.models import ModelConfig
    # Add any other core imports you want to test
    print("MARSYS framework core modules imported successfully!")
    print("Installation appears to be successful.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("There might be an issue with the installation or your Python environment.")

# Run it with: python verify_install.py
```
<!-- TODO: Provide a more concrete example or a simple CLI command if available for verification -->

## Docker Installation
<!-- TODO: Verify Docker entrypoint as src/main.py is currently empty. -->
<!-- TODO: Ensure Dockerfile is up-to-date with pyproject.toml based installation. -->

If a `Dockerfile` and `docker-compose.yml` are provided and maintained:

### Using Docker Compose

Example `docker-compose.yml` (ensure this matches the actual file):
```yaml
version: '3.8'
services:
  multi-agent-ai:
    build:
      context: .
      # dockerfile: Dockerfile # Specify if not named Dockerfile
    environment:
      # Pass API keys and other necessary environment variables
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      # - HEADLESS=${HEADLESS}
    volumes:
      - ./src:/app/src # Mount your source code
      # Add other volume mounts if needed, e.g., for logs or data
    # command: python -m src.main # Adjust if your entry point is different
    # user: "${UID}:${GID}" # Optional: For running as non-root user
    # ports: # Optional: Expose ports if your application runs a server
    #  - "8000:8000"
```

### Build and Run with Docker

```bash
# Ensure you have Docker and Docker Compose installed
# Create a .env file with your API keys at the project root, Docker Compose will pick it up

docker-compose build
docker-compose up
```

## Troubleshooting

### Common Issues

1.  **Import Errors**:
    *   Ensure you are in the project root directory.
    *   Confirm that your virtual environment is activated (`source venv/bin/activate` or `venv\\Scripts\\activate`).
    *   Verify that `pip install .` completed successfully.
2.  **API Key Errors**:
    *   Double-check that your `.env` file is in the project root and correctly formatted.
    *   Ensure the environment variables (e.g., `OPENAI_API_KEY`) are correctly named and have valid keys.
3.  **Playwright Issues**:
    *   If you encounter errors related to browser automation, try running `playwright install` to ensure necessary browser binaries are downloaded.
    *   On some Linux systems, you might need to install additional system dependencies for Playwright: `playwright install-deps`.
4.  **Dependency Conflicts or Issues**:
    *   Ensure `pip`, `setuptools`, and `wheel` are up to date: `pip install --upgrade pip setuptools wheel`.
    *   If you encounter persistent issues, try creating a fresh virtual environment and reinstalling dependencies.

### Getting Help

- Check the [FAQ](../project/faq.md) for answers to common questions.
- Open an issue on the project's [GitHub Issues page](https://github.com/rezaho/MARSys/issues). 
- Join our community forum or chat (e.g., Discord, Slack) if available. <!-- TODO: Update with actual community link if one exists, e.g., https://discord.gg/YOUR_INVITE_CODE -->

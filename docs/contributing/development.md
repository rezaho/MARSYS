# Development Setup

Set up your development environment for contributing to the framework.

## Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

## Setup Steps

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/Multi-agent_AI_Learning.git
   cd Multi-agent_AI_Learning
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_agents.py
```

## Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
make lint
```

## Next Steps

- Read [Architecture Guide](architecture.md)
- Review [Contributing Guidelines](guidelines.md)
- Explore [Testing Guide](testing.md)

# Contributing

Join the MARSYS community and help build the future of multi-agent AI systems!

## 🤝 Ways to Contribute

<div class="grid cards" markdown="1">

- :material-bug:{ .lg .middle } **Report Issues**

    ---

    Found a bug? Let us know!

    - Bug reports
    - Performance issues
    - Documentation gaps
    - Feature requests

    [Report Issue →](https://github.com/yourusername/marsys/issues/new)

- :material-code-tags:{ .lg .middle } **Code Contributions**

    ---

    Improve the framework

    - Bug fixes
    - New features
    - Performance optimizations
    - Refactoring

    [View Guidelines →](#code-contributions)

- :material-file-document:{ .lg .middle } **Documentation**

    ---

    Help others learn

    - Fix typos and errors
    - Improve explanations
    - Add examples
    - Write tutorials

    [Documentation Guide →](#documentation)

- :material-test-tube:{ .lg .middle } **Testing**

    ---

    Ensure quality

    - Write unit tests
    - Add integration tests
    - Test edge cases
    - Report test failures

    [Testing Guide →](#testing)

</div>

## 🚀 Quick Start

### 1. Fork & Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/yourusername/marsys.git
cd marsys

# Add upstream remote
git remote add upstream https://github.com/original/marsys.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 4. Make Changes

```bash
# Make your changes
# Add tests
# Update documentation

# Run tests
pytest tests/

# Check code style
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Commit changes
git add .
git commit -m "feat: add amazing feature"
```

### 5. Submit Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Open PR on GitHub
# Link any related issues
# Describe your changes
```

## 📝 Code Contributions

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add parallel agent execution
fix: resolve memory leak in agent pools
docs: update topology documentation
test: add tests for branch executor
refactor: simplify validation logic
perf: optimize message passing
```

### Pull Request Guidelines

✅ **DO:**
- Keep PRs focused on a single feature/fix
- Include tests for new functionality
- Update documentation
- Add type hints
- Follow existing code patterns
- Link related issues

❌ **DON'T:**
- Mix unrelated changes
- Break existing tests
- Introduce unnecessary dependencies
- Change code style arbitrarily
- Submit incomplete work

## 📚 Documentation

### Documentation Structure

```
docs/
├── getting-started/    # Beginner guides
├── concepts/          # Core concepts
├── tutorials/         # Step-by-step tutorials
├── api/              # API reference
├── use-cases/        # Real-world examples
└── contributing/     # This section
```

### Writing Style

- **Clear**: Use simple, direct language
- **Concise**: Be brief but complete
- **Practical**: Include working examples
- **Visual**: Add diagrams where helpful

### Example Template

```markdown
# Feature Name

Brief description of what this feature does.

## Overview

Explain the purpose and use cases.

## Example

\```python
# Complete, working example
from marsys import Feature

feature = Feature()
result = feature.do_something()
\```

## API Reference

Document all public methods and parameters.

## Best Practices

Tips for effective usage.
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test fixtures
```

### Writing Tests

```python
import pytest
from marsys.agents import Agent

@pytest.mark.asyncio
async def test_agent_creation():
    """Test agent can be created with config."""
    agent = Agent(
        model_config=test_config,
        agent_name="TestAgent"
    )

    assert agent.name == "TestAgent"
    assert agent.model is not None

@pytest.mark.asyncio
async def test_agent_execution():
    """Test agent can execute tasks."""
    agent = create_test_agent()
    result = await agent.run("Test task")

    assert result.success
    assert result.response is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_agent.py

# Run with markers
pytest -m "not slow"
```

## 🏗️ Architecture

### Key Principles

1. **Pure Functions**: Agents use pure `_run()` methods
2. **Centralized Validation**: Single validation processor
3. **Dynamic Branching**: Runtime parallel execution
4. **Topology-Driven**: Graph-based routing

### Adding New Features

1. **Identify Component**: Where does it belong?
2. **Design API**: Keep it consistent
3. **Implement**: Follow existing patterns
4. **Test**: Comprehensive test coverage
5. **Document**: Update relevant docs

## 🔄 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version bumped
- [ ] Tagged release
- [ ] Published to PyPI

## 👥 Community

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/).

**Be:**
- Respectful
- Constructive
- Inclusive
- Professional

**Don't:**
- Harass or discriminate
- Troll or spam
- Use inappropriate language
- Share private information

### Getting Help

- **Discord**: [Join our server](https://discord.gg/marsys)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/marsys/discussions)
- **Issues**: [GitHub Issues](https://github.com/yourusername/marsys/issues)
- **Email**: support@marsys.ai

## 🎯 Current Priorities

### High Priority
- Performance optimization
- Browser agent stability
- Documentation improvements
- Test coverage expansion

### Feature Requests
- Streaming responses
- Distributed execution
- Advanced learning agents
- Custom storage backends

### Known Issues
Check our [issue tracker](https://github.com/yourusername/marsys/issues) for current bugs and feature requests.

## 📜 License

By contributing, you agree that your contributions will be licensed under the same license as the project (Apache License 2.0).

## 🙏 Recognition

### Contributors

Thanks to all our contributors!

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

### Special Thanks

- OpenAI, Anthropic, Google for AI models
- The open-source community
- All our users and testers

## 🚦 Next Steps

<div class="grid cards" markdown="1">

- :material-github:{ .lg .middle } **[GitHub Repo](https://github.com/yourusername/marsys)**

    ---

    Star and watch the repository

- :material-book:{ .lg .middle } **[Documentation](../index.md)**

    ---

    Learn the framework

- :material-discord:{ .lg .middle } **[Join Discord](https://discord.gg/marsys)**

    ---

    Chat with the community

- :material-email:{ .lg .middle } **[Contact Us](mailto:support@marsys.ai)**

    ---

    Get in touch

</div>

---

!!! success "Thank You!"
    Every contribution makes MARSYS better. Whether it's fixing a typo, adding a feature, or helping others - we appreciate your effort!

!!! tip "First Time?"
    If this is your first open-source contribution, check out [First Contributions](https://github.com/firstcontributions/first-contributions) for a gentle introduction.
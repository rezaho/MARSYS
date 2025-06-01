# Contributing to Multi-Agent Reasoning Systems (MARSYS)

Thank you for your interest in contributing to the Multi-Agent Reasoning Systems (MARSYS) framework! This guide will help you get started.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/rezaho/MARSYS.git
cd MARSYS

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/MARSYS.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_agents.py
```

## Development Guidelines

### Code Style

We follow PEP 8 with some modifications:
- Line length: 100 characters
- Use type hints for all function signatures
- Use docstrings for all public methods

```python
def process_message(
    self,
    message: Message,
    timeout: float = 30.0
) -> Optional[Message]:
    """
    Process a message with the specified timeout.
    
    Args:
        message: The message to process
        timeout: Maximum time to wait for processing
        
    Returns:
        Processed message or None if timeout
        
    Raises:
        ValueError: If message format is invalid
    """
    # Implementation
```

### Async/Await Guidelines

All I/O operations must be asynchronous:

```python
# Good
async def fetch_data(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Bad
def fetch_data(url: str) -> str:
    response = requests.get(url)
    return response.text()
```

### Message Handling

Always return Message objects:

```python
# Good
return Message(
    role="assistant",
    content="Operation completed successfully",
    name=self.name
)

# Bad
return "Operation completed successfully"
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

Follow these principles:
- Write tests for new functionality
- Update documentation
- Keep commits focused and atomic
- Write clear commit messages

### 3. Test Your Changes

```bash
# Run tests
pytest

# Check code style
flake8 src/
mypy src/

# Run documentation build
cd docs && mkdocs build
```

### 4. Submit Pull Request

1. Push your branch to your fork
2. Create a pull request against the main branch
3. Fill out the PR template completely
4. Wait for review and address feedback

## Areas for Contribution

### 🐛 Bug Fixes
- Check the [issue tracker](https://github.com/rezaho/MARSYS/issues)
- Look for "good first issue" labels
- Reproduce the bug and write a test
- Submit a fix with the test

### ✨ New Features
- Discuss in an issue first
- Write comprehensive tests
- Update documentation
- Add examples if applicable

### 📚 Documentation
- Fix typos and clarify existing docs
- Add examples and tutorials
- Improve API documentation
- Translate documentation

### 🧪 Testing
- Increase test coverage
- Add integration tests
- Improve test performance
- Add edge case tests

### 🎯 Examples
- Create new example applications
- Improve existing examples
- Add domain-specific examples
- Create video tutorials

## Architecture Guidelines

### Adding New Agents

1. Inherit from `BaseAgent` or `Agent`
2. Implement required methods
3. Add tests in `tests/test_agents/`
4. Document in `docs/api/`

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    """
    Custom agent implementation.
    
    This agent specializes in...
    """
    
    async def process(self, task: str) -> Message:
        """Process the given task."""
        # Implementation
        pass
```

### Adding New Tools

1. Create function with type hints and docstring
2. Add to `AVAILABLE_TOOLS` or agent-specific tools
3. Write tests
4. Document usage

```python
def new_tool(param1: str, param2: int = 10) -> str:
    """
    Brief description of the tool.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Description of return value
    """
    # Implementation
    return result
```

### Adding New Models

1. Inherit from appropriate base class
2. Implement `run` method
3. Handle errors gracefully
4. Add configuration support

```python
from src.models.base_models import BaseLLM

class NewModelProvider(BaseLLM):
    async def run(
        self,
        messages: List[Message],
        tools: Optional[List[Dict]] = None,
        output_json: bool = False
    ) -> Dict[str, Any]:
        """Run the model."""
        # Implementation
        pass
```

## Testing Requirements

### Unit Tests

Every new function/method needs tests:

```python
import pytest
from src.agents.agent import Agent

@pytest.mark.asyncio
async def test_agent_creation():
    """Test agent can be created with valid config."""
    agent = Agent(
        name="test_agent",
        model_config=ModelConfig(
            provider="openai",
            model_name="gpt-4.1-mini"
        )
    )
    assert agent.name == "test_agent"
    assert agent.model is not None
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.asyncio
async def test_agent_communication():
    """Test two agents can communicate."""
    agent1 = Agent(name="agent1", register=True, ...)
    agent2 = Agent(name="agent2", register=True, ...)
    
    response = await agent1.invoke_agent("agent2", "Hello")
    assert response.role != "error"
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: Optional[int] = None
) -> Tuple[str, int]:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose
    and any important details about the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to None.
        
    Returns:
        A tuple containing:
            - str: Description of first return value
            - int: Description of second return value
            
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result)
        ('processed_test', 42)
    """
    # Implementation
```

### Markdown Documentation

- Use clear headings
- Include code examples
- Add diagrams where helpful
- Link to related sections

## Review Process

### What We Look For

1. **Code Quality**
   - Follows style guidelines
   - Well-tested
   - Properly documented
   - No breaking changes

2. **Functionality**
   - Works as intended
   - Handles edge cases
   - Performs well
   - Backwards compatible

3. **Documentation**
   - Clear and complete
   - Examples provided
   - API docs updated
   - Changelog updated

### Review Timeline

- Initial review: 2-3 business days
- Follow-up reviews: 1-2 business days
- Merge after approval from maintainer

## Release Process

1. **Version Numbering**: We use semantic versioning (MAJOR.MINOR.PATCH)
2. **Changelog**: Update CHANGELOG.md with your changes
3. **Documentation**: Ensure docs are updated
4. **Tests**: All tests must pass
5. **Release Notes**: Maintainers will create release notes

## Getting Help

### Resources

- [Documentation](https://rezaho.github.io/MARSYS/)
- [Issue Tracker](https://github.com/rezaho/MARSYS/issues)
- [Discussions](https://github.com/rezaho/MARSYS/discussions)
- [Discord Community](https://discord.gg/yourinvite)

### Questions?

- Check existing issues and discussions
- Ask in Discord for quick questions
- Open an issue for bugs or feature requests
- Email maintainers for sensitive issues

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- Annual contributor spotlight

Thank you for contributing to Multi-Agent Reasoning Systems (MARSYS)! 🎉

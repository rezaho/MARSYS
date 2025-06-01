# Frequently Asked Questions

## General Questions

### What is the Multi-Agent Reasoning Systems (MARSYS) Framework?

The Multi-Agent Reasoning Systems (MARSYS) Framework is a Python library for building systems where multiple AI agents can work together, communicate, and solve complex tasks. It provides:
- A unified interface for different AI models (OpenAI, Anthropic, Google)
- Tools and browser automation capabilities
- Inter-agent communication protocols
- Memory management systems
- Extensible architecture

### Who is this framework for?

This framework is designed for:
- **Developers** building AI-powered applications
- **Researchers** exploring multi-agent systems
- **Teams** creating automated workflows
- **Enterprises** needing scalable AI solutions

### What are the main use cases?

Common use cases include:
- Research and analysis systems
- Automated customer support
- Code review and generation
- Data processing pipelines
- Web scraping and automation
- Content creation workflows

## Installation & Setup

### What are the system requirements?

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Operating System: Windows, macOS, or Linux
- Internet connection for API-based models

### How do I install the framework?

```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

### What API keys do I need?

Depending on your model choices:
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_API_KEY`

Set them as environment variables or in a `.env` file.

### Can I use local models?

Yes! The framework supports local models through the model abstraction layer. You'll need to implement a custom model class inheriting from `BaseLLM`.

## Agent Questions

### How do I create an agent?

Basic agent creation:
```python
from src.agents import Agent
from src.models.models import ModelConfig

agent = Agent(
    name="my_agent",
    model_config=ModelConfig(
        provider="openai",
        model_name="gpt-4"
    ),
    instructions="You are a helpful assistant"
)
```

### What's the difference between Agent and BaseAgent?

- **BaseAgent**: Abstract base class with core functionality (registration, logging)
- **Agent**: Concrete implementation with model integration and memory management
- **BrowserAgent**: Specialized agent with web automation capabilities
- **LearnableAgent**: Agent that can improve through feedback

### How do agents communicate?

Agents communicate through the registry:
```python
# Agent A invokes Agent B
response = await agent_a.invoke_agent(
    "agent_b",
    "Please analyze this data..."
)
```

### Can agents share memory?

Agents have individual memory by default, but you can implement shared memory patterns:
```python
from src.agents.memory import MemoryManager

shared_memory = MemoryManager()

# Both agents use the same memory
agent1 = Agent(..., memory=shared_memory)
agent2 = Agent(..., memory=shared_memory)
```

## Tool Questions

### How do I create custom tools?

Define a function with type hints and docstring:
```python
def my_tool(param: str) -> str:
    """Tool description here."""
    return f"Processed: {param}"

agent = Agent(
    tools={"my_tool": my_tool}
)
```

### What built-in tools are available?

- `calculate`: Mathematical calculations
- `get_time`: Current time in any timezone
- File operations (with custom implementation)
- Web search (with custom implementation)
- Browser automation (through BrowserAgent)

### Can tools be async?

Yes! Async tools are preferred:
```python
async def async_tool(url: str) -> str:
    """Fetch data asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

### How do I handle tool errors?

The framework automatically catches tool errors and returns them as error messages. You can also implement custom error handling:
```python
def safe_tool(param: str) -> str:
    try:
        # Tool logic
        return result
    except SpecificError as e:
        return f"Error: {str(e)}"
```

## Memory & Messages

### How does memory work?

Memory is append-only and chronological:
```python
# Messages are automatically stored
response = await agent.auto_run(initial_request="...")

# Retrieve all messages
all_messages = agent.memory.retrieve_all()

# Search messages
results = agent.memory.search("keyword")
```

### What message roles are available?

Standard roles:
- `system`: System instructions
- `user`: User input
- `assistant`: AI responses
- `tool`: Tool execution results
- `error`: Error messages

Extended roles:
- `agent_call`: Agent invoking another agent
- `agent_response`: Response from invoked agent

### How do I clear memory?

```python
# Clear all memory
agent.memory = MemoryManager()

# Or selectively keep recent messages
recent = agent.memory.retrieve_last_n(10)
agent.memory = MemoryManager()
for msg in recent:
    agent.memory.update_memory(msg)
```

### What's the message format?

Messages follow this structure (see `src/agents/memory.py` for the full definition):
```python
from src.agents.memory import Message

# Example:
new_message = Message(
    role="assistant",
    content="This is the assistant's response.",
    # message_id is generated automatically if not provided
    # Other optional fields: name, tool_calls, tool_call_id, agent_call
)
```

## Performance & Scaling

### How many agents can run simultaneously?

The framework can handle dozens of agents simultaneously. Limiting factors:
- Available memory
- API rate limits
- Model provider constraints

### How do I optimize performance?

1. Use appropriate models for tasks (GPT-4.1-MINI for simple, GPT-4 for complex)
2. Implement caching for repeated operations
3. Batch requests when possible
4. Use async operations throughout
5. Monitor and limit memory growth

### Can I distribute agents across servers?

The current implementation is designed for single-server deployment. For distributed systems, you would need to:
- Implement a distributed registry
- Add network communication layer
- Handle distributed state management

### How do I handle rate limits?

The framework includes automatic retry logic in some API model interactions. For custom rate limiting, you would typically implement it around your API calls.
(The previously mentioned `RateLimiter` class example has been removed as `src/utils/rate_limiter.py` was not found.)

## Troubleshooting

### Common Errors and Solutions

**Import Error: "No module named 'src'"**
- Run from project root directory
- Add project root to PYTHONPATH

**API Key Error**
- Check environment variables are set
- Verify `.env` file is in project root
- Ensure API key is valid

**Timeout Error**
- Increase timeout in ModelConfig
- Check network connection
- Verify API service status

**Memory Error**
- Clear agent memory periodically
- Reduce max_steps parameter
- Use more efficient model

### How do I enable debug logging?

Set the log level:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
export LOG_LEVEL=DEBUG
```

### Where can I get help?

1. Check the [documentation](../index.md)
2. Search [existing issues](https://github.com/yourusername/MARSYS/issues) <!-- TODO: Update placeholder URL -->
3. Ask in [discussions](https://github.com/yourusername/MARSYS/discussions) <!-- TODO: Update placeholder URL -->
4. Join our [Discord community](https://discord.gg/yourinvite) <!-- TODO: Update placeholder URL -->

## Advanced Topics

### Can I use custom model providers?

Yes! Implement the required model interface. For example, for a custom LLM, you might inherit from `BaseLLM` (see `src/models/models.py`):
```python
from src.models.models import BaseLLM
from typing import List, Dict, Optional

class CustomModel(BaseLLM):
    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        # Add other parameters as per BaseLLM.run or your custom needs
    ) -> str:
        # Your implementation to interact with the custom model
        # This method should be synchronous as per BaseLLM.run
        pass
```
Ensure your implementation matches the method signature of the base class you choose (e.g., `BaseLLM`, `BaseVLM`, `BaseAPIModel`).

### How do I implement learning agents?

Use the `LearnableAgent` class from `src/agents/learnable_agents.py` or implement learning in a custom agent inheriting from `BaseLearnableAgent`.
```python
from src.agents.learnable_agents import LearnableAgent # Or BaseLearnableAgent
from typing import Dict, Any

class MyLearningAgent(LearnableAgent): # Inherit from LearnableAgent
    def __init__(self, model, description, **kwargs): # Ensure constructor matches
        super().__init__(model=model, description=description, **kwargs)
        # Custom initialization for learning

    def learn_from_feedback(self, feedback: str, training_data: Any):
        # Example method: Update agent behavior based on feedback
        # This might involve updating PEFT heads or other learnable parameters
        # (Actual learning mechanisms need to be implemented based on your strategy)
        print(f"Learning from feedback: {feedback}")
        # Example: self.model.train_on_data(training_data) # Hypothetical
        pass

# Example usage (simplified):
# from src.models.models import BaseLLM
# my_local_model = BaseLLM(model_name="path/to/your/local/model")
# learning_agent = MyLearningAgent(
#     model=my_local_model,
#     description="An agent that learns from feedback",
#     learning_head="peft", # If using PEFT
#     learning_head_config={...} # PEFT config
# )
# learning_agent.learn_from_feedback("Good job!", example_data)

```

### Can agents modify their own code?

While agents can generate code, they cannot directly modify their runtime behavior for safety. You can implement controlled adaptation through:
- Dynamic tool registration
- Adjustable parameters
- Strategy pattern implementations

### Is there a GUI for managing agents?

Currently, the framework is API-only. Community contributions for GUI tools are welcome!

## Contributing

### How can I contribute?

See our [Contributing Guide](../contributing/guidelines.md) for details on:
- Setting up development environment
- Code style guidelines
- Submitting pull requests
- Testing requirements

### Can I use this commercially?

Check the LICENSE file for terms. Generally, the framework is open source and can be used commercially with attribution (verify this against the actual LICENSE).

### Who maintains this project?

The project is maintained by [Your Organization] <!-- TODO: Update placeholder --> with contributions from the community. See `CONTRIBUTORS.md` (if available) for the full list.

---

Still have questions? [Open an issue](https://github.com/yourusername/MARSYS/issues/new) <!-- TODO: Update placeholder URL --> or ask in our [Discord community](https://discord.gg/yourinvite) <!-- TODO: Update placeholder URL -->!

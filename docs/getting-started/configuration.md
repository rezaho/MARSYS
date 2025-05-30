# Configuration

Learn how to configure the Multi-Agent Reasoning Systems (MARSYS) Framework.

## Environment Variables

Create a `.env` file in your project root:

```bash
# API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
HUGGINGFACE_TOKEN=your-hf-token

# Logging
LOG_LEVEL=SUMMARY
```

## Model Configuration

```python
from src.models.models import ModelConfig

# OpenAI Configuration
openai_config = ModelConfig(
    type="api",  # Specify API-based model
    provider="openai",
    name="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Anthropic Configuration
anthropic_config = ModelConfig(
    type="api",
    provider="anthropic", 
    name="claude-3-opus-20240229",
    temperature=0.5
)

# Local Model with Hugging Face
local_config = ModelConfig(
    type="local",  # Specify local model
    model_class="llm",  # or "vlm" for vision-language models
    name="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype="float16",  # Optimize memory usage
    device_map="auto"  # Automatically distribute across available devices
)
```

## Agent Configuration

```python
from src.agents import Agent

# Create agent with specific configuration
agent = Agent(
    name="my_assistant",
    model_config=openai_config,
    instructions="You are a helpful assistant",
    max_tokens=1000,  # Override model config default
    allowed_peers=["researcher", "writer"]  # List of agents this one can invoke
)
```

## Memory Configuration

```python
# Default conversation memory
agent = Agent(
    name="assistant",
    model_config=config,
    instructions="...",
    memory_type="conversation_history"  # Default memory type
)

# Knowledge graph memory (if implemented)
agent = Agent(
    name="kg_agent",
    model_config=config,
    instructions="...",
    memory_type="kg"  # Use knowledge graph for complex reasoning
)
```

## Logging Configuration

Set the log level via environment variable or in code:

```python
import os
from src.agents.utils import LogLevel, RequestContext

# Via environment variable
os.environ['LOG_LEVEL'] = 'SUMMARY'  # Options: MINIMAL, SUMMARY, DETAILED, DEBUG

# Or in code when creating RequestContext
context = RequestContext(
    request_id="task-123",
    agent_name="my_agent",
    log_level=LogLevel.DETAILED  # Override default log level
)
```

## Next Steps

- Explore [agent types](../concepts/agents.md)
- Learn about [tools](../concepts/tools.md)
- See [examples](../use-cases/index.md)

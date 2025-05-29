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
from src.utils.config import ModelConfig

# OpenAI Configuration
openai_config = ModelConfig(
    provider="openai",
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000
)

# Anthropic Configuration
anthropic_config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    temperature=0.5
)

# Local Model with Hugging Face
local_config = ModelConfig(
    provider="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"  # or "cpu"
)
```

## Next Steps

- Explore [agent types](../concepts/agents.md)
- Learn about [tools](../concepts/tools.md)
- See [examples](../use-cases/index.md)

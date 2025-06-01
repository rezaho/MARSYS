# Configuration

Learn how to configure the Multi-Agent Reasoning Systems (MARSYS) Framework.

## Environment Variables

Create a `.env` file in your project root. The framework's `ModelConfig` will automatically attempt to load API keys from these environment variables if not provided directly in the configuration.

```env
# .env

# API Keys (ensure these match the expected variables by ModelConfig)
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key" # For Anthropic models
GOOGLE_API_KEY="your-google-key"     # For Google models
GROQ_API_KEY="your-groq-key"         # For Groq models
# OPENROUTER_API_KEY="your-openrouter-key" # For OpenRouter models (if used)
# HUGGINGFACE_TOKEN="your-hf-token" # Optional: May be used by Hugging Face libraries for private models

# Logging Configuration
# This can be used by your application or specific logging setups.
# The framework's core logging level for RequestContext can also be set programmatically.
LOG_LEVEL="SUMMARY" # Example: MINIMAL, SUMMARY, DETAILED, DEBUG
```

## Model Configuration (`ModelConfig`)

The `src.models.models.ModelConfig` class is used to define settings for language models, whether they are API-based or local.

```python
from src.models.models import ModelConfig

# --- API Model Configurations ---

# OpenAI Configuration
# ModelConfig will try to get OPENAI_API_KEY from environment if api_key is not set.
# base_url is automatically determined from 'provider' if not set.
openai_config = ModelConfig(
    type="api",
    provider="openai",
    name="gpt-4-turbo", # Or any other valid OpenAI model
    temperature=0.7,
    max_tokens=2000
    # api_key="sk-...", # Optionally provide directly
)

# Anthropic Configuration
anthropic_config = ModelConfig(
    type="api",
    provider="anthropic",
    name="claude-3-opus-20240229", # Or other Anthropic model
    temperature=0.5
    # api_key="sk-ant-...", # Optionally provide directly
)

# Google Configuration
google_config = ModelConfig(
    type="api",
    provider="google",
    name="gemini-pro", # Or other Google model
    # api_key="...", # Optionally provide directly
)

# Groq Configuration
groq_config = ModelConfig(
    type="api",
    provider="groq",
    name="llama3-8b-8192", # Or other Groq model
    # api_key="gsk_...", # Optionally provide directly
)


# --- Local Model Configuration ---
# For models loaded using Hugging Face Transformers library

# Example for a local LLM
local_llm_config = ModelConfig(
    type="local",
    model_class="llm", # Required for local models: "llm" or "vlm"
    name="mistralai/Mistral-7B-Instruct-v0.2", # Hugging Face model identifier
    torch_dtype="auto",  # Or "bfloat16", "float16" for optimization
    device_map="auto",   # For model distribution across devices
    # quantization_config={"load_in_8bit": True} # Optional: example quantization
)

# Example for a local VLM (Vision Language Model)
local_vlm_config = ModelConfig(
    type="local",
    model_class="vlm", # Required for local models
    name="Salesforce/blip-image-captioning-large", # Example VLM identifier
    torch_dtype="auto",
    device_map="auto"
)
```
**Note:** `ModelConfig` performs validation, including checking for required API keys for API-type models (by looking at environment variables based on the provider) and ensuring `model_class` is specified for local models.

## Agent Configuration

When creating an agent, you pass a `ModelConfig` instance to it.

```python
from src.agents.agents import Agent # Assuming Agent class is in src.agents.agents
# from src.models.models import ModelConfig # Already imported above

# Example: Create an agent with the OpenAI configuration from above
my_openai_agent = Agent(
    model=openai_config, # Pass the ModelConfig object
    description="An assistant that uses OpenAI's GPT-4 Turbo.",
    agent_name="OpenAIAssistant", # Optional: specify a name
    max_tokens=1500,  # Optional: Override default max_tokens from ModelConfig for this agent
    allowed_peers=["ResearcherAgent", "DataAnalyzerAgent"] # Optional: List of agents this one can invoke
)

# Example: Create an agent with a local model configuration
my_local_agent = Agent(
    model=local_llm_config, # Pass the ModelConfig for the local LLM
    description="An assistant that uses a local Mistral model.",
    agent_name="LocalMistralAssistant"
)
```

<!--
## Memory Configuration (Placeholder - Verify Agent Constructor)

The following shows how memory *might* be configured if the Agent class supports a `memory_type` parameter.
Please verify the `Agent` class constructor and available memory types.

```python
# Default conversation memory
# agent_with_default_memory = Agent(
#     name="assistant_default_mem",
#     model=openai_config, # Use a defined ModelConfig
#     description="Assistant with default memory.",
#     memory_type="conversation_history"  # Example: Default memory type
# )

# Knowledge graph memory (if implemented and supported by the Agent class)
# agent_with_kg_memory = Agent(
#     name="kg_agent_example",
#     model=openai_config, # Use a defined ModelConfig
#     description="Assistant with knowledge graph memory.",
#     memory_type="kg"  # Example: Use knowledge graph
# )
```
-->

## Logging Configuration

The framework uses a `RequestContext` object that carries logging information, including the `log_level`. You can set a default log level or override it per request. Log levels are defined in `src.agents.utils.LogLevel`.

```python
import os
from src.agents.utils import LogLevel, RequestContext # Assuming these are in src.agents.utils

# --- Setting Log Level via Environment Variable (Application-specific setup) ---
# Your application can read this environment variable to set a global default log level.
# os.environ['LOG_LEVEL'] = 'DEBUG' # Options: MINIMAL, SUMMARY, DETAILED, DEBUG

# --- Setting Log Level Programmatically in RequestContext ---
# When you create a RequestContext, you can specify its log_level.
# This is typically done when initiating a new task or operation.

# Example: Creating a new RequestContext for a task
initial_task_context = RequestContext(
    task_id="unique-task-identifier-123",
    initial_prompt="Analyze the provided data.",
    # progress_queue can be an asyncio.Queue() if using progress monitoring
    log_level=LogLevel.DETAILED  # Set specific log level for this task
)

# Agents will use the log_level from the RequestContext passed to them.
# await my_openai_agent.auto_run(
#     initial_request="Tell me a joke.",
#     request_context=initial_task_context
# )

# If no RequestContext is passed to an agent's auto_run, it might create one
# with a default log level (e.g., LogLevel.SUMMARY).
# await my_local_agent.auto_run(initial_request="What is the capital of France?")

```
The `ProgressLogger` within the agent methods uses the `log_level` from the `RequestContext`.

## Next Steps

- Explore agent concepts in more detail: `../concepts/agents.md`
- Learn about using tools with agents: `../concepts/tools.md`
- Review usage examples and tutorials: `../use-cases/index.md` and `../tutorials/index.md` (adjust paths if needed)

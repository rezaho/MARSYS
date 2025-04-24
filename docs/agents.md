# Agents Framework Documentation

This document explains the core concepts and usage of the Agent framework defined in `src/agents/agents.py`.

## 1. Introduction

### What are Agents?

Agents are the fundamental building blocks of the multi-agent system. They represent individual AI entities capable of performing tasks, communicating with each other, using tools, maintaining memory, and interacting with external environments (like web browsers).

### Why Agents?

The agent framework promotes modularity, reusability, and observability. Key design goals include:

*   **Model Flexibility:** Agents can use different underlying language models (local or API-based). Multiple agents can potentially share the same base model instance.
*   **Specialization:** Agents can be customized with unique system prompts, different memory types (conversation history, knowledge graphs), learnable components (e.g., PEFT heads), and specific tools.
*   **Structured Communication:** Provides a standardized, asynchronous way (`invoke_agent`) for agents to call each other, respecting permissions and resource limits.
*   **Context Management:** Tracks the flow of execution, call depth, and interaction counts within a task using the `RequestContext` dataclass.
*   **Progress Monitoring & Logging:** Offers a mechanism to observe agent activity and internal state via `ProgressUpdate`s and standard logging.
*   **Asynchronous Operations:** Built with `asyncio` to handle I/O-bound tasks like API calls or browser interactions efficiently.
*   **Validated Configuration:** Uses Pydantic's `ModelConfig` schema to ensure model settings are correct and consistent.

## 2. Core Concepts

### `BaseAgent` (Abstract Class)

*   **Purpose:** The foundation for all agent types. Defines the common interface and core functionalities.
*   **Key Attributes:**
    *   `model`: The underlying language model instance (`BaseLLM`, `BaseVLM`, `BaseAPIModel`).
    *   `system_prompt`: Default instructions defining the agent's base persona or goal.
    *   `tools`: Dictionary mapping tool names to callable functions (optional).
    *   `tools_schema`: OpenAI-compatible JSON schema describing the available tools (required if `tools` are provided).
    *   `allowed_peers`: A `set` of agent names this agent is permitted to call via `invoke_agent`.
    *   `name`: The unique registered name of the agent instance (assigned during registration).
    *   `communication_log`: Dictionary storing a summarized history of `invoke_agent` and `handle_invocation` calls per task ID.
    *   `memory`: An instance of `MemoryManager` handling the agent's state.
*   **Key Methods:**
    *   `handle_invocation(request, request_context)`: The primary *asynchronous* entry point for an agent to receive a request (from the user or another agent). It determines the `run_mode` (e.g., 'chat', 'think', 'plan', 'critic') based on the request structure or agent type, logs the interaction, and calls the agent's core logic in `_run`.
    *   `invoke_agent(target_agent_name, request, request_context)`: Allows an agent to *asynchronously* call another registered agent. It handles permission checks (`allowed_peers`), depth/interaction limits (`max_depth`, `max_interactions` from `request_context`), propagates a modified `RequestContext` (updating depth, count, caller/callee names), logs the invocation attempt and result/error, and returns the target agent's response.
    *   `_run(prompt, request_context, run_mode, **kwargs)` (Abstract): The core *asynchronous* logic implementation required by subclasses. This method typically involves:
        1.  Updating memory with the input `prompt`.
        2.  Selecting the appropriate system prompt based on `run_mode` (e.g., checking for `self.system_prompt_<run_mode>`).
        3.  Preparing messages for the language model using `self.memory.to_llm_format()`.
        4.  Calling the language model (`self.model.run(...)` or `self.model_instance.run(...)`) with appropriate parameters (messages, tools, `json_mode`, `temperature`, etc.), potentially influenced by `run_mode` and `kwargs`.
        5.  Updating memory with the model's response.
        6.  Performing post-processing like tool calls (if applicable for the `run_mode`, e.g., 'think') or invoking other agents based on the model's response.
        7.  Returning the final result for this step.
    *   `_log_progress(...)`: Helper for sending `ProgressUpdate`s via the `ProgressLogger`.
    *   `_add_interaction_to_log(...)`: Helper to add summarized interaction data to `communication_log`.

### `RequestContext` (Dataclass)

*   **Key Attributes:**
    *   `task_id`: Unique ID for the overall task initiated by the user/system.
    *   `initial_prompt`: The very first prompt that started the task.
    *   `progress_queue`: An `asyncio.Queue` for sending `ProgressUpdate`s back to the task initiator.
    *   `log_level`: Controls the verbosity of progress updates sent to the queue (e.g., `LogLevel.SUMMARY`, `LogLevel.DEBUG`).
    *   `max_depth`: Maximum allowed call stack depth for `invoke_agent`.
    *   `max_interactions`: Maximum total number of `invoke_agent` calls allowed within the task.
    *   `interaction_id`: Unique ID for the *current specific* agent call/interaction. Regenerated for each `invoke_agent` call.
    *   `depth`: Current call stack depth (starts at 0, increments with each `invoke_agent`).
    *   `interaction_count`: Current number of agent calls made within the task (increments with each `invoke_agent`).
    *   `caller_agent_name`: Name of the agent that made the current `invoke_agent` call (or 'user' for the initial call).
    *   `callee_agent_name`: Name of the agent whose `handle_invocation` is currently being executed.

**Example: Creating Initial RequestContext**
```python
# filepath: /path/to/your/scripts/request_context_example.py
import asyncio
from src.agents.agents import RequestContext, LogLevel

async def main():
    progress_queue = asyncio.Queue()
    initial_context = RequestContext(
        task_id="task-123",
        initial_prompt="Analyze the sentiment of this review: 'This product is amazing!'",
        progress_queue=progress_queue,
        log_level=LogLevel.DEBUG,
        max_depth=5,
        max_interactions=20,
    )
    print(f"Initial Context Task ID: {initial_context.task_id}")

# asyncio.run(main())
```

### Memory (`BaseMemory`, `ConversationMemory`, `KGMemory`, `MemoryManager`)

*   **Purpose:** Allows agents to maintain state across multiple interactions within a task.
*   **`BaseMemory`:** Abstract base class defining the memory interface (`update_memory`, `retrieve_all`, `to_llm_format`, etc.).
*   **`ConversationMemory`:** Stores interaction history as a simple list of messages (`{'role': ..., 'content': ...}`). Suitable for standard chat interactions.
*   **`KGMemory`:** Stores information as timestamped knowledge graph triplets (Subject, Predicate, Object). Requires an LLM instance (`model`) passed during initialization for its `extract_and_update_from_text` method. Useful for accumulating structured knowledge.
*   **`MemoryManager`:** A factory and wrapper. Each agent instance holds a `MemoryManager` (`self.memory`). The manager instantiates the chosen memory type (`ConversationMemory` or `KGMemory`) based on the `memory_type` string and potentially other configuration (like passing the `model` instance if required by the memory type) and delegates calls to the underlying memory module.

**Example: Using `ConversationMemory` via `MemoryManager`**

```python
# filepath: /path/to/your/scripts/memory_example.py
from src.agents.agents import MemoryManager

# Initialize MemoryManager for conversation history
memory_manager = MemoryManager(
    memory_type="conversation_history",
    system_prompt="You are a helpful AI."
)

# Update memory
memory_manager.update_memory("user", "Hello there!")
memory_manager.update_memory("assistant", "Hi! How can I help you today?")
memory_manager.update_memory("user", "What is 2+2?")

# Retrieve all messages formatted for LLM
llm_input_messages = memory_manager.to_llm_format()
print("LLM Input Messages:")
for msg in llm_input_messages:
    print(f"- {msg}")

# Retrieve the last 2 messages
recent_messages = memory_manager.retrieve_recent(2)
print("\nRecent Messages:")
for msg in recent_messages:
    print(f"- {msg}")

# Reset memory (keeps system prompt by default)
memory_manager.reset_memory()
print("\nMemory after reset:")
print(memory_manager.retrieve_all())
```

### `ModelConfig` (Pydantic Schema)

*   **Purpose:** Defines and validates the configuration for language models used by agents. Ensures required fields are present and sets defaults.
*   **Location:** Defined in `src/models/models.py`.
*   **Key Fields:**
    *   `type`: `'local'` or `'api'`.
    *   `name`: Model identifier (e.g., `'gpt-4-turbo'`, `'mistralai/Mistral-7B-Instruct-v0.1'`).
    *   `provider`: Optional API provider name (`'openai'`, `'openrouter'`, `'google'`). Used to determine `base_url` if not explicitly set.
    *   `base_url`: Specific API endpoint URL. Overrides `provider`.
    *   `api_key`: API authentication key. Reads from environment variables (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`) if not provided.
    *   `max_tokens`: Default maximum tokens for generation.
    *   `temperature`: Default sampling temperature.
    *   `model_class`: For `type='local'`, specifies `'llm'` or `'vlm'`.
    *   `torch_dtype`, `device_map`: For `type='local'`.
*   **Validation:** Automatically determines `base_url` from `provider`, reads `api_key` from environment, and ensures API models have necessary credentials.

**Example: Creating `ModelConfig` Instances**

```python
# filepath: /path/to/your/scripts/model_config_example.py
from src.models.models import ModelConfig
import os

# Example 1: OpenAI API (API key from env)
config_openai = ModelConfig(
    type="api",
    provider="openai",
    name="gpt-4-turbo",
    temperature=0.5
)
print(f"OpenAI Config Base URL: {config_openai.base_url}")
# print(f"OpenAI Config API Key: {config_openai.api_key}") # Key is read from env

# Example 2: OpenRouter API (Explicit API key)
config_openrouter = ModelConfig(
    type="api",
    provider="openrouter",
    name="anthropic/claude-3-haiku",
    api_key="YOUR_OPENROUTER_KEY_HERE", # Replace with your key
    max_tokens=2000
)
print(f"OpenRouter Config Base URL: {config_openrouter.base_url}")

# Example 3: Local LLM
config_local = ModelConfig(
    type="local",
    name="mistralai/Mistral-7B-Instruct-v0.1",
    model_class="llm",
    torch_dtype="bfloat16",
    device_map="auto"
)
print(f"Local Config Name: {config_local.name}")

# Example 4: API with custom base URL
config_custom_api = ModelConfig(
    type="api",
    name="custom-model-name",
    base_url="http://localhost:8080/v1", # Your custom endpoint
    api_key="dummy-key-if-needed"
)
print(f"Custom API Config Base URL: {config_custom_api.base_url}")

```

## 3. Agent Types

### `Agent`

*   **Purpose:** A general-purpose agent using either local models (`BaseLLM`, `BaseVLM`) or API models (`BaseAPIModel`).
*   **Initialization:** Configured via a `ModelConfig` instance which specifies model `type`, `name`, and other relevant parameters (API keys, model settings, etc.). Uses `_create_model_from_config` internally. An optional `max_tokens` parameter in the `Agent` constructor overrides the default from `ModelConfig`.
*   **Core Logic (`_run`)**: Implements the basic flow: updates memory, selects system prompt (using `system_prompt_<run_mode>` if available, otherwise `system_prompt`), prepares messages, calls the configured model (`self.model_instance.run`), updates memory with the response, and returns the result. Handles `json_mode` automatically if `run_mode == 'plan'`. Can handle tool calls if the model supports them and `tools_schema` is provided (typically used in 'think' mode or similar). Passes API-specific parameters from `ModelConfig` (and any runtime `kwargs`) to the model's `run` method.

**Example: Basic `Agent` Instantiation and Invocation**

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import os
import logging
from src.agents.agents import Agent, RequestContext, LogLevel
from src.models.models import ModelConfig # Import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the ModelConfig instance (reads API key from env)
try:
    model_config_api = ModelConfig(
        type="api",
        provider="openai", # Specify provider
        name="gpt-3.5-turbo",
        temperature=0.7,
    )
except ValueError as e:
     logging.error(f"Failed to create ModelConfig: {e}")
     # Handle error appropriately, e.g., exit or use a fallback
     exit()


# Create the agent instance using ModelConfig
basic_agent = Agent(
    agent_name="EchoBot",
    model_config=model_config_api, # Pass the config object
    system_prompt="Repeat the user's message exactly.",
    max_tokens=50 # Agent-specific override
)

async def run_echo_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="echo-01",
        initial_prompt="Hello Agent!",
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY,
    )

    try:
        logging.info(f"Invoking agent {basic_agent.name}")
        response = await basic_agent.handle_invocation(
            request="Hello Agent!",
            request_context=request_context
        )
        logging.info(f"\nAgent Response: {response}")
    except Exception as e:
        logging.error(f"\nError: {e}", exc_info=True)
    finally:
        await progress_queue.put(None)
        await progress_queue.join()

# asyncio.run(run_echo_task())
```

### `LearnableAgent`

*   **Purpose:** Extends `BaseAgent` for agents using local models (`BaseLLM`, `BaseVLM`) that might have learnable components.
*   **Initialization:** Takes a `model` instance directly (not a `ModelConfig`). Can initialize PEFT heads if `learning_head='peft'` and `learning_head_config` are provided, wrapping the base model in a `PeftHead`.
*   **Core Logic (`_run`)**: Similar to `Agent`, but calls `self.model.run` (which might be the base model or the `PeftHead` wrapper). Requires the actual model instance to be passed during initialization. Selects system prompt based on `run_mode`. Handles `json_mode` if `run_mode == 'plan'`. Passes `kwargs` to the model's `run` method.

**Example: `LearnableAgent` Instantiation (Conceptual)**

```python
# filepath: /path/to/your/scripts/learnable_agent_example.py
from src.agents.agents import LearnableAgent
from src.models.models import BaseLLM

# Assume BaseLLM is initialized correctly
local_model_instance = BaseLLM(model_name="mistralai/Mistral-7B-Instruct-v0.1")

peft_config = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}

learnable_agent = LearnableAgent(
    agent_name="Researcher",
    model=local_model_instance, # Pass the model instance directly
    system_prompt="You are a research assistant. Summarize the key points.",
    learning_head="peft",
    learning_head_config=peft_config,
    max_tokens=256
)

print(f"Learnable Agent '{learnable_agent.name}' created.")
```

### `BrowserAgent`

*   **Purpose:** Specialized `Agent` subclass for web browsing tasks. Uses different `run_mode` values ('think', 'critic') for distinct behaviors.
*   **Initialization:**
    *   Uses `ModelConfig` like `Agent`.
    *   Requires *asynchronous* creation using class methods `create` or `create_safe`. These handle `BrowserTool` initialization.
    *   Takes optional `generation_system_prompt` (used for `run_mode='think'`) and `critic_system_prompt` (used for `run_mode='critic'`).
    *   `create_safe` includes parameters like `temp_dir`, `headless_browser`, `timeout`.
    *   Dynamically generates `tools_schema` for browser actions.
*   **Core Logic (`_run`)**:
    *   Selects system prompt based on `run_mode` ('think' uses `generation_system_prompt`, 'critic' uses `critic_system_prompt`, falls back to default `system_prompt`).
    *   Calls the LLM (`self.model_instance.run`). Passes the generated `tools_schema` and `tools` *only* when `run_mode == 'think'`.
    *   **Crucially, in 'think' mode:** Parses the LLM response for tool calls. If a valid browser action is identified, it executes the corresponding method on `self.browser_tool` and updates memory with the result using the `tool` role (or `tool_error` on failure).
*   **Cleanup:** Provides an `async close_browser()` method.

**Example: `BrowserAgent` Creation and Invocation**

```python
# filepath: /path/to/your/scripts/browser_agent_creation.py
import asyncio
import os
import logging
from src.agents.agents import BrowserAgent, RequestContext, LogLevel
from src.models.models import ModelConfig # Import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create ModelConfig (reads API key from env)
try:
    model_config_browser = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4-turbo",
        temperature=0.2,
    )
except ValueError as e:
     logging.error(f"Failed to create ModelConfig: {e}")
     exit()

async def manage_browser_agent():
    browser_agent = None
    progress_queue = asyncio.Queue()
    task_id = "browser-example-01"

    try:
        logging.info("Creating BrowserAgent using create_safe...")
        browser_agent = await BrowserAgent.create_safe(
            agent_name="SafeBrowser",
            model_config=model_config_browser, # Pass ModelConfig instance
            generation_system_prompt="You control a browser. Use tools to navigate.",
            headless_browser=True,
            timeout=45
        )
        logging.info(f"BrowserAgent '{browser_agent.name}' created successfully.")

        initial_prompt = "Navigate to example.com and get the main heading."
        request_context = RequestContext(
            task_id=task_id,
            initial_prompt=initial_prompt,
            progress_queue=progress_queue,
            log_level=LogLevel.DETAILED,
            max_interactions=5
        )

        logging.info(f"Invoking BrowserAgent for task: {initial_prompt}")
        # BrowserAgent's handle_invocation implicitly uses 'think' mode if not specified
        response = await browser_agent.handle_invocation(
            request=initial_prompt,
            request_context=request_context
        )
        logging.info(f"BrowserAgent initial response/action: {response}")

    except Exception as e:
        logging.error(f"Error creating or using BrowserAgent: {e}", exc_info=True)
    finally:
        if browser_agent:
            logging.info("Closing browser...")
            await browser_agent.close_browser()
            logging.info("Browser closed.")
        await progress_queue.put(None)
        await progress_queue.join()

# asyncio.run(manage_browser_agent())
```

## 4. Defining Agent Behavior (`_run` method)

The `_run` method is the heart of an agent's custom logic. When implementing `_run` in a subclass, consider these steps:

1.  **Log Entry:** Log entry with `run_mode`.
    ```python
    await self._log_progress(request_context, LogLevel.DETAILED, f"Executing _run mode='{run_mode}'.")
    ```
2.  **Update Memory (Input):** Store the incoming `prompt`.
    ```python
    self.memory.update_memory("user", str(prompt))
    ```
3.  **Select System Prompt:** Choose based on `run_mode`.
    ```python
    system_prompt_content = getattr(self, f"system_prompt_{run_mode}", self.system_prompt)
    ```
4.  **Prepare Model Input:** Get formatted memory, ensure system prompt is correctly included.
    ```python
    llm_messages = self.memory.to_llm_format()
    # Ensure system prompt is correctly placed/updated in llm_messages
    ```
5.  **Determine Model Parameters:** Set parameters based on `run_mode` or `kwargs`. Get defaults from `self._model_config` if applicable.
    ```python
    json_mode = (run_mode == "plan")
    use_tools = (run_mode == "think")
    max_tokens_override = kwargs.get("max_tokens", self.max_tokens) # self.max_tokens already considers agent override
    temperature_override = kwargs.get("temperature", self._model_config.temperature if hasattr(self, '_model_config') else 0.7)
    model_params = {"max_tokens": max_tokens_override, "temperature": temperature_override, "json_mode": json_mode}
    # Add other API specific kwargs from self._get_api_kwargs() if needed
    api_kwargs = self._get_api_kwargs() if hasattr(self, '_get_api_kwargs') else {}
    model_params.update(api_kwargs)
    ```
6.  **Call the Model:** Execute the LLM call.
    ```python
    try:
        result = await self.model_instance.run(
             messages=llm_messages,
             tools=self.tools_schema if use_tools and self.tools_schema else None,
             tool_choice="auto" if use_tools and self.tools_schema else None,
             **model_params
        )
        # Process result (might be string or dict with content/tool_calls)
        output_content = result.get("content") if isinstance(result, dict) else str(result)
        tool_calls = result.get("tool_calls") if isinstance(result, dict) else None
        await self._log_progress(request_context, LogLevel.DEBUG, f"LLM Output: {str(result)[:100]}...", data={"tool_calls": tool_calls})
    except Exception as e:
        await self._log_progress(request_context, LogLevel.MINIMAL, f"LLM call failed: {e}", data={"error": str(e)})
        raise
    ```
7.  **Update Memory (Output):** Store the model's response content or the raw response if it contains tool calls.
    ```python
    # Store the raw response string or JSON string representation
    output_str = json.dumps(result) if isinstance(result, dict) else str(result)
    self.memory.update_memory("assistant", output_str)
    ```
8.  **Post-processing (Tool Calls / Agent Invocation):** Analyze `result`.
    *   **Tool Calls:** If `tool_calls` exist (and `run_mode` supports tools), parse them, execute functions from `self.tools`, update memory with results using role `tool` (or `tool_error`), and potentially call the LLM again. The `Agent` and `BrowserAgent` `_run` methods return the raw `result` containing tool calls, expecting a higher-level loop (like `auto_run`) to handle execution.
    *   **Agent Invocation:** If delegation is needed based on the `result`, use `await self.invoke_agent(...)`.
9.  **Return Result:** Return the final processed result (often the raw `result` from the LLM, especially if tool calls are involved).

## 5. Tool Usage

Agents can be equipped with tools (Python functions) to interact with external systems or perform specific calculations.

1.  **Define Tools:** Create Python functions that perform the desired actions.
2.  **Define Schema:** Create an OpenAI-compatible JSON schema list (`tools_schema`) that describes each function, its purpose, and its parameters. Pydantic models can be helpful here.
3.  **Provide to Agent:** Pass the dictionary of functions (`tools`) and the schema list (`tools_schema`) during agent initialization.
    ```python
    # filepath: /path/to/your/scripts/tool_agent_example.py
    import json
    from src.agents.agents import Agent
    from src.models.models import ModelConfig

    def get_current_weather(location: str, unit: str = "celsius") -> str:
        # In a real scenario, this would call a weather API
        return json.dumps({"location": location, "temperature": "25", "unit": unit, "condition": "Sunny"})

    weather_schema = [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }]

    # Assume model_config_api is defined as in previous examples
    # ...

    weather_agent = Agent(
        agent_name="WeatherBot",
        model_config=model_config_api, # Use a ModelConfig instance
        system_prompt="You are a weather bot. Use tools to find the weather.",
        tools={"get_current_weather": get_current_weather},
        tools_schema=weather_schema
    )
    print(f"Agent {weather_agent.name} created with tools.")
    ```
4.  **Handle in `_run` or `auto_run`:** When the LLM response contains `tool_calls` (and the `run_mode` supports tools):
    *   The `_run` method typically returns the raw response containing the `tool_calls`.
    *   A higher-level method like `auto_run` (or custom orchestration logic) intercepts this response.
    *   It parses the tool name and arguments from each call.
    *   Finds the corresponding function in `self.tools`.
    *   Executes the function (await if async) with the arguments, handling errors.
    *   Formats the function's return value or error.
    *   Updates the agent's memory with the tool result using the `tool` role (or `tool_error`).
    *   Calls the agent's `_run` method *again* with the history including the `tool` role message, so it can generate the final user-facing response based on the tool's output.

## 6. Logging and Monitoring

*   **Standard Logging:** The framework uses Python's built-in `logging`. Configure the root logger to control the level and format of messages printed to the console/file (e.g., `logging.basicConfig(level=logging.INFO)`).
*   **Progress Updates:** For real-time monitoring of tasks, especially in UI applications, use the `progress_queue` in the `RequestContext`. Consume `ProgressUpdate` objects from this queue asynchronously. The level of detail is controlled by the `log_level` parameter in `RequestContext`. `LogLevel.DEBUG` provides the most verbose information, including structured data.

## 7. Error Handling

*   **Agent Initialization:** Errors during `Agent` or `LearnableAgent` init are typically synchronous. `BrowserAgent.create_safe` handles internal retries/timeouts but will raise exceptions (like `TimeoutError`) on persistent failure. `ModelConfig` validation errors will also occur during initialization.
*   **Model Calls:** The `_run` method should wrap `self.model.run(...)` or `self.model_instance.run(...)` in a `try...except` block to catch model errors (API issues, runtime errors) and log them using `_log_progress`.
*   **Tool Calls:** Execution of tools within `_run` or `auto_run` should also be wrapped in `try...except`. Log errors and update memory with a `tool_error` role.
*   **Agent Invocation:** `invoke_agent` handles `PermissionError`, `ValueError` (limits), and `AgentNotFound` errors. Exceptions raised by the *target* agent's `handle_invocation` or `_run` are propagated back to the caller. The caller's `invoke_agent` call should be within a `try...except` block if specific error handling is needed.
*   **Task Execution:** The initial `handle_invocation` call (or a wrapper like `BaseCrew.run_task`) should wrap the call in `try...except`, log task failures, potentially put `None` on the progress queue, and handle or re-raise the exception.

## 8. Usage Examples (Revised & Expanded)

### Example 1: Basic API Agent

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import logging
import os
from src.agents.agents import Agent, RequestContext, LogLevel
from src.models.models import ModelConfig # Import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the ModelConfig instance (reads API key from env)
try:
    model_config_api = ModelConfig(
        type="api",
        provider="openai", # Specify provider
        name="gpt-3.5-turbo",
        temperature=0.7,
    )
except ValueError as e:
     logging.error(f"Failed to create ModelConfig: {e}")
     # Handle error appropriately, e.g., exit or use a fallback
     exit()


# Create the agent instance using ModelConfig
basic_agent = Agent(
    agent_name="EchoBot",
    model_config=model_config_api, # Pass the config object
    system_prompt="Repeat the user's message exactly.",
    max_tokens=50 # Agent-specific override
)

async def run_echo_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="echo-01",
        initial_prompt="Hello Agent!",
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY,
    )

    try:
        logging.info(f"Invoking agent {basic_agent.name}")
        response = await basic_agent.handle_invocation(
            request="Hello Agent!",
            request_context=request_context
        )
        logging.info(f"\nAgent Response: {response}")
    except Exception as e:
        logging.error(f"\nError: {e}", exc_info=True)
    finally:
        await progress_queue.put(None)
        await progress_queue.join()

# asyncio.run(run_echo_task())
```

### Example 2: Browser Agent Task

```python
# filepath: /path/to/your/scripts/browser_agent_example.py
import asyncio
import logging
import os
from src.agents.agents import BrowserAgent, RequestContext, LogLevel
from src.models.models import ModelConfig # Import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create ModelConfig (reads API key from env)
try:
    model_config_browser = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4-turbo",
        temperature=0.2,
    )
except ValueError as e:
    logging.error(f"Failed to create ModelConfig: {e}")
    exit()

async def run_browser_task():
    browser_agent = None
    progress_queue = asyncio.Queue()
    task_id = "browser-task-01"

    try:
        browser_agent = await BrowserAgent.create_safe(
            agent_name="WebNavigator",
            model_config=model_config_browser, # Pass ModelConfig instance
            generation_system_prompt="You are a web browsing assistant.",
            headless_browser=True,
            timeout=45,
            temp_dir="./browser_output"
        )

        initial_prompt = "What is the current top headline on BBC News?"
        request_context = RequestContext(
            task_id=task_id,
            initial_prompt=initial_prompt,
            progress_queue=progress_queue,
            log_level=LogLevel.DETAILED,
            max_interactions=5
        )

        # BrowserAgent's handle_invocation uses 'think' mode by default for string prompts
        response = await browser_agent.handle_invocation(
            request=initial_prompt,
            request_context=request_context
        )
        logging.info(f"Task {task_id} initial invocation completed. Response: {response}")

    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
    finally:
        if browser_agent:
            await browser_agent.close_browser()
        await progress_queue.put(None)
        await progress_queue.join()

# asyncio.run(run_browser_task())
```

### Example 3: Inter-Agent Communication (Planner & Executor)

```python
# filepath: /path/to/your/scripts/multi_agent_example.py
import asyncio
import logging
import os
import json
from src.agents.agents import Agent, RequestContext, LogLevel
from src.models.models import ModelConfig # Import ModelConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(agent_name)s] %(message)s')

# Add agent_name to log records
class AgentLogFilter(logging.Filter):
    def filter(self, record):
        record.agent_name = getattr(record, 'agent_name', 'System')
        return True

logger = logging.getLogger()
logger.addFilter(AgentLogFilter())


# --- Model Configurations using ModelConfig ---
try:
    # Use a more capable model for orchestration and synthesis
    model_config_capable = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4.1", # Use specified capable model name
        temperature=0.3,
        max_tokens=1500, # Allow more tokens for planning/synthesis
        # API key is read from env by ModelConfig
    )

    # Use a potentially faster/cheaper model for retrieval and research
    model_config_worker = ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4.1-mini", # Use specified worker model name
        temperature=0.1,
        max_tokens=500,
        # API key is read from env by ModelConfig
    )
except ValueError as e:
    logger.error(f"Failed to create ModelConfig: {e}. Ensure API keys are set in environment.")
    exit()


# --- Mock Tools for Retrieval Agent ---

def mock_google_search(query: str) -> str:
    """Mocks a Google search, returning simulated results."""
    logger.info(f"Mock Google Search for: {query}", extra={"agent_name": "Tool"})
    # Simulate finding some results with varying relevance
    results = []
    num_results = random.randint(2, 5)
    for i in range(num_results):
        relevance = random.choice(["high", "medium", "low"])
        results.append({
            "title": f"Simulated Google Result {i+1} for '{query}' ({relevance})",
            "content": f"This is simulated content about '{query}'. Relevance: {relevance}. Source: Google Search simulation.",
            "source": "Mock Google Search",
            "url": f"http://mock.google.com/search?q={query.replace(' ', '+')}&result={i+1}"
        })
    # Simulate occasional failure
    if random.random() < 0.05:
        logger.warning("Mock Google Search failed simulation.", extra={"agent_name": "Tool"})
        return json.dumps({"error": "Simulated search failure"})
    return json.dumps(results)

def mock_scholar_search(query: str) -> str:
    """Mocks a Google Scholar search, returning simulated paper results."""
    logger.info(f"Mock Scholar Search for: {query}", extra={"agent_name": "Tool"})
    results = []
    num_results = random.randint(1, 3)
    for i in range(num_results):
        year = random.randint(2018, 2023)
        results.append({
            "title": f"Mock Paper {i+1}: Aspects of '{query}' ({year})",
            "content": f"Simulated abstract discussing '{query}'. Mentions keywords like synthesis, analysis, data. Published in {year}. Source: Google Scholar simulation.",
            "source": "Mock Google Scholar",
            "url": f"http://mock.scholar.google.com/citations?view_op=view_citation&citation_for_view=mock{i+1}_{query.replace(' ', '_')}"
        })
    return json.dumps(results)

# --- Tool Schemas ---
google_search_schema = [{
    "type": "function",
    "function": {
        "name": "mock_google_search",
        "description": "Performs a simulated Google web search for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    },
}]

scholar_search_schema = [{
    "type": "function",
    "function": {
        "name": "mock_scholar_search",
        "description": "Performs a simulated Google Scholar search for academic papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query for academic papers."},
            },
            "required": ["query"],
        },
    },
}]

# Combine schemas for the Retrieval Agent
retrieval_tools = {
    "mock_google_search": mock_google_search,
    "mock_scholar_search": mock_scholar_search,
}
retrieval_tools_schema = google_search_schema + scholar_search_schema

# --- Agent Prompts ---

ORCHESTRATOR_PROMPT = """
You are a meticulous Research Orchestrator. Your goal is to manage a team of agents (Retrieval, Researcher, Synthesizer) to answer a user's complex query thoroughly.

**Your Process:**
1.  **Analyze Query:** Understand the user's request, identify key concepts, and determine the scope.
2.  **Formulate Thesis & Plan:** Break down the query into specific, answerable sub-questions. Create a step-by-step plan outlining which questions to ask and in what order. Think about potential angles and necessary background information.
3.  **Delegate Retrieval:** For each sub-question, formulate a *precise and targeted* query for the `RetrievalAgent`. Do NOT pass the original user query. Request information needed for *that specific sub-question*.
4.  **Manage Information:** Keep track of the questions asked and the information received.
5.  **Delegate Research/Validation:** Pass the specific sub-question and the *corresponding retrieved information* to the `ResearcherAgent` for validation of relevance and sufficiency *for that sub-question*.
6.  **Iterate based on Feedback:**
    *   If the `ResearcherAgent` confirms sufficiency, mark the sub-question as complete and move to the next step in your plan.
    *   If the `ResearcherAgent` identifies gaps, analyze the feedback. Formulate a *new, refined query* for the `RetrievalAgent` to address the gaps for that *same sub-question*. Repeat steps 3-6 for this sub-question.
7.  **Synthesize:** Once your plan indicates all critical sub-questions have sufficiently validated information, compile *all relevant validated information* gathered across all sub-questions. Pass the *original user query* and this *complete set of information* to the `SynthesizerAgent`.
8.  **Final Output:** Return the final report received from the `SynthesizerAgent`.

**Constraints:**
*   NEVER answer the user's query directly. Your role is orchestration.
*   NEVER call the `SynthesizerAgent` until the `ResearcherAgent` has validated sufficient information for *all* critical parts of your plan.
*   Be specific when calling other agents. Provide only the necessary context for their task.
*   Manage the flow and decide the next step. Your output should clearly indicate the next action (e.g., call RetrievalAgent with query X, call ResearcherAgent with question Y and data Z, call SynthesizerAgent with collected data). Structure your response to clearly indicate the next intended action and its parameters. For example: {"next_action": "invoke_agent", "agent_name": "RetrievalAgent", "request": "Specific query for retrieval"} or {"next_action": "synthesize", "data": collected_info}. If the process is complete, respond with {"next_action": "final_response", "response": synthesizer_output}.
*   If limits (interactions, depth) are reached, attempt to synthesize with the information gathered so far, clearly stating it might be incomplete.
"""

RETRIEVAL_PROMPT = """
You are a Retrieval Agent. Your task is to find information relevant to a *specific query* given to you, using the available search tools (mock_google_search, mock_scholar_search).

**Your Process:**
1.  Analyze the specific query you received.
2.  Choose the most appropriate tool(s) (Google for general info, Scholar for academic). You might need to use tools multiple times or refine your search based on initial results.
3.  Execute the tool(s) with precise search terms based on the query.
4.  Gather the results from the tool calls.
5.  Format the results as a JSON list of objects, where each object has keys: "title", "content", "source", "url".
    Example: `[{"title": "...", "content": "...", "source": "Mock Google Search", "url": "..."}, ...]`
6.  Return ONLY the JSON list. Do not add explanations or summaries outside the JSON structure. If no relevant information is found after searching, return an empty list `[]`. If a tool fails, report the error within the JSON if possible or return an error message.
"""

RESEARCHER_PROMPT = """
You are a critical Researcher Agent. Your task is to evaluate if the provided information *sufficiently and relevantly* answers a *specific question* you are given. You are NOT answering the original user query.

**Your Input:**
1.  A specific question.
2.  A list of information snippets (dictionaries with 'title', 'content', 'source', 'url') supposedly retrieved to answer that question.

**Your Process:**
1.  Carefully read the specific question.
2.  Analyze each information snippet provided. Is the 'content' relevant to the specific question?
3.  Assess if the *combined* relevant information is *sufficient* to answer the specific question thoroughly.
4.  Provide a clear assessment:
    *   **If Sufficient:** Respond with a JSON object: `{"status": "sufficient", "reason": "The provided information adequately addresses the specific question regarding X and Y."}`
    *   **If Insufficient:** Respond with a JSON object clearly stating *what is missing*. `{"status": "insufficient", "reason": "The information is relevant but lacks details on Z.", "missing_info_request": "Need specific examples of Z implementation."}`
    *   **If Irrelevant:** Respond with a JSON object: `{"status": "irrelevant", "reason": "The provided information discusses A, but the question was about B."}`

**Constraints:**
*   Focus *only* on the specific question and the provided information snippets.
*   Be precise in your reasoning, especially when information is insufficient.
*   Return ONLY the JSON assessment object.
"""

SYNTHESIZER_PROMPT = """
You are a Synthesizer Agent. Your task is to write a comprehensive, well-structured report answering the *original user query*, based *only* on the validated information provided to you.

**Your Input:**
1.  The original user query.
2.  A collection of validated information snippets (likely a list of lists or a combined list of dictionaries with 'title', 'content', 'source', 'url').

**Your Process:**
1.  Deeply analyze the original user query to understand the core requirements.
2.  Review all the provided information snippets.
3.  Formulate your own thesis or structure for the report based on the query and the available evidence. Organize the information logically.
4.  Write a clear, coherent, and comprehensive report that directly addresses the user's query.
5.  Ground your report *strictly* in the provided information. Do NOT add external knowledge or hallucinate. You can cite sources implicitly by mentioning findings from different sources.
6.  Ensure the report flows well and is easy to understand.
7.  Return *only* the final report as a string. Start the report directly, without introductory phrases like "Here is the report:".
"""

# --- Agent Initialization ---

orchestrator_agent = Agent(
    agent_name="Orchestrator",
    model_config=model_config_capable, # Use capable model config
    system_prompt=ORCHESTRATOR_PROMPT,
    allowed_peers=["RetrievalAgent", "ResearcherAgent", "SynthesizerAgent"],
    memory_type="conversation_history", # Keeps track of the conversation flow
)

retrieval_agent = Agent(
    agent_name="RetrievalAgent",
    model_config=model_config_worker, # Use worker model config
    system_prompt=RETRIEVAL_PROMPT,
    tools=retrieval_tools,
    tools_schema=retrieval_tools_schema,
    allowed_peers=[], # Cannot call other agents
    memory_type="conversation_history", # Remembers past retrieval attempts in a task if needed
)

researcher_agent = Agent(
    agent_name="ResearcherAgent",
    model_config=model_config_worker, # Use worker model config
    system_prompt=RESEARCHER_PROMPT,
    allowed_peers=[],
    memory_type="conversation_history", # Simple memory is likely sufficient
)

synthesizer_agent = Agent(
    agent_name="SynthesizerAgent",
    model_config=model_config_capable, # Use capable model config
    system_prompt=SYNTHESIZER_PROMPT,
    allowed_peers=[],
    memory_type="conversation_history", # Needs to see the prompt + data
)

# --- Main Execution Logic ---

async def run_deep_research_task(user_query: str, max_iterations: int = 5):
    """
    Runs the deep research multi-agent system.

    Args:
        user_query: The initial query from the user.
        max_iterations: Max number of Orchestrator -> Retrieve -> Research cycles.
    """
    task_id = f"deep-research-{uuid.uuid4()}"
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id=task_id,
        initial_prompt=user_query,
        progress_queue=progress_queue,
        log_level=LogLevel.DETAILED,
        max_depth=4, # Orchestrator -> Worker -> Tool (if applicable)
        max_interactions=max_iterations * 3 + 2, # Rough estimate: Plan + N*(Retrieve+Research) + Synthesize
    )

    # Start progress monitor
    async def progress_monitor(q: asyncio.Queue):
        while True:
            update = await q.get()
            if update is None:
                q.task_done()
                break
            log_data_str = f" Data: {json.dumps(update.data)}" if update.data else ""
            logger.info(
                f"[Progress L{update.level.value}] {update.message}{log_data_str}",
                extra={"agent_name": update.agent_name or "System"}
            )
            q.task_done()

    monitor_task = asyncio.create_task(progress_monitor(progress_queue))

    logger.info(f"--- Starting Deep Research Task {task_id} ---", extra={"agent_name": "System"})
    logger.info(f"User Query: {user_query}", extra={"agent_name": "System"})

    current_orchestrator_request: Any = user_query
    all_validated_data = {} # Store validated data keyed by sub-question
    iteration = 0
    final_report = "Error: Research process did not complete."

    try:
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Orchestrator Iteration {iteration} ---", extra={"agent_name": "System"})

            # 1. Call Orchestrator to decide next step
            orchestrator_response_raw = await orchestrator_agent.handle_invocation(
                request=current_orchestrator_request,
                request_context=request_context
            )

            # Parse Orchestrator's decision
            try:
                # Expecting JSON like {"next_action": "invoke_agent", "agent_name": "...", "request": ...}
                # or {"next_action": "synthesize", "data": ...}
                # or {"next_action": "final_response", "response": ...}
                decision = json.loads(orchestrator_response_raw)
                next_action = decision.get("next_action")
                logger.info(f"Orchestrator decision: {next_action}", extra={"agent_name": "Orchestrator"})
            except json.JSONDecodeError:
                logger.error(f"Orchestrator returned non-JSON response: {orchestrator_response_raw}", extra={"agent_name": "Orchestrator"})
                final_report = "Error: Orchestrator failed to provide a valid next step."
                break

            # --- Execute based on Orchestrator's decision ---

            if next_action == "invoke_agent":
                target_agent_name = decision.get("agent_name")
                request_payload = decision.get("request")

                if not target_agent_name or request_payload is None:
                    logger.error(f"Orchestrator invoke decision missing agent_name or request.", extra={"agent_name": "Orchestrator"})
                    final_report = "Error: Orchestrator provided invalid invocation details."
                    break

                # --- Call Retrieval Agent ---
                if target_agent_name == "RetrievalAgent":
                    if not isinstance(request_payload, str):
                         logger.error(f"Orchestrator request to RetrievalAgent is not a string query.", extra={"agent_name": "Orchestrator"})
                         final_report = "Error: Invalid query format for RetrievalAgent."
                         break
                    current_question = request_payload # Store the question asked
                    logger.info(f"Orchestrator requests Retrieval for: {current_question}", extra={"agent_name": "Orchestrator"})
                    retrieved_data_raw = await orchestrator_agent.invoke_agent(
                        target_agent_name="RetrievalAgent",
                        request=current_question, # Pass the specific query string
                        request_context=request_context,
                    )
                    # Prepare for Orchestrator's next step (which should involve calling Researcher)
                    current_orchestrator_request = {
                        "action": "process_retrieval_result",
                        "question": current_question,
                        "retrieved_data": retrieved_data_raw
                    }

                # --- Call Researcher Agent ---
                # This happens implicitly when the orchestrator gets the retrieval result back
                # The orchestrator's *next* call should be to the researcher based on its logic
                elif target_agent_name == "ResearcherAgent":
                     if not isinstance(request_payload, dict) or "question" not in request_payload or "data" not in request_payload:
                         logger.error(f"Orchestrator request to ResearcherAgent has invalid format.", extra={"agent_name": "Orchestrator"})
                         final_report = "Error: Invalid request format for ResearcherAgent."
                         break

                     question_for_researcher = request_payload["question"]
                     data_for_researcher = request_payload["data"] # Should be the list from Retrieval

                     logger.info(f"Orchestrator requests Researcher to validate info for: {question_for_researcher}", extra={"agent_name": "Orchestrator"})
                     researcher_assessment_raw = await orchestrator_agent.invoke_agent(
                         target_agent_name="ResearcherAgent",
                         request={"question": question_for_researcher, "data": data_for_researcher},
                         request_context=request_context,
                     )
                     # Prepare for Orchestrator's next planning step
                     current_orchestrator_request = {
                         "action": "process_researcher_assessment",
                         "question": question_for_researcher,
                         "assessment": researcher_assessment_raw,
                         "retrieved_data": data_for_researcher # Pass back the data that was assessed
                     }
                     # Store validated data if assessment is sufficient
                     try:
                         assessment_json = json.loads(researcher_assessment_raw)
                         if assessment_json.get("status") == "sufficient":
                             logger.info(f"Researcher deemed info sufficient for: {question_for_researcher}", extra={"agent_name": "ResearcherAgent"})
                             if question_for_researcher not in all_validated_data:
                                 all_validated_data[question_for_researcher] = []
                             # Attempt to parse retrieved data if it's a string
                             parsed_data = []
                             if isinstance(data_for_researcher, str):
                                 try:
                                     parsed_data = json.loads(data_for_researcher)
                                 except json.JSONDecodeError:
                                     logger.warning(f"Could not parse retrieved data for storage: {data_for_researcher}", extra={"agent_name": "System"})
                             elif isinstance(data_for_researcher, list):
                                 parsed_data = data_for_researcher

                             if isinstance(parsed_data, list):
                                all_validated_data[question_for_researcher].extend(parsed_data)
                             else:
                                logger.warning(f"Retrieved data was not a list after parsing: {type(parsed_data)}", extra={"agent_name": "System"})

                     except json.JSONDecodeError:
                         logger.warning(f"Researcher assessment was not valid JSON: {researcher_assessment_raw}", extra={"agent_name": "ResearcherAgent"})


                # --- Call Synthesizer Agent ---
                elif target_agent_name == "SynthesizerAgent":
                    if not isinstance(request_payload, dict) or "user_query" not in request_payload or "validated_data" not in request_payload:
                         logger.error(f"Orchestrator request to SynthesizerAgent has invalid format.", extra={"agent_name": "Orchestrator"})
                         final_report = "Error: Invalid request format for SynthesizerAgent."
                         break

                    logger.info(f"Orchestrator requests Synthesizer to generate report.", extra={"agent_name": "Orchestrator"})
                    final_report_raw = await orchestrator_agent.invoke_agent(
                        target_agent_name="SynthesizerAgent",
                        request=request_payload, # Contains user_query and validated_data
                        request_context=request_context,
                    )
                    # Process is complete, set final report
                    final_report = final_report_raw
                    logger.info(f"Synthesizer finished report.", extra={"agent_name": "SynthesizerAgent"})
                    # Tell orchestrator loop to finish
                    current_orchestrator_request = {"action": "final_response", "response": final_report}
                    break # Exit loop after synthesis

                else:
                    logger.error(f"Orchestrator requested unknown agent: {target_agent_name}", extra={"agent_name": "Orchestrator"})
                    final_report = f"Error: Orchestrator requested unknown agent '{target_agent_name}'."
                    break

            elif next_action == "synthesize":
                 # Orchestrator decided to synthesize based on its internal state
                 data_for_synthesis = decision.get("data", all_validated_data) # Use accumulated data
                 logger.info(f"Orchestrator requests Synthesizer to generate report.", extra={"agent_name": "Orchestrator"})
                 final_report_raw = await orchestrator_agent.invoke_agent(
                     target_agent_name="SynthesizerAgent",
                     request={"user_query": user_query, "validated_data": data_for_synthesis},
                     request_context=request_context,
                 )
                 final_report = final_report_raw
                 logger.info(f"Synthesizer finished report.", extra={"agent_name": "SynthesizerAgent"})
                 break # Exit loop

            elif next_action == "final_response":
                # Orchestrator indicates completion (e.g., after getting report from synthesizer)
                final_report = decision.get("response", "Error: Orchestrator indicated completion but provided no response.")
                logger.info(f"Orchestrator indicated final response received.", extra={"agent_name": "Orchestrator"})
                break # Exit loop

            else:
                logger.error(f"Orchestrator returned unknown next_action: {next_action}", extra={"agent_name": "Orchestrator"})
                final_report = f"Error: Orchestrator returned unknown action '{next_action}'."
                break

            # Check interaction limits (safeguard)
            if request_context.interaction_count >= request_context.max_interactions:
                logger.warning(f"Max interactions ({request_context.max_interactions}) reached. Attempting synthesis with gathered data.", extra={"agent_name": "System"})
                try:
                    final_report_raw = await orchestrator_agent.invoke_agent(
                         target_agent_name="SynthesizerAgent",
                         request={"user_query": user_query, "validated_data": all_validated_data},
                         request_context=request_context,
                    )
                    final_report = f"[Warning: Max interactions reached]\n{final_report_raw}"
                except Exception as synth_error:
                    logger.error(f"Synthesis attempt after max interactions failed: {synth_error}", extra={"agent_name": "System"})
                    final_report = "Error: Max interactions reached, and final synthesis failed."
                break


        if iteration >= max_iterations and "Error" in final_report:
             logger.warning(f"Max iterations ({max_iterations}) reached without successful synthesis.", extra={"agent_name": "System"})
             final_report = "Error: Research process timed out after maximum iterations."


    except Exception as e:
        logger.error(f"An error occurred during the research task: {e}", exc_info=True, extra={"agent_name": "System"})
        final_report = f"Error: An unexpected error occurred during the research process: {e}"
    finally:
        logger.info(f"--- Deep Research Task {task_id} Finished ---", extra={"agent_name": "System"})
        # Signal monitor to stop and wait
        await progress_queue.put(None)
        await monitor_task

    return final_report


# --- Example Usage ---
if __name__ == "__main__":
    # Example Query:
    query = "What are the latest advancements in using synthetic data for training large language models, focusing on efficiency and quality?"

    # Run the task
    final_result = asyncio.run(run_deep_research_task(query, max_iterations=5))

    print("\n" + "="*30 + " Final Research Report " + "="*30)
    print(final_result)
    print("="*80)
```

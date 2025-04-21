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
    *   `handle_invocation(request, request_context)`: The primary *asynchronous* entry point for an agent to receive a request (from the user or another agent). It determines the `run_mode` (e.g., 'chat', 'think', 'plan') based on the request structure or agent type, logs the interaction, and calls the agent's core logic in `_run`.
    *   `invoke_agent(target_agent_name, request, request_context)`: Allows an agent to *asynchronously* call another registered agent. It handles permission checks (`allowed_peers`), depth/interaction limits (`max_depth`, `max_interactions` from `request_context`), propagates a modified `RequestContext` (updating depth, count, caller/callee names), logs the invocation attempt and result/error, and returns the target agent's response.
    *   `_run(prompt, request_context, run_mode, **kwargs)` (Abstract): The core *asynchronous* logic implementation required by subclasses. This method typically involves:
        1.  Updating memory with the input `prompt`.
        2.  Selecting the appropriate system prompt based on `run_mode`.
        3.  Preparing messages for the language model using `self.memory.to_llm_format()`.
        4.  Calling the language model (`self.model.run(...)`) with appropriate parameters (messages, tools, `json_mode`, etc.).
        5.  Updating memory with the model's response.
        6.  Performing post-processing like tool calls or invoking other agents based on the model's response.
        7.  Returning the final result for this step.
    *   `_log_progress(...)`: Helper for sending `ProgressUpdate`s via the `ProgressLogger`.
    *   `_add_interaction_to_log(...)`: Helper to add summarized interaction data to `communication_log`.

### `RequestContext` (Dataclass)

*   **Purpose:** Carries essential information about the current state of a task's execution flow as it passes between agents during `invoke_agent` calls.
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
        log_level=LogLevel.DEBUG, # Set desired verbosity
        max_depth=5,
        max_interactions=20,
    )
    print(f"Initial Context Task ID: {initial_context.task_id}")
    # In a real scenario, this context would be passed to the first agent's handle_invocation

# asyncio.run(main())
```

### `ProgressLogger` and `ProgressUpdate`

*   **Purpose:** Provide a structured way for agents to report their status, actions, and internal data during task execution, primarily for monitoring and debugging.
*   **`ProgressUpdate`:** A dataclass containing:
    *   `timestamp`: Time of the update.
    *   `level`: Severity (`LogLevel`).
    *   `message`: Human-readable status message.
    *   `task_id`, `interaction_id`, `agent_name`: Context identifiers.
    *   `data`: Optional dictionary for structured data (e.g., LLM input/output snippets, tool arguments).
*   **`ProgressLogger.log(...)`:** An *async* static method used internally by agents (via `_log_progress`) to create a `ProgressUpdate` and put it onto the `RequestContext`'s `progress_queue` if the update's level meets the context's `log_level`. It also logs to standard Python logging as a fallback or for different verbosity levels.

**Example: Consuming Progress Updates**

```python
# filepath: /path/to/your/scripts/progress_consumer_example.py
import asyncio
import logging
from src.agents.agents import ProgressUpdate, LogLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def consume_progress(progress_queue: asyncio.Queue[Optional[ProgressUpdate]]):
    """Consumes and logs progress updates from a queue."""
    logging.info("Progress consumer started.")
    while True:
        update = await progress_queue.get()
        if update is None: # Sentinel value indicates the end
            logging.info("Received end signal. Progress consumer finished.")
            break

        # Log based on the update level
        log_level_map = {
            LogLevel.MINIMAL: logging.INFO,
            LogLevel.SUMMARY: logging.INFO,
            LogLevel.DETAILED: logging.INFO, # Or logging.DEBUG
            LogLevel.DEBUG: logging.DEBUG,
        }
        std_log_level = log_level_map.get(update.level, logging.INFO)

        log_msg = (
            f"PROGRESS [Task:{update.task_id}] "
            f"[Agent:{update.agent_name or 'N/A'}] "
            f"({update.level.name}): {update.message}"
        )
        if update.data:
            log_msg += f" | Data: {str(update.data)[:150]}" # Truncate data

        logging.log(std_log_level, log_msg)
        progress_queue.task_done() # Mark task as done for queue.join()

async def run_mock_task():
    """Simulates a task producing progress updates."""
    progress_queue = asyncio.Queue()
    consumer_task = asyncio.create_task(consume_progress(progress_queue))

    # Simulate agent work and sending updates
    await asyncio.sleep(0.1)
    await progress_queue.put(ProgressUpdate(time.time(), LogLevel.SUMMARY, "Task started", "task-sim-01", agent_name="AgentA"))
    await asyncio.sleep(0.2)
    await progress_queue.put(ProgressUpdate(time.time(), LogLevel.DETAILED, "Processing step 1", "task-sim-01", agent_name="AgentA", data={"step": 1, "value": 10}))
    await asyncio.sleep(0.3)
    await progress_queue.put(ProgressUpdate(time.time(), LogLevel.DEBUG, "Internal calculation", "task-sim-01", agent_name="AgentA", data={"temp_result": 42}))
    await asyncio.sleep(0.1)
    await progress_queue.put(ProgressUpdate(time.time(), LogLevel.SUMMARY, "Task finished", "task-sim-01", agent_name="AgentA"))

    # Signal end of task
    await progress_queue.put(None)

    # Wait for consumer to finish processing all items
    await progress_queue.join()
    await consumer_task

# asyncio.run(run_mock_task())
```

### `AgentRegistry`

*   **Purpose:** A central, thread-safe dictionary holding *weak references* to all active agent instances, keyed by their unique names.
*   **Functionality:**
    *   `register(agent, name, prefix)`: Adds an agent. Assigns a unique name if `name` is None. Uses weak references to avoid preventing garbage collection.
    *   `unregister(name)`: Removes an agent.
    *   `get(name)`: Retrieves an agent instance by name, returning `None` if not found or if the agent has been garbage collected. Used by `invoke_agent`.
    *   `all()`: Returns a dictionary of all currently registered agents.

### Memory (`BaseMemory`, `ConversationMemory`, `KGMemory`, `MemoryManager`)

*   **Purpose:** Allows agents to maintain state across multiple interactions within a task.
*   **`BaseMemory`:** Abstract base class defining the memory interface (`update_memory`, `retrieve_all`, `to_llm_format`, etc.).
*   **`ConversationMemory`:** Stores interaction history as a simple list of messages (`{'role': ..., 'content': ...}`). Suitable for standard chat interactions.
*   **`KGMemory`:** Stores information as timestamped knowledge graph triplets (Subject, Predicate, Object). Requires an LLM instance (`model`) passed during initialization for its `extract_and_update_from_text` method. Useful for accumulating structured knowledge.
*   **`MemoryManager`:** A factory and wrapper. Each agent instance holds a `MemoryManager` (`self.memory`). The manager instantiates the chosen memory type (`ConversationMemory` or `KGMemory`) and delegates calls to the underlying memory module.

**Example: Using `ConversationMemory` via `MemoryManager`**

```python
# filepath: /path/to/your/scripts/memory_example.py
from src.agents.agents import MemoryManager

# Initialize MemoryManager for conversation history
memory_manager = MemoryManager(memory_type="conversation_history", system_prompt="You are a helpful AI.")

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

## 3. Agent Types

### `Agent`

*   **Purpose:** A general-purpose agent using either local models (`BaseLLM`, `BaseVLM`) or API models (`BaseAPIModel`).
*   **Initialization:** Configured via `model_config` dictionary which specifies model `type` ('local' or 'api'), `name`, and other relevant parameters (API keys, local model settings, etc.). See `_create_model_from_config` for details.
*   **Core Logic (`_run`)**: Implements the basic flow: updates memory, selects system prompt (can use `system_prompt_<run_mode>` attributes as overrides), prepares messages, calls the configured model (`self.model_instance.run`), updates memory with the response, and returns the result. Handles `json_mode` based on `run_mode` ('plan'). Can handle tool calls if the model supports them and `tools_schema` is provided.

**Example: Basic `Agent` Instantiation and Invocation**

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import os
from src.agents.agents import Agent, RequestContext, LogLevel

# Assume OPENAI_API_KEY is set in environment
model_config_api = {
    "type": "api",
    "name": "gpt-3.5-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,
}

# Create the agent instance
basic_agent = Agent(
    agent_name="EchoBot",
    model_config=model_config_api,
    system_prompt="Repeat the user's message exactly.",
    max_tokens=50
)

async def run_echo_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="echo-01",
        initial_prompt="Hello Agent!",
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY,
        interaction_id="int-echo-001",
        caller_agent_name="user",
        callee_agent_name=basic_agent.name
    )

    # Consume progress updates (simplified logging)
    async def log_progress():
        while True:
            update = await progress_queue.get()
            if update is None: break
            print(f"PROGRESS: {update.message}")
            progress_queue.task_done()
    log_task = asyncio.create_task(log_progress())

    # Invoke the agent
    try:
        response = await basic_agent.handle_invocation(
            request="Hello Agent!",
            request_context=request_context
        )
        print(f"\nAgent Response: {response}")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await progress_queue.put(None) # Signal end
        await progress_queue.join()
        await log_task

# asyncio.run(run_echo_task())
```

### `LearnableAgent`

*   **Purpose:** Extends `BaseAgent` for agents using local models (`BaseLLM`, `BaseVLM`) that might have learnable components.
*   **Initialization:** Takes a `model` instance directly. Can initialize PEFT heads if `learning_head='peft'` and `learning_head_config` are provided, wrapping the base model in a `PeftHead`.
*   **Core Logic (`_run`)**: Similar to `Agent`, but calls `self.model.run` (which might be the base model or the `PeftHead` wrapper). Requires the actual model instance to be passed during initialization (unlike `Agent` which uses `model_config`).

**Example: `LearnableAgent` Instantiation (Conceptual)**

```python
# filepath: /path/to/your/scripts/learnable_agent_example.py
from src.agents.agents import LearnableAgent
from src.models.models import BaseLLM # Assuming a local model class

# --- This part is application-specific ---
# Assume 'load_my_local_model' loads a Hugging Face model or similar
def load_my_local_model(model_name_or_path: str) -> BaseLLM:
    # Replace with your actual model loading logic
    print(f"Loading local model: {model_name_or_path}...")
    # Example: return BaseLLM(model_name=model_name_or_path, device_map="auto")
    class MockLocalLLM(BaseLLM): # Mock for example purposes
        def __init__(self, model_name, **kwargs): super().__init__(model_name, **kwargs)
        def run(self, messages, **kwargs): return f"Mock response for: {messages[-1]['content']}"
    return MockLocalLLM(model_name=model_name_or_path)
# --- End application-specific part ---

# Load the base model instance
local_model_instance = load_my_local_model("mistralai/Mistral-7B-Instruct-v0.1") # Example model

# Configuration for PEFT (if used)
peft_config = {
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"], # Example target modules
    # "peft_model_id": "path/to/your/adapter/weights" # Optional: Load existing adapter
}

# Create the LearnableAgent instance
learnable_agent = LearnableAgent(
    agent_name="Researcher",
    model=local_model_instance, # Pass the loaded model instance
    system_prompt="You are a research assistant. Summarize the key points.",
    learning_head="peft", # Specify PEFT head
    learning_head_config=peft_config, # Provide PEFT config
    max_tokens=256
)

print(f"Learnable Agent '{learnable_agent.name}' created.")
# The agent's self.model is now potentially a PeftHead wrapper
# You would invoke it similarly to the basic Agent example using handle_invocation

# Example conceptual invocation (requires async context)
# async def run_learnable_task():
#     # ... create RequestContext ...
#     response = await learnable_agent.handle_invocation("Summarize the theory of relativity.", request_context)
#     print(response)
```

### `BrowserAgent`

*   **Purpose:** Specialized `Agent` subclass for web browsing tasks.
*   **Initialization:**
    *   Uses `model_config` like `Agent`.
    *   Requires *asynchronous* creation using class methods `create` or `create_safe` (recommended for robustness). These methods handle the initialization of the underlying `BrowserTool`.
    *   Takes optional `generation_system_prompt` (for 'think' mode) and `critic_system_prompt` (for 'critic' mode).
    *   `create_safe` includes parameters like `temp_dir`, `headless_browser`, and `timeout` for browser setup.
    *   Dynamically generates `tools_schema` based on the available methods/Pydantic models in `src.environment.web_browser`.
*   **Core Logic (`_run`)**:
    *   Selects system prompt based on `run_mode` ('think', 'critic', or default).
    *   Calls the LLM, passing the generated `tools_schema` only when `run_mode == 'think'`.
    *   **Crucially, in 'think' mode:** Parses the LLM response for tool calls (preferred) or specific command patterns (fallback). If a valid browser action is identified, it executes the corresponding method on the `self.browser_tool` instance (via `self.browser_methods`) and updates memory with the result (`tool_result` or `tool_error` role).
*   **Cleanup:** Provides an `async close_browser()` method that must be called to shut down the browser instance gracefully.

**Example: `BrowserAgent` Creation and Cleanup**

```python
# filepath: /path/to/your/scripts/browser_agent_creation.py
import asyncio
import os
from src.agents.agents import BrowserAgent

# Assume OPENAI_API_KEY is set
model_config_browser = {
    "type": "api",
    "name": "gpt-4-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

async def manage_browser_agent():
    browser_agent = None
    try:
        print("Creating BrowserAgent using create_safe...")
        browser_agent = await BrowserAgent.create_safe(
            agent_name="SafeBrowser",
            model_config=model_config_browser,
            generation_system_prompt="You control a browser. Use tools to navigate.",
            headless_browser=True,
            timeout=45 # Timeout for initialization
        )
        print(f"BrowserAgent '{browser_agent.name}' created successfully.")
        print(f"Generated Tools Schema: {browser_agent.tools_schema}")

        # --- Agent is ready to be used here ---
        # Example: await browser_agent.handle_invocation(...)

    except Exception as e:
        print(f"Error creating or using BrowserAgent: {e}")
    finally:
        if browser_agent:
            print("Closing browser...")
            await browser_agent.close_browser()
            print("Browser closed.")

# asyncio.run(manage_browser_agent())
```

## 4. Defining Agent Behavior (`_run` method)

The `_run` method is the heart of an agent's custom logic. When implementing `_run` in a subclass, consider these steps:

1.  **Log Entry:** Start by logging the entry into the method with relevant context.
    ```python
    await self._log_progress(request_context, LogLevel.DETAILED, f"Executing _run mode='{run_mode}'.")
    ```
2.  **Update Memory (Input):** Store the incoming prompt in the agent's memory.
    ```python
    user_prompt = str(prompt)
    self.memory.update_memory("user", user_prompt)
    ```
3.  **Select System Prompt:** Choose the system prompt based on `run_mode`.
    ```python
    system_prompt_content = getattr(self, f"system_prompt_{run_mode}", self.system_prompt)
    ```
4.  **Prepare Model Input:** Get formatted memory and ensure the correct system prompt is included.
    ```python
    llm_messages = self.memory.to_llm_format()
    # Ensure system prompt is correctly set/prepended in a copy
    llm_messages_copy = ... # Prepare messages list
    ```
5.  **Determine Model Parameters:** Set `json_mode`, `max_tokens`, `temperature`, etc., based on `run_mode` or `kwargs`. Check for `tools_schema` if applicable.
    ```python
    json_mode = (run_mode == "plan")
    max_tokens_override = kwargs.get("max_tokens", self.max_tokens)
    ```
6.  **Call the Model:** Execute the LLM call within a `try...except` block for error handling.
    ```python
    try:
        result = await self.model.run(messages=llm_messages_copy, ..., tools=self.tools_schema if use_tools else None)
        output_str = str(result) # Or json.dumps if needed
        await self._log_progress(request_context, LogLevel.DEBUG, f"LLM Output: {output_str[:100]}...")
    except Exception as e:
        await self._log_progress(request_context, LogLevel.MINIMAL, f"LLM call failed: {e}", data={"error": str(e)})
        raise
    ```
7.  **Update Memory (Output):** Store the model's response.
    ```python
    self.memory.update_memory("assistant", output_str)
    ```
8.  **Post-processing (Tool Calls / Agent Invocation):** Analyze `result`.
    *   **Tool Calls:** If `result` contains tool calls (e.g., `result.get("tool_calls")`), parse them, execute the corresponding functions from `self.tools`, and update memory with the results (using a specific role like `tool_result`).
    *   **Agent Invocation:** If the agent needs to delegate work, use `await self.invoke_agent(...)`, passing the `request_context`.
9.  **Return Result:** Return the processed result or the raw LLM output.

## 5. Tool Usage

Agents can be equipped with tools (Python functions) to interact with external systems or perform specific calculations.

1.  **Define Tools:** Create Python functions that perform the desired actions.
2.  **Define Schema:** Create an OpenAI-compatible JSON schema list (`tools_schema`) that describes each function, its purpose, and its parameters. Pydantic models can be helpful here (as seen in `BrowserAgent._generate_tool_schema`).
3.  **Provide to Agent:** Pass the dictionary of functions (`tools`) and the schema list (`tools_schema`) during agent initialization.
    ```python
    def get_current_weather(location: str, unit: str = "celsius") -> str:
        # ... implementation ...
        return f"The weather in {location} is..."

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

    weather_agent = Agent(
        # ... model_config, system_prompt ...
        tools={"get_current_weather": get_current_weather},
        tools_schema=weather_schema
    )
    ```
4.  **Handle in `_run`:** When the LLM response indicates a tool call (e.g., in `result.get("tool_calls")`), the `_run` method must:
    *   Parse the tool name and arguments.
    *   Find the corresponding function in `self.tools`.
    *   Execute the function with the arguments.
    *   Format the function's return value.
    *   Update the agent's memory with the tool result (e.g., using role `tool_result`).
    *   Potentially call the LLM *again* with the tool result included in the history, so it can generate the final user-facing response.

## 6. Logging and Monitoring

*   **Standard Logging:** The framework uses Python's built-in `logging`. Configure the root logger to control the level and format of messages printed to the console/file (e.g., `logging.basicConfig(level=logging.INFO)`).
*   **Progress Updates:** For real-time monitoring of tasks, especially in UI applications, use the `progress_queue` returned by `BaseCrew.run_task`. Consume `ProgressUpdate` objects from this queue asynchronously. The level of detail is controlled by the `log_level` parameter passed to `run_task`. `LogLevel.DEBUG` provides the most verbose information, including structured data.

## 7. Error Handling

*   **Agent Initialization:** Errors during `Agent` or `LearnableAgent` init are typically synchronous. `BrowserAgent.create_safe` handles internal retries/timeouts but will raise exceptions (like `TimeoutError`) on persistent failure.
*   **Model Calls:** The `_run` method should wrap `self.model.run(...)` in a `try...except` block to catch model errors (API issues, runtime errors) and log them using `_log_progress`.
*   **Tool Calls:** Execution of tools within `_run` should also be wrapped in `try...except` to handle errors gracefully (e.g., invalid arguments, external service failures). Log errors and update memory with a `tool_error` role.
*   **Agent Invocation:** `invoke_agent` handles `PermissionError`, `ValueError` (limits), and `AgentNotFound` errors. Exceptions raised by the *target* agent's `handle_invocation` or `_run` are propagated back to the caller. The caller's `invoke_agent` call should be within a `try...except` block if specific error handling is needed.
*   **Task Execution:** `BaseCrew.run_task` wraps the initial agent call in `try...except`, logs task failures, puts `None` on the progress queue, and re-raises the exception.

## 8. Usage Examples (Revised & Expanded)

### Example 1: Basic API Agent

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import logging
import os
from src.agents.agents import Agent, RequestContext, LogLevel, ProgressUpdate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Agent Definition ---
# Use environment variables for API keys
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

model_config_gpt4 = {
    "type": "api",
    "name": "gpt-4-turbo-preview", # Or another suitable model
    "api_key": api_key,
    "temperature": 0.7, # Example API parameter
}

responder_agent = Agent(
    agent_name="HelpfulResponder",
    model_config=model_config_gpt4,
    system_prompt="You are a concise and helpful assistant.",
    max_tokens=150
)

# --- Task Execution Logic ---
async def process_progress_queue(queue: asyncio.Queue[Optional[ProgressUpdate]]):
    """Asynchronously consumes progress updates from the queue."""
    while True:
        update = await queue.get()
        if update is None: # Sentinel indicates task completion
            logging.info("Progress queue finished.")
            break
        log_msg = (
            f"PROGRESS: [Task:{update.task_id}]"
            f"[Interaction:{update.interaction_id or 'N/A'}]"
            f"[Agent:{update.agent_name or 'N/A'}]"
            f"[{update.level.name}] {update.message}"
        )
        if update.data:
            try:
                data_str = json.dumps(update.data)
                log_msg += f" Data: {data_str[:200]}" # Log truncated data
                if len(data_str) > 200: log_msg += "..."
            except TypeError:
                 log_msg += " Data: [Unserializable]"
        logging.info(log_msg)
        queue.task_done() # Notify queue that item processing is complete

async def run_responder_task():
    """Runs a simple task with the responder agent."""
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="responder-task-01",
        initial_prompt="What is the capital of Canada?",
        progress_queue=progress_queue,
        log_level=LogLevel.DEBUG, # Get detailed updates
        interaction_id="int-resp-001",
        caller_agent_name="user",
        callee_agent_name=responder_agent.name
    )

    # Start the progress queue consumer
    progress_task = asyncio.create_task(process_progress_queue(progress_queue))

    logging.info(f"Starting task {request_context.task_id}...")
    try:
        # Invoke the agent's handler
        response = await responder_agent.handle_invocation(
            request="What is the capital of Canada?", # The actual request payload
            request_context=request_context
        )
        logging.info(f"Task {request_context.task_id} completed. Final Response: {response}")

    except Exception as e:
        logging.error(f"Task {request_context.task_id} failed: {e}", exc_info=True)
        # Ensure queue consumer finishes even on error
        if not progress_queue.empty():
             await progress_queue.put(None) # Send sentinel if not already sent
    finally:
        # Wait for the progress queue to be fully processed
        await progress_queue.join()
        await progress_task # Ensure the consumer task finishes

if __name__ == "__main__":
    asyncio.run(run_responder_task())
```

### Example 2: Browser Agent Task

```python
# filepath: /path/to/your/scripts/browser_agent_example.py
import asyncio
import logging
import os
import json
from src.agents.agents import BrowserAgent, RequestContext, LogLevel, ProgressUpdate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Agent Definition ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Model config suitable for function/tool calling
model_config_browser = {
    "type": "api",
    "name": "gpt-4-turbo",
    "api_key": api_key,
    "temperature": 0.2, # Lower temperature for more deterministic actions
}

# --- Task Execution Logic ---
# Re-use process_progress_queue from Example 1
async def process_progress_queue(queue: asyncio.Queue[Optional[ProgressUpdate]]):
    # ... (same implementation as Example 1) ...
    while True:
        update = await queue.get()
        if update is None: # Sentinel indicates task completion
            logging.info("Progress queue finished.")
            break
        log_msg = (
            f"PROGRESS: [Task:{update.task_id}]"
            f"[Interaction:{update.interaction_id or 'N/A'}]"
            f"[Agent:{update.agent_name or 'N/A'}]"
            f"[{update.level.name}] {update.message}"
        )
        if update.data:
            try:
                # Limit data logging size
                data_str = json.dumps(update.data)
                log_msg += f" Data: {data_str[:200]}"
                if len(data_str) > 200: log_msg += "..."
            except TypeError:
                 log_msg += " Data: [Unserializable]"
        logging.info(log_msg)
        queue.task_done()

async def run_browser_task():
    """Creates and runs a task with the browser agent."""
    browser_agent: Optional[BrowserAgent] = None
    progress_queue = asyncio.Queue()
    task_id = "browser-task-01"

    # Start the progress queue consumer
    progress_task = asyncio.create_task(process_progress_queue(progress_queue))

    try:
        logging.info("Creating Browser Agent...")
        # Use create_safe for robust initialization
        browser_agent = await BrowserAgent.create_safe(
            agent_name="WebNavigator",
            model_config=model_config_browser,
            generation_system_prompt=(
                "You are a web browsing assistant. Based on the user goal and the current web page content (if any), "
                "decide the single best next action to take using the available browser tools. "
                "If the goal is achieved based on the current content, respond with the final answer directly without using a tool."
            ),
            # critic_system_prompt=... (optional)
            headless_browser=True, # Set to False to see the browser UI
            timeout=45, # Increased timeout for potentially slow CI environments
            temp_dir="./browser_output" # Directory for screenshots
        )
        logging.info(f"Browser Agent '{browser_agent.name}' created successfully.")

        initial_prompt = "What is the current top headline on BBC News?"
        request_context = RequestContext(
            task_id=task_id,
            initial_prompt=initial_prompt,
            progress_queue=progress_queue,
            log_level=LogLevel.DETAILED, # Use DETAILED to see browser actions
            interaction_id="int-browse-001",
            caller_agent_name="user",
            callee_agent_name=browser_agent.name,
            max_interactions=5 # Limit browser steps for this example
        )

        logging.info(f"Starting task {task_id} with prompt: '{initial_prompt}'")
        # The BrowserAgent's handle_invocation will trigger its _run method in 'think' mode.
        # The _run method will call the LLM, parse tool calls, execute browser actions,
        # update memory, and potentially loop (though this simple example doesn't show looping logic).
        # The result here is likely the *first* thought/action plan from the LLM.
        # A full browsing session would require a loop managed externally or within a coordinator agent.
        response = await browser_agent.handle_invocation(
            request=initial_prompt,
            request_context=request_context
        )
        logging.info(f"Task {task_id} initial invocation completed. LLM Response/Action Plan: {str(response)[:500]}...")
        # Note: This example only shows the first step. A real task might involve multiple
        # agent._run calls triggered by a controlling loop based on browser action results.

    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
    finally:
        # Ensure browser is closed and progress queue is handled
        if browser_agent:
            logging.info("Closing browser...")
            await browser_agent.close_browser()
        # Send sentinel to progress queue consumer
        if not progress_queue.empty() or not progress_task.done():
             await progress_queue.put(None)
        await progress_queue.join()
        await progress_task

if __name__ == "__main__":
    # Ensure the output directory exists
    if not os.path.exists("./browser_output"):
        os.makedirs("./browser_output")
    asyncio.run(run_browser_task())

```

### Example 3: Inter-Agent Communication (Planner & Executor)

```python
# filepath: /path/to/your/scripts/multi_agent_example.py
import asyncio
import logging
import os
import json
from typing import Any, Dict, List
from src.agents.agents import Agent, RequestContext, LogLevel, ProgressUpdate, BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Agent Definitions ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

model_config_shared = {
    "type": "api",
    "name": "gpt-4-turbo-preview",
    "api_key": api_key,
    "temperature": 0.5,
}

# --- Executor Agent ---
executor_agent = Agent(
    agent_name="Executor",
    model_config=model_config_shared,
    system_prompt="You are an expert in executing specific, single tasks. Provide a concise summary of the result.",
    allowed_peers=[], # Executor doesn't call others
    max_tokens=200
)

# --- Planner Agent (Subclassing Agent to override _run) ---
class PlannerAgent(Agent):
    """An agent that plans steps and delegates execution."""

    # Override the _run method for custom planning and delegation logic
    async def _run(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any) -> Any:
        """Plans steps and invokes the Executor agent for each step."""
        await self._log_progress(request_context, LogLevel.DETAILED, f"Planner entering _run. Prompt: {str(prompt)[:100]}...")
        self.memory.update_memory("user", str(prompt))

        # 1. Call LLM to get the plan (using the parent Agent's _run logic but forcing json_mode)
        system_prompt_content = getattr(self, f"system_prompt_{run_mode}", self.system_prompt)
        llm_messages = self.memory.to_llm_format()
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        # Ensure system prompt is set
        if not any(m['role'] == 'system' for m in llm_messages_copy):
            llm_messages_copy.insert(0, {"role": "system", "content": system_prompt_content})
        else:
            for msg in llm_messages_copy:
                if msg["role"] == "system": msg["content"] = system_prompt_content; break

        plan_steps: List[str] = []
        try:
            await self._log_progress(request_context, LogLevel.DETAILED, "Calling LLM for planning...")
            # Force JSON mode for planning
            plan_result_raw = await self.model.run(
                messages=llm_messages_copy,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                json_mode=True, # Force JSON for planning
                **self._get_api_kwargs() # Pass API specific kwargs
            )
            await self._log_progress(request_context, LogLevel.DEBUG, f"LLM Plan Output: {plan_result_raw}")

            # Attempt to parse the plan (assuming LLM returns a JSON list of strings)
            parsed_plan = json.loads(plan_result_raw) if isinstance(plan_result_raw, str) else plan_result_raw
            if isinstance(parsed_plan, list) and all(isinstance(step, str) for step in parsed_plan):
                plan_steps = parsed_plan
                self.memory.update_memory("assistant", f"Generated Plan: {plan_steps}") # Store plan in memory
            else:
                raise ValueError(f"LLM did not return a valid JSON list of strings for the plan. Got: {type(parsed_plan)}")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            error_msg = f"Failed to generate or parse plan: {e}. LLM Raw: {str(plan_result_raw)[:200]}"
            await self._log_progress(request_context, LogLevel.MINIMAL, error_msg, data={"error": str(e)})
            self.memory.update_memory("assistant", f"Error during planning: {error_msg}")
            return {"error": "Planning failed.", "details": error_msg} # Return error state

        if not plan_steps:
             await self._log_progress(request_context, LogLevel.MINIMAL, "LLM returned an empty plan.")
             return {"final_results": [], "message": "No steps planned."}

        # 2. Invoke Executor for each step
        execution_results = []
        await self._log_progress(request_context, LogLevel.SUMMARY, f"Generated plan with {len(plan_steps)} steps. Starting execution...")

        for i, step in enumerate(plan_steps):
            step_request = f"Execute this step: {step}"
            await self._log_progress(request_context, LogLevel.SUMMARY, f"Requesting execution of step {i+1}/{len(plan_steps)}: '{step}'")
            try:
                # Invoke the executor agent
                step_result = await self.invoke_agent(
                    target_agent_name="Executor",
                    request=step_request,
                    request_context=request_context # Pass context - limits apply!
                )
                await self._log_progress(request_context, LogLevel.SUMMARY, f"Executor response for step {i+1}: {str(step_result)[:100]}...")
                execution_results.append({"step": step, "status": "success", "result": step_result})
                # Update planner's memory with execution result
                self.memory.update_memory("assistant", f"Step {i+1} ('{step}') result: {str(step_result)[:100]}...")

            except Exception as e:
                error_detail = f"Error invoking Executor for step {i+1} ('{step}'): {e}"
                await self._log_progress(request_context, LogLevel.MINIMAL, error_detail, data={"error": str(e)})
                execution_results.append({"step": step, "status": "error", "error": str(e)})
                self.memory.update_memory("assistant", f"Step {i+1} ('{step}') failed: {e}")
                # Decide whether to stop or continue on error
                # For this example, we'll stop
                await self._log_progress(request_context, LogLevel.MINIMAL, "Stopping execution due to error.")
                break

        final_summary = f"Finished execution. {len([r for r in execution_results if r['status'] == 'success'])} steps succeeded, {len([r for r in execution_results if r['status'] == 'error'])} failed."
        await self._log_progress(request_context, LogLevel.SUMMARY, final_summary)
        self.memory.update_memory("assistant", final_summary)
        return {"final_results": execution_results}

# Instantiate the PlannerAgent
planner_agent = PlannerAgent(
    agent_name="Planner",
    model_config=model_config_shared,
    system_prompt=(
        "You are a meticulous planner. Break down the user's request into a sequence of "
        "clear, actionable steps. Respond ONLY with a JSON list of strings, where each string is one step. "
        "Example: [\"Step 1 description\", \"Step 2 description\"]"
    ),
    allowed_peers=["Executor"], # Planner can call Executor
    max_tokens=500 # Allow more tokens for planning
)

# --- Task Execution Logic ---
# Re-use process_progress_queue from Example 1
async def process_progress_queue(queue: asyncio.Queue[Optional[ProgressUpdate]]):
    # ... (same implementation as Example 1) ...
    while True:
        update = await queue.get()
        if update is None: # Sentinel indicates task completion
            logging.info("Progress queue finished.")
            break
        log_msg = (
            f"PROGRESS: [Task:{update.task_id}]"
            f"[Interaction:{update.interaction_id or 'N/A'}]"
            f"[Agent:{update.agent_name or 'N/A'}]"
            f"[{update.level.name}] {update.message}"
        )
        if update.data:
            try:
                # Limit data logging size
                data_str = json.dumps(update.data)
                log_msg += f" Data: {data_str[:200]}"
                if len(data_str) > 200: log_msg += "..."
            except TypeError:
                 log_msg += " Data: [Unserializable]"
        logging.info(log_msg)
        queue.task_done()

async def run_multi_agent_task():
    """Runs a task involving the Planner and Executor agents."""
    progress_queue = asyncio.Queue()
    task_id = "plan-exec-task-01"
    initial_prompt = "Research the company 'NVIDIA', find their latest stock price, and summarize their main business areas."

    # Start the progress queue consumer
    progress_task = asyncio.create_task(process_progress_queue(progress_queue))

    request_context = RequestContext(
        task_id=task_id,
        initial_prompt=initial_prompt,
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY, # Use SUMMARY for less verbose output
        max_depth=3, # Planner -> Executor is depth 1
        max_interactions=15, # Allow for planning + multiple execution steps
        interaction_id="int-multi-001",
        caller_agent_name="user",
        callee_agent_name=planner_agent.name
    )

    logging.info(f"Starting task {task_id} with prompt: '{initial_prompt}'")
    try:
        # Start the task with the Planner agent
        final_result = await planner_agent.handle_invocation(
            request=initial_prompt,
            request_context=request_context
        )
        logging.info(f"Task {task_id} completed.")
        logging.info(f"Final Task Result:\n{json.dumps(final_result, indent=2)}")

    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
    finally:
        # Ensure queue consumer finishes
        if not progress_queue.empty() or not progress_task.done():
             await progress_queue.put(None)
        await progress_queue.join()
        await progress_task

if __name__ == "__main__":
    # Ensure agents are registered (instantiation handles this)
    logging.info(f"Agents registered: {list(AgentRegistry.all().keys())}")
    asyncio.run(run_multi_agent_task())

```

## 9. Extending the Framework (Developer Guide)

This section provides guidance for developers looking to extend the agent framework with custom components.

*   **Creating New Agent Types:**

    The most common extension is creating specialized agents. Subclass `BaseAgent` (for maximum control) or `Agent` (if using the standard model config loading). The key is implementing the `async def _run(...)` method.

    **Example: Custom `CodeReviewAgent`**

    ```python
    # filepath: /path/to/your/custom_agents/code_review_agent.py
    from src.agents.agents import Agent, RequestContext, LogLevel
    from typing import Any, Dict

    class CodeReviewAgent(Agent):
        """An agent specialized in reviewing code snippets."""

        # You can add custom attributes or override __init__ if needed
        # For example, adding specific prompts for different languages
        # def __init__(self, ..., python_review_prompt: str, js_review_prompt: str):
        #     super().__init__(...)
        #     self.system_prompt_python = python_review_prompt
        #     self.system_prompt_javascript = js_review_prompt

        async def _run(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any) -> Any:
            """Reviews code provided in the prompt."""
            await self._log_progress(request_context, LogLevel.DETAILED, f"CodeReviewAgent entering _run. Mode: {run_mode}")

            # Assume prompt is a dictionary like {"language": "python", "code": "..."}
            if not isinstance(prompt, dict) or "language" not in prompt or "code" not in prompt:
                await self._log_progress(request_context, LogLevel.MINIMAL, "Invalid prompt format for CodeReviewAgent.")
                return {"error": "Invalid prompt format. Expected {'language': str, 'code': str}."}

            language = prompt["language"]
            code_snippet = prompt["code"]

            # 1. Update Memory (Optional, maybe just log the request)
            # self.memory.update_memory("user", f"Review request for {language} code.")

            # 2. Select System Prompt based on language (or run_mode if applicable)
            # Example: Use specific prompts defined in __init__ or default
            system_prompt_content = getattr(self, f"system_prompt_{language}", self.system_prompt)
            await self._log_progress(request_context, LogLevel.DEBUG, f"Using system prompt for {language}.")

            # 3. Prepare Model Input (System prompt + code snippet)
            # For review, maybe don't need full history, just system + code
            llm_messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": f"Please review the following {language} code:\n```\n{code_snippet}\n```"}
            ]

            # 4. Determine Model Parameters
            max_tokens_override = kwargs.get("max_tokens", self.max_tokens)
            api_kwargs = self._get_api_kwargs() # Get API specific params from config
            api_kwargs.update(kwargs) # Allow runtime overrides

            # 5. Call the Model
            await self._log_progress(request_context, LogLevel.DETAILED, "Calling LLM for code review.")
            try:
                review_result = await self.model_instance.run(
                    messages=llm_messages,
                    max_tokens=max_tokens_override,
                    **api_kwargs
                )
                output_str = str(review_result)
                await self._log_progress(request_context, LogLevel.DEBUG, f"LLM Review Output: {output_str[:150]}...")
            except Exception as e:
                await self._log_progress(request_context, LogLevel.MINIMAL, f"LLM call failed during review: {e}")
                raise

            # 6. Update Memory (Optional)
            # self.memory.update_memory("assistant", f"Review complete: {output_str[:100]}...")

            # 7. Post-processing (e.g., format review, call another agent for scoring)
            # Example: Parse review_result if structured, or just return it
            formatted_review = {"language": language, "review": output_str}

            await self._log_progress(request_context, LogLevel.DETAILED, "Code review finished.")
            return formatted_review # Return the structured review
    ```
    Remember to handle registration if you use this agent within a `BaseCrew` (see `crew.md` Advanced Topics on Custom Agent Initialization).

*   **Adding New Memory Types:**

    If `ConversationMemory` or `KGMemory` don't fit your needs (e.g., you need a vector database memory), you can create your own.
    1.  **Subclass `BaseMemory`:** Create a new class inheriting from `src.agents.agents.BaseMemory`.
    2.  **Implement Abstract Methods:** Implement all methods defined in `BaseMemory` (`update_memory`, `retrieve_all`, `to_llm_format`, etc.) according to your memory structure's logic.
    3.  **Update `MemoryManager`:** Modify the `MemoryManager.__init__` method in `src.agents.agents.py` to recognize your new `memory_type` string and instantiate your custom memory class.

    **Conceptual Example: `VectorDBMemory`**

    ```python
    # filepath: /path/to/your/memory/vector_memory.py
    from src.agents.agents import BaseMemory
    from typing import List, Dict, Any
    # Assume 'vector_db_client' is your library for interacting with a vector DB
    # import vector_db_client

    class VectorDBMemory(BaseMemory):
        def __init__(self, embedding_model, db_connection_params, system_prompt: Optional[str] = None):
            super().__init__(memory_type="vector_db")
            self.embedding_model = embedding_model # Model to create text embeddings
            # self.db = vector_db_client.connect(**db_connection_params)
            self.system_prompt = system_prompt # Store system prompt if needed
            # Store recent conversation history separately if needed for context
            self.recent_history: List[Dict[str, str]] = []
            if system_prompt:
                 self.recent_history.append({"role": "system", "content": system_prompt})

        def update_memory(self, role: str, content: str, metadata: Dict = None) -> None:
            """Adds content to vector DB and recent history."""
            # Add to recent history for short-term context
            self.recent_history.append({"role": role, "content": content})
            # Embed and store in vector DB for long-term retrieval
            # embedding = self.embedding_model.embed(content)
            # self.db.add(embedding=embedding, text=content, metadata=metadata or {"role": role})
            print(f"VectorDBMemory: Added '{content[:30]}...' to history and vector DB.")

        def retrieve_recent(self, n: int = 5) -> List[Dict[str, Any]]:
            """Retrieves recent conversation history."""
            return self.recent_history[-n:] if n > 0 else []

        def retrieve_relevant(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
            """Retrieves relevant documents from vector DB."""
            # query_embedding = self.embedding_model.embed(query)
            # results = self.db.search(query_embedding, k=k)
            # Format results for LLM (e.g., as context snippets)
            # formatted_results = [{"role": "system", "content": f"Relevant context: {res['text']}"} for res in results]
            # return formatted_results
            print(f"VectorDBMemory: Retrieving {k} relevant docs for '{query[:30]}...'")
            return [{"role": "system", "content": f"Relevant context for {query}: Doc {i+1}"} for i in range(k)] # Mock

        def to_llm_format(self, query_for_retrieval: Optional[str] = None, k_relevant: int = 3, n_recent: int = 5) -> List[Dict[str, Any]]:
            """Formats memory including recent history and relevant context."""
            messages = []
            # Add relevant context retrieved from vector DB based on a query
            if query_for_retrieval:
                relevant_context = self.retrieve_relevant(query_for_retrieval, k=k_relevant)
                messages.extend(relevant_context) # Add relevant context first
            # Add recent conversation history
            recent = self.retrieve_recent(n=n_recent)
            messages.extend(recent)
            return messages

        # ... Implement other abstract methods (replace, delete, retrieve_all, reset) ...
        def replace_memory(self, *args: Any, **kwargs: Any) -> None: raise NotImplementedError
        def delete_memory(self, *args: Any, **kwargs: Any) -> None: raise NotImplementedError
        def retrieve_all(self) -> List[Dict[str, Any]]: raise NotImplementedError # Might not be feasible for large DBs
        def reset_memory(self) -> None:
            # self.db.clear() # Clear vector DB
            self.recent_history.clear() # Clear recent history
            if self.system_prompt: self.recent_history.append({"role": "system", "content": self.system_prompt})
            print("VectorDBMemory: Reset.")

    # --- In src/agents/agents.py ---
    # class MemoryManager:
    #     def __init__(self, memory_type: str, ...):
    #         ...
    #         elif memory_type == "vector_db":
    #             # Ensure necessary args (embedding_model, db_params) are passed
    #             if model is None: raise ValueError("VectorDBMemory requires an embedding model.")
    #             # Assume db_params are passed via agent config somehow
    #             db_params = kwargs.get("db_connection_params", {})
    #             self.memory_module = VectorDBMemory(embedding_model=model, db_connection_params=db_params, system_prompt=system_prompt)
    #         ...
    ```

*   **Customizing `handle_invocation`:**

    Overriding `handle_invocation` is less common but can be useful for:
    *   **Custom Run Mode Logic:** Implementing complex logic to determine the `run_mode` based on the request content or agent state, beyond the default dictionary key check.
    *   **Pre/Post Processing:** Adding logic that needs to run *before* or *after* the main `_run` method executes, regardless of the `run_mode`.
    *   **Input/Output Transformation:** Modifying the `request` before it reaches `_run` or the `result` after `_run` finishes.

    **Conceptual Example: Adding Request Validation**

    ```python
    # filepath: /path/to/your/custom_agents/validated_agent.py
    from src.agents.agents import Agent, RequestContext, LogLevel
    from pydantic import BaseModel, ValidationError # For validation

    class RequestSchema(BaseModel): # Define expected request structure
        action: str
        payload: Dict[str, Any]

    class ValidatedAgent(Agent):
        async def handle_invocation(self, request: Any, request_context: RequestContext) -> Any:
            """Handles invocation with added request validation."""
            await self._log_progress(request_context, LogLevel.DETAILED, "ValidatedAgent handling invocation.")

            # --- Pre-processing: Validate Request ---
            try:
                if not isinstance(request, dict):
                     raise TypeError("Request must be a dictionary.")
                validated_request = RequestSchema.model_validate(request)
                await self._log_progress(request_context, LogLevel.DEBUG, "Request validated successfully.")
                # Use validated data for run_mode or pass to _run
                run_mode = validated_request.action
                prompt_data = validated_request.payload
            except (ValidationError, TypeError) as e:
                error_msg = f"Invalid request format: {e}"
                await self._log_progress(request_context, LogLevel.MINIMAL, error_msg)
                # Update communication log with error status
                log_entry = { ... "status": "error", "error": error_msg } # Simplified
                self._add_interaction_to_log(request_context.task_id, log_entry)
                # Return an error response instead of raising to prevent task crash
                return {"error": "Invalid request", "details": error_msg}
            # --- End Pre-processing ---

            # Call the original BaseAgent handle_invocation logic (or reimplement parts)
            # This is tricky; often better to call _run directly if overriding handle_invocation
            # For simplicity, let's call _run directly here:
            log_entry_callee = { ... "status": "processing" } # Simplified
            self._add_interaction_to_log(request_context.task_id, log_entry_callee)
            try:
                result = await self._run(prompt=prompt_data, request_context=request_context, run_mode=run_mode)
                log_entry_callee["status"] = "success"; log_entry_callee["response"] = result
                await self._log_progress(request_context, LogLevel.SUMMARY, "Finished processing validated request.")
                return result
            except Exception as e:
                log_entry_callee["status"] = "error"; log_entry_callee["error"] = str(e)
                await self._log_progress(request_context, LogLevel.MINIMAL, f"Error during _run: {e}")
                raise # Re-raise exception from _run

            # --- Post-processing (if needed) ---
            # result = self.transform_output(result)
            # await self._log_progress(request_context, LogLevel.DEBUG, "Output transformed.")
            # return result
    ```
    **Caution:** Overriding `handle_invocation` requires careful handling of logging, context propagation, and error states to maintain consistency with the framework. It's often simpler to put logic within `_run`.

*   **Model Integration:**

    To use a new local model library or API:
    1.  **Subclass Model Base:** Create a new class inheriting from `BaseLLM`, `BaseVLM` (in `src.models.models`), or `BaseAPIModel` (in `src.models.api_models`).
    2.  **Implement `run`:** Implement the `async def run(self, messages, **kwargs)` method. This involves:
        *   Formatting the input `messages` list into the format required by your model/API.
        *   Making the actual inference call or API request.
        *   Handling potential errors (API errors, connection issues, runtime errors).
        *   Parsing the response from the model/API.
        *   **Crucially:** If your model/API supports tool calling, parse the response to identify tool calls and return them in a structured format (e.g., similar to OpenAI's `tool_calls` list). If not, return the text response.
    3.  **Update `Agent._create_model_from_config`:** Modify this method in `src.agents.agents.py` to recognize a new `type` or `class` in the `model_config` and instantiate your new model class, passing necessary configuration parameters (API keys, model paths, etc.).

*   **Tool Implementation:**

    *   **Robustness:** Tools should handle potential errors gracefully (e.g., invalid inputs, external service failures) and return informative error messages or raise specific exceptions if appropriate.
    *   **Schema Accuracy:** The `tools_schema` provided to the agent *must* accurately describe the function's name, purpose, parameters (including types and descriptions), and required parameters. Inaccurate schemas will confuse the LLM.
    *   **Asynchronicity:** If a tool performs I/O operations (network requests, file access), define it as an `async def` function to avoid blocking the agent's event loop. The agent's `_run` method will need to `await` the tool's execution.

    **Conceptual Example: Async Tool**

    ```python
    # filepath: /path/to/your/tools/async_tools.py
    import asyncio
    import aiohttp # Example async HTTP library

    async def fetch_web_page_summary(url: str) -> str:
        """Fetches the content of a URL and returns a summary (simulated)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response.raise_for_status() # Raise error for bad status codes
                    # In reality, you'd process the content (e.g., extract text, summarize)
                    # For example: text_content = await response.text()
                    # summary = await summarize_text(text_content) # Another async call
                    await asyncio.sleep(0.5) # Simulate processing time
                    return f"Summary for {url}: Content looks good."
        except aiohttp.ClientError as e:
            return f"Error fetching URL {url}: {e}"
        except asyncio.TimeoutError:
            return f"Error: Timeout fetching URL {url}"
        except Exception as e:
            return f"An unexpected error occurred fetching {url}: {e}"

    # --- In Agent's _run method ---
    # async def _run(...):
    #     ...
    #     if tool_calls:
    #         for tool_call in tool_calls:
    #             if function_name == "fetch_web_page_summary":
    #                 ...
    #                 # Await the async tool function
    #                 function_response = await function_to_call(**function_args)
    #                 ...
    #                 self.memory.update_memory(role="tool", content=str(function_response))
    #     ...
    ```

---

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
    *   `handle_invocation(request, request_context)`: The primary *asynchronous* entry point for an agent to receive a request (from the user or another agent). It determines the `run_mode` (e.g., 'chat', 'think', 'plan', 'critic') based on the request structure or agent type, logs the interaction, and calls the agent's core logic in `_run`.
    *   `invoke_agent(target_agent_name, request, request_context)`: Allows an agent to *asynchronously* call another registered agent. It handles permission checks (`allowed_peers`), depth/interaction limits (`max_depth`, `max_interactions` from `request_context`), propagates a modified `RequestContext` (updating depth, count, caller/callee names), logs the invocation attempt and result/error, and returns the target agent's response.
    *   `_run(prompt, request_context, run_mode, **kwargs)` (Abstract): The core *asynchronous* logic implementation required by subclasses. This method typically involves:
        1.  Updating memory with the input `prompt`.
        2.  Selecting the appropriate system prompt based on `run_mode` (e.g., checking for `self.system_prompt_<run_mode>`).
        3.  Preparing messages for the language model using `self.memory.to_llm_format()`.
        4.  Calling the language model (`self.model.run(...)` or `self.model_instance.run(...)`) with appropriate parameters (messages, tools, `json_mode`, etc.), potentially influenced by `run_mode` and `kwargs`.
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

## 3. Agent Types

### `Agent`

*   **Purpose:** A general-purpose agent using either local models (`BaseLLM`, `BaseVLM`) or API models (`BaseAPIModel`).
*   **Initialization:** Configured via `model_config` dictionary which specifies model `type` ('local' or 'api'), `name`, and other relevant parameters (API keys, model settings, etc.). May also take `memory_config`. Uses `_create_model_from_config` internally.
*   **Core Logic (`_run`)**: Implements the basic flow: updates memory, selects system prompt (using `system_prompt_<run_mode>` if available, otherwise `system_prompt`), prepares messages, calls the configured model (`self.model_instance.run`), updates memory with the response, and returns the result. Handles `json_mode` automatically if `run_mode == 'plan'`. Can handle tool calls if the model supports them and `tools_schema` is provided (typically used in 'think' mode or similar). Passes API-specific parameters from `model_config` to the model's `run` method.

**Example: Basic `Agent` Instantiation and Invocation**

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import os
import logging
from src.agents.agents import Agent, RequestContext, LogLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
*   **Initialization:** Takes a `model` instance directly. Can initialize PEFT heads if `learning_head='peft'` and `learning_head_config` are provided, wrapping the base model in a `PeftHead`. May also take `memory_config`.
*   **Core Logic (`_run`)**: Similar to `Agent`, but calls `self.model.run` (which might be the base model or the `PeftHead` wrapper). Requires the actual model instance to be passed during initialization. Selects system prompt based on `run_mode`. Handles `json_mode` if `run_mode == 'plan'`. Passes `kwargs` to the model's `run` method.

**Example: `LearnableAgent` Instantiation (Conceptual)**

```python
# filepath: /path/to/your/scripts/learnable_agent_example.py
from src.agents.agents import LearnableAgent
from src.models.models import BaseLLM

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
    model=local_model_instance,
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
    *   Uses `model_config` like `Agent`.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_config_browser = {
    "type": "api",
    "name": "gpt-4-turbo",
    "api_key": os.getenv("OPENAI_API_KEY"),
}

async def manage_browser_agent():
    browser_agent = None
    progress_queue = asyncio.Queue()
    task_id = "browser-example-01"

    try:
        logging.info("Creating BrowserAgent using create_safe...")
        browser_agent = await BrowserAgent.create_safe(
            agent_name="SafeBrowser",
            model_config=model_config_browser,
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
        response = await browser_agent.handle_invocation(
            request={"mode": "think", "prompt": initial_prompt},
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
    ```
5.  **Determine Model Parameters:** Set parameters based on `run_mode` or `kwargs`.
    ```python
    json_mode = (run_mode == "plan")
    use_tools = (run_mode == "think")
    max_tokens_override = kwargs.get("max_tokens", self.max_tokens)
    model_params = {"max_tokens": max_tokens_override, "json_mode": json_mode}
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
        output_content = result.get("content") if isinstance(result, dict) else str(result)
        tool_calls = result.get("tool_calls") if isinstance(result, dict) else None
        await self._log_progress(request_context, LogLevel.DEBUG, f"LLM Output: {output_content[:100]}...", data={"tool_calls": tool_calls})
    except Exception as e:
        await self._log_progress(request_context, LogLevel.MINIMAL, f"LLM call failed: {e}", data={"error": str(e)})
        raise
    ```
7.  **Update Memory (Output):** Store the model's response content.
    ```python
    if output_content:
        self.memory.update_memory("assistant", output_content)
    ```
8.  **Post-processing (Tool Calls / Agent Invocation):** Analyze `result`.
    *   **Tool Calls:** If `tool_calls` exist (and `run_mode` supports tools), parse them, execute functions from `self.tools`, update memory with results using role `tool` (or `tool_error`), and potentially call the LLM again.
    *   **Agent Invocation:** If delegation is needed, use `await self.invoke_agent(...)`.
9.  **Return Result:** Return the final processed result.

## 5. Tool Usage

Agents can be equipped with tools (Python functions) to interact with external systems or perform specific calculations.

1.  **Define Tools:** Create Python functions that perform the desired actions.
2.  **Define Schema:** Create an OpenAI-compatible JSON schema list (`tools_schema`) that describes each function, its purpose, and its parameters. Pydantic models can be helpful here.
3.  **Provide to Agent:** Pass the dictionary of functions (`tools`) and the schema list (`tools_schema`) during agent initialization.
    ```python
    def get_current_weather(location: str, unit: str = "celsius") -> str:
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
        tools={"get_current_weather": get_current_weather},
        tools_schema=weather_schema
    )
    ```
4.  **Handle in `_run`:** When the LLM response contains `tool_calls` (and the `run_mode` supports tools):
    *   Parse the tool name and arguments from each call.
    *   Find the corresponding function in `self.tools`.
    *   Execute the function (await if async) with the arguments, handling errors.
    *   Format the function's return value or error.
    *   Update the agent's memory with the tool result using the `tool` role (or `tool_error`).
    *   Typically, call the LLM *again* with the history including the `tool` role message, so it can generate the final user-facing response based on the tool's output.

## 6. Logging and Monitoring

*   **Standard Logging:** The framework uses Python's built-in `logging`. Configure the root logger to control the level and format of messages printed to the console/file (e.g., `logging.basicConfig(level=logging.INFO)`).
*   **Progress Updates:** For real-time monitoring of tasks, especially in UI applications, use the `progress_queue` returned by `BaseCrew.run_task`. Consume `ProgressUpdate` objects from this queue asynchronously. The level of detail is controlled by the `log_level` parameter passed to `run_task`. `LogLevel.DEBUG` provides the most verbose information, including structured data.

## 7. Error Handling

*   **Agent Initialization:** Errors during `Agent` or `LearnableAgent` init are typically synchronous. `BrowserAgent.create_safe` handles internal retries/timeouts but will raise exceptions (like `TimeoutError`) on persistent failure.
*   **Model Calls:** The `_run` method should wrap `self.model.run(...)` in a `try...except` block to catch model errors (API issues, runtime errors) and log them using `_log_progress`.
*   **Tool Calls:** Execution of tools within `_run` should also be wrapped in `try...except`. Log errors and update memory with a `tool_error` role.
*   **Agent Invocation:** `invoke_agent` handles `PermissionError`, `ValueError` (limits), and `AgentNotFound` errors. Exceptions raised by the *target* agent's `handle_invocation` or `_run` are propagated back to the caller. The caller's `invoke_agent` call should be within a `try...except` block if specific error handling is needed.
*   **Task Execution:** `BaseCrew.run_task` wraps the initial agent call in `try...except`, logs task failures, puts `None` on the progress queue, and re-raises the exception.

## 8. Usage Examples (Revised & Expanded)

### Example 1: Basic API Agent

```python
# filepath: /path/to/your/scripts/basic_agent_example.py
import asyncio
import logging
import os
from src.agents.agents import Agent, RequestContext, LogLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

model_config_gpt4 = {
    "type": "api",
    "name": "gpt-4-turbo-preview",
    "api_key": api_key,
    "temperature": 0.7,
}

responder_agent = Agent(
    agent_name="HelpfulResponder",
    model_config=model_config_gpt4,
    system_prompt="You are a concise and helpful assistant.",
    max_tokens=150
)

async def run_responder_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="responder-task-01",
        initial_prompt="What is the capital of Canada?",
        progress_queue=progress_queue,
        log_level=LogLevel.DEBUG,
    )

    logging.info(f"Starting task {request_context.task_id}...")
    try:
        response = await responder_agent.handle_invocation(
            request="What is the capital of Canada?",
            request_context=request_context
        )
        logging.info(f"Task {request_context.task_id} completed. Final Response: {response}")

    except Exception as e:
        logging.error(f"Task {request_context.task_id} failed: {e}", exc_info=True)
    finally:
        await progress_queue.put(None)
        await progress_queue.join()

# asyncio.run(run_responder_task())
```

### Example 2: Browser Agent Task

```python
# filepath: /path/to/your/scripts/browser_agent_example.py
import asyncio
import logging
import os
from src.agents.agents import BrowserAgent, RequestContext, LogLevel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

model_config_browser = {
    "type": "api",
    "name": "gpt-4-turbo",
    "api_key": api_key,
    "temperature": 0.2,
}

async def run_browser_task():
    browser_agent = None
    progress_queue = asyncio.Queue()
    task_id = "browser-task-01"

    try:
        browser_agent = await BrowserAgent.create_safe(
            agent_name="WebNavigator",
            model_config=model_config_browser,
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

model_config_shared = {
    "type": "api",
    "name": "gpt-4-turbo-preview",
    "api_key": api_key,
    "temperature": 0.5,
}

executor_agent = Agent(
    agent_name="Executor",
    model_config=model_config_shared,
    system_prompt="You are an expert in executing specific tasks.",
    allowed_peers=[]
)

class PlannerAgent(Agent):
    async def _run(self, prompt, request_context, run_mode, **kwargs):
        self.memory.update_memory("user", str(prompt))
        system_prompt_content = getattr(self, f"system_prompt_{run_mode}", self.system_prompt)
        llm_messages = self.memory.to_llm_format()
        llm_messages_copy = [msg.copy() for msg in llm_messages]
        if not any(m['role'] == 'system' for m in llm_messages_copy):
            llm_messages_copy.insert(0, {"role": "system", "content": system_prompt_content})

        plan_steps = []
        try:
            plan_result_raw = await self.model.run(
                messages=llm_messages_copy,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                json_mode=True
            )
            parsed_plan = json.loads(plan_result_raw) if isinstance(plan_result_raw, str) else plan_result_raw
            if isinstance(parsed_plan, list) and all(isinstance(step, str) for step in parsed_plan):
                plan_steps = parsed_plan
                self.memory.update_memory("assistant", f"Generated Plan: {plan_steps}")
            else:
                raise ValueError("Invalid plan format.")
        except Exception as e:
            return {"error": "Planning failed.", "details": str(e)}

        execution_results = []
        for i, step in enumerate(plan_steps):
            step_request = f"Execute this step: {step}"
            try:
                step_result = await self.invoke_agent(
                    target_agent_name="Executor",
                    request=step_request,
                    request_context=request_context
                )
                execution_results.append({"step": step, "status": "success", "result": step_result})
            except Exception as e:
                execution_results.append({"step": step, "status": "error", "error": str(e)})
                break

        return {"final_results": execution_results}

planner_agent = PlannerAgent(
    agent_name="Planner",
    model_config=model_config_shared,
    system_prompt="You are a meticulous planner.",
    allowed_peers=["Executor"]
)

async def run_multi_agent_task():
    progress_queue = asyncio.Queue()
    task_id = "plan-exec-task-01"
    initial_prompt = "Research the company 'NVIDIA', find their latest stock price, and summarize their main business areas."

    request_context = RequestContext(
        task_id=task_id,
        initial_prompt=initial_prompt,
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY,
        max_depth=3,
        max_interactions=15
    )

    try:
        final_result = await planner_agent.handle_invocation(
            request=initial_prompt,
            request_context=request_context
        )
        logging.info(f"Task {task_id} completed. Result: {final_result}")
    except Exception as e:
        logging.error(f"Task {task_id} failed: {e}", exc_info=True)
    finally:
        await progress_queue.join()

# asyncio.run(run_multi_agent_task())
```

## 9. Extending the Framework (Developer Guide)

This section provides guidance for developers looking to extend the agent framework with custom components.

*   **Creating New Agent Types:**

    Subclass `BaseAgent` or `Agent` and implement the `async def _run(...)` method.

    **Example: Custom `CodeReviewAgent`**

    ```python
    from src.agents.agents import Agent, RequestContext, LogLevel

    class CodeReviewAgent(Agent):
        async def _run(self, prompt, request_context, run_mode, **kwargs):
            self.memory.update_memory("user", str(prompt))
            system_prompt_content = getattr(self, f"system_prompt_{run_mode}", self.system_prompt)
            llm_messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": f"Review the following code:\n{prompt}"}
            ]
            try:
                review_result = await self.model.run(
                    messages=llm_messages,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens)
                )
                return {"review": review_result}
            except Exception as e:
                return {"error": str(e)}
    ```

*   **Adding New Memory Types:**

    Subclass `BaseMemory` and implement methods like `update_memory`, `retrieve_all`, and `to_llm_format`.

    **Example: `VectorDBMemory`**

    ```python
    from src.agents.agents import BaseMemory

    class VectorDBMemory(BaseMemory):
        def __init__(self, embedding_model, db_connection_params):
            super().__init__(memory_type="vector_db")
            self.embedding_model = embedding_model

        def update_memory(self, role, content):
            pass

        def to_llm_format(self):
            pass
    ```

*   **Customizing `handle_invocation`:**

    Override `handle_invocation` for custom request validation or pre/post-processing.

    **Example: Adding Request Validation**

    ```python
    from src.agents.agents import Agent, RequestContext

    class ValidatedAgent(Agent):
        async def handle_invocation(self, request, request_context):
            if not isinstance(request, dict):
                return {"error": "Invalid request format."}
            return await self._run(prompt=request, request_context=request_context, run_mode="default")
    ```

*   **Model Integration:**

    Subclass `BaseLLM`, `BaseVLM`, or `BaseAPIModel` and implement the `run` method.

*   **Tool Implementation:**

    Define tools as Python functions and provide schemas for integration.

    **Example: Async Tool**

    ```python
    import aiohttp

    async def fetch_web_page_summary(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.text()
    ```

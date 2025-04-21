# Agents Framework Documentation

This document explains the core concepts and usage of the Agent framework defined in `src/agents/agents.py`.

## 1. Introduction

### What are Agents?

Agents are the fundamental building blocks of the multi-agent system. They represent individual AI entities capable of performing tasks, communicating with each other, and interacting with external tools or environments.

### Why Agents?

The agent framework promotes modularity and reusability. Key design goals include:

*   **Shared Base Models:** Allowing multiple agents to leverage the same powerful (and potentially large) base language model efficiently.
*   **Agent-Specific Extensions:** Enabling customization through unique system prompts, specialized memory types, learnable components (like PEFT heads), and access to different tools.
*   **Structured Communication:** Providing a standardized way for agents to invoke each other and handle requests.
*   **Context Management:** Tracking the flow of execution, depth, and interaction counts within a task using `RequestContext`.
*   **Progress Monitoring:** Offering a mechanism to observe the agent's activity through `ProgressUpdate`s.

## 2. Core Concepts

### `BaseAgent` (Abstract Class)

*   **Purpose:** The foundation for all agent types. It defines the common interface and core functionalities.
*   **Key Attributes:**
    *   `model`: The underlying language model (local or API).
    *   `system_prompt`: Default instructions defining the agent's persona or goal.
    *   `tools`/`tools_schema`: Definitions for external functions the agent can use (optional).
    *   `allowed_peers`: A set of agent names this agent is permitted to call.
    *   `name`: The unique registered name of the agent instance.
    *   `communication_log`: Stores interaction history per task.
*   **Key Methods:**
    *   `handle_invocation(request, request_context)`: The primary entry point for an agent to receive a request (from the user or another agent). It determines the `run_mode` and calls `_run`.
    *   `invoke_agent(target_agent_name, request, request_context)`: Allows an agent to call another agent, handling permissions, context propagation, and limits.
    *   `_run(prompt, request_context, run_mode, **kwargs)` (Abstract): The core logic implementation required by subclasses. Selects prompts, calls the model, manages memory based on `run_mode`.
    *   `_log_progress(...)`: Helper for sending `ProgressUpdate`s.

### `RequestContext` (Dataclass)

*   **Purpose:** Carries essential information about the current state of a task's execution flow as it passes between agents.
*   **Key Attributes:**
    *   `task_id`: ID of the overall task.
    *   `interaction_id`: ID of the specific call being made.
    *   `progress_queue`: Queue for sending `ProgressUpdate`s.
    *   `log_level`: Controls verbosity of progress updates.
    *   `depth`/`max_depth`: Tracks call stack depth.
    *   `interaction_count`/`max_interactions`: Tracks the number of agent calls.
    *   `caller_agent_name`/`callee_agent_name`: Tracks the invocation path.

### `ProgressLogger` and `ProgressUpdate`

*   **Purpose:** Provide a mechanism for agents to report their status and actions during task execution.
*   **`ProgressUpdate`:** A data structure containing the timestamp, log level, message, task/interaction IDs, agent name, and optional data.
*   **`ProgressLogger.log(...)`:** A static method used by agents (via `_log_progress`) to send updates to the `RequestContext`'s `progress_queue`.

### `AgentRegistry`

*   **Purpose:** A central, weak-reference dictionary to keep track of all active agent instances by name. This allows agents to find each other via `AgentRegistry.get(name)` for `invoke_agent` calls without creating strong circular dependencies.

### Memory (`BaseMemory`, `ConversationMemory`, `KGMemory`, `MemoryManager`)

*   **Purpose:** Allows agents to maintain state across interactions.
*   **`BaseMemory`:** Abstract base class defining the memory interface (`update_memory`, `retrieve_all`, `to_llm_format`, etc.).
*   **`ConversationMemory`:** Stores interaction history as a simple list of messages (`{'role': ..., 'content': ...}`).
*   **`KGMemory`:** Stores information as knowledge graph triplets (Subject, Predicate, Object), requiring an LLM for fact extraction.
*   **`MemoryManager`:** A factory that instantiates and manages the chosen memory type for an agent. Agents interact with their memory through this manager.

## 3. Agent Types

### `Agent`

*   **Purpose:** A general-purpose agent that can use either a locally loaded model (`BaseLLM`, `BaseVLM`) or an API-based model (`BaseAPIModel`), configured via `model_config`.
*   **Key Features:** Implements `_run` to handle different modes (like 'chat', 'plan') by selecting the appropriate system prompt and calling the configured model.

### `LearnableAgent`

*   **Purpose:** Extends `BaseAgent` for scenarios involving learnable components, typically using local models.
*   **Key Features:**
    *   Can initialize components like PEFT heads via `learning_head` and `learning_head_config`.
    *   Implements `_run` similarly to `Agent`, interacting with its potentially adapted internal model.

### `BrowserAgent`

*   **Purpose:** Specialized `Agent` subclass designed for web browsing tasks.
*   **Key Features:**
    *   Uses distinct system prompts: `generation_system_prompt` (for 'think' mode) and `critic_system_prompt` (for 'critic' mode).
    *   Initializes a `BrowserTool` instance (usually via async `create` or `create_safe` methods) to interact with a web browser.
    *   The `_run` method handles 'think' (generating browser actions) and 'critic' modes. The 'think' mode implementation is expected to parse the LLM output and execute corresponding browser actions using `self.browser_methods`.
    *   Dynamically generates `tools_schema` based on the available methods in `BrowserTool`.

## 4. Usage Examples

*(Note: These are conceptual examples focusing on the agent framework structure. Actual model loading and interaction details depend on the specific model classes used.)*

### Example 1: Simple `Agent` Instantiation

```python
# filepath: /path/to/your/script.py
from src.agents.agents import Agent, RequestContext, LogLevel
import asyncio

# Assume model_config defines an API model (e.g., OpenAI GPT-4)
# See Agent._create_model_from_config for structure
model_config_api = {
    "type": "api",
    "name": "gpt-4",
    "api_key": "YOUR_API_KEY", # Use env variables in practice
    # Add other API params like temperature if needed
}

# Create an agent instance
simple_agent = Agent(
    agent_name="SimpleResponder",
    model_config=model_config_api,
    system_prompt="You are a helpful assistant.",
    max_tokens=100
)

async def run_simple_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="task-001",
        initial_prompt="Hello Agent!",
        progress_queue=progress_queue,
        log_level=LogLevel.DEBUG,
        interaction_id="int-001", # Initial interaction ID
        caller_agent_name="user",
        callee_agent_name=simple_agent.name
    )

    # Invoke the agent
    response = await simple_agent.handle_invocation(
        request="Hello Agent!", # Simple prompt request
        request_context=request_context
    )
    print(f"Agent Response: {response}")

    # Process progress updates (optional)
    while True:
        update = await progress_queue.get()
        if update is None: # Sentinel value indicates end
            break
        print(f"Progress: [{update.level.name}] {update.message}")

# asyncio.run(run_simple_task())
```

### Example 2: `BrowserAgent` Creation

```python
# filepath: /path/to/your/script.py
from src.agents.agents import BrowserAgent, RequestContext, LogLevel
import asyncio

# Assume model_config defines a suitable model (local or API)
model_config_browser = {
    "type": "api",
    "name": "gpt-4-turbo", # Example model
    "api_key": "YOUR_API_KEY",
}

async def create_browser_agent():
    browser_agent = await BrowserAgent.create_safe( # Use create_safe for robustness
        agent_name="WebSurfer",
        model_config=model_config_browser,
        generation_system_prompt="You control a web browser. Determine the next action based on the goal and current page state.",
        critic_system_prompt="Evaluate the effectiveness of the last browser action.",
        headless_browser=True, # Run without UI
        timeout=30 # Timeout for browser init
    )
    print(f"Browser Agent '{browser_agent.name}' created.")
    # The agent's tools_schema is now populated based on BrowserTool

    # Example invocation (conceptual - BrowserAgent defaults to 'think' mode)
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="task-browse-01",
        initial_prompt="Find the capital of France.",
        progress_queue=progress_queue,
        log_level=LogLevel.DETAILED,
        interaction_id="int-browse-001",
        caller_agent_name="user",
        callee_agent_name=browser_agent.name
    )
    try:
        # The initial prompt goes to the 'think' mode via handle_invocation
        first_thought = await browser_agent.handle_invocation(
            request="Find the capital of France.",
            request_context=request_context
        )
        print(f"Agent's first thought/action plan: {first_thought}")
        # In a real scenario, the BrowserAgent's _run method (for 'think')
        # would parse 'first_thought' and execute browser actions.
    finally:
        await browser_agent.close_browser() # Important cleanup

# asyncio.run(create_browser_agent())

```

### Example 3: Inter-Agent Communication

```python
# filepath: /path/to/your/script.py
from src.agents.agents import Agent, RequestContext, LogLevel, AgentRegistry
import asyncio

# Assume model_config_api is defined as in Example 1

# Agent 1: Planner
planner_agent = Agent(
    agent_name="Planner",
    model_config=model_config_api,
    system_prompt="You break down tasks into steps. Respond with a JSON list of steps.",
    allowed_peers=["Executor"] # Planner can call Executor
)

# Agent 2: Executor
executor_agent = Agent(
    agent_name="Executor",
    model_config=model_config_api,
    system_prompt="You execute a given step.",
    allowed_peers=[] # Executor cannot call others in this example
)

# Add a custom _run method to Planner to invoke Executor
async def planner_run_override(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any) -> Any:
    # 1. Call LLM to get plan (simplified)
    plan_steps_str = await super(Agent, self)._run(prompt, request_context, run_mode, **kwargs) # Call original _run for planning
    try:
        plan_steps = json.loads(plan_steps_str)
        if not isinstance(plan_steps, list): raise ValueError("Plan is not a list")
    except (json.JSONDecodeError, ValueError) as e:
        await self._log_progress(request_context, LogLevel.MINIMAL, f"Failed to parse plan: {e}")
        return f"Error: Could not generate valid plan. LLM output: {plan_steps_str}"

    # 2. Invoke Executor for each step
    results = []
    for i, step in enumerate(plan_steps):
        await self._log_progress(request_context, LogLevel.SUMMARY, f"Requesting execution of step {i+1}: {step}")
        try:
            # Prepare a new request context if needed, or reuse/modify existing one carefully
            # For simplicity, we pass the current context, but depth/count limits apply
            step_result = await self.invoke_agent(
                target_agent_name="Executor",
                request=f"Execute this step: {step}", # Send request to Executor
                request_context=request_context # Pass context
            )
            await self._log_progress(request_context, LogLevel.SUMMARY, f"Executor response for step {i+1}: {str(step_result)[:100]}...")
            results.append({"step": step, "result": step_result})
        except Exception as e:
            await self._log_progress(request_context, LogLevel.MINIMAL, f"Error invoking Executor for step {i+1}: {e}")
            results.append({"step": step, "error": str(e)})
            # Decide whether to stop or continue on error
            break

    return {"final_results": results}

# Monkey-patch the Planner's _run method for this example
planner_agent._run = planner_run_override.__get__(planner_agent, Agent)


async def run_planning_task():
    progress_queue = asyncio.Queue()
    request_context = RequestContext(
        task_id="task-plan-exec-01",
        initial_prompt="Plan and execute: Bake a cake.",
        progress_queue=progress_queue,
        log_level=LogLevel.SUMMARY,
        max_depth=3, # Allow Planner -> Executor calls
        max_interactions=10,
        interaction_id="int-plan-001",
        caller_agent_name="user",
        callee_agent_name=planner_agent.name
    )

    # Start the task with the Planner
    final_result = await planner_agent.handle_invocation(
        request="Plan and execute: Bake a cake.",
        request_context=request_context
    )
    print(f"Final Task Result:\n{json.dumps(final_result, indent=2)}")

    # Process progress updates
    # ... (queue processing loop as in Example 1) ...

# asyncio.run(run_planning_task())
```

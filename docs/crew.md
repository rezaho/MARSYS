# Crew Framework Documentation

This document explains how to configure, initialize, and run groups of agents using the `BaseCrew` class defined in `src/topology/crew.py`.

## 1. Introduction

### What is a Crew?

A "Crew" represents a collection of agents designed to collaborate on a task. The `BaseCrew` class provides a way to:

*   Define the agents in the crew using configuration (`AgentConfig`).
*   Initialize all agents, handling dependencies and asynchronous setup (like `BrowserAgent`).
*   Provide a unified entry point (`run_task`) to start a task with a specific agent.
*   Manage cleanup of resources used by agents (e.g., closing browser instances).

### Why Use a Crew?

*   **Configuration Driven:** Define your multi-agent setup declaratively.
*   **Simplified Initialization:** Handles the potentially complex setup of multiple agents, especially those requiring async initialization.
*   **Task Management:** Provides a standard way to initiate and monitor tasks involving multiple agents.
*   **Resource Management:** Ensures resources like browsers are properly cleaned up.

## 2. Core Concepts

### `AgentConfig` (Pydantic Model)

*   **Purpose:** Defines the configuration for a *single* agent within the crew. A list of these configs is passed to `BaseCrew`.
*   **Key Attributes:**
    *   `name` (str): A unique name for this agent instance within the crew and the `AgentRegistry`.
    *   `agent_class` (str): The class name of the agent to instantiate (e.g., `"Agent"`, `"BrowserAgent"`, `"LearnableAgent"`). Must match a key in `BaseCrew._initialize_agents.agent_classes`.
    *   `model_config` (Optional[Dict]): Required for `Agent` and `BrowserAgent`. A dictionary specifying the model details (type, name, API key, etc.). See `Agent._create_model_from_config`.
    *   `model_ref` (Optional[str]): Required for `LearnableAgent`. A string identifier used to load the specific local model instance (loading logic is external to `BaseCrew`).
    *   `system_prompt` (str): The base system prompt for the agent.
    *   `tools` (Optional[Dict]): Dictionary mapping tool names to *already loaded* callable functions. Tool loading/resolution must happen before creating the crew.
    *   `tools_schema` (Optional[List]): OpenAI-compatible JSON schema for the tools.
    *   `memory_type` (Optional[str]): Type of memory ('conversation_history' or 'kg'). Defaults to 'conversation_history'.
    *   `max_tokens` (Optional[int]): Default max tokens for this agent's LLM responses.
    *   `allowed_peers` (Optional[List[str]]): List of agent `name`s that this agent is allowed to invoke. Defaults to an empty list.
    *   `generation_system_prompt` (Optional[str]): Specific prompt for `BrowserAgent`'s 'think' mode.
    *   `critic_system_prompt` (Optional[str]): Specific prompt for `BrowserAgent`'s 'critic' mode.
    *   `temp_dir` (Optional[str]): Directory for `BrowserAgent` temporary files (e.g., screenshots). Defaults to `./tmp/screenshots`.
    *   `headless_browser` (bool): Whether `BrowserAgent` should run headlessly. Defaults to `True`.
    *   `browser_init_timeout` (Optional[int]): Timeout (in seconds) for each `BrowserAgent` initialization attempt. Defaults to 30.
    *   `learning_head` / `learning_head_config`: For `LearnableAgent` PEFT setup.

**Example: `AgentConfig` Structure (YAML)**

```yaml
# filepath: /path/to/your/configs/crew_config.yaml
# Example configuration for a simple crew

- name: "InfoGatherer"
  agent_class: "Agent"
  model_config:
    type: "api"
    name: "gpt-3.5-turbo"
    api_key: "${OPENAI_API_KEY}" # Use env var substitution or load separately
    temperature: 0.6
  system_prompt: "You gather information based on user requests."
  memory_type: "conversation_history"
  max_tokens: 300
  allowed_peers: ["Summarizer"] # Can call the Summarizer

- name: "Summarizer"
  agent_class: "Agent"
  model_config:
    type: "api"
    name: "gpt-4-turbo-preview"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.3
  system_prompt: "You summarize the provided text concisely."
  memory_type: "conversation_history"
  max_tokens: 150
  allowed_peers: [] # Cannot call other agents

# Example BrowserAgent config
# - name: "WebResearcher"
#   agent_class: "BrowserAgent"
#   model_config:
#     type: "api"
#     name: "gpt-4-turbo"
#     api_key: "${OPENAI_API_KEY}"
#   generation_system_prompt: "Use browser tools to find information online."
#   headless_browser: true
#   browser_init_timeout: 45
#   temp_dir: "/app/data/screenshots"
#   allowed_peers: ["Summarizer"]
```

**Loading `AgentConfig` from YAML (Conceptual)**

```python
# filepath: /path/to/your/scripts/load_config_example.py
import yaml
from pydantic import TypeAdapter
from typing import List
import os
# Assuming AgentConfig is importable
from src.topology.crew import AgentConfig

def load_agent_configs_from_yaml(filepath: str) -> List[AgentConfig]:
    """Loads and validates agent configurations from a YAML file."""
    try:
        with open(filepath, 'r') as f:
            # Simple environment variable substitution (use more robust method in production)
            raw_content = f.read()
            expanded_content = os.path.expandvars(raw_content)
            raw_configs = yaml.safe_load(expanded_content)

        # Use Pydantic's TypeAdapter for validation
        adapter = TypeAdapter(List[AgentConfig])
        validated_configs = adapter.validate_python(raw_configs)
        print(f"Successfully loaded and validated {len(validated_configs)} agent configs.")
        return validated_configs
    except FileNotFoundError:
        print(f"Error: Config file not found at {filepath}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filepath}: {e}")
        raise
    except Exception as e: # Catch Pydantic validation errors etc.
        print(f"Error validating agent configurations from {filepath}: {e}")
        raise

# Example usage:
# config_file = "/path/to/your/configs/crew_config.yaml"
# agent_configs = load_agent_configs_from_yaml(config_file)
# Now 'agent_configs' can be passed to BaseCrew.create
```

### `BaseCrew`

*   **Purpose:** Manages the lifecycle of a group of agents defined by `AgentConfig`s.
*   **Initialization (Asynchronous Factory):**
    *   `BaseCrew` instances **must** be created using the asynchronous class method `await BaseCrew.create(agent_configs, ...)`.
    *   The `__init__` method only stores the configuration.
    *   `BaseCrew.create` calls the internal `async _initialize_agents` method after basic instantiation.
    *   `_initialize_agents` iterates through the `agent_configs`, instantiates the corresponding agent classes (`Agent`, `LearnableAgent`, or `BrowserAgent`), handles asynchronous initialization (like `BrowserAgent.create_safe`), and populates the `self.agents` dictionary. It also registers each agent with the `AgentRegistry`.
*   **Key Attributes:**
    *   `agent_configs`: The list of `AgentConfig` objects passed during creation.
    *   `learning_config`: Optional configuration for RL (not detailed here).
    *   `agents`: A dictionary mapping agent names to the initialized `BaseAgent` instances. Populated by `_initialize_agents`.
*   **Key Methods:**
    *   `create(cls, agent_configs, learning_config)` (Class Method): The **required** async factory method to create and initialize a `BaseCrew`.
    *   `run_task(initial_agent_name, initial_prompt, log_level, max_depth, max_interactions)`:
        *   The primary method to start a task.
        *   Takes the name of the agent to start with and the initial prompt.
        *   Creates the initial `RequestContext`, including a `progress_queue`.
        *   Invokes the `handle_invocation` method of the specified initial agent.
        *   Returns a tuple: `(final_result, progress_queue)`. The caller is responsible for consuming updates from the `progress_queue`.
        *   Handles top-level task exceptions, logs them, ensures the `progress_queue` receives a `None` sentinel, and re-raises the exception.
    *   `cleanup()`: An async method to clean up resources used by agents. Currently, it iterates through agents and calls `close_browser()` on any `BrowserAgent` instances concurrently. Should be called when the crew is no longer needed.

**Example: Creating a Crew using `BaseCrew.create`**

```python
# filepath: /path/to/your/scripts/create_crew_example.py
import asyncio
import logging
from typing import List
from src.topology.crew import BaseCrew, AgentConfig

logging.basicConfig(level=logging.INFO)

# Assume agent_configs is loaded (e.g., from YAML or defined directly)
# Using simplified direct definition for this example:
agent_configs: List[AgentConfig] = [
    AgentConfig(
        name="Greeter", agent_class="Agent",
        model_config={"type": "api", "name": "gpt-3.5-turbo", "api_key": "YOUR_API_KEY"},
        system_prompt="Say hello."
    )
    # Add more AgentConfig objects here
]

async def initialize_crew():
    crew = None
    try:
        print("Initializing crew...")
        # Use the async factory method
        crew = await BaseCrew.create(agent_configs=agent_configs)
        print(f"Crew initialized successfully!")
        print(f"Initialized agents: {list(crew.agents.keys())}")
        # Crew is ready to run tasks
        return crew
    except Exception as e:
        print(f"Error initializing crew: {e}")
        # Handle initialization errors (e.g., invalid config, browser init failure)
        return None

async def main():
    # Replace "YOUR_API_KEY" in agent_configs before running
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Using placeholder.")
        api_key = "YOUR_API_KEY" # Placeholder
    for config in agent_configs:
        if config.model_config and config.model_config.get("api_key") == "YOUR_API_KEY":
            config.model_config["api_key"] = api_key

    crew_instance = await initialize_crew()
    if crew_instance:
        # ... proceed to run tasks using crew_instance.run_task(...) ...
        print("Crew ready.")
        # Remember to call cleanup when done
        # await crew_instance.cleanup()

# asyncio.run(main())
```

**Example: Running a Task with `run_task`**

```python
# filepath: /path/to/your/scripts/run_task_example.py
import asyncio
import logging
from typing import Optional
from src.topology.crew import BaseCrew
from src.agents.agents import LogLevel, ProgressUpdate

# Assume 'crew_instance' is an already initialized BaseCrew object
# Assume 'consume_progress' function exists (see agents.md example)

async def consume_progress(progress_queue: asyncio.Queue[Optional[ProgressUpdate]]):
    # ... (implementation from agents.md example) ...
    logging.info("Progress consumer started.")
    while True:
        update = await progress_queue.get()
        if update is None: logging.info("Progress consumer finished."); break
        logging.info(f"PROGRESS [{update.level.name}]: {update.message}")
        progress_queue.task_done()

async def execute_crew_task(crew_instance: BaseCrew):
    if not crew_instance:
        print("Crew instance is not available.")
        return

    task_prompt = "Give me a short greeting."
    initial_agent = "Greeter" # Name must match an agent in the crew config

    print(f"\nRunning task with initial agent '{initial_agent}'...")
    try:
        # Call run_task
        final_result, progress_queue = await crew_instance.run_task(
            initial_agent_name=initial_agent,
            initial_prompt=task_prompt,
            log_level=LogLevel.SUMMARY # Control progress update verbosity
        )

        # Start consuming progress updates in the background
        progress_consumer = asyncio.create_task(consume_progress(progress_queue))

        # Wait for the progress queue to be fully processed
        await progress_queue.join()
        await progress_consumer # Ensure consumer task finishes

        print(f"\nTask completed. Final Result: {final_result}")

    except ValueError as e:
        print(f"Task failed: Invalid agent name? {e}")
    except Exception as e:
        print(f"Task failed with an unexpected error: {e}")

# Conceptual main execution flow
# async def main():
#     # ... (Initialize crew as in create_crew_example.py) ...
#     crew = await initialize_crew()
#     if crew:
#         await execute_crew_task(crew)
#         await crew.cleanup() # Important: Call cleanup
# asyncio.run(main())
```

**Example: Calling `cleanup`**

```python
# filepath: /path/to/your/scripts/cleanup_example.py
import asyncio
from src.topology.crew import BaseCrew

# Assume 'crew_instance' is an already initialized BaseCrew object

async def shutdown_crew(crew_instance: BaseCrew):
    if crew_instance:
        print("\nInitiating crew cleanup...")
        try:
            await crew_instance.cleanup()
            print("Crew cleanup finished.")
        except Exception as e:
            print(f"Error during crew cleanup: {e}")
    else:
        print("No crew instance to clean up.")

# Conceptual main execution flow
# async def main():
#     # ... (Initialize crew) ...
#     crew = await initialize_crew()
#     if crew:
#         # ... (Run tasks) ...
#         await shutdown_crew(crew)
# asyncio.run(main())
```

## 4. Advanced Topics

This section delves into more complex usage patterns and customization options for the Crew framework.

*   **Configuration Loading:**

    As mentioned, `AgentConfig` is a Pydantic model. This makes loading configurations from structured files like YAML or JSON straightforward and provides automatic validation.

    **Why use external configuration?**
    *   **Separation of Concerns:** Keeps agent definitions separate from the core application logic.
    *   **Reusability:** Easily reuse or modify crew setups without changing Python code.
    *   **Clarity:** Provides a clear overview of the agents, their roles (prompts), models, and permissions.

    **How to load from YAML:**
    You can use libraries like `PyYAML` along with Pydantic's `TypeAdapter` to load and validate a list of agent configurations. Remember to handle potential environment variable substitution for sensitive data like API keys.

    ```python
    # filepath: /path/to/your/scripts/load_config_example.py
    import yaml
    from pydantic import TypeAdapter, ValidationError
    from typing import List
    import os
    import logging
    # Assuming AgentConfig is importable
    from src.topology.crew import AgentConfig # Adjust import path as needed

    logging.basicConfig(level=logging.INFO)

    def load_agent_configs_from_yaml(filepath: str) -> List[AgentConfig]:
        """Loads and validates agent configurations from a YAML file."""
        logging.info(f"Attempting to load agent configurations from: {filepath}")
        try:
            with open(filepath, 'r') as f:
                # Use os.path.expandvars for simple ${VAR} or $VAR substitution
                # For more complex templating, consider libraries like Jinja2
                raw_content = f.read()
                expanded_content = os.path.expandvars(raw_content)
                logging.debug(f"Expanded YAML content:\n{expanded_content[:500]}...") # Log truncated content
                raw_configs = yaml.safe_load(expanded_content)

            if not isinstance(raw_configs, list):
                raise TypeError(f"Expected YAML root to be a list of agent configs, got {type(raw_configs)}")

            # Use Pydantic's TypeAdapter for validating the list of AgentConfig objects
            adapter = TypeAdapter(List[AgentConfig])
            validated_configs = adapter.validate_python(raw_configs)
            logging.info(f"Successfully loaded and validated {len(validated_configs)} agent configs.")
            return validated_configs
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {filepath}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {filepath}: {e}")
            raise
        except ValidationError as e:
            logging.error(f"Validation error loading agent configurations from {filepath}:\n{e}")
            raise
        except TypeError as e:
            logging.error(f"Type error during configuration loading (check YAML structure): {e}")
            raise
        except Exception as e: # Catch other potential errors
            logging.error(f"Unexpected error loading agent configurations from {filepath}: {e}")
            raise

    # --- Example Usage ---
    # async def main():
    #     try:
    #         # Ensure OPENAI_API_KEY is set in the environment before loading
    #         if not os.getenv("OPENAI_API_KEY"):
    #              print("Warning: OPENAI_API_KEY environment variable not set.")
    #              # Decide how to handle missing keys - raise error or use placeholder?
    #
    #         config_file = "/path/to/your/configs/crew_config.yaml" # Adjust path
    #         agent_configs = load_agent_configs_from_yaml(config_file)
    #
    #         # Now pass the loaded configs to BaseCrew.create
    #         crew = await BaseCrew.create(agent_configs=agent_configs)
    #         # ... use the crew ...
    #         await crew.cleanup()
    #
    #     except Exception as e:
    #         logging.error(f"Failed to initialize or run crew: {e}")
    #
    # # import asyncio
    # # asyncio.run(main())
    ```
    *(See the `AgentConfig` YAML example in Section 2 for the file structure)*

*   **Custom Agent Initialization:**

    The `BaseCrew._initialize_agents` method currently knows how to instantiate specific, hardcoded agent classes (`Agent`, `LearnableAgent`, `BrowserAgent`). If you create a custom agent subclass (e.g., `MySpecialAgent(Agent)`), `BaseCrew` won't automatically know how to instantiate it directly from the `agent_class` string in the config.

    **Workaround:**
    The recommended approach, as shown in the main `crew_example.py`, is:
    1.  **Instantiate Manually:** Create an instance of your custom agent *outside* the `BaseCrew.create` call. Ensure its constructor handles registration (which `BaseAgent.__init__` does).
    2.  **Use Placeholder Config:** In your `agent_configs` list passed to `BaseCrew.create`, include an `AgentConfig` entry for your custom agent, but use a known `agent_class` like `"Agent"` as a placeholder. Make sure the `name` matches the name you used (or let be generated) when manually instantiating your agent.
    3.  **Replace After Creation:** After `crew = await BaseCrew.create(...)` successfully returns, manually replace the placeholder agent instance in the `crew.agents` dictionary with your pre-initialized custom agent instance. You might need to unregister the placeholder first if names clash during manual instantiation vs. crew instantiation.

    ```python
    # Conceptual Example (building on crew_example.py)

    # 1. Define and Instantiate Custom Agent
    class MySpecialAgent(Agent):
        # ... custom methods, overrides ...
        async def _run(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any) -> Any:
            await self._log_progress(request_context, LogLevel.SUMMARY, f"MySpecialAgent running! Mode: {run_mode}")
            # ... custom logic ...
            return f"Special result for: {prompt}"

    # Instantiate manually (registers with AgentRegistry)
    special_agent_instance = MySpecialAgent(
        agent_name="SpecialOne", # Explicit name
        model_config=model_config_shared, # Use appropriate config
        system_prompt="I am a very special agent.",
        allowed_peers=["Executor"]
    )

    # 2. Define Configs (Use placeholder for custom agent)
    agent_configs = [
        AgentConfig(
            name="SpecialOne", # MUST match the manually instantiated name
            agent_class="Agent", # Placeholder class BaseCrew knows
            model_config=model_config_shared, # Provide config for placeholder init
            system_prompt="Placeholder prompt", # Doesn't matter much
            allowed_peers=["Executor"]
        ),
        AgentConfig(name="Executor", agent_class="Agent", ...), # Other agents
    ]

    # 3. Create Crew and Replace
    async def setup_special_crew():
        crew = await BaseCrew.create(agent_configs=agent_configs)

        # Check if placeholder exists and replace it
        if "SpecialOne" in crew.agents and isinstance(crew.agents["SpecialOne"], Agent):
            logging.info("Replacing placeholder 'SpecialOne' with custom instance.")
            # No need to unregister if names match and manual instance was created first
            crew.agents["SpecialOne"] = special_agent_instance
        else:
            logging.warning("Could not find placeholder 'SpecialOne' or it wasn't the expected type.")

        return crew

    # ... rest of the main execution logic using the modified crew ...
    ```
    This workaround allows leveraging `BaseCrew` for managing standard agents and task execution while incorporating custom agent types. A future enhancement to `BaseCrew` could involve a mechanism to register custom agent classes for direct loading.

*   **Reinforcement Learning:**

    The `learning_config: Optional[GRPOConfig]` parameter in `BaseCrew.create` is a placeholder for integrating reinforcement learning, specifically envisioned for algorithms like GRPO (Generalized Reward Policy Optimization).

    *   **`GRPOConfig`:** This would be a Pydantic model (defined in `src.learning.rl`) holding hyperparameters for the GRPO algorithm (learning rates, batch sizes, reward models, etc.).
    *   **Integration:** A `GRPOTrainer` class (also likely in `src.learning.rl`) would consume this configuration and interact with `LearnableAgent` instances within the crew (potentially via hooks or callbacks) to collect trajectories, compute rewards, and update the agents' learnable parameters (e.g., PEFT heads).
    *   **Current Status:** This integration is **not fully implemented** in the provided code structure. The `learning_config` parameter serves as a design intention for future development. Using it currently requires implementing the `GRPOTrainer` and the necessary interaction logic between the trainer and the agents.

*   **Task Control Flow:**

    `BaseCrew.run_task` provides a simple entry point by invoking the specified `initial_agent_name`. However, many tasks require more complex sequences of agent interactions than a single linear call.

    **Patterns for Complex Flows:**

    1.  **Internal Orchestration (Agent-Managed):** An agent (like the `PlannerAgent` example) takes responsibility for the control flow within its `_run` method. It calls other agents using `self.invoke_agent` based on its internal logic, loops, or the results of previous steps.
        *   **Pros:** Keeps task-specific logic encapsulated within the responsible agent.
        *   **Cons:** Can make the orchestrating agent complex; harder to get a high-level view of the entire task flow from outside the agent.

    2.  **External Orchestration (Function-Managed):** An external `async` function manages the overall task flow. This function might:
        *   Call `crew.run_task` once to get an initial plan or result.
        *   Iteratively call specific agents' `handle_invocation` methods directly (retrieving agents via `AgentRegistry.get(name)` or `crew.agents[name]`) based on the state or previous results.
        *   Manage loops, conditional logic, and data aggregation outside the agents.

    **Conceptual Example: External Orchestrator**

    ```python
    # filepath: /path/to/your/scripts/external_orchestrator_example.py
    import asyncio
    import logging
    from src.topology.crew import BaseCrew
    from src.agents.agents import RequestContext, LogLevel, AgentRegistry, ProgressUpdate

    # Assume 'crew' is an initialized BaseCrew instance
    # Assume 'consume_progress' function exists

    async def consume_progress(progress_queue: asyncio.Queue[Optional[ProgressUpdate]]):
        # ... (implementation from agents.md example) ...
        logging.info("Progress consumer started.")
        while True:
            update = await progress_queue.get()
            if update is None: logging.info("Progress consumer finished."); break
            logging.info(f"PROGRESS [{update.level.name}]: {update.message}")
            progress_queue.task_done()

    async def run_research_task_orchestrated(crew: BaseCrew, topic: str):
        """Orchestrates a research task using Planner and Executor agents."""
        task_id = f"research-{topic.replace(' ','-').lower()}-{uuid.uuid4()}"
        progress_queue = asyncio.Queue()
        progress_task = asyncio.create_task(consume_progress(progress_queue))

        planner = crew.agents.get("Planner")
        executor = crew.agents.get("Executor")

        if not planner or not executor:
            logging.error("Required agents (Planner, Executor) not found in crew.")
            await progress_queue.put(None); await progress_task
            return {"error": "Missing required agents."}

        logging.info(f"Starting orchestrated task {task_id} for topic: '{topic}'")

        # --- Step 1: Get Plan from Planner ---
        plan_result = None
        try:
            initial_context = RequestContext(
                task_id=task_id, initial_prompt=topic, progress_queue=progress_queue,
                log_level=LogLevel.SUMMARY, max_depth=5, max_interactions=20,
                interaction_id=str(uuid.uuid4()), caller_agent_name="orchestrator", callee_agent_name="Planner"
            )
            await ProgressLogger.log(initial_context, LogLevel.SUMMARY, f"Requesting plan for '{topic}' from Planner.")
            # Directly call handle_invocation
            plan_result = await planner.handle_invocation(
                request={"action": "plan", "prompt": f"Create a plan to research: {topic}"}, # Use dict request
                request_context=initial_context
            )
            logging.info(f"Received plan result: {str(plan_result)[:200]}...")
            # Basic validation (adapt based on Planner's actual output structure)
            if isinstance(plan_result, dict) and "final_results" in plan_result:
                 plan_steps = [step_info.get("step") for step_info in plan_result["final_results"] if step_info.get("status") == "success"] # Example extraction
                 if not plan_steps: raise ValueError("Planner returned no successful steps.")
            else:
                 # Handle cases where Planner might return steps directly or error structure
                 raise ValueError(f"Unexpected plan format received from Planner: {type(plan_result)}")

        except Exception as e:
            logging.error(f"Failed to get plan from Planner: {e}")
            await progress_queue.put(None); await progress_task
            return {"error": "Planning failed.", "details": str(e)}

        # --- Step 2: Execute Steps using Executor ---
        execution_summary = []
        logging.info(f"Executing {len(plan_steps)} steps...")
        for i, step in enumerate(plan_steps):
            try:
                # Create new context for each executor call, linked to the main task
                exec_context = RequestContext(
                    task_id=task_id, initial_prompt=topic, progress_queue=progress_queue,
                    log_level=LogLevel.SUMMARY, max_depth=5, max_interactions=20,
                    interaction_id=str(uuid.uuid4()),
                    # Depth could be 1 if orchestrator is depth 0, or higher if nested
                    depth=1, # Assuming orchestrator is depth 0
                    # Interaction count needs careful management in external orchestrator
                    interaction_count=i + 1, # Simple count for this example
                    caller_agent_name="orchestrator", callee_agent_name="Executor"
                )
                await ProgressLogger.log(exec_context, LogLevel.SUMMARY, f"Executing step {i+1}/{len(plan_steps)}: '{step}'")
                step_result = await executor.handle_invocation(
                    request=f"Execute: {step}",
                    request_context=exec_context
                )
                logging.info(f"Step {i+1} result: {str(step_result)[:150]}...")
                execution_summary.append({"step": step, "result": step_result})
            except Exception as e:
                logging.error(f"Failed to execute step {i+1} ('{step}'): {e}")
                execution_summary.append({"step": step, "error": str(e)})
                # Decide whether to break or continue
                break

        # --- Step 3: Final Aggregation (Optional) ---
        final_output = {
            "topic": topic,
            "plan": plan_steps,
            "execution_summary": execution_summary
        }
        logging.info(f"Orchestrated task {task_id} finished.")

        # Signal end and wait for progress consumer
        await progress_queue.put(None)
        await progress_queue.join()
        await progress_task

        return final_output

    # --- Conceptual Main ---
    # async def main():
    #     # ... load configs, create crew ...
    #     crew = await BaseCrew.create(...)
    #     if crew:
    #         try:
    #             final_research = await run_research_task_orchestrated(crew, "Quantum Computing impact on AI")
    #             print("\n--- Final Research Output ---")
    #             print(json.dumps(final_research, indent=2))
    #         finally:
    #             await crew.cleanup()
    # asyncio.run(main())
    ```
    *   **Pros:** Clear separation of task flow logic from agent capabilities; easier high-level understanding and modification of the flow.
    *   **Cons:** Can lead to more boilerplate code in the orchestrator; requires careful management of `RequestContext` (especially `depth` and `interaction_count`) if nesting calls deeply.

    Choosing between internal and external orchestration depends on the complexity of the task and the desired level of modularity.

---

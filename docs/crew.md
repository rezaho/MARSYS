# Crew Framework Documentation

This document explains the concepts and usage of the Crew framework defined in `src/topology/crew.py`.

## 1. Introduction

### What are Crews?

A "Crew" represents a collection of configured agents designed to work together to accomplish a larger task. The `BaseCrew` class acts as a manager for these agents.

### Why Crews?

*   **Configuration Driven:** Define the agents participating in a task, their types, models, prompts, and allowed interactions declaratively using `AgentConfig`.
*   **Simplified Task Execution:** Provides a single entry point (`run_task`) to initiate a task with a specific starting agent and prompt.
*   **Resource Management:** Handles the initialization of agents based on configurations and provides a `cleanup` method for releasing resources (like closing browser instances managed by `BrowserAgent`s).
*   **Centralized Setup:** Organizes the setup and orchestration of multiple agents for a given application.

## 2. Core Concepts

### `BaseCrew`

*   **Purpose:** Manages the lifecycle and execution of tasks within a group of configured agents.
*   **Initialization:** Takes a list of `AgentConfig` objects. During `__init__`, it calls `_initialize_agents` to create and register the agent instances.
*   **Key Attributes:**
    *   `agent_configs`: The list of configurations used to initialize the crew.
    *   `agents`: A dictionary mapping agent names to the live `BaseAgent` instances.
*   **Key Methods:**
    *   `_initialize_agents()`: Instantiates agents based on `agent_configs` and registers them with `AgentRegistry`.
    *   `run_task(initial_agent_name, initial_prompt, ...)`: The main method to start a task. It finds the initial agent, creates the initial `RequestContext`, calls the agent's `handle_invocation`, and returns the final result and progress queue.
    *   `cleanup()`: Iterates through agents and calls cleanup methods where necessary (e.g., `BrowserAgent.close_browser`).

### `AgentConfig` (Pydantic Model)

*   **Purpose:** Defines the configuration for a single agent within the crew. This allows defining the entire agent setup in a structured format (e.g., loaded from YAML or JSON).
*   **Key Attributes:**
    *   `name`: Unique name for the agent.
    *   `agent_class`: Specifies which agent class to use ('Agent', 'LearnableAgent', 'BrowserAgent').
    *   `model_config` / `model_ref`: Specifies the model to use (details depend on `agent_class`).
    *   `system_prompt`: Base prompt for the agent.
    *   `allowed_peers`: Crucial for defining the communication topology within the crew.
    *   Other attributes specific to agent types (e.g., `generation_system_prompt` for `BrowserAgent`).

## 3. Usage Examples

### Example 1: Defining Configuration and Running a Simple Task

```python
# filepath: /path/to/your/crew_script.py
import asyncio
import json
from src.topology.crew import BaseCrew, AgentConfig
from src.agents.agents import LogLevel, ProgressUpdate # Import ProgressUpdate

# 1. Define Agent Configurations
agent_configs = [
    AgentConfig(
        name="Researcher",
        agent_class="Agent", # Using the general Agent class
        model_config={ # Configuration for the Agent's model
            "type": "api",
            "name": "claude-3-sonnet-20240229", # Example API model
            "api_key": "YOUR_CLAUDE_KEY", # Use env variables
            # Add other necessary API params for Claude
        },
        system_prompt="You are a research assistant. Find information on the given topic.",
        allowed_peers=["Writer"] # Can call the Writer agent
    ),
    AgentConfig(
        name="Writer",
        agent_class="Agent",
        model_config={ # Can use the same or a different model
            "type": "api",
            "name": "gpt-3.5-turbo", # Example API model
            "api_key": "YOUR_OPENAI_KEY", # Use env variables
        },
        system_prompt="You are a writer. Summarize the provided information concisely.",
        allowed_peers=[] # Cannot call other agents
    )
]

# 2. Initialize the Crew
my_crew = BaseCrew(agent_configs=agent_configs)

# 3. Define the Async Task Execution
async def run_crew_task():
    initial_agent = "Researcher"
    prompt = "What are the main benefits of using asynchronous programming?"
    print(f"Starting task with agent '{initial_agent}' and prompt: '{prompt}'")

    try:
        # Run the task starting with the Researcher agent
        result, progress_queue = await my_crew.run_task(
            initial_agent_name=initial_agent,
            initial_prompt={ # Example using dict for potential future routing
                "action": "chat", # Explicitly start with chat mode
                "prompt": prompt
            },
            log_level=LogLevel.SUMMARY # Set desired log level
        )

        print("\n--- Task Finished ---")
        print(f"Final Result:\n{result}")

        print("\n--- Progress Updates ---")
        while True:
            update: Optional[ProgressUpdate] = await progress_queue.get()
            if update is None: # End signal
                break
            log_line = f"[{update.timestamp:.2f}][{update.level.name}]"
            if update.agent_name:
                log_line += f"[{update.agent_name}]"
            log_line += f" {update.message}"
            if update.data:
                log_line += f" Data: {json.dumps(update.data)}"
            print(log_line)

    except Exception as e:
        print(f"\n--- Task Failed ---")
        print(f"Error: {e}")
    finally:
        # 4. Cleanup Crew Resources (important for BrowserAgent, etc.)
        await my_crew.cleanup()

# 5. Run the Async Function
# asyncio.run(run_crew_task())
```

### Example 2: Crew with a BrowserAgent

```python
# filepath: /path/to/your/browser_crew_script.py
import asyncio
import json
from src.topology.crew import BaseCrew, AgentConfig
from src.agents.agents import LogLevel, ProgressUpdate, BrowserAgent # Import BrowserAgent for cleanup check

# 1. Define Agent Configurations including a BrowserAgent
agent_configs_with_browser = [
    AgentConfig(
        name="Planner",
        agent_class="Agent",
        model_config={"type": "api", "name": "gpt-4", "api_key": "YOUR_KEY"},
        system_prompt="Plan the steps to find information online using a browser agent.",
        allowed_peers=["Browser"]
    ),
    AgentConfig(
        name="Browser",
        agent_class="BrowserAgent", # Specify BrowserAgent
        model_config={"type": "api", "name": "gpt-4-turbo", "api_key": "YOUR_KEY"},
        # Provide specific prompts for BrowserAgent
        generation_system_prompt="You control a browser. Decide the next action (e.g., navigate, click, type).",
        critic_system_prompt="Evaluate the last browser action.",
        allowed_peers=[], # Browser might not need to call others directly
        memory_type="conversation_history" # Or another suitable type
    )
]

# 2. Initialize the Crew (Sync part)
# Note: This creates the BrowserAgent object but doesn't fully initialize the browser tool yet.
# For full initialization, the crew setup or task run needs to handle the async 'create' call.
# This example assumes a simplified setup where the BrowserAgent might be pre-initialized
# or the framework handles async init later. A more robust approach might involve an async crew init.
print("Initializing crew (BrowserAgent requires async 'create'/'create_safe' for full setup)...")
browser_crew = BaseCrew(agent_configs=agent_configs_with_browser)

# 3. Define Async Task Execution
async def run_browser_crew_task():
    # --- Crucial Step for BrowserAgent ---
    # Ensure the BrowserAgent's browser tool is initialized *before* running the task.
    # This might happen here, or during an async crew initialization process.
    browser_agent_instance = browser_crew.agents.get("Browser")
    if isinstance(browser_agent_instance, BrowserAgent) and not browser_agent_instance.browser_tool:
         print("Initializing BrowserAgent's browser tool asynchronously...")
         # Re-create using the async factory method to ensure browser is ready
         # This replaces the instance created synchronously by BaseCrew._initialize_agents
         # We need the original config used by the crew.
         browser_config = next(c for c in browser_crew.agent_configs if c.name == "Browser")
         browser_agent_instance = await BrowserAgent.create_safe(
             agent_name=browser_config.name,
             model_config=browser_config.model_config,
             generation_system_prompt=browser_config.generation_system_prompt,
             critic_system_prompt=browser_config.critic_system_prompt,
             memory_type=browser_config.memory_type,
             max_tokens=browser_config.max_tokens,
             allowed_peers=browser_config.allowed_peers,
             # Add temp_dir, headless settings if needed
             timeout=30
         )
         # Update the crew's dictionary (and AgentRegistry via BrowserAgent init)
         browser_crew.agents["Browser"] = browser_agent_instance
         print("Browser tool initialized.")
    # --- End BrowserAgent Init ---


    initial_agent = "Planner"
    prompt = "Find the current weather in London using the browser."
    print(f"Starting task with agent '{initial_agent}' and prompt: '{prompt}'")

    try:
        result, progress_queue = await browser_crew.run_task(
            initial_agent_name=initial_agent,
            initial_prompt=prompt,
            log_level=LogLevel.DETAILED,
            max_depth=5 # Allow Planner -> Browser -> LLM calls etc.
        )

        print("\n--- Task Finished ---")
        print(f"Final Result:\n{result}") # Result likely involves browser actions/summary

        # Process progress queue... (as in Example 1)

    except Exception as e:
        print(f"\n--- Task Failed ---")
        print(f"Error: {e}")
    finally:
        # Cleanup is essential for BrowserAgent
        await browser_crew.cleanup()

# 4. Run the Async Function
# asyncio.run(run_browser_crew_task())

```

# Custom Agents

Build specialized agents tailored to your specific domain and requirements.

## Overview

While the framework provides powerful base agents, creating custom agents allows you to:
- Implement domain-specific logic
- Add specialized tools and capabilities
- Optimize for particular use cases
- Integrate with external systems

## Creating a Custom Agent

### Basic Custom Agent

Inherit from `BaseAgent` or `Agent`:

```python
from src.agents.agents import BaseAgent # Corrected import
from src.models.message import Message
from src.models.types import RequestContext # Corrected import
from src.models.models import ModelConfig # Added for completeness
from src.utils.logging_utils import get_logger # Added for example usage

class CustomBaseStyleAgent(BaseAgent): # Renamed for clarity
    """Custom agent inheriting from BaseAgent with specialized behavior."""

    def __init__(self, name: str, model_config: ModelConfig, specialization: str, **kwargs): # Explicit params
        super().__init__(name=name, model_config=model_config, **kwargs) # Pass specific BaseAgent params
        self.specialization = specialization

    async def _run(self, request: Message, context: RequestContext) -> Message: # Aligned with BaseAgent
        """Custom task processing logic for BaseAgent."""
        context.logger.info(f"{self.name} processing task: '{request.content}' with specialization '{self.specialization}'")

        result_content = f"Processed '{request.content}' with {self.specialization} specialization by {self.name}"

        return Message(
            role="assistant",
            content=result_content,
            name=self.name
        )

# Example conceptual usage (not for direct execution in docs):
# async def main_base_custom():
#     logger = get_logger("CustomBaseStyleAgentTest")
#     # Ensure you have a valid ModelConfig, e.g., for a dummy or a configured LLM
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model") 
#     custom_agent = CustomBaseStyleAgent(
#         name="base_ops_agent",
#         model_config=model_cfg,
#         specialization="simple_tasks"
#     )
#     req_context = RequestContext(request_id="test_base_agent", logger=logger)
#     user_message = Message(role="user", content="Perform a simple task.")
#     response_message = await custom_agent.run(user_message, req_context) # Use the public run method
#     print(f"Agent Response: {response_message.content}")
```

### Agent with State Management

```python
from src.agents.agents import Agent # Corrected import
from src.models.models import ModelConfig # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import get_logger # Added for example usage
from typing import Dict, Any, Optional
import json

class StatefulAgent(Agent):
    """Agent that maintains internal state."""

    def __init__(self, name: str, model_config: ModelConfig, state_file_name: Optional[str] = None, **kwargs): # Explicit Agent params
        super().__init__(name=name, model_config=model_config, **kwargs) # Pass specific Agent params
        self.state: Dict[str, Any] = {}
        self.state_file = state_file_name or f"{self.name}_state.json" # Use provided name or default
        self._load_state()

    def _load_state(self):
        """Load state from persistent storage."""
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        except FileNotFoundError:
            # It's okay if the state file doesn't exist yet
            self.state = {}
        except json.JSONDecodeError:
            # Handle cases where the file is corrupted
            print(f"Warning: Could not decode JSON from state file {self.state_file}. Starting with empty state.")
            self.state = {}


    def _save_state(self):
        """Save state to persistent storage."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2) # Added indent for readability

    async def update_state(self, key: str, value: Any, context: RequestContext) -> None:
        """Update internal state and save."""
        context.logger.info(f"Updating state: {key} = {value}")
        self.state[key] = value
        self._save_state()

    async def get_state(self, key: str, context: RequestContext) -> Optional[Any]:
        """Retrieve state value."""
        value = self.state.get(key)
        context.logger.info(f"Retrieved state for key '{key}': {'Found' if value is not None else 'Not found'}")
        return value

# Example conceptual usage:
# async def main_stateful():
#     logger = get_logger("StatefulAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model")
#     stateful_agent = StatefulAgent(name="state_keeper", model_config=model_cfg)
#     req_context = RequestContext(request_id="test_stateful", logger=logger)
#
#     await stateful_agent.update_state("user_preference", "dark_mode", req_context)
#     pref = await stateful_agent.get_state("user_preference", req_context)
#     print(f"User preference: {pref}")
```

### Domain-Specific Agent

```python
from src.agents.agents import Agent # Corrected import
from src.models.models import ModelConfig # Added
from src.models.message import Message # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import get_logger # Added for example usage
from typing import AsyncGenerator # Added for _process_request
import json

class DataAnalystAgent(Agent):
    """Specialized agent for data analysis tasks."""

    def __init__(self, name: str, model_config: ModelConfig, **kwargs): # Explicit Agent params
        custom_instructions = """
        You are a data analyst agent. Your responsibilities:
        1. Analyze datasets and identify patterns.
        2. Generate statistical summaries.
        3. Create visualizations (describe them in text).
        4. Provide actionable insights based on data.
        Always be precise with numbers and cite your calculations if possible.
        """
        # Combine with any instructions passed via kwargs or set as default
        existing_instructions = kwargs.pop('instructions', "")
        instructions = f"{existing_instructions}\n{custom_instructions}".strip()

        # Prepare specialized tools
        specialized_tools = {
            "calculate_statistics": self._calculate_statistics,
            "detect_anomalies": self._detect_anomalies, # Assuming implementation
            "generate_report_outline": self._generate_report_outline # Assuming implementation
        }
        # Combine with any tools passed in kwargs
        existing_tools = kwargs.pop('tools', {})
        all_tools = {**existing_tools, **specialized_tools}

        super().__init__(name=name, model_config=model_config, instructions=instructions, tools=all_tools, **kwargs)

    async def _calculate_statistics(self, data_csv_string: str, context: RequestContext) -> str:
        """Calculate basic statistics for a dataset provided as a CSV string."""
        context.logger.info(f"Calculating statistics for data: {data_csv_string[:50]}...")
        try:
            # Example: expects comma-separated numbers
            values = [float(x.strip()) for x in data_csv_string.split(',') if x.strip()]
            if not values:
                return "Error: No numerical data provided."
            stats = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "median": sorted(values)[len(values) // 2] if values else 0, # Simplified median
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values)
            }
            return json.dumps(stats, indent=2)
        except ValueError:
            return "Error: Data contains non-numeric values."
        except Exception as e:
            context.logger.error(f"Error in _calculate_statistics: {e}", exc_info=True)
            return f"Error calculating statistics: {str(e)}"

    async def _detect_anomalies(self, data_csv_string: str, threshold_std_dev: float = 2.0, context: RequestContext) -> str:
        """Detect anomalies using simple standard deviation method."""
        context.logger.info(f"Detecting anomalies with threshold {threshold_std_dev} std dev.")
        # Placeholder implementation - replace with actual anomaly detection logic
        # This would typically involve parsing data, calculating mean/std dev, and identifying outliers.
        # For now, returns a placeholder message.
        # Example:
        # values = [float(x.strip()) for x in data_csv_string.split(',') if x.strip()]
        # mean = sum(values) / len(values)
        # std_dev = (sum([(x - mean) ** 2 for x in values]) / len(values)) ** 0.5
        # anomalies = [v for v in values if abs(v - mean) > threshold_std_dev * std_dev]
        # return json.dumps({"anomalies_found": anomalies, "count": len(anomalies)})
        return json.dumps({"message": f"Anomaly detection for data (first 50 chars: {data_csv_string[:50]}) with threshold {threshold_std_dev} not fully implemented in this example."})

    async def _generate_report_outline(self, analysis_summary_json: str, context: RequestContext) -> str:
        """Generate a formatted analysis report outline based on a summary."""
        context.logger.info("Generating report outline.")
        # Placeholder implementation
        try:
            analysis_data = json.loads(analysis_summary_json)
            outline = f"# Analysis Report Outline\n\n## 1. Introduction\n   - Purpose of analysis\n\n## 2. Key Findings\n   - Summary: {analysis_data.get('summary', 'N/A')}\n\n## 3. Detailed Sections (based on available data)\n"
            if 'stats' in analysis_data:
                outline += f"   - Statistical Overview: {analysis_data['stats']}\n"
            if 'anomalies_found' in analysis_data:
                 outline += f"   - Anomaly Detection Results: {analysis_data['anomalies_found']}\n"
            outline += "\n## 4. Conclusion\n   - Main takeaways"
            return outline
        except Exception as e:
            context.logger.error(f"Error in _generate_report_outline: {e}", exc_info=True)
            return f"Error generating report outline: {str(e)}"

# Example conceptual usage:
# async def main_data_analyst():
#     logger = get_logger("DataAnalystAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model") # Use a real model for actual LLM capabilities
#     analyst_agent = DataAnalystAgent(name="data_cruncher", model_config=model_cfg)
#
#     req_context = RequestContext(request_id="test_data_analysis", logger=logger)
#     # Simulate a request that might involve tool use
#     # The Agent's _process_request would handle LLM interaction and tool calls.
#     # For direct tool call testing (if tools were public/exposed):
#     # stats_result = await analyst_agent._calculate_statistics("1,2,3,4,5,100", req_context)
#     # print(f"Stats: {stats_result}")
#
#     # To test the full agent flow:
#     user_query = Message(role="user", content="Calculate statistics for the data: 10,20,30,25,15 and generate a report outline.")
#     async for response in analyst_agent.process_request(user_query, req_context):
#         print(f"Analyst response: {response.content}")

```

## Advanced Custom Agent Patterns

### Plugin-Based Agent

```python
from src.agents.agents import Agent # Corrected
from src.models.models import ModelConfig # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import get_logger # Added
from abc import ABC, abstractmethod
from typing import List, Dict, Callable # Corrected 'callable' to 'Callable'

class AgentPlugin(ABC):
    """Base class for agent plugins."""

    @abstractmethod
    def get_tools(self) -> Dict[str, Callable]: # Corrected 'callable'
        """Return tools provided by this plugin."""
        pass

    @abstractmethod
    def get_instructions_delta(self) -> str: # Renamed for clarity
        """Return additional instructions provided by this plugin."""
        pass

class PluggableAgent(Agent):
    """Agent that supports plugins."""

    def __init__(self, name: str, model_config: ModelConfig, plugins: List[AgentPlugin] = None, **kwargs): # Explicit Agent params
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.plugins = plugins or []
        
        # Aggregate instructions and tools from plugins
        plugin_instructions_parts = [kwargs.get('instructions', "")]
        combined_tools = kwargs.get('tools', {}).copy()

        for plugin in self.plugins:
            plugin_tools = plugin.get_tools()
            for tool_name, tool_impl in plugin_tools.items():
                if tool_name in combined_tools:
                    # Handle potential tool name conflicts, e.g., log a warning or raise error
                    print(f"Warning: Tool '{tool_name}' from plugin is overwriting an existing tool.")
                combined_tools[tool_name] = tool_impl
            plugin_instructions_parts.append(plugin.get_instructions_delta())
        
        final_instructions = "\n\n".join(filter(None, plugin_instructions_parts))

        super().__init__(name=name, model_config=model_config, instructions=final_instructions, tools=combined_tools, **kwargs)
        # self._load_plugins() # Logic moved to __init__ for proper setup before super().__init__

    # def _load_plugins(self): # This logic is now integrated into __init__
    #     """Load tools and instructions from plugins."""
    #     # ...

    def add_plugin(self, plugin: AgentPlugin, context: RequestContext): # Added context for logging
        """Dynamically add a plugin after initialization."""
        context.logger.info(f"Adding plugin: {type(plugin).__name__}")
        self.plugins.append(plugin)
        
        # Update tools
        new_tools = plugin.get_tools()
        for tool_name, tool_impl in new_tools.items():
            if tool_name in self.tools:
                context.logger.warning(f"Dynamically added plugin tool '{tool_name}' is overwriting an existing tool.")
            self.add_tool(tool_name, tool_impl) # Use Agent's add_tool method

        # Append plugin instructions
        # Note: Modifying self.instructions directly after init might not be reflected
        # in the LLM context without re-initializing or specific handling by the LLM provider.
        # For simplicity, we'll append here; a more robust solution might involve
        # re-constructing the prompt or having the LLM provider support dynamic instruction updates.
        plugin_instr_delta = plugin.get_instructions_delta()
        if plugin_instr_delta:
            self.instructions = f"{self.instructions}\n\n{plugin_instr_delta}".strip()
            context.logger.info(f"Appended instructions from plugin {type(plugin).__name__}.")


# Example plugin
class DatabasePlugin(AgentPlugin):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        # In a real scenario, initialize db connection or pool here if needed by tools

    def get_tools(self) -> Dict[str, Callable]: # Corrected 'callable'
        return {
            "query_database": self._query_database,
            # "update_database": self._update_database # Assuming implementation
        }

    def get_instructions_delta(self) -> str: # Renamed
        return """
        Database Plugin Instructions:
        - Use 'query_database' to retrieve data using SQL.
        - Ensure SQL queries are safe and targeted.
        """
        # - Use 'update_database' to modify records (if implemented)

    async def _query_database(self, query: str, context: RequestContext) -> str: # Added context
        """Execute a SQL query against the database."""
        context.logger.info(f"Querying database with: {query[:100]}...")
        # Database query implementation (e.g., using asyncpg, sqlalchemy)
        # This is a placeholder.
        # For example:
        # try:
        #   conn = await asyncpg.connect(self.connection_string)
        #   result = await conn.fetch(query)
        #   await conn.close()
        #   return json.dumps([dict(row) for row in result], default=str)
        # except Exception as e:
        #   context.logger.error(f"Database query error: {e}", exc_info=True)
        #   return f"Error querying database: {str(e)}"
        return f"Executed query (simulated): {query}. Results would be here."

# Example conceptual usage:
# async def main_pluggable():
#     logger = get_logger("PluggableAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model")
#     db_plugin = DatabasePlugin(connection_string="postgresql://user:pass@host:port/db")
#
#     pluggable_agent = PluggableAgent(
#         name="plugin_master",
#         model_config=model_cfg,
#         plugins=[db_plugin],
#         instructions="You are a helpful assistant with plugin capabilities."
#     )
#     req_context = RequestContext(request_id="test_pluggable", logger=logger)
#     print(f"Agent Instructions: {pluggable_agent.instructions}")
#     print(f"Agent Tools: {list(pluggable_agent.tools.keys())}")
#
#     # To test the agent flow:
#     # user_query = Message(role="user", content="Query the database for all users.")
#     # async for response in pluggable_agent.process_request(user_query, req_context):
#     #     print(f"Pluggable Agent response: {response.content}")

```

### Reactive Agent

```python
from src.agents.agents import Agent # Corrected
from src.models.models import ModelConfig # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import LogLevel, get_logger # Added LogLevel, get_logger
import asyncio
from typing import Callable, Dict, List, Any # Corrected 'callable', added Any

class ReactiveAgent(Agent):
    """Agent that reacts to events from an internal queue."""

    def __init__(self, name: str, model_config: ModelConfig, **kwargs): # Explicit Agent params
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.event_handlers: Dict[str, List[Callable[[Any, RequestContext], None]]] = {} # Handler takes data and context
        self.event_queue = asyncio.Queue()
        self.running = False
        self._event_processing_task: Optional[asyncio.Task] = None
        # It's good practice to have a dedicated logger for the agent instance
        self.logger = kwargs.get('logger', get_logger(f"ReactiveAgent_{name}"))


    def on(self, event_type: str, handler: Callable[[Any, RequestContext], None]): # Changed 'event' to 'event_type' for clarity
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Registered handler for event type '{event_type}'.")

    async def emit(self, event_type: str, data: Any = None): # Changed 'event' to 'event_type'
        """Emit an event to the internal queue."""
        await self.event_queue.put((event_type, data))
        self.logger.debug(f"Emitted event '{event_type}' with data: {str(data)[:50]}...")

    async def start_event_loop(self, parent_context: RequestContext): # Renamed, added parent_context
        """Start the event processing loop."""
        if not self.running:
            self.running = True
            # Create a derived RequestContext for the event loop if needed, or use parent_context's logger
            loop_logger = parent_context.logger or self.logger
            loop_context = RequestContext(request_id=f"{parent_context.request_id}_event_loop", logger=loop_logger)
            self._event_processing_task = asyncio.create_task(self._process_events(loop_context))
            self.logger.info("Event processing loop started.")
        else:
            self.logger.info("Event processing loop already running.")


    async def stop_event_loop(self):
        """Stop event processing."""
        if self.running:
            self.running = False
            if self._event_processing_task:
                try:
                    # Give the loop a chance to finish processing the current event
                    await asyncio.wait_for(self._event_processing_task, timeout=2.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Event processing loop did not finish cleanly on stop.")
                    self._event_processing_task.cancel()
                except asyncio.CancelledError:
                    self.logger.info("Event processing task was cancelled.")
            self.logger.info("Event processing loop stopped.")


    async def _process_events(self, context: RequestContext): # Takes RequestContext
        """Process events from the queue."""
        while self.running:
            try:
                event_type, data = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0  # Check for running flag periodically
                )
                context.logger.debug(f"Processing event '{event_type}'.")
                if event_type in self.event_handlers:
                    for handler in self.event_handlers[event_type]:
                        try:
                            # Pass both data and the current processing context to the handler
                            await handler(data, context)
                        except Exception as e:
                            # Use self._log_progress (from BaseAgent) for structured logging if appropriate
                            # For direct logging here:
                            context.logger.error(f"Event handler for '{event_type}' raised an error: {e}", exc_info=True)
                            # Example using _log_progress if this agent has LLM interaction context
                            # error_context = RequestContext(request_id=f"{context.request_id}_handler_error", logger=context.logger)
                            # await self._log_progress(error_context, LogLevel.ERROR, f"Event handler error for '{event_type}': {e}")
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                # This is normal, allows checking self.running
                continue
            except Exception as e:
                context.logger.error(f"Unexpected error in event processing loop: {e}", exc_info=True)
                # Avoid continuous error loops by adding a small delay
                await asyncio.sleep(0.1)
        context.logger.info("Event processing loop finished.")

# Example conceptual usage:
# async def my_event_handler(data: Any, context: RequestContext):
#     context.logger.info(f"Event handler received data: {data}")
#     # Do something with the data
#
# async def main_reactive():
#     logger = get_logger("ReactiveAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model")
#     reactive_agent = ReactiveAgent(name="event_listener", model_config=model_cfg, logger=logger)
#
#     reactive_agent.on("user_login", my_event_handler)
#
#     # Create a main context for starting the loop
#     main_req_context = RequestContext(request_id="main_reactive_test", logger=logger)
#     await reactive_agent.start_event_loop(main_req_context)
#
#     await reactive_agent.emit("user_login", {"user_id": "123", "timestamp": "2023-01-01T12:00:00Z"})
#     await asyncio.sleep(0.1) # Allow event to be processed
#
#     await reactive_agent.stop_event_loop()

```

### Learning Agent Implementation

This section should demonstrate how to create a custom agent that extends `LearnableAgent` from `src.agents.learnable_agents`.

```python
from src.agents.learnable_agents import LearnableAgent # Corrected import
from src.models.models import ModelConfig
from src.models.types import RequestContext, Feedback, Message # Added Feedback, Message
from src.agents.adaptation_strategies import AdaptationStrategy, SimpleAdaptationStrategy # Example strategy
from src.agents.feedback_processors import FeedbackProcessor, BasicFeedbackProcessor # Example processor
from src.utils.logging_utils import get_logger # Added for example usage
from datetime import datetime, timezone # Added timezone for utcnow
from typing import Dict, Any, Optional, AsyncGenerator # Added Optional, AsyncGenerator

class MyCustomLearningAgent(LearnableAgent):
    """
    Custom learning agent that extends LearnableAgent.
    It can have its own specific parameters, override learning behaviors,
    or add new capabilities while leveraging the base learning framework.
    """

    def __init__(self,
                 name: str,
                 model_config: ModelConfig,
                 custom_learning_param: str = "default_value",
                 # Pass LearnableAgent specific params or allow defaults
                 learning_rate: float = 0.05,
                 feedback_processor: Optional[FeedbackProcessor] = None,
                 adaptation_strategy: Optional[AdaptationStrategy] = None,
                 **kwargs):

        # Initialize default feedback processor and adaptation strategy if not provided
        # These are components used by LearnableAgent's learning loop.
        effective_feedback_processor = feedback_processor or BasicFeedbackProcessor()
        effective_adaptation_strategy = adaptation_strategy or SimpleAdaptationStrategy(learning_rate=learning_rate)

        super().__init__(name=name,
                         model_config=model_config,
                         learning_rate=learning_rate, # Passed to super for base class use
                         feedback_processor=effective_feedback_processor,
                         adaptation_strategy=effective_adaptation_strategy,
                         **kwargs)
        self.custom_learning_param = custom_learning_param
        self.logger = kwargs.get('logger', get_logger(f"MyCustomLearningAgent_{name}"))
        self.logger.info(f"Initialized MyCustomLearningAgent with custom_param: {self.custom_learning_param}")
        # experience_buffer and performance_history are managed by LearnableAgent

    async def _process_request(self, request: Message, context: RequestContext) -> AsyncGenerator[Message, None]:
        """
        Override Agent's _process_request if custom pre/post LLM call logic is needed.
        The LearnableAgent itself doesn't override _process_request by default.
        This is where the agent's primary task execution happens.
        """
        self.logger.info(f"{self.name} (learning enabled) processing request: {request.content[:50]}...")
        # Example: Add custom learning param to the context for LLM
        # (This depends on how your LLM prompt incorporates context)
        # enriched_prompt_context = {
        #     **(self.system_prompt_context or {}),
        #     "custom_learning_info": f"Current custom param: {self.custom_learning_param}"
        # }
        # Call super()._process_request or reimplement LLM call logic
        # For simplicity, let's assume we call super's _process_request
        # which handles the main LLM interaction and tool use.
        # To do this, we'd need to ensure system_prompt_context is correctly passed.
        # original_spc = self.system_prompt_context
        # self.system_prompt_context = enriched_prompt_context
        
        async for message in super()._process_request(request, context):
            yield message
        
        # self.system_prompt_context = original_spc # Restore
        self.logger.info(f"{self.name} finished processing request.")


    async def process_feedback(self, feedback: Feedback, context: RequestContext) -> None:
        """
        Override to customize how feedback is processed before learning.
        The base LearnableAgent.process_feedback stores the feedback in experience_buffer
        and then calls adapt_behavior.
        """
        self.logger.info(f"{self.name} received feedback (score: {feedback.score}) for interaction {feedback.interaction_id}. Custom param: {self.custom_learning_param}")
        
        # Custom logic before base processing:
        if feedback.score < 0.2 and "urgent" in feedback.metadata.get("tags", []):
            self.logger.warning("Urgent low score feedback received! Triggering immediate alert (simulated).")
            # self.custom_learning_param = "under_review_urgent" # Example custom state change

        # Call super().process_feedback to ensure base class logic runs
        # (e.g., adding to experience_buffer, calling adapt_behavior).
        await super().process_feedback(feedback, context)

        self.logger.info(f"{self.name} finished custom feedback processing for {feedback.interaction_id}.")


    async def adapt_behavior(self, feedback: Feedback, context: RequestContext) -> None:
        """
        Override to customize how the agent adapts its behavior.
        The base LearnableAgent.adapt_behavior uses the configured AdaptationStrategy.
        You can:
        1. Provide a custom AdaptationStrategy in __init__.
        2. Override this method for entirely custom adaptation logic,
           optionally still calling super().adapt_behavior or self.adaptation_strategy.execute().
        """
        self.logger.info(f"{self.name} adapting behavior based on feedback for {feedback.interaction_id}. Current LR: {self.learning_rate}")

        # Example custom adaptation:
        if self.custom_learning_param == "aggressive_mode" and feedback.score > 0.8:
            # If in aggressive mode and performing well, slightly increase learning rate
            # self.learning_rate = min(0.5, self.learning_rate * 1.05) # Managed by strategy now
            self.logger.info("Aggressive mode positive feedback: strategy will handle LR.")

        # Call the adaptation strategy defined in LearnableAgent (which uses self.adaptation_strategy)
        await super().adapt_behavior(feedback, context)

        # Example: Post-adaptation logic
        self.logger.info(f"{self.name} finished adapting behavior. New LR (from strategy): {self.learning_rate}, Instructions length: {len(self.instructions)}")


    def get_custom_performance_summary(self) -> Dict[str, Any]:
        """Get a custom summary of agent performance, including inherited metrics."""
        # get_performance_metrics() is available from LearnableAgent
        base_metrics = self.get_performance_metrics()
        
        summary = {
            "custom_learning_parameter": self.custom_learning_param,
            "total_interactions_in_buffer": len(self.experience_buffer), # experience_buffer from LearnableAgent
        }
        # Combine with base metrics
        summary.update(base_metrics)
        return summary

# Example conceptual usage:
# async def main_custom_learning():
#     logger = get_logger("MyCustomLearningAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model") # Use a real model for LLM features
#
#     # Optionally, define a custom strategy or processor
#     # class MyStrategy(SimpleAdaptationStrategy):
#     #     async def execute(self, agent: 'LearnableAgent', feedback: Feedback, context: RequestContext):
#     #         await super().execute(agent, feedback, context)
#     #         agent.instructions += "\n[System Note] My custom strategy was here!"
#
#     learning_agent = MyCustomLearningAgent(
#         name="adaptive_learner",
#         model_config=model_cfg,
#         custom_learning_param="initial_phase",
#         learning_rate=0.02
#         # adaptation_strategy=MyStrategy(learning_rate=0.02) # Example of custom strategy
#     )
#
#     req_context = RequestContext(request_id="learn_test_001", logger=logger)
#
#     # Simulate an agent interaction (e.g., processing a user request)
#     user_request = Message(role="user", content="Tell me a joke.")
#     interaction_id = req_context.request_id # Or a more specific ID
#     
#     # In a real scenario, the agent's response comes from its _process_request method
#     # For this example, let's assume a dummy response message
#     agent_response_content = "Why don't scientists trust atoms? Because they make up everything!"
#     agent_response_message = Message(role="assistant", content=agent_response_content, name=learning_agent.name, interaction_id=interaction_id)
#
#     # Simulate receiving feedback for this interaction
#     feedback_data = Feedback(
#         interaction_id=interaction_id,
#         request_message=user_request, 
#         response_message=agent_response_message,
#         score=0.9, # Example score (0.0 to 1.0)
#         comment="Good joke, very relevant!",
#         timestamp=datetime.now(timezone.utc), # Use timezone-aware datetime
#         metadata={"task_type": "creative", "user_rating": 5}
#     )
#
#     await learning_agent.process_feedback(feedback_data, req_context)
#
#     # Check performance
#     performance_summary = learning_agent.get_custom_performance_summary()
#     print(f"Performance Summary: {performance_summary}")
#
#     # The agent's instructions or other parameters might have changed due to adaptation
#     # print(f"Updated Instructions: {learning_agent.instructions}")
```

## Integration Patterns

### External API Integration

```python
from src.agents.agents import Agent # Corrected
from src.models.models import ModelConfig # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import get_logger # Added
import aiohttp # Ensure this is installed in your environment
import json

class APIIntegrationAgent(Agent):
    """Agent that integrates with external APIs using aiohttp."""

    def __init__(self, name: str, model_config: ModelConfig, api_key: str, api_base_url: str, **kwargs): # Explicit Agent params
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.api_key = api_key
        self.api_base_url = api_base_url
        self._session: Optional[aiohttp.ClientSession] = None # Private attribute for session
        self.logger = kwargs.get('logger', get_logger(f"APIIntegrationAgent_{name}"))

        # Add API tools - tools should be passed to super().__init__ or added via self.add_tool
        # For this example, let's assume they are added post-init for clarity,
        # though passing to constructor is often cleaner.
        self.add_tool("fetch_data_from_endpoint", self._fetch_data)
        self.add_tool("post_data_to_endpoint", self._post_data) # Assuming _post_data implementation

    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns the aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            # You might want to configure timeouts, SSL verification, etc.
            self._session = aiohttp.ClientSession(headers=headers)
            self.logger.info("aiohttp.ClientSession initialized.")
        return self._session

    async def close_session(self):
        """Closes the aiohttp ClientSession if it's open."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("aiohttp.ClientSession closed.")
            self._session = None # Reset to allow re-creation

    # It's good practice to manage session lifecycle, e.g., via context manager protocols
    # or explicit start/stop methods if the agent has a defined lifecycle.
    # For simplicity, this example initializes on first use and provides a close method.

    async def _fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None, context: RequestContext) -> str:
        """Fetch data from an API endpoint."""
        session = await self._get_session()
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        context.logger.info(f"Fetching data from API: {url} with params: {params}")
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()  # Raises an exception for 4XX/5XX responses
                data = await response.json() # Assumes JSON response
                return json.dumps(data, indent=2)
        except aiohttp.ClientResponseError as e:
            context.logger.error(f"API ClientResponseError: {e.status} {e.message} for URL {url}")
            return f"API Error: {e.status} - {e.message}"
        except aiohttp.ClientError as e: # Handles other client errors like connection issues
            context.logger.error(f"API ClientError: {e} for URL {url}")
            return f"API Connection Error: {str(e)}"
        except Exception as e:
            context.logger.error(f"Unexpected error fetching API data: {e}", exc_info=True)
            return f"Unexpected API Error: {str(e)}"

    async def _post_data(self, endpoint: str, data: Dict[str, Any], context: RequestContext) -> str:
        """Post data to an API endpoint."""
        session = await self._get_session()
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        context.logger.info(f"Posting data to API: {url} with data: {str(data)[:100]}...")
        try:
            async with session.post(url, json=data) as response: # Sending data as JSON
                response.raise_for_status()
                response_data = await response.json() # Assumes JSON response
                return json.dumps(response_data, indent=2)
        except aiohttp.ClientResponseError as e:
            context.logger.error(f"API ClientResponseError on POST: {e.status} {e.message} for URL {url}")
            return f"API Error on POST: {e.status} - {e.message}"
        except aiohttp.ClientError as e:
            context.logger.error(f"API ClientError on POST: {e} for URL {url}")
            return f"API Connection Error on POST: {str(e)}"
        except Exception as e:
            context.logger.error(f"Unexpected error posting API data: {e}", exc_info=True)
            return f"Unexpected API Error on POST: {str(e)}"

# Example conceptual usage:
# async def main_api_integration():
#     logger = get_logger("APIIntegrationAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model") # Use real model for LLM interaction
#     
#     # Replace with your actual API details
#     api_agent = APIIntegrationAgent(
#         name="api_user",
#         model_config=model_cfg,
#         api_key="YOUR_API_KEY",
#         api_base_url="https://api.example.com/v1",
#         logger=logger
#     )
#     req_context = RequestContext(request_id="test_api_call", logger=logger)
#
#     # Example: Fetch data (replace 'items' and params with actual endpoint and parameters)
#     # fetched_data_str = await api_agent._fetch_data("items", {"limit": 5}, req_context)
#     # print(f"Fetched data: {fetched_data_str}")
#
#     # Example: Post data (replace 'items' and payload with actual endpoint and data)
#     # new_item_payload = {"name": "New Gadget", "price": 99.99}
#     # post_response_str = await api_agent._post_data("items", new_item_payload, req_context)
#     # print(f"Post response: {post_response_str}")
#
#     # To test the full agent flow (LLM deciding to use the tool):
#     # user_query = Message(role="user", content="Fetch the first 5 items from the API.")
#     # async for response in api_agent.process_request(user_query, req_context):
#     #     print(f"API Agent LLM response: {response.content}")
#
#     await api_agent.close_session() # Important to clean up session

```

### Database Integration

```python
from src.agents.agents import Agent # Corrected
from src.models.models import ModelConfig # Added
from src.models.types import RequestContext # Added
from src.utils.logging_utils import get_logger # Added
import asyncpg # Ensure this is installed in your environment
import json
from typing import Optional, List, Dict, Any # Added for type hints

class DatabaseAgent(Agent):
    """Agent with database capabilities using asyncpg."""

    def __init__(self, name: str, model_config: ModelConfig, db_url: str, **kwargs): # Explicit Agent params
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None # Private attribute for connection pool
        self.logger = kwargs.get('logger', get_logger(f"DatabaseAgent_{name}"))

        # Add database tools
        self.add_tool("execute_sql_query", self._query) # Renamed for clarity
        self.add_tool("execute_sql_insert", self._insert) # Assuming _insert implementation
        # self.add_tool("execute_sql_update", self._update) # Assuming _update implementation

    async def _get_pool(self) -> asyncpg.Pool:
        """Establishes and returns the database connection pool."""
        if self._pool is None:
            try:
                # min_size and max_size can be configured based on expected load
                self._pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=10)
                self.logger.info(f"Database connection pool established for {self.db_url}.")
            except Exception as e:
                self.logger.error(f"Failed to create database connection pool: {e}", exc_info=True)
                raise # Re-raise to indicate critical failure
        return self._pool

    async def disconnect_pool(self):
        """Closes the database connection pool if it's open."""
        if self._pool:
            try:
                await self._pool.close()
                self.logger.info("Database connection pool closed.")
            except Exception as e:
                self.logger.error(f"Error closing database connection pool: {e}", exc_info=True)
            finally:
                self._pool = None


    async def _query(self, sql_query: str, params: Optional[List[Any]] = None, context: RequestContext) -> str:
        """Execute a SELECT SQL query and return results as JSON string."""
        context.logger.info(f"Executing SQL query: {sql_query[:100]}... with params: {params}")
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # For SELECT queries that return rows
                if sql_query.strip().upper().startswith("SELECT"):
                    rows = await conn.fetch(sql_query, *params if params else [])
                    # Convert asyncpg.Record objects to dictionaries for JSON serialization
                    return json.dumps([dict(row) for row in rows], default=str) # default=str for datetime etc.
                else:
                    # For INSERT, UPDATE, DELETE that might not return rows but status
                    status = await conn.execute(sql_query, *params if params else [])
                    return json.dumps({"status": status, "message": "Command executed successfully."})
        except asyncpg.PostgresError as e: # Catch specific asyncpg errors
            context.logger.error(f"Database query error: {e.message} (SQLSTATE: {e.sqlstate})", exc_info=True)
            return f"Database Query Error: {e.message}"
        except Exception as e:
            context.logger.error(f"Unexpected error executing query: {e}", exc_info=True)
            return f"Unexpected Query Error: {str(e)}"

    async def _insert(self, table_name: str, data: Dict[str, Any], context: RequestContext) -> str:
        """Insert data into a table. Returns the ID or status."""
        context.logger.info(f"Inserting data into table '{table_name}': {str(data)[:100]}...")
        if not data:
            return "Error: No data provided for insert."
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(data))])
        values = list(data.values())
        
        # Example: Constructing SQL for INSERT. Be cautious about SQL injection if table_name is dynamic.
        # It's safer if table_name is from a predefined list or validated.
        sql_insert = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING id;" # Example, RETURNING id
        
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Using fetchval for RETURNING id, or execute for simple status
                inserted_id = await conn.fetchval(sql_insert, *values)
                if inserted_id is not None:
                    return json.dumps({"status": "success", "id": inserted_id, "message": f"Data inserted into {table_name}."})
                else: # If RETURNING id is not used or applicable
                    return json.dumps({"status": "success", "message": f"Data insertion into {table_name} attempted."})

        except asyncpg.PostgresError as e:
            context.logger.error(f"Database insert error into '{table_name}': {e.message}", exc_info=True)
            return f"Database Insert Error: {e.message}"
        except Exception as e:
            context.logger.error(f"Unexpected error during insert into '{table_name}': {e}", exc_info=True)
            return f"Unexpected Insert Error: {str(e)}"

# Example conceptual usage:
# async def main_db_integration():
#     logger = get_logger("DatabaseAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model")
#     
#     # Replace with your actual database URL
#     db_agent = DatabaseAgent(
#         name="db_operator",
#         model_config=model_cfg,
#         db_url="postgresql://user:password@host:port/database",
#         logger=logger
#     )
#     req_context = RequestContext(request_id="test_db_op", logger=logger)
#
#     # Ensure pool is connected (or _get_pool will handle it on first query)
#     # await db_agent._get_pool() # Optional: pre-connect
#
#     # Example: Query data (replace with your actual query)
#     # query_result_str = await db_agent._query("SELECT * FROM users WHERE id = $1;", [1], req_context)
#     # print(f"Query result: {query_result_str}")
#
#     # Example: Insert data
#     # new_user_data = {"username": "testuser", "email": "test@example.com"}
#     # insert_result_str = await db_agent._insert("users", new_user_data, req_context)
#     # print(f"Insert result: {insert_result_str}")
#
#     # To test the full agent flow (LLM deciding to use the tool):
#     # user_db_query = Message(role="user", content="Find user with ID 1 from the database.")
#     # async for response in db_agent.process_request(user_db_query, req_context):
#     #     print(f"DB Agent LLM response: {response.content}")
#
#     await db_agent.disconnect_pool() # Important to clean up pool

```

## Testing Custom Agents

```python
import pytest
from src.agents.agents import BaseAgent # For the example CustomBaseStyleAgent
from src.models.models import ModelConfig
from src.models.types import RequestContext, Message # Added Message
from src.utils.logging_utils import get_logger # Added

# Define or import the custom agent to be tested.
# For this example, let's use the CustomBaseStyleAgent defined earlier in this document.
class CustomBaseStyleAgentForTest(BaseAgent):
    """Simplified CustomBaseStyleAgent for testing purposes."""
    def __init__(self, name: str, model_config: ModelConfig, specialization: str, **kwargs):
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.specialization = specialization

    async def _run(self, request: Message, context: RequestContext) -> Message:
        context.logger.info(f"TESTING: {self.name} processing '{request.content}' with spec '{self.specialization}'")
        result_content = f"Test processed '{request.content}' with {self.specialization} by {self.name}"
        return Message(role="assistant", content=result_content, name=self.name)

@pytest.fixture
def logger_fixture():
    return get_logger("TestCustomAgentPytest")

@pytest.fixture
def model_config_fixture():
    # Using a dummy provider for tests not hitting actual LLMs
    return ModelConfig(type="api", provider="dummy", name="test_model")

@pytest.fixture
def custom_agent_fixture(logger_fixture, model_config_fixture):
    return CustomBaseStyleAgentForTest(
        name="test_pytest_agent",
        model_config=model_config_fixture,
        specialization="pytest_testing"
    )

class TestCustomAgent:
    @pytest.mark.asyncio
    async def test_custom_agent_creation(self, custom_agent_fixture: CustomBaseStyleAgentForTest): # Added self, used fixture
        """Test custom agent can be created and attributes are set."""
        agent = custom_agent_fixture
        assert agent.name == "test_pytest_agent"
        assert agent.specialization == "pytest_testing"
        assert agent.model_config.provider == "dummy"

    @pytest.mark.asyncio
    async def test_custom_agent_processing(self, custom_agent_fixture: CustomBaseStyleAgentForTest, logger_fixture): # Added self, used fixtures
        """Test custom processing logic of the agent."""
        agent = custom_agent_fixture
        # Create RequestContext with the logger from fixture
        context = RequestContext(request_id="pytest_test_proc_001", logger=logger_fixture)
        
        # Create a user request message
        user_request_message = Message(role="user", content="Perform test task.", name="pytest_user")
        
        # Call the agent's public `run` method (which internally calls _run for BaseAgent)
        result_message = await agent.run(user_request_message, context)
        
        assert result_message.role == "assistant"
        assert "pytest_testing" in result_message.content # From specialization
        assert "Perform test task" in result_message.content # From request
        assert agent.name in result_message.content # From agent name

# To run these tests, you would typically use `pytest` from your terminal
# Ensure pytest and pytest-asyncio are installed: pip install pytest pytest-asyncio
```

## Best Practices

1.  **Follow Framework Conventions**:
    *   Inherit from `BaseAgent` for simpler request-response agents or `Agent` for more complex, LLM-driven agents with tool use and streaming.
    *   Implement `_run(request, context)` for `BaseAgent` children, returning a single `Message`.
    *   Implement `_process_request(request, context)` for `Agent` children, yielding `Message` objects (e.g., for thoughts, tool calls, final answer).
    *   Utilize `RequestContext` for logging (`context.logger`) and passing request-specific data.
    *   Use `ModelConfig` for specifying LLM configurations.
2.  **Handle Errors Gracefully**:
    *   Inside tool implementations or agent logic, catch exceptions and return informative error messages, possibly as part of a `Message` object or a structured error string if a tool.
    *   Use `context.logger.error()` for logging exceptions with tracebacks.
3.  **Document Your Agent**:
    *   Write clear docstrings for your custom agent class and its methods, explaining its purpose, parameters, and behavior.
    *   Provide usage examples if the agent has a complex setup or unique interaction patterns.
4.  **Test Thoroughly**:
    *   Write unit tests for your agent's specific logic, including any custom tools or state management.
    *   Use `pytest` and `pytest-asyncio` for testing asynchronous code.
    *   Mock external dependencies (LLMs, APIs, databases) during unit tests to ensure they are fast and reliable.
5.  **Manage State Carefully**:
    *   If your agent needs to maintain state across requests or sessions, consider how it will be stored (e.g., in-memory, file, database).
    *   Think about persistence, concurrency, and potential race conditions if multiple instances or requests might access the same state.
6.  **Use Type Hints**:
    *   Add type hints to your agent's methods and attributes. This improves code readability, helps catch errors early, and enables better tooling.
7.  **Configuration and Initialization**:
    *   Pass necessary configurations (like API keys, model settings, tool configurations) to your agent's `__init__` method. Avoid hardcoding sensitive information.
    *   Ensure `super().__init__(...)` is called correctly with all required parameters for the base class (`name`, `model_config`, etc.).
8.  **Tool Design**:
    *   If adding custom tools, ensure they are well-defined, take necessary context (often including `RequestContext`), and return clear, parseable results (typically strings, often JSON formatted for complex data).
    *   Tool methods should be `async` if they perform I/O operations.

## Common Patterns

### Specialized Instructions

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig
from src.utils.logging_utils import get_logger # For example usage

def create_specialized_agent(domain: str, agent_name: str, model_config: ModelConfig) -> Agent:
    """Factory function for creating domain-specific agents with tailored instructions."""
    logger = get_logger(f"SpecializedAgentFactory_{domain}")
    
    instructions_map = {
        "legal": "You are a legal expert. Always cite relevant laws and precedents. Provide disclaimers that your advice is not a substitute for a qualified lawyer.",
        "medical": "You are a medical information assistant. Always include disclaimers about consulting human medical professionals. Do not provide diagnoses or treatment plans.",
        "financial": "You are a financial analyst. Provide risk warnings and state that your analysis is not financial advice. Mention regulatory compliance where applicable."
    }
    
    base_instructions = "You are a specialized assistant."
    domain_specific_instructions = instructions_map.get(domain.lower(), f"You are specialized in {domain}.")
    
    full_instructions = f"{base_instructions}\n{domain_specific_instructions}"
    
    logger.info(f"Creating agent '{agent_name}' for domain '{domain}' with instructions: {full_instructions[:100]}...")
    
    return Agent(
        name=agent_name,
        instructions=full_instructions,
        model_config=model_config
        # Add domain-specific tools here if needed
        # tools={"domain_tool": some_tool_func}
    )

# Example conceptual usage:
# model_cfg = ModelConfig(type="api", provider="openai", name="gpt-4.1-mini) # Or your preferred model
# legal_eagle = create_specialized_agent(domain="legal", agent_name="LegalEagleBot", model_config=model_cfg)
# financial_guru = create_specialized_agent(domain="financial", agent_name="FinanceGuru", model_config=model_cfg)
```

### Capability Composition (Tool Management)

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig
from src.models.types import RequestContext # For tool signature
from src.utils.logging_utils import get_logger # For example usage
from typing import Callable, Dict, Any

class ComposableAgent(Agent):
    """Agent that allows dynamic addition of capabilities (tools)."""

    def __init__(self, name: str, model_config: ModelConfig, **kwargs):
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.logger = kwargs.get('logger', get_logger(f"ComposableAgent_{name}"))

    def add_new_capability(self, capability_name: str, implementation: Callable[..., Any], instruction_hint: Optional[str] = None):
        """
        Add a new capability (tool) dynamically to the agent.
        The agent's LLM should be informed about this new capability via its instructions.
        """
        if capability_name in self.tools:
            self.logger.warning(f"Capability '{capability_name}' already exists. Overwriting.")
        
        self.add_tool(capability_name, implementation) # Use Agent's built-in method
        self.logger.info(f"Added capability: '{capability_name}'")

        # Update instructions to inform the LLM about the new tool.
        # How effective this is depends on the LLM and how it processes instructions.
        # A more robust approach might involve re-initializing the LLM prompt system if possible.
        # For simplicity, this example appends to instructions; adjust as needed.
        if instruction_hint is None:
            instruction_hint = f"You can now use the '{capability_name}' capability to perform its designated task."
        
        if self.instructions:
            self.instructions += f"\n\n[New Capability Available] {instruction_hint}"
        else:
            self.instructions = f"[New Capability Available] {instruction_hint}"
        
        self.logger.info(f"Updated instructions with hint for '{capability_name}'.")


# Example conceptual usage:
# async def custom_weather_tool(location: str, context: RequestContext) -> str:
#     context.logger.info(f"Custom weather tool called for {location}")
#     # Actual implementation to fetch weather
#     return f"The weather in {location} is sunny and 25°C (simulated)."

# async def main_composable():
#     logger = get_logger("ComposableAgentTest")
#     model_cfg = ModelConfig(provider="dummy", model_name="dummy_model")
#
#     agent = ComposableAgent(name="versatile_agent", model_config=model_cfg, logger=logger)
#     
#     # Initial instructions
#     agent.instructions = "You are a helpful assistant."
#     print(f"Initial tools: {list(agent.tools.keys())}")
#     print(f"Initial instructions: {agent.instructions}")
#
#     # Add a new capability
#     weather_instruction = "Use 'get_weather' to find out the current weather for a city."
#     agent.add_new_capability(
#         capability_name="get_weather",
#         implementation=custom_weather_tool,
#         instruction_hint=weather_instruction
#     )
#
#     print(f"Updated tools: {list(agent.tools.keys())}")
#     print(f"Updated instructions: {agent.instructions}")
#
#     # Now the agent (if its LLM understands the updated instructions) could use this tool.
#     # req_context = RequestContext(request_id="composable_test", logger=logger)
#     # user_query = Message(role="user", content="What's the weather in London?")
#     # async for response in agent.process_request(user_query, req_context):
#     #     print(f"Composable Agent response: {response.content}")

```

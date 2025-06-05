# Agents

Agents are the core building blocks of the Multi-Agent Reasoning Systems (MARSYS) framework. They are autonomous entities that can perceive their environment, make decisions, and take actions based on their configuration and interactions.

## What is an Agent in MARSYS?

An agent in this framework is typically characterized by:
- An **identity**: A unique name within the `AgentRegistry`.
- An **AI Model**: Powered by an underlying language model (LLM), vision-language model (VLM), or API-based model, configured via `ModelConfig`.
- **Capabilities**: Ability to use tools (functions) and invoke other registered agents.
- **Memory**: Maintains conversational context and history through a `MemoryManager` instance.
- **Instructions/Description**: A textual description defining its role, persona, and objectives.

## Agent Class Hierarchy (Simplified)

The primary classes involved are `BaseAgent`, `Agent`, `BrowserAgent`, and `LearnableAgent`.

```mermaid
classDiagram
    BaseAgent <|-- Agent
    Agent <|-- LearnableAgent
    Agent <|-- BrowserAgent # BrowserAgent inherits from Agent

    class ModelConfig {
        +type: str
        +name: str
        +provider: Optional[str]
        +api_key: Optional[str]
        +max_tokens: int
        +temperature: float
        +model_class: Optional[str] # For local models ('llm', 'vlm')
    }

    class BaseAgent {
        +name: str
        +model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel] # Renamed from model to model_instance
        +description: str
        +tools: Dict[str, Callable]
        +tools_schema: List[Dict]
        +allowed_peers: Set[str]
        +logger: logging.Logger
        +__init__(model_config: ModelConfig, description: str, tools, agent_name, allowed_peers)
        +invoke_agent(target_agent_name: str, request: Any, request_context: RequestContext) Message
        # ... other methods ...
    }

    class Agent {
        +memory: MemoryManager
        +__init__(model_config: ModelConfig, description: str, tools, agent_name, allowed_peers, memory_type, max_tokens, system_message_processor, user_message_processor, assistant_message_processor)
        +auto_run(initial_prompt: Any, request_context: Optional[RequestContext], max_steps: int) Message # Return type updated
        # +_run(prompt: Any, request_context: RequestContext, run_mode: str) Message
        # ... other methods ...
    }

    class BrowserAgent {
        +browser_tool: Optional[BrowserTool]
        +__init__(model_config: ModelConfig, generation_description: str, critic_description: str, memory_type, max_tokens, agent_name, allowed_peers)
        +create_safe(model_config: ModelConfig, ...) BrowserAgent
        # ... browser specific methods ...
    }

    class LearnableAgent {
        +learning_memory: LearningMemory
        +feedback_mechanism: FeedbackMechanism
        +__init__(..., learning_config: LearningConfig)
        # +learn_from_feedback(feedback: Feedback) None
        # +adapt_behavior() None
        # ... learning specific methods ...
    }

    class MemoryManager {
        +history: List[Message]
        +update_memory(message: Message)
        +retrieve_recent(limit: int) List[Message]
        +to_llm_format() List[Dict]
    }

    class Message {
        +role: str
        +content: Any
        +message_id: str
        +tool_calls: Optional[List[Dict]]
        +tool_call_id: Optional[str]
        +name: Optional[str]
    }

    class RequestContext {
        +request_id: str # Renamed from task_id
        +interaction_id: str
        +interaction_count: int
        +depth: int
        +caller_agent_name: Optional[str]
    }

    ModelConfig ..> BaseAgent : provides configuration for
    BaseAgent ..> AgentRegistry : registers with
    Agent ..> MemoryManager : uses
    Agent ..> Message : processes and stores
    Agent ..> RequestContext : uses for operations
    BaseAgent ..> "BaseLLM, BaseVLM, BaseAPIModel" : uses model_instance of
    BrowserAgent ..> "BrowserTool" : uses
    LearnableAgent ..> "LearningMemory" : uses
    LearnableAgent ..> "FeedbackMechanism" : uses
```
*Note: The diagram is a conceptual representation. Refer to the source code for exact method signatures and relationships.* 

## Creating an Agent

Agents are instantiated by providing a model configuration, a description, and optionally tools and other settings.

### Basic Agent Example

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig

# Define the model configuration
openai_model_config = ModelConfig(
    type="api",
    provider="openai",
    name="gpt-4-turbo", # Ensure this is a valid model name for your provider
    temperature=0.7,
    max_tokens=1500
    # API key will be read from OPENAI_API_KEY environment variable by default
)

# Create the agent instance
assistant_agent = Agent(
    model=openai_model_config, # Pass the ModelConfig object
    description="You are a helpful and friendly AI assistant. Your goal is to provide accurate and concise answers.",
    agent_name="HelpfulAssistant" # Optional: specify a unique name
)

# To run the agent (example)
# async def main():
#     response = await assistant_agent.auto_run(initial_request="Hello, who are you?")
#     print(response)
# import asyncio
# asyncio.run(main())
```

### Agent with Tools

Tools are Python functions that the agent can decide to call to perform actions.

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig
from datetime import datetime

def get_current_time(timezone: Optional[str] = None) -> str:
    """Returns the current date and time, optionally for a specific timezone."""
    # Basic implementation, for real timezone handling, use pytz or similar
    return f"The current date and time is: {datetime.now().isoformat()}" 

research_model_config = ModelConfig(
    type="api", provider="openai", name="gpt-4-turbo"
)

researcher_agent = Agent(
    model=research_model_config,
    description="You are a research assistant. You can use tools to find information.",
    tools={"get_current_time_tool": get_current_time},
    agent_name="Researcher"
)

# Example of running the researcher agent
# async def main():
#     response = await researcher_agent.auto_run(initial_request="What time is it now?")
#     print(response)
# import asyncio
# asyncio.run(main())
```
Tool schemas are generated automatically from the function signature and docstring.

## Agent Input/Output Schema Validation

The MARS framework supports optional schema validation for agent-to-agent communication to ensure data consistency and catch errors early. You can define schemas for both incoming requests (`input_schema`) and outgoing responses (`output_schema`).

### Schema Formats

Schemas can be specified in three user-friendly formats:

#### 1. List of Strings (Simple)
Each string becomes a required field with string type:

```python
from src.agents.agents import Agent
from src.models.models import ModelConfig

# ResearcherAgent expects a 'sub_question' field
researcher_agent = Agent(
    model_config=ModelConfig(type="api", provider="openai", name="gpt-4o"),
    description="You are a research assistant that answers specific sub-questions.",
    input_schema=["sub_question"],  # Requires: {"sub_question": "string value"}
    output_schema=["answer"],       # Returns: {"answer": "string response"}
    agent_name="ResearcherAgent"
)
```

#### 2. Dict of Key:Type (Typed)
Specify exact types for each field:

```python
# SynthesizerAgent with typed input/output
synthesizer_agent = Agent(
    model_config=ModelConfig(type="api", provider="openai", name="gpt-4o"),
    description="You synthesize research data into reports.",
    input_schema={
        "user_query": str,
        "validated_data": list,
        "priority": int
    },
    output_schema={
        "report": str,
        "confidence": float
    },
    agent_name="SynthesizerAgent"
)
```

#### 3. Full JSON Schema (Advanced)
Use complete JSON Schema for complex validation:

```python
# Advanced schema with constraints
advanced_agent = Agent(
    model_config=ModelConfig(type="api", provider="openai", name="gpt-4o"),
    description="You process complex structured data.",
    input_schema={
        "type": "object",
        "properties": {
            "document": {
                "type": "string",
                "minLength": 10
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "timestamp": {"type": "number"}
                },
                "required": ["source"]
            }
        },
        "required": ["document"]
    },
    output_schema={
        "type": "object", 
        "properties": {
            "summary": {"type": "string", "minLength": 5},
            "key_points": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            }
        },
        "required": ["summary", "key_points"]
    },
    agent_name="AdvancedAgent"
)
```

### How Schema Validation Works

1. **Input Validation**: When one agent invokes another via `invoke_agent()`, the request is validated against the target agent's `input_schema` before execution.

2. **Output Validation**: When an agent returns a final response, it's validated against the agent's own `output_schema`.

3. **Error Handling**: Validation failures return error messages instead of proceeding with invalid data.

### Schema Validation in Practice

```python
# Define agents with schemas
researcher = Agent(
    model_config=model_config,
    description="Research assistant that answers sub-questions.",
    input_schema=["sub_question"],
    output_schema=["answer"],
    agent_name="Researcher",
    allowed_peers=[]
)

orchestrator = Agent(
    model_config=model_config,
    description="Main agent that coordinates research.",
    input_schema=["main_query"],
    output_schema=["final_report"],
    agent_name="Orchestrator",
    allowed_peers=["Researcher"]
)

# This will work - matches input_schema
result = await orchestrator.invoke_agent(
    "Researcher",
    {"sub_question": "What is machine learning?"},
    request_context
)

# This will fail validation - missing required field
result = await orchestrator.invoke_agent(
    "Researcher", 
    {"wrong_field": "value"},  # Missing 'sub_question'
    request_context
)
# Returns: Message(role="error", content="Request validation failed...")
```

### Best Practices for Schemas

1. **Start Simple**: Use list format for basic string fields, then upgrade to dict format when you need types.

2. **Match Communication Patterns**: Design schemas to match how agents actually communicate in your system.

3. **Graceful Degradation**: Schema validation logs errors but doesn't crash agents - invalid data is caught early.

4. **Document Expected Formats**: Include schema information in your agent descriptions so LLMs understand the expected format.

**Note**: Schema validation only applies to agent-to-agent communication and final responses. Internal framework fields like `"status"` and monitoring data are handled separately and should not appear in user-defined schemas.


## Creating Custom Agents

For more specialized behavior, you can create custom agent classes by inheriting from `BaseAgent` or `Agent`.

```python
from src.agents.agents import Agent # Or BaseAgent for more fundamental customization
from src.models.models import ModelConfig
from src.agents.memory import Message # Correct import for Message
from src.agents.utils import RequestContext # Correct import for RequestContext
from typing import Any

class MyCustomAgent(Agent): # Inherit from Agent for auto_run, memory etc.
    def __init__(self, model: ModelConfig, description: str, agent_name: Optional[str] = None, **kwargs):
        super().__init__(model=model, description=description, agent_name=agent_name, **kwargs)
        # Custom initialization for your agent
        self.my_custom_state = "initialized"

    async def _run(
        self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs: Any
    ) -> Message:
        # Override _run for custom step logic within auto_run
        # This is a simplified example. A real _run would involve:
        # 1. Updating memory with the input prompt.
        # 2. Preparing messages for the LLM using self.memory.to_llm_format().
        # 3. Constructing the system prompt using self._construct_full_system_prompt().
        # 4. Calling self.model.run() with messages, tools_schema, json_mode, etc.
        # 5. Processing the LLM response into a Message object (e.g. using Message.from_response_dict).
        # 6. Updating memory with the response Message.

        await self._log_progress(request_context, LogLevel.INFO, f"MyCustomAgent ({self.name}) received prompt: {str(prompt)[:100]}")

        # Example: Add custom behavior before calling the model or processing its response
        if "special_trigger" in str(prompt):
            # Perform a custom action
            pass

        # Call the parent's _run or reimplement its logic carefully
        # For this example, we'll simulate a simple response
        # In a real scenario, you would interact with self.model here
        # and use self.memory, self.tools_schema etc.

        system_prompt_str = self._construct_full_system_prompt(
            base_description=self.description,
            json_mode_for_output=(run_mode == "auto_step") # auto_step expects JSON
        )

        # Prepare messages for the model
        # This would typically involve self.memory.update_memory(input_message_obj)
        # and then self.memory.to_llm_format()
        # For simplicity, creating a dummy input message list:
        current_messages_for_llm = [
            {"role": "system", "content": system_prompt_str},
            {"role": "user", "content": str(prompt)} # Simplified
        ]

        # Simulate model call
        # llm_response_data = await self.model.run(
        #     messages=current_messages_for_llm,
        #     json_mode=(run_mode == "auto_step"),
        #     tools=self.tools_schema if self.tools_schema else None,
        #     max_tokens=self.max_tokens
        # )

        # For this example, let's assume llm_response_data is a dict like:
        # {"role": "assistant", "content": "...", "tool_calls": [...]}
        # This would come from self.model.run()
        # Ensure the response from self.model.run() is compatible with Message.from_response_dict
        # or process it accordingly.

        # Example: Directly creating a Message if not calling LLM or for specific logic
        response_content = f"Custom agent {self.name} processed: {str(prompt)[:50]}"
        if run_mode == "auto_step": # auto_step expects a JSON response for next_action
            response_content = json.dumps({
                "thought": "This is a custom thought from MyCustomAgent.",
                "next_action": "final_response",
                "action_input": {"response": response_content}
            })

        output_message = Message(
            role="assistant",
            content=response_content,
            name=self.name
        )
        
        # Update memory with the output message
        # await self.memory.update_memory(output_message)

        return output_message

# Example usage:
# custom_agent_config = ModelConfig(type="api", provider="openai", name="gpt-4.1-mini")
# my_agent = MyCustomAgent(model=custom_agent_config, description="A custom agent example.", agent_name="CustomBot")
# async def main():
#    final_response = await my_agent.auto_run(initial_request="Hello custom agent!")
#    print(final_response)
# import asyncio
# asyncio.run(main())
```

## Agent Capabilities

### 1. **Autonomous Multi-Step Execution (`auto_run`)**

Agents can perform tasks over multiple steps, deciding to call tools, invoke other agents, or provide a final response.

```python
# Assuming 'assistant_agent' is an initialized Agent instance
# async def main():
#     response_string = await assistant_agent.auto_run(
#         initial_request="Research the latest trends in AI for 2025 and then ask the SummarizerAgent to summarize it.",
#         max_steps=10 # Max iterations of thought-action-observation loop
#     )
#     print(f"Final response from auto_run: {response_string}")
# import asyncio
# asyncio.run(main())
```

### 2. **Tool Usage**

If an agent has tools, its underlying LLM can decide to use them. The `Agent` class handles the orchestration.

```python
# Assuming 'researcher_agent' has the 'get_current_time_tool'
# async def main():
#     response = await researcher_agent.auto_run(
#         initial_request="What is the current time? Please use your tools.",
#         max_steps=3
#     )
#     print(response)
# import asyncio
# asyncio.run(main())
```

### 3. **Inter-Agent Communication (`invoke_agent`)**

Agents can invoke other registered agents if they are in their `allowed_peers` list. This is typically handled within the `auto_run` loop if the LLM decides to take the `invoke_agent` action.

```python
# orchestrator_config = ModelConfig(type="api", provider="openai", name="gpt-4-turbo")
# researcher_config = ModelConfig(type="api", provider="openai", name="gpt-4-turbo")

# researcher = Agent(model=researcher_config, description="I find information.", agent_name="ResearcherAgent")
# orchestrator = Agent(
#     model=orchestrator_config, 
#     description="I coordinate tasks and can talk to other agents.", 
#     agent_name="OrchestratorAgent",
#     allowed_peers=["ResearcherAgent"] # Allows OrchestratorAgent to call ResearcherAgent
# )

# async def main():
#     # The orchestrator might decide to call ResearcherAgent based on the task
#     response = await orchestrator.auto_run(
#         initial_request="Ask the ResearcherAgent to find out about quantum computing progress.",
#         max_steps=5
#     )
#     print(response)
# import asyncio
# asyncio.run(main())
```

## Agent Description (System Prompt Foundation)

The `description` provided during agent initialization is a crucial part of its system prompt, defining its role, personality, and high-level goals.

```python
code_reviewer_description = """
You are an expert Python code reviewer. Your primary goal is to help developers write better code.
Key Responsibilities:
1. Review Python code snippets for adherence to PEP 8 style guidelines.
2. Identify potential bugs, logical errors, or inefficiencies.
3. Suggest improvements for clarity, performance, and maintainability.
4. Always provide constructive, polite, and actionable feedback.

Output Format:
- For code suggestions, use Markdown code blocks with the `python` language identifier.
- Clearly explain the reasoning behind each suggestion.
"""

# code_reviewer_config = ModelConfig(type="api", provider="openai", name="gpt-4-turbo")
# code_reviewer_agent = Agent(
#     model=code_reviewer_config,
#     description=code_reviewer_description,
#     agent_name="CodeReviewer"
# )
```

## Agent Registration and Discovery

Agents are automatically registered with the `AgentRegistry` upon instantiation using their `agent_name` (or a generated one if `agent_name` is not provided). This allows them to be discovered and invoked by other agents.

```python
from src.agents.registry import AgentRegistry

# Agent is registered automatically during __init__
# helper_config = ModelConfig(type="api", provider="openai", name="gpt-4.1-mini")
# helper_agent = Agent(model=helper_config, description="A simple helper.", agent_name="MyHelperAgent")

# Retrieve an agent from the registry
# retrieved_agent = AgentRegistry.get("MyHelperAgent")
# if retrieved_agent:
#     print(f"Agent '{retrieved_agent.name}' is registered and retrieved.")
# else:
#     print("Agent not found in registry.")

# Agents are also unregistered automatically when they are deleted (via __del__).
```

## Agent Memory (`MemoryManager`)

Each `Agent` instance has its own `MemoryManager` (by default, `ConversationMemory`) which stores the history of interactions (user inputs, agent responses, tool calls, tool responses).

- The memory is updated automatically within the `_run` method of the `Agent` class.
- Messages are stored in chronological order.
- The `memory.to_llm_format()` method prepares the history in a format suitable for the LLM.

```python
# Memory is managed internally by the Agent during auto_run
# async def main():
#     chat_config = ModelConfig(type="api", provider="openai", name="gpt-4.1-mini")
#     chat_agent = Agent(model=chat_config, description="A conversational agent.", agent_name="Chatty")

#     response1 = await chat_agent.auto_run(initial_request="Hi there! My favorite color is blue.")
#     print(f"Response 1: {response1}")

#     # The agent's memory now contains the first interaction.
#     response2 = await chat_agent.auto_run(initial_request="What was my favorite color?")
#     print(f"Response 2: {response2}") # Agent should remember "blue"
# import asyncio
# asyncio.run(main())
```
For more details, see [Memory Concepts](./memory.md).

## Best Practices for Agent Design

1.  **Clear and Specific Descriptions**: The agent's `description` is key. Make it detailed about the agent's role, capabilities, limitations, and expected output format.
2.  **Appropriate Model Selection**: Choose a `ModelConfig` (provider, model name, parameters) that suits the complexity and requirements of the agent's tasks.
3.  **Well-Defined Tools**: If using tools, ensure they are robust, have clear docstrings (for schema generation), and handle potential errors.
4.  **Consider `allowed_peers`**: Explicitly define which other agents an agent can invoke to control interaction flows and prevent unintended calls.
5.  **Iterative Testing**: Test agents with various inputs and scenarios to refine their descriptions, tools, and overall behavior.
6.  **Error Handling in Custom Agents**: If creating custom agents by overriding `_run`, ensure robust error handling and that `Message` objects (even for errors) are returned as per framework expectations.

## Advanced Usage

### Customizing Agent Behavior by Overriding `_run`

As shown in the "Creating Custom Agents" section, inheriting from `Agent` and overriding the `_run` method allows for fine-grained control over each step of the agent's execution cycle within `auto_run`. This is where you can inject custom logic for prompt engineering, model interaction, response parsing, and memory updates.

### Agent Composition and Orchestration

Design multiple specialized agents and an orchestrator agent to manage them. The orchestrator's `description` would guide it to delegate sub-tasks to the appropriate specialized agents via `invoke_agent`.

```python
# Conceptual example:
# research_writer_config = ModelConfig(type="api", provider="openai", name="gpt-4-turbo")
# editor_config = ModelConfig(type="api", provider="openai", name="gpt-4.1-mini")
# main_orchestrator_config = ModelConfig(type="api", provider="openai", name="gpt-4-turbo")

# research_writer = Agent(
#     model=research_writer_config,
#     description="I research topics and write initial drafts.",
#     agent_name="ResearchWriterAgent"
# )
# editor = Agent(
#     model=editor_config,
#     description="I review and edit drafts for clarity and grammar.",
#     agent_name="EditorAgent"
# )

# orchestrator = Agent(
#     model=main_orchestrator_config,
#     description=(
#         "You are a project manager. Your goal is to produce a high-quality article. "
#         "First, ask ResearchWriterAgent to draft an article on a given topic. "
#         "Then, ask EditorAgent to review and refine the draft from ResearchWriterAgent. "
#         "Finally, present the edited article as your final response."
#     ),
#     agent_name="ProjectManagerAgent",
#     allowed_peers=["ResearchWriterAgent", "EditorAgent"]
# )

# async def main():
#     final_article = await orchestrator.auto_run(
#         initial_request="Produce an article about the future of renewable energy."
#     )
#     print("--- Final Article ---")
#     print(final_article)
# import asyncio
# asyncio.run(main())
```

## Next Steps

- Learn about [Messages](./messages.md) - The structure of communication data.
- Explore [Memory](./memory.md) - How agents maintain context and history.
- Understand [Tools](./tools.md) - How to extend agent capabilities with functions.
- Review [Core Concepts](./core-concepts.md) for a broader overview.

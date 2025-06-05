---
applyTo: '**'
---

# MARSYS Mental Map Instructions# Multi-Agent Reasoning Systems (MARSYS) – Complete Mental Map

This document provides a comprehensive understanding of the MARSYS framework's architecture, design patterns, and usage guidelines derived from deep analysis of the source code.


---

## 1. Framework Architecture Overview

MARSYS (Multi-Agent Reasoning Systems) is a sophisticated framework for building collaborative AI agent systems. The architecture follows a layered approach with clear separation of concerns:

```
src/
├── agents/          # Agent implementations and core logic
│   ├── agents.py         # BaseAgent, Agent classes
│   ├── browser_agent.py  # Browser automation agent
│   ├── learnable_agents.py # Learning-capable agents
│   ├── memory.py        # Memory management system
│   ├── registry.py      # Agent discovery & registration
│   └── utils.py         # Logging, context management
├── models/          # AI model abstractions
│   ├── models.py        # Model interfaces & configs
│   ├── processors.py    # Vision/image processing
│   └── utils.py         # Model utilities
├── environment/     # External integrations
│   ├── tools.py         # Tool system
│   ├── web_browser.py   # Browser automation
│   ├── operator.py      # Operations management
│   └── utils.py         # Environment utilities
├── learning/        # Learning algorithms
│   └── rl.py           # Reinforcement learning
└── utils/           # General utilities
    └── monitoring.py    # Progress monitoring


┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (User Applications, Multi-Agent Workflows)                  │
├─────────────────────────────────────────────────────────────┤
│                       Agent Layer                            │
│  (BaseAgent, Agent, BrowserAgent, LearnableAgent)          │
├─────────────────────────────────────────────────────────────┤
│                    Abstraction Layer                         │
│  (Model Abstractions, Memory Abstractions)                   │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  (Registry, Tools, Monitoring, Utils)                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles:
1. **Message-Centric Communication**: All agent interactions use standardized Message objects
2. **Async-First Design**: All I/O operations are asynchronous for maximum concurrency
3. **Weak Reference Registry**: Prevents memory leaks while enabling agent discovery
4. **OpenAI-Compatible Format**: Messages and tools follow OpenAI's function calling spec
5. **Pluggable Components**: Models, memory types, and tools are easily extensible

---

## 2. Core Components Deep Dive

### 2.1 Agent System (`src/agents/agents.py`)

#### **BaseAgent** (Abstract Base Class)
- **Location**: `src/agents/agents.py`
- **Purpose**: Foundation for all agent types, providing core infrastructure
- **Constructor Parameters**:
  - `model: Union[BaseVLM, BaseLLM, BaseAPIModel]` - The AI model instance
  - `description: str` - Agent's role and behavioral guidelines
  - `tools: Optional[Dict[str, Callable]]` - Tool name to function mapping
  - `max_tokens: Optional[int]` - Default token limit for responses
  - `agent_name: Optional[str]` - Unique identifier (auto-generated if not provided)
  - `allowed_peers: Optional[List[str]]` - Whitelist of agents this one can invoke

- **Key Attributes**:
  - `tools_schema: List[Dict[str, Any]]` - Auto-generated OpenAI tool schemas
  - `communication_log: Dict[str, List[Dict]]` - Per-task communication history
  - `logger: logging.Logger` - Agent-specific logger instance
  - `name: str` - Registered name in AgentRegistry

- **Critical Methods**:

  1. **`__init__`**: 
     - Generates tool schemas using `generate_openai_tool_schema()`
     - Registers with AgentRegistry
     - Initializes communication log
     - Sets up agent-specific logger

  2. **`_construct_full_system_prompt`**:
     - Strips duplicate JSON instructions using regex
     - Combines base description with tool instructions
     - Adds peer agent invocation guidelines
     - Includes response format specifications

  3. **`async invoke_agent`**:
     - **Input**: `target_agent_name`, `request`, `request_context`
     - **Process**:
       - Verifies target exists in registry
       - Checks permission against `allowed_peers`
       - Validates depth/interaction limits
       - Creates new RequestContext with incremented counters
       - Calls target's `auto_run` method
       - Wraps response in Message object
     - **Return**: `Message` with agent response or error

  4. **`async auto_run`**:
     - **Input**: `initial_request`, `request_context`, `max_steps`, `max_re_prompts`
     - **Process**:
       - Creates RequestContext if not provided
       - Starts progress monitor
       - Iterative execution loop:
         - Calls `_run` with current state
         - Parses JSON response
         - Executes actions (tool calls, agent invocations)
         - Updates memory
       - Handles re-prompting for malformed responses
     - **Return**: `str` final answer

  5. **`async _execute_tool_calls`**:
     - Sanitizes tool names (removes "functions." prefix)
     - Executes tools (async or sync via thread pool)
     - Updates memory with tool results
     - Returns structured results for LLM

  6. **`update_from_response`**  
     Delegates an LLM response dict to the underlying MemoryManager so it can be
     transformed (via input processors) and stored as a `Message`.

#### **Agent** (Concrete Implementation)
- **Location**: `src/agents/agents.py`
- **Purpose**: Standard agent for API-based models (OpenAI, Anthropic, etc.)
- **Extends**: BaseAgent
- **Constructor Additional Parameters**:
  - `model_config: ModelConfig`
  - `tools: Optional[Dict[str, Callable]]`
  - `memory_type: Optional[str]` – `"conversation_history"` (default) or `"kg"`

- **Key Attributes**:
  - `memory: MemoryManager` - Manages conversation/knowledge storage
  - `_model_config: ModelConfig` - Stored configuration
  - `model_instance: Union[BaseLLM, BaseVLM, BaseAPIModel]` - Created from config

- **Critical Methods**:

  1. **`_create_model_from_config`**:
     - Factory method for model instantiation
     - Handles "local" vs "api" model types
     - Extracts provider-specific kwargs
     - Creates appropriate model class

  2. **`async _run`** (Implements abstract method):
     - **Process**:
       - Extracts prompt and context using `_extract_prompt_and_context`
       - Updates memory with context messages
       - Constructs system prompt based on run_mode
       - Retrieves messages in LLM format via `memory.to_llm_format()`
       - Calls model with appropriate parameters
       - Uses `memory.update_from_response()` for storing response
     - **Return**: `Message` object

  3. **`_input_message_processor`**:
     - Transforms LLM responses to Message format
     - Extracts `agent_call` from JSON content
     - Preserves thought as content

  4. **`_output_message_processor`**:
     - Transforms Messages to LLM format
     - Synthesizes JSON content when `agent_call` present
     - Removes non-OpenAI fields

#### **BrowserAgent** (`browser_agent.py`)
- **Location**: `src/agents/browser_agent.py`
- **Purpose**: Web automation agent (placeholder for Playwright integration)
- **Status**: _Implemented (beta)_ – Playwright/Selenium ready, dynamic tool loading.

#### **LearnableAgent** (`learnable_agents.py`)
- **Location**: `src/agents/learnable_agents.py`
- **Purpose**: Agents with learning/adaptation capabilities
- **Status**: _Implemented_ – supports PEFT heads and conversation memory.

### 2.2 Memory System (`src/agents/memory.py`)

#### **Message** (Dataclass)
- **Location**: `src/agents/memory.py`
- **Purpose**: Universal message format for agent communication
- **Fields**:
  - `role: str` - One of: system, user, assistant, tool, error
  - `content: Optional[str]` - Text content (can be None for tool-only messages)
  - `message_id: str` - UUID for tracking
  - `name: Optional[str]` - Agent/tool identifier
  - `tool_calls: Optional[List[Dict]]` - Tool invocation requests
  - `tool_call_id: Optional[str]` - Links tool results to requests
  - `agent_call: Optional[Dict]` - Agent invocation metadata

- **Methods**:
  - `to_llm_dict()`: Converts to OpenAI-compatible format
    - Filters None values appropriately
    - Handles content=None for assistant+tool_calls
  - `from_response_dict()`: Creates Message from LLM response
    - Accepts optional processor for transformation
    - Handles missing fields gracefully

#### **MemoryManager**
- **Location**: `src/agents/memory.py`
- **Purpose**: Factory and facade for memory implementations
- **Constructor Parameters**:
  - `memory_type: str` - "conversation_history" or "kg"
  - `description: Optional[str]` - Initial system message
  - `model: Optional[Union[BaseLLM, BaseVLM]]` - Required for KG memory
  - `input_processor: Optional[Callable]` - Transform from LLM format
  - `output_processor: Optional[Callable]` - Transform to LLM format

- **Key Methods**:
  - `update_from_response()`: Special method for LLM responses
    - Applies input processor transformation
    - Creates Message and stores in memory
  - `to_llm_format()`: Gets messages with output transformation
  - `_extract_prompt_and_context()`: Parses complex prompt dicts

#### **BaseMemory** (Abstract)
- **Location**: `src/agents/memory.py`, Lines 174-296
- **Purpose**: Interface for memory implementations
- **Abstract Methods**:
  - `update_memory()`, `replace_memory()`, `delete_memory()`
  - `retrieve_all()`, `retrieve_recent()`, `retrieve_by_id()`, `retrieve_by_role()`
  - `reset_memory()`, `to_llm_format()`

#### **ConversationMemory**
- **Location**: `src/agents/memory.py`, Lines 299-459
- **Storage**: `List[Message]` - Sequential message history
- **Features**:
  - Preserves system message on reset
  - Supports message transformation
  - Maintains chronological order

#### **KGMemory** (Knowledge Graph)
- **Location**: `src/agents/memory.py`, Lines 462-732
- **Storage**: `List[Dict]` - Fact triplets with timestamps
- **Unique Features**:
  - `extract_and_update_from_text()`: Uses LLM to extract facts
  - Stores facts as (subject, predicate, object) triplets
  - Converts facts to Messages on retrieval
  - Requires model instance for extraction

#### **MessageMemory** (Legacy)
- **Location**: `src/agents/memory.py`, Lines 11-52
- **Purpose**: Older Pydantic-based memory (being phased out)
- **Note**: Uses different structure, included for backwards compatibility

### 2.3 Registry System (`src/agents/registry.py`)

#### **AgentRegistry**
- **Location**: `src/agents/registry.py`, Lines 8-89
- **Purpose**: Central service discovery with weak references
- **Class Attributes**:
  - `_agents: WeakValueDictionary[str, BaseAgent]` - No strong references
  - `_lock: threading.Lock` - Thread-safe operations
  - `_counter: int` - For auto-generated names

- **Methods**:
  - `register(agent, name, prefix) -> str`:
    - Generates unique names if not provided
    - Prevents duplicate names
    - Returns final registered name
  - `get(name) -> Optional[BaseAgent]`:
    - Returns None if agent was garbage collected
  - `all() -> Dict[str, BaseAgent]`:
    - Snapshot of all live agents

### 2.4 Utils System (`src/agents/utils.py`)

#### **LogLevel** (IntEnum)
- **Values**: NONE(0), MINIMAL(1), SUMMARY(2), DETAILED(3), DEBUG(4)
- **Usage**: Controls verbosity of progress logging

#### **RequestContext** (Dataclass)
- **Location**: `src/agents/utils.py`, Lines 82-123
- **Purpose**: Request lifecycle tracking and limit enforcement
- **Key Fields**:
  - `task_id`: Overall task identifier
  - `interaction_id`: Current step ID
  - `depth`: Agent invocation chain depth
  - `interaction_count`: Total interactions
  - `progress_queue`: Async queue for updates
  - `max_depth/max_interactions`: Safety limits
  - `current_tokens_used`: Token tracking
  - `max_tokens_soft/hard_limit`: Token limits

#### **ProgressLogger**
- **Location**: `src/agents/utils.py`, Lines 128-184
- **Key Method**: `async log(request_context, level, message, **data)`
  - Routes to async queue if available and level appropriate
  - Falls back to standard logging
  - Prevents double logging

#### **AgentLogFilter**
- **Location**: `src/agents/utils.py`, Lines 18-43
- **Purpose**: Ensures consistent log formatting
- **Behavior**: 
  - Adds `agent_name` to all log records
  - Normalizes logger names

### 2.5 Model System (`src/models/`)

#### **Expected ModelConfig** (Pydantic BaseModel)
- **Fields** (inferred from usage):
  ```python
  type: Literal["local", "api"]
  provider: Optional[str]  # "openai", "anthropic", etc.
  name: str  # Model identifier
  api_key: Optional[str]
  base_url: Optional[str]
  max_tokens: int = 512
  temperature: float = 0.7
  model_class: Optional[Literal["llm", "vlm"]]  # For local models
  torch_dtype: Optional[str]  # For local models
  device_map: Optional[str]  # For local models
  ```

#### **Expected Model Interfaces**:
- **BaseLLM**: Language models
- **BaseVLM**: Vision-language models  
- **BaseAPIModel**: API-based models
- **Common Interface**:
  ```python
  def run(messages, max_tokens, temperature, json_mode, tools, **kwargs) -> Dict/str
  ```

### 2.6 Environment System (`src/environment/`)

#### **Expected Tools Module** (`tools.py`):
- **AVAILABLE_TOOLS**: Global tool registry
- **Tool Requirements**:
  - Full type annotations
  - Descriptive docstring
  - Parameter descriptions in docstring
  - Return type annotation

#### **Expected generate_openai_tool_schema** (`src/environment/utils.py`):
- **Purpose**: Convert Python functions to OpenAI tool schemas
- **Process**:
  - Inspects function signature
  - Parses docstring for descriptions
  - Generates JSON schema for parameters

### 2.7 Learning System (`src/learning/`)

#### **LearnableAgent** (Placeholder)
- **Expected Features**:
  - Extends Agent with trainable components
  - PeftHead integration for parameter-efficient fine-tuning
  - Experience buffer for learning
  - Performance metrics tracking

### 2.8 Monitoring System (`src/utils/monitoring.py`)

#### **Expected default_progress_monitor**:
- **Purpose**: Background task for progress monitoring
- **Interface**: `async def default_progress_monitor(queue: asyncio.Queue, logger: Logger)`
- **Behavior**: Consumes ProgressUpdate objects until None received

---

## 3. Data Flow & Communication Patterns

### 3.1 Agent Invocation Flow
```
1. User → agent.auto_run(task)
2. auto_run creates/uses RequestContext
3. Loop: _run() → parse JSON → execute action
4. Actions:
   a. Tool Call: _execute_tool_calls() → update memory
   b. Agent Call: invoke_agent() → target.auto_run()
   c. Final Response: return answer
5. Progress updates → async queue → monitor
```

### 3.2 Memory Update Flow
```
1. Input Processing:
   - User prompt → memory.update_memory()
   - Context messages → memory.update_memory()

2. Model Interaction:
   - memory.to_llm_format() → model.run()
   - Response → memory.update_from_response()

3. Transformations:
   - input_processor: LLM format → Message format
   - output_processor: Message format → LLM format
```

### 3.3 Tool Execution Flow
```
1. Model returns tool_calls in response
2. Agent updates memory with tool_calls
3. _execute_tool_calls():
   - Sanitize tool names
   - Execute each tool
   - Create tool result messages
4. Update memory with results
5. Continue execution with results in context
```

---

## 4. Memory System Architecture

### 4.1 Message Lifecycle
```
Creation → Storage → Retrieval → Transformation → Transmission
    ↓         ↓          ↓             ↓              ↓
from_response append  retrieve_*  to_llm_dict    model/agent
```

### 4.2 Memory Types Comparison

| Feature | ConversationMemory | KGMemory |
|---------|-------------------|----------|
| Storage | Sequential list | Fact triplets |
| Ordering | Chronological | Timestamp-based |
| Retrieval | By role/ID/recent | By subject/predicate |
| Use Case | Dialogues | Knowledge extraction |
| Model Required | No | Yes (for extraction) |

### 4.3 Message Transformation Pipeline
```
LLM Response → input_processor → Message → Memory
Memory → Message → output_processor → LLM Format
```

---

## 5. Model Abstraction Layer

### 5.1 Model Type Hierarchy
```
BaseModel (Abstract)
    ├── BaseLLM (Local language models)
    ├── BaseVLM (Local vision-language models)
    └── BaseAPIModel (Cloud API models)
```

### 5.2 Model Configuration Flow
```
ModelConfig → Agent._create_model_from_config() → Model Instance
     ↓                      ↓                           ↓
Validation          Type Detection              Initialization
```

---

## 6. Tool System & Environment Integration

### 6.1 Tool Schema Generation
```python
Python Function → generate_openai_tool_schema() → OpenAI Schema
        ↓                     ↓                         ↓
   Signature            Docstring                 JSON Schema
```

### 6.2 Tool Invocation Protocol
1. Model requests tool via `tool_calls`
2. Agent validates tool exists
3. Sanitizes tool name
4. Parses arguments JSON
5. Executes tool (async/sync)
6. Returns result as tool message

---

## 7. Registry & Service Discovery

### 7.1 Registration Lifecycle
```
Agent Created → AgentRegistry.register() → Weak Reference Stored
      ↓                    ↓                        ↓
  __init__            Name Generation         Return Name
```

### 7.2 Garbage Collection
- Weak references allow automatic cleanup
- No explicit unregister needed in most cases
- `__del__` method handles edge cases

---

## 8. Logging & Monitoring Infrastructure

### 8.1 Logging Hierarchy
```
ProgressLogger → RequestContext Queue → Progress Monitor
      ↓                ↓                      ↓
 Standard Log    Async Updates          Background Task
```

### 8.2 Log Levels Usage
- **MINIMAL**: Errors, critical events
- **SUMMARY**: Major steps, agent invocations
- **DETAILED**: Step details, parsing info
- **DEBUG**: Full payloads, internal state

---

## 9. Implementation Patterns & Best Practices

### 9.1 Creating a Basic Agent
```python
from src.agents import Agent
from src.models.models import ModelConfig

config = ModelConfig(
    type="api",
    provider="openai",
    name="gpt-4.1-mini,
    temperature=0.7,
    max_tokens=2000
)

agent = Agent(
    model_config=config,
    description="You are a helpful assistant",
    tools={"calculate": calculate_fn},
    allowed_peers=["researcher"]
)

result = await agent.auto_run("Solve this problem...")
```

### 9.2 Multi-Agent System
```python
# Specialized agents
researcher = Agent(
    name="researcher",
    model_config=config,
    description="You research and gather information",
    allowed_peers=["analyzer"]
)

analyzer = Agent(
    name="analyzer", 
    model_config=config,
    description="You analyze data and draw conclusions",
    allowed_peers=["researcher"]
)

coordinator = Agent(
    name="coordinator",
    model_config=config,
    description="You coordinate between researcher and analyzer",
    allowed_peers=["researcher", "analyzer"]
)

# Coordinator manages workflow
result = await coordinator.auto_run(
    "Research climate change impacts and analyze the data"
)
```

### 9.3 Custom Memory Processing
```python
class CustomAgent(Agent):
    def _input_message_processor(self):
        def processor(llm_response):
            # Extract custom fields
            if "custom_action" in llm_response:
                llm_response["agent_call"] = {
                    "agent_name": llm_response["custom_action"]["target"],
                    "request": llm_response["custom_action"]["payload"]
                }
            return llm_response
        return processor
    
    def _output_message_processor(self):
        def processor(message_dict):
            # Transform for specific LLM
            if message_dict.get("agent_call"):
                # Synthesize custom format
                message_dict["custom_action"] = {
                    "type": "delegate",
                    "target": message_dict["agent_call"]["agent_name"]
                }
            return message_dict
        return processor
```

### 9.4 Tool Development Pattern
```python
async def advanced_search(
    query: str,
    filters: Dict[str, Any] = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform advanced search with filters.
    
    Args:
        query: Search query string
        filters: Optional filters as key-value pairs
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    # Implementation
    results = await search_api.search(query, filters)
    return results[:max_results]

# Tool automatically gets schema generated
agent = Agent(
    model_config=config,
    tools={"advanced_search": advanced_search}
)
```

### 9.5 Error Handling Pattern
```python
# All errors return Message objects
try:
    response = await agent.invoke_agent("helper", request, context)
    if response.role == "error":
        # Handle error gracefully
        fallback_response = await agent._run(
            "The helper agent failed. Try solving yourself.",
            context, 
            run_mode="fallback"
        )
        return fallback_response
except Exception as e:
    return Message(
        role="error",
        content=f"System error: {str(e)}",
        name=agent.name
    )
```

---

## 10. Advanced Use Cases & Extensions

### 10.1 Hierarchical Agent Systems
```python
class TeamLead(Agent):
    def __init__(self, team_members: List[str], **kwargs):
        super().__init__(
            allowed_peers=team_members,
            **kwargs
        )
        self.team = team_members
    
    async def delegate_task(self, task: str):
        # Analyze task
        analysis = await self._run(
            f"Analyze this task and determine which team member should handle it: {task}",
            run_mode="analyze"
        )
        
        # Extract chosen agent
        chosen_agent = self._extract_agent_from_analysis(analysis)
        
        # Delegate
        return await self.invoke_agent(chosen_agent, task)
```

### 10.2 Learning Agent Implementation
```python
class LearnableAgent(Agent):
    def __init__(self, learning_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.experience_buffer = []
        self.performance_history = []
    
    async def learn_from_feedback(
        self, 
        task: str, 
        response: str, 
        feedback: float
    ):
        # Store experience
        self.experience_buffer.append({
            "task": task,
            "response": response,
            "feedback": feedback,
            "timestamp": time.time()
        })
        
        # Update behavior based on feedback
        if feedback < 0.5:
            # Poor performance - adjust approach
            self.description += f"\nNote: Previous approach to '{task[:50]}...' was ineffective."
        
        self.performance_history.append(feedback)
```

### 10.3 Dynamic Tool Creation
```python
async def create_dynamic_tool(agent: Agent, tool_spec: str):
    # Generate tool code
    code_response = await agent._run(
        f"Generate a Python async function for: {tool_spec}",
        run_mode="code_generation"
    )
    
    # Extract and validate code
    code = extract_code_block(code_response.content)
    
    # Create function in controlled namespace
    namespace = {"asyncio": asyncio, "aiohttp": aiohttp}
    exec(code, namespace)
    
    # Get function and add to agent
    func_name = extract_function_name(code)
    new_tool = namespace[func_name]
    
    # Generate schema and register
    schema = generate_openai_tool_schema(new_tool, func_name)
    agent.tools[func_name] = new_tool
    agent.tools_schema.append(schema)
    
    return func_name
```

### 10.4 Consensus Mechanisms
```python
async def multi_agent_consensus(
    agents: List[Agent],
    task: str,
    consensus_threshold: float = 0.7
) -> str:
    # Gather all responses
    responses = await asyncio.gather(*[
        agent.auto_run(task) for agent in agents
    ])
    
    # Create consensus coordinator
    coordinator = Agent(
        name="consensus_coordinator",
        model_config=strong_model_config,
        description="You analyze multiple responses and determine consensus"
    )
    
    # Analyze consensus
    consensus_prompt = {
        "prompt": "Analyze these responses and provide a consensus answer",
        "responses": responses,
        "threshold": consensus_threshold
    }
    
    return await coordinator.auto_run(consensus_prompt)
```

### 10.5 Agent Specialization Pipeline
```python
class SpecializationPipeline:
    def __init__(self, base_config: ModelConfig):
        self.base_config = base_config
        self.specializations = {}
    
    async def create_specialist(
        self, 
        domain: str,
        training_examples: List[Dict]
    ) -> Agent:
        # Create base agent
        agent = Agent(
            name=f"{domain}_specialist",
            model_config=self.base_config,
            description=f"You are a specialist in {domain}"
        )
        
        # Train on examples
        for example in training_examples:
            response = await agent.auto_run(example["input"])
            if response != example["expected"]:
                # Refine behavior
                agent.description += f"\nWhen asked about {example['topic']}, consider: {example['guidance']}"
        
        self.specializations[domain] = agent
        return agent
```

### 10.6 Performance Optimization
```python
class CachedAgent(Agent):
    def __init__(self, cache_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.response_cache = {}
        self.cache_size = cache_size
    
    async def _run(self, prompt: Any, request_context: RequestContext, run_mode: str, **kwargs):
        # Create cache key
        cache_key = self._create_cache_key(prompt, run_mode)
        
        # Check cache
        if cache_key in self.response_cache:
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                "Cache hit for prompt"
            )
            return self.response_cache[cache_key]
        
        # Get fresh response
        response = await super()._run(prompt, request_context, run_mode, **kwargs)
        
        # Update cache
        if len(self.response_cache) >= self.cache_size:
            # Evict oldest
            oldest = min(self.response_cache.items(), key=lambda x: x[1].timestamp)
            del self.response_cache[oldest[0]]
        
        self.response_cache[cache_key] = response
        return response
```

### 10.7 Meta-Learning Capabilities

```python
class MetaLearningAgent(Agent):
    """Agent that learns how to learn"""
    
    async def optimize_learning_strategy(self, task_history: List[Dict]):
        """Analyze past learning experiences to improve learning approach"""
        
        # Analyze what worked and what didn't
        analysis = await self._run(
            {
                "prompt": "Analyze these learning experiences and identify patterns",
                "experiences": task_history
            },
            run_mode="analysis"
        )
        
        # Generate improved learning strategy
        new_strategy = await self._run(
            {
                "prompt": "Based on this analysis, design an improved learning strategy",
                "analysis": analysis.content
            },
            run_mode="strategy"
        )
        
        # Update agent's approach
        self.description = f"{self.description}\n\nLearning Strategy:\n{new_strategy.content}"
        
        return new_strategy
```

---

## Key Framework Insights

### Architecture Principles:
1. **Message-Centric**: Everything flows through Message objects for consistency
2. **Weak References**: Registry prevents memory leaks while enabling discovery
3. **Context Propagation**: RequestContext maintains state across call chains
4. **Processor Pipeline**: Flexible format transformation between different APIs
5. **Tool Abstraction**: Simple function → OpenAI schema transformation

### Design Patterns Used:
1. **Factory Pattern**: ModelConfig → Model creation
2. **Registry Pattern**: Agent discovery and management
3. **Template Method**: BaseAgent defines skeleton, subclasses implement specifics
4. **Facade Pattern**: MemoryManager simplifies memory interactions
5. **Strategy Pattern**: Different memory types for different use cases

### Safety & Robustness:
1. **Depth Limits**: Prevent infinite recursion in agent calls
2. **Interaction Limits**: Cap total operations per task
3. **Permission System**: allowed_peers restricts agent communication
4. **Error Messages**: Errors return Message objects, not exceptions
5. **Re-prompting**: Automatic retry for malformed responses

### Extensibility Points:
1. **Custom Agents**: Extend BaseAgent or Agent
2. **Memory Types**: Implement BaseMemory interface
3. **Model Providers**: Add new BaseAPIModel implementations
4. **Tool Systems**: Any function with proper annotations
5. **Message Processors**: Transform between format
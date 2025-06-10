# Topology & Crew Management

The MARSYS topology system provides structured management of agent collections ("crews") with configuration-driven initialization, resource management, and task execution coordination.

## Overview

The topology system enables:

- **Declarative Agent Configuration**: Define agent crews through configuration objects
- **Async Resource Management**: Proper initialization and cleanup of browser agents and other resources
- **Task Orchestration**: Coordinate multi-agent workflows with progress tracking
- **Learning Integration**: Support for reinforcement learning configurations

## Core Components

### AgentConfig

Pydantic model for configuring individual agents within a crew.

```python
from src.topology.crew import AgentConfig

config = AgentConfig(
    name="researcher",
    agent_class="Agent",
    model_config={
        "type": "api",
        "name": "gpt-4o",
        "provider": "openai",
        "max_tokens": 2048
    },
    system_prompt="You are a research assistant specializing in data analysis.",
    memory_type="conversation_history",
    allowed_peers=["analyzer", "reporter"],
    max_tokens=1024
)
```

#### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique agent identifier |
| `agent_class` | `str` | Agent class name ("Agent", "LearnableAgent", "BrowserAgent") |
| `model_config` | `Optional[Dict]` | Model configuration for Agent/BrowserAgent |
| `model_ref` | `Optional[str]` | Model reference for LearnableAgent |
| `system_prompt` | `str` | Base system prompt defining agent behavior |
| `learning_head` | `Optional[str]` | Learning head type for LearnableAgent |
| `learning_head_config` | `Optional[Dict]` | Learning head configuration |
| `tools` | `Optional[Dict]` | Available tool functions |
| `tools_schema` | `Optional[List[Dict]]` | JSON schema for tools |
| `generation_system_prompt` | `Optional[str]` | BrowserAgent "think" mode prompt |
| `critic_system_prompt` | `Optional[str]` | BrowserAgent "critic" mode prompt |
| `memory_type` | `Optional[str]` | Memory module type (default: "conversation_history") |
| `max_tokens` | `Optional[int]` | Default response length limit |
| `allowed_peers` | `Optional[List[str]]` | Agents this agent can invoke |
| `temp_dir` | `Optional[str]` | Temporary file directory (BrowserAgent) |
| `headless_browser` | `bool` | Headless browser mode (default: True) |
| `browser_init_timeout` | `Optional[int]` | Browser initialization timeout |

### BaseCrew

Main class for managing agent collections and task execution.

```python
from src.topology.crew import BaseCrew, AgentConfig

# Define agent configurations
agent_configs = [
    AgentConfig(
        name="coordinator",
        agent_class="Agent",
        model_config={"type": "api", "name": "gpt-4o", "provider": "openai"},
        system_prompt="Coordinate research tasks and delegate to specialists.",
        allowed_peers=["researcher", "analyzer"]
    ),
    AgentConfig(
        name="researcher", 
        agent_class="BrowserAgent",
        model_config={"type": "api", "name": "gpt-4o-mini", "provider": "openai"},
        generation_system_prompt="Research topics using web browsing.",
        temp_dir="./tmp/research_screenshots",
        allowed_peers=["analyzer"]
    ),
    AgentConfig(
        name="analyzer",
        agent_class="Agent", 
        model_config={"type": "api", "name": "claude-3-sonnet", "provider": "anthropic"},
        system_prompt="Analyze research data and generate insights.",
        allowed_peers=[]
    )
]

# Create crew asynchronously
crew = await BaseCrew.create(agent_configs)
```

## Agent Initialization Patterns

### Standard Agent Initialization

```python
# Agent class initialization
if agent_cls is Agent:
    if not config.model_config:
        raise ValueError(f"Agent '{config.name}' requires 'model_config'.")
    
    agent = Agent(
        model_config=config.model_config,
        agent_name=config.name,
        system_prompt=config.system_prompt,
        tools=config.tools,
        memory_type=config.memory_type,
        max_tokens=config.max_tokens,
        allowed_peers=config.allowed_peers
    )
```

### BrowserAgent Async Initialization

```python
# BrowserAgent requires async initialization
async def init_browser_agent(args_dict):
    try:
        instance = await BrowserAgent.create_safe(**args_dict)
        self.agents[instance.name] = instance
        logging.info(f"Initialized agent: {instance.name}")
    except Exception as e:
        logging.error(f"Failed to initialize BrowserAgent: {e}")
        raise

# Add to initialization tasks
initialization_tasks.append(init_browser_agent(browser_args))

# Execute all async initializations
await asyncio.gather(*initialization_tasks)
```

### LearnableAgent with Model Loading

```python
# LearnableAgent with custom model loading
if agent_cls is LearnableAgent:
    if not config.model_ref:
        raise ValueError(f"LearnableAgent needs model reference.")
    
    # Load or create model based on model_ref
    if config.model_ref not in loaded_models:
        loaded_models[config.model_ref] = load_model(config.model_ref)
    
    agent = LearnableAgent(
        model=loaded_models[config.model_ref],
        description=config.system_prompt,
        learning_head=config.learning_head,
        learning_head_config=config.learning_head_config,
        agent_name=config.name,
        allowed_peers=config.allowed_peers
    )
```

## Task Execution

### Running Multi-Agent Tasks

```python
from src.agents.utils import LogLevel

# Execute task starting with specific agent
result, progress_queue = await crew.run_task(
    initial_agent_name="coordinator",
    initial_prompt="Research the latest developments in quantum computing",
    log_level=LogLevel.SUMMARY,
    max_depth=5,
    max_interactions=10
)

# Monitor progress
async def monitor_progress():
    while True:
        try:
            update = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
            if update is None:  # Task completed
                break
            print(f"[{update.agent_name}] {update.message}")
        except asyncio.TimeoutError:
            continue

# Run monitoring alongside task
await asyncio.gather(
    crew.run_task(...),
    monitor_progress()
)
```

### Task Context and Control

```python
# Task execution creates RequestContext
from src.agents.utils import RequestContext

task_id = str(uuid.uuid4())
progress_queue = asyncio.Queue()

request_context = RequestContext(
    task_id=task_id,
    depth=0,
    max_depth=max_depth,
    interaction_count=0, 
    max_interactions=max_interactions,
    progress_queue=progress_queue,
    log_level=log_level
)

# Invoke initial agent
result = await initial_agent.handle_invocation(
    request=initial_prompt,
    request_context=request_context
)
```

## Advanced Configuration Patterns

### Multi-Modal Research Crew

```python
research_crew_config = [
    # Coordinator agent
    AgentConfig(
        name="coordinator",
        agent_class="Agent",
        model_config={
            "type": "api",
            "name": "gpt-4o", 
            "provider": "openai",
            "max_tokens": 2048
        },
        system_prompt="""
        You coordinate research tasks across multiple specialists.
        Break down complex research requests and delegate to appropriate agents.
        """,
        allowed_peers=["web_researcher", "data_analyst", "vision_analyst"],
        memory_type="conversation_history"
    ),
    
    # Web browsing specialist
    AgentConfig(
        name="web_researcher",
        agent_class="BrowserAgent", 
        model_config={
            "type": "api",
            "name": "gpt-4o-mini",
            "provider": "openai"
        },
        generation_system_prompt="""
        Research topics using web browsing. Navigate websites, extract information,
        and take screenshots for visual analysis.
        """,
        critic_system_prompt="Evaluate the quality and relevance of gathered web information.",
        temp_dir="./tmp/research",
        headless_browser=True,
        allowed_peers=["vision_analyst"]
    ),
    
    # Vision analysis specialist  
    AgentConfig(
        name="vision_analyst",
        agent_class="Agent",
        model_config={
            "type": "local",
            "name": "microsoft/kosmos-2-patch14-224",
            "model_class": "vlm",
            "device_map": "auto"
        },
        system_prompt="""
        Analyze images, charts, and visual content. Extract text, describe content,
        and identify relevant information from visual sources.
        """,
        allowed_peers=["data_analyst"]
    ),
    
    # Data analysis specialist
    AgentConfig(
        name="data_analyst", 
        agent_class="LearnableAgent",
        model_ref="analysis_model",
        system_prompt="""
        Analyze structured and unstructured data. Generate insights, summaries,
        and conclusions from research findings.
        """,
        learning_head="peft",
        learning_head_config={
            "target_modules": ["q_proj", "v_proj"],
            "lora_rank": 16,
            "lora_alpha": 32
        },
        allowed_peers=[]
    )
]
```

### Learning-Enhanced Crew

```python
from src.learning.rl import GRPOConfig

# Configure reinforcement learning
learning_config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=3
)

# Create crew with learning configuration
crew = await BaseCrew.create(
    agent_configs=research_crew_config,
    learning_config=learning_config
)
```

## Resource Management

### Automatic Cleanup

```python
class BaseCrew:
    async def cleanup(self):
        """Clean up crew resources"""
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logging.warning(f"Cleanup failed for {agent.name}: {e}")
        
        # Clear agent registry
        for name in list(self.agents.keys()):
            AgentRegistry.unregister(name)

# Usage with context manager
async def run_research_task():
    crew = await BaseCrew.create(agent_configs)
    try:
        result = await crew.run_task("coordinator", "Research quantum computing")
        return result
    finally:
        await crew.cleanup()
```

### Error Handling in Initialization

```python
async def _initialize_agents(self):
    """Initialize agents with proper error handling"""
    initialization_tasks = []
    
    for config in self.agent_configs:
        try:
            if config.agent_class == "BrowserAgent":
                # Async initialization for browser agents
                task = self._init_browser_agent(config)
                initialization_tasks.append(task)
            else:
                # Sync initialization for other agents
                agent = self._create_agent(config)
                self.agents[config.name] = agent
                
        except Exception as e:
            logging.error(f"Failed to configure agent '{config.name}': {e}")
            raise ValueError(f"Agent configuration failed: {config.name}")
    
    # Execute async initializations
    if initialization_tasks:
        try:
            await asyncio.gather(*initialization_tasks)
        except Exception as e:
            # Cleanup partial initializations
            await self.cleanup()
            raise
    
    # Verify all agents initialized
    missing_agents = [
        config.name for config in self.agent_configs 
        if config.name not in self.agents
    ]
    if missing_agents:
        raise RuntimeError(f"Agents failed to initialize: {missing_agents}")
```

## Best Practices

### Configuration Design

1. **Modular Configs**: Create reusable configuration templates
2. **Clear Responsibilities**: Define specific roles for each agent
3. **Permission Management**: Carefully configure `allowed_peers`
4. **Resource Limits**: Set appropriate `max_tokens` and timeouts

### Agent Orchestration

1. **Task Decomposition**: Use coordinator agents for complex workflows
2. **Dependency Management**: Order agent initialization by dependencies
3. **Error Recovery**: Implement fallback strategies for failed agents
4. **Progress Monitoring**: Use progress queues for task visibility

### Performance Optimization

1. **Async Operations**: Leverage async initialization for heavy resources
2. **Resource Pooling**: Share models across compatible agents
3. **Cleanup Protocols**: Implement proper resource cleanup
4. **Memory Management**: Monitor agent memory usage in long-running tasks

### Development Workflow

```python
# Development configuration with debugging
debug_config = [
    AgentConfig(
        name="debug_agent",
        agent_class="Agent",
        model_config={
            "type": "api",
            "name": "gpt-4o-mini",  # Faster for development
            "provider": "openai",
            "temperature": 0.1  # More deterministic
        },
        system_prompt="Debug agent for testing workflows.",
        allowed_peers=[],
        max_tokens=512  # Shorter responses for faster iteration
    )
]

# Production configuration
production_config = [
    AgentConfig(
        name="production_agent",
        agent_class="Agent", 
        model_config={
            "type": "api",
            "name": "gpt-4o",  # Best performance
            "provider": "openai",
            "temperature": 0.7
        },
        system_prompt="Production agent with full capabilities.",
        allowed_peers=["specialist_1", "specialist_2"],
        max_tokens=2048
    )
]
```

The topology system provides a robust foundation for building complex multi-agent systems with proper resource management, error handling, and task coordination capabilities. 
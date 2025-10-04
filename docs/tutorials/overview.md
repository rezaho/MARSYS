# Tutorials

Step-by-step guides to master MARSYS through practical examples and real-world projects.

## 🎯 Learning Path

Progressive tutorials from beginner to advanced:

<div class="grid cards" markdown="1">

- :material-play-circle:{ .lg .middle } **Getting Started**

    ---

    New to MARSYS? Start here!

    1. [Installation & Setup](../getting-started/installation.md)
    2. [Your First Agent](../getting-started/first-agent.md)
    3. [Basic Usage](basic-usage.md)

- :material-rocket-launch:{ .lg .middle } **Core Concepts**

    ---

    Build your foundation

    1. [Agent Communication](../concepts/communication.md)
    2. [Memory Management](../concepts/memory.md)
    3. [Tool Integration](../concepts/tools.md)
    4. [Error Handling](../concepts/error-handling.md)

- :material-layers:{ .lg .middle } **Multi-Agent Systems**

    ---

    Coordinate multiple agents

    1. [Topology Patterns](../concepts/advanced/topology.md)
    2. [Orchestra Coordination](../api/orchestra.md)
    3. [State Management](../concepts/state-management.md)

- :material-brain:{ .lg .middle } **Advanced Topics**

    ---

    Master advanced features

    1. [Custom Agents](../concepts/custom-agents.md)
    2. [Learning Agents](../concepts/learning-agents.md)
    3. [Browser Automation](../concepts/browser-automation.md)
    4. [Memory Patterns](../concepts/memory-patterns.md)

</div>

## 📚 Tutorial Tracks

### Track 1: Build a Research Assistant

**Goal**: Create a multi-agent system for comprehensive research

#### Part 1: Basic Research Agent
```python
from marsys.coordination import Orchestra
from marsys.agents import Agent
from marsys.models import ModelConfig

# Create research agent
researcher = Agent(
    model_config=ModelConfig(
        type="api",
        provider="openai",
        name="gpt-4"
    ),
    agent_name="Researcher",
    description="Research and analyze topics",
    system_prompt="You are a thorough researcher..."
)

# Simple research task
result = await Orchestra.run(
    task="Research quantum computing applications",
    topology={"nodes": ["Researcher"], "edges": []}
)
```

#### Part 2: Add Web Search
```python
def search_web(query: str) -> List[Dict]:
    """Search the web for information."""
    # Implementation here
    return results

researcher_with_tools = Agent(
    model_config=config,
    agent_name="WebResearcher",
    tools={"search_web": search_web}
)
```

#### Part 3: Multi-Agent Research Team
```python
from marsys.coordination.topology.patterns import PatternConfig

# Create research team
data_collector = Agent(agent_name="DataCollector", ...)
analyzer = Agent(agent_name="Analyzer", ...)
writer = Agent(agent_name="Writer", ...)

# Hub-and-spoke pattern
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["DataCollector", "Analyzer", "Writer"],
    parallel_spokes=True
)

result = await Orchestra.run(
    task="Comprehensive research on AI trends",
    topology=topology
)
```

### Track 2: Build a Customer Service Bot

**Goal**: Create an intelligent customer service system

#### Part 1: Basic Support Agent
```python
support_agent = Agent(
    agent_name="SupportAgent",
    description="Customer support specialist",
    system_prompt="""You are a helpful customer support agent.
    - Be polite and professional
    - Ask clarifying questions
    - Provide accurate information
    - Escalate when needed"""
)
```

#### Part 2: Add Knowledge Base
```python
from marsys.concepts.memory_patterns import SemanticMemory

class KnowledgeBaseAgent(BaseAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.knowledge = SemanticMemory()

        # Load knowledge base
        self.load_faqs()
        self.load_documentation()

    async def _run(self, prompt, context, **kwargs):
        # Search knowledge base
        relevant_info = self.knowledge.query_facts(prompt)

        # Enhanced prompt with context
        enhanced = f"{prompt}\n\nRelevant info: {relevant_info}"

        # Generate response
        response = await self.model.run(enhanced)
        return response
```

#### Part 3: Escalation System
```python
topology = {
    "nodes": ["User", "L1Support", "L2Support", "Manager"],
    "edges": [
        "User -> L1Support",
        "L1Support -> User",
        "L1Support -> L2Support",  # Escalation
        "L2Support -> User",
        "L2Support -> Manager",    # Further escalation
        "Manager -> User"
    ]
}

# Agents can escalate complex issues
l1_response = {
    "next_action": "invoke_agent",
    "action_input": {
        "agent_name": "L2Support",
        "request": "Complex technical issue needs expertise"
    }
}
```

### Track 3: Build a Data Analysis Pipeline

**Goal**: Create an automated data analysis system

#### Part 1: Data Processor
```python
class DataProcessor(BaseAgent):
    async def _run(self, prompt, context, **kwargs):
        # Extract data request
        data_request = self.parse_request(prompt)

        # Process data
        if data_request["type"] == "csv":
            data = self.process_csv(data_request["path"])
        elif data_request["type"] == "json":
            data = self.process_json(data_request["path"])

        return Message(
            role="assistant",
            content="Data processed successfully",
            structured_data=data
        )
```

#### Part 2: Analysis Pipeline
```python
topology = PatternConfig.pipeline(
    stages=[
        {"name": "ingestion", "agents": ["DataIngester"]},
        {"name": "cleaning", "agents": ["DataCleaner"]},
        {"name": "analysis", "agents": ["StatAnalyzer", "MLAnalyzer"]},
        {"name": "visualization", "agents": ["Visualizer"]},
        {"name": "reporting", "agents": ["ReportWriter"]}
    ],
    parallel_within_stage=True
)
```

#### Part 3: Real-time Monitoring
```python
from marsys.coordination.state import StateManager

# Enable state persistence
state_manager = StateManager(storage)

# Run with monitoring
result = await Orchestra.run(
    task="Analyze Q4 sales data",
    topology=topology,
    state_manager=state_manager,
    execution_config=ExecutionConfig(
        status=StatusConfig(
            enabled=True,
            verbosity=2,
            show_agent_thoughts=True
        )
    )
)
```

## 🎓 Best Practices Examples

### Error Handling Pattern
```python
class RobustAgent(BaseAgent):
    async def _run(self, prompt, context, **kwargs):
        try:
            result = await self.process(prompt)
            return Message(
                role="assistant",
                content=result
            )
        except ValidationError as e:
            # Recoverable error
            return Message(
                role="error",
                content=f"Validation issue: {e}",
                metadata={"recoverable": True}
            )
        except APIError as e:
            # Route to user for help
            return Message(
                role="error",
                content="API error - need user intervention",
                metadata={"route_to_user": True}
            )
```

### Memory Management Pattern
```python
class MemoryAgent(BaseAgent):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.episodic = EpisodicMemory(max_episodes=100)
        self.working = WorkingMemory(capacity=7)

    async def _run(self, prompt, context, **kwargs):
        # Start new episode
        episode_id = self.episodic.start_episode(prompt)

        # Use working memory
        self.working.add(prompt, priority=1.0)

        # Process with memory context
        relevant_memories = self.episodic.retrieve_similar(prompt)
        enhanced_prompt = self.enhance_with_memory(prompt, relevant_memories)

        response = await self.model.run(enhanced_prompt)

        # End episode (in actual implementation)
        self.episodic.end_episode(episode_id, "success")

        return response
```

### Parallel Processing Pattern
```python
from marsys.agents import AgentPool

# Create pool for parallel processing
pool = AgentPool(
    agent_class=DataAnalyzer,
    num_instances=5,
    model_config=config,
    agent_name="AnalyzerPool"
)

# Coordinator distributes work
coordinator_response = {
    "next_action": "parallel_invoke",
    "agents": ["AnalyzerPool"] * 5,
    "agent_requests": {
        "AnalyzerPool_0": "Analyze dataset A",
        "AnalyzerPool_1": "Analyze dataset B",
        "AnalyzerPool_2": "Analyze dataset C",
        "AnalyzerPool_3": "Analyze dataset D",
        "AnalyzerPool_4": "Analyze dataset E"
    }
}
```

## 🚀 Quick Start Projects

### 1. **Question Answering System**
```python
# 10 minutes to build
qa_agent = Agent(
    agent_name="QABot",
    system_prompt="Answer questions accurately and concisely."
)

result = await Orchestra.run(
    task="What is the capital of France?",
    topology={"nodes": ["QABot"], "edges": []}
)
```

### 2. **Document Summarizer**
```python
# 20 minutes to build
summarizer = Agent(
    agent_name="Summarizer",
    system_prompt="""Summarize documents:
    - Extract key points
    - Maintain accuracy
    - Be concise"""
)

result = await Orchestra.run(
    task=f"Summarize this document: {document_text}",
    topology={"nodes": ["Summarizer"], "edges": []}
)
```

### 3. **Code Assistant**
```python
# 30 minutes to build
code_helper = Agent(
    agent_name="CodeAssistant",
    system_prompt="""You are a coding assistant.
    - Explain code clearly
    - Suggest improvements
    - Fix bugs
    - Write clean code"""
)

result = await Orchestra.run(
    task="Fix this Python function: ...",
    topology={"nodes": ["CodeAssistant"], "edges": []}
)
```

## 📖 Tutorial Resources

### Example Code
All tutorial code is available in the [examples/](https://github.com/yourusername/marsys/tree/main/examples) directory.

### Interactive Notebooks
Jupyter notebooks for hands-on learning are in [tutorials/notebooks/](https://github.com/yourusername/marsys/tree/main/tutorials/notebooks).

### Video Tutorials
Video walkthroughs available on our [YouTube channel](https://youtube.com/@marsys).

## 🎯 Next Steps

<div class="grid cards" markdown="1">

- :material-file-document:{ .lg .middle } **[API Reference](../api/overview.md)**

    ---

    Complete API documentation

- :material-lightbulb:{ .lg .middle } **[Use Cases](../use-cases/index.md)**

    ---

    Real-world applications

- :material-tools:{ .lg .middle } **[Contributing](../contributing/index.md)**

    ---

    Join the community

- :material-help-circle:{ .lg .middle } **[Support](../support.md)**

    ---

    Get help and support

</div>

---

!!! tip "Pro Tip"
    Start with the Quick Start Projects to get a feel for MARSYS, then progress through the tutorial tracks to build complete systems. Each track is designed to be completed in 2-4 hours.

!!! success "Ready to Build!"
    You now have everything you need to start building with MARSYS. Pick a tutorial track that matches your interests and start creating!
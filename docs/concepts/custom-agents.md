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
from src.agents.base_agent import BaseAgent
from src.models.message import Message
from src.utils.types import RequestContext

class CustomAgent(BaseAgent):
    """Custom agent with specialized behavior."""
    
    def __init__(self, specialization: str, **kwargs):
        super().__init__(**kwargs)
        self.specialization = specialization
    
    async def process_task(self, task: str, context: RequestContext) -> Message:
        """Custom task processing logic."""
        # Your implementation here
        result = f"Processed '{task}' with {self.specialization} specialization"
        
        return Message(
            role="assistant",
            content=result,
            name=self.name
        )
```

### Agent with State Management

```python
from typing import Dict, Any, Optional
import json

class StatefulAgent(Agent):
    """Agent that maintains internal state."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state: Dict[str, Any] = {}
        self.state_file = f"{self.name}_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load state from persistent storage."""
        try:
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        except FileNotFoundError:
            self.state = {}
    
    def _save_state(self):
        """Save state to persistent storage."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
    
    async def update_state(self, key: str, value: Any) -> None:
        """Update internal state."""
        self.state[key] = value
        self._save_state()
    
    async def get_state(self, key: str) -> Optional[Any]:
        """Retrieve state value."""
        return self.state.get(key)
```

### Domain-Specific Agent

```python
class DataAnalystAgent(Agent):
    """Specialized agent for data analysis tasks."""
    
    def __init__(self, **kwargs):
        # Set specific instructions for data analysis
        kwargs['instructions'] = """
        You are a data analyst agent. Your responsibilities:
        1. Analyze datasets and identify patterns
        2. Generate statistical summaries
        3. Create visualizations (describe them)
        4. Provide actionable insights
        
        Always be precise with numbers and cite your calculations.
        """
        
        super().__init__(**kwargs)
        
        # Add specialized tools
        self.tools.update({
            "calculate_statistics": self._calculate_statistics,
            "detect_anomalies": self._detect_anomalies,
            "generate_report": self._generate_report
        })
    
    async def _calculate_statistics(self, data: str) -> str:
        """Calculate basic statistics for dataset."""
        # Parse data
        try:
            values = [float(x) for x in data.split(',')]
            stats = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values)
            }
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    async def _detect_anomalies(self, data: str, threshold: float = 2.0) -> str:
        """Detect anomalies using simple z-score method."""
        # Implementation here
        pass
    
    async def _generate_report(self, analysis: str) -> str:
        """Generate formatted analysis report."""
        # Implementation here
        pass
```

## Advanced Custom Agent Patterns

### Plugin-Based Agent

```python
from abc import ABC, abstractmethod
from typing import List

class AgentPlugin(ABC):
    """Base class for agent plugins."""
    
    @abstractmethod
    def get_tools(self) -> Dict[str, callable]:
        """Return tools provided by this plugin."""
        pass
    
    @abstractmethod
    def get_instructions(self) -> str:
        """Return additional instructions for the agent."""
        pass

class PluggableAgent(Agent):
    """Agent that supports plugins."""
    
    def __init__(self, plugins: List[AgentPlugin] = None, **kwargs):
        super().__init__(**kwargs)
        self.plugins = plugins or []
        self._load_plugins()
    
    def _load_plugins(self):
        """Load tools and instructions from plugins."""
        for plugin in self.plugins:
            # Add plugin tools
            self.tools.update(plugin.get_tools())
            
            # Append plugin instructions
            self.instructions += f"\n\n{plugin.get_instructions()}"
    
    def add_plugin(self, plugin: AgentPlugin):
        """Dynamically add a plugin."""
        self.plugins.append(plugin)
        self.tools.update(plugin.get_tools())
        self.instructions += f"\n\n{plugin.get_instructions()}"

# Example plugin
class DatabasePlugin(AgentPlugin):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def get_tools(self) -> Dict[str, callable]:
        return {
            "query_database": self._query_database,
            "update_database": self._update_database
        }
    
    def get_instructions(self) -> str:
        return """
        Database Plugin Instructions:
        - Use query_database to retrieve data
        - Use update_database to modify records
        - Always validate SQL queries before execution
        """
    
    async def _query_database(self, query: str) -> str:
        # Database query implementation
        pass
```

### Reactive Agent

```python
import asyncio
from typing import Callable

class ReactiveAgent(Agent):
    """Agent that reacts to events."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    async def emit(self, event: str, data: Any = None):
        """Emit an event."""
        await self.event_queue.put((event, data))
    
    async def start(self):
        """Start event processing loop."""
        self.running = True
        asyncio.create_task(self._process_events())
    
    async def stop(self):
        """Stop event processing."""
        self.running = False
    
    async def _process_events(self):
        """Process events from queue."""
        while self.running:
            try:
                event, data = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                if event in self.event_handlers:
                    for handler in self.event_handlers[event]:
                        try:
                            await handler(data)
                        except Exception as e:
                            await self._log_progress(
                                RequestContext(request_id="event"),
                                LogLevel.MINIMAL,
                                f"Event handler error: {e}"
                            )
            except asyncio.TimeoutError:
                continue
```

### Learning Agent Implementation

```python
from collections import deque
import numpy as np

class CustomLearningAgent(Agent):
    """Agent that learns from interactions."""
    
    def __init__(self, learning_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.experience_buffer = deque(maxlen=1000)
        self.performance_metrics = []
    
    async def learn_from_feedback(self, task: str, response: str, feedback: float):
        """Learn from feedback on task performance."""
        # Store experience
        self.experience_buffer.append({
            "task": task,
            "response": response,
            "feedback": feedback,
            "timestamp": datetime.now()
        })
        
        # Update performance metrics
        self.performance_metrics.append(feedback)
        
        # Adjust behavior based on feedback
        if feedback < 0.5:  # Poor performance
            await self._adjust_strategy("increase_detail")
        elif feedback > 0.8:  # Good performance
            await self._adjust_strategy("maintain")
    
    async def _adjust_strategy(self, adjustment: str):
        """Adjust agent strategy based on learning."""
        if adjustment == "increase_detail":
            self.instructions += "\nProvide more detailed explanations."
        # Other adjustments...
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of agent performance."""
        if not self.performance_metrics:
            return {"average": 0, "trend": "neutral"}
        
        recent = self.performance_metrics[-10:]
        return {
            "average": np.mean(recent),
            "trend": "improving" if np.mean(recent) > np.mean(self.performance_metrics) else "declining",
            "total_interactions": len(self.experience_buffer)
        }
```

## Integration Patterns

### External API Integration

```python
import aiohttp

class APIIntegrationAgent(Agent):
    """Agent that integrates with external APIs."""
    
    def __init__(self, api_key: str, api_base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.session = None
        
        # Add API tools
        self.tools.update({
            "fetch_data": self._fetch_data,
            "post_data": self._post_data
        })
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _fetch_data(self, endpoint: str) -> str:
        """Fetch data from API endpoint."""
        if not self.session:
            return "Error: Session not initialized"
        
        try:
            url = f"{self.api_base_url}/{endpoint}"
            async with self.session.get(url) as response:
                data = await response.json()
                return json.dumps(data)
        except Exception as e:
            return f"API Error: {str(e)}"
```

### Database Integration

```python
import asyncpg

class DatabaseAgent(Agent):
    """Agent with database capabilities."""
    
    def __init__(self, db_url: str, **kwargs):
        super().__init__(**kwargs)
        self.db_url = db_url
        self.pool = None
        
        self.tools.update({
            "query": self._query,
            "insert": self._insert,
            "update": self._update
        })
    
    async def connect(self):
        """Establish database connection pool."""
        self.pool = await asyncpg.create_pool(self.db_url)
    
    async def disconnect(self):
        """Close database connections."""
        if self.pool:
            await self.pool.close()
    
    async def _query(self, sql: str) -> str:
        """Execute SELECT query."""
        if not self.pool:
            return "Error: Database not connected"
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(sql)
                return json.dumps([dict(row) for row in rows], default=str)
        except Exception as e:
            return f"Query Error: {str(e)}"
```

## Testing Custom Agents

```python
import pytest

class TestCustomAgent:
    @pytest.mark.asyncio
    async def test_custom_agent_creation():
        """Test custom agent can be created."""
        agent = CustomAgent(
            name="test_agent",
            specialization="testing",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        )
        
        assert agent.name == "test_agent"
        assert agent.specialization == "testing"
    
    @pytest.mark.asyncio
    async def test_custom_processing():
        """Test custom processing logic."""
        agent = CustomAgent(
            name="processor",
            specialization="data",
            model_config=ModelConfig(provider="openai", model_name="gpt-3.5-turbo")
        )
        
        context = RequestContext(request_id="test")
        result = await agent.process_task("test task", context)
        
        assert result.role == "assistant"
        assert "data" in result.content
```

## Best Practices

1. **Follow Framework Conventions**: Always return Message objects, use proper logging
2. **Handle Errors Gracefully**: Return error messages, not exceptions
3. **Document Your Agent**: Clear docstrings and usage examples
4. **Test Thoroughly**: Unit tests for all custom functionality
5. **Maintain State Carefully**: Consider persistence and concurrent access
6. **Use Type Hints**: Makes code more maintainable and self-documenting

## Common Patterns

### Specialized Instructions

```python
def create_specialized_agent(domain: str) -> Agent:
    """Factory for domain-specific agents."""
    instructions_map = {
        "legal": "You are a legal expert. Always cite relevant laws and precedents.",
        "medical": "You are a medical advisor. Always include disclaimers about consulting professionals.",
        "financial": "You are a financial analyst. Provide risk warnings and regulatory compliance notes."
    }
    
    return Agent(
        name=f"{domain}_specialist",
        instructions=instructions_map.get(domain, "You are a specialized assistant."),
        model_config=ModelConfig(provider="openai", model_name="gpt-4")
    )
```

### Capability Composition

```python
class ComposableAgent(Agent):
    """Agent with composable capabilities."""
    
    def add_capability(self, name: str, implementation: Callable):
        """Add a new capability dynamically."""
        self.tools[name] = implementation
        self.instructions += f"\n- You can now use '{name}' capability"
```

## Next Steps

- Explore [Memory Patterns](memory-patterns.md) - Advanced memory for custom agents
- Learn about [Browser Automation](browser-automation.md) - Web capabilities
- See [Examples](../use-cases/examples/advanced-examples.md#custom-agents) - Real implementations

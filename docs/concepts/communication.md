# Agent Communication

## Communication Protocols

### Handshake Protocol

Establish communication between agents:

```python
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext
import logging

class HandshakeProtocol:
    async def initiate_handshake(
        self,
        from_agent: Agent,
        to_agent_name: str
    ) -> bool:
        """Establish communication with another agent."""
        context = RequestContext(
            request_id=f"handshake_{from_agent.name}_{to_agent_name}",
            agent_name=from_agent.name,
            logger=logging.getLogger(__name__)
        )
        
        # Send handshake request
        handshake_msg = Message(
            role="assistant",
            content="HANDSHAKE_REQUEST",
            name=to_agent_name,
            metadata={
                "protocol": "handshake",
                "version": "1.0",
                "capabilities": from_agent.get_capabilities()
            }
        )
        
        response = await from_agent.invoke_agent(
            to_agent_name,
            handshake_msg.content,
            context=context
        )
        
        # Validate handshake response
        if response and hasattr(response, 'metadata'):
            return response.metadata.get("status") == "accepted"
        return False
```

### Request-Reply Protocol

Structured request-reply communication:

```python
import asyncio
import uuid
import json
import time
from typing import Any, Dict, List, Optional
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext

class RequestReplyProtocol:
    def __init__(self):
        self.pending_requests = {}
    
    async def send_request(
        self,
        from_agent: Agent,
        to_agent: str,
        request_type: str,
        payload: Any,
        timeout: float = 30.0
    ) -> Message:
        """Send request and wait for reply."""
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        context = RequestContext(
            request_id=request_id,
            agent_name=from_agent.name,
            logger=logging.getLogger(__name__)
        )
        
        # Create request message
        request_content = json.dumps({
            "request_type": request_type,
            "payload": payload,
            "reply_to": from_agent.name
        })
        
        # Track pending request
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request
        try:
            response = await from_agent.invoke_agent(
                to_agent, 
                request_content,
                context=context
            )
            
            # For immediate responses, resolve the future
            if request_id in self.pending_requests:
                self.pending_requests[request_id].set_result(response)
                del self.pending_requests[request_id]
            
            return response
            
        except asyncio.TimeoutError:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            raise
    
    def handle_reply(self, message: Message):
        """Process incoming reply."""
        request_id = message.metadata.get("in_reply_to")
        if request_id in self.pending_requests:
            self.pending_requests[request_id].set_result(message)
            del self.pending_requests[request_id]
```

## Communication Patterns Examples

### Map-Reduce Pattern

Distribute work and aggregate results:

```python
from src.agents.agents import Agent
from src.agents.utils import RequestContext
import asyncio
import json

class MapReduceCoordinator(Agent):
    def __init__(self, name: str, model_config: dict, **kwargs):
        super().__init__(name=name, model_config=model_config, **kwargs)
    
    async def map_reduce(
        self,
        data: List[Any],
        mapper_agents: List[str],
        reducer_agent: str,
        context: RequestContext
    ) -> Any:
        """Execute map-reduce pattern."""
        # Map phase - distribute work
        chunk_size = len(data) // len(mapper_agents)
        map_tasks = []
        
        for i, mapper in enumerate(mapper_agents):
            start = i * chunk_size
            end = start + chunk_size if i < len(mapper_agents) - 1 else len(data)
            chunk = data[start:end]
            
            task_context = RequestContext(
                request_id=f"{context.request_id}_map_{i}",
                agent_name=self.name,
                logger=context.logger
            )
            
            task = self.invoke_agent(
                mapper,
                f"Process this data chunk: {json.dumps(chunk)}",
                context=task_context
            )
            map_tasks.append(task)
        
        # Gather map results
        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)
        
        # Filter out exceptions and get successful results
        successful_results = [
            r for r in map_results 
            if not isinstance(r, Exception)
        ]
        
        # Reduce phase - aggregate results
        reduce_context = RequestContext(
            request_id=f"{context.request_id}_reduce",
            agent_name=self.name,
            logger=context.logger
        )
        
        reduce_result = await self.invoke_agent(
            reducer_agent,
            f"Aggregate these results: {json.dumps([r.content if hasattr(r, 'content') else str(r) for r in successful_results])}",
            context=reduce_context
        )
        
        return reduce_result
```

### Chain of Responsibility

Pass requests through a chain of handlers:

```python
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext
from typing import Optional, List

class ChainOfResponsibilityAgent(Agent):
    def __init__(
        self, 
        name: str, 
        model_config: dict,
        next_handler: Optional[str] = None,
        supported_types: List[str] = None,
        **kwargs
    ):
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.next_handler = next_handler
        self.supported_types = supported_types or []
    
    async def _process_request(self, message: Message, context: RequestContext) -> Message:
        """Process request or pass to next handler."""
        # Check if this agent can handle the request
        if self.can_handle(message):
            return await super()._process_request(message, context)
        
        # Pass to next handler
        if self.next_handler:
            return await self.invoke_agent(
                self.next_handler,
                message.content,
                context=context
            )
        
        # No handler found
        return Message(
            role="assistant",
            content="No handler found for request",
            metadata={"error": "no_handler", "agent": self.name}
        )
    
    def can_handle(self, message: Message) -> bool:
        """Check if agent can handle this request type."""
        request_type = message.metadata.get("type") if message.metadata else None
        return request_type in self.supported_types
```

### Observer Pattern

Notify multiple agents of state changes:

```python
from src.agents.agents import Agent
from src.agents.utils import RequestContext
import asyncio
from typing import Any, List

class ObservableAgent(Agent):
    def __init__(self, name: str, model_config: dict, **kwargs):
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.observers: List[str] = []
    
    def attach_observer(self, observer_name: str):
        """Add observer to notification list."""
        if observer_name not in self.observers:
            self.observers.append(observer_name)
    
    def detach_observer(self, observer_name: str):
        """Remove observer from notification list."""
        if observer_name in self.observers:
            self.observers.remove(observer_name)
    
    async def notify_observers(self, event: str, data: Any, context: RequestContext):
        """Notify all observers of state change."""
        notifications = []
        
        for observer in self.observers:
            notification_context = RequestContext(
                request_id=f"{context.request_id}_notify_{observer}",
                agent_name=self.name,
                logger=context.logger
            )
            
            notification = self.invoke_agent(
                observer,
                f"Event: {event}",
                context=notification_context
            )
            notifications.append(notification)
        
        # Fire and forget - don't wait for responses
        asyncio.gather(*notifications, return_exceptions=True)
```

## Error Handling in Communication

### Retry with Circuit Breaker

Implement circuit breaker pattern:

```python
import time
from typing import Any
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(
        self, 
        agent: Agent, 
        target: str, 
        task: str,
        context: RequestContext
    ) -> Message:
        """Execute call with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await agent.invoke_agent(target, task, context=context)
            
            # Success - reset on half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise
```

### Dead Letter Queue

Handle failed messages:

```python
from datetime import datetime
from typing import Optional, List, Dict
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext
import asyncio

class DeadLetterQueue:
    def __init__(self):
        self.failed_messages: List[Dict] = []
    
    async def process_with_dlq(
        self,
        agent: Agent,
        message: Message,
        context: RequestContext,
        max_retries: int = 3
    ) -> Optional[Message]:
        """Process message with dead letter queue for failures."""
        for attempt in range(max_retries):
            try:
                result = await agent._process_request(message, context)
                return result
            except Exception as e:
                context.logger.warning(
                    f"Attempt {attempt + 1} failed for agent {agent.name}: {e}"
                )
                
                if attempt == max_retries - 1:
                    # Add to dead letter queue
                    self.failed_messages.append({
                        "message": message,
                        "error": str(e),
                        "timestamp": datetime.now(),
                        "attempts": max_retries,
                        "agent": agent.name
                    })
                    return None
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return None
```

## Performance Optimization

### Message Batching

Batch multiple messages for efficiency:

```python
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext
import asyncio
from typing import List

class BatchingAgent(Agent):
    def __init__(
        self, 
        name: str, 
        model_config: dict,
        batch_size: int = 10, 
        batch_timeout: float = 1.0, 
        **kwargs
    ):
        super().__init__(name=name, model_config=model_config, **kwargs)
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.message_queue = asyncio.Queue()
        self.batch_processor_task = None
    
    async def start_batch_processor(self):
        """Start the batch processing loop."""
        self.batch_processor_task = asyncio.create_task(
            self._process_batches()
        )
    
    async def _process_batches(self):
        """Process messages in batches."""
        while True:
            batch = []
            deadline = asyncio.get_event_loop().time() + self.batch_timeout
            
            while len(batch) < self.batch_size:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(),
                        timeout=timeout
                    )
                    batch.append(message)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                await self._process_batch(batch)
    
    async def _process_batch(self, messages: List[Message]):
        """Process a batch of messages."""
        # Combine messages for efficient processing
        combined_content = "\n".join([
            f"[{i+1}] {msg.content}" 
            for i, msg in enumerate(messages)
        ])
        
        batch_message = Message(
            role="user", 
            content=f"Process these requests:\n{combined_content}"
        )
        
        context = RequestContext(
            request_id=f"batch_{len(messages)}",
            agent_name=self.name,
            logger=logging.getLogger(__name__)
        )
        
        result = await self._process_request(batch_message, context)
        
        # Parse and distribute results
        # Implementation depends on your specific batching logic
        return result
```

### Connection Pooling

Manage agent connections efficiently:

```python
from src.agents.registry import AgentRegistry
from src.agents.agents import Agent
from src.models.message import Message
from src.agents.utils import RequestContext
import asyncio
from typing import Dict

class AgentConnectionPool:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections: Dict[str, Agent] = {}
        self.semaphore = asyncio.Semaphore(max_connections)
    
    async def get_connection(self, agent_name: str) -> Agent:
        """Get or create connection to agent."""
        async with self.semaphore:
            if agent_name not in self.connections:
                agent = AgentRegistry.get_agent(agent_name)
                if not agent:
                    raise ValueError(f"Agent {agent_name} not found")
                self.connections[agent_name] = agent
            
            return self.connections[agent_name]
    
    async def invoke_with_pool(
        self,
        from_agent: Agent,
        to_agent: str,
        task: str,
        context: RequestContext
    ) -> Message:
        """Invoke agent using connection pool."""
        # Ensure target agent is available
        target = await self.get_connection(to_agent)
        return await from_agent.invoke_agent(to_agent, task, context=context)
```

## Best Practices

1. **Message Design**
   - Keep messages self-contained
   - Include necessary context in metadata
   - Use structured formats (JSON) for complex data
   - Always include proper RequestContext

2. **Error Handling**
   - Always handle communication failures
   - Implement timeouts for all remote calls
   - Use circuit breakers for unreliable connections
   - Log errors with proper context

3. **Performance**
   - Batch messages when possible
   - Use async operations throughout
   - Monitor communication latency
   - Use connection pooling for frequent communications

4. **Security**
   - Validate message sources
   - Sanitize message content
   - Implement access controls
   - Use proper authentication in RequestContext

5. **Monitoring**
   - Log all inter-agent communications
   - Track message flow and latency
   - Monitor failed communications
   - Use structured logging with RequestContext

## Next Steps

- Explore [Topologies](topologies.md) - Organizational patterns
- Learn about [Custom Agents](custom-agents.md) - Build specialized agents
- See [Memory Patterns](memory-patterns.md) - Advanced memory strategies
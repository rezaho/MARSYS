# Agent Communication

## Communication Protocols

### Handshake Protocol

Establish communication between agents:

```python
from src.agents import Agent
from src.agents.memory import Message
from src.agents.utils import RequestContext

class HandshakeProtocol:
    async def initiate_handshake(
        self,
        from_agent: Agent,
        to_agent_name: str
    ) -> bool:
        """Establish communication with another agent."""
        # Send handshake request
        handshake_msg = Message(
            role="agent_call",
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
            context=RequestContext(
                request_id=f"handshake_{from_agent.name}_{to_agent_name}",
                agent_name=from_agent.name
            )
        )
        
        # Validate handshake response
        if response.metadata.get("protocol") == "handshake":
            return response.metadata.get("status") == "accepted"
        return False
```

### Request-Reply Protocol

Structured request-reply communication:

```python
import asyncio
import uuid
import json
from typing import Any, Callable, Dict, List, Optional

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
        
        # Create request message
        request = Message(
            role="agent_call",
            content=json.dumps(payload),
            name=to_agent,
            metadata={
                "request_id": request_id,
                "request_type": request_type,
                "reply_to": from_agent.name
            }
        )
        
        # Track pending request
        future = asyncio.Future()
        self.pending_requests[request_id] = future
        
        # Send request
        asyncio.create_task(
            from_agent.invoke_agent(to_agent, request.content)
        )
        
        # Wait for reply with timeout
        try:
            reply = await asyncio.wait_for(future, timeout)
            return reply
        except asyncio.TimeoutError:
            del self.pending_requests[request_id]
            raise
    
    def handle_reply(self, message: Message):
        """Process incoming reply."""
        request_id = message.metadata.get("in_reply_to")
        if request_id in self.pending_requests:
            self.pending_requests[request_id].set_result(message)
```

## Communication Patterns Examples

### Map-Reduce Pattern

Distribute work and aggregate results:

```python
class MapReduceCoordinator(Agent):
    async def map_reduce(
        self,
        data: List[Any],
        mapper_agents: List[str],
        reducer_agent: str
    ) -> Any:
        """Execute map-reduce pattern."""
        # Map phase - distribute work
        chunk_size = len(data) // len(mapper_agents)
        map_tasks = []
        
        for i, mapper in enumerate(mapper_agents):
            start = i * chunk_size
            end = start + chunk_size if i < len(mapper_agents) - 1 else len(data)
            chunk = data[start:end]
            
            task = self.invoke_agent(
                mapper,
                f"Process this data chunk: {json.dumps(chunk)}"
            )
            map_tasks.append(task)
        
        # Gather map results
        map_results = await asyncio.gather(*map_tasks)
        
        # Reduce phase - aggregate results
        reduce_result = await self.invoke_agent(
            reducer_agent,
            f"Aggregate these results: {[r.content for r in map_results]}"
        )
        
        return reduce_result
```

### Chain of Responsibility

Pass requests through a chain of handlers:

```python
class ChainOfResponsibilityAgent(Agent):
    def __init__(self, next_handler: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.next_handler = next_handler
    
    async def handle_request(self, request: Message) -> Message:
        """Process request or pass to next handler."""
        # Check if this agent can handle the request
        if self.can_handle(request):
            return await self.process_request(request)
        
        # Pass to next handler
        if self.next_handler:
            return await self.invoke_agent(
                self.next_handler,
                request.content,
                metadata=request.metadata
            )
        
        # No handler found
        return Message(
            role="error",
            content="No handler found for request",
            name=self.name
        )
    
    def can_handle(self, request: Message) -> bool:
        """Check if agent can handle this request type."""
        request_type = request.metadata.get("type")
        return request_type in self.supported_types
```

### Observer Pattern

Notify multiple agents of state changes:

```python
class ObservableAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observers = []
    
    def attach_observer(self, observer_name: str):
        """Add observer to notification list."""
        if observer_name not in self.observers:
            self.observers.append(observer_name)
    
    def detach_observer(self, observer_name: str):
        """Remove observer from notification list."""
        if observer_name in self.observers:
            self.observers.remove(observer_name)
    
    async def notify_observers(self, event: str, data: Any):
        """Notify all observers of state change."""
        notifications = []
        
        for observer in self.observers:
            notification = self.invoke_agent(
                observer,
                f"Event: {event}",
                metadata={"event_data": data}
            )
            notifications.append(notification)
        
        # Fire and forget - don't wait for responses
        asyncio.gather(*notifications, return_exceptions=True)
```

## Error Handling in Communication

### Retry with Circuit Breaker

Implement circuit breaker pattern:

```python
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
    
    async def call(self, agent: Agent, target: str, task: str) -> Message:
        """Execute call with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await agent.invoke_agent(target, task)
            
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
class DeadLetterQueue:
    def __init__(self):
        self.failed_messages = []
    
    async def process_with_dlq(
        self,
        agent: Agent,
        message: Message,
        max_retries: int = 3
    ) -> Optional[Message]:
        """Process message with dead letter queue for failures."""
        for attempt in range(max_retries):
            try:
                result = await agent.process(message)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    # Add to dead letter queue
                    self.failed_messages.append({
                        "message": message,
                        "error": str(e),
                        "timestamp": datetime.now(),
                        "attempts": max_retries
                    })
                    return None
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
```

## Performance Optimization

### Message Batching

Batch multiple messages for efficiency:

```python
class BatchingAgent(Agent):
    def __init__(self, batch_size: int = 10, batch_timeout: float = 1.0, **kwargs):
        super().__init__(**kwargs)
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
        
        result = await self.model.run([
            Message(role="user", content=f"Process these requests:\n{combined_content}")
        ])
        
        # Parse and distribute results
        # ... implementation
```

### Connection Pooling

Manage agent connections efficiently:

```python
class AgentConnectionPool:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connections = {}
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
        task: str
    ) -> Message:
        """Invoke agent using connection pool."""
        target = await self.get_connection(to_agent)
        return await from_agent.invoke_agent(to_agent, task)
```

## Best Practices

1. **Message Design**
   - Keep messages self-contained
   - Include necessary context in metadata
   - Use structured formats (JSON) for complex data

2. **Error Handling**
   - Always handle communication failures
   - Implement timeouts for all remote calls
   - Use circuit breakers for unreliable connections

3. **Performance**
   - Batch messages when possible
   - Use async operations throughout
   - Monitor communication latency

4. **Security**
   - Validate message sources
   - Sanitize message content
   - Implement access controls

5. **Monitoring**
   - Log all inter-agent communications
   - Track message flow and latency
   - Monitor failed communications

## Next Steps

- Explore [Topologies](topologies.md) - Organizational patterns
- Learn about [Custom Agents](custom-agents.md) - Build specialized agents
- See [Memory Patterns](memory-patterns.md) - Advanced memory strategies
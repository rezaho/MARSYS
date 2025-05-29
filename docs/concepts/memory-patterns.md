# Memory Patterns

Advanced memory management strategies for complex agent systems.

## Overview

While basic memory management handles conversation history, advanced patterns enable:
- Shared memory between agents
- Long-term knowledge storage
- Selective memory retrieval
- Memory optimization
- Specialized memory types

## Memory Types

### 1. Episodic Memory

Store and retrieve specific experiences:

```python
from datetime import datetime
from typing import List, Optional
import json

class EpisodicMemory:
    """Memory system for storing discrete episodes."""
    
    def __init__(self, max_episodes: int = 100):
        self.episodes: List[Dict] = []
        self.max_episodes = max_episodes
    
    def start_episode(self, context: str) -> str:
        """Start a new episode."""
        episode_id = f"ep_{datetime.now().timestamp()}"
        episode = {
            "id": episode_id,
            "context": context,
            "start_time": datetime.now(),
            "messages": [],
            "outcome": None
        }
        self.episodes.append(episode)
        return episode_id
    
    def add_to_episode(self, episode_id: str, message: Message):
        """Add message to current episode."""
        for episode in self.episodes:
            if episode["id"] == episode_id:
                episode["messages"].append(message.to_dict())
                break
    
    def end_episode(self, episode_id: str, outcome: str):
        """Mark episode as complete."""
        for episode in self.episodes:
            if episode["id"] == episode_id:
                episode["end_time"] = datetime.now()
                episode["outcome"] = outcome
                break
        
        # Maintain size limit
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
    
    def find_similar_episodes(self, context: str, limit: int = 5) -> List[Dict]:
        """Find episodes with similar context."""
        # Simple keyword matching - could use embeddings for better similarity
        context_words = set(context.lower().split())
        
        scored_episodes = []
        for episode in self.episodes:
            episode_words = set(episode["context"].lower().split())
            similarity = len(context_words & episode_words) / len(context_words | episode_words)
            scored_episodes.append((similarity, episode))
        
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored_episodes[:limit]]
```

### 2. Semantic Memory

Store facts and knowledge:

```python
from collections import defaultdict
import numpy as np

class SemanticMemory:
    """Memory for storing facts and relationships."""
    
    def __init__(self):
        self.facts: Dict[str, List[Dict]] = defaultdict(list)
        self.relationships: Dict[str, Dict[str, str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def add_fact(self, entity: str, attribute: str, value: Any, confidence: float = 1.0):
        """Store a fact about an entity."""
        fact = {
            "attribute": attribute,
            "value": value,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "source": None
        }
        self.facts[entity].append(fact)
    
    def add_relationship(self, entity1: str, relation: str, entity2: str):
        """Store relationship between entities."""
        if entity1 not in self.relationships:
            self.relationships[entity1] = {}
        self.relationships[entity1][relation] = entity2
    
    def query_facts(self, entity: str, attribute: Optional[str] = None) -> List[Dict]:
        """Query facts about an entity."""
        facts = self.facts.get(entity, [])
        
        if attribute:
            facts = [f for f in facts if f["attribute"] == attribute]
        
        # Sort by confidence and recency
        facts.sort(key=lambda f: (f["confidence"], f["timestamp"]), reverse=True)
        return facts
    
    def get_related_entities(self, entity: str, relation_type: Optional[str] = None) -> List[str]:
        """Get entities related to the given entity."""
        relations = self.relationships.get(entity, {})
        
        if relation_type:
            return [relations.get(relation_type)] if relation_type in relations else []
        
        return list(relations.values())
```

### 3. Working Memory

Short-term, task-focused memory:

```python
class WorkingMemory:
    """Limited capacity working memory for current task."""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[Dict] = []
        self.focus_stack: List[str] = []
    
    def add(self, item: Any, priority: int = 1):
        """Add item to working memory."""
        entry = {
            "content": item,
            "priority": priority,
            "access_count": 0,
            "last_access": datetime.now()
        }
        
        self.items.append(entry)
        
        # Remove least important items if over capacity
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda x: (x["priority"], x["access_count"]), reverse=True)
            self.items = self.items[:self.capacity]
    
    def get(self, index: int) -> Optional[Any]:
        """Retrieve item from working memory."""
        if 0 <= index < len(self.items):
            self.items[index]["access_count"] += 1
            self.items[index]["last_access"] = datetime.now()
            return self.items[index]["content"]
        return None
    
    def push_focus(self, topic: str):
        """Push new focus topic."""
        self.focus_stack.append(topic)
    
    def pop_focus(self) -> Optional[str]:
        """Pop focus topic."""
        return self.focus_stack.pop() if self.focus_stack else None
    
    def clear(self):
        """Clear working memory."""
        self.items.clear()
        self.focus_stack.clear()
```

## Shared Memory Patterns

### 1. Blackboard Pattern

Shared workspace for multiple agents:

```python
import asyncio
from typing import Dict, Any, Set

class Blackboard:
    """Shared memory space for agent collaboration."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.history: List[Dict] = []
    
    async def write(self, key: str, value: Any, agent_name: str):
        """Write data to blackboard."""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            old_value = self.data.get(key)
            self.data[key] = value
            
            # Record history
            self.history.append({
                "timestamp": datetime.now(),
                "agent": agent_name,
                "action": "write",
                "key": key,
                "old_value": old_value,
                "new_value": value
            })
            
            # Notify subscribers
            await self._notify_subscribers(key, value, agent_name)
    
    async def read(self, key: str, agent_name: str) -> Optional[Any]:
        """Read data from blackboard."""
        value = self.data.get(key)
        
        # Record access
        self.history.append({
            "timestamp": datetime.now(),
            "agent": agent_name,
            "action": "read",
            "key": key,
            "value": value
        })
        
        return value
    
    def subscribe(self, key: str, agent_name: str):
        """Subscribe to changes for a key."""
        self.subscribers[key].add(agent_name)
    
    async def _notify_subscribers(self, key: str, value: Any, writer: str):
        """Notify subscribers of changes."""
        for subscriber in self.subscribers.get(key, set()):
            if subscriber != writer:
                # In real implementation, would trigger agent notification
                pass
```

### 2. Memory Pool Pattern

Shared memory pool with allocation:

```python
class MemoryPool:
    """Managed memory pool for agent system."""
    
    def __init__(self, total_capacity: int = 10000):
        self.total_capacity = total_capacity
        self.allocated: Dict[str, int] = {}
        self.memories: Dict[str, MemoryManager] = {}
        self.available = total_capacity
    
    def allocate(self, agent_name: str, requested_size: int) -> Optional[MemoryManager]:
        """Allocate memory to an agent."""
        if requested_size > self.available:
            # Try to free up memory
            self._garbage_collect()
            
            if requested_size > self.available:
                return None
        
        # Create memory manager with size limit
        memory = BoundedMemoryManager(max_size=requested_size)
        
        self.memories[agent_name] = memory
        self.allocated[agent_name] = requested_size
        self.available -= requested_size
        
        return memory
    
    def release(self, agent_name: str):
        """Release allocated memory."""
        if agent_name in self.allocated:
            self.available += self.allocated[agent_name]
            del self.allocated[agent_name]
            del self.memories[agent_name]
    
    def _garbage_collect(self):
        """Free memory from inactive agents."""
        # Implementation would check for inactive agents
        pass

class BoundedMemoryManager(MemoryManager):
    """Memory manager with size limits."""
    
    def __init__(self, max_size: int, **kwargs):
        super().__init__(**kwargs)
        self.max_size = max_size
        self.current_size = 0
    
    def update_memory(self, message: Message) -> None:
        """Update memory with size checking."""
        message_size = len(json.dumps(message.to_dict()))
        
        # Make room if needed
        while self.current_size + message_size > self.max_size and self.messages:
            removed = self.messages.pop(0)
            self.current_size -= len(json.dumps(removed.to_dict()))
        
        super().update_memory(message)
        self.current_size += message_size
```

## Memory Optimization Patterns

### 1. Memory Compression

Compress older memories:

```python
import zlib
from typing import List

class CompressedMemory:
    """Memory system with compression for old messages."""
    
    def __init__(self, compression_threshold: int = 100):
        self.recent: List[Message] = []
        self.compressed: List[bytes] = []
        self.compression_threshold = compression_threshold
    
    def add_message(self, message: Message):
        """Add message to memory."""
        self.recent.append(message)
        
        # Compress old messages
        if len(self.recent) > self.compression_threshold:
            to_compress = self.recent[:50]
            self.recent = self.recent[50:]
            
            # Compress batch
            batch_data = json.dumps([m.to_dict() for m in to_compress])
            compressed_data = zlib.compress(batch_data.encode())
            self.compressed.append(compressed_data)
    
    def get_all_messages(self) -> List[Message]:
        """Retrieve all messages, decompressing as needed."""
        all_messages = []
        
        # Decompress old messages
        for compressed_batch in self.compressed:
            batch_data = zlib.decompress(compressed_batch).decode()
            batch_dicts = json.loads(batch_data)
            all_messages.extend([Message.from_dict(d) for d in batch_dicts])
        
        # Add recent messages
        all_messages.extend(self.recent)
        
        return all_messages
```

### 2. Memory Summarization

Summarize old conversations:

```python
class SummarizedMemory:
    """Memory that summarizes old content."""
    
    def __init__(self, summary_threshold: int = 50):
        self.current_messages: List[Message] = []
        self.summaries: List[Dict] = []
        self.summary_threshold = summary_threshold
    
    async def add_message(self, message: Message, summarizer_agent: Agent):
        """Add message and summarize if needed."""
        self.current_messages.append(message)
        
        if len(self.current_messages) >= self.summary_threshold:
            # Create summary
            summary = await self._create_summary(
                self.current_messages,
                summarizer_agent
            )
            
            self.summaries.append({
                "summary": summary,
                "message_count": len(self.current_messages),
                "timestamp": datetime.now()
            })
            
            # Keep only recent messages
            self.current_messages = self.current_messages[-10:]
    
    async def _create_summary(self, messages: List[Message], agent: Agent) -> str:
        """Create summary of messages."""
        conversation = "\n".join([
            f"{m.role}: {m.content[:100]}..." 
            for m in messages
        ])
        
        response = await agent.auto_run(
            task=f"Summarize this conversation:\n{conversation}",
            max_steps=1
        )
        
        return response.content
```

## Retrieval Patterns

### 1. Similarity-Based Retrieval

Retrieve relevant memories:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMemory:
    """Memory with similarity-based retrieval."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.vectors = None
    
    def add_message(self, message: Message):
        """Add message and update vectors."""
        self.messages.append(message)
        self._update_vectors()
    
    def _update_vectors(self):
        """Update TF-IDF vectors."""
        if len(self.messages) > 0:
            contents = [m.content for m in self.messages]
            self.vectors = self.vectorizer.fit_transform(contents)
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Message]:
        """Find messages similar to query."""
        if not self.messages or self.vectors is None:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [self.messages[i] for i in top_indices]
```

### 2. Time-Based Retrieval

Retrieve memories by time:

```python
class TemporalMemory:
    """Memory with time-based retrieval."""
    
    def __init__(self):
        self.timeline: Dict[str, List[Message]] = defaultdict(list)
    
    def add_message(self, message: Message):
        """Add message to timeline."""
        date_key = message.timestamp.strftime("%Y-%m-%d")
        self.timeline[date_key].append(message)
    
    def get_date_range(self, start_date: datetime, end_date: datetime) -> List[Message]:
        """Get messages within date range."""
        messages = []
        
        current = start_date
        while current <= end_date:
            date_key = current.strftime("%Y-%m-%d")
            messages.extend(self.timeline.get(date_key, []))
            current += timedelta(days=1)
        
        return sorted(messages, key=lambda m: m.timestamp)
    
    def get_context_window(self, timestamp: datetime, window_minutes: int = 30) -> List[Message]:
        """Get messages around a specific time."""
        start = timestamp - timedelta(minutes=window_minutes)
        end = timestamp + timedelta(minutes=window_minutes)
        
        return self.get_date_range(start, end)
```

## Best Practices

1. **Memory Lifecycle Management**
   - Clear old memories periodically
   - Implement memory limits
   - Use compression for long-term storage

2. **Efficient Retrieval**
   - Index memories for fast search
   - Cache frequently accessed memories
   - Use appropriate data structures

3. **Memory Sharing**
   - Use locks for concurrent access
   - Implement read/write permissions
   - Monitor memory usage

4. **Performance Optimization**
   - Batch memory operations
   - Use async for I/O operations
   - Profile memory usage

## Advanced Integration

### Memory-Augmented Agent

```python
class MemoryAugmentedAgent(Agent):
    """Agent with advanced memory capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.working_memory = WorkingMemory()
    
    async def remember_fact(self, entity: str, attribute: str, value: Any):
        """Store fact in semantic memory."""
        self.semantic_memory.add_fact(entity, attribute, value)
    
    async def recall_facts(self, entity: str) -> str:
        """Recall facts about entity."""
        facts = self.semantic_memory.query_facts(entity)
        return json.dumps(facts, default=str)
    
    async def start_task(self, task: str):
        """Start new task with episodic memory."""
        episode_id = self.episodic_memory.start_episode(task)
        self.working_memory.push_focus(task)
        return episode_id
```

## Next Steps

- Explore [Browser Automation](browser-automation.md) - Web-based memory
- Learn about [Learning Agents](learning-agents.md) - Memory and learning
- See [Examples](../use-cases/examples/advanced-examples.md#memory-patterns) - Real implementations

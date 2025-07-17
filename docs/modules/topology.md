# Topology Module Documentation

## Overview

The Topology module provides a flexible three-way system for defining multi-agent relationships in the MARS coordination framework. The system supports three input formats (string notation, object-based, and pattern configuration) that are all converted to a canonical internal representation for consistency and type safety.

## Three-Way Topology System

### 1. String Notation (Simplest)
Define topologies using simple dictionaries with string values:

```python
topology = {
    "nodes": ["User", "Agent1", "Agent2"],
    "edges": ["User -> Agent1", "Agent1 <-> Agent2"],
    "rules": ["parallel(Agent1, Agent2)", "timeout(300)"],
    "metadata": {"description": "Simple workflow"}
}
```

### 2. Object-Based (Type-Safe)
Mix typed objects with strings for more control:

```python
from src.coordination.topology import Edge, Node
from src.coordination.rules import TimeoutRule

topology = {
    "nodes": [agent1_instance, agent2_instance],  # Actual agent objects
    "edges": [
        Edge(source="User", target="Agent1", edge_type=EdgeType.INVOKE),
        ("Agent1", "Agent2")  # Tuple notation
    ],
    "rules": [TimeoutRule(max_duration=300)],
    "metadata": {"pattern": "hub_and_spoke"}
}
```

### 3. Pattern Configuration
Use pre-defined patterns for common architectures:

```python
from src.coordination.topology.patterns import PatternConfig

# Hub-and-spoke pattern
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3"],
    parallel_spokes=True
)

# Hierarchical pattern
topology = PatternConfig.hierarchical(
    tree={
        "Manager": ["Lead1", "Lead2"],
        "Lead1": ["Worker1", "Worker2"],
        "Lead2": ["Worker3", "Worker4"]
    }
)
```

## Core Components

### Topology (Canonical Representation)
The internal representation that all formats convert to:

```python
@dataclass
class Topology:
    """The canonical topology representation."""
    nodes: List[Node]        # Typed Node objects
    edges: List[Edge]        # Typed Edge objects  
    rules: List[Rule]        # Typed Rule objects
    metadata: Dict[str, Any]
    
    # Methods for flexible mutation
    def add_node(self, node: Union[str, Node, Any]) -> Node
    def add_edge(self, edge: Union[str, Edge, tuple]) -> Edge
    def add_rule(self, rule: Union[str, Rule]) -> Rule
```

### Node
Represents an agent or system component:

```python
@dataclass
class Node:
    name: str
    node_type: NodeType = NodeType.AGENT
    agent_ref: Optional[Any] = None  # Reference to actual agent
    metadata: Dict[str, Any] = field(default_factory=dict)

# Node types
class NodeType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
```

### Edge
Represents a connection between nodes:

```python
@dataclass
class Edge:
    source: str  # Node name
    target: str  # Node name
    edge_type: EdgeType = EdgeType.INVOKE
    bidirectional: bool = False
    pattern: Optional[EdgePattern] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Edge types
class EdgeType(Enum):
    INVOKE = "invoke"    # Standard agent invocation
    NOTIFY = "notify"    # Notification without response
    QUERY = "query"      # Query with expected response
    STREAM = "stream"    # Streaming connection

# Special patterns
class EdgePattern(Enum):
    REFLEXIVE = "reflexive"      # A <=> B (boomerang)
    ALTERNATING = "alternating"  # A <~> B (ping-pong)
    SYMMETRIC = "symmetric"      # A <|> B (peer)
```

### TopologyAnalyzer
Parses topology definitions and builds executable graphs:

```python
analyzer = TopologyAnalyzer()
# Accepts dict, Topology, or PatternConfig
graph = analyzer.analyze(topology)
```

### TopologyGraph
Runtime graph structure for efficient analysis:

```python
graph = TopologyGraph(nodes, edges)
# Automatic detection of patterns
divergence_points = graph.divergence_points
convergence_points = graph.convergence_points
```

## Edge Notation Syntax

### Basic Edges
```python
edges = [
    # Unidirectional edge
    "User -> PlannerAgent",
    
    # Bidirectional edge (creates two edges internally)
    "Agent1 <-> Agent2",
    
    # Special patterns
    "Agent1 <=> Agent2",  # Reflexive (boomerang)
    "Agent1 <~> Agent2",  # Alternating (ping-pong)
    "Agent1 <|> Agent2",  # Symmetric (peer)
    
    # Tuple notation
    ("PlannerAgent", "ExecutorAgent"),
    
    # Object notation
    Edge(source="Reviewer", target="FinalAgent")
]
```

### Edge Types
Edges can have specific types for different communication patterns:

```python
# In object notation
Edge(source="A", target="B", edge_type=EdgeType.QUERY)
Edge(source="A", target="B", edge_type=EdgeType.STREAM)

# String notation uses INVOKE by default
"A -> B"  # EdgeType.INVOKE
```

## Rule Syntax

### String Rules
```python
rules = [
    # Parallel execution
    "parallel(Agent1, Agent2, Agent3)",
    
    # Conversation limits
    "max_turns(Agent1 <-> Agent2, 5)",
    
    # Concurrency limits
    "max_agents(3)",
    
    # Timeouts
    "timeout(Agent1, 30s)",
    "timeout(300)",  # Global timeout in seconds
    
    # Synchronization
    "wait_all(Agent1, Agent2) -> Agent3",
    
    # Resource limits
    "max_memory(Agent1, 1GB)",
    "rate_limit(Agent2, 10/min)"
]
```

### Object Rules
```python
from src.coordination.rules import (
    TimeoutRule, ParallelRule, MaxTurnsRule,
    MaxAgentsRule, SynchronizationRule
)

rules = [
    TimeoutRule(max_duration=300),
    ParallelRule(agents=["Agent1", "Agent2"]),
    MaxTurnsRule(edge=("A", "B"), max_turns=5),
    MaxAgentsRule(max_concurrent=3)
]
```

## Pattern Detection

### Divergence Points
Agents with multiple outgoing edges:

```python
# Topology
edges = [
    "Coordinator -> Worker1",
    "Coordinator -> Worker2",
    "Coordinator -> Worker3"
]

# Detection
graph.divergence_points = {"Coordinator"}
graph.is_divergence_point("Coordinator")  # True
```

### Convergence Points
Agents with multiple incoming edges:

```python
# Topology
edges = [
    "Worker1 -> Aggregator",
    "Worker2 -> Aggregator",
    "Worker3 -> Aggregator"
]

# Detection
graph.convergence_points = {"Aggregator"}
graph.is_convergence_point("Aggregator")  # True
```

### Conversation Loops
Bidirectional communication patterns:

```python
# Topology
edges = ["Analyst <-> Reviewer"]

# Detection
graph.is_in_conversation_loop("Analyst", "Reviewer")  # True
```

## Usage Examples

### Simple Linear Workflow

```python
# Using string notation
topology = {
    "nodes": ["User", "Researcher", "Writer", "Editor"],
    "edges": [
        "User -> Researcher",
        "Researcher -> Writer",
        "Writer -> Editor",
        "Editor -> User"
    ]
}

# Convert and analyze
from src.coordination import Orchestra
result = await Orchestra.run(task="Write article", topology=topology)
```

### Parallel Execution Pattern

```python
# Using pattern configuration
from src.coordination.topology.patterns import PatternConfig

topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3"],
    parallel_spokes=True,
    bidirectional=False  # Spokes can't talk back
)

# Or using string notation
topology = {
    "nodes": ["User", "Coordinator", "Worker1", "Worker2", "Worker3", "Aggregator"],
    "edges": [
        "User -> Coordinator",
        "Coordinator -> Worker1",
        "Coordinator -> Worker2",
        "Coordinator -> Worker3",
        "Worker1 -> Aggregator",
        "Worker2 -> Aggregator",
        "Worker3 -> Aggregator"
    ],
    "rules": ["parallel(Worker1, Worker2, Worker3)"]
}
```

### Conversation Pattern

```python
# Using mixed notation
topology = {
    "nodes": ["User", "Assistant", expert_agent, reviewer_agent],
    "edges": [
        "User -> Assistant",
        "Assistant <-> Expert",      # Bidirectional
        Edge(source="Expert", target="Reviewer", bidirectional=True),
        "Assistant -> User"
    ],
    "rules": [
        MaxTurnsRule(edge=("Assistant", "Expert"), max_turns=3),
        "timeout(Expert, 60s)"
    ]
}
```

## TopologyGraph API

### Query Methods

```python
# Get adjacent agents
next_agents = graph.get_next_agents("PlannerAgent")
previous_agents = graph.get_previous_agents("ExecutorAgent")

# Check connections
has_connection = graph.has_edge("Agent1", "Agent2")
is_bidirectional = graph.is_bidirectional("Agent1", "Agent2")

# Pattern queries
is_divergence = graph.is_divergence_point("Coordinator")
is_convergence = graph.is_convergence_point("Aggregator")
in_conversation = graph.is_in_conversation_loop("Expert", "Reviewer")
```

### Analysis Methods

```python
# Find paths
path = graph.find_path("User", "FinalAgent")
all_paths = graph.find_all_paths("Start", "End")

# Detect cycles
has_cycles = graph.has_cycles()
cycles = graph.find_cycles()

# Component analysis
components = graph.get_connected_components()
is_connected = graph.is_fully_connected()
```

## Converter Classes

### StringNotationConverter
Converts pure string notation to Topology:

```python
from src.coordination.topology.converters import StringNotationConverter

notation = {
    "nodes": ["A", "B", "C"],
    "edges": ["A -> B", "B -> C"],
    "rules": ["parallel(B, C)"]
}

topology = StringNotationConverter.convert(notation)
```

### ObjectNotationConverter
Handles mixed object/string notation:

```python
from src.coordination.topology.converters import ObjectNotationConverter

notation = {
    "nodes": [agent1, "Agent2"],  # Mix of objects and strings
    "edges": [Edge(...), ("A", "B")],  # Mix of Edge objects and tuples
    "rules": [TimeoutRule(300), "parallel(A, B)"]  # Mix of rules
}

topology = ObjectNotationConverter.convert(notation, agent_registry)
```

### PatternConfigConverter
Converts pattern configurations to Topology:

```python
from src.coordination.topology.converters import PatternConfigConverter
from src.coordination.topology.patterns import PatternConfig

config = PatternConfig(
    pattern=PatternType.HUB_AND_SPOKE,
    params={"hub": "Coordinator", "spokes": ["W1", "W2", "W3"]}
)

topology = PatternConfigConverter.convert(config)
```

## Orchestra Integration

The Orchestra automatically detects which format you're using:

```python
from src.coordination import Orchestra

# String notation
await Orchestra.run(task, {"nodes": [...], "edges": [...], "rules": [...]})

# Pattern configuration
await Orchestra.run(task, PatternConfig.hub_and_spoke(...))

# Already converted Topology
await Orchestra.run(task, topology_object)
```

## Pattern Templates

### Hub-and-Spoke
```python
PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3"],
    parallel_spokes=True,
    bidirectional=True,  # Whether spokes can respond to hub
    spoke_to_spoke=False  # Whether spokes can communicate
)
```

### Hierarchical
```python
PatternConfig.hierarchical(
    tree={
        "CEO": ["CTO", "CFO"],
        "CTO": ["DevLead", "QALead"],
        "DevLead": ["Dev1", "Dev2"]
    },
    bidirectional=True  # Whether lower levels can report up
)
```

### Pipeline
```python
PatternConfig.pipeline(
    stages=["Input", "Process", "Transform", "Output"],
    parallel_stages={"Process": ["ProcessA", "ProcessB"]}
)
```

### Mesh
```python
PatternConfig.mesh(
    agents=["Agent1", "Agent2", "Agent3", "Agent4"],
    fully_connected=True,  # Every agent connects to every other
    connection_pattern="bidirectional"  # or "unidirectional"
)
```

## Best Practices

1. **Choose the Right Format**: 
   - Use string notation for simple topologies
   - Use object notation when you need type safety
   - Use patterns for standard architectures

2. **Node Naming**: 
   - Use clear, descriptive names
   - Avoid spaces (use PascalCase or snake_case)
   - Be consistent across your topology

3. **Edge Design**:
   - Use unidirectional edges for clear flow
   - Use bidirectional edges only for conversations
   - Avoid circular dependencies unless needed

4. **Rule Application**:
   - Apply timeouts to prevent hanging
   - Use parallel rules to improve performance
   - Set conversation limits to prevent infinite loops

5. **Validation**:
   - The system validates topology at creation time
   - Missing nodes referenced in edges generate warnings
   - Rules are validated when applied

## Performance Considerations

- **O(1) lookups**: Internal indices for fast node/edge access
- **Lazy computation**: Pattern detection only when accessed
- **Efficient storage**: Minimal memory footprint
- **Type safety**: Canonical representation prevents runtime errors

## Error Handling

### Common Issues

```python
# Missing nodes (generates warning, not error)
topology = {
    "nodes": ["A", "B"],
    "edges": ["A -> C"]  # Warning: C not in nodes
}

# Invalid edge syntax
try:
    topology = {
        "edges": ["A --> B"]  # Wrong arrow syntax
    }
except ValueError as e:
    print(f"Invalid edge: {e}")

# Pattern validation
try:
    PatternConfig.hub_and_spoke(
        # Missing required 'spokes' parameter
        hub="Coordinator"
    )
except ValueError as e:
    print(f"Invalid pattern: {e}")
```

## Testing Topologies

```python
# Validate topology before execution
def validate_topology(topology: Union[dict, Topology]) -> List[str]:
    errors = []
    
    # Convert to Topology if needed
    if isinstance(topology, dict):
        from src.coordination.topology.converters import StringNotationConverter
        topology = StringNotationConverter.convert(topology)
    
    # Analyze
    analyzer = TopologyAnalyzer()
    graph = analyzer.analyze(topology)
    
    # Check connectivity
    if not graph.is_fully_connected():
        errors.append("Topology has disconnected components")
    
    # Check for entry points
    if not graph.get_entry_points():
        errors.append("No entry points found")
    
    # Check for potential deadlocks
    if graph.has_circular_dependencies():
        errors.append("Circular dependencies detected")
    
    return errors
```

## Migration from TopologyDefinition

The old `TopologyDefinition` class has been completely removed. Update your code:

```python
# Old way (no longer works)
from src.coordination.topology import TopologyDefinition
topology = TopologyDefinition(nodes=[...], edges=[...], rules=[...])

# New way (use dict notation)
topology = {
    "nodes": [...],
    "edges": [...],
    "rules": [...]
}
```

## API Reference

### Topology class
```python
class Topology:
    def add_node(self, node: Union[str, Node, Any]) -> Node
    def remove_node(self, node_name: str) -> bool
    def update_node(self, node_name: str, **kwargs) -> bool
    def get_node(self, node_name: str) -> Optional[Node]
    
    def add_edge(self, edge: Union[str, Edge, tuple]) -> Edge
    def remove_edge(self, source: str, target: str) -> bool
    def get_edge(self, source: str, target: str) -> Optional[Edge]
    
    def add_rule(self, rule: Union[str, Rule]) -> Rule
    def remove_rule(self, rule: Rule) -> bool
    
    def clear(self) -> None
    def to_dict(self) -> dict
```

### TopologyAnalyzer.analyze()
```python
def analyze(self, topology_def: Union[Topology, Dict[str, Any]]) -> TopologyGraph:
    """Parse topology and build executable graph."""
```

### Pattern Configuration
```python
@classmethod
def hub_and_spoke(cls, hub: str, spokes: List[str], **kwargs) -> PatternConfig

@classmethod  
def hierarchical(cls, tree: Dict[str, List[str]], **kwargs) -> PatternConfig

@classmethod
def pipeline(cls, stages: List[str], **kwargs) -> PatternConfig

@classmethod
def mesh(cls, agents: List[str], **kwargs) -> PatternConfig
```

The Topology module provides a flexible and powerful three-way system for defining complex agent relationships, enabling the MARS framework to handle sophisticated multi-agent coordination patterns with ease while maintaining type safety internally.
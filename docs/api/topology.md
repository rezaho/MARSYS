# Topology API

Complete API reference for the topology system that defines agent communication patterns and workflow structures.

## 🎯 Overview

The Topology API provides flexible ways to define multi-agent workflows through graph structures, pre-defined patterns, and string notation.

## 📦 Core Classes

### Topology

Main topology container that holds nodes, edges, and rules.

**Import:**
```python
from marsys.coordination.topology import Topology
```

**Constructor:**
```python
Topology(
    nodes: List[Union[Node, str]] = None,
    edges: List[Union[Edge, str]] = None,
    rules: List[Union[Rule, str]] = None,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `nodes` | `List[Union[Node, str]]` | Graph nodes (agents) | `[]` |
| `edges` | `List[Union[Edge, str]]` | Connections between nodes | `[]` |
| `rules` | `List[Union[Rule, str]]` | Execution rules | `[]` |
| `metadata` | `Dict[str, Any]` | Additional metadata | `{}` |

**Methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `add_node(node)` | `Node` | Add a node to topology |
| `add_edge(edge)` | `Edge` | Add an edge to topology |
| `add_rule(rule)` | `Rule` | Add a rule to topology |
| `remove_node(name)` | `bool` | Remove node by name |
| `remove_edge(source, target)` | `bool` | Remove specific edge |
| `get_node(name)` | `Optional[Node]` | Get node by name |
| `validate()` | `bool` | Validate topology consistency |

**Example:**
```python
from marsys.coordination.topology import Topology, Node, Edge

topology = Topology(
    nodes=[
        Node("Coordinator", node_type=NodeType.AGENT),
        Node("Worker1", node_type=NodeType.AGENT),
        Node("Worker2", node_type=NodeType.AGENT)
    ],
    edges=[
        Edge("Coordinator", "Worker1"),
        Edge("Coordinator", "Worker2"),
        Edge("Worker1", "Coordinator"),
        Edge("Worker2", "Coordinator")
    ]
)
```

---

### Node

Represents an agent or system component in the topology.

**Import:**
```python
from marsys.coordination.topology import Node, NodeType
```

**Constructor:**
```python
Node(
    name: str,
    node_type: NodeType = NodeType.AGENT,
    agent_ref: Optional[Any] = None,
    is_convergence_point: bool = False,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | `str` | Unique node identifier | Required |
| `node_type` | `NodeType` | Type of node | `AGENT` |
| `agent_ref` | `Optional[Any]` | Reference to agent instance | `None` |
| `is_convergence_point` | `bool` | Marks convergence point | `False` |
| `metadata` | `Dict[str, Any]` | Additional node data | `{}` |

**NodeType Enum:**
```python
class NodeType(Enum):
    USER = "user"        # User interaction node
    AGENT = "agent"      # AI agent node
    SYSTEM = "system"    # System component
    TOOL = "tool"        # Tool node
```

**Example:**
```python
from marsys.coordination.topology import Node, NodeType

# Create different node types
user_node = Node("User", node_type=NodeType.USER)
agent_node = Node("Assistant", node_type=NodeType.AGENT)
convergence = Node(
    "Aggregator",
    node_type=NodeType.AGENT,
    is_convergence_point=True
)
```

---

### Edge

Defines connections and communication paths between nodes.

**Import:**
```python
from marsys.coordination.topology import Edge, EdgeType, EdgePattern
```

**Constructor:**
```python
Edge(
    source: str,
    target: str,
    edge_type: EdgeType = EdgeType.INVOKE,
    bidirectional: bool = False,
    pattern: Optional[EdgePattern] = None,
    metadata: Dict[str, Any] = None
)
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `source` | `str` | Source node name | Required |
| `target` | `str` | Target node name | Required |
| `edge_type` | `EdgeType` | Type of connection | `INVOKE` |
| `bidirectional` | `bool` | Two-way communication | `False` |
| `pattern` | `Optional[EdgePattern]` | Communication pattern | `None` |
| `metadata` | `Dict[str, Any]` | Additional edge data | `{}` |

**EdgeType Enum:**
```python
class EdgeType(Enum):
    INVOKE = "invoke"      # Agent invocation
    NOTIFY = "notify"      # Notification only
    QUERY = "query"        # Query/response
    STREAM = "stream"      # Streaming data
```

**EdgePattern Enum:**
```python
class EdgePattern(Enum):
    ALTERNATING = "alternating"  # Take turns
    SYMMETRIC = "symmetric"      # Equal access
```

**Example:**
```python
# One-way edge
edge1 = Edge("Manager", "Worker")

# Bidirectional conversation
edge2 = Edge(
    "Agent1",
    "Agent2",
    bidirectional=True,
    pattern=EdgePattern.ALTERNATING
)

# Notification edge
edge3 = Edge(
    "Monitor",
    "Logger",
    edge_type=EdgeType.NOTIFY
)
```

---

### PatternConfig

Pre-defined topology patterns for common use cases.

**Import:**
```python
from marsys.coordination.topology.patterns import PatternConfig
```

**Static Methods:**

#### hub_and_spoke
```python
@staticmethod
def hub_and_spoke(
    hub: str,
    spokes: List[str],
    parallel_spokes: bool = False,
    bidirectional: bool = True
) -> Topology
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `hub` | `str` | Central coordinator name | Required |
| `spokes` | `List[str]` | Spoke agent names | Required |
| `parallel_spokes` | `bool` | Execute spokes in parallel | `False` |
| `bidirectional` | `bool` | Two-way communication | `True` |

#### pipeline
```python
@staticmethod
def pipeline(
    stages: List[Dict[str, Any]],
    parallel_within_stage: bool = False
) -> Topology
```

**Parameters:**
| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `stages` | `List[Dict]` | Pipeline stages | Required |
| `parallel_within_stage` | `bool` | Parallel execution in stages | `False` |

#### mesh
```python
@staticmethod
def mesh(
    agents: List[str],
    fully_connected: bool = True
) -> Topology
```

#### hierarchical
```python
@staticmethod
def hierarchical(
    tree: Dict[str, List[str]]
) -> Topology
```

**Example:**
```python
from marsys.coordination.topology.patterns import PatternConfig

# Hub and spoke pattern
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2", "Worker3"],
    parallel_spokes=True
)

# Pipeline pattern
topology = PatternConfig.pipeline(
    stages=[
        {"name": "extract", "agents": ["Extractor"]},
        {"name": "transform", "agents": ["Transformer1", "Transformer2"]},
        {"name": "load", "agents": ["Loader"]}
    ],
    parallel_within_stage=True
)

# Hierarchical pattern
topology = PatternConfig.hierarchical(
    tree={
        "CEO": ["VP1", "VP2"],
        "VP1": ["Manager1", "Manager2"],
        "VP2": ["Manager3"],
        "Manager1": ["Worker1", "Worker2"]
    }
)
```

---

## 🔄 Topology Converters

### String Notation

Convert string-based topology definitions.

**Format:**
```python
topology = {
    "nodes": ["Agent1", "Agent2", "Agent3"],
    "edges": [
        "Agent1 -> Agent2",      # One-way
        "Agent2 <-> Agent3",      # Bidirectional
        "Agent1 => Agent3"        # Strong connection
    ],
    "rules": [
        "timeout(300)",
        "max_agents(10)"
    ]
}
```

### Object Notation

Convert object-based topology definitions.

**Format:**
```python
topology = {
    "nodes": [
        {"name": "Agent1", "type": "agent"},
        {"name": "User", "type": "user"}
    ],
    "edges": [
        {"source": "User", "target": "Agent1"},
        {"source": "Agent1", "target": "User"}
    ],
    "rules": [
        {"type": "timeout", "value": 300}
    ]
}
```

---

## 📊 TopologyGraph

Internal graph representation for routing decisions.

**Import:**
```python
from marsys.coordination.topology.graph import TopologyGraph
```

**Key Methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `get_entry_points()` | `List[str]` | Find nodes with no incoming edges |
| `get_allowed_agents(node)` | `List[str]` | Get allowed targets from node |
| `is_conversation_edge(s, t)` | `bool` | Check if edge is bidirectional |
| `is_convergence_point(node)` | `bool` | Check if node is convergence point |
| `get_convergence_points()` | `List[str]` | Get all convergence points |

---

## 📐 TopologyAnalyzer

Analyzes and validates topology structures.

**Import:**
```python
from marsys.coordination.topology.analyzer import TopologyAnalyzer
```

**Constructor:**
```python
analyzer = TopologyAnalyzer(topology: Topology)
```

**Methods:**
| Method | Returns | Description |
|--------|---------|-------------|
| `create_graph()` | `TopologyGraph` | Create graph representation |
| `validate()` | `Tuple[bool, List[str]]` | Validate topology |
| `find_cycles()` | `List[List[str]]` | Detect cycles |
| `get_execution_order()` | `List[str]` | Topological sort |

**Example:**
```python
analyzer = TopologyAnalyzer(topology)
graph = analyzer.create_graph()

# Validate topology
is_valid, errors = analyzer.validate()
if not is_valid:
    print(f"Topology errors: {errors}")

# Get execution order
order = analyzer.get_execution_order()
```

---

## 📋 Best Practices

### ✅ DO:
- Keep topologies simple and focused
- Use pre-defined patterns when possible
- Validate topology before execution
- Set convergence points for parallel branches
- Add timeout rules for safety

### ❌ DON'T:
- Create cyclic dependencies without limits
- Mix too many patterns
- Forget error recovery paths
- Skip validation

---

## 🎨 Common Patterns

### Sequential Chain
```python
topology = {
    "nodes": ["A", "B", "C"],
    "edges": ["A -> B", "B -> C"]
}
```

### Parallel Execution
```python
topology = PatternConfig.hub_and_spoke(
    hub="Coordinator",
    spokes=["Worker1", "Worker2"],
    parallel_spokes=True
)
```

### Conversation Loop
```python
topology = {
    "nodes": ["Agent1", "Agent2"],
    "edges": ["Agent1 <-> Agent2"],
    "rules": ["max_turns(5)"]
}
```

---

## 🚦 Related Documentation

- [Topology Concepts](../concepts/advanced/topology.md) - Conceptual overview
- [Orchestra API](orchestra.md) - Using topologies with Orchestra
- [Execution API](execution.md) - How topologies are executed
- [Rules API](rules.md) - Topology rules reference

---

!!! tip "Pro Tip"
    Start with pre-defined patterns from `PatternConfig` and customize as needed. This ensures proper structure and reduces errors.
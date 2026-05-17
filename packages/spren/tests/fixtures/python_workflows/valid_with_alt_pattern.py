"""Pipeline with an alternating-edge pattern.

The framework supports `EdgePattern.ALTERNATING` for back-and-forth turns
between two agents (the `<~>` operator in dict-DSL form). Spren's canvas
in v0.3 does NOT render this pattern — Q2 lock — so the importer auto-
converts these edges to plain bidirectional with a non-blocking warning
surfaced via the import response's `warnings[]` array.
"""
from marsys.agents import Agent
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.topology.core import (
    Edge,
    EdgePattern,
    EdgeType,
    Node,
    NodeKind,
    Topology,
)
from marsys.models import ModelConfig


def stub(query: str) -> str:
    return query


a = Agent(
    name="A",
    goal="ask probing questions",
    instruction="alternate with B",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"stub": stub},
    memory_retention="session",
    allowed_peers=["B"],
)

b = Agent(
    name="B",
    goal="answer probing questions",
    instruction="alternate with A",
    model_config=ModelConfig(type="api", name="claude-opus-4-7", provider="anthropic"),
    tools={"stub": stub},
    memory_retention="session",
    allowed_peers=["A"],
)

topology = Topology(
    nodes=[
        Node(name="A", kind=NodeKind.AGENT, agent_ref="A"),
        Node(name="B", kind=NodeKind.AGENT, agent_ref="B"),
    ],
    edges=[
        Edge(
            source="A",
            target="B",
            edge_type=EdgeType.INVOKE,
            bidirectional=True,
            pattern=EdgePattern.ALTERNATING,
        ),
    ],
)

exec_config = ExecutionConfig(
    convergence_timeout=120.0,
    response_format="json",
    user_interaction="none",
)

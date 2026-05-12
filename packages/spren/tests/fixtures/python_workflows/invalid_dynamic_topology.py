"""Rejected — builds edges via list comprehension."""
from marsys.agents import Agent
from marsys.coordination.topology.core import Edge, EdgeType, Node, NodeType, Topology
from marsys.models import ModelConfig


def stub() -> None:
    return None


names = ["A", "B", "C"]

agent_a = Agent(
    name="A",
    goal="g",
    instruction="i",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"stub": stub},
)

topology = Topology(
    nodes=[Node(name=n, node_type=NodeType.AGENT) for n in names],
    edges=[Edge(source=a, target=b, edge_type=EdgeType.INVOKE) for a, b in zip(names, names[1:])],
)

"""Rejected — uses an f-string in a user-facing field."""
from marsys.agents import Agent
from marsys.coordination.topology.core import Node, NodeKind, Topology
from marsys.models import ModelConfig

USER_NAME = "Reza"


def stub() -> None:
    return None


agent = Agent(
    name=f"Researcher for {USER_NAME}",
    goal="g",
    instruction="i",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"stub": stub},
)
topology = Topology(
    nodes=[Node(name="Researcher", kind=NodeKind.AGENT, agent_ref="Researcher")],
    edges=[],
)

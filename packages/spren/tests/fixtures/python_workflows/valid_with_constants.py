"""Constants-Backed Pipeline.

Same shape as valid_minimal.py but funnels Agent/Node/ModelConfig kwargs
through module-level constants. Verifies the importer's two-pass walker
resolves ``Name`` references against the constants table.
"""
from marsys.agents import Agent
from marsys.coordination.topology.core import Edge, EdgeType, Node, NodeKind, Topology
from marsys.models import ModelConfig

AGENT_NAME = "Reza"
GOAL = "do research"
INSTRUCTION = "be helpful"
MODEL_NAME = "claude-opus-4-7"
PROVIDER = "anthropic"


def stub() -> None:
    return None


agent = Agent(
    name=AGENT_NAME,
    goal=GOAL,
    instruction=INSTRUCTION,
    model_config=ModelConfig(type="api", name=MODEL_NAME, provider=PROVIDER),
    tools={"stub": stub},
)

topology = Topology(
    nodes=[Node(name=AGENT_NAME, kind=NodeKind.AGENT, agent_ref=AGENT_NAME)],
    edges=[],
)

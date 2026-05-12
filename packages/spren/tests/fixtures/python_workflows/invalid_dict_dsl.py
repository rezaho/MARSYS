"""Rejected — uses the dict-of-arrows topology DSL.

The framework's own example files use this style; clean AST parsing of it is
out of scope for v0.3. Users rewrite to constructor style or use the visual
builder.
"""
from marsys.agents import Agent
from marsys.coordination.topology.core import Topology
from marsys.models import ModelConfig


def stub() -> None:
    return None


agent = Agent(
    name="Researcher",
    goal="research",
    instruction="research",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"stub": stub},
)

topology = Topology(nodes={"Start -> Researcher": "invoke"})

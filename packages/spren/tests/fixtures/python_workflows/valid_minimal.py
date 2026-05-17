"""Minimal Research Pipeline.

Exercises the full kwarg surface of Spren's AgentSpec / NodeSpec / EdgeSpec
mirrors. The two stub callables below exist solely so the parser sees real
``Dict[str, Callable]`` AST shapes — the parser keeps only the dict keys.
"""
from marsys.agents import Agent
from marsys.coordination.config import ExecutionConfig
from marsys.coordination.topology.core import Edge, EdgeType, Node, NodeKind, Topology
from marsys.models import ModelConfig


def search_web(query: str) -> str:
    return query


def write_doc(content: str) -> str:
    return content


researcher = Agent(
    name="Researcher",
    goal="research the user's question thoroughly",
    instruction="search the web, read sources, return a synthesized answer with citations",
    model_config=ModelConfig(type="api", name="gpt-4o", provider="openai"),
    tools={"search_web": search_web},
    memory_retention="session",
    allowed_peers=["Writer"],
)

writer = Agent(
    name="Writer",
    goal="turn the research into a polished document",
    instruction="rewrite for clarity and tone; preserve citations",
    model_config=ModelConfig(type="api", name="claude-opus-4-7", provider="anthropic"),
    tools={"write_doc": write_doc},
    memory_retention="session",
    allowed_peers=[],
)

topology = Topology(
    nodes=[
        Node(name="Researcher", kind=NodeKind.AGENT, agent_ref="Researcher", is_convergence_point=False),
        Node(name="Writer", kind=NodeKind.AGENT, agent_ref="Writer", is_convergence_point=True),
    ],
    edges=[
        Edge(source="Researcher", target="Writer", edge_type=EdgeType.INVOKE, bidirectional=False),
    ],
)

exec_config = ExecutionConfig(
    convergence_timeout=120.0,
    response_format="json",
    user_interaction="none",
)

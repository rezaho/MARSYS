from src.agents.agents import BaseAgent
from src.learning.rl import GRPOTrainer, GRPOConfig
from pydantic import BaseModel

class AgentConfig(BaseModel):
    agnent_prompt: str
    model_name: str
    learning_head: str
    learning_config: Optional[Dict] = None
    memory_type: Optional[str] = "conversation_history"
    max_tokens: Optional[int] = 512


class BaseCrew:
    def __init__(self, agents: List[AgentConfig], learning_config: GRPOConfig):
        self.agents = agents
        self.learning_config = learning_config
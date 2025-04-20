"""
This module defines the essential framework for constructing AI agents that leverage shared large-scale
language models while allowing for task-specific extensions. The design emphasizes modularity, where the
base language model (e.g., from the Llama or Qwen families) is managed externally and shared among multiple
agents, while each agent is responsible for its own customizations.

Key Components:

1. Shared Base Model:
    - The foundational language processing capabilities are provided by a base model that is instantiated
      separately and shared across different agents. This separation enables efficient resource usage and
      consistency in language understanding.

2. Agent-Specific Extensions:
    - Custom Prompt:
         * Each agent defines a unique prompt that guides its behavior and tailors its responses to specific
           contexts or tasks.
    - Specialized Learning Head:
         * Agents may incorporate task-specific learning modules (e.g., a LoRA head optimized via GRPO RL for
           re-writing user queries). Although the core logic of these learning heads is implemented in a
           separate module (such as models.py), agents are responsible for integrating and managing them.
    - Memory Module:
         * A dedicated memory component maintains the agent's internal state. This can include hidden state
           representations or a history of interactions, enabling context-aware processing across multiple turns.

3. Interaction Layers:
    - Environment and Tools:
         * Agents interface with external systems – whether digital platforms, physical devices, or
           simulation environments – through specialized communication layers.
    - Inter-Agent Communication:
         * Designed for networked interactions, this mechanism facilitates communication between multiple agents,
           organized following a topological structure that is defined in other modules.

4. Methods of Communication:
    - The module exposes APIs and methods for external invocation by users or other agents, enabling:
         * Task requests and query processing.
         * Interaction with diverse environments and collaboration among agents.

Overall Design:
    - This module’s structure separates the shared base model from agent-specific functionalities, ensuring that
      while agents leverage a common language foundation, each can exhibit unique behaviors through tailored
      prompts, learning heads, and memory management.
    - The clear delineation between shared and specific components encourages extendibility and supports the
      development of diverse agents capable of performing a wide range of tasks.

This file defines the necessary classes and methods to:
- Initialize and manage the base model.
- Interface with environmental tools.
- Facilitate inter-agent communications.
- Manage agent-specific prompting, learning adaptations, and memory.
- Provide a standardized communication channel for invoking agent actions.

Future implementations should extend this module to incorporate concrete classes and methods
to fully realize the operational capabilities of each agent based on the above specifications.
"""

import asyncio
import importlib
import json
import os
import random
import re
import threading
import uuid
import weakref
from datetime import datetime
from queue import Queue
from typing import Any, Dict, List, Optional
from weakref import WeakValueDictionary

from src.agents.memory import MessageMemory
from src.environment.web_browser import BrowserTool
from src.models.models import BaseAPIModel, BaseLLM, BaseVLM, PeftHead


class AgentRegistry:
    _agents = weakref.WeakValueDictionary()
    _lock = threading.Lock()
    _counter = 0

    @classmethod
    def register(cls, agent, name=None, prefix="BaseAgent"):
        with cls._lock:
            if name is None:
                cls._counter += 1
                name = f"{prefix}-{cls._counter}"
            else:
                if name in cls._agents:
                    raise ValueError(f"Agent name '{name}' already exists.")
            cls._agents[name] = agent
            return name

    @classmethod
    def unregister(cls, name):
        with cls._lock:
            cls._agents.pop(name, None)

    @classmethod
    def get(cls, name):
        return cls._agents.get(name)

    @classmethod
    def all(cls):
        return dict(cls._agents)


class BaseAgent:
    """
    Minimal base class for all agents.
    Handles agent registration, task queue, and basic properties.
    """

    def __init__(
        self,
        model: BaseVLM | BaseLLM,
        system_prompt: str,
        tools: Optional[Dict] = None,
        tools_schema: Optional[Dict] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
    ):
        # Check tools/tools_schema consistency
        if tools and not tools_schema:
            raise ValueError("The tools schema is required if the tools are provided.")
        if tools_schema and not tools:
            raise ValueError("The tools are required if the tools schema is provided.")
        self.tools = tools
        self.tools_schema = tools_schema
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.tasks = Queue()
        self.name = AgentRegistry.register(
            self, agent_name, prefix=self.__class__.__name__
        )

    def __del__(self):
        AgentRegistry.unregister(self.name)


class BaseLearnableAgent(BaseAgent):
    """
    Base class for learnable agents (handles learning head logic).
    """

    def __init__(
        self,
        model: BaseVLM | BaseLLM,
        system_prompt: str,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=kwargs.get("tools"),
            tools_schema=kwargs.get("tools_schema"),
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
        self._learning_head_name = learning_head
        self._learning_config = learning_head_config
        if learning_head == "peft":
            self.model = PeftHead(model=self.model)
            self.model.prepare_peft_model(**learning_head_config)


import time
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """
    Abstract base class for memory modules.
    Defines the interface for memory operations.
    """

    def __init__(self, memory_type: str):
        self.memory_type = memory_type

    @abstractmethod
    def update_memory(self, *args, **kwargs):
        raise NotImplementedError("update_memory must be implemented in subclasses.")

    @abstractmethod
    def replace_memory(self, *args, **kwargs):
        raise NotImplementedError("replace_memory must be implemented in subclasses.")

    @abstractmethod
    def delete_memory(self, *args, **kwargs):
        raise NotImplementedError("delete_memory must be implemented in subclasses.")

    @abstractmethod
    def retrieve_recent(self, n=1):
        raise NotImplementedError("retrieve_recent must be implemented in subclasses.")

    @abstractmethod
    def retrieve_all(self):
        raise NotImplementedError("retrieve_all must be implemented in subclasses.")

    @abstractmethod
    def reset_memory(self):
        raise NotImplementedError("reset_memory must be implemented in subclasses.")

    @abstractmethod
    def to_llm_format(self, *args, **kwargs):
        raise NotImplementedError("to_llm_format must be implemented in subclasses.")


class ConversationMemory(BaseMemory):
    """
    Memory module that stores conversation history.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__(memory_type="conversation_history")
        self.memory = []
        if system_prompt:
            self.memory.append({"role": "system", "content": system_prompt})

    def update_memory(self, role: str, content: str):
        self.memory.append({"role": role, "content": content})

    def replace_memory(self, idx: int, role: str, content: str):
        if 0 <= idx < len(self.memory):
            self.memory[idx] = {"role": role, "content": content}
        else:
            raise IndexError("Memory index out of range.")

    def delete_memory(self, idx: int):
        if 0 <= idx < len(self.memory):
            del self.memory[idx]
        else:
            raise IndexError("Memory index out of range.")

    def retrieve_recent(self, n=1):
        return self.memory[-n:] if n > 0 else []

    def retrieve_all(self):
        return list(self.memory)

    def retrieve_by_role(self, role: str, n=None):
        filtered = [m for m in self.memory if m["role"] == role]
        return filtered[-n:] if n else filtered

    def reset_memory(self):
        self.memory.clear()

    def to_llm_format(self):
        return list(self.memory)


class KGMemory(BaseMemory):
    """
    Memory module that stores knowledge as triplets in a knowledge graph.
    Requires an LLM for fact extraction from text.
    """

    def __init__(
        self,
        model: BaseVLM | BaseLLM | BaseAPIModel,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(memory_type="kg")
        self.model = model
        self.kg = []
        if system_prompt:
            self.kg.append(
                {
                    "role": "system",
                    "subject": "system",
                    "predicate": "init",
                    "object": system_prompt,
                    "timestamp": time.time(),
                }
            )

    def update_memory(self, role: str, subject: str, predicate: str, obj: str):
        timestamp = time.time()
        self.kg.append(
            {
                "role": role,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "timestamp": timestamp,
            }
        )

    def replace_memory(
        self, idx: int, role: str, subject: str, predicate: str, obj: str
    ):
        if 0 <= idx < len(self.kg):
            self.kg[idx] = {
                "role": role,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "timestamp": time.time(),
            }
        else:
            raise IndexError("KG index out of range.")

    def delete_memory(self, idx: int):
        if 0 <= idx < len(self.kg):
            del self.kg[idx]
        else:
            raise IndexError("KG index out of range.")

    def retrieve_recent(self, n=1):
        sorted_kg = sorted(self.kg, key=lambda x: x["timestamp"], reverse=True)
        return [self._kg_to_llm_format(fact) for fact in sorted_kg[:n]] if n > 0 else []

    def retrieve_all(self):
        return [self._kg_to_llm_format(fact) for fact in self.kg]

    def retrieve_by_role(self, role: str, n=None):
        filtered = [fact for fact in self.kg if fact["role"] == role]
        filtered = sorted(filtered, key=lambda x: x["timestamp"], reverse=True)
        if n:
            filtered = filtered[:n]
        return [self._kg_to_llm_format(fact) for fact in filtered]

    def reset_memory(self):
        self.kg.clear()

    def to_llm_format(self):
        return [self._kg_to_llm_format(fact) for fact in self.kg]

    def _kg_to_llm_format(self, fact):
        content = f"{fact['subject']} {fact['predicate']} {fact['object']} (added at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(fact['timestamp']))})"
        return {"role": fact["role"], "content": content}

    def extract_and_update_from_text(self, input_text: str, role: str = "user"):
        extraction_prompt = (
            "Extract all knowledge graph facts from the following text. "
            "Return a JSON list of triplets, where each triplet is a dict with keys: subject, predicate, object. "
            'Example: [{"subject": "Paris", "predicate": "is the capital of", "object": "France"}, ...]'
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": role, "content": input_text},
        ]
        try:
            result = self.model.run(messages=messages, json_mode=True)
        except Exception:
            result = self.model.run(messages=messages)
        if isinstance(result, str):
            try:
                facts = json.loads(result)
            except Exception:
                facts = []
        else:
            facts = result if isinstance(result, list) else []
        for fact in facts:
            if (
                isinstance(fact, dict)
                and "subject" in fact
                and "predicate" in fact
                and "object" in fact
            ):
                self.update_memory(
                    role, fact["subject"], fact["predicate"], fact["object"]
                )
        return facts


class MemoryManager:
    """
    Factory/manager for memory modules. Instantiates and delegates to the correct memory type.
    """

    def __init__(
        self,
        memory_type: str,
        system_prompt: Optional[str] = None,
        model: Optional[BaseVLM | BaseLLM | BaseAPIModel] = None,
    ):
        self.memory_type = memory_type
        if memory_type == "conversation_history":
            self.memory_module = ConversationMemory(system_prompt=system_prompt)
        elif memory_type == "kg":
            if model is None:
                raise ValueError(
                    "KGMemory requires a 'model' instance for fact extraction."
                )
            self.memory_module = KGMemory(model=model, system_prompt=system_prompt)
        else:
            raise ValueError(f"Unknown memory_type: {memory_type}")

    def update_memory(self, *args, **kwargs):
        return self.memory_module.update_memory(*args, **kwargs)

    def replace_memory(self, *args, **kwargs):
        return self.memory_module.replace_memory(*args, **kwargs)

    def delete_memory(self, *args, **kwargs):
        return self.memory_module.delete_memory(*args, **kwargs)

    def retrieve_recent(self, *args, **kwargs):
        return self.memory_module.retrieve_recent(*args, **kwargs)

    def retrieve_all(self):
        return self.memory_module.retrieve_all()

    def retrieve_by_role(self, *args, **kwargs):
        return self.memory_module.retrieve_by_role(*args, **kwargs)

    def reset_memory(self):
        return self.memory_module.reset_memory()

    def to_llm_format(self):
        return self.memory_module.to_llm_format()

    def extract_and_update_from_text(self, *args, **kwargs):
        if self.memory_type == "kg":
            return self.memory_module.extract_and_update_from_text(*args, **kwargs)
        else:
            raise NotImplementedError(
                "extract_and_update_from_text is only available for KGMemory."
            )


class LearnableAgent(BaseLearnableAgent):
    """
    A full-featured agent capable of learning (e.g., via PEFT) and utilizing memory.
    """

    def __init__(
        self,
        model: BaseVLM | BaseLLM,
        system_prompt: str,
        tools: Optional[Dict] = None,
        tools_schema: Optional[Dict] = None,
        learning_head: str = None,
        learning_head_config: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            max_tokens=max_tokens,
            agent_name=agent_name,
            tools=tools,
            tools_schema=tools_schema,
        )
        kg_model = self.model.model if hasattr(self.model, "peft_head") else self.model
        self.memory = MemoryManager(
            memory_type=memory_type,
            system_prompt=system_prompt,
            model=kg_model if memory_type == "kg" else None,
        )

    def plan(self, user_prompt: str, max_tokens: int = None, role: str = "user") -> str:
        self.memory.update_memory(role, user_prompt)
        context = self.memory.to_llm_format()
        system_prompt_planning = getattr(self, "system_prompt_planning", None)
        if system_prompt_planning:
            system_updated = False
            for msg in context:
                if msg["role"] == "system":
                    msg["content"] = system_prompt_planning
                    system_updated = True
                    break
            if not system_updated:
                context.insert(0, {"role": "system", "content": system_prompt_planning})
        output = self.model.run(
            messages=context,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=True,
            tools=self.tools_schema,
        )
        output_str = json.dumps(output) if isinstance(output, dict) else str(output)
        self.memory.update_memory("assistant", output_str)
        return output

    def execute(
        self, user_prompt: str, max_tokens: int = None, role: str = "user"
    ) -> str:
        self.memory.update_memory(role, user_prompt)
        context = self.memory.to_llm_format()
        output = self.model.run(
            messages=context,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=False,
            tools=self.tools_schema,
        )
        output_str = json.dumps(output) if isinstance(output, dict) else str(output)
        self.memory.update_memory("assistant", output_str)
        return output


class Agent(BaseAgent):
    """
    A non-learnable agent that utilizes memory and can interact with local models or external APIs.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        system_prompt: str,
        tools: Optional[Dict] = None,
        tools_schema: Optional[Dict] = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
    ):
        model_max_tokens = model_config.get("max_tokens", max_tokens)
        self.model_instance = self._create_model_from_config(
            model_config, default_max_tokens=model_max_tokens
        )
        super().__init__(
            model=self.model_instance,
            system_prompt=system_prompt,
            tools=tools,
            tools_schema=tools_schema,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
        self.memory = MemoryManager(
            memory_type=memory_type,
            system_prompt=system_prompt,
            model=self.model_instance if memory_type == "kg" else None,
        )
        self._model_config = model_config

    def _create_model_from_config(
        self, config: Dict[str, Any], default_max_tokens: int
    ) -> BaseLLM | BaseVLM | BaseAPIModel:
        model_type = config.get("type")
        model_name = config.get("name")
        max_tokens = config.get("max_tokens", default_max_tokens)
        if not model_name:
            raise ValueError("Model configuration must include a 'name'.")
        if model_type == "local":
            model_class_type = config.get("class", "llm")
            torch_dtype = config.get("torch_dtype", "auto")
            device_map = config.get("device_map", "auto")
            extra_kwargs = {
                k: v
                for k, v in config.items()
                if k
                not in [
                    "type",
                    "name",
                    "class",
                    "max_tokens",
                    "torch_dtype",
                    "device_map",
                ]
            }
            if model_class_type == "llm":
                return BaseLLM(
                    model_name=model_name,
                    max_tokens=max_tokens,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            elif model_class_type == "vlm":
                return BaseVLM(
                    model_name=model_name,
                    max_tokens=max_tokens,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    **extra_kwargs,
                )
            else:
                raise ValueError(f"Unsupported local model class: {model_class_type}")
        elif model_type == "api":
            api_key = config.get("api_key")
            base_url = config.get("base_url")
            extra_kwargs = {
                k: v
                for k, v in config.items()
                if k not in ["type", "name", "api_key", "base_url", "max_tokens"]
            }
            return BaseAPIModel(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported model type in config: {model_type}. Must be 'local' or 'api'."
            )

    def _get_api_kwargs(self) -> Dict[str, Any]:
        if isinstance(self.model_instance, BaseAPIModel):
            kwargs = {
                k: v
                for k, v in self._model_config.items()
                if k
                not in [
                    "type",
                    "name",
                    "class",
                    "api_key",
                    "base_url",
                    "max_tokens",
                    "torch_dtype",
                    "device_map",
                ]
            }
            return kwargs
        return {}

    def chat(
        self,
        user_prompt: str,
        max_tokens: Optional[int] = None,
        role: str = "user",
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        self.memory.update_memory(role, user_prompt)
        context = self.memory.to_llm_format()
        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)
        output = self.model_instance.run(
            messages=context,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=json_mode,
            tools=self.tools_schema,
            **api_kwargs,
        )
        output_str = (
            json.dumps(output)
            if isinstance(output, dict) and json_mode
            else str(output)
        )
        self.memory.update_memory("assistant", output_str)
        return output

    def plan(
        self,
        user_prompt: str,
        max_tokens: int = None,
        role: str = "user",
        **kwargs,
    ) -> str:
        self.memory.update_memory(role, user_prompt)
        context = self.memory.to_llm_format()
        system_prompt_planning = getattr(self, "system_prompt_planning", None)
        if system_prompt_planning:
            system_updated = False
            for msg in context:
                if msg["role"] == "system":
                    msg["content"] = system_prompt_planning
                    system_updated = True
                    break
            if not system_updated:
                context.insert(0, {"role": "system", "content": system_prompt_planning})
        api_kwargs = self._get_api_kwargs()
        api_kwargs.update(kwargs)
        output = self.model_instance.run(
            messages=context,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=True,
            tools=self.tools_schema,
            **api_kwargs,
        )
        output_str = json.dumps(output) if isinstance(output, dict) else str(output)
        self.memory.update_memory("assistant", output_str)
        return output


class BrowserAgent(Agent):
    """BrowserAgent is an agent that leverages the Playwright library to automate browser interactions with the web."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
    ):
        self.generation_system_prompt = generation_system_prompt
        self.critic_system_prompt = critic_system_prompt
        super().__init__(
            model_config=model_config,
            system_prompt=generation_system_prompt,
            tools=None,
            tools_schema=None,
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )

    def think(self, user_prompt: str = None, max_tokens: int = None) -> str:
        if not max_tokens:
            max_tokens = self.max_tokens
        if user_prompt:
            self.memory.update_memory("user", user_prompt)
        messages = self.memory.retrieve_all()
        system_updated = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = self.generation_system_prompt
                system_updated = True
                break
        if not system_updated:
            messages.insert(
                0, {"role": "system", "content": self.generation_system_prompt}
            )
        api_kwargs = self._get_api_kwargs()
        output = self.model_instance.run(
            messages=messages,
            role="agent",
            tools=self.tools_schema,
            max_tokens=max_tokens or self.max_tokens,
            json_mode=False,
            **api_kwargs,
        )
        self.memory.update_memory("agent", output)
        return output

    def critic(self):
        messages = self.memory.retrieve_all()
        system_updated = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = self.critic_system_prompt
                system_updated = True
                break
        if not system_updated:
            messages.insert(0, {"role": "system", "content": self.critic_system_prompt})
        api_kwargs = self._get_api_kwargs()
        output = self.model_instance.run(
            messages=messages,
            role="critic",
            max_tokens=self.max_tokens,
            json_mode=False,
            **api_kwargs,
        )
        self.memory.update_memory("critic", output)
        return output

    @classmethod
    async def create(
        cls,
        model_config: Dict[str, Any],
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        agent_name: Optional[str] = None,
    ):
        agent = cls(
            model_config=model_config,
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
        web_browser = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser, name)
            for name in dir(web_browser)
            if name.startswith("FN_")
        ]
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        await agent.initialize_browser_tool(
            temp_dir=temp_dir, headless=headless_browser
        )
        return agent

    @classmethod
    async def create_safe(
        cls,
        model_config: Dict[str, Any],
        generation_system_prompt: str = None,
        critic_system_prompt: str = None,
        memory_type: Optional[str] = "conversation_history",
        max_tokens: Optional[int] = 512,
        temp_dir: Optional[str] = "./tmp/screenshots",
        headless_browser: bool = True,
        timeout: Optional[int] = None,
        agent_name: Optional[str] = None,
    ) -> "BrowserAgent":
        agent = cls(
            model_config=model_config,
            generation_system_prompt=generation_system_prompt,
            critic_system_prompt=critic_system_prompt,
            memory_type=memory_type,
            max_tokens=max_tokens,
            agent_name=agent_name,
        )
        web_browser = importlib.import_module("src.environment.web_browser")
        agent.tools_schema = [
            getattr(web_browser, name)
            for name in dir(web_browser)
            if name.startswith("FN_")
        ]
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        for attempt in range(3):
            try:
                await asyncio.wait_for(
                    agent.initialize_browser_tool(
                        temp_dir=temp_dir, headless=headless_browser
                    ),
                    timeout=timeout or 5,
                )
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise TimeoutError("BrowserAgent initialization timed out.")
        return agent

    async def initialize_browser_tool(self, **kwargs):
        self.browser_tool = await BrowserTool.create(**kwargs)
        self.browser_methods = {}
        for attr in dir(self.browser_tool):
            if not attr.startswith("_"):
                method = getattr(self.browser_tool, attr)
                if callable(method):
                    self.browser_methods[attr] = method

"""
This module defines learnable agent classes that can incorporate trainable components.

It includes the BaseLearnableAgent abstract class and the LearnableAgent implementation
that supports PEFT (Parameter-Efficient Fine-Tuning) and other learning heads for
customizing model behavior through training.

LearnableAgent only supports local models (not API models) because:
- Training requires direct access to model weights
- PEFT/LoRA needs to modify the model architecture
- RL training frameworks (trl, PPO, DPO) need raw model/tokenizer access
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .agents import BaseAgent
from .memory import MemoryManager, Message
from .utils import RequestContext

# Type-only imports for optional local model support (requires marsys[local-models])
if TYPE_CHECKING:
    from marsys.models.models import (
        LocalProviderAdapter,
        HuggingFaceLLMAdapter,
        HuggingFaceVLMAdapter,
        ModelConfig,
    )


class BaseLearnableAgent(BaseAgent, ABC):
    """
    Base class for agents that can incorporate learnable components (e.g., PEFT heads).

    Inherits from BaseAgent and adds handling for learning head initialization.

    IMPORTANT: LearnableAgent only supports local HuggingFace models (not API models or vLLM).
    This is because training requires direct access to model weights and tokenizer.

    Attributes:
        _learning_head_name: Name of the learning head type (e.g., 'peft').
        _learning_config: Configuration dictionary for the learning head.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        goal: str,
        instruction: str,
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the BaseLearnableAgent.

        Args:
            model_config: ModelConfig for creating the local model (must have type="local").
            goal: A 1-2 sentence summary of what this agent accomplishes.
            instruction: Detailed instructions on how the agent should behave and operate.
            learning_head: Optional name of the learning head type (e.g., 'peft').
            learning_head_config: Optional configuration for the learning head.
            tools: Optional dictionary of tools.
            max_tokens: Default maximum tokens for model generation.
            name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            bidirectional_peers: If True, creates bidirectional edges with allowed_peers.
            input_schema: Optional schema for validating agent input.
            output_schema: Optional schema for validating agent output.
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")
            **kwargs: Additional arguments.

        Raises:
            TypeError: If model_config.type is not "local" (API models not supported).
            TypeError: If backend is "vllm" (vLLM not supported for training).
        """
        # Lazy import for local models (requires marsys[local-models])
        try:
            from marsys.models.models import (
                LocalProviderAdapter,
                HuggingFaceLLMAdapter,
                HuggingFaceVLMAdapter,
                LocalAdapterFactory,
                PeftHead,
            )
        except ImportError as e:
            raise ImportError(
                "Learnable agents require local model support. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        # Enforce local models only
        if model_config.type != "local":
            raise TypeError(
                f"LearnableAgent only supports local models (type='local'), "
                f"got type='{model_config.type}'. API models cannot be used for training."
            )

        # Enforce HuggingFace backend only (vLLM doesn't support training)
        backend = getattr(model_config, "backend", "huggingface") or "huggingface"
        if backend == "vllm":
            raise TypeError(
                "LearnableAgent does not support vLLM backend. "
                "Use backend='huggingface' for training support."
            )

        # Create adapter using factory
        model_class = model_config.model_class or "llm"
        effective_max_tokens = max_tokens if max_tokens is not None else model_config.max_tokens

        adapter_kwargs = {
            "max_tokens": effective_max_tokens,
            "thinking_budget": model_config.thinking_budget,
        }

        # Add HuggingFace-specific parameters
        if hasattr(model_config, "torch_dtype") and model_config.torch_dtype:
            adapter_kwargs["torch_dtype"] = model_config.torch_dtype
        if hasattr(model_config, "device_map") and model_config.device_map:
            adapter_kwargs["device_map"] = model_config.device_map
        if hasattr(model_config, "trust_remote_code"):
            adapter_kwargs["trust_remote_code"] = model_config.trust_remote_code
        if hasattr(model_config, "attn_implementation") and model_config.attn_implementation:
            adapter_kwargs["attn_implementation"] = model_config.attn_implementation

        model = LocalAdapterFactory.create_adapter(
            backend=backend,
            model_name=model_config.name,
            model_class=model_class,
            **adapter_kwargs,
        )

        super().__init__(
            model=model,
            goal=goal,
            instruction=instruction,
            tools=tools,
            max_tokens=effective_max_tokens,
            name=name,
            allowed_peers=allowed_peers,
            bidirectional_peers=bidirectional_peers,
            input_schema=input_schema,
            output_schema=output_schema,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
        )

        self._learning_head_name = learning_head
        self._learning_config = learning_head_config

        if learning_head == "peft":
            if not learning_head_config:
                raise ValueError(
                    "learning_head_config is required when learning_head is 'peft'"
                )
            self.model = PeftHead(model=self.model)
            self.model.prepare_peft_model(**learning_head_config)
    
    async def run_step(
        self, 
        request: Any, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step of agent reasoning with learning-aware context.
        
        This method extends BaseAgent.run_step() to add support for learning-specific
        operations like training iterations, loss tracking, and PEFT state management.
        
        Args:
            request: The request for this step
            context: Execution context from the coordination system
        
        Returns:
            Dictionary containing response and optional context selection
        """
        # Extract learning-specific context if present
        learning_context = context.get('learning_context', {})
        training_iteration = learning_context.get('iteration', 0)
        is_training = learning_context.get('is_training', False)
        
        # Log learning context if in training mode
        if is_training and hasattr(self, 'logger'):
            self.logger.debug(
                f"LearnableAgent '{self.name}' in training mode, iteration {training_iteration}"
            )
        
        # Call parent run_step to handle standard agent operations
        result = await super().run_step(request, context)
        
        # Add learning-specific metadata to result if in training mode
        if is_training:
            result['learning_metadata'] = {
                'iteration': training_iteration,
                'has_peft': self._learning_head_name == 'peft',
                'model_type': type(self.model).__name__
            }
            
            # Future: Add hooks for PEFT gradient updates, loss calculation, etc.
            # This will be implemented when StateManager supports model checkpoints
            
        return result




class LearnableAgent(BaseLearnableAgent):
    """
    An agent implementation that uses a local, potentially learnable (e.g., PEFT) model.

    It utilizes a MemoryManager to handle its internal state and implements the
    `_run` method for core logic execution based on different run modes.

    **Key Distinction**: LearnableAgent is designed to work with open-source models that run locally
    where you have direct access to model weights. This allows attaching learning heads (like PEFT)
    to train the model for specific agent workflows. Unlike the Agent class which uses API-based models,
    LearnableAgent can be fine-tuned and adapted through techniques like parameter-efficient fine-tuning.

    Use LearnableAgent when:
    - You need to customize model behavior through training
    - You have access to local GPU/compute resources
    - You want to use open-source models (LLaMA, Mistral, Qwen, etc.)
    - You need full control over the model architecture

    Training Access:
    - Access raw PyTorch model: `agent.model.model` (for HuggingFace adapter)
    - Access tokenizer: `agent.model.tokenizer`
    - These can be used with training frameworks like trl, PEFT, etc.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        goal: str,
        instruction: str,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        memory_type: Optional[str] = "conversation_history",
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        bidirectional_peers: bool = False,
        input_schema: Optional[Any] = None,
        output_schema: Optional[Any] = None,
        memory_retention: str = "session",
        memory_storage_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the LearnableAgent.

        Args:
            model_config: ModelConfig for creating the local model (must have type="local").
            goal: A 1-2 sentence summary of what this agent accomplishes.
            instruction: Detailed instructions on how the agent should behave and operate.
            tools: Optional dictionary of tools.
            memory_type: Type of memory module to use.
            learning_head: Optional type of learning head ('peft').
            learning_head_config: Optional configuration for the learning head.
            max_tokens: Default maximum tokens for model generation.
            name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            bidirectional_peers: If True, creates bidirectional edges with allowed_peers.
            input_schema: Optional schema for validating agent input.
            output_schema: Optional schema for validating agent output.
            memory_retention: Memory retention policy - "single_run", "session", or "persistent"
            memory_storage_path: Path for persistent memory storage (if retention is "persistent")
            **kwargs: Additional arguments.
        """
        super().__init__(
            model_config=model_config,
            goal=goal,
            instruction=instruction,
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            tools=tools,
            max_tokens=max_tokens,
            name=name,
            allowed_peers=allowed_peers,
            bidirectional_peers=bidirectional_peers,
            input_schema=input_schema,
            output_schema=output_schema,
            memory_retention=memory_retention,
            memory_storage_path=memory_storage_path,
            **kwargs,
        )

        # Lazy import for PeftHead type checking (requires marsys[local-models])
        try:
            from marsys.models.models import PeftHead, LocalProviderAdapter
        except ImportError:
            PeftHead = None
            LocalProviderAdapter = None

        # Get underlying model for KG memory if needed
        kg_model = None
        if PeftHead and isinstance(self.model, PeftHead):
            kg_model = self.model.model  # Get the adapter from PeftHead
        elif LocalProviderAdapter and isinstance(self.model, LocalProviderAdapter):
            kg_model = self.model

        self.memory = MemoryManager(
            memory_type=memory_type or "conversation_history",
            description=self.instruction,
            model=kg_model,
        )

    async def _run(
        self,
        messages: List[Dict[str, Any]],
        request_context: RequestContext,
        run_mode: str,
        **kwargs: Any
    ) -> Message:
        """
        PURE execution logic for the LearnableAgent.

        This method ONLY handles:
        1. Calling the language model asynchronously with the provided messages
        2. Creating a Message object from the model's HarmonizedResponse

        All message preparation including system prompt is handled by run_step().
        All memory operations are handled by run_step().

        Args:
            messages: List of message dictionaries ready for the LLM (including system prompt)
            request_context: The context for this specific run (managed by ExecutionEngine)
            run_mode: A string indicating the type of operation (e.g., 'chat', 'plan', 'think', 'auto_step')
            **kwargs: Additional keyword arguments specific to the run mode or model call

        Returns:
            A Message object representing the agent's raw response.
        """
        # Extract model parameters from kwargs
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", 0.7)  # Default for learnable models
        json_mode = kwargs.pop("json_mode", False)
        tools_schema = kwargs.pop("tools_schema", None)

        try:
            # Call the model asynchronously (BaseLocalModel.arun returns HarmonizedResponse)
            harmonized_response = await self.model.arun(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools_schema,
                **kwargs,
            )

            # Create Message from HarmonizedResponse (consistent with Agent._run pattern)
            assistant_message = Message.from_harmonized_response(
                harmonized_response,
                name=self.name
            )

            return assistant_message

        except Exception as e:
            # Return error as a Message
            error_message = Message(
                role="error",
                content=f"LLM call failed: {e}",
                name=self.name
            )
            return error_message

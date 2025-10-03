"""
This module defines learnable agent classes that can incorporate trainable components.

It includes the BaseLearnableAgent abstract class and the LearnableAgent implementation
that supports PEFT (Parameter-Efficient Fine-Tuning) and other learning heads for
customizing model behavior through training.
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

from .agents import BaseAgent
from .memory import MemoryManager, Message
from .utils import LogLevel, RequestContext
from .exceptions import (
    AgentFrameworkError,
    MessageError,
    MessageFormatError,
    MessageContentError,
    ActionValidationError,
    ToolCallError,
    SchemaValidationError,
    AgentError,
    AgentImplementationError,
    AgentConfigurationError,
    AgentPermissionError,
    AgentLimitError,
    ModelError,
    ModelResponseError,
    create_error_from_exception,
)


class BaseLearnableAgent(BaseAgent, ABC):
    """
    Base class for agents that can incorporate learnable components (e.g., PEFT heads).

    Inherits from BaseAgent and adds handling for learning head initialization.

    Attributes:
        _learning_head_name: Name of the learning head type (e.g., 'peft').
        _learning_config: Configuration dictionary for the learning head.
    """

    def __init__(
        self,
        model: Union["BaseVLM", "BaseLLM"],
        description: str,  # Renamed from system_prompt
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the BaseLearnableAgent.

        Args:
            model: The base language model instance (typically local).
            description: The base description of the agent's role and purpose.
            learning_head: Optional name of the learning head type (e.g., 'peft').
            learning_head_config: Optional configuration for the learning head.
            max_tokens: Default maximum tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            **kwargs: Additional arguments passed to BaseAgent.__init__ (e.g., tools).
        """
        super().__init__(
            model=model,
            description=description,  # Renamed
            tools=kwargs.get("tools"),
            # tools_schema=kwargs.get("tools_schema"), # Removed as BaseAgent.__init__ no longer takes it
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
        )
        self._learning_head_name = learning_head
        self._learning_config = learning_head_config
        if learning_head == "peft":
            # Lazy import for local models (requires marsys[local-models])
            try:
                from src.models.models import BaseLLM, BaseVLM, PeftHead
            except ImportError as e:
                raise ImportError(
                    "Learnable agents require local model support. Install with:\n"
                    "  pip install marsys[local-models]\n"
                    "or:\n"
                    "  uv pip install marsys[local-models]\n\n"
                    f"Original error: {str(e)}"
                ) from e

            if not learning_head_config:
                raise ValueError(
                    "learning_head_config is required when learning_head is 'peft'"
                )
            if not isinstance(model, (BaseLLM, BaseVLM)):
                raise TypeError(
                    f"Base model for PEFT must be BaseLLM or BaseVLM, got {type(model)}"
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
    - You want to use open-source models (LLaMA, Mistral, etc.)
    - You need full control over the model architecture
    """

    def __init__(
        self,
        model: Union["BaseVLM", "BaseLLM"],
        description: str,  # Renamed from system_prompt
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        # tools_schema: Optional[List[Dict[str, Any]]] = None, # Removed from signature
        learning_head: Optional[str] = None,
        learning_head_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 512,
        agent_name: Optional[str] = None,
        allowed_peers: Optional[List[str]] = None,
        **kwargs: Any,  # Keep kwargs for other potential BaseAgent args if any in future
    ) -> None:
        """
        Initializes the LearnableAgent.

        Args:
            model: The local language model instance (potentially wrapped by PeftHead).
            description: The base description of the agent's role and purpose.
            tools: Optional dictionary of tools.
            learning_head: Optional type of learning head ('peft').
            learning_head_config: Optional configuration for the learning head.
            max_tokens: Default maximum tokens for model generation.
            agent_name: Optional specific name for registration.
            allowed_peers: List of agent names this agent can call.
            **kwargs: Additional arguments passed to BaseAgent.__init__.
        """
        super().__init__(
            model=model,
            description=description,  # Renamed
            learning_head=learning_head,
            learning_head_config=learning_head_config,
            max_tokens=max_tokens,
            agent_name=agent_name,
            allowed_peers=allowed_peers,
            tools=tools,  # Pass tools
            # tools_schema is not passed as BaseLearnableAgent's super call to BaseAgent handles it
            **kwargs,
        )
        # Lazy import for PeftHead type checking (requires marsys[local-models])
        try:
            from src.models.models import PeftHead
        except ImportError:
            PeftHead = None  # If not installed, PeftHead won't be used

        kg_model: Union["BaseVLM", "BaseLLM"]
        if PeftHead and isinstance(self.model, PeftHead):
            kg_model = self.model.model
        else:
            kg_model = self.model
        self.memory = MemoryManager(
            memory_type="conversation_history",  # Default memory type
            description=self.instruction,  # Pass agent's instruction for initial system message
            model=None,  # kg_model is not needed since memory_type is "conversation_history"
            input_processor=self._input_message_processor(),
            output_processor=self._output_message_processor(),
        )

    def _input_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts LLM JSON responses to Message-compatible format.
        Extracts agent_calls information from JSON content when present.
        """

        def transform_from_llm(data: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy of the original data
            result = data.copy()

            # Check if content contains agent call info in JSON
            content = data.get("content")
            if data.get("role") == "assistant" and content and isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                    if (
                        isinstance(parsed_content, dict)
                        and parsed_content.get("next_action") == "invoke_agent"
                    ):
                        # Extract agent_calls information as raw dict list - Message.__post_init__ will convert
                        action_input = parsed_content.get("action_input", {})
                        if (
                            isinstance(action_input, dict)
                            and "agent_name" in action_input
                        ):
                            result["agent_calls"] = [action_input]  # Create list with single agent call

                            # Keep only thought in content if present
                            thought = parsed_content.get("thought")
                            if thought:
                                result["content"] = thought
                            else:
                                result["content"] = None
                except (json.JSONDecodeError, TypeError):
                    # Content is not JSON or parsing failed, keep as is
                    pass

            return result

        return transform_from_llm

    def _output_message_processor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Creates a processor function that converts Message dicts to LLM-compatible format.
        Synthesizes JSON content when agent_calls is present.
        """

        def transform_to_llm(msg_dict: Dict[str, Any]) -> Dict[str, Any]:
            # Start with a copy
            result = msg_dict.copy()

            # If agent_calls is present and role is assistant, synthesize JSON content
            if msg_dict.get("role") == "assistant" and msg_dict.get("agent_calls"):
                agent_calls = msg_dict["agent_calls"]
                thought = msg_dict.get("content", "I need to invoke another agent.")

                # For now, we only support single agent invocation, so take the first one
                if agent_calls and len(agent_calls) > 0:
                    first_agent_call_msg = agent_calls[0]
                    
                    # Handle both AgentCallMsg objects and raw dict format
                    if hasattr(first_agent_call_msg, 'to_dict'):
                        # It's an AgentCallMsg object
                        agent_call_data = first_agent_call_msg.to_dict()
                    else:
                        # It's already a dict
                        agent_call_data = first_agent_call_msg

                synthesized_content = {
                    "thought": thought,
                    "next_action": "invoke_agent",
                        "action_input": agent_call_data,
                }
                result["content"] = json.dumps(synthesized_content)
                
                # Remove agent_calls from result as it's not part of OpenAI API
                result.pop("agent_calls", None)
            else:
                # Remove agent_calls if present (not part of OpenAI API)
                result.pop("agent_calls", None)

            return result

        return transform_to_llm
    
    async def run_step(
        self, 
        request: Any, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute one step of agent reasoning for LearnableAgent.
        
        This method handles all memory management, context extraction, and logging,
        then calls the pure _run() method for model interaction.
        
        Args:
            request: The request for this step. Can be a prompt string, dict with
                    'prompt' and 'passed_referenced_context', or other formats.
            context: Execution context from the coordination system
        
        Returns:
            Dictionary containing response and optional context selection
        """
        # Extract context information
        step_id = context.get('step_id')
        session_id = context.get('session_id')
        is_continuation = context.get('is_continuation', False)
        execution_mode = context.get('execution_mode', 'auto_step')
        request_context = context['request_context']  # Must be provided by ExecutionEngine
        
        # Apply memory retention policy (handled by BaseAgent.run_step if we call super())
        # But we need custom handling for LearnableAgent's prompt extraction
        
        # Extract prompt and context messages
        user_actual_prompt_content, passed_context_messages = self._extract_prompt_and_context(request)
        prompt_sender_name = request_context.caller_agent_name or "user"
        
        # Log the execution start
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"LearnableAgent executing run_step with mode='{execution_mode}'. Actual prompt: {str(user_actual_prompt_content)[:100]}...",
            data={"has_passed_context": bool(passed_context_messages)},
        )
        
        # Add passed context messages to memory
        for ref_msg in passed_context_messages:
            self.memory.update_memory(message=ref_msg)
            await self._log_progress(
                request_context,
                LogLevel.DEBUG,
                f"LearnableAgent added referenced message ID {ref_msg.message_id} (Role: {ref_msg.role}) to memory.",
            )
        
        # Add the current user prompt to memory
        if user_actual_prompt_content:
            self.memory.update_memory(
                role="user",
                content=user_actual_prompt_content,
                name=prompt_sender_name,
            )
        
        # Get base instruction for this run mode
        base_instruction_for_run = getattr(
            self,
            f"instruction_{execution_mode}",
            self.instruction,
        )
        
        # Determine JSON mode settings
        json_mode_for_guidelines = execution_mode == "auto_step"
        has_tools = bool(self.tools_schema)
        json_mode_for_llm_native = (
            json_mode_for_guidelines or execution_mode == "plan"
        ) and not has_tools
        
        # Construct system prompt
        operational_system_prompt = self._construct_full_system_prompt(
            base_description=base_instruction_for_run,
            json_mode_for_output=json_mode_for_guidelines,
        )
        
        # Get messages from memory and prepare for LLM
        llm_messages_for_model = self.memory.to_llm_format()
        
        # Update or insert system message
        system_message_found = False
        for i, msg_dict in enumerate(llm_messages_for_model):
            if msg_dict["role"] == "system":
                llm_messages_for_model[i] = {
                    "role": "system", 
                    "content": operational_system_prompt
                }
                system_message_found = True
                break
        
        if not system_message_found:
            llm_messages_for_model.insert(
                0,
                {"role": "system", "content": operational_system_prompt},
            )
        
        # Extract model parameters
        model_kwargs = context.get('model_kwargs', {})
        max_tokens_override = model_kwargs.pop("max_tokens", self.max_tokens)
        temperature_override = model_kwargs.pop("temperature", 0.7)  # Default for learnable
        
        # Log before calling the model
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"LearnableAgent calling internal LLM (mode: {execution_mode}, LLM_JSON_mode: {json_mode_for_llm_native})",
            data={"system_prompt_length": len(operational_system_prompt)},
        )
        
        try:
            # Call pure _run() method
            assistant_message = await self._run(
                messages=llm_messages_for_model,
                request_context=request_context,
                run_mode=execution_mode,
                max_tokens=max_tokens_override,
                temperature=temperature_override,
                json_mode=json_mode_for_llm_native,
                tools_schema=self.tools_schema if has_tools else None,
                **model_kwargs
            )
            
            # Update memory with the response
            if isinstance(assistant_message.content, dict) or (
                isinstance(assistant_message.content, str) and assistant_message.content
            ):
                self.memory.update_from_response(
                    {
                        "role": assistant_message.role,
                        "content": assistant_message.content,
                        "name": assistant_message.name,
                        "tool_calls": assistant_message.tool_calls,
                    },
                    message_id=assistant_message.message_id,
                    default_role="assistant",
                    default_name=self.name
                )
            
            # Log successful completion
            await self._log_progress(
                request_context,
                LogLevel.DETAILED,
                f"LearnableAgent LLM call successful. Output content: {str(assistant_message.content)[:100]}...",
                data={
                    "tool_calls": assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else None,
                    "message_id": assistant_message.message_id
                },
            )
            
        except Exception as e:
            await self._log_progress(
                request_context,
                LogLevel.MINIMAL,
                f"LearnableAgent LLM call failed: {e}",
                data={"error": str(e)},
            )
            # Create error message
            assistant_message = Message(
                role="error", content=f"LLM call failed: {e}", name=self.name
            )
        
        # Log run completion
        await self._log_progress(
            request_context,
            LogLevel.DETAILED,
            f"LearnableAgent run_step mode='{execution_mode}' finished.",
        )
        
        # Return result in coordination format
        return {
            "response": assistant_message,
            "context_selection": None  # LearnableAgent doesn't use context selection yet
        }

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
        1. Calling the language model with the provided messages
        2. Creating a Message object from the model's output
        
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
            # Call the model with prepared messages
            raw_model_output: Any = self.model.run(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools_schema,
                **kwargs,
            )
            
            # Generate message ID for the response
            new_message_id = str(uuid.uuid4())
            
            # Create Message from response
            if isinstance(raw_model_output, dict) and "role" in raw_model_output:
                # Use Message.from_response_dict which handles transformations
                assistant_message = Message.from_response_dict(
                    raw_model_output,
                    default_id=new_message_id,
                    default_role="assistant",
                    default_name=self.name,
                    processor=self._input_message_processor()
                )
            else:
                # String response or other format - create message directly
                content = str(raw_model_output) if raw_model_output else None
                assistant_message = Message(
                    role="assistant",
                    content=content,
                    name=self.name,
                    message_id=new_message_id
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

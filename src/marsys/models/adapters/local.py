"""Local model provider adapters for HuggingFace and vLLM backends."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from marsys.models.response_models import (
    HarmonizedResponse,
    ResponseMetadata,
    ToolCall,
)
from marsys.models.utils import apply_tools_template, parse_local_model_tool_calls

logger = logging.getLogger(__name__)

# --- Utilities for Local Models ---


class ThinkingTokenBudgetProcessor:
    """
    LogitsProcessor that limits thinking tokens for models like Qwen3-Thinking.

    After max_thinking_tokens are generated within <think>...</think> blocks,
    forces the model to output </think> and continue with the response.

    If the model doesn't support thinking tokens (no <think>/<//think> in vocabulary),
    this processor is automatically disabled and passes through scores unchanged.
    """

    def __init__(self, tokenizer, max_thinking_tokens: int = 1000):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.enabled = True  # Will be set to False if thinking tokens not found

        # Get token IDs for thinking delimiters
        # Check if <think> and </think> actually exist in the vocabulary
        self.think_start_token_id = self._get_token_id("<think>")
        self.think_end_token_id = self._get_token_id("</think>")

        # If either token is not found or maps to unknown token, disable the processor
        unk_token_id = getattr(tokenizer, 'unk_token_id', None)
        if (self.think_start_token_id is None or
            self.think_end_token_id is None or
            self.think_start_token_id == unk_token_id or
            self.think_end_token_id == unk_token_id):
            self.enabled = False

        # Get newline token (optional, used for gradual boosting)
        self.newline_token_id = self._get_token_id("\n")

        self.thinking_started = False
        self.thinking_ended = False
        self.tokens_in_thinking = 0

    def _get_token_id(self, token_str: str):
        """
        Safely get token ID for a string. Returns None if not found.
        """
        try:
            # Try encoding first
            encoded = self.tokenizer.encode(token_str, add_special_tokens=False)
            if encoded and len(encoded) == 1:
                return encoded[0]
            # If encoding produces multiple tokens, it's not a single special token
            if encoded and len(encoded) > 1:
                return None
        except Exception:
            pass

        try:
            # Fallback: try convert_tokens_to_ids
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            # Check if it returned the unknown token
            unk_id = getattr(self.tokenizer, 'unk_token_id', None)
            if token_id != unk_id:
                return token_id
        except Exception:
            pass

        return None

    def __call__(self, input_ids, scores):
        """Process logits during generation."""
        # If processor is disabled (non-thinking model), pass through unchanged
        if not self.enabled:
            return scores

        # If thinking already ended, pass through unchanged
        if self.thinking_ended:
            return scores

        # Get the last generated token
        last_token = input_ids[0, -1].item()

        # Check if thinking phase started
        if not self.thinking_started:
            if last_token == self.think_start_token_id:
                self.thinking_started = True
                self.tokens_in_thinking = 0
            return scores

        # Check if thinking ended naturally
        if last_token == self.think_end_token_id:
            self.thinking_ended = True
            return scores

        # Count tokens in thinking phase
        self.tokens_in_thinking += 1

        # At 90% of budget, start boosting </think> logits
        if self.tokens_in_thinking >= self.max_thinking_tokens * 0.90:
            if self.newline_token_id is not None:
                scores[:, self.newline_token_id] += 3.0
            scores[:, self.think_end_token_id] += 8.0

        # At 100% of budget, force </think>
        if self.tokens_in_thinking >= self.max_thinking_tokens:
            import torch
            scores = torch.full_like(scores, float('-inf'))
            scores[:, self.think_end_token_id] = 0
            self.thinking_ended = True

        return scores

    def reset(self):
        """Reset state for new generation."""
        self.thinking_started = False
        self.thinking_ended = False
        self.tokens_in_thinking = 0

    @property
    def is_enabled(self) -> bool:
        """Check if this processor is active (model supports thinking tokens)."""
        return self.enabled



# --- Local Model Adapter Pattern ---


class LocalProviderAdapter(ABC):
    """
    Abstract base class for local model provider adapters (HuggingFace, vLLM, etc.).

    Training Access:
        HuggingFace adapters expose `model` and `tokenizer` for training frameworks:
        - adapter.model: Raw PyTorch model (AutoModelForCausalLM or AutoModelForVision2Seq)
        - adapter.tokenizer: HuggingFace tokenizer

        vLLM adapters do NOT expose these (vLLM doesn't support training).
        Use `supports_training` property to check.

    Example for training integration:
        ```python
        if adapter.supports_training:
            # Access raw PyTorch model and tokenizer
            pytorch_model = adapter.model
            tokenizer = adapter.tokenizer
            # Use with trl, PEFT, or custom training loops
        ```
    """

    # Type hints for training access (set by HuggingFace adapters, None for vLLM)
    model: Any = None  # PyTorch model (when available)
    tokenizer: Any = None  # HuggingFace tokenizer (when available)

    def __init__(self, model_name: str, model_class: str = "llm", **config):
        """
        Initialize the local adapter.

        Args:
            model_name: The model identifier (e.g., "Qwen/Qwen3-VL-8B-Thinking")
            model_class: Either "llm" or "vlm" for text or vision models
            **config: Backend-specific configuration (torch_dtype, device_map, etc.)
        """
        self.model_name = model_name
        self.model_class = model_class
        self._config = config

    @abstractmethod
    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference synchronously.

        Returns:
            Dictionary with: {"role": "assistant", "content": "...", "thinking": "...", "tool_calls": []}
        """
        pass

    @abstractmethod
    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Run inference asynchronously.

        Returns:
            HarmonizedResponse for compatibility with the Agent framework.
        """
        pass

    @property
    def backend(self) -> str:
        """Return the backend name (e.g., 'huggingface', 'vllm')"""
        return self.__class__.__name__.replace("Adapter", "").lower()

    @property
    def supports_training(self) -> bool:
        """
        Check if this adapter supports training (exposes model and tokenizer).

        HuggingFace adapters return True, vLLM returns False.
        """
        return self.model is not None and self.tokenizer is not None


class HuggingFaceLLMAdapter(LocalProviderAdapter):
    """HuggingFace adapter for text-only language models."""

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for transformers (requires marsys[local-models])
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local LLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        # Extract trust_remote_code for tokenizer
        trust_remote_code = kwargs.get("trust_remote_code", False)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

        # Create thinking budget processor if specified
        self._thinking_processor = None
        if thinking_budget is not None:
            self._thinking_processor = ThinkingTokenBudgetProcessor(
                self.tokenizer, max_thinking_tokens=thinking_budget
            )

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run LLM inference using HuggingFace transformers."""
        # Format the input with the tokenizer
        text: str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if json_mode:
            text += "```json\n"
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": max_tokens if max_tokens else self._max_tokens,
        }

        # Add thinking budget processor if configured
        if self._thinking_processor is not None:
            self._thinking_processor.reset()
            generate_kwargs["logits_processor"] = [self._thinking_processor]

        generated_ids = self.model.generate(**model_inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded: List[str] = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Parse thinking content from <think>...</think> blocks
        raw_content = decoded[0]
        thinking_content = None
        final_content = raw_content

        if "<think>" in raw_content:
            import re
            think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # Parse tool calls from <tool_call>...</tool_call> blocks
        final_content, parsed_tool_calls = parse_local_model_tool_calls(final_content)

        if json_mode:
            final_content = "\n".join(final_content.split("```")[:-1]).strip()
            final_content = json.loads(final_content.replace("\n", ""))

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": "assistant",
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": parsed_tool_calls,
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """Async LLM inference - wraps sync in thread."""
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        # Convert parsed tool calls to ToolCall objects
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=tc.get("function", {}),
            )
            for tc in raw_result.get("tool_calls", [])
        ]

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=tool_calls,
            metadata=ResponseMetadata(
                provider="huggingface",
                model=self.model.config.name_or_path if hasattr(self.model, 'config') else self.model_name,
            ),
        )


class HuggingFaceVLMAdapter(LocalProviderAdapter):
    """HuggingFace adapter for vision-language models."""

    def __init__(
        self,
        model_name: str,
        model_class: str = "vlm",
        max_tokens: int = 1024,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for transformers (requires marsys[local-models])
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local VLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        # Extract trust_remote_code
        trust_remote_code = kwargs.get("trust_remote_code", False)

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, **kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self._device = device_map
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

        # Create thinking budget processor if specified
        self._thinking_processor = None
        if thinking_budget is not None:
            self._thinking_processor = ThinkingTokenBudgetProcessor(
                self.tokenizer, max_thinking_tokens=thinking_budget
            )

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        role: str = "assistant",
        **kwargs,
    ) -> Dict[str, Any]:
        """Run VLM inference using HuggingFace transformers."""
        # Format the input with the tokenizer
        if tools:
            apply_tools_template(messages, tools)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = f"{text}\n<|im_start|>{role}"
        if json_mode:
            text += "```json\n"

        # Lazy import for vision processing
        try:
            from marsys.models.processors import process_vision_info
        except ImportError as e:
            raise ImportError(
                "Vision processing requires PyTorch and torchvision. Install with:\n"
                "  pip install marsys[local-models]\n"
                "or:\n"
                "  uv pip install marsys[local-models]\n\n"
                f"Original error: {str(e)}"
            ) from e

        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.model.device)

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": max_tokens if max_tokens else self._max_tokens,
        }

        # Add thinking budget processor if configured
        if self._thinking_processor is not None:
            self._thinking_processor.reset()
            generate_kwargs["logits_processor"] = [self._thinking_processor]

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Parse thinking content
        raw_content = decoded[0]
        thinking_content = None
        final_content = raw_content

        if "<think>" in raw_content:
            import re
            think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        # Parse tool calls from <tool_call>...</tool_call> blocks
        final_content, parsed_tool_calls = parse_local_model_tool_calls(final_content)

        if json_mode:
            final_content = "\n".join(final_content.split("```")[:-1]).strip()
            final_content = json.loads(final_content.replace("\n", ""))

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": role,
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": parsed_tool_calls,
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        role: str = "assistant",
        **kwargs,
    ) -> HarmonizedResponse:
        """Async VLM inference - wraps sync in thread."""
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            role=role,
            **kwargs,
        )

        # Convert parsed tool calls to ToolCall objects
        tool_calls = [
            ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=tc.get("function", {}),
            )
            for tc in raw_result.get("tool_calls", [])
        ]

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=tool_calls,
            metadata=ResponseMetadata(
                provider="huggingface",
                model=self.model.config.name_or_path if hasattr(self.model, 'config') else self.model_name,
            ),
        )


class VLLMAdapter(LocalProviderAdapter):
    """
    vLLM adapter for high-throughput production inference.

    vLLM provides:
    - Continuous batching for high throughput
    - PagedAttention for memory efficiency
    - FP8/AWQ/GPTQ quantization support
    - Tensor parallelism for multi-GPU inference
    - Native chat completion with llm.chat()

    Supports both text-only LLMs and vision-language models (VLMs).

    Note: Requires marsys[production] installation.

    References:
    - https://docs.vllm.ai/en/stable/getting_started/quickstart/
    - https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html
    """

    def __init__(
        self,
        model_name: str,
        model_class: str = "llm",
        max_tokens: int = 1024,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize vLLM adapter.

        Args:
            model_name: HuggingFace model name or path
            model_class: "llm" or "vlm" (both use same vLLM interface)
            max_tokens: Maximum tokens to generate (vLLM default is only 16!)
            tensor_parallel_size: Number of GPUs for distributed inference
            gpu_memory_utilization: Fraction of GPU memory to use (0-1, default 0.9)
            dtype: Data type - "auto", "float16", "bfloat16", "float32"
            quantization: Quantization method - "awq", "gptq", "fp8", or None
            thinking_budget: Token budget for thinking (used for thinking models)
            **kwargs: Additional vLLM engine arguments (trust_remote_code, etc.)
        """
        super().__init__(model_name, model_class, **kwargs)

        # Lazy import for vLLM (requires marsys[production])
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM support requires additional dependencies. Install with:\n"
                "  pip install marsys[production]\n"
                "or:\n"
                "  uv pip install marsys[production]\n\n"
                f"Original error: {str(e)}"
            ) from e

        self._SamplingParams = SamplingParams

        # Build vLLM initialization kwargs
        vllm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": dtype,
            "trust_remote_code": kwargs.get("trust_remote_code", False),
        }

        # Add quantization if specified
        if quantization:
            vllm_kwargs["quantization"] = quantization

        # Add any additional engine arguments
        for key in ["max_model_len", "enforce_eager", "seed", "swap_space", "cpu_offload_gb"]:
            if key in kwargs:
                vllm_kwargs[key] = kwargs[key]

        self.model = LLM(**vllm_kwargs)
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget

    def run(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference using vLLM.

        Uses llm.chat() for chat completion format which handles:
        - Message formatting with chat templates
        - Multi-modal inputs (images via image_url in content)
        """
        # Create sampling params (vLLM default max_tokens is only 16!)
        sampling_params = self._SamplingParams(
            max_tokens=max_tokens if max_tokens else self._max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            seed=kwargs.get("seed"),
            stop=kwargs.get("stop"),
        )

        # Use vLLM's native chat method which handles:
        # - Chat template application
        # - Multi-modal content (images as {"type": "image_url", ...})
        # Reference: https://docs.vllm.ai/en/v0.7.1/serving/multimodal_inputs.html
        outputs = self.model.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        # Extract generated text from first output
        output_text = outputs[0].outputs[0].text

        # Parse thinking content from <think>...</think> blocks
        thinking_content = None
        final_content = output_text

        if "<think>" in output_text:
            import re
            think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
            if think_match:
                thinking_content = think_match.group(1).strip()
                final_content = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

        # Handle JSON mode
        if json_mode:
            # Try to extract JSON from code blocks
            if "```json" in final_content:
                final_content = "\n".join(final_content.split("```json")[1].split("```")[0].strip().split("\n"))
            elif "```" in final_content:
                final_content = "\n".join(final_content.split("```")[1].split("```")[0].strip().split("\n"))
            try:
                final_content = json.loads(final_content)
            except json.JSONDecodeError:
                pass  # Keep as string if JSON parsing fails

        result_content = final_content
        if json_mode and isinstance(result_content, dict):
            result_content = json.dumps(result_content)

        return {
            "role": "assistant",
            "content": result_content,
            "thinking": thinking_content,
            "tool_calls": [],
        }

    async def arun(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        max_tokens: int = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List] = None,
        **kwargs,
    ) -> HarmonizedResponse:
        """
        Async inference using vLLM.

        Note: vLLM's offline LLM class is synchronous. For true async,
        use AsyncLLMEngine or the OpenAI-compatible server.
        This wraps sync in asyncio.to_thread for non-blocking behavior.
        """
        import asyncio

        raw_result = await asyncio.to_thread(
            self.run,
            messages=messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            tools=tools,
            images=images,
            **kwargs,
        )

        return HarmonizedResponse(
            role=raw_result.get("role", "assistant"),
            content=raw_result.get("content"),
            thinking=raw_result.get("thinking"),
            tool_calls=[],
            metadata=ResponseMetadata(
                provider="vllm",
                model=self.model_name,
            ),
        )



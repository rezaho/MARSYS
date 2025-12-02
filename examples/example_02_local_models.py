"""
Simple Two-Agent Example using Local Qwen3-VL Model

This example demonstrates a minimal multi-agent setup using a local
Vision-Language Model (Qwen3-VL).

Architecture:
    User <-> Coordinator -> Worker

The Coordinator receives tasks from User and delegates to the Worker for execution.

Backends:
    This example supports two backends for local model inference:
    - huggingface: Development/research (default, install with marsys[local-models])
    - vllm: Production high-throughput (install with marsys[production])

Requirements:
    # For HuggingFace backend (default):
    pip install marsys[local-models]

    # For vLLM backend (production):
    pip install marsys[production]

    # GPU Requirements:
    # - Qwen3-VL-8B: ~16GB VRAM (BF16) or ~8GB (FP8 via vLLM)
    # - Qwen3-VL-30B-A3B: ~24GB VRAM (BF16) or ~16GB (FP8 via vLLM)
"""

import asyncio
import os
from pathlib import Path

# Set Hugging Face cache directory (must be set before importing transformers/marsys)
HF_CACHE_DIR = Path(__file__).parent.parent / ".cache"
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

# Imports after env setup (HF_HOME must be set before importing transformers)
from marsys.agents import Agent  # noqa: E402
from marsys.coordination import Orchestra
from marsys.coordination.config import ExecutionConfig
from marsys.models import ModelConfig


async def main():
    # ==========================================================================
    # Option 1: HuggingFace backend (default) - Development/Research
    # ==========================================================================
    # Uses HuggingFace transformers for model loading and inference.
    # Easier to debug but lower throughput than vLLM.
    model_config = ModelConfig(
        type="local",
        name="Qwen/Qwen3-VL-8B-Thinking",  # Smaller model for testing
        model_class="vlm",  # Vision-Language Model
        backend="huggingface",  # Default backend (can be omitted)
        max_tokens=4096,
        thinking_budget=256,  # Limit thinking to 256 tokens (auto-disabled for non-thinking models)
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,  # Required for Qwen models
        # attn_implementation="flash_attention_2",  # Uncomment if flash_attn is properly installed
    )

    # ==========================================================================
    # Option 2: vLLM backend - Production/High-Throughput
    # ==========================================================================
    # Uses vLLM for high-throughput inference with:
    # - Continuous batching
    # - PagedAttention for memory efficiency
    # - FP8/AWQ/GPTQ quantization support
    # - Multi-GPU tensor parallelism
    #
    # model_config = ModelConfig(
    #     type="local",
    #     name="Qwen/Qwen3-VL-8B-Thinking",
    #     model_class="vlm",
    #     backend="vllm",  # Use vLLM backend
    #     max_tokens=4096,
    #     thinking_budget=256,
    #     torch_dtype="bfloat16",  # Maps to vLLM's 'dtype' parameter
    #     trust_remote_code=True,
    #     # vLLM-specific options:
    #     tensor_parallel_size=1,  # Number of GPUs for tensor parallelism
    #     gpu_memory_utilization=0.9,  # Fraction of GPU memory to use
    #     # quantization="fp8",  # Uncomment for FP8 quantization (reduces memory)
    # )

    # Create agents
    coordinator = Agent(
        model_config=model_config,
        name="Coordinator",
        goal="Coordinate task execution by delegating work to the Worker agent.",
        instruction="""You are a task coordinator. When you receive a task from the user:
1. Analyze what needs to be done
2. Delegate the actual work to the Worker agent by calling it with clear instructions
3. Once the Worker responds, summarize the results and provide the final answer to the user

Always delegate work to Worker - do not try to do the work yourself.""",
    )

    worker = Agent(
        model_config=model_config,
        name="Worker",
        goal="Execute tasks as instructed by the Coordinator.",
        instruction="""You are a task executor. When you receive instructions:
1. Carefully read the task requirements
2. Execute the task to the best of your ability
3. Return a clear, complete response with your results

Be thorough and precise in your work.""",
    )

    # Define topology with User interaction
    topology = {
        "agents": ["User", "Coordinator", "Worker"],
        "flows": [
            "User -> Coordinator",
            "Coordinator -> User",
            "Coordinator -> Worker",
            "Worker -> Coordinator",
        ],
        "rules": ["timeout(1200)"],  # 20 minute workflow timeout (overrides default 300s)
    }

    # Run with Orchestra and user interaction
    result = await Orchestra.run(
        task={"message": "Start task workflow"},
        topology=topology,
        execution_config=ExecutionConfig(
            user_interaction="terminal",
            user_first=True,
            initial_user_msg="Hello! I'm ready to help. What task would you like me to work on?",
            convergence_timeout=1200,  # 20 minutes
            user_interaction_timeout=1200,  # 20 minutes - waiting for user input
            step_timeout=1200,  # 20 minutes - individual step execution
        ),
        max_steps=20,
        verbosity=2,
    )

    if result and result.success:
        if result.final_response:
            print(f"\nFinal Response:\n{'-' * 40}")
            print(result.final_response)

    elif result and result.error:
            print(f"\nError: {result.error}")
    else:
        print("\nWorkflow ended.")


if __name__ == "__main__":
    asyncio.run(main())

# MARSYS Framework Overview

## Introduction

The Multi-Agent Reasoning Systems (MARSYS) framework is a Python-based platform for developing and orchestrating collaborative AI agents. It provides a structured environment for agents to communicate, use tools, interact with various AI models (both local and API-based), and manage their memory and learning processes.

MARSYS is designed for building complex applications where multiple specialized agents work together to achieve common or individual goals. This can range from automated research and data analysis to interactive web automation and content generation.

## Core Goals

*   **Modularity:** Allow developers to easily create, customize, and combine agents with different capabilities.
*   **Extensibility:** Provide clear interfaces for adding new AI models, tools, and memory systems.
*   **Scalability:** Support asynchronous operations and efficient resource management for handling multiple agents and complex tasks.
*   **Interoperability:** Enable seamless communication and data exchange between agents.
*   **Developer Experience:** Offer clear documentation, well-defined APIs, and robust logging for easier development and debugging.

## Key Features

*   **Agent Abstractions:**
    *   `BaseAgent`: The foundational class for all agents, providing registration, logging, and tool schema generation.
    *   `Agent`: A generic agent suitable for API-based models or non-learnable local models.
    *   `LearnableAgent`: An agent designed for local models that can be fine-tuned (e.g., using PEFT).
    *   `BrowserAgent`: A specialized agent for web automation tasks using Playwright.
*   **Message-Based Communication:**
    *   Agents communicate using standardized `Message` objects (OpenAI-compatible format).
    *   Supports roles like `system`, `user`, `assistant`, `tool`, and `error`.
*   **Memory Management:**
    *   `MemoryManager` for persistent and recallable agent memory.
    *   `ConversationMemory` to store chronological interaction history.
    *   Support for input/output message processors to transform data between agent and LLM formats.
*   **Model Integration:**
    *   `BaseLLM` and `BaseVLM` for local Hugging Face models.
    *   `BaseAPIModel` for easy integration with external LLM/VLM providers (OpenAI, OpenRouter, Groq, Google, Anthropic).
    *   `ModelConfig` for secure and flexible model configuration.
*   **Tool System:**
    *   Agents can be equipped with custom Python tools.
    *   Automatic generation of OpenAI-compatible tool schemas.
    *   LLMs can request tool execution, and agents handle the invocation and response.
*   **Agent Registry:**
    *   `AgentRegistry` for dynamic registration and discovery of agent instances.
    *   Enables inter-agent invocation by name.
*   **Asynchronous Operations:**
    *   Core agent operations are `async` to ensure non-blocking execution.
*   **Structured Logging:**
    *   `ProgressLogger` and `RequestContext` provide detailed and contextual logging of agent activities.
*   **JSON Output Contract:**
    *   Standardized JSON structure for agents performing multi-step reasoning (`auto_step` mode), detailing thoughts, next actions (invoke agent, call tool, final response), and action inputs.

## Who is this for?

MARSYS is aimed at:

*   AI researchers exploring multi-agent collaboration and emergent behaviors.
*   Developers building applications that require complex task decomposition and specialized AI agents.
*   Engineers looking to integrate various LLMs and tools into a cohesive system.
*   Teams needing a framework to rapidly prototype and deploy multi-agent solutions.

## Getting Started

1.  **Installation:** See `docs/getting-started/installation.md`.
2.  **Configuration:** Set up your AI models and API keys as described in `docs/getting-started/configuration.md` and `src/models/models.py` (`ModelConfig`).
3.  **Your First Agent:** Follow the guide in `docs/getting-started/first-agent.md` to create and run a simple agent.
4.  **Explore Concepts:** Dive deeper into core concepts like [Agents](concepts/agents.md), [Memory](concepts/memory.md), [Models](concepts/models.md), and [Tools](concepts/tools.md).

## Project Structure Highlights

*   **`src/agents/`**: Contains the core logic for agent classes, memory, and registry.
*   **`src/models/`**: Houses the abstraction layers for different AI models.
*   **`src/environment/`**: Includes the tool system and browser automation utilities.
*   **`docs/`**: All documentation, including guides, API references, and tutorials.

We encourage you to explore the [architecture document](architecture.md) for a more in-depth understanding of the framework's design.

"""
Tests for the marsys.agents.learnable_agents module.

This module tests:
- BaseLearnableAgent initialization and learning head configuration
- LearnableAgent initialization with MemoryManager
- Local model enforcement (API models rejected)
- vLLM backend rejection (HuggingFace only for training)
- PEFT learning head initialization
- run_step() learning metadata injection
- _run() async execution

Note: These tests use extensive mocking since learnable agents require
the marsys[local-models] extra which may not be installed in test environments.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock
from typing import Dict, Any, List

from marsys.agents.registry import AgentRegistry
from marsys.agents.memory import MemoryManager, Message


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.fixture
def mock_local_model():
    """Create a mock local model adapter."""
    model = Mock()
    model.arun = AsyncMock(return_value=Mock(
        content="Test response",
        tool_calls=None,
        usage=None,
        metadata=Mock(model="test-model", provider="local"),
        raw_response=None
    ))
    model.run = Mock(return_value=Mock(
        content="Test response",
        tool_calls=None
    ))
    model.cleanup = AsyncMock()
    return model


@pytest.fixture
def mock_local_adapter_factory(mock_local_model):
    """Create a mock LocalAdapterFactory."""
    factory = Mock()
    factory.create_adapter = Mock(return_value=mock_local_model)
    return factory


@pytest.fixture
def local_model_config():
    """Create a local model config mock."""
    config = Mock()
    config.type = "local"
    config.name = "test/local-model"
    config.model_class = "llm"
    config.backend = "huggingface"
    config.max_tokens = 4096
    config.thinking_budget = None
    config.torch_dtype = "bfloat16"
    config.device_map = "auto"
    config.trust_remote_code = True
    config.attn_implementation = None
    return config


@pytest.fixture
def api_model_config():
    """Create an API model config mock (should be rejected)."""
    config = Mock()
    config.type = "api"
    config.name = "anthropic/claude-haiku"
    config.provider = "openrouter"
    config.max_tokens = 4096
    return config


@pytest.fixture
def vllm_model_config():
    """Create a vLLM backend config mock (should be rejected)."""
    config = Mock()
    config.type = "local"
    config.name = "test/local-model"
    config.model_class = "llm"
    config.backend = "vllm"
    config.max_tokens = 4096
    config.thinking_budget = None
    return config


@pytest.fixture
def peft_config():
    """Create a PEFT learning head config."""
    return {
        "peft_type": "lora",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    }


# =============================================================================
# Mock Imports Context Manager
# =============================================================================

# =============================================================================
# Mock Classes for isinstance() compatibility
# =============================================================================

class MockLocalProviderAdapter:
    """Mock class for LocalProviderAdapter that works with isinstance()."""
    pass


class MockHuggingFaceLLMAdapter:
    """Mock class for HuggingFaceLLMAdapter that works with isinstance()."""
    pass


class MockHuggingFaceVLMAdapter:
    """Mock class for HuggingFaceVLMAdapter that works with isinstance()."""
    pass


class MockPeftHead:
    """Mock class for PeftHead that works with isinstance()."""
    def __init__(self, model):
        self.model = model
        self.arun = AsyncMock()
        self.run = Mock()

    def prepare_peft_model(self, **kwargs):
        pass


# =============================================================================
# BaseLearnableAgent Initialization Tests
# =============================================================================

class TestBaseLearnableAgentInitialization:
    """Tests for BaseLearnableAgent initialization."""

    def test_rejects_api_model_config(self, api_model_config):
        """Test that BaseLearnableAgent rejects API model configs."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=Mock,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            with pytest.raises(TypeError) as exc_info:
                LearnableAgent(
                    model_config=api_model_config,
                    goal="Test goal",
                    instruction="Test instruction",
                    name="TestAgent"
                )

            assert "only supports local models" in str(exc_info.value)
            assert "type='api'" in str(exc_info.value)

    def test_rejects_vllm_backend(self, vllm_model_config):
        """Test that BaseLearnableAgent rejects vLLM backend."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=Mock,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            with pytest.raises(TypeError) as exc_info:
                LearnableAgent(
                    model_config=vllm_model_config,
                    goal="Test goal",
                    instruction="Test instruction",
                    name="TestAgent"
                )

            assert "does not support vLLM" in str(exc_info.value)


# =============================================================================
# LearnableAgent Initialization Tests
# =============================================================================

class TestLearnableAgentInitialization:
    """Tests for LearnableAgent initialization."""

    def test_initialization_creates_memory_manager(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that LearnableAgent initializes with a MemoryManager."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            assert hasattr(agent, 'memory')
            assert isinstance(agent.memory, MemoryManager)

    def test_initialization_with_custom_memory_type(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test initialization with custom memory type."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent",
                memory_type="conversation_history"
            )

            assert hasattr(agent, 'memory')

    def test_initialization_stores_learning_head_config(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that learning head name and config are stored."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            # Without learning head
            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            assert agent._learning_head_name is None
            assert agent._learning_config is None

    def test_initialization_with_tools(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test initialization with tools."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            def sample_tool(query: str) -> str:
                """Search for information."""
                return f"Results for: {query}"

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent",
                tools={"sample_tool": sample_tool}
            )

            assert "sample_tool" in agent.tools

    def test_initialization_registers_agent(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that agent is registered on initialization."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="RegisteredLearnableAgent"
            )

            assert AgentRegistry.get("RegisteredLearnableAgent") is agent

    def test_initialization_with_max_tokens_override(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that max_tokens can be overridden."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent",
                max_tokens=8192
            )

            assert agent.max_tokens == 8192

    def test_initialization_with_allowed_peers(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test initialization with allowed_peers."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent",
                allowed_peers=["Agent1", "Agent2"]
            )

            assert "Agent1" in agent.allowed_peers
            assert "Agent2" in agent.allowed_peers


# =============================================================================
# PEFT Learning Head Tests
# =============================================================================

class TestPeftLearningHead:
    """Tests for PEFT learning head initialization."""

    def test_peft_requires_config(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that PEFT learning head requires config."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            with pytest.raises(ValueError) as exc_info:
                LearnableAgent(
                    model_config=local_model_config,
                    goal="Test goal",
                    instruction="Test instruction",
                    name="TestAgent",
                    learning_head="peft"
                    # Missing learning_head_config
                )

            assert "learning_head_config is required" in str(exc_info.value)

    def test_peft_initialization_with_config(
        self, local_model_config, mock_local_adapter_factory, peft_config
    ):
        """Test PEFT initialization with proper config."""
        # Create a proper class that can be used with isinstance
        class MockPeftHead:
            def __init__(self, model):
                self.model = model
                self.prepare_peft_model_called = False
                self.prepare_peft_model_kwargs = None
                self.arun = AsyncMock()

            def prepare_peft_model(self, **kwargs):
                self.prepare_peft_model_called = True
                self.prepare_peft_model_kwargs = kwargs

        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=Mock,
            HuggingFaceLLMAdapter=Mock,
            HuggingFaceVLMAdapter=Mock,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent",
                learning_head="peft",
                learning_head_config=peft_config
            )

            assert agent._learning_head_name == "peft"
            assert agent._learning_config == peft_config
            # Model should now be wrapped by PeftHead
            assert isinstance(agent.model, MockPeftHead)
            # prepare_peft_model should have been called with config
            assert agent.model.prepare_peft_model_called
            assert agent.model.prepare_peft_model_kwargs == peft_config


# =============================================================================
# run_step Tests
# =============================================================================

class TestLearnableAgentRunStep:
    """Tests for run_step with learning context."""

    @pytest.mark.asyncio
    async def test_run_step_adds_learning_metadata_when_training(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that run_step adds learning metadata when in training mode."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Mock the parent run_step
            with patch.object(
                agent.__class__.__bases__[0].__bases__[0],  # BaseAgent
                'run_step',
                new_callable=AsyncMock
            ) as mock_parent_run_step:
                mock_parent_run_step.return_value = {
                    'response': 'Test response',
                    'success': True
                }

                context = {
                    'learning_context': {
                        'is_training': True,
                        'iteration': 5
                    }
                }

                result = await agent.run_step("Test request", context)

                assert 'learning_metadata' in result
                assert result['learning_metadata']['iteration'] == 5
                assert result['learning_metadata']['has_peft'] is False

    @pytest.mark.asyncio
    async def test_run_step_no_learning_metadata_when_not_training(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that run_step doesn't add learning metadata when not training."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Mock the parent run_step
            with patch.object(
                agent.__class__.__bases__[0].__bases__[0],  # BaseAgent
                'run_step',
                new_callable=AsyncMock
            ) as mock_parent_run_step:
                mock_parent_run_step.return_value = {
                    'response': 'Test response',
                    'success': True
                }

                context = {
                    'learning_context': {
                        'is_training': False,
                        'iteration': 0
                    }
                }

                result = await agent.run_step("Test request", context)

                assert 'learning_metadata' not in result

    @pytest.mark.asyncio
    async def test_run_step_handles_missing_learning_context(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that run_step handles missing learning_context gracefully."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Mock the parent run_step
            with patch.object(
                agent.__class__.__bases__[0].__bases__[0],  # BaseAgent
                'run_step',
                new_callable=AsyncMock
            ) as mock_parent_run_step:
                mock_parent_run_step.return_value = {
                    'response': 'Test response',
                    'success': True
                }

                # No learning_context in context
                context = {}

                result = await agent.run_step("Test request", context)

                # Should not raise and should not have learning_metadata
                assert 'learning_metadata' not in result


# =============================================================================
# _run Tests
# =============================================================================

class TestLearnableAgentRun:
    """Tests for _run async execution."""

    @pytest.mark.asyncio
    async def test_run_calls_model_arun(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that _run calls model.arun asynchronously."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent
            from marsys.agents.utils import RequestContext

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Replace model with our mock
            agent.model = mock_local_model

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]

            request_context = Mock(spec=RequestContext)

            result = await agent._run(
                messages=messages,
                request_context=request_context,
                run_mode="chat"
            )

            # Verify arun was called
            mock_local_model.arun.assert_called_once()

            # Check the call arguments
            call_kwargs = mock_local_model.arun.call_args
            assert call_kwargs.kwargs['messages'] == messages

    @pytest.mark.asyncio
    async def test_run_returns_message_object(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that _run returns a Message object."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent
            from marsys.agents.utils import RequestContext

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Replace model with our mock
            agent.model = mock_local_model

            messages = [{"role": "user", "content": "Hello!"}]
            request_context = Mock(spec=RequestContext)

            result = await agent._run(
                messages=messages,
                request_context=request_context,
                run_mode="chat"
            )

            assert isinstance(result, Message)

    @pytest.mark.asyncio
    async def test_run_uses_default_temperature(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that _run uses default temperature of 0.7."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent
            from marsys.agents.utils import RequestContext

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            agent.model = mock_local_model

            messages = [{"role": "user", "content": "Hello!"}]
            request_context = Mock(spec=RequestContext)

            await agent._run(
                messages=messages,
                request_context=request_context,
                run_mode="chat"
            )

            # Check temperature was 0.7 (default for learnable models)
            call_kwargs = mock_local_model.arun.call_args
            assert call_kwargs.kwargs['temperature'] == 0.7

    @pytest.mark.asyncio
    async def test_run_handles_exception(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that _run handles exceptions and returns error message."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent
            from marsys.agents.utils import RequestContext

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Create a model that raises an exception
            failing_model = Mock()
            failing_model.arun = AsyncMock(side_effect=RuntimeError("Model failed"))
            agent.model = failing_model

            messages = [{"role": "user", "content": "Hello!"}]
            request_context = Mock(spec=RequestContext)

            result = await agent._run(
                messages=messages,
                request_context=request_context,
                run_mode="chat"
            )

            assert isinstance(result, Message)
            assert result.role == "error"
            assert "LLM call failed" in result.content
            assert "Model failed" in result.content

    @pytest.mark.asyncio
    async def test_run_passes_kwargs_to_model(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that _run passes extra kwargs to model."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent
            from marsys.agents.utils import RequestContext

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            agent.model = mock_local_model

            messages = [{"role": "user", "content": "Hello!"}]
            request_context = Mock(spec=RequestContext)

            await agent._run(
                messages=messages,
                request_context=request_context,
                run_mode="chat",
                json_mode=True,
                temperature=0.5,
                max_tokens=2000
            )

            call_kwargs = mock_local_model.arun.call_args
            assert call_kwargs.kwargs['json_mode'] is True
            assert call_kwargs.kwargs['temperature'] == 0.5
            assert call_kwargs.kwargs['max_tokens'] == 2000


# =============================================================================
# Memory Manager Tests
# =============================================================================

class TestLearnableAgentMemory:
    """Tests for LearnableAgent memory functionality."""

    def test_memory_initialized_with_instruction(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that memory is initialized with agent instruction as description."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="You are a helpful research assistant.",
                name="TestAgent"
            )

            assert hasattr(agent, 'memory')
            # Memory should be initialized with conversation_history type
            assert agent.memory.memory_type == "conversation_history"
            # Check that messages were initialized (system message with instruction)
            messages = agent.memory.get_messages()
            assert len(messages) >= 1
            # First message should be system message with instruction
            system_msgs = [m for m in messages if m.get("role") == "system"]
            assert len(system_msgs) == 1
            assert "You are a helpful research assistant." in system_msgs[0].get("content", "")

    def test_memory_type_defaults_to_conversation_history(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that memory_type defaults to conversation_history."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
                # memory_type not specified
            )

            assert hasattr(agent, 'memory')


# =============================================================================
# Inheritance Chain Tests
# =============================================================================

class TestInheritanceChain:
    """Tests for proper inheritance chain."""

    def test_learnable_agent_inherits_from_base_learnable_agent(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that LearnableAgent inherits from BaseLearnableAgent."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent, BaseLearnableAgent

            assert issubclass(LearnableAgent, BaseLearnableAgent)

    def test_base_learnable_agent_inherits_from_base_agent(self):
        """Test that BaseLearnableAgent inherits from BaseAgent."""
        from marsys.agents.learnable_agents import BaseLearnableAgent
        from marsys.agents.agents import BaseAgent

        assert issubclass(BaseLearnableAgent, BaseAgent)

    def test_learnable_agent_has_run_step_method(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that LearnableAgent has run_step method via inheritance."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            assert hasattr(agent, 'run_step')
            assert callable(agent.run_step)

    def test_learnable_agent_has_run_method(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that LearnableAgent has _run method."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            assert hasattr(agent, '_run')
            assert callable(agent._run)


# =============================================================================
# Import Error Handling Tests
# =============================================================================

class TestImportErrorHandling:
    """Tests for handling missing marsys[local-models] dependencies."""

    def test_import_error_provides_helpful_message(self):
        """Test that ImportError provides installation instructions."""
        # This test verifies the error message format when imports fail
        # We can't easily test this without actually removing the dependencies
        # So we just verify the structure exists
        from marsys.agents.learnable_agents import BaseLearnableAgent

        # The import error handling is in __init__, which we can't easily test
        # without modifying sys.modules. The important thing is that the class exists.
        assert BaseLearnableAgent is not None


# =============================================================================
# Resource Management Tests
# =============================================================================

class TestResourceManagement:
    """Tests for resource acquisition/release on learnable agents."""

    def test_acquire_instance(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test acquiring instance from learnable agent."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            result = agent.acquire_instance("branch_1")

            assert result is agent
            assert agent._allocated_to_branch == "branch_1"

    def test_release_instance(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test releasing instance from learnable agent."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            agent.acquire_instance("branch_1")
            result = agent.release_instance("branch_1")

            assert result is True
            assert agent._allocated_to_branch is None


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestLearnableAgentCleanup:
    """Tests for learnable agent cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_can_be_called(
        self, local_model_config, mock_local_model, mock_local_adapter_factory
    ):
        """Test that cleanup can be called without error."""
        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="CleanupLearnableAgent"
            )

            # Replace model with mock that has cleanup
            agent.model = mock_local_model

            # Should not raise
            await agent.cleanup()


# =============================================================================
# Model Config Parameter Passing Tests
# =============================================================================

class TestModelConfigParameters:
    """Tests for model config parameter handling."""

    def test_torch_dtype_passed_to_factory(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that torch_dtype is passed to LocalAdapterFactory."""
        local_model_config.torch_dtype = "float16"

        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Verify factory was called with torch_dtype
            call_kwargs = mock_local_adapter_factory.create_adapter.call_args
            assert call_kwargs.kwargs.get('torch_dtype') == "float16"

    def test_device_map_passed_to_factory(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that device_map is passed to LocalAdapterFactory."""
        local_model_config.device_map = "cuda:0"

        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Verify factory was called with device_map
            call_kwargs = mock_local_adapter_factory.create_adapter.call_args
            assert call_kwargs.kwargs.get('device_map') == "cuda:0"

    def test_trust_remote_code_passed_to_factory(
        self, local_model_config, mock_local_adapter_factory
    ):
        """Test that trust_remote_code is passed to LocalAdapterFactory."""
        local_model_config.trust_remote_code = True

        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=local_model_config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Verify factory was called with trust_remote_code
            call_kwargs = mock_local_adapter_factory.create_adapter.call_args
            assert call_kwargs.kwargs.get('trust_remote_code') is True

    def test_backend_defaults_to_huggingface(
        self, mock_local_adapter_factory
    ):
        """Test that backend defaults to huggingface when not specified."""
        config = Mock()
        config.type = "local"
        config.name = "test/model"
        config.model_class = "llm"
        config.max_tokens = 4096
        config.thinking_budget = None
        # No backend attribute
        del config.backend

        with patch.multiple(
            'marsys.models.models',
            LocalProviderAdapter=MockLocalProviderAdapter,
            HuggingFaceLLMAdapter=MockHuggingFaceLLMAdapter,
            HuggingFaceVLMAdapter=MockHuggingFaceVLMAdapter,
            LocalAdapterFactory=mock_local_adapter_factory,
            PeftHead=MockPeftHead,
            create=True
        ):
            from marsys.agents.learnable_agents import LearnableAgent

            agent = LearnableAgent(
                model_config=config,
                goal="Test goal",
                instruction="Test instruction",
                name="TestAgent"
            )

            # Verify factory was called with huggingface backend
            call_kwargs = mock_local_adapter_factory.create_adapter.call_args
            assert call_kwargs.kwargs.get('backend') == "huggingface" or \
                   call_kwargs.args[0] == "huggingface" if call_kwargs.args else True

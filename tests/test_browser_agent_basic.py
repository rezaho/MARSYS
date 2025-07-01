import asyncio
import os
from pathlib import Path

import pytest

from src.agents.browser_agent import BrowserAgent
from src.models.models import ModelConfig


@pytest.fixture
async def browser_agent():
    """Create a BrowserAgent instance for testing."""
    config = ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
    )

    agent = await BrowserAgent.create(
        model_config=config, headless=True, temp_dir="./tmp/test_browser"
    )

    yield agent

    # Cleanup
    await agent.cleanup()


@pytest.mark.asyncio
async def test_browser_agent_initialization(browser_agent):
    """Test that BrowserAgent initializes correctly."""
    assert browser_agent is not None
    assert browser_agent.browser_tool is not None
    assert len(browser_agent.tools) > 0
    assert len(browser_agent.tools_schema) > 0

    # Check that web tools are included
    assert "clean_and_extract_html" in browser_agent.tools
    assert "read_file" in browser_agent.tools


@pytest.mark.asyncio
async def test_browser_navigation(browser_agent):
    """Test basic browser navigation."""
    # Navigate to example.org
    await browser_agent.browser_tool.goto("https://example.org")

    # Take screenshot
    screenshot_path = await browser_agent._take_step_screenshot(highlight=True)
    assert screenshot_path is not None
    assert os.path.exists(screenshot_path)

    # Check memory
    assert len(browser_agent.memory.memories) >= 1


@pytest.mark.asyncio
async def test_browser_agent_auto_run():
    """Test BrowserAgent auto_run with a simple task."""
    config = ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
    )

    agent = await BrowserAgent.create_safe(
        model_config=config, headless=True, timeout=30
    )

    try:
        result = await agent.auto_run(
            "Navigate to example.org and tell me what the main heading says",
            max_steps=5,
        )

        assert result is not None
        assert isinstance(result, (dict, str))

        if isinstance(result, dict):
            assert "text" in result or "final_response" in result
            assert "metadata" in result

    finally:
        await agent.cleanup()


@pytest.mark.asyncio
async def test_screenshot_persistence():
    """Test that screenshots are properly stored in memory."""
    config = ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
    )

    agent = await BrowserAgent.create(model_config=config, headless=True)

    try:
        # Navigate and take screenshot
        await agent.browser_tool.goto("https://example.org")
        screenshot_path = await agent._take_step_screenshot(highlight=True)

        # Check that screenshot path is in memory metadata
        screenshot_paths = agent._get_screenshot_paths()
        assert len(screenshot_paths) > 0
        assert screenshot_path in screenshot_paths

    finally:
        await agent.cleanup()

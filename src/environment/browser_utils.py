import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def is_playwright_installed() -> bool:
    """Check if Playwright browsers are installed."""
    try:
        # Check for Playwright browsers path
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        if not browsers_path:
            # Default location varies by OS
            if platform.system() == "Windows":
                browsers_path = Path.home() / "AppData" / "Local" / "ms-playwright"
            else:
                browsers_path = Path.home() / ".cache" / "ms-playwright"

        browsers_path = Path(browsers_path)

        # Check if chromium executable exists
        chromium_paths = [
            browsers_path / "chromium-*" / "chrome-linux" / "chrome",
            browsers_path / "chromium-*" / "chrome-mac" / "Chromium.app",
            browsers_path / "chromium-*" / "chrome-win" / "chrome.exe",
        ]

        for pattern in chromium_paths:
            if any(browsers_path.glob(str(pattern).split("/")[-1])):
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking Playwright installation: {e}")
        return False


def ensure_playwright() -> None:
    """Ensure Playwright and browser binaries are installed."""
    if is_playwright_installed():
        logger.info("Playwright browsers are already installed.")
        return

    logger.info("Installing Playwright browsers...")
    try:
        # First ensure playwright is installed
        subprocess.run(
            ["pip", "install", "playwright"], check=True, capture_output=True, text=True
        )

        # Then install browsers
        result = subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Playwright browsers installed successfully.")
        logger.debug(f"Installation output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Playwright browsers: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(
            "Failed to install Playwright browsers. "
            "Please run 'playwright install chromium' manually."
        ) from e
    except FileNotFoundError:
        logger.error("Playwright command not found. Please install playwright first.")
        raise RuntimeError(
            "Playwright not found. Please run 'pip install playwright' first."
        )


def get_browser_config(
    headless: bool = True, viewport: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Get optimal browser configuration based on environment."""
    config = {
        "headless": headless,
        "args": (
            [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--single-process",
                "--disable-gpu",
            ]
            if headless
            else []
        ),
    }

    if viewport:
        config["viewport"] = viewport
    else:
        config["viewport"] = {"width": 1280, "height": 720}

    # Add user agent to avoid detection
    config["user_agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    return config

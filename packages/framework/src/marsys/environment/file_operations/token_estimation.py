"""
Token estimation utilities for text and images.

This module provides utilities to estimate token counts for vision-language models (VLMs)
based on character counts (for text) and pixel dimensions (for images).

Supports provider-specific estimation for: OpenAI (GPT), Anthropic (Claude), Google (Gemini), xAI (Grok)
"""

import logging
import math
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ====== Text Token Estimation ======

def estimate_text_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate number of tokens for text content.

    Args:
        text: Text content
        chars_per_token: Average characters per token (default 4.0)
            - English: ~4 chars/token
            - Code: ~3.5 chars/token
            - Dense technical text: ~4.5 chars/token

    Returns:
        Estimated token count
    """
    return max(1, int(len(text) / chars_per_token))


# ====== Image Token Estimation ======

def estimate_image_tokens(
    width: int,
    height: int,
    provider: str = "generic",
    detail: str = "high"
) -> int:
    """
    Estimate number of tokens for an image based on pixel dimensions.

    Different providers tokenize images differently based on resolution and tiling strategy.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        provider: Provider type ("openai", "anthropic", "google", "xai", "generic")
        detail: Detail level ("high" or "low") - affects some providers

    Returns:
        Estimated token count for the image

    Provider-Specific Formulas:
        - OpenAI (GPT): 512x512 tiles, 170 tokens/tile + 85 base tokens
        - Anthropic (Claude): (width * height) / 750 tokens
        - Google (Gemini): 768x768 tiles, 258 tokens/tile
        - xAI (Grok): 448x448 tiles, 256 tokens/tile, (tiles + 1) * 256
    """
    provider = provider.lower()

    if provider in ["openai", "gpt"]:
        return _estimate_openai_tokens(width, height, detail)
    elif provider in ["anthropic", "claude"]:
        return _estimate_anthropic_tokens(width, height)
    elif provider in ["google", "gemini"]:
        return _estimate_google_tokens(width, height)
    elif provider in ["xai", "grok"]:
        return _estimate_xai_tokens(width, height, detail)
    else:
        # Generic conservative estimate
        return _estimate_generic_tokens(width, height)


def _estimate_openai_tokens(width: int, height: int, detail: str = "high") -> int:
    """
    Estimate tokens for OpenAI GPT vision models.

    Formula: total_tokens = 85 + 170 * n
    Where n = number of 512x512 tiles

    Process:
    1. If detail="low", return 85 tokens (fixed)
    2. Resize image if > 2048px on any side
    3. Scale shortest side to 768px if needed
    4. Divide into 512x512 tiles
    5. Calculate: 85 + (170 * num_tiles)

    Args:
        width: Image width in pixels
        height: Image height in pixels
        detail: "high" or "low" detail mode

    Returns:
        Estimated token count
    """
    if detail == "low":
        return 85

    # Step 1: Resize if > 2048px on any side
    if width > 2048 or height > 2048:
        if width > height:
            height = int(height * (2048 / width))
            width = 2048
        else:
            width = int(width * (2048 / height))
            height = 2048

    # Step 2: Scale shortest side to 768px if needed
    if min(width, height) > 768:
        if width < height:
            height = int(height * (768 / width))
            width = 768
        else:
            width = int(width * (768 / height))
            height = 768

    # Step 3: Calculate number of 512x512 tiles (round up)
    tiles_width = math.ceil(width / 512)
    tiles_height = math.ceil(height / 512)
    num_tiles = tiles_width * tiles_height

    # Step 4: Calculate total tokens
    return 85 + (170 * num_tiles)


def _estimate_anthropic_tokens(width: int, height: int) -> int:
    """
    Estimate tokens for Anthropic Claude vision models.

    Formula: tokens = (width * height) / 750

    Notes:
    - Images > 1568px on long edge are scaled down (preserving aspect ratio)
    - Images > 8000px on any edge are rejected (but we still estimate)
    - Very small images < 200px may degrade performance

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count
    """
    # Apply scaling if needed (long edge > 1568px)
    if max(width, height) > 1568:
        if width > height:
            height = int(height * (1568 / width))
            width = 1568
        else:
            width = int(width * (1568 / height))
            height = 1568

    # Calculate tokens: (width * height) / 750
    tokens = (width * height) / 750

    return max(1, int(tokens))


def _estimate_google_tokens(width: int, height: int) -> int:
    """
    Estimate tokens for Google Gemini vision models.

    Formula:
    - If width <= 384 AND height <= 384: 258 tokens
    - Otherwise: divide into 768x768 tiles, 258 tokens per tile

    Tile calculation:
    - tiles_width = ceil(width / 768)
    - tiles_height = ceil(height / 768)
    - total_tokens = tiles_width * tiles_height * 258

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count
    """
    # Small images: fixed 258 tokens
    if width <= 384 and height <= 384:
        return 258

    # Large images: tile-based calculation
    tiles_width = math.ceil(width / 768)
    tiles_height = math.ceil(height / 768)
    num_tiles = tiles_width * tiles_height

    return num_tiles * 258


def _estimate_xai_tokens(width: int, height: int, detail: str = "high") -> int:
    """
    Estimate tokens for xAI Grok vision models.

    Formula: (num_tiles + 1) * 256
    Where tiles are 448x448 pixels

    Notes:
    - Each tile is 448x448 pixels
    - Each tile = 256 tokens
    - Final generation includes an extra tile
    - Maximum 6 tiles (< 1,792 tokens per image)
    - detail="low" processes low-res version (fewer tokens)

    Args:
        width: Image width in pixels
        height: Image height in pixels
        detail: "high" or "low" detail mode

    Returns:
        Estimated token count
    """
    if detail == "low":
        # Low detail mode: single tile + 1
        return 2 * 256  # 512 tokens

    # Calculate number of 448x448 tiles (round up)
    tiles_width = math.ceil(width / 448)
    tiles_height = math.ceil(height / 448)
    num_tiles = tiles_width * tiles_height

    # Cap at 6 tiles maximum
    num_tiles = min(num_tiles, 6)

    # Formula: (tiles + 1) * 256
    return (num_tiles + 1) * 256


def _estimate_generic_tokens(width: int, height: int) -> int:
    """
    Generic conservative token estimation for unknown providers.

    Uses a middle-ground approach based on common VLM patterns:
    - Assumes tile-based processing
    - Conservative estimate to avoid underestimation

    Formula: (pixels / 750) tokens (similar to Claude)
    This provides a reasonable estimate across different providers.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count
    """
    total_pixels = width * height
    # Conservative estimate: similar to Claude's formula
    return max(256, int(total_pixels / 750))


# ====== Helper Functions ======

def get_optimal_image_dimensions(
    target_tokens: int,
    aspect_ratio: Optional[Tuple[int, int]] = None,
    provider: str = "generic"
) -> Tuple[int, int]:
    """
    Calculate optimal image dimensions for a target token budget.

    Args:
        target_tokens: Desired maximum token count
        aspect_ratio: Optional (width, height) ratio to maintain
        provider: Provider type for estimation

    Returns:
        Tuple of (width, height) in pixels

    Example:
        >>> get_optimal_image_dimensions(1000, aspect_ratio=(16, 9), provider="openai")
        (1024, 576)  # Approximately 1000 tokens, 16:9 ratio
    """
    if aspect_ratio is None:
        aspect_ratio = (1, 1)  # Square by default

    aspect_width, aspect_height = aspect_ratio
    ratio = aspect_width / aspect_height

    # Binary search for optimal dimensions
    low, high = 128, 4096
    best_width = 512
    best_height = 512

    for _ in range(20):  # Max 20 iterations
        mid = (low + high) // 2
        width = mid
        height = int(mid / ratio)

        estimated = estimate_image_tokens(width, height, provider)

        if abs(estimated - target_tokens) < 50:  # Within 50 tokens is good enough
            best_width = width
            best_height = height
            break
        elif estimated < target_tokens:
            low = mid + 1
            best_width = width
            best_height = height
        else:
            high = mid - 1

    return best_width, best_height


def should_downsample_image(
    width: int,
    height: int,
    max_pixels: int,
    max_tokens: Optional[int] = None,
    provider: str = "generic"
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Determine if an image should be downsampled and calculate target dimensions.

    Args:
        width: Current width in pixels
        height: Current height in pixels
        max_pixels: Maximum allowed total pixels
        max_tokens: Optional maximum allowed tokens
        provider: Provider type for token estimation

    Returns:
        Tuple of (should_downsample, target_dimensions)
        - should_downsample: True if image exceeds limits
        - target_dimensions: (width, height) if downsampling needed, None otherwise
    """
    current_pixels = width * height

    # Check pixel limit
    exceeds_pixels = current_pixels > max_pixels

    # Check token limit if specified
    exceeds_tokens = False
    if max_tokens is not None:
        estimated_tokens = estimate_image_tokens(width, height, provider)
        exceeds_tokens = estimated_tokens > max_tokens

    # If neither limit exceeded, no downsampling needed
    if not exceeds_pixels and not exceeds_tokens:
        return False, None

    # Calculate target dimensions maintaining aspect ratio
    aspect_ratio = width / height

    # Determine limiting factor
    if max_tokens is not None and exceeds_tokens:
        # Use token budget if specified and exceeded
        target_width, target_height = get_optimal_image_dimensions(
            max_tokens,
            aspect_ratio=(width, height),
            provider=provider
        )
    else:
        # Use pixel budget
        scale_factor = (max_pixels / current_pixels) ** 0.5
        target_width = int(width * scale_factor)
        target_height = int(height * scale_factor)

    return True, (target_width, target_height)


# ====== Combined Estimation ======

def estimate_total_tokens(
    text_content: str,
    images: Optional[list] = None,
    provider: str = "generic"
) -> dict:
    """
    Estimate total tokens for text + images content.

    Args:
        text_content: Text content string
        images: List of (width, height) tuples for images, or list of dicts with 'width', 'height', 'detail'
        provider: Provider type for image token estimation

    Returns:
        Dictionary with breakdown:
        {
            "text_tokens": int,
            "image_tokens": int,
            "total_tokens": int,
            "text_chars": int,
            "total_pixels": int,
            "image_count": int,
            "images": [{"width": int, "height": int, "tokens": int}, ...]
        }
    """
    text_tokens = estimate_text_tokens(text_content)
    image_tokens = 0
    total_pixels = 0
    image_details = []

    if images:
        for img in images:
            # Support both tuple format and dict format
            if isinstance(img, dict):
                width = img['width']
                height = img['height']
                detail = img.get('detail', 'high')
            else:
                width, height = img
                detail = 'high'

            img_tokens = estimate_image_tokens(width, height, provider, detail)
            image_tokens += img_tokens
            total_pixels += width * height

            image_details.append({
                "width": width,
                "height": height,
                "tokens": img_tokens,
                "detail": detail
            })

    return {
        "text_tokens": text_tokens,
        "image_tokens": image_tokens,
        "total_tokens": text_tokens + image_tokens,
        "text_chars": len(text_content),
        "total_pixels": total_pixels,
        "image_count": len(images) if images else 0,
        "images": image_details
    }

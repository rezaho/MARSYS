## Description of function calling definitions for each method in the class
import asyncio
import io
import json
import logging
import math
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from playwright.async_api import BrowserContext, Page, async_playwright

# Import BeautifulSoup and markdownify if available
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    BeautifulSoup = None
    Comment = None

try:
    import markdownify
except ImportError:
    markdownify = None
import json
import logging
import re
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup, Comment
import markdownify

# Build the absolute path to the assets directory (assuming the repo structure provided)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, "..", "assets")

logger = logging.getLogger(__name__)
##############################################
###   Helper Functions for Browser Tool   ###
##############################################


# Helper function to check rectangle overlap.
def rects_overlap(
    rect1: Tuple[float, float, float, float], rect2: Tuple[float, float, float, float]
) -> bool:
    """
    Check if two rectangles overlap.

    Each rectangle is represented as a tuple (x1, y1, x2, y2).

    Parameters:
        rect1 (tuple): (x1, y1, x2, y2) for the first rectangle.
        rect2 (tuple): (x1, y1, x2, y2) for the second rectangle.

    Returns:
        bool: True if the rectangles overlap, False otherwise.
    """
    # Each rect is (x1, y1, x2, y2)
    return not (
        rect1[2] <= rect2[0]
        or rect1[0] >= rect2[2]
        or rect1[3] <= rect2[1]
        or rect1[1] >= rect2[3]
    )


def find_label_position_with_masking(
    bbox: Tuple[int, int, int, int],
    text_width: int,
    text_height: int,
    image_width: int,
    image_height: int,
    global_mask: 'np.ndarray',
    padding: int = 10,
    label_margin: int = 5
) -> Tuple[int, int, 'np.ndarray']:
    """
    Find optimal label position using masking-based algorithm.
    
    This algorithm creates a mask-based approach to find the best position for labels:
    1. Creates a hollow rectangle mask around the current bbox with padding
    2. Intersects with global occupancy mask to find valid placement areas
    3. Searches systematically for the optimal label position with proper spacing
    4. Updates global mask to mark the chosen label area as occupied
    
    Args:
        bbox: Current element bounding box (x1, y1, x2, y2)
        text_width: Width of the label text
        text_height: Height of the label text  
        image_width: Width of the image
        image_height: Height of the image
        global_mask: Occupancy mask (1=available, 0=occupied)
        padding: Padding around bbox for label placement search
        label_margin: Minimum spacing between label and bbox border (default: 5px)
        
    Returns:
        Tuple of (label_x, label_y, updated_global_mask)
    """
    import numpy as np
    
    x1, y1, x2, y2 = bbox
    
    # Ensure the mask is the correct size
    if global_mask.shape != (image_height, image_width):
        raise ValueError(f"Mask shape {global_mask.shape} doesn't match image size ({image_height}, {image_width})")
    
    # Create hollow rectangle mask around the bbox with padding
    # This represents the search area for label placement
    search_x1 = max(0, x1 - padding)
    search_y1 = max(0, y1 - padding)
    search_x2 = min(image_width, x2 + padding)
    search_y2 = min(image_height, y2 + padding)
    
    # Create the hollow rectangle mask (1=valid for label, 0=invalid)
    hollow_mask = np.ones((image_height, image_width), dtype=np.uint8)
    
    # Set the bbox area to 0 (invalid for label placement)
    hollow_mask[y1:y2, x1:x2] = 0
    
    # Only consider the search area around the bbox
    search_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    search_mask[search_y1:search_y2, search_x1:search_x2] = 1
    
    # Combine hollow mask with search area
    hollow_mask = hollow_mask & search_mask
    
    # Intersect with global occupancy mask to find actually available areas
    available_mask = global_mask & hollow_mask
    
    # Define search priority order: above, below, left, right of bbox
    search_positions = []
    
    # Above the bbox (label bottom should be at least label_margin pixels above bbox top)
    above_y_end = max(0, y1 - text_height - label_margin)
    if above_y_end >= search_y1:
        for y in range(above_y_end, max(0, y1 - text_height - padding), -1):
            for x in range(max(0, x1 - padding), min(image_width - text_width, x2 + padding)):
                search_positions.append((x, y, 'above'))
    
    # Below the bbox (label top should be at least label_margin pixels below bbox bottom)
    below_y_start = min(image_height - text_height, y2 + label_margin)
    if below_y_start < search_y2 - text_height:
        for y in range(below_y_start, min(image_height - text_height, search_y2)):
            for x in range(max(0, x1 - padding), min(image_width - text_width, x2 + padding)):
                search_positions.append((x, y, 'below'))
    
    # Left of the bbox (label right should be at least label_margin pixels left of bbox left)
    left_x_end = max(0, x1 - text_width - label_margin)
    if left_x_end >= search_x1:
        for x in range(left_x_end, max(0, x1 - text_width - padding), -1):
            for y in range(max(0, y1 - padding), min(image_height - text_height, y2 + padding)):
                search_positions.append((x, y, 'left'))
    
    # Right of the bbox (label left should be at least label_margin pixels right of bbox right)
    right_x_start = min(image_width - text_width, x2 + label_margin)
    if right_x_start < search_x2 - text_width:
        for x in range(right_x_start, min(image_width - text_width, search_x2)):
            for y in range(max(0, y1 - padding), min(image_height - text_height, y2 + padding)):
                search_positions.append((x, y, 'right'))
    
    # Find the best position where the label rectangle fits entirely in available area
    best_position = None
    
    for label_x, label_y, direction in search_positions:
        # Check bounds
        if (label_x < 0 or label_y < 0 or 
            label_x + text_width > image_width or 
            label_y + text_height > image_height):
            continue
            
        # Check if the entire label rectangle fits in available area
        label_region = available_mask[label_y:label_y + text_height, label_x:label_x + text_width]
        
        if label_region.shape == (text_height, text_width) and np.all(label_region == 1):
            best_position = (label_x, label_y)
            break
    
    # Fallback: if no perfect position found, use a position near the bbox
    if best_position is None:
        # Try simple fallback positions with proper spacing
        fallback_positions = [
            (x1, max(0, y1 - text_height - label_margin)),  # Above with margin
            (x1, min(image_height - text_height, y2 + label_margin)),  # Below with margin
            (max(0, x1 - text_width - label_margin), y1),  # Left with margin
            (min(image_width - text_width, x2 + label_margin), y1),  # Right with margin
            (x1, y1)  # Last resort: overlap with bbox
        ]
        
        for fallback_x, fallback_y in fallback_positions:
            if (0 <= fallback_x <= image_width - text_width and 
                0 <= fallback_y <= image_height - text_height):
                best_position = (fallback_x, fallback_y)
                break
    
    # Use bbox position as absolute fallback
    if best_position is None:
        best_position = (x1, y1)
    
    label_x, label_y = best_position
    
    # Update global mask to mark label area as occupied
    updated_mask = global_mask.copy()
    
    # Ensure coordinates are within bounds before updating mask
    safe_x = max(0, min(label_x, image_width - 1))
    safe_y = max(0, min(label_y, image_height - 1))
    safe_x2 = max(safe_x, min(label_x + text_width, image_width))
    safe_y2 = max(safe_y, min(label_y + text_height, image_height))
    
    updated_mask[safe_y:safe_y2, safe_x:safe_x2] = 0
    
    return (label_x, label_y, updated_mask)


# Legacy helper function to find a valid label position.
def find_label_position(
    x: float,
    y: float,
    x2: float,
    y2: float,
    text_width: float,
    text_height: float,
    image_width: float,
    image_height: float,
    current_box: Tuple[float, float, float, float],
    all_boxes: List[Tuple[float, float, float, float]],
    placed_labels: List[Tuple[float, float, float, float]],
    margin: int = 5,
    max_adjust: int = 5,
) -> Tuple[float, float]:
    """
    Find a valid position for a label relative to a bounding box.

    The function attempts to place the label (with dimensions text_width x text_height)
    near the bounding box defined by (x, y, x2, y2) while maintaining a given margin and
    avoiding overlaps with other bounding boxes (from all_boxes) and already placed label texts
    (from placed_labels). It first checks vertical candidates (above and below the box) and, if necessary,
    tries small adjustments; if vertical placements fail, it then checks horizontal candidates.

    Parameters:
        x (int/float): x-coordinate of the top-left corner of the bounding box.
        y (int/float): y-coordinate of the top-left corner of the bounding box.
        x2 (int/float): x-coordinate of the bottom-right corner of the bounding box.
        y2 (int/float): y-coordinate of the bottom-right corner of the bounding box.
        text_width (int): Width of the label text.
        text_height (int): Height of the label text.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        current_box (tuple): The bounding box of the current element (x, y, x2, y2).
        all_boxes (list): List of bounding boxes for all interactive elements.
        placed_labels (list): List of rectangles representing already placed labels.
        margin (int, optional): Gap (in pixels) to maintain between the label and the bounding box. Default is 5.
        max_adjust (int, optional): Maximum adjustment (in pixels) for trying alternative positions. Default is 20.

    Returns:
        tuple: (label_x, label_y) position for the top-left corner of the label.
    """

    # Compute vertical candidate positions.
    candidate_above = (
        (x, y - text_height - margin) if (y - text_height - margin >= 0) else None
    )
    candidate_below = (
        (x, y2 + margin) if (y2 + margin + text_height <= image_height) else None
    )

    # Function to test a candidate position.
    def candidate_valid(candidate):
        if candidate is None:
            return False
        rect = (
            candidate[0] - margin,
            candidate[1] - margin,
            candidate[0] + text_width + margin,
            candidate[1] + text_height + margin,
        )
        if candidate[1] < 0 or candidate[1] + text_height > image_height:
            return False
        # Check candidate_rect does not overlap any other bounding box.
        for box in all_boxes:
            if box == current_box:
                continue
            if rects_overlap(rect, box):
                return False
        # Also ensure it does not overlap any previously placed label.
        for pl in placed_labels:
            if rects_overlap(rect, pl):
                return False
        return True

    # First, try vertical candidates.
    if candidate_above is not None and candidate_valid(candidate_above):
        return candidate_above
    if candidate_below is not None and candidate_valid(candidate_below):
        return candidate_below

    # Try vertical adjustments on candidate_above.
    if candidate_above is not None:
        for d in range(5, max_adjust + 1, 5):
            cand = (x, y - text_height - margin - d)
            if candidate_valid(cand):
                return cand
        # for d in range(5, max_adjust + 1, 5):
        #     cand = (x, y - text_height - margin + d)
        #     if candidate_valid(cand):
        #         return cand

    # Try vertical adjustments on candidate_below.
    if candidate_below is not None:
        for d in range(5, max_adjust + 1, 5):
            cand = (x, y2 + text_height + margin + d)
            if candidate_valid(cand):
                return cand
        # for d in range(5, max_adjust + 1, 5):
        #     cand = (x, y2 + text_height + margin - d)
        #     if candidate_valid(cand):
        #         return cand

    # If vertical adjustments fail, try horizontal positions.
    candidate_left = (
        (x - text_width - margin, y) if (x - text_width - margin >= 0) else None
    )
    candidate_right = (
        (x2 + margin, y) if (x2 + margin + text_width <= image_width) else None
    )

    if candidate_left is not None and candidate_valid(candidate_left):
        return candidate_left
    if candidate_right is not None and candidate_valid(candidate_right):
        return candidate_right

    # Try horizontal adjustments on candidate_left.
    if candidate_left is not None:
        for d in range(5, max_adjust + 1, 5):
            cand = (x - text_width - margin - d, y)
            if candidate_valid(cand):
                return cand
        for d in range(5, max_adjust + 1, 5):
            cand = (x - text_width - margin + d, y)
            if candidate_valid(cand):
                return cand

    # Try horizontal adjustments on candidate_right.
    if candidate_right is not None:
        for d in range(5, max_adjust + 1, 5):
            cand = (x2 + margin + d, y)
            if candidate_valid(cand):
                return cand
        for d in range(5, max_adjust + 1, 5):
            cand = (x2 + margin - d, y)
            if candidate_valid(cand):
                return cand

    # Fallback: return candidate_below if available, else candidate_above.
    return candidate_below if candidate_below is not None else candidate_above


# async def get_interactive_elements(
#     page: Page, visible_only: bool = True, output_details: bool = True
# ) -> Tuple[
#     List[Tuple[Dict[str, float], Tuple[int, int]]],
#     List[Dict[str, Union[str, Tuple[int, int]]]],
# ]:
#     # Define selectors for common interactive elements.
#     selectors = [
#         "button",
#         "a",
#         "input",
#         "select",
#         "textarea",
#         "[role='button']",
#         "[onclick]",
#         "[tabindex]",
#     ]
#     combined_selector = ", ".join(selectors)

#     # Query interactive elements from the page.
#     elements = await page.query_selector_all(combined_selector)

#     # Lists to store data for drawing and for details.
#     annotated_elements = []  # Each item is (bbox, center)
#     interactive_details = []  # Collected if output_details is True.

#     for element in elements:
#         bbox = await element.bounding_box()
#         if bbox is None:
#             continue
#         # Compute the center and cast to integer.
#         center_x = int(bbox["x"] + bbox["width"] / 2)
#         center_y = int(bbox["y"] + bbox["height"] / 2)

#         # If filtering by visibility, perform a hit-test.
#         if visible_only:
#             is_on_top = await page.evaluate(
#                 """arg => {
#                     const { element, centerX, centerY } = arg;
#                     const topElement = document.elementFromPoint(centerX, centerY);
#                     return topElement === element || element.contains(topElement);
#                 }""",
#                 {"element": element, "centerX": center_x, "centerY": center_y},
#             )
#             if not is_on_top:
#                 continue

#         annotated_elements.append((bbox, (center_x, center_y)))

#         if output_details:
#             role = await element.get_attribute("role") or ""
#             aria_label = await element.get_attribute("aria-label")
#             title = await element.get_attribute("title")
#             description = (
#                 aria_label
#                 if aria_label is not None
#                 else (title if title is not None else "")
#             )
#             interactive_details.append(
#                 {
#                     "role": role,
#                     "description": description,
#                     "center": (center_x, center_y),
#                     "top_left": (int(bbox["x"]), int(bbox["y"])),
#                     "bottom_right": (
#                         int(bbox["x"] + bbox["width"]),
#                         int(bbox["y"] + bbox["height"]),
#                     ),
#                 }
#             )
#     return annotated_elements, interactive_details


async def get_interactive_elements(
    page: Page, visible_only: bool = True, output_details: bool = True
) -> Tuple[
    List[Tuple[Dict[str, float], Tuple[int, int]]],
    List[Dict[str, Union[str, Tuple[int, int]]]],
]:
    """
    Queries all frames (including iframes) for interactive elements and returns two lists:
      1. A list of tuples (bbox, center) where:
         - bbox: A dictionary with bounding box details (x, y, width, height) in main-page coordinates.
         - center: A tuple of ints representing the center coordinate.
      2. A list of dictionaries with element details, including:
         - "role": The element's role (if any).
         - "description": The element's aria-label or title (if available).
         - "center": The center coordinate (tuple of ints).
         - "top_left": The top-left corner of the bounding box.
         - "bottom_right": The bottom-right corner of the bounding box.

    This function iterates over all frames (including the main frame). For each frame, it queries interactive
    elements using standard selectors. For frames other than the main frame, it retrieves the iframe element via
    frame.frame_element() and uses its bounding box as an offset, so that the returned coordinates are in the main
    page coordinate system.

    Parameters:
        page (Page): The Playwright page object.
        visible_only (bool, optional): If True, only returns elements that pass a hit-test (are visible). Default is True.
        output_details (bool, optional): If True, returns detailed information for each element. Default is True.

    Returns:
        Tuple[
            List[Tuple[Dict[str, float], Tuple[int, int]]],
            List[Dict[str, Union[str, Tuple[int, int]]]]
        ]: A tuple containing:
            - A list of (bounding_box, center) tuples.
            - A list of dictionaries with element details.
    """
    selectors = [
        "button",
        "a",
        "input",
        "select",
        "textarea",
        "[role='button']",
        "[onclick]",
        "[tabindex]",
    ]
    combined_selector = ", ".join(selectors)

    all_annotated: List[Tuple[Dict[str, float], Tuple[int, int]]] = []
    all_details: List[Dict[str, Union[str, Tuple[int, int]]]] = []

    # Iterate over all frames (page.frames is a property, not a function).
    for frame in page.frames:
        # Default offset is (0, 0) for the main frame.
        offset_x, offset_y = 0.0, 0.0
        if frame.parent_frame is not None:
            try:
                frame_element = (
                    await frame.frame_element()
                )  # The iframe element in the parent.
                box = await frame_element.bounding_box()
                if box is not None:
                    offset_x = box["x"]
                    offset_y = box["y"]
            except Exception:
                offset_x, offset_y = 0.0, 0.0

        elements = await frame.query_selector_all(combined_selector)
        for element in elements:
            bbox = await element.bounding_box()
            if bbox is None:
                continue
            # Adjust bbox coordinates by the frame's offset.
            adjusted_bbox = {
                "x": bbox["x"] + offset_x,
                "y": bbox["y"] + offset_y,
                "width": bbox["width"],
                "height": bbox["height"],
            }
            center_x = int(adjusted_bbox["x"] + adjusted_bbox["width"] / 2)
            center_y = int(adjusted_bbox["y"] + adjusted_bbox["height"] / 2)

            if visible_only:
                is_on_top = await page.evaluate(
                    """(arg) => {
                           const { element, centerX, centerY } = arg;
                           const topElement = document.elementFromPoint(centerX, centerY);
                           return topElement === element || element.contains(topElement);
                       }""",
                    {"element": element, "centerX": center_x, "centerY": center_y},
                )
                if not is_on_top:
                    continue

            all_annotated.append((adjusted_bbox, (center_x, center_y)))

            if output_details:
                role = await element.get_attribute("role") or ""
                aria_label = await element.get_attribute("aria-label")
                title = await element.get_attribute("title")
                description = (
                    aria_label
                    if aria_label is not None
                    else (title if title is not None else "")
                )
                all_details.append(
                    {
                        "role": role,
                        "description": description,
                        "center": (center_x, center_y),
                        "top_left": (int(adjusted_bbox["x"]), int(adjusted_bbox["y"])),
                        "bottom_right": (
                            int(adjusted_bbox["x"] + adjusted_bbox["width"]),
                            int(adjusted_bbox["y"] + adjusted_bbox["height"]),
                        ),
                    }
                )

    return all_annotated, all_details


async def highlight_interactive_elements(
    page: Page,
    visible_only: bool = True,
    output_image: bool = True,
    draw_center_dot: bool = False,
    output_details: bool = True,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Highlights interactive elements on a webpage by drawing bounding boxes and center coordinate labels on a screenshot.

    The function queries interactive elements (using selectors for buttons, links, inputs, etc.) from the
    provided Playwright state object. It optionally filters elements based on visibility (using a hit-test)
    and annotates a fresh screenshot with bounding boxes, an optional center dot, and coordinate labels.
    The labels are positioned using a helper function that attempts to avoid overlapping other labels or
    the bounding boxes of neighboring elements.

    Parameters:
        state (Any): The Playwright state object containing the page and screenshot_path.
        visible_only (bool, optional): If True, only elements that pass the visibility hit-test are processed. Default is True.
        output_image (bool, optional): If True, includes the annotated image in the returned result. Default is True.
        draw_center_dot (bool, optional): If True, draws a center dot at the element's center. Default is True.
        output_details (bool, optional): If True, includes a list of interactive element details in the result. Default is False.
        save_path (str, optional): If provided, the annotated screenshot is saved to this file path; otherwise, the image is not saved to disk.

    Returns:
        dict: A dictionary with the following keys:
            - "image": The annotated image (if output_image is True). This is either a file path (if save_path is provided)
                       or a PIL Image object.
            - "elements": A list of dictionaries (if output_details is True) containing details for each interactive element
                          (role, description, center, top_left, bottom_right).

    Note:
        When save_path is not provided, the function calls page.screenshot() without a path to obtain image bytes,
        then opens the image with PIL, annotates it, and returns the annotated image without saving to disk.
    """

    # Ensure at least one output type is requested.
    if not (output_image or output_details):
        raise ValueError("At least one of output_image or output_details must be True.")

    annotated_elements, interactive_details = await get_interactive_elements(
        page=page, visible_only=visible_only, output_details=output_details
    )

    result = {}

    # Take a fresh screenshot.
    screenshot_bytes = await page.screenshot()
    image = PILImage.open(io.BytesIO(screenshot_bytes))

    if output_image:
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except Exception:
            font = ImageFont.load_default()

        # Build a list of all bounding box rectangles.
        all_boxes = []
        for bbox, _ in annotated_elements:
            all_boxes.append(
                (
                    int(bbox["x"]),
                    int(bbox["y"]),
                    int(bbox["x"] + bbox["width"]),
                    int(bbox["y"] + bbox["height"]),
                )
            )

        # List to store already placed label rectangles.
        placed_labels = []
        margin = 5  # 5px gap from the bounding box.

        for bbox, center in annotated_elements:
            x, y = int(bbox["x"]), int(bbox["y"])
            x2, y2 = int(bbox["x"] + bbox["width"]), int(bbox["y"] + bbox["height"])
            current_box = (x, y, x2, y2)

            # Draw the bounding box in Imperial Red (#ED2939).
            draw.rectangle([(x, y), (x2, y2)], outline="#ED2939", width=2)
            # Optionally, draw the center dot.
            if draw_center_dot:
                r = 3
                draw.ellipse(
                    [(center[0] - r, center[1] - r), (center[0] + r, center[1] + r)],
                    fill="#ED2939",
                )

            # The label text using integer center coordinates.
            label = f"({center[0]}, {center[1]})"
            bbox_text = font.getbbox(label)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]

            # Find a valid label position.
            candidate = find_label_position(
                x,
                y,
                x2,
                y2,
                text_width,
                text_height,
                image_width,
                image_height,
                current_box,
                all_boxes,
                placed_labels,
                margin=margin,
                max_adjust=20,
            )
            candidate_rect = (
                candidate[0],
                candidate[1],
                candidate[0] + text_width,
                candidate[1] + text_height,
            )
            placed_labels.append(candidate_rect)
            
            # Draw white background with alpha 200 for the label
            bg_padding = 3
            draw.rectangle([
                candidate[0] - bg_padding,
                candidate[1] - bg_padding,
                candidate[0] + text_width + bg_padding,
                candidate[1] + text_height + bg_padding
            ], fill=(255, 255, 255, 150))
            
            draw.text(candidate, label, fill="#3DED97", font=font)

        # If save_path is provided, then save the annotated image to that path.
        if save_path:
            image.save(save_path)
            result["image_path"] = save_path
            result["image"] = None
        else:
            # If not saving, just display the annotated image.
            result["image_path"] = None
            result["image"] = image

    if output_details:
        result["elements"] = interactive_details

    return result


async def mouse_move_smooth(
    page: Page,
    target: Tuple[int, int],
    start: Optional[Tuple[int, int]] = None,
    smoothness_factor: float = 1.0,
    noise: float = 2.0,
    save_path: Optional[str] = None,
) -> Union[PILImage.Image, str]:
    """
    Moves the mouse pointer smoothly to a target coordinate using a dynamic velocity profile and smoothness factor,
    then takes a screenshot with an overlaid pointer icon.

    If no starting position is provided, the function injects JS to track the mouse position (via a 'mousemove'
    listener) and uses that value. The movement is divided into a number of steps determined by the total distance and
    the smoothness_factor. For a default smoothness_factor of 1.0, an 800-pixel movement uses 50 steps with a 0.002 s delay
    per step (0.1 s total). The velocity profile is defined as:
         v(i) = v_min + (v_max - v_min) * sin(pi * (i/actual_steps))
    and is scaled so that the sum of increments equals the total displacement.
    After the movement, a screenshot is taken. The function uses shared interactive element detection logic to check if
    the target point is inside any interactive element's bounding box. If it is, a hand pointer icon is used; otherwise,
    a normal cursor icon is overlaid on the screenshot. The icon is resized to 32×32 pixels.

    Parameters:
        page (Page): The Playwright page object.
        target (Tuple[int, int]): The final (x, y) coordinate for the pointer.
        start (Optional[Tuple[int, int]], optional): The starting (x, y) coordinate. If not provided, the current mouse
            position (tracked in the page) is used.
        smoothness_factor (float, optional): Controls movement smoothness. Default is 1.0.
        noise (float, optional): Maximum random noise (in pixels) added per step. Default is 2.0.
        save_path (Optional[str], optional): If provided, saves the final annotated screenshot to this path.

    Returns:
        Union[PILImage.Image, str]: The annotated screenshot as a PIL Image (or file path if saved).

    Note:
        - If no starting position is provided, this function injects a 'mousemove' listener to store the last mouse position
          in window.__lastMousePosition.
        - Baseline for an 800-pixel movement is 50 steps with 0.002 s delay per step (0.1 s total).
        - Dynamic velocity parameters used are: v_min = 400 and v_max = 1600 (pixels/second).
    """
    # If start is not provided, retrieve the current mouse pointer position from the page.
    if start is None:
        # Inject mousemove listener if not already present.
        await page.evaluate(
            """() => {
            if (!window.__lastMousePosition) {
                window.__lastMousePosition = { x: 0, y: 0 };
                document.addEventListener("mousemove", (e) => {
                    window.__lastMousePosition = { x: e.clientX, y: e.clientY };
                });
            }
        }"""
        )
        pos = await page.evaluate("() => window.__lastMousePosition")
        start = (int(pos["x"]), int(pos["y"]))

    # Build absolute paths for pointer icons.
    cursor_icon_path = os.path.join(ASSETS_DIR, "img", "cursor.png")
    hand_pointer_icon_path = os.path.join(ASSETS_DIR, "img", "hand-pointer.png")

    start_x, start_y = start
    target_x, target_y = target

    dx = target_x - start_x
    dy = target_y - start_y
    total_distance = math.hypot(dx, dy)
    if total_distance == 0:
        return

    # Baseline: for an 800-pixel movement, 50 steps with 0.002 s delay (0.1 s total).
    default_steps_for_800 = 50
    default_delay = 0.002
    baseline_steps = max(10, int((total_distance / 800.0) * default_steps_for_800))
    actual_steps = max(1, int(baseline_steps * smoothness_factor))
    total_time = baseline_steps * default_delay
    step_delay = total_time / actual_steps

    v_min = 400.0
    v_max = 1600.0

    velocity_profile = []
    for i in range(1, actual_steps + 1):
        f = i / actual_steps
        v = v_min + (v_max - v_min) * math.sin(math.pi * f)
        velocity_profile.append(v)
    profile_sum = sum(velocity_profile)
    scale = total_distance / (step_delay * profile_sum)
    increments = [v * scale * step_delay for v in velocity_profile]

    unit_dx = dx / total_distance
    unit_dy = dy / total_distance

    current_x, current_y = start_x, start_y
    for d in increments:
        disp_x = d * unit_dx
        disp_y = d * unit_dy
        noise_x = random.uniform(-noise, noise)
        noise_y = random.uniform(-noise, noise)
        current_x += disp_x + noise_x
        current_y += disp_y + noise_y
        await page.mouse.move(int(current_x), int(current_y))
        await asyncio.sleep(step_delay)

    await page.mouse.move(target_x, target_y)

    screenshot_bytes = await page.screenshot()
    image = PILImage.open(io.BytesIO(screenshot_bytes))

    # Use get_interactive_elements to get interactive elements.
    _, interactive_details = await get_interactive_elements(page, visible_only=True)
    # Check if target point is inside any interactive element bounding box.
    target_is_interactive = any(
        (
            elem["top_left"][0] <= target_x <= elem["bottom_right"][0]
            and elem["top_left"][1] <= target_y <= elem["bottom_right"][1]
        )
        for elem in interactive_details
    )

    try:
        if target_is_interactive:
            pointer_icon = PILImage.open(hand_pointer_icon_path)
        else:
            pointer_icon = PILImage.open(cursor_icon_path)
    except Exception as e:
        raise RuntimeError("Could not load pointer icon images.") from e

    desired_icon_size = (32, 32)
    pointer_icon = pointer_icon.resize(desired_icon_size, PILImage.Resampling.LANCZOS)

    image.paste(pointer_icon, (target_x, target_y), pointer_icon.convert("RGBA"))

    if save_path:
        image.save(save_path)
        return save_path
    else:
        return image


# A dictionary mapping color names to their hex codes.
DEFAULT_COLOR_MAP = {
    "RED": "#D0312D",
    "CHERRY RED": "#990F02",
    "ROSE": "#E3242B",
    "CRIMSON": "#B90E0A",
    "RUBY": "#900603",
    "SCARLET": "#900D09",
    "APPLE": "#A91B0D",
    "MAHOGANY": "#420C09",
    "BLOOD": "#710C04",
    "BERRY": "#7A1712",
    "Imperial Red": "#ED2939",
    "Indian Red": "#CD5C5C",
    "Desire": "#EA3C53",
    "Raspberry": "#D21F3C",
    "Candy Apple": "#FF0800",
    "Chili Red": "#C21807",
    "Orange": "#ED7014",
    "TANGERINE": "#FA8128",
    "MERIGOLD": "#FCAE1E",
    "FIRE": "#DD571C",
    "CIDER": "#B56727",
    "CANTALOUPE": "#FDA172",
    "APRICOT": "#ED820E",
    "HONEY": "#EC9706",
    "TANGELO": "#FC4C02",
    "Deep Saffron": "#FFA52C",
    "Pastel Orange": "#FEBA4F",
    "Royal Orange": "#FF9944",
    "Coral": "#FF7F50",
    "Dark Coral": "#D75341",
    "Yellow": "#F5E653",
    "TAN": "#E6DBAC",
    "BEIGE": "#EEDC9A",
    "MACAROON": "#F9E076",
    "HAZEL WOOD": "#C9BB8E",
    "SAND": "#D8B863",
    "SEPIA": "#E3B778",
    "Clover Lime": "#FCE883",
    "Royal Yellow": "#FADA5E",
    "Gold": "#FFD700",
    "LATTE": "#E7C27D",
    "OYSTER": "#DCD7A0",
    "BISCOTTI": "#E3C565",
    "PARMESAN": "#FDE992",
    "HAZELNUT": "#BDA55D",
    "Yellow Tan": "#FFE36E",
    "Banana": "#FFE135",
    "Aureolin": "#FDEE00",
    "Electric Yellow": "#FFFF33",
    "Pastel Yellow": "#FFFE71",
    "GREEN": "#3CB043",
    "CHARTREUSE": "#B0FC38",
    "SAGE": "#728C69",
    "LIME": "#AEF359",
    "PARAKEET": "#03C04A",
    "MINT": "#99EDC3",
    "Jungle green": "#29AB87",
    "Tropical Rainforest": "#00755E",
    "Persian green": "#00A693",
    "Jade green": "#00A86B",
    "FERN": "#5DBB63",
    "OLIVE": "#98BF64",
    "EMERALD": "#028A0F",
    "PEAR": "#74B72E",
    "MOSS": "#466D1D",
    "SHAMROCK": "#03AC13",
    "SEAFOAM": "#3DED97",
    "Spring": "#00F0A8",
    "BLUE": "#3944BC",
    "SLATE": "#757C88",
    "SKY": "#63C5DA",
    "NAVY": "#0A1172",
    "INDIGO": "#281E5D",
    "COBALT": "#1338BE",
    "TEAL": "#48AAAD",
    "OCEAN": "#016064",
    "PEACOCK": "#022D36",
    "AZURE": "#1520A6",
    "CERULEAN": "#0492C2",
    "LAPIS": "#2832C2",
    "BLUEBERRY": "#241571",
    "DENIM": "#151E3D",
    "ADMIRAL": "#051094",
    "SAPPHIRE": "#52B2BF",
    "ARCTIC": "#82EEFD",
    "TURQUOISE": "#40E0D0",
    "STEEL": "#4682B4",
    "Tiffany": "#81D8D0",
    "Carolina blue": "#4B9CD3",
    "Blue Sapphire": "#126180",
    "Yale blue": "#00356B",
    "PURPLE": "#A32CC4",
    "MAUVE": "#7A4988",
    "DARK VIOLET": "#710193",
    "BOYSENBERRY": "#630436",
    "ELECTRIC LAVENDER": "#E39FF6",
    "PLUM": "#601A35",
    "STRONG MAGENTA": "#A1045A",
    "DEEP LILAC": "#B65FCF",
    "GRAPE": "#663046",
    "ROYAL PERIWINKLE": "#BE93D4",
    "SANGRIA PURPLE": "#4D0F28",
    "EGGPLANT": "#311432",
    "JAZZBERRY JAM": "#67032F",
    "IRIS": "#9867C5",
    "HEATHER": "#9E7BB5",
    "AMETHYST": "#A45EE5",
    "RAISIN": "#290916",
    "ORCHID": "#DA70D6",
    "ELECTRIC VIOLET": "#8F00FF",
    "Medium Purple": "#9370DB",
    "Dark Orchid": "#9932CC",
    "Dark Magenta": "#8B008B",
    "Veronica": "#A020F0",
    "PINK": "#F699CD",
    "PUNCH": "#F25278",
    "BLUSH PINK": "#FEC5E5",
    "WATERMELON": "#FE7F9C",
    "FLAMINGO": "#FDA4BA",
    "ROUGE": "#F26B8A",
    "LIGHT SALMON": "#FDAB9F",
    "CORAL PINK": "#FE7D6A",
    "PEACH": "#FC9483",
    "STRAWBERRY": "#FC4C4E",
    "ROSEWOOD": "#9E4244",
    "FRENCH FUCHSIA": "#FC46AA",
    "Brown": "#795548",
    "MOCHA": "#3C280D",
    "PEANUT BROWN": "#795C34",
    "CAROB": "#362511",
    "HICKORY": "#371D10",
    "WOOD": "#3F301D",
    "PECAN": "#4A2511",
    "WALNUT": "#432616",
    "CARAMEL": "#65350F",
    "GINGERBREAD": "#5E2C04",
    "SYRUP": "#481F01",
    "Copper": "#B87333",
    "Dark brown": "#654321",
    "Golden brown": "#996515",
    "Light brown": "#B5651D",
    "CHARCOAL": "#28231D",
    "BLACK": "#000000",
    "GRAY": "#808080",
    "SHADOW": "#373737",
    "FLINT": "#7F7D9C",
    "CHARCOAL GRAY": "#232023",
    "Dim Gray": "#696969",
    "Gray Cloud": "#B6B6B4",
    "Medium Gray": "#BEBEBE",
    "Nickel": "#727472",
    "Stone Gray": "#928E85",
    "GRAPHITE": "#594D5B",
    "IRON": "#322D31",
    "PEWTER": "#696880",
    "CLOUD": "#C5C6D0",
    "SILVER": "#ADADC9",
    "SMOKE": "#59515E",
    "DARK SLATE": "#3E3D53",
    "ANCHOR": "#41424C",
    "ASH": "#564C4D",
    "PORPOISE": "#4D4C5C",
    "WHITE": "#FFFFFF",
    "PEARL": "#FBFCF8",
    "CREAM": "#FFFADA",
    "EGGSHELL": "#FFF9E3",
    "ALABASTER": "#FEF9F3",
    "BONE": "#E7DECC",
}


def resolve_color(color_name_or_hex: str) -> str:
    """
    Resolves a color input to its hex code. If the input matches a key in DEFAULT_COLOR_MAP,
    returns that hex code. If it starts with '#' and has length 4 or 7, returns it as is.
    Otherwise, raises a ValueError.
    """
    color_name_or_hex = color_name_or_hex.strip()
    if color_name_or_hex in DEFAULT_COLOR_MAP:
        return DEFAULT_COLOR_MAP[color_name_or_hex]
    if color_name_or_hex.startswith("#") and len(color_name_or_hex) in [4, 7]:
        return color_name_or_hex
    raise ValueError(
        f"Unknown color '{color_name_or_hex}'. Must be a valid hex code or one of {list(DEFAULT_COLOR_MAP.keys())}."
    )


async def highlight_screen_grid(
    page: Page,
    nrows: int = 3,
    ncols: int = 3,
    dot_color: str = "Raspberry",
    font_size: int = 12,
    bg_opacity: float = 0.7,  # 1.0 = fully opaque, 0.0 = fully transparent
) -> Dict[str, Union[PILImage.Image, Dict[int, Tuple[int, int]]]]:
    """
    Divides the screenshot of the current page into an nrows x ncols grid, draws a dot at the center of each cell,
    and labels each cell with a numeric index.

    The dot and label color is determined by dot_color, which can be a hex code or one of the predefined
    color names in DEFAULT_COLOR_MAP (default is "Raspberry" (#D21F3C)). The background for each grid number is drawn
    with a white fill at the specified opacity (bg_opacity, between 0.0 and 1.0, where 1.0 is fully opaque and 0.0 is fully transparent).

    Cells are indexed in row-major order (0 is top-left, then left-to-right, then next row).

    Returns a dictionary with:
      - "image": The annotated PIL Image.
      - "grid_coordinates": A dictionary mapping each cell index to its center coordinates.

    Parameters:
        page (Page): The Playwright page object.
        nrows (int): Number of rows in the grid. Default is 3.
        ncols (int): Number of columns in the grid. Default is 3.
        dot_color (str): Either a hex code (e.g. "#D21F3C") or a predefined color name. Default is "Raspberry".
        bg_opacity (float): Opacity for the background behind the grid numbers (0.0 to 1.0). 1.0 means fully opaque; 0.0 means fully transparent.

    Returns:
        Dict[str, Union[PILImage.Image, Dict[int, Tuple[int, int]]]]:
            {
              "image": PILImage.Image,
              "grid_coordinates": {index: (center_x, center_y), ...}
            }
    """
    # Resolve the provided dot color.
    color_hex = resolve_color(dot_color)

    # Capture a fresh screenshot and ensure it's in RGBA mode.
    screenshot_bytes = await page.screenshot()
    image = PILImage.open(io.BytesIO(screenshot_bytes)).convert("RGBA")

    width, height = image.size
    cell_width = width / ncols
    cell_height = height / nrows

    grid_coordinates: Dict[int, Tuple[int, int]] = {}

    try:
        font = ImageFont.truetype(font="DejaVuSans.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default(size=font_size)

    # Create an overlay for drawing the transparent background for grid numbers.
    overlay = PILImage.new("RGBA", image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Convert the bg_opacity (0.0 to 1.0) to an integer alpha (0 to 255).
    # 1.0 means fully opaque (alpha=255), 0.0 means fully transparent (alpha=0).
    bg_alpha = int(bg_opacity * 255)

    # Draw the transparent backgrounds in the overlay.
    index = 0
    r = 3  # dot radius
    pad = 2  # padding for background rectangle
    for row in range(nrows):
        for col in range(ncols):
            cell_x = col * cell_width
            cell_y = row * cell_height
            center_x = int(cell_x + cell_width / 2)
            center_y = int(cell_y + cell_height / 2)
            grid_coordinates[index] = (center_x, center_y)

            label = str(index)
            label_offset_x = r + 2
            label_offset_y = -r
            text_x = center_x + label_offset_x
            text_y = center_y + label_offset_y

            # Get bounding box of the label.
            text_bbox = font.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Calculate background rectangle with minimal padding.
            bg_rect = [
                text_x - pad,
                text_y - pad,
                text_x + text_width + pad,
                text_y + text_height + pad,
            ]
            # Draw the background rectangle with white color and alpha 200 as requested.
            overlay_draw.rectangle(bg_rect, fill=(255, 255, 255, 200))

            index += 1

    # Composite the overlay onto the main image.
    composite = PILImage.alpha_composite(image, overlay)

    # Draw the opaque dots and numbers on the composite image.
    draw_composite = ImageDraw.Draw(composite)
    index = 0
    for row in range(nrows):
        for col in range(ncols):
            cell_x = col * cell_width
            cell_y = row * cell_height
            center_x = int(cell_x + cell_width / 2)
            center_y = int(cell_y + cell_height / 2)

            # Draw the dot.
            draw_composite.ellipse(
                [(center_x - r, center_y - r), (center_x + r, center_y + r)],
                fill=color_hex,
            )

            # Draw the grid number completely opaque.
            label = str(index)
            label_offset_x = r + 2
            label_offset_y = -r
            text_x = center_x + label_offset_x
            text_y = center_y + label_offset_y
            draw_composite.text(
                (text_x, text_y), label, font=font, fill=color_hex, stroke_width=0
            )

            index += 1

    return {"image": composite, "grid_coordinates": grid_coordinates}


#############################################################
############                                     ############
############     TOOL USE SCHEMA DEFINITIONS     ############
############                                     ############
#############################################################


class PropertySchema(TypedDict):
    type: str
    description: Optional[str]


class ParameterSchema(TypedDict):
    type: str
    properties: dict[str, PropertySchema]
    required: List[str]


class FnCallSchema(TypedDict):
    name: str
    description: Optional[str]
    parameters: Optional[ParameterSchema]


class ToolUseSchema(TypedDict):
    type: str
    function: FnCallSchema


REASONING_PROMPT = "Describe your reasoning for the action you are about to take."


FN_WEB_NAVIGATE_URL: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "goto",
        "description": (
            'Uses the Playwright library to navigate to the provided URL directly and perform necessary actions. You are not allowed to use http://example.com . If you don\'t have a specific URL in your context, use must navigate to "https://google.com" to search for the required information.'
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "url": {
                    "type": "string",
                    "description": "The target URL that the browser should navigate to.",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the navigation (default is library dependent)."
                    ),
                },
            },
            "required": ["reasoning", "url"],
        },
    },
}

FN_WEB_BACK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "go_back",
        "description": (
            "Uses the Playwright library to navigate back to the previous page in the browser history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the navigation (default is library dependent)."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_SCROLL_TOP: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to_top",
        "description": (
            "Uses the Playwright library to scroll to the beginning of the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_SCROLL_BOTTOM: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to_bottom",
        "description": (
            "Uses the Playwright library to scroll to the end of the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
            },
            "required": ["reasoning"],
        },
    },
}

# FN_WEB_SCROLL_UP: ToolUseSchema = {
#     "type": "function",
#     "function": {
#         "name": "scroll_up",
#         "description": (
#             "Uses the Playwright library to scroll up by a specified number of pixels."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "reasoning": {
#                     "type": "string",
#                     "description": REASONING_PROMPT,
#                 },
#                 "distance": {
#                     "type": "integer",
#                     "description": (
#                         "The number of pixels to scroll up (default is library dependent)."
#                     ),
#                 },
#             },
#             "required": ["distance", "reasoning"],
#         },
#     },
# }

# FN_WEB_SCROLL_DOWN: ToolUseSchema = {
#     "type": "function",
#     "function": {
#         "name": "scroll_down",
#         "description": (
#             "Uses the Playwright library to scroll down by a specified number of pixels."
#         ),
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "reasoning": {
#                     "type": "string",
#                     "description": REASONING_PROMPT,
#                 },
#                 "distance": {
#                     "type": "integer",
#                     "description": (
#                         "The number of pixels to scroll down (default is library dependent)."
#                     ),
#                 },
#             },
#             "required": ["distance", "reasoning"],
#         },
#     },
# }

FN_WEB_SCROLL_TO: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "scroll_to",
        "description": (
            "Uses the Playwright library to scroll to a specific element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": "The CSS selector of the element to scroll to. Either selector or role must be provided.",
                },
                "role": {
                    "type": "string",
                    "description": "The ARIA role of the element to scroll to. Either selector or role must be provided.",
                },
            },
            "required": ["reasoning"],
            "oneOf": [{"required": ["selector"]}, {"required": ["role"]}],
        },
    },
}

FN_WEB_MOUSE_SCROLL: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_scroll",
        "description": (
            "Uses the Playwright library's mouse wheel method to scroll the page both horizontally and vertically. "
            "A positive y value scrolls down, while a negative y scrolls up. Similarly, a positive x scrolls right and "
            "a negative x scrolls left. This unified method allows scrolling in any direction."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "x": {
                    "type": "integer",
                    "description": (
                        "The horizontal scroll offset. Use a positive value to scroll right and a negative value to scroll left. "
                        "Defaults to 0 if not provided."
                    ),
                },
                "y": {
                    "type": "integer",
                    "description": (
                        "The vertical scroll offset. Use a positive value to scroll down and a negative value to scroll up. "
                        "Defaults to 0 if not provided."
                    ),
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_MOVE_MOUSE_SMOOTHLY: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "move_mouse_smoothly",
        "description": (
            "Moves the mouse pointer smoothly to a specified target coordinate using a dynamic velocity profile. "
            "Unlike a basic move, this simulates human-like motion, making it harder for bot detectors to block the agent. "
            "This method is equivalent to hovering over an element but with a natural movement pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "target": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The target (x, y) coordinate for the mouse movement.",
                },
                "start": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The starting (x, y) coordinate (optional). If you don't provide a starting point, the current mouse position will be used.",
                },
                "smoothness_factor": {
                    "type": "number",
                    "description": "Controls the smoothness of the mouse movement. Default is 1.0.",
                },
                "noise": {
                    "type": "number",
                    "description": "Maximum random noise (in pixels) added per step. Default is 2.0.",
                },
                "save_path": {
                    "type": "string",
                    "description": "If provided, the annotated screenshot will be saved to this file path.",
                },
            },
            "required": ["target", "reasoning"],
        },
    },
}

FN_WEB_MOUSE_MOVE: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_move",
        "description": (
            "Moves the mouse pointer to a specified (x, y) coordinate on the page using a basic move action. "
            "This is equivalent to a hover action where the pointer is repositioned instantly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "x": {
                    "type": "integer",
                    "description": "The x-coordinate for the mouse move.",
                },
                "y": {
                    "type": "integer",
                    "description": "The y-coordinate for the mouse move.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the move action.",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
}

FN_WEB_MOUSE_MOVE_SMOOTH: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_move_smooth",
        "description": (
            "Moves the mouse pointer smoothly to a specified target coordinate using a dynamic velocity profile. "
            "Unlike a basic move, this simulates human-like motion, making it harder for bot detectors to block the agent. "
            "This method is equivalent to hovering over an element but with a natural movement pattern."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "target": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The target (x, y) coordinate for the mouse movement.",
                },
                "start": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "The starting (x, y) coordinate (optional).",
                },
                "smoothness_factor": {
                    "type": "number",
                    "description": "Controls the smoothness of the mouse movement. Default is 1.0.",
                },
                "noise": {
                    "type": "number",
                    "description": "Maximum random noise (in pixels) added per step. Default is 2.0.",
                },
                "save_path": {
                    "type": "string",
                    "description": "If provided, the annotated screenshot will be saved to this file path.",
                },
            },
            "required": ["target", "reasoning"],
        },
    },
}

FN_WEB_MOUSE_CLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_click",
        "description": (
            "Uses the Playwright library to perform a mouse click at the specified (x, y) coordinates on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "x": {
                    "type": "integer",
                    "description": "The x-coordinate for the mouse click.",
                },
                "y": {
                    "type": "integer",
                    "description": "The y-coordinate for the mouse click.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the click action.",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
}

FN_WEB_MOUSE_DBLCLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_dbclick",
        "description": (
            "Uses the Playwright library to perform a double click at the specified (x, y) coordinates on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "x": {
                    "type": "integer",
                    "description": "The x-coordinate for the mouse double click.",
                },
                "y": {
                    "type": "integer",
                    "description": "The y-coordinate for the mouse double click.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the double click action.",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
}

FN_WEB_MOUSE_RIGHT_CLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "mouse_right_click",
        "description": (
            "Uses the Playwright library to perform a right click at the specified (x, y) coordinates on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "x": {
                    "type": "integer",
                    "description": "The x-coordinate for the mouse right click.",
                },
                "y": {
                    "type": "integer",
                    "description": "The y-coordinate for the mouse right click.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the right click action.",
                },
            },
            "required": ["x", "y", "reasoning"],
        },
    },
}

FN_WEB_KEYBOARD_TYPE: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "keyboard_type",
        "description": (
            "Uses the Playwright library's keyboard API to type the provided text into the element that currently has focus. "
            "An optional delay between keystrokes can be specified to simulate natural typing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "text": {
                    "type": "string",
                    "description": "The text that will be typed using the keyboard.",
                },
                "delay": {
                    "type": "integer",
                    "description": "Optional delay in milliseconds between keystrokes.",
                },
            },
            "required": ["reasoning", "text"],
        },
    },
}


FN_WEB_KEYBOARD_PRESS: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "keyboard_press",
        "description": (
            "Uses the Playwright keyboard API to press a specified special key (e.g., 'Enter', 'Escape', etc.) on the page. "
            "Ensure that the key is one of the supported keys: 'Enter', 'Tab', 'Escape', 'ArrowUp', 'ArrowDown', 'ArrowLeft', "
            "'ArrowRight', 'Backspace', 'Delete', 'Home', 'End', 'PageUp', 'PageDown', 'Insert', 'Meta', 'Shift', 'Control', 'Alt'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "key": {
                    "type": "string",
                    "description": "The special key to press (must be one of the allowed keys).",
                },
                "delay": {
                    "type": "integer",
                    "description": "Optional delay in milliseconds before releasing the key.",
                },
            },
            "required": ["reasoning", "key"],
        },
    },
}


FN_WEB_CLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "click",
        "description": (
            "Uses the Playwright library to click on a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "The CSS selector of the element to click on. Either selector or role must be provided."
                    ),
                },
                "role": {
                    "type": "string",
                    "description": (
                        "The ARIA role of the element to click on. Either selector or role must be provided."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Optional timeout in milliseconds for the click action (default is library dependent)."
                    ),
                },
            },
            "required": ["reasoning"],
            "oneOf": [{"required": ["selector"]}, {"required": ["role"]}],
        },
    },
}

FN_WEB_DBLCLICK: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "dblclick",
        "description": "Uses the Playwright library to double click on a specified element on the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": "The CSS selector of the element to double click. Either selector or role must be provided.",
                },
                "role": {
                    "type": "string",
                    "description": "The ARIA role of the element to double click. Either selector or role must be provided.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the double click action.",
                },
            },
            "required": ["reasoning"],
            "oneOf": [{"required": ["selector"]}, {"required": ["role"]}],
        },
    },
}


FN_WEB_INPUT_TEXT: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "input_text",
        "description": (
            "Uses the Playwright library to input text into a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": "The CSS selector of the element to input text into. Either selector or role must be provided.",
                },
                "role": {
                    "type": "string",
                    "description": "The ARIA role of the element to input text into. Either selector or role must be provided.",
                },
                "text": {
                    "type": "string",
                    "description": "The text to input into the element.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the input action.",
                },
            },
            "required": ["text", "reasoning"],
            "oneOf": [{"required": ["selector"]}, {"required": ["role"]}],
        },
    },
}

FN_WEB_HOVER: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "hover",
        "description": (
            "Uses the Playwright library to hover over a specified element on the page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": "The CSS selector of the element to hover over. Either selector or role must be provided.",
                },
                "role": {
                    "type": "string",
                    "description": "The ARIA role of the element to hover over. Either selector or role must be provided.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the hover action.",
                },
            },
            "required": ["reasoning"],
            "oneOf": [{"required": ["selector"]}, {"required": ["role"]}],
        },
    },
}

FN_WEB_DRAG_N_DROP: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "drag_and_drop",
        "description": "Uses the Playwright library to drag an element from a source to a target element on the page.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "source_selector": {
                    "type": "string",
                    "description": "The CSS selector of the source element. Either source_selector or source_role must be provided.",
                },
                "source_role": {
                    "type": "string",
                    "description": "The ARIA role of the source element. Either source_selector or source_role must be provided.",
                },
                "target_selector": {
                    "type": "string",
                    "description": "The CSS selector of the target element. Either target_selector or target_role must be provided.",
                },
                "target_role": {
                    "type": "string",
                    "description": "The ARIA role of the target element. Either target_selector or target_role must be provided.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds for the drag and drop action.",
                },
            },
            "required": ["reasoning"],
            "allOf": [
                {
                    "oneOf": [
                        {"required": ["source_selector"]},
                        {"required": ["source_role"]},
                    ]
                },
                {
                    "oneOf": [
                        {"required": ["target_selector"]},
                        {"required": ["target_role"]},
                    ]
                },
            ],
        },
    },
}

FN_WEB_SCREENSHOT: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "screenshot",
        "description": (
            "Uses the Playwright library to take a screenshot of the current page. If 'highlight_bbox' is True, "
            "interactive elements are highlighted with bounding boxes. When a filename is provided, the image is saved "
            "and the full file path is returned; if not, the screenshot is returned as a PIL Image object."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "filename": {
                    "type": "string",
                    "description": (
                        "Optional filename (with path relative to temp_dir) to save the screenshot. "
                        "If not provided, the screenshot is returned as a PIL Image object."
                    ),
                },
                "highlight_bbox": {
                    "type": "boolean",
                    "description": "Whether to highlight interactive elements in the screenshot (default False).",
                },
            },
            "required": ["reasoning"],
        },
    },
}

FN_WEB_GET_OBJECT_BY_LABEL: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "get_object_by_label",
        "description": "Retrieves the outer HTML of the element with the specified label (using the aria-label attribute).",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "label": {
                    "type": "string",
                    "description": "The label of the element to retrieve.",
                },
            },
            "required": ["reasoning", "label"],
        },
    },
}

FN_WEB_GET_OBJECT_BY_SELECTOR: ToolUseSchema = {
    "type": "function",
    "function": {
        "name": "get_object_by_selector",
        "description": "Retrieves the outer HTML of the element matching the given CSS selector.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": REASONING_PROMPT,
                },
                "selector": {
                    "type": "string",
                    "description": "The CSS selector of the element to retrieve.",
                },
            },
            "required": ["reasoning", "selector"],
        },
    },
}


##############################################################
############                                      ############
############     BROWSER TOOL CLASS DEFINITION    ############
############                                      ############
##############################################################


class BrowserTool:
    """
    A tool to perform browser actions using Playwright and record the reasoning,
    screenshots, and outputs for each step.
    """

    def __init__(
        self,
        playwright: async_playwright,
        browser,
        context: BrowserContext,
        page: Page,
        temp_dir: str,
        screenshot_dir: Optional[str] = None,
        downloads_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the BrowserTool instance.

        Parameters:
            playwright (Playwright): The Playwright instance.
            browser: The browser instance.
            context (BrowserContext): The browser context.
            page (Page): The page to perform actions on.
            temp_dir (str): Temporary directory for downloads and other files.
            screenshot_dir (Optional[str]): Dedicated directory for screenshots. Defaults to temp_dir/screenshots.
        """
        self.playwright = playwright
        self.browser = browser
        self.context = context
        self.page = page
        self.temp_dir = temp_dir
        
        # Set downloads path - use provided path or default to temp_dir/downloads
        self.download_path = downloads_path or os.path.join(temp_dir, "downloads")

        # Use dedicated screenshot directory or default to temp_dir/screenshots
        self.screenshot_path = screenshot_dir or os.path.join(temp_dir, "screenshots")

        os.makedirs(self.download_path, exist_ok=True)
        os.makedirs(self.screenshot_path, exist_ok=True)
        # History to store the reasoning chains, actions, screenshots, and outputs
        self.history: List[Dict[str, Any]] = []

    @classmethod
    async def create(
        cls,
        temp_dir: Optional[str] = None,
        default_browser: str = "chrome",
        headless: bool = True,
        viewport: Optional[Dict[str, int]] = None,
        screenshot_dir: Optional[str] = None,
    ) -> "BrowserTool":
        """
        Create and initialize a BrowserTool instance using the async Playwright API.

        Parameters:
            temp_dir (Optional[str]): Optional temporary directory for files. Defaults to system temp if not provided.
            default_browser (str): Browser channel to use.
            headless (bool): Whether to launch the browser in headless mode.
            viewport (Optional[Dict[str, int]]): Browser viewport dimensions.
            screenshot_dir (Optional[str]): Dedicated directory for screenshots.

        Returns:
            BrowserTool: An instance with a page navigated to a default URL (https://google.com).
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            channel=default_browser, headless=headless
        )
        context_kwargs = {}
        if viewport:
            context_kwargs["viewport"] = viewport
        context: BrowserContext = await browser.new_context(**context_kwargs)
        page: Page = await context.new_page()
        tool = cls(playwright, browser, context, page, temp_dir, screenshot_dir)

        return tool

    @classmethod
    async def create_safe(
        cls,
        temp_dir: Optional[str] = None,
        default_browser: str = "chrome",
        headless: bool = True,
        timeout: Optional[int] = None,
        viewport: Optional[Dict[str, int]] = None,
        screenshot_dir: Optional[str] = None,
        browser_channel: Optional[str] = None,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        downloads_path: Optional[str] = None,
    ) -> "BrowserTool":
        """
        Create and initialize a BrowserTool instance using the async Playwright API with a timeout.

        Parameters:
            temp_dir (Optional[str]): Optional temporary directory for files. Defaults to system temp if not provided.
            default_browser (str): Browser channel to use.
            headless (bool): Whether to launch the browser in headless mode.
            timeout (Optional[int]): Optional timeout in milliseconds for the creation process.
            viewport (Optional[Dict[str, int]]): Browser viewport dimensions.
            screenshot_dir (Optional[str]): Dedicated directory for screenshots.
            browser_channel (Optional[str]): Browser channel override.
            viewport_width (int): Browser viewport width.
            viewport_height (int): Browser viewport height.
            downloads_path (Optional[str]): Custom downloads directory path.

        Returns:
            BrowserTool: An instance with a page navigated to a default URL (https://google.com).
        """
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        
        # Use browser_channel if provided, otherwise use default_browser
        channel = browser_channel or default_browser
        
        playwright = await async_playwright().start()
        
        # Set up launch options with enhanced configuration
        launch_options = {
            "headless": headless,
            # Add args to speed up browser and disable font loading
            "args": [
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--disable-font-subpixel-positioning",
                "--disable-remote-fonts",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        }
        if channel:
            launch_options["channel"] = channel
        
        browser = await playwright.chromium.launch(**launch_options)
        
        # Set up context options
        context_kwargs = {
            "accept_downloads": True,
            "java_script_enabled": True,
            "bypass_csp": True,  # Bypass Content Security Policy for better compatibility
        }
        
        # Handle viewport settings
        if viewport:
            context_kwargs["viewport"] = viewport
        else:
            context_kwargs["viewport"] = {"width": viewport_width, "height": viewport_height}
        
        # Enable downloads - Playwright handles downloads automatically to a temp directory
        # We can't set a custom downloads path in context creation, but downloads are accessible via the Download API
        context_kwargs["accept_downloads"] = True
        
        # Create downloads directory for manual file operations if needed
        if downloads_path:
            os.makedirs(downloads_path, exist_ok=True)
        elif temp_dir:
            downloads_dir = os.path.join(temp_dir, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
        
        for attempt in range(3):
            try:
                context = await asyncio.wait_for(
                    browser.new_context(**context_kwargs), timeout=1.5
                )
                
                # Set default timeouts for the context
                if timeout:
                    context.set_default_navigation_timeout(timeout)
                    context.set_default_timeout(timeout)
                
                page = await asyncio.wait_for(context.new_page(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise TimeoutError("BrowserTool creation timed out.")
        
        tool = cls(playwright, browser, context, page, temp_dir, screenshot_dir, downloads_path)
        
        # Don't navigate to default page automatically - let the caller decide
        return tool

    async def goto(
        self, url: str, reasoning: Optional[str] = None, timeout: Optional[int] = None
    ) -> None:
        """
        Navigate the page to a given URL.

        Parameters:
            url (str): The URL to navigate to.
            reasoning (Optional[str]): The reasoning chain for this navigation. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.goto(url, timeout=timeout)
        try:
            await self.page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await self.page.wait_for_timeout(1000)
        title = await self.page.title()
        self.history.append(
            {
                "action": "goto",
                "reasoning": r,
                "url": url,
                "output": title,
            }
        )

    async def go_back(
        self, reasoning: Optional[str] = None, timeout: Optional[int] = None
    ) -> None:
        """
        Navigate back in the browser history.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.go_back(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "go_back",
                "reasoning": r,
                "output": title,
            }
        )

    async def scroll_to_top(self, reasoning: Optional[str] = None) -> None:
        """
        Scroll to the top of the page using the mouse wheel and wait until rendering is complete.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this scrolling action. Defaults to empty string.
        """
        r = reasoning or ""
        # Scroll up using a large negative delta to ensure reaching the top.
        await self.page.mouse.wheel(0, -10000)
        # Wait until the scroll position is at the top, indicating the page has been rendered.
        await self.page.wait_for_function("window.scrollY === 0")
        self.history.append(
            {
                "action": "scroll_to_top",
                "reasoning": r,
            }
        )

    async def scroll_to_bottom(self, reasoning: Optional[str] = None) -> None:
        """
        Scroll to the bottom of the page using the mouse wheel.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this scrolling action. Defaults to empty string.
        """
        r = reasoning or ""
        # Use a large positive delta to scroll downwards
        await self.page.mouse.wheel(0, 10000)
        self.history.append(
            {
                "action": "scroll_to_bottom",
                "reasoning": r,
            }
        )

    async def scroll_up(self, distance: int, reasoning: Optional[str] = None) -> None:
        """
        Scroll up by a specified distance (in pixels) using the evaluate method.

        This method performs an instant scroll upward by the specified pixel distance using
        JavaScript's window.scrollBy() function. The scroll is immediate and does not simulate
        natural mouse wheel scrolling.

        Parameters:
            distance (int): The number of pixels to scroll up. Must be a positive integer.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.evaluate(f"window.scrollBy(0, -{distance})")
        self.history.append(
            {
                "action": "scroll_up_eval",
                "reasoning": r,
                "distance": distance,
            }
        )

    async def scroll_down(self, distance: int, reasoning: Optional[str] = None) -> None:
        """
        Scroll down by a specified distance (in pixels) using the evaluate method.

        This method performs an instant scroll downward by the specified pixel distance using
        JavaScript's window.scrollBy() function. The scroll is immediate and does not simulate
        natural mouse wheel scrolling.

        Parameters:
            distance (int): The number of pixels to scroll down. Must be a positive integer.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.evaluate(f"window.scrollBy(0, {distance})")
        self.history.append(
            {
                "action": "scroll_down_eval",
                "reasoning": r,
                "distance": distance,
            }
        )

    async def mouse_scroll(
        self,
        x: int = 0,
        y: int = 0,
        reasoning: Optional[str] = None,
    ) -> None:
        """
        Scroll the page horizontally and/or vertically using the mouse wheel.

        This method accepts both horizontal (x) and vertical (y) scroll offsets. A positive y value scrolls
        downwards while a negative y value scrolls upwards. Similarly, a positive x value scrolls to the right,
        and a negative x value scrolls to the left.

        The method uses the mouse wheel action and waits until the page's scroll position reaches the expected
        offsets based on the initial position and the provided x and y values.

        Parameters:
            x (int): The horizontal scroll offset. Set to a negative value to scroll left, and positive to scroll right.
            y (int): The vertical scroll offset. Set to a negative value to scroll up, and positive to scroll down.
            reasoning (Optional[str]): A description of the reasoning for this scroll action. Defaults to an empty string.

        Returns:
            None
        """
        r = reasoning or ""

        # Get the initial scroll positions.
        initial_x = await self.page.evaluate("() => window.scrollX")
        initial_y = await self.page.evaluate("() => window.scrollY")

        # Calculate expected positions, clamping horizontal scroll to at least 0 and at most the maximum scrollable width.
        max_x = await self.page.evaluate(
            "() => document.body.scrollWidth - window.innerWidth"
        )
        expected_x = initial_x + x
        if x < 0:
            expected_x = max(0, expected_x)
        else:
            expected_x = min(expected_x, max_x)

        # Calculate expected vertical position (similarly, clamp to a minimum of 0).
        max_y = await self.page.evaluate(
            "() => document.body.scrollHeight - window.innerHeight"
        )
        expected_y = initial_y + y
        if y < 0:
            expected_y = max(0, expected_y)
        else:
            expected_y = min(expected_y, max_y)

        # Perform the scroll action.
        await self.page.mouse.wheel(x, y)

        # Poll until the scroll positions reach (or surpass) the expected positions.
        while True:
            current_x = await self.page.evaluate("() => window.scrollX")
            current_y = await self.page.evaluate("() => window.scrollY")
            x_done = (
                (x >= 0 and current_x >= expected_x)
                or (x < 0 and current_x <= expected_x)
                or (x == 0)
            )
            y_done = (
                (y >= 0 and current_y >= expected_y)
                or (y < 0 and current_y <= expected_y)
                or (y == 0)
            )
            if x_done and y_done:
                break
            await self.page.wait_for_timeout(100)  # wait 100ms before rechecking

        self.history.append(
            {
                "action": "mouse_scroll",
                "reasoning": r,
                "x": x,
                "y": y,
            }
        )

    async def scroll_to(
        self,
        selector: Optional[str] = None,
        role: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        """
        Scroll to a specific element on the page identified by a selector or role.

        Parameters:
            selector (Optional[str]): The CSS selector of the element. Either selector or role must be provided.
            role (Optional[str]): The ARIA role of the element. Either selector or role must be provided.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

        Raises:
            ValueError: If both selector and role are provided or neither is provided.
        """
        r = reasoning or ""

        if selector and role:
            raise ValueError(
                "Only one of 'selector' or 'role' should be provided, not both."
            )

        if not selector and not role:
            raise ValueError("Either 'selector' or 'role' must be provided.")

        # If role is provided, construct a selector based on the role
        actual_selector = selector if selector else f'[role="{role}"]'

        element = await self.page.query_selector(actual_selector)
        if element:
            await element.scroll_into_view_if_needed()
        self.history.append(
            {
                "action": "scroll_to",
                "reasoning": r,
                "selector" if selector else "role": selector if selector else role,
            }
        )

    async def mouse_click(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
        use_smooth_movement: bool = False,
    ) -> None:
        """
        Click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse click.
            y (int): The y-coordinate for the mouse click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the click action.
            use_smooth_movement (bool): If True, uses smooth human-like movement before clicking. Defaults to False.

        Returns:
            None
        """
        r = reasoning or ""
        
        # Use smooth movement if requested
        if use_smooth_movement:
            await self.mouse_move_smooth(
                target=(x, y),
                reasoning=f"Smooth movement before click: {r}"
            )
        
        await self.page.mouse.click(x, y)
        self.history.append(
            {
                "action": "mouse_click",
                "reasoning": r,
                "x": x,
                "y": y,
                "use_smooth_movement": use_smooth_movement,
            }
        )

    async def mouse_dbclick(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
        use_smooth_movement: bool = False,
    ) -> None:
        """
        Perform a double click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse double click.
            y (int): The y-coordinate for the mouse double click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the double click action.
            use_smooth_movement (bool): If True, uses smooth human-like movement before double clicking. Defaults to False.

        Returns:
            None
        """
        r = reasoning or ""
        
        # Use smooth movement if requested
        if use_smooth_movement:
            await self.mouse_move_smooth(
                target=(x, y),
                reasoning=f"Smooth movement before double click: {r}"
            )
        
        await self.page.mouse.dblclick(x, y, timeout=timeout)
        self.history.append(
            {
                "action": "mouse_dbclick",
                "reasoning": r,
                "x": x,
                "y": y,
                "use_smooth_movement": use_smooth_movement,
            }
        )

    async def mouse_right_click(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
        use_smooth_movement: bool = False,
    ) -> None:
        """
        Perform a right click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse right click.
            y (int): The y-coordinate for the mouse right click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the right click action.
            use_smooth_movement (bool): If True, uses smooth human-like movement before right clicking. Defaults to False.

        Returns:
            None
        """
        r = reasoning or ""
        
        # Use smooth movement if requested
        if use_smooth_movement:
            await self.mouse_move_smooth(
                target=(x, y),
                reasoning=f"Smooth movement before right click: {r}"
            )
        
        await self.page.mouse.click(x, y, button="right", timeout=timeout)
        self.history.append(
            {
                "action": "mouse_right_click",
                "reasoning": r,
                "x": x,
                "y": y,
                "use_smooth_movement": use_smooth_movement,
            }
        )

    async def click(
        self,
        selector: Optional[str] = None,
        role: Optional[str] = None,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Click on a specified element on the page.

        Parameters:
            selector (Optional[str]): The CSS selector of the element to click. Either selector or role must be provided.
            role (Optional[str]): The ARIA role of the element to click. Either selector or role must be provided.
            reasoning (Optional[str]): The reasoning chain for this click action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.

        Raises:
            ValueError: If both selector and role are provided or neither is provided.
        """
        r = reasoning or ""

        if selector and role:
            raise ValueError(
                "Only one of 'selector' or 'role' should be provided, not both."
            )

        if not selector and not role:
            raise ValueError("Either 'selector' or 'role' must be provided.")

        # If role is provided, construct a selector based on the role
        actual_selector = selector if selector else f'[role="{role}"]'

        await self.page.click(actual_selector, timeout=timeout)
        self.history.append(
            {
                "action": "mouse_click",
                "reasoning": r,
                "selector" if selector else "role": selector if selector else role,
            }
        )

    async def mouse_move(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Move the mouse to the specified (x, y) coordinate on the page.

        This method performs a basic mouse move operation (similar to hovering over an element).

        Parameters:
            x (int): The x-coordinate to move the mouse to.
            y (int): The y-coordinate to move the mouse to.
            reasoning (Optional[str]): A description of the reasoning for this action.
            timeout (Optional[int]): Optional timeout in milliseconds for the move action.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.mouse.move(x, y, timeout=timeout)
        self.history.append(
            {
                "action": "mouse_move",
                "reasoning": r,
                "x": x,
                "y": y,
            }
        )

    async def mouse_move_smooth(
        self,
        target: Tuple[int, int],
        reasoning: Optional[str] = None,
        start: Optional[Tuple[int, int]] = None,
        smoothness_factor: float = 1.0,
        noise: float = 2.0,
        save_path: Optional[str] = None,
    ) -> Union[PILImage.Image, str]:
        """
        Moves the mouse pointer smoothly to a target coordinate using a dynamic human-like velocity profile.

        This method simulates a natural, human-like movement, which makes it harder for bot detectors to block the agent.
        After the move, a screenshot with an overlaid pointer icon is taken. This method is equivalent in effect to
        hovering over a coordinate but with the additional stealth benefit of natural movement.

        Parameters:
            target (Tuple[int, int]): The final (x, y) coordinate for the mouse pointer.
            reasoning (Optional[str]): A description of the reasoning behind this action.
            start (Optional[Tuple[int, int]]): The starting (x, y) coordinate (if not provided, retrieved from the page).
            smoothness_factor (float): Controls the smoothness of the movement (default is 1.0).
            noise (float): Maximum random noise added per movement step (in pixels, default is 2.0).
            save_path (Optional[str]): If provided, the annotated screenshot is saved to this path.

        Returns:
            Union[PILImage.Image, str]: The annotated screenshot as a PIL Image object or the file path if saved.
        """
        r = reasoning or ""
        result = await mouse_move_smooth(
            page=self.page,
            target=target,
            start=start,
            smoothness_factor=smoothness_factor,
            noise=noise,
            save_path=save_path,
        )
        self.history.append(
            {
                "action": "mouse_move_smooth",
                "reasoning": r,
                "target": target,
                "start": start,
                "smoothness_factor": smoothness_factor,
                "noise": noise,
                "save_path": save_path,
                "output": (
                    "screenshot saved"
                    if isinstance(result, str)
                    else "PIL Image returned"
                ),
            }
        )
        return result

    async def keyboard_type(
        self,
        text: str,
        reasoning: Optional[str] = None,
        delay: Optional[int] = None,
    ) -> None:
        """
        Type the provided text using the keyboard, sending the input to the element that currently has focus.

        Parameters:
            text (str): The text to type using the keyboard.
            reasoning (Optional[str]): A description of the reasoning behind this action. Defaults to an empty string.
            delay (Optional[int]): Optional delay (in milliseconds) between keystrokes for a more natural typing simulation.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.keyboard.type(text, delay=delay)
        self.history.append(
            {
                "action": "keyboard_type",
                "reasoning": r,
                "text": text,
                "delay": delay,
            }
        )

    async def keyboard_press(
        self,
        key: str,
        reasoning: Optional[str] = None,
        delay: Optional[int] = None,
    ) -> None:
        """
        Press a special keyboard key (such as 'Enter', 'Escape', etc.) on the page using the Playwright keyboard API.

        This method sends a key press event to the element that currently has focus (or globally if no specific element is focused).
        It validates that the provided key is one of the recognized special keys.

        Parameters:
            key (str): The special key to press (e.g., "Enter", "Escape", "ArrowUp", etc.). Must be one of the allowed keys.
            reasoning (Optional[str]): A description of the reasoning for this action. Defaults to an empty string.
            delay (Optional[int]): Optional delay (in milliseconds) before releasing the key, for a more natural simulation.

        Raises:
            ValueError: If the provided key is not one of the recognized special keys.

        Returns:
            None
        """
        allowed_keys = {
            "Enter",
            "Tab",
            "Escape",
            "ArrowUp",
            "ArrowDown",
            "ArrowLeft",
            "ArrowRight",
            "Backspace",
            "Delete",
            "Home",
            "End",
            "PageUp",
            "PageDown",
            "Insert",
            "Meta",
            "Shift",
            "Control",
            "Alt",
        }
        if key not in allowed_keys:
            raise ValueError(
                f"Special key '{key}' is not recognized. Allowed keys: {sorted(allowed_keys)}"
            )

        r = reasoning or ""
        await self.page.keyboard.press(key, delay=delay)
        self.history.append(
            {
                "action": "keyboard_press",
                "reasoning": r,
                "key": key,
                "delay": delay,
            }
        )

    async def dbclick(
        self,
        selector: Optional[str] = None,
        role: Optional[str] = None,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Double click on a specified element on the page.

        Parameters:
            selector (Optional[str]): The CSS selector of the element to double click.
            role (Optional[str]): The ARIA role of the element to double click.
            reasoning (Optional[str]): The reasoning chain for this double click action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.

        Raises:
            ValueError: If both selector and role are provided or neither is provided.
        """
        r = reasoning or ""
        if selector and role:
            raise ValueError(
                "Only one of 'selector' or 'role' should be provided, not both."
            )
        if not selector and not role:
            raise ValueError("Either 'selector' or 'role' must be provided.")
        actual_selector = selector if selector else f'[role="{role}"]'

        await self.page.dblclick(actual_selector, timeout=timeout)
        self.history.append(
            {
                "action": "mouse_dblclick",
                "reasoning": r,
                "selector" if selector else "role": selector if selector else role,
            }
        )

    async def input_text(
        self,
        text: str,
        selector: Optional[str] = None,
        role: Optional[str] = None,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Input text into a specified element on the page.

        Parameters:
            text (str): The text to be input.
            selector (Optional[str]): The CSS selector of the element. Either selector or role must be provided.
            role (Optional[str]): The ARIA role of the element. Either selector or role must be provided.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.

        Raises:
            ValueError: If both selector and role are provided or neither is provided.
        """
        r = reasoning or ""

        if selector and role:
            raise ValueError(
                "Only one of 'selector' or 'role' should be provided, not both."
            )

        if not selector and not role:
            raise ValueError("Either 'selector' or 'role' must be provided.")

        # If role is provided, construct a selector based on the role
        actual_selector = selector if selector else f'[role="{role}"]'

        await self.page.fill(actual_selector, text, timeout=timeout)
        self.history.append(
            {
                "action": "input_text",
                "reasoning": r,
                "selector" if selector else "role": selector if selector else role,
                "text": text,
            }
        )

    async def hover(
        self,
        selector: Optional[str] = None,
        role: Optional[str] = None,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Hover over a specified element on the page.

        Parameters:
            selector (Optional[str]): The CSS selector of the element. Either selector or role must be provided.
            role (Optional[str]): The ARIA role of the element. Either selector or role must be provided.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.

        Raises:
            ValueError: If both selector and role are provided or neither is provided.
        """
        r = reasoning or ""

        if selector and role:
            raise ValueError(
                "Only one of 'selector' or 'role' should be provided, not both."
            )

        if not selector and not role:
            raise ValueError("Either 'selector' or 'role' must be provided.")

        # If role is provided, construct a selector based on the role
        actual_selector = selector if selector else f'[role="{role}"]'

        await self.page.hover(actual_selector, timeout=timeout)
        self.history.append(
            {
                "action": "hover",
                "reasoning": r,
                "selector" if selector else "role": selector if selector else role,
            }
        )

    async def drag_and_drop(
        self,
        source_selector: Optional[str] = None,
        source_role: Optional[str] = None,
        target_selector: Optional[str] = None,
        target_role: Optional[str] = None,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Drag an element from a source selector/role and drop it onto a target selector/role.

        Parameters:
            source_selector (Optional[str]): The CSS selector of the source element.
            source_role (Optional[str]): The ARIA role of the source element.
            target_selector (Optional[str]): The CSS selector of the target element.
            target_role (Optional[str]): The ARIA role of the target element.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.

        Raises:
            ValueError: If both source_selector and source_role are provided or neither is provided.
            ValueError: If both target_selector and target_role are provided or neither is provided.
        """
        r = reasoning or ""

        if (source_selector and source_role) or (
            not source_selector and not source_role
        ):
            raise ValueError(
                "Provide either 'source_selector' or 'source_role' (but not both) for the source element."
            )

        if (target_selector and target_role) or (
            not target_selector and not target_role
        ):
            raise ValueError(
                "Provide either 'target_selector' or 'target_role' (but not both) for the target element."
            )

        actual_source = (
            source_selector if source_selector else f'[role="{source_role}"]'
        )
        actual_target = (
            target_selector if target_selector else f'[role="{target_role}"]'
        )

        await self.page.drag_and_drop(actual_source, actual_target, timeout=timeout)
        self.history.append(
            {
                "action": "drag_and_drop",
                "reasoning": r,
                "source": source_selector if source_selector else source_role,
                "target": target_selector if target_selector else target_role,
            }
        )

    async def screenshot(
        self,
        filename: Optional[str] = None,
        reasoning: Optional[str] = None,
        highlight_bbox: bool = True,
        save_dir: Optional[str] = None,
    ) -> Union[str, PILImage.Image]:
        """
        Take a screenshot of the current page.

        If a filename is provided, the screenshot is saved to that file in the screenshot directory and the full file path is returned.
        If no filename is provided (i.e. filename is None), the screenshot is returned as a PIL Image object.

        Additionally, if 'highlight_bbox' is True, interactive elements on the page are highlighted with bounding boxes.

        Parameters:
            filename (Optional[str]): The filename to save the screenshot.
                                      If None, the screenshot is not saved to disk but returned as a PIL Image.
            reasoning (Optional[str]): A description of the reasoning for taking this screenshot.
            highlight_bbox (bool): Whether to highlight interactive elements in the screenshot. Defaults to True.
            save_dir (Optional[str]): Override the default screenshot directory for this screenshot.

        Returns:
            Union[str, PILImage.Image]: The full file path where the screenshot is saved, or the PIL Image object if no filename is provided.
        """
        r = reasoning or ""
        if highlight_bbox:
            result = await highlight_interactive_elements(
                self.page,
                visible_only=True,
                output_image=True,
                draw_center_dot=False,
                output_details=False,
                save_path=None,
            )
            image = result.get("image", None)
            if not image and filename:
                image = PILImage.open(filename)
        else:
            screenshot_bytes = await self.page.screenshot()
            image = PILImage.open(io.BytesIO(screenshot_bytes))

        if filename:
            # Use provided save_dir or default to self.screenshot_path
            screenshot_dir = save_dir or self.screenshot_path
            os.makedirs(screenshot_dir, exist_ok=True)

            # Ensure filename has a valid extension
            if not os.path.splitext(filename)[1]:
                filename = filename + ".png"

            full_path = os.path.join(screenshot_dir, filename)
            image.save(full_path)
            self.history.append(
                {
                    "action": "screenshot",
                    "reasoning": r,
                    "filename": filename,
                    "path": full_path,
                }
            )
            return full_path

        self.history.append(
            {
                "action": "screenshot",
                "reasoning": r,
                "output": "PIL Image object returned",
            }
        )
        return image

    async def on_download(self, filename: str, reasoning: Optional[str] = None) -> None:
        """
        Wait for a download event and save the downloaded file in the download directory.

        Parameters:
            filename (str): The filename to save the downloaded file.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
        """
        r = reasoning or ""
        download = await self.page.wait_for_event("download")
        path = os.path.join(self.download_path, filename)
        await download.save_as(path)
        self.history.append(
            {
                "action": "download",
                "reasoning": r,
                "filename": filename,
                "path": self.download_path,
            }
        )

    async def get_html(self, reasoning: Optional[str] = None) -> str:
        """
        Retrieve the HTML source code of the current page as text.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

        Returns:
            str: The current page HTML source.
        """
        r = reasoning or ""
        # # Wait for the body element to ensure the page is fully loaded
        # await self.page.wait_for_selector("body", timeout=5000)
        # # Get the inner HTML of the body element
        # body_locator = self.page.locator("body")
        # html = await body_locator.inner_html()
        # first wait until the page is fully loaded
        await self.page.wait_for_load_state("networkidle")
        # then get the html content
        html = await self.page.content()
        self.history.append(
            {
                "action": "get_html",
                "reasoning": r,
                "output": html,
            }
        )
        return html

    async def get_text(self, selector: str, timeout: Optional[int] = None, reasoning: Optional[str] = None) -> str:
        """
        Get text content of an element.

        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait for element in milliseconds
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Text content of the element
        """
        r = reasoning or ""
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                text = await element.text_content()
                result = text or ""
            else:
                result = ""
        except Exception as e:
            result = f"Error: Failed to get text from selector '{selector}': {e}"
        
        self.history.append(
            {
                "action": "get_text",
                "reasoning": r,
                "selector": selector,
                "output": result[:200] + "..." if len(result) > 200 else result,
            }
        )
        return result

    async def get_attribute(self, selector: str, attribute: str, timeout: Optional[int] = None, reasoning: Optional[str] = None) -> str:
        """
        Get an attribute value from an element.

        Args:
            selector: CSS selector for the element
            attribute: Name of the attribute to get
            timeout: Maximum time to wait for element in milliseconds
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Attribute value or empty string if not found
        """
        r = reasoning or ""
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                value = await element.get_attribute(attribute)
                result = value or ""
            else:
                result = ""
        except Exception as e:
            result = f"Error: Failed to get attribute '{attribute}' from selector '{selector}': {e}"
        
        self.history.append(
            {
                "action": "get_attribute",
                "reasoning": r,
                "selector": selector,
                "attribute": attribute,
                "output": result,
            }
        )
        return result

    async def wait_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = "visible", reasoning: Optional[str] = None) -> str:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector to wait for
            timeout: Maximum time to wait in milliseconds
            state: State to wait for ('visible', 'hidden', 'attached', 'detached')
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Success message
        """
        r = reasoning or ""
        try:
            await self.page.wait_for_selector(selector, timeout=timeout, state=state)
            result = f"Element matching '{selector}' is now {state}"
        except Exception as e:
            result = f"Error: Timeout waiting for selector '{selector}' to be {state}: {e}"
        
        self.history.append(
            {
                "action": "wait_for_selector",
                "reasoning": r,
                "selector": selector,
                "state": state,
                "output": result,
            }
        )
        return result

    async def go_forward(self, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Navigate forward in the browser history.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.go_forward(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "go_forward",
                "reasoning": r,
                "output": title,
            }
        )

    async def reload(self, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Reload the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.reload(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "reload",
                "reasoning": r,
                "output": title,
            }
        )

    async def get_url(self, reasoning: Optional[str] = None) -> str:
        """
        Get the current page URL.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            str: Current page URL
        """
        r = reasoning or ""
        url = self.page.url
        self.history.append(
            {
                "action": "get_url",
                "reasoning": r,
                "output": url,
            }
        )
        return url

    async def get_title(self, reasoning: Optional[str] = None) -> str:
        """
        Get the current page title.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            str: Current page title
        """
        r = reasoning or ""
        title = await self.page.title()
        self.history.append(
            {
                "action": "get_title",
                "reasoning": r,
                "output": title,
            }
        )
        return title

    async def extract_links(self, reasoning: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract all links from the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            List of dictionaries containing link information
        """
        r = reasoning or ""
        links = await self.page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links.map(link => ({
                    text: link.textContent.trim(),
                    href: link.href,
                    title: link.title || ''
                }));
            }
        """)
        self.history.append(
            {
                "action": "extract_links",
                "reasoning": r,
                "output": f"Found {len(links)} links",
            }
        )
        return links

    async def fill_form(self, form_data: Dict[str, str], reasoning: Optional[str] = None) -> None:
        """
        Fill out a form with the provided data.

        Parameters:
            form_data: Dictionary mapping field names/selectors to values
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        for selector, value in form_data.items():
            await self.page.fill(selector, value)
        self.history.append(
            {
                "action": "fill_form",
                "reasoning": r,
                "form_data": form_data,
            }
        )

    async def select_option(self, selector: str, value: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Select an option from a dropdown.

        Parameters:
            selector: CSS selector for the select element
            value: Value to select
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.select_option(selector, value, timeout=timeout)
        self.history.append(
            {
                "action": "select_option",
                "reasoning": r,
                "selector": selector,
                "value": value,
            }
        )

    async def check_checkbox(self, selector: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Check a checkbox.

        Parameters:
            selector: CSS selector for the checkbox
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.check(selector, timeout=timeout)
        self.history.append(
            {
                "action": "check_checkbox",
                "reasoning": r,
                "selector": selector,
            }
        )

    async def uncheck_checkbox(self, selector: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Uncheck a checkbox.

        Parameters:
            selector: CSS selector for the checkbox
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.uncheck(selector, timeout=timeout)
        self.history.append(
            {
                "action": "uncheck_checkbox",
                "reasoning": r,
                "selector": selector,
            }
        )

    async def get_text_original(self, selector: str, timeout: Optional[int] = None) -> str:
        """
        Get text content of an element.

        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait for element in milliseconds

        Returns:
            Text content of the element
        """
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                text = await element.text_content()
                return text or ""
            return ""
        except Exception as e:
            raise RuntimeError(f"Failed to get text from selector '{selector}': {e}")

    async def get_attribute(self, selector: str, attribute: str, timeout: Optional[int] = None) -> str:
        """
        Get an attribute value from an element.

        Args:
            selector: CSS selector for the element
            attribute: Name of the attribute to get
            timeout: Maximum time to wait for element in milliseconds

        Returns:
            Attribute value or empty string if not found
        """
        try:
            element = await self.page.wait_for_selector(selector, timeout=timeout)
            if element:
                value = await element.get_attribute(attribute)
                return value or ""
            return ""
        except Exception as e:
            raise RuntimeError(f"Failed to get attribute '{attribute}' from selector '{selector}': {e}")

    async def wait_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = "visible") -> str:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector to wait for
            timeout: Maximum time to wait in milliseconds
            state: State to wait for ('visible', 'hidden', 'attached', 'detached')

        Returns:
            Success message
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout, state=state)
            return f"Element matching '{selector}' is now {state}"
        except Exception as e:
            raise RuntimeError(f"Timeout waiting for selector '{selector}' to be {state}: {e}")

    async def go_forward(self, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Navigate forward in the browser history.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.go_forward(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "go_forward",
                "reasoning": r,
                "output": title,
            }
        )

    async def reload(self, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Reload the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.reload(timeout=timeout)
        title = await self.page.title()
        self.history.append(
            {
                "action": "reload",
                "reasoning": r,
                "output": title,
            }
        )

    async def get_url(self, reasoning: Optional[str] = None) -> str:
        """
        Get the current page URL.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            str: Current page URL
        """
        r = reasoning or ""
        url = self.page.url
        self.history.append(
            {
                "action": "get_url",
                "reasoning": r,
                "output": url,
            }
        )
        return url

    async def get_title(self, reasoning: Optional[str] = None) -> str:
        """
        Get the current page title.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            str: Current page title
        """
        r = reasoning or ""
        title = await self.page.title()
        self.history.append(
            {
                "action": "get_title",
                "reasoning": r,
                "output": title,
            }
        )
        return title

    async def extract_links(self, reasoning: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Extract all links from the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            List of dictionaries containing link information
        """
        r = reasoning or ""
        links = await self.page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links.map(link => ({
                    text: link.textContent.trim(),
                    href: link.href,
                    title: link.title || ''
                }));
            }
        """)
        self.history.append(
            {
                "action": "extract_links",
                "reasoning": r,
                "output": f"Found {len(links)} links",
            }
        )
        return links

    async def fill_form(self, form_data: Dict[str, str], reasoning: Optional[str] = None) -> None:
        """
        Fill out a form with the provided data.

        Parameters:
            form_data: Dictionary mapping field names/selectors to values
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        for selector, value in form_data.items():
            await self.page.fill(selector, value)
        self.history.append(
            {
                "action": "fill_form",
                "reasoning": r,
                "form_data": form_data,
            }
        )

    async def select_option(self, selector: str, value: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Select an option from a dropdown.

        Parameters:
            selector: CSS selector for the select element
            value: Value to select
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.select_option(selector, value, timeout=timeout)
        self.history.append(
            {
                "action": "select_option",
                "reasoning": r,
                "selector": selector,
                "value": value,
            }
        )

    async def check_checkbox(self, selector: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Check a checkbox.

        Parameters:
            selector: CSS selector for the checkbox
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.check(selector, timeout=timeout)
        self.history.append(
            {
                "action": "check_checkbox",
                "reasoning": r,
                "selector": selector,
            }
        )

    async def uncheck_checkbox(self, selector: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Uncheck a checkbox.

        Parameters:
            selector: CSS selector for the checkbox
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.uncheck(selector, timeout=timeout)
        self.history.append(
            {
                "action": "uncheck_checkbox",
                "reasoning": r,
                "selector": selector,
            }
        )

    async def press_key(self, key: str, reasoning: Optional[str] = None, delay: Optional[int] = None) -> None:
        """
        Press a key.

        Parameters:
            key: Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')
            reasoning (Optional[str]): The reasoning chain for this action.
            delay (Optional[int]): Delay between key presses in milliseconds.
        """
        r = reasoning or ""
        await self.page.keyboard.press(key, delay=delay)
        self.history.append(
            {
                "action": "press_key",
                "reasoning": r,
                "key": key,
            }
        )

    async def wait_for_navigation(self, timeout: Optional[int] = None, reasoning: Optional[str] = None) -> None:
        """
        Wait for navigation to complete.

        Parameters:
            timeout (Optional[int]): Optional timeout in milliseconds.
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.wait_for_load_state("networkidle", timeout=timeout)
        self.history.append(
            {
                "action": "wait_for_navigation",
                "reasoning": r,
            }
        )

    async def evaluate_javascript(self, script: str, reasoning: Optional[str] = None) -> Any:
        """
        Execute JavaScript in the page context.

        Parameters:
            script: JavaScript code to execute
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Result of the JavaScript execution
        """
        r = reasoning or ""
        result = await self.page.evaluate(script)
        self.history.append(
            {
                "action": "evaluate_javascript",
                "reasoning": r,
                "script": script[:100] + "..." if len(script) > 100 else script,
                "output": str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
            }
        )
        return result

    async def get_cookies(self, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all cookies for the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            List of cookie dictionaries
        """
        r = reasoning or ""
        cookies = await self.context.cookies()
        self.history.append(
            {
                "action": "get_cookies",
                "reasoning": r,
                "output": f"Found {len(cookies)} cookies",
            }
        )
        return cookies

    async def set_cookie(self, cookie: Dict[str, Any], reasoning: Optional[str] = None) -> None:
        """
        Set a cookie.

        Parameters:
            cookie: Cookie dictionary with name, value, domain, etc.
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.context.add_cookies([cookie])
        self.history.append(
            {
                "action": "set_cookie",
                "reasoning": r,
                "cookie": cookie,
            }
        )

    async def delete_cookies(self, reasoning: Optional[str] = None) -> None:
        """
        Delete all cookies.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.context.clear_cookies()
        self.history.append(
            {
                "action": "delete_cookies",
                "reasoning": r,
            }
        )

    async def get_local_storage(self, reasoning: Optional[str] = None) -> Dict[str, str]:
        """
        Get all local storage items.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Dictionary of local storage key-value pairs
        """
        r = reasoning or ""
        storage = await self.page.evaluate("""
            () => {
                const storage = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    storage[key] = localStorage.getItem(key);
                }
                return storage;
            }
        """)
        self.history.append(
            {
                "action": "get_local_storage",
                "reasoning": r,
                "output": f"Found {len(storage)} items",
            }
        )
        return storage

    async def set_local_storage(self, key: str, value: str, reasoning: Optional[str] = None) -> None:
        """
        Set a local storage item.

        Parameters:
            key: Storage key
            value: Storage value
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.evaluate(f"localStorage.setItem('{key}', '{value}')")
        self.history.append(
            {
                "action": "set_local_storage",
                "reasoning": r,
                "key": key,
                "value": value,
            }
        )

    async def clear_local_storage(self, reasoning: Optional[str] = None) -> None:
        """
        Clear all local storage.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.evaluate("localStorage.clear()")
        self.history.append(
            {
                "action": "clear_local_storage",
                "reasoning": r,
            }
        )

    async def download_file(self, url: str, filename: Optional[str] = None, reasoning: Optional[str] = None) -> str:
        """
        Download a file from a URL.

        Parameters:
            url: URL to download
            filename: Optional filename to save as
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Path to the downloaded file
        """
        r = reasoning or ""
        
        # Start waiting for download before clicking
        async with self.page.expect_download() as download_info:
            # Navigate to the URL to trigger download
            await self.page.goto(url)
        
        download = await download_info.value
        
        # Determine filename
        if not filename:
            filename = download.suggested_filename or "downloaded_file"
        
        # Save the file
        file_path = os.path.join(self.download_path, filename)
        await download.save_as(file_path)
        
        self.history.append(
            {
                "action": "download_file",
                "reasoning": r,
                "url": url,
                "filename": filename,
                "path": file_path,
            }
        )
        return file_path

    async def get_clean_html(
        self, 
        selector: Optional[str] = None, 
        max_text_length: Optional[int] = None,
        preserve_structure: bool = True,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Get cleaned HTML content with unnecessary elements removed.

        Parameters:
            selector: Optional CSS selector to target specific element
            max_text_length: Maximum length of text content to return
            preserve_structure: Whether to preserve HTML structure
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Cleaned HTML content
        """
        r = reasoning or ""
        
        # Get HTML content
        if selector:
            element = await self.page.query_selector(selector)
            if element:
                html = await element.inner_html()
            else:
                html = ""
        else:
            html = await self.page.content()
        
        # Clean the HTML
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()
            
            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Get text or HTML based on preserve_structure
            if preserve_structure:
                cleaned = str(soup)
            else:
                cleaned = soup.get_text(separator=' ', strip=True)
            
            # Truncate if needed
            if max_text_length and len(cleaned) > max_text_length:
                cleaned = cleaned[:max_text_length] + "..."
        else:
            cleaned = ""
        
        self.history.append(
            {
                "action": "get_clean_html",
                "reasoning": r,
                "selector": selector,
                "output_length": len(cleaned),
            }
        )
        return cleaned

    async def html_to_markdown(
        self, 
        selector: str = "body",
        preserve_links: bool = True,
        preserve_tables: bool = True,
        preserve_images: bool = True,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Convert HTML content to Markdown format.

        Parameters:
            selector: CSS selector for the element to convert
            preserve_links: Whether to preserve links in markdown format
            preserve_tables: Whether to preserve table formatting
            preserve_images: Whether to preserve image references
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Markdown formatted content
        """
        r = reasoning or ""
        
        # Get HTML content
        element = await self.page.query_selector(selector)
        if element:
            html = await element.inner_html()
        else:
            html = await self.page.content()
        
        # Convert to markdown
        markdown = markdownify.markdownify(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
            convert=["p", "h1", "h2", "h3", "h4", "h5", "h6", "br", "strong", "em", "ul", "ol", "li"] +
                   (["a"] if preserve_links else []) +
                   (["table", "thead", "tbody", "tr", "th", "td"] if preserve_tables else []) +
                   (["img"] if preserve_images else [])
        )
        
        self.history.append(
            {
                "action": "html_to_markdown",
                "reasoning": r,
                "selector": selector,
                "output_length": len(markdown),
            }
        )
        return markdown

    async def fill(self, selector: str, text: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Fill an input field with text (alias for input_text for compatibility).

        Parameters:
            selector: CSS selector for the input field
            text: Text to fill
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        await self.input_text(text, selector=selector, reasoning=reasoning, timeout=timeout)

    async def press(self, selector: str, key: str, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> None:
        """
        Press a key on a specific element.

        Parameters:
            selector: CSS selector for the element
            key: Key to press
            reasoning (Optional[str]): The reasoning chain for this action.
            timeout (Optional[int]): Optional timeout in milliseconds.
        """
        r = reasoning or ""
        await self.page.press(selector, key, timeout=timeout)
        self.history.append(
            {
                "action": "press",
                "reasoning": r,
                "selector": selector,
                "key": key,
            }
        )

    async def get_attribute_all(self, selector: str, attribute: str, reasoning: Optional[str] = None) -> List[str]:
        """
        Get an attribute value from all matching elements.

        Parameters:
            selector: CSS selector for the elements
            attribute: Name of the attribute to get
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            List of attribute values
        """
        r = reasoning or ""
        values = await self.page.evaluate(f"""
            () => {{
                const elements = document.querySelectorAll('{selector}');
                return Array.from(elements).map(el => el.getAttribute('{attribute}') || '');
            }}
        """)
        self.history.append(
            {
                "action": "get_attribute_all",
                "reasoning": r,
                "selector": selector,
                "attribute": attribute,
                "count": len(values),
            }
        )
        return values

    async def open_new_tab(self, url: str, reasoning: Optional[str] = None) -> str:
        """
        Open a new tab and navigate to URL.

        Parameters:
            url: URL to navigate to in the new tab
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Success message with tab count
        """
        r = reasoning or ""
        new_page = await self.context.new_page()
        await new_page.goto(url)
        
        # Update the current page reference to the new tab
        self.page = new_page
        
        tab_count = len(self.context.pages)
        self.history.append(
            {
                "action": "open_new_tab",
                "reasoning": r,
                "url": url,
                "tab_count": tab_count,
            }
        )
        return f"Opened new tab and navigated to {url}. Total tabs: {tab_count}"

    async def get_page_count(self, reasoning: Optional[str] = None) -> int:
        """
        Get the number of open tabs/pages.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Number of open pages
        """
        r = reasoning or ""
        count = len(self.context.pages)
        self.history.append(
            {
                "action": "get_page_count",
                "reasoning": r,
                "count": count,
            }
        )
        return count

    async def switch_to_tab(self, index: int, reasoning: Optional[str] = None) -> str:
        """
        Switch to a tab by index.

        Parameters:
            index: Zero-based index of the tab to switch to
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Success message with new tab URL
        """
        r = reasoning or ""
        pages = self.context.pages
        if 0 <= index < len(pages):
            self.page = pages[index]
            await self.page.bring_to_front()
            url = self.page.url
            self.history.append(
                {
                    "action": "switch_to_tab",
                    "reasoning": r,
                    "index": index,
                    "url": url,
                }
            )
            return f"Switched to tab {index}: {url}"
        else:
            raise IndexError(f"Tab index {index} out of range. Available tabs: 0-{len(pages)-1}")

    async def extract_content_from_url(
        self, 
        url: str, 
        selector: str = "main, article, .content, .post, .entry",
        max_text_length: Optional[int] = 5000,
        return_markdown: bool = True,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced version with anti-bot evasion and retry strategies.

        Parameters:
            url: URL to extract content from
            selector: CSS selector for main content area
            max_text_length: Maximum length of extracted text
            return_markdown: Whether to return content as markdown
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Dictionary with extracted content information
        """
        import random
        import asyncio
        
        r = reasoning or ""
        
        # Save current URL to return to later
        original_url = self.page.url
        
        # Define retry strategies with different user agents and settings
        strategies = [
            {
                'name': 'standard_desktop',
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'viewport': {'width': 1920, 'height': 1080},
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            },
            {
                'name': 'mobile_safari',
                'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1',
                'viewport': {'width': 390, 'height': 844},
                'headers': {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9'
                }
            },
            {
                'name': 'googlebot',  # Some sites allow Googlebot
                'user_agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
                'viewport': {'width': 1024, 'height': 768}
            }
        ]
        
        for i, strategy in enumerate(strategies):
            context = None
            try:
                logger.info(f"Attempting extraction with strategy: {strategy['name']} (attempt {i+1}/{len(strategies)})")
                
                # Create new context with user agent
                context = await self.browser.new_context(
                    user_agent=strategy['user_agent'],
                    viewport=strategy['viewport'],
                    extra_http_headers=strategy.get('headers', {})
                )
                page = await context.new_page()
                
                # Add random delay to appear more human
                await asyncio.sleep(random.uniform(0.5, 2.0))
                
                # First, check if it's a PDF
                try:
                    head_response = await page.request.head(url)
                    content_type = head_response.headers.get('content-type', '').lower()
                except:
                    content_type = ''
                
                is_pdf = (
                    'application/pdf' in content_type or
                    url.lower().endswith('.pdf') or
                    '/pdf/' in url.lower()
                )
                
                if is_pdf:
                    # Use dedicated PDF extraction
                    result = await self.extract_pdf_content_from_url(
                        url,
                        output_format="markdown" if return_markdown else "json",
                        max_text_length=max_text_length,
                        reasoning=reasoning
                    )
                    await context.close()
                    return result
                
                # Navigate with timeout
                response = await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                
                # Check HTTP status
                if response and response.status >= 400:
                    logger.warning(f"HTTP {response.status} for {url}")
                    await context.close()
                    continue
                
                # Wait for content with shorter timeout
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                except:
                    # Try body as fallback
                    await page.wait_for_selector('body', timeout=2000)
                
                # Check for bot detection
                content = await page.content()
                if self._is_bot_blocked(content):
                    logger.warning(f"Bot detection triggered with strategy {strategy['name']}")
                    await context.close()
                    continue
                
                # Extract and validate content
                extracted = await self._extract_with_validation(page, selector, max_text_length, return_markdown)
                
                if extracted and len(extracted.get('content', '')) > 100:
                    logger.info(f"Successfully extracted content with strategy: {strategy['name']}")
                    
                    result = {
                        "success": True,
                        "url": url,
                        "title": extracted['title'],
                        "content": extracted['content'],
                        "content_length": extracted['length'],
                        "format": "markdown" if return_markdown else "text",
                        "type": "html",
                        "strategy_used": strategy['name']
                    }
                    
                    self.history.append(
                        {
                            "action": "extract_content_from_url",
                            "reasoning": r,
                            "url": url,
                            "title": extracted['title'],
                            "content_length": extracted['length'],
                            "type": "html",
                            "success": True,
                            "strategy": strategy['name']
                        }
                    )
                    
                    await context.close()
                    return result
                
                await context.close()
                
            except Exception as e:
                logger.warning(f"Strategy {strategy['name']} failed: {str(e)[:100]}")
                if context:
                    await context.close()
                continue
        
        # All strategies failed
        logger.error(f"All extraction strategies failed for {url}")
        error_msg = f"Unable to extract content from {url} after {len(strategies)} attempts"
        
        self.history.append(
            {
                "action": "extract_content_from_url",
                "reasoning": r,
                "url": url,
                "success": False,
                "error": error_msg
            }
        )
        
        return {
            "success": False,
            "error": error_msg,
            "url": url,
            "message": error_msg
        }
        
        # Note: removed finally block since we're not navigating with self.page anymore

    def _is_bot_blocked(self, content: str) -> bool:
        """Check for common bot detection/blocking indicators."""
        if not content:
            return False
            
        content_lower = content.lower()
        
        # Bot detection services
        indicators = [
            # Cloudflare
            'cf-browser-verification',
            'cloudflare ray id',
            'cf_clearance',
            'checking your browser',
            'ddos protection by cloudflare',
            
            # Incapsula/Imperva
            'incapsula incident',
            'incap_ses',
            'visid_incap',
            
            # Distil Networks
            'distil_referrer',
            'distilnetworks',
            
            # PerimeterX
            'perimeterx',
            '_px',
            
            # DataDome
            'datadome',
            
            # Generic blocks
            '403 forbidden',
            'access denied',
            'permission denied',
            'unauthorized access',
            'bot detection',
            'are you a robot',
            'prove you are human'
        ]
        
        for indicator in indicators:
            if indicator in content_lower:
                logger.debug(f"Bot detection indicator found: {indicator}")
                return True
        
        # Check for suspiciously short content that might be a block page
        text_only = re.sub(r'<[^>]+>', '', content)  # Strip HTML
        if len(text_only.strip()) < 100 and any(word in text_only.lower() for word in ['blocked', 'denied', 'forbidden']):
            return True
        
        return False

    async def _extract_with_validation(self, page, selector, max_text_length, return_markdown):
        """Extract content and validate it's meaningful."""
        try:
            # Try primary selector
            element = await page.query_selector(selector)
            if not element:
                # Fallback to body
                element = await page.query_selector('body')
            
            if element:
                if return_markdown:
                    html = await element.inner_html()
                    import markdownify
                    content = markdownify.markdownify(html, strip=['script', 'style'])
                else:
                    content = await element.text_content()
                
                # Validate content is not just JavaScript
                if content:
                    # Check if content is mostly JavaScript/minified code
                    js_indicators = ['function(', 'var ', 'const ', 'let ', '=>', '});', 'window.', 'document.']
                    js_count = sum(1 for ind in js_indicators if ind in content[:500])
                    
                    if js_count > 3:
                        logger.warning("Content appears to be JavaScript code, not article text")
                        return None
                    
                    # Truncate if needed
                    if max_text_length and len(content) > max_text_length:
                        content = content[:max_text_length] + "..."
                    
                    title = await page.title()
                    
                    return {
                        'url': page.url,
                        'title': title,
                        'content': content,
                        'length': len(content)
                    }
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return None

    async def extract_pdf_content_from_url(
        self,
        url: str,
        output_format: str = "markdown",  # "markdown" or "json"
        max_text_length: Optional[int] = 10000,
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract text content from PDF URLs with proper loading.
        
        Parameters:
            url: URL of the PDF document
            output_format: Return format - "markdown" or "json"
            max_text_length: Maximum length of extracted text
            reasoning: The reasoning chain for this action
            
        Returns:
            Dictionary with extracted content and metadata
        """
        import io
        
        r = reasoning or ""
        
        try:
            # Method 1: Use request context for more reliable PDF download
            # This ensures we get the complete PDF content
            response = await self.page.request.get(url)
            
            # Check if request was successful
            if not response.ok:
                raise Exception(f"Failed to fetch PDF: HTTP {response.status}")
            
            # Get the complete PDF content
            pdf_bytes = await response.body()
            
            # Verify we have actual PDF content
            if not pdf_bytes or len(pdf_bytes) < 100:
                raise Exception("Received empty or invalid PDF content")
            
            # Check PDF header (PDF files start with %PDF)
            if not pdf_bytes[:5] == b'%PDF-':
                raise Exception("Content is not a valid PDF file")
            
            # Try to extract text using available PDF libraries
            extracted_text = ""
            
            # Try pdfminer.six first (better for complex PDFs)
            try:
                from pdfminer.high_level import extract_text_to_fp
                from pdfminer.layout import LAParams
                
                output_string = io.StringIO()
                extract_text_to_fp(
                    io.BytesIO(pdf_bytes),
                    output_string,
                    laparams=LAParams(),
                    output_type='text',
                    codec='utf-8'
                )
                extracted_text = output_string.getvalue()
                
            except ImportError:
                # Fallback to PyPDF2 if pdfminer.six is not available
                try:
                    from PyPDF2 import PdfReader
                    
                    pdf_file = io.BytesIO(pdf_bytes)
                    reader = PdfReader(pdf_file)
                    for page in reader.pages:
                        extracted_text += page.extract_text() + "\n"
                        
                except ImportError:
                    raise Exception("No PDF parsing library available. Please install pdfminer.six or PyPDF2.")
            
            # Truncate if needed
            if max_text_length and len(extracted_text) > max_text_length:
                extracted_text = extracted_text[:max_text_length] + "..."
            
            # Format output
            if output_format == "markdown":
                # Convert to markdown format with title from URL
                title = url.split('/')[-1].replace('.pdf', '')
                content = f"# PDF Document: {title}\n\n**Source:** {url}\n\n---\n\n{extracted_text}"
            else:  # json
                content = {
                    "url": url,
                    "text": extracted_text,
                    "format": "pdf",
                    "length": len(extracted_text)
                }
            
            result = {
                "success": True,
                "content": content,
                "url": url,
                "format": output_format,
                "type": "pdf"
            }
            
            self.history.append(
                {
                    "action": "extract_pdf_content_from_url",
                    "reasoning": r,
                    "url": url,
                    "content_length": len(extracted_text),
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            # Fallback: Try navigation method with proper waiting
            try:
                response = await self.page.goto(url, wait_until="networkidle")
                
                # Wait for response to be fully loaded
                await response.finished()
                
                pdf_bytes = await response.body()
                
                # Verify we have actual PDF content
                if not pdf_bytes or len(pdf_bytes) < 100:
                    raise Exception("Received empty or invalid PDF content")
                
                # Check PDF header
                if not pdf_bytes[:5] == b'%PDF-':
                    raise Exception("Content is not a valid PDF file")
                
                # Try to extract text using available PDF libraries
                extracted_text = ""
                
                # Try pdfminer.six first (better for complex PDFs)
                try:
                    from pdfminer.high_level import extract_text_to_fp
                    from pdfminer.layout import LAParams
                    
                    output_string = io.StringIO()
                    extract_text_to_fp(
                        io.BytesIO(pdf_bytes),
                        output_string,
                        laparams=LAParams(),
                        output_type='text',
                        codec='utf-8'
                    )
                    extracted_text = output_string.getvalue()
                    
                except ImportError:
                    # Fallback to PyPDF2 if pdfminer.six is not available
                    try:
                        from PyPDF2 import PdfReader
                        
                        pdf_file = io.BytesIO(pdf_bytes)
                        reader = PdfReader(pdf_file)
                        for page in reader.pages:
                            extracted_text += page.extract_text() + "\n"
                            
                    except ImportError:
                        raise Exception("No PDF parsing library available. Please install pdfminer.six or PyPDF2.")
                
                # Truncate if needed
                if max_text_length and len(extracted_text) > max_text_length:
                    extracted_text = extracted_text[:max_text_length] + "..."
                
                # Format output
                if output_format == "markdown":
                    title = url.split('/')[-1].replace('.pdf', '')
                    content = f"# PDF Document: {title}\n\n**Source:** {url}\n\n---\n\n{extracted_text}"
                else:  # json
                    content = {
                        "url": url,
                        "text": extracted_text,
                        "format": "pdf",
                        "length": len(extracted_text)
                    }
                
                result = {
                    "success": True,
                    "content": content,
                    "url": url,
                    "format": output_format,
                    "type": "pdf"
                }
                
                self.history.append(
                    {
                        "action": "extract_pdf_content_from_url",
                        "reasoning": r,
                        "url": url,
                        "content_length": len(extracted_text),
                        "success": True
                    }
                )
                
                return result
                
            except Exception as e2:
                error_msg = f"Primary method: {str(e)}, Fallback method: {str(e2)}"
                
                self.history.append(
                    {
                        "action": "extract_pdf_content_from_url",
                        "reasoning": r,
                        "url": url,
                        "success": False,
                        "error": error_msg
                    }
                )
                
                return {
                    "success": False,
                    "error": error_msg,
                    "url": url,
                    "message": f"Failed to extract PDF content"
                }

    async def press_key(self, key: str, reasoning: Optional[str] = None, delay: Optional[int] = None) -> None:
        """
        Press a key.

        Parameters:
            key: Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')
            reasoning (Optional[str]): The reasoning chain for this action.
            delay (Optional[int]): Delay between key presses in milliseconds.
        """
        r = reasoning or ""
        await self.page.keyboard.press(key, delay=delay)
        self.history.append(
            {
                "action": "press_key",
                "reasoning": r,
                "key": key,
            }
        )

    async def wait_for_navigation(self, timeout: Optional[int] = None, reasoning: Optional[str] = None) -> None:
        """
        Wait for navigation to complete.

        Parameters:
            timeout (Optional[int]): Optional timeout in milliseconds.
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.wait_for_load_state("networkidle", timeout=timeout)
        self.history.append(
            {
                "action": "wait_for_navigation",
                "reasoning": r,
            }
        )

    async def evaluate_javascript(self, script: str, reasoning: Optional[str] = None) -> Any:
        """
        Execute JavaScript in the page context.

        Parameters:
            script: JavaScript code to execute
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Result of the JavaScript execution
        """
        r = reasoning or ""
        result = await self.page.evaluate(script)
        self.history.append(
            {
                "action": "evaluate_javascript",
                "reasoning": r,
                "script": script[:100] + "..." if len(script) > 100 else script,
                "output": str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
            }
        )
        return result

    async def get_cookies(self, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all cookies for the current page.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            List of cookie dictionaries
        """
        r = reasoning or ""
        cookies = await self.context.cookies()
        self.history.append(
            {
                "action": "get_cookies",
                "reasoning": r,
                "output": f"Found {len(cookies)} cookies",
            }
        )
        return cookies

    async def set_cookie(self, cookie: Dict[str, Any], reasoning: Optional[str] = None) -> None:
        """
        Set a cookie.

        Parameters:
            cookie: Cookie dictionary with name, value, domain, etc.
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.context.add_cookies([cookie])
        self.history.append(
            {
                "action": "set_cookie",
                "reasoning": r,
                "cookie": cookie,
            }
        )

    async def delete_cookies(self, reasoning: Optional[str] = None) -> None:
        """
        Delete all cookies.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.context.clear_cookies()
        self.history.append(
            {
                "action": "delete_cookies",
                "reasoning": r,
            }
        )

    async def get_local_storage(self, reasoning: Optional[str] = None) -> Dict[str, str]:
        """
        Get all local storage items.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.

        Returns:
            Dictionary of local storage key-value pairs
        """
        r = reasoning or ""
        storage = await self.page.evaluate("""
            () => {
                const storage = {};
                for (let i = 0; i < localStorage.length; i++) {
                    const key = localStorage.key(i);
                    storage[key] = localStorage.getItem(key);
                }
                return storage;
            }
        """)
        self.history.append(
            {
                "action": "get_local_storage",
                "reasoning": r,
                "output": f"Found {len(storage)} items",
            }
        )
        return storage

    async def set_local_storage(self, key: str, value: str, reasoning: Optional[str] = None) -> None:
        """
        Set a local storage item.

        Parameters:
            key: Storage key
            value: Storage value
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.evaluate(f"localStorage.setItem('{key}', '{value}')")
        self.history.append(
            {
                "action": "set_local_storage",
                "reasoning": r,
                "key": key,
                "value": value,
            }
        )

    async def clear_local_storage(self, reasoning: Optional[str] = None) -> None:
        """
        Clear all local storage.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action.
        """
        r = reasoning or ""
        await self.page.evaluate("localStorage.clear()")
        self.history.append(
            {
                "action": "clear_local_storage",
                "reasoning": r,
            }
        )

    async def close(self) -> None:
        """
        Close the browser and stop the Playwright instance.
        """
        await self.browser.close()
        await self.playwright.stop()
        self.history.append(
            {
                "action": "close",
                "reasoning": "Browser closed.",
            }
        )

    def reset_history(self) -> None:
        """
        Reset the internal history to an empty list.
        """
        self.history = []

    def truncate_history(self, n: int) -> None:
        """
        Truncate the history keeping only the last 'n' items.

        Parameters:
            n (int): The number of most recent history items to keep.
        """
        self.history = self.history[-n:]
        self.history = self.history[-n:]

    # Add missing methods to BrowserTool class

    async def type_text(self, selector: str, text: str, delay: int = 0, reasoning: Optional[str] = None, timeout: Optional[int] = None) -> str:
        """
        Type text into an element specified by selector.
        
        Args:
            selector: CSS selector for the target element
            text: Text to type
            delay: Delay between keystrokes in milliseconds
            reasoning: Optional reasoning for this action
            timeout: Optional timeout in milliseconds
            
        Returns:
            Success message
        """
        if reasoning:
            logging.info(f"BrowserTool Type Text: {reasoning}")
        
        await self.page.type(selector, text, delay=delay, timeout=timeout)
        return f"Successfully typed text into element: {selector}"

    async def detect_interactive_elements_rule_based(self, visible_only: bool = True, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Detect interactive elements on the page using rule-based approach.
        
        This method finds clickable elements like buttons, links, inputs, etc. using
        CSS selectors and DOM analysis.
        
        Args:
            visible_only: If True, only return visible elements
            reasoning: Optional reasoning for this detection
            
        Returns:
            List of interactive elements with their properties and positions
        """
        if reasoning:
            logging.info(f"BrowserTool Detect Interactive Elements: {reasoning}")
        
        # Define selectors for interactive elements
        interactive_selectors = [
            'button',
            'input[type="button"]',
            'input[type="submit"]', 
            'input[type="reset"]',
            'input[type="image"]',
            'input[type="checkbox"]',
            'input[type="radio"]',
            'input[type="text"]',
            'input[type="password"]',
            'input[type="email"]',
            'input[type="number"]',
            'input[type="tel"]',
            'input[type="url"]',
            'input[type="search"]',
            'input[type="date"]',
            'input[type="time"]',
            'input[type="datetime-local"]',
            'input[type="file"]',
            'textarea',
            'select',
            'a[href]',
            '[onclick]',
            '[role="button"]',
            '[role="link"]',
            '[tabindex]',
            '.btn',
            '.button',
            '.link'
        ]
        
        elements = []
        seen_bboxes = {}  # Track elements by bounding box to prevent duplicates
        
        for selector in interactive_selectors:
            try:
                # Get all elements matching this selector
                element_handles = await self.page.query_selector_all(selector)
                
                for element in element_handles:
                    try:
                        # Get bounding box first (needed for both visibility and positioning)
                        bbox = await element.bounding_box()
                        if not bbox:
                            continue
                        
                        # Calculate center point (needed for both visibility checks and element info)
                        center_x = int(bbox['x'] + bbox['width'] / 2)
                        center_y = int(bbox['y'] + bbox['height'] / 2)
                        
                        # Check if element is visible if required
                        if visible_only:
                            # First check basic DOM visibility
                            is_visible = await element.is_visible()
                            if not is_visible:
                                continue
                        
                            # Then check if element is actually on top (not obscured by modals/overlays)
                            # Use document.elementFromPoint to check if this element is on top
                            is_on_top = await self.page.evaluate(
                                """(args) => {
                                    const { element, centerX, centerY } = args;
                                    const topElement = document.elementFromPoint(centerX, centerY);
                                    return topElement === element || element.contains(topElement);
                                }""",
                                {"element": element, "centerX": center_x, "centerY": center_y}
                            )
                            if not is_on_top:
                                continue
                        
                        # Create a unique key for the bounding box (rounded to avoid floating point issues)
                        bbox_key = (
                            round(bbox['x'], 2), 
                            round(bbox['y'], 2), 
                            round(bbox['x'] + bbox['width'], 2), 
                            round(bbox['y'] + bbox['height'], 2)
                        )
                        
                        # Get element attributes
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        element_type = await element.evaluate('el => el.type || ""')
                        text_content = await element.evaluate('el => el.textContent || el.value || el.alt || el.title || ""')
                        href = await element.evaluate('el => el.href || ""')
                        
                        # Create label from tag name and text content
                        text_content_trimmed = text_content.strip()[:100]  # Limit text length
                        if text_content_trimmed:
                            label = f"{element_type}: {text_content_trimmed}"
                        else:
                            label = element_type
                        
                        element_info = {
                            'label': label,
                            'href': href,
                            'bbox': [int(bbox['x']), int(bbox['y']), int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height'])],
                            'center': [center_x, center_y],
                            'selector': selector,
                            'source': 'rule_based'  # Mark as rule-based
                        }
                        
                        # Check if we've already seen this bounding box
                        if bbox_key in seen_bboxes:
                            # Element already exists - decide which one to keep based on selector priority
                            existing_element = seen_bboxes[bbox_key]
                            
                            # Define selector priority (more specific selectors are preferred)
                            selector_priority = {
                                'button': 10,
                                'input[type="button"]': 9,
                                'input[type="submit"]': 9,
                                'a[href]': 8,
                                '[role="button"]': 7,
                                '[role="link"]': 6,
                                '[onclick]': 5,
                                'input[type="text"]': 5,
                                'input[type="password"]': 5,
                                'select': 5,
                                'textarea': 5,
                                '[tabindex]': 3,
                                '.btn': 2,
                                '.button': 2,
                                '.link': 1
                            }
                            
                            current_priority = selector_priority.get(selector, 0)
                            existing_priority = selector_priority.get(existing_element['selector'], 0)
                            
                            # Keep the element with higher priority selector
                            if current_priority > existing_priority:
                                seen_bboxes[bbox_key] = element_info
                            # If same priority, prefer the one with more descriptive label
                            elif current_priority == existing_priority:
                                if len(element_info['label']) > len(existing_element['label']):
                                    seen_bboxes[bbox_key] = element_info
                        else:
                            # New element - add it
                            seen_bboxes[bbox_key] = element_info
                        
                    except Exception as e:
                        logging.debug(f"Error processing element: {e}")
                        continue
                        
            except Exception as e:
                logging.debug(f"Error with selector {selector}: {e}")
                continue
        
        # Convert the deduplicated elements back to a list
        elements = list(seen_bboxes.values())
        
        return elements

    async def predict_interactive_elements(self, reasoning: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Predict interactive elements using AI vision analysis.
        
        This method takes a screenshot and uses AI to identify interactive elements
        that might not be easily detected through DOM analysis.
        
        Args:
            reasoning: Optional reasoning for this prediction
            
        Returns:
            List of predicted interactive elements with confidence scores
        """
        if reasoning:
            logging.info(f"BrowserTool Predict Interactive Elements: {reasoning}")
        
        # Take screenshot for analysis
        screenshot_path = await self.screenshot(filename="interactive_prediction", reasoning="For AI vision analysis")
        
        # This is a placeholder implementation
        # In a real implementation, this would use a vision model to analyze the screenshot
        # For now, return empty list as this requires integration with vision models
        logging.warning("predict_interactive_elements is a placeholder - requires vision model integration")
        
        return []



    async def highlight_bbox(
        self, 
        elements: List[Dict[str, Any]], 
        reasoning: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Highlight bounding boxes on the current page with numbered labels using advanced masking algorithm.
        
        This method takes a list of elements with bbox coordinates and renders
        rectangles with numbered labels (1, 2, 3...) using a sophisticated 
        masking-based approach to prevent label overlaps and collisions.
        
        Args:
            elements: List of element dictionaries, each containing:
                - 'bbox': [x1, y1, x2, y2] coordinates
                - 'label': Element description (for metadata, not shown in image)
                - 'selector': CSS selector (optional)
                - 'href': URL for links (optional)
                - 'confidence': Prediction confidence (optional)
                - 'source': Detection source (optional)
            reasoning: Optional reasoning for taking the screenshot
            filename: Optional custom filename (without extension). If None, uses timestamp-based naming.
            
        Returns:
            Path to the screenshot with highlighted bounding boxes and numbers
            
        Raises:
            Exception: If browser is not initialized or screenshot fails
        """
        if not self.page:
            raise Exception("Browser page is not initialized.")
        
        if reasoning:
            logging.info(f"BrowserTool Highlight Bbox: {reasoning}")
        
        # Take a fresh screenshot for annotation
        screenshot_bytes = await self.page.screenshot()
        image = PILImage.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        
        # Use the same font as highlight_interactive_elements
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
        
        # Create global occupancy mask (1=available, 0=occupied)
        import numpy as np
        global_mask = np.ones((image_height, image_width), dtype=np.uint8)
        
        # Mark all bounding box areas as occupied (unavailable for label placement)
        for element in elements:
            bbox = element.get('bbox', [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Ensure coordinates are within image bounds
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(x1, min(x2, image_width))
            y2 = max(y1, min(y2, image_height))
            
            # Mark bbox area as occupied in the global mask
            if x2 > x1 and y2 > y1:
                global_mask[y1:y2, x1:x2] = 0
        
        # Draw bounding boxes and labels using masking algorithm
        for i, element in enumerate(elements):
            bbox = element.get('bbox', [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are valid
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Convert to integers for drawing
            x, y, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image_width))
            y = max(0, min(y, image_height))
            x2 = max(x, min(x2, image_width))
            y2 = max(y, min(y2, image_height))
            
            # Skip if bbox is invalid after bounds checking
            if x2 <= x or y2 <= y:
                continue
            
            # Draw the bounding box in Imperial Red (#ED2939) - same as highlight_interactive_elements
            draw.rectangle([(x, y), (x2, y2)], outline="#ED2939", width=2)
            
            # Use only the element number as label (simple and clean)
            label = f"{i+1}"
            
            # Calculate text dimensions
            bbox_text = font.getbbox(label)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Find optimal label position using masking algorithm
            try:
                label_x, label_y, global_mask = find_label_position_with_masking(
                    bbox=(x, y, x2, y2),
                    text_width=text_width,
                    text_height=text_height,
                    image_width=image_width,
                    image_height=image_height,
                    global_mask=global_mask,
                    padding=10,
                    label_margin=5
                )
                
                # Draw white background with alpha 200 for the label
                bg_padding = 2
                draw.rectangle([
                    label_x - bg_padding,
                    label_y - bg_padding,
                    label_x + text_width + bg_padding,
                    label_y + text_height + bg_padding
                ], fill=(255, 255, 255, 200))
                
                # Draw the text in Imperial Red (#ED2939) - same color as the bounding box borders
                draw.text((label_x, label_y), label, fill="#ED2939", font=font)
                
            except Exception as e:
                logging.warning(f"Error positioning label for element {i+1}: {e}")
                # Fallback to simple positioning with proper spacing
                fallback_x = max(0, min(x, image_width - text_width))
                margin = 5  # Consistent margin for fallback
                fallback_y = max(0, y - text_height - margin) if y - text_height - margin >= 0 else min(y2 + margin, image_height - text_height)
            
                # Draw white background with alpha 200 for the fallback label
                bg_padding = 2
                draw.rectangle([
                    fallback_x - bg_padding,
                    fallback_y - bg_padding,
                    fallback_x + text_width + bg_padding,
                    fallback_y + text_height + bg_padding
                ], fill=(255, 255, 255, 200))
                
                draw.text((fallback_x, fallback_y), label, fill="#ED2939", font=font)
        
        # Save the annotated screenshot with custom or timestamp-based filename
        timestamp = int(time.time() * 1000)
        if filename:
            image_filename = f"{filename}.png"
        else:
            image_filename = f"highlighted_elements_{timestamp}.png"
        filepath = os.path.join(self.screenshot_path, image_filename)
        
        image.save(filepath)
        
        # Add to history
        self.history.append(
            {
                "action": "highlight_bbox",
                "timestamp": timestamp,
                "reasoning": reasoning or "Highlight bounding boxes on page with masking-based positioning",
                "elements_count": len(elements),
                "screenshot_path": filepath,
            }
        )
        
        return filepath

    async def highlight_pixel_grid(
        self, 
        pixel_spacing: int = 100,
        reasoning: Optional[str] = None
    ) -> str:
        """
        Creates a grid overlay on the current page with specified pixel spacing between grid lines.
        
        Vertical lines are red with x-coordinate labels written vertically.
        Horizontal lines are yellow-orange with y-coordinate labels written horizontally.
        Coordinate labels show the actual pixel position from origin (0,0) at top-left.
        
        Args:
            pixel_spacing: Space between consecutive grid lines in pixels (default: 100)
            reasoning: Optional reasoning for this action
            
        Returns:
            Path to the saved screenshot with grid overlay
        """
        if reasoning:
            logger.info(f"Browser Tool - Highlighting pixel grid: {reasoning}")
        
        # Take a screenshot to get current page dimensions
        screenshot_bytes = await self.page.screenshot()
        image = PILImage.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
        width, height = image.size
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Define colors
        vertical_color = "#FF0000"  # Red for vertical lines and x-values
        horizontal_color = "#FFA500"  # Orange for horizontal lines and y-values
        
        # Load font for coordinate labels
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=12)
        except Exception:
            font = ImageFont.load_default(size=12)
        
        # Draw vertical lines (red) with x-coordinate labels
        x = 0
        while x <= width:
            # Draw vertical line
            draw.line([(x, 0), (x, height)], fill=vertical_color, width=1)
            
            # Draw x-coordinate label vertically
            if x > 0:  # Skip x=0 to avoid overlap with y-labels
                label = f"x= {x}"
                # For vertical text, we need to create a temporary image and rotate it
                temp_img = PILImage.new('RGBA', (150, 30), (255, 255, 255, 0))
                temp_draw = ImageDraw.Draw(temp_img)
                
                # Get text dimensions for background
                bbox = font.getbbox(label)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw semi-transparent white background (200 alpha as requested)
                bg_padding = 2
                temp_draw.rectangle([
                    2 - bg_padding,
                    2 - bg_padding,
                    2 + text_width + bg_padding,
                    2 + text_height + bg_padding
                ], fill=(255, 255, 255, 200))
                
                # Draw the text
                temp_draw.text((2, 2), label, font=font, fill=vertical_color)
                
                # Rotate the text 90 degrees counterclockwise for vertical orientation
                rotated_text = temp_img.rotate(90, expand=True)
                
                # Calculate position to place the rotated text
                text_width, text_height = rotated_text.size
                paste_x = max(0, min(x - text_width // 2, width - text_width))
                paste_y = 5  # Small offset from top
                
                # Paste the rotated text onto the main image
                image.paste(rotated_text, (paste_x, paste_y), rotated_text)
            
            x += pixel_spacing
        
        # Draw horizontal lines (yellow-orange) with y-coordinate labels  
        y = 0
        while y <= height:
            # Draw horizontal line
            draw.line([(0, y), (width, y)], fill=horizontal_color, width=1)
            
            # Draw y-coordinate label horizontally
            if y > 0:  # Skip y=0 to avoid overlap
                label = f"y={y}"
                # Get text dimensions
                bbox = font.getbbox(label)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position label with small offset from left edge
                label_x = 5
                label_y = max(0, min(y - text_height // 2, height - text_height))
                
                # Draw background rectangle with alpha 200 as requested
                bg_padding = 2
                draw.rectangle([
                    label_x - bg_padding,
                    label_y - bg_padding,
                    label_x + text_width + bg_padding,
                    label_y + text_height + bg_padding
                ], fill=(255, 255, 255, 200))
                
                # Draw the y-coordinate label
                draw.text((label_x, label_y), label, font=font, fill=horizontal_color)
            
            y += pixel_spacing
        
        # Save the image with grid overlay
        timestamp = int(time.time() * 1000)
        filename = f"pixel_grid_{timestamp}.png"
        save_path = os.path.join(self.screenshot_path, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert back to RGB for saving (remove alpha channel)
        rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
        rgb_image.save(save_path)
        
        logger.info(f"Pixel grid screenshot saved to: {save_path}")
        return save_path

    async def type_text(
        self,
        text: str,
        reasoning: Optional[str] = None,
        min_delay: int = 50,
        max_delay: int = 150,
        variation_factor: float = 0.3,
    ) -> None:
        """
        Type text with realistic human-like behavior including random delays and typing variations.

        This method simulates natural human typing patterns by introducing random delays between
        keystrokes and slight variations in typing speed to make the interaction appear more
        human-like and less detectable as automated input.

        Parameters:
            text (str): The text to type. Each character will be typed individually with realistic delays.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            min_delay (int): Minimum delay between keystrokes in milliseconds. Defaults to 50ms.
            max_delay (int): Maximum delay between keystrokes in milliseconds. Defaults to 150ms.
            variation_factor (float): Factor to add randomness to delays (0.0 = no variation, 1.0 = high variation). 
                                    Defaults to 0.3 for natural variation.

        Returns:
            None

        Raises:
            ValueError: If min_delay >= max_delay or variation_factor is negative.
        """
        import random
        import asyncio

        if min_delay >= max_delay:
            raise ValueError("min_delay must be less than max_delay")
        if variation_factor < 0:
            raise ValueError("variation_factor must be non-negative")

        r = reasoning or ""
        
        # Track the typing for history
        start_time = asyncio.get_event_loop().time()
        
        for i, char in enumerate(text):
            # Type the character
            await self.page.keyboard.type(char)
            
            # Don't add delay after the last character
            if i < len(text) - 1:
                # Calculate base delay
                base_delay = random.randint(min_delay, max_delay)
                
                # Add variation
                variation = random.uniform(-variation_factor, variation_factor)
                actual_delay = max(10, int(base_delay * (1 + variation)))  # Minimum 10ms delay
                
                # Convert to seconds for asyncio.sleep
                await asyncio.sleep(actual_delay / 1000.0)
        
        end_time = asyncio.get_event_loop().time()
        total_time = int((end_time - start_time) * 1000)  # Convert to milliseconds
        
        self.history.append(
            {
                "action": "type_text",
                "reasoning": r,
                "text": text,
                "character_count": len(text),
                "min_delay": min_delay,
                "max_delay": max_delay,
                "variation_factor": variation_factor,
                "total_time_ms": total_time,
            }
        )

    async def get_accessibility_tree(
        self,
        reasoning: Optional[str] = None,
        include_hidden: bool = False,
        max_depth: Optional[int] = None,
        simplify: bool = True,
    ) -> Dict[str, Any]:
        """
        Get and return a clean accessibility tree of the current page for AI agent consumption.

        This method retrieves the accessibility tree that assistive technologies use to understand
        the page structure, then formats it in a clean, simplified format that's optimized for
        AI agents to understand page content and structure without complex HTML parsing.

        Parameters:
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            include_hidden (bool): Whether to include hidden/invisible elements. Defaults to False.
            max_depth (Optional[int]): Maximum depth to traverse in the tree. None for no limit.
            simplify (bool): Whether to simplify the tree by removing redundant information. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the cleaned accessibility tree with metadata.
                Structure:
                {
                    "url": "current_page_url",
                    "title": "page_title", 
                    "tree": {...}, # The accessibility tree structure with href information
                    "summary": {
                        "total_nodes": int,
                        "interactive_elements": int,
                        "text_nodes": int,
                        "links_with_href": int,
                        "depth": int
                    }
                }
                
                Each node in the tree may include:
                - role: Element's ARIA role
                - name: Element's accessible name/text
                - value: Element's value (for inputs)
                - href: URL destination for links and buttons (newly added)
                - position: Bounding box coordinates for clicking
                - disabled/checked/selected/expanded: Interactive states
        """
        r = reasoning or ""
        
        try:
            # Get the accessibility tree snapshot
            snapshot = await self.page.accessibility.snapshot()
            
            if not snapshot:
                return {
                    "url": self.page.url,
                    "title": await self.page.title(),
                    "tree": None,
                    "summary": {
                        "total_nodes": 0,
                        "interactive_elements": 0,
                        "text_nodes": 0,
                        "links_with_href": 0,
                        "depth": 0
                    },
                    "error": "No accessibility tree available"
                }

            # Extract href information for links and buttons
            href_info = await self._extract_href_information()

            # Clean and process the tree
            cleaned_tree = self._clean_accessibility_node(
                snapshot, 
                include_hidden=include_hidden,
                current_depth=0,
                max_depth=max_depth,
                simplify=simplify,
                href_info=href_info
            )
            
            # Generate summary statistics
            summary = self._analyze_accessibility_tree(cleaned_tree)
            
            result = {
                "url": self.page.url,
                "title": await self.page.title(),
                "tree": cleaned_tree,
                "summary": summary
            }
            
            self.history.append(
                {
                    "action": "get_accessibility_tree",
                    "reasoning": r,
                    "include_hidden": include_hidden,
                    "max_depth": max_depth,
                    "simplify": simplify,
                    "summary": summary,
                }
            )
            
            return result
            
        except Exception as e:
            error_result = {
                "url": self.page.url,
                "title": await self.page.title(),
                "tree": None,
                "summary": {
                    "total_nodes": 0,
                    "interactive_elements": 0,
                    "text_nodes": 0,
                    "links_with_href": 0,
                    "depth": 0
                },
                "error": f"Failed to get accessibility tree: {str(e)}"
            }
            
            self.history.append(
                {
                    "action": "get_accessibility_tree",
                    "reasoning": r,
                    "error": str(e),
                }
            )
            
            return error_result

    async def _extract_href_information(self) -> Dict[str, str]:
        """
        Extract href information for all links and buttons on the page.
        
        Returns:
            Dictionary mapping element text content to href values
        """
        try:
            # JavaScript to extract href information with element text content for matching
            href_script = """
            () => {
                const hrefMap = {};
                
                // Get all elements with href attributes (links, buttons with href, etc.)
                const elementsWithHref = document.querySelectorAll('a[href], button[href], area[href]');
                
                elementsWithHref.forEach(element => {
                    const href = element.getAttribute('href');
                    if (href && href.trim()) {
                        // Get text content for matching
                        let textContent = element.textContent || element.innerText || '';
                        textContent = textContent.trim().replace(/\\s+/g, ' ');
                        
                        // Also try aria-label, title, or alt attributes
                        if (!textContent) {
                            textContent = element.getAttribute('aria-label') || 
                                         element.getAttribute('title') || 
                                         element.getAttribute('alt') || '';
                            textContent = textContent.trim();
                        }
                        
                        if (textContent) {
                            // Use text content as key for matching
                            hrefMap[textContent] = href.trim();
                        }
                    }
                });
                
                // Also check for buttons with formaction or data-href
                const buttonsWithAction = document.querySelectorAll('button[formaction], button[data-href], input[formaction]');
                buttonsWithAction.forEach(element => {
                    const action = element.getAttribute('formaction') || element.getAttribute('data-href');
                    if (action && action.trim()) {
                        let textContent = element.textContent || element.innerText || '';
                        textContent = textContent.trim().replace(/\\s+/g, ' ');
                        
                        if (!textContent) {
                            textContent = element.getAttribute('aria-label') || 
                                         element.getAttribute('title') || 
                                         element.getAttribute('value') || '';
                            textContent = textContent.trim();
                        }
                        
                        if (textContent) {
                            hrefMap[textContent] = action.trim();
                        }
                    }
                });
                
                return hrefMap;
            }
            """
            
            return await self.page.evaluate(href_script)
            
        except Exception as e:
            # If href extraction fails, return empty dict and continue with tree building
            return {}

    def _clean_accessibility_node(
        self, 
        node: Dict[str, Any], 
        include_hidden: bool = False,
        current_depth: int = 0,
        max_depth: Optional[int] = None,
        simplify: bool = True,
        href_info: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Clean and simplify an accessibility tree node for AI consumption.
        
        Args:
            node: The accessibility tree node to clean
            include_hidden: Whether to include hidden elements
            current_depth: Current depth in the tree
            max_depth: Maximum depth to process
            simplify: Whether to simplify the output
            href_info: Dictionary mapping element text to href values for link matching
            
        Returns:
            Cleaned node dictionary or None if node should be filtered out
        """
        if not node:
            return None
            
        # Check depth limit
        if max_depth is not None and current_depth > max_depth:
            return None
        
        # Filter out hidden elements if requested
        if not include_hidden and node.get('hidden', False):
            return None
            
        # Start with a clean node
        cleaned = {}
        
        # Essential properties
        if 'role' in node:
            cleaned['role'] = node['role']
        if 'name' in node and node['name']:
            cleaned['name'] = node['name'].strip()
        if 'value' in node and node['value']:
            cleaned['value'] = str(node['value']).strip()
        
        # Interactive properties
        if node.get('disabled'):
            cleaned['disabled'] = True
        if node.get('checked') is not None:
            cleaned['checked'] = node['checked']
        if node.get('selected') is not None:
            cleaned['selected'] = node['selected']
        if node.get('expanded') is not None:
            cleaned['expanded'] = node['expanded']
            
        # Only include level if it's meaningful
        if 'level' in node and node['level'] > 0:
            cleaned['level'] = node['level']
            
        # Include coordinates if available (useful for clicking)
        if 'boundingBox' in node:
            bbox = node['boundingBox']
            position = {
                'x': int(bbox.get('x', 0)),
                'y': int(bbox.get('y', 0)),
                'width': int(bbox.get('width', 0)),
                'height': int(bbox.get('height', 0))
            }
            cleaned['position'] = position
        
        # Try to find href information for this element using text matching
        if href_info and cleaned.get('role') in {'link', 'button'}:
            element_text = cleaned.get('name', '').strip()
            if element_text and element_text in href_info:
                cleaned['href'] = href_info[element_text]
        
        # Process children recursively
        if 'children' in node and node['children']:
            children = []
            for child in node['children']:
                cleaned_child = self._clean_accessibility_node(
                    child, 
                    include_hidden=include_hidden,
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                    simplify=simplify,
                    href_info=href_info
                )
                if cleaned_child:
                    children.append(cleaned_child)
            
            if children:
                cleaned['children'] = children
        
        # If simplifying, filter out nodes with no meaningful content
        if simplify:
            # Keep nodes that have:
            # - A role that indicates interactivity or content
            # - Text content (name or value)
            # - Children
            # - Position information
            # - Href information (links/buttons with destinations)
            interactive_roles = {
                'button', 'link', 'textbox', 'combobox', 'listbox', 'menuitem',
                'tab', 'tabpanel', 'slider', 'spinbutton', 'checkbox', 'radio'
            }
            content_roles = {
                'heading', 'text', 'paragraph', 'article', 'main', 'navigation',
                'banner', 'contentinfo', 'complementary', 'region'
            }
            
            has_role = cleaned.get('role') in (interactive_roles | content_roles)
            has_content = bool(cleaned.get('name') or cleaned.get('value'))
            has_children = bool(cleaned.get('children'))
            has_position = bool(cleaned.get('position'))
            has_href = bool(cleaned.get('href'))
            
            if not (has_role or has_content or has_children or has_position or has_href):
                return None
        
        return cleaned if cleaned else None

    def _analyze_accessibility_tree(self, tree: Optional[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze the accessibility tree and generate summary statistics.
        
        Args:
            tree: The cleaned accessibility tree
            
        Returns:
            Dictionary with summary statistics
        """
        if not tree:
            return {
                "total_nodes": 0,
                "interactive_elements": 0,
                "text_nodes": 0,
                "links_with_href": 0,
                "depth": 0
            }
        
        interactive_roles = {
            'button', 'link', 'textbox', 'combobox', 'listbox', 'menuitem',
            'tab', 'tabpanel', 'slider', 'spinbutton', 'checkbox', 'radio'
        }
        
        def count_nodes(node, current_depth=0):
            if not node:
                return {
                    "total_nodes": 0,
                    "interactive_elements": 0, 
                    "text_nodes": 0,
                    "links_with_href": 0,
                    "max_depth": current_depth
                }
            
            stats = {
                "total_nodes": 1,
                "interactive_elements": 0,
                "text_nodes": 0,
                "links_with_href": 0,
                "max_depth": current_depth
            }
            
            # Check if this node is interactive
            if node.get('role') in interactive_roles:
                stats["interactive_elements"] = 1
                
            # Check if this node has text content
            if node.get('name') or node.get('value'):
                stats["text_nodes"] = 1
            
            # Check if this node has href information
            if node.get('href'):
                stats["links_with_href"] = 1
            
            # Process children
            if 'children' in node:
                for child in node['children']:
                    child_stats = count_nodes(child, current_depth + 1)
                    stats["total_nodes"] += child_stats["total_nodes"]
                    stats["interactive_elements"] += child_stats["interactive_elements"]
                    stats["text_nodes"] += child_stats["text_nodes"]
                    stats["links_with_href"] += child_stats["links_with_href"]
                    stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
            
            return stats
        
        stats = count_nodes(tree)
        return {
            "total_nodes": stats["total_nodes"],
            "interactive_elements": stats["interactive_elements"],
            "text_nodes": stats["text_nodes"],
            "links_with_href": stats["links_with_href"],
            "depth": stats["max_depth"]
        }

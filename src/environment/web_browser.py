## Description of function calling definitions for each method in the class
import asyncio
import io
import math
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from playwright.async_api import BrowserContext, Page, async_playwright

# Build the absolute path to the assets directory (assuming the repo structure provided)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, "..", "assets")


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


# Helper function to find a valid label position.
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
    the target point is inside any interactive element’s bounding box. If it is, a hand pointer icon is used; otherwise,
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
            # Draw the background rectangle with white color and the computed alpha.
            overlay_draw.rectangle(bg_rect, fill=(255, 255, 255, bg_alpha))

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
        self.download_path = os.path.join(temp_dir, "downloads")

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
        for attempt in range(3):
            try:
                context = await asyncio.wait_for(
                    browser.new_context(**context_kwargs), timeout=1.5
                )
                page = await asyncio.wait_for(context.new_page(), timeout=0.5)
                break
            except asyncio.TimeoutError:
                if attempt == 2:
                    raise TimeoutError("BrowserTool creation timed out.")
        tool = cls(playwright, browser, context, page, temp_dir, screenshot_dir)
        title = await page.title()
        tool.history.append(
            {
                "action": "goto",
                "reasoning": "Initial navigation to default page.",
                "url": "https://google.com",
                "output": title,
            }
        )
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

    # async def scroll_up(self, distance: int, reasoning: Optional[str] = None) -> None:
    #     """
    #     Scroll up by a specified distance (in pixels) using the evaluate method.

    #     Parameters:
    #         distance (int): The number of pixels to scroll up.
    #         reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
    #     """
    #     r = reasoning or ""
    #     await self.page.evaluate(f"window.scrollBy(0, -{distance})")
    #     self.history.append(
    #         {
    #             "action": "scroll_up_eval",
    #             "reasoning": r,
    #             "distance": distance,
    #         }
    #     )

    # async def scroll_down(self, distance: int, reasoning: Optional[str] = None) -> None:
    #     """
    #     Scroll down by a specified distance (in pixels) using the evaluate method.

    #     Parameters:
    #         distance (int): The number of pixels to scroll down.
    #         reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
    #     """
    #     r = reasoning or ""
    #     await self.page.evaluate(f"window.scrollBy(0, {distance})")
    #     self.history.append(
    #         {
    #             "action": "scroll_down_eval",
    #             "reasoning": r,
    #             "distance": distance,
    #         }
    #     )

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
    ) -> None:
        """
        Click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse click.
            y (int): The y-coordinate for the mouse click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the click action.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.mouse.click(x, y)
        self.history.append(
            {
                "action": "mouse_click",
                "reasoning": r,
                "x": x,
                "y": y,
            }
        )

    async def mouse_dbclick(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Perform a double click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse double click.
            y (int): The y-coordinate for the mouse double click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the double click action.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.mouse.dblclick(x, y, timeout=timeout)
        self.history.append(
            {
                "action": "mouse_dbclick",
                "reasoning": r,
                "x": x,
                "y": y,
            }
        )

    async def mouse_right_click(
        self,
        x: int,
        y: int,
        reasoning: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Perform a right click at the specified (x, y) coordinate on the page using the mouse.

        Parameters:
            x (int): The x-coordinate for the mouse right click.
            y (int): The y-coordinate for the mouse right click.
            reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.
            timeout (Optional[int]): Optional timeout in milliseconds for the right click action.

        Returns:
            None
        """
        r = reasoning or ""
        await self.page.mouse.click(x, y, button="right", timeout=timeout)
        self.history.append(
            {
                "action": "mouse_right_click",
                "reasoning": r,
                "x": x,
                "y": y,
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

    # async def get_object_by_label(
    #     self, label: str, reasoning: Optional[str] = None
    # ) -> Optional[str]:
    #     """
    #     Retrieve the outer HTML for the element with the given label (using the aria-label attribute).

    #     Parameters:
    #         label (str): The label of the element to retrieve.
    #         reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

    #     Returns:
    #         Optional[str]: The outer HTML of the found element or None if no matching element exists.
    #     """
    #     r = reasoning or ""
    #     element = await self.page.query_selector(f'[aria-label="{label}"]')
    #     if element:
    #         outer_html = await self.page.evaluate(
    #             "(element) => element.outerHTML", element
    #         )
    #         self.history.append(
    #             {
    #                 "action": "get_object_by_label",
    #                 "reasoning": r,
    #                 "label": label,
    #                 "output": outer_html,
    #             }
    #         )
    #         return outer_html
    #     else:
    #         self.history.append(
    #             {
    #                 "action": "get_object_by_label",
    #                 "reasoning": r,
    #                 "label": label,
    #                 "output": None,
    #             }
    #         )
    #         return None

    # async def get_object_by_selector(
    #     self, selector: str, reasoning: Optional[str] = None
    # ) -> Optional[str]:
    #     """
    #     Retrieve the outer HTML for the element matching the given CSS selector.

    #     Parameters:
    #         selector (str): The CSS selector of the element to retrieve.
    #         reasoning (Optional[str]): The reasoning chain for this action. Defaults to empty string.

    #     Returns:
    #         Optional[str]: The outer HTML of the found element or None if no matching element exists.
    #     """
    #     r = reasoning or ""
    #     element = await self.page.query_selector(selector)
    #     if element:
    #         outer_html = await self.page.evaluate("(el) => el.outerHTML", element)
    #         self.history.append(
    #             {
    #                 "action": "get_object_by_selector",
    #                 "reasoning": r,
    #                 "selector": selector,
    #                 "output": outer_html,
    #             }
    #         )
    #         return outer_html
    #     else:
    #         self.history.append(
    #             {
    #                 "action": "get_object_by_selector",
    #                 "reasoning": r,
    #                 "selector": selector,
    #                 "output": None,
    #             }
    #         )
    #         return None

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

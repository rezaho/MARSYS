"""
Shared Element Detection Logic

This module provides unified element detection capabilities used by both
BrowserTool.detect_interactive_elements_rule_based() and BrowserTool.get_page_elements().

Features:
- Shadow DOM piercing (open shadow roots)
- Cross-origin iframe traversal (via Playwright frame API)
- Configurable selector sets
- Visibility and occlusion checking
- Coordinate transformation for iframes
- Deduplication strategies
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Frame, Page

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Comprehensive list of interactive element selectors
# Used by both bbox detection and element listing
INTERACTIVE_SELECTORS = [
    # Buttons
    'button',
    'input[type="button"]',
    'input[type="submit"]',
    'input[type="reset"]',
    'input[type="image"]',
    # Text inputs
    'input[type="text"]',
    'input[type="password"]',
    'input[type="email"]',
    'input[type="number"]',
    'input[type="tel"]',
    'input[type="url"]',
    'input[type="search"]',
    # Date/time inputs
    'input[type="date"]',
    'input[type="time"]',
    'input[type="datetime-local"]',
    # Other inputs
    'input[type="checkbox"]',
    'input[type="radio"]',
    'input[type="file"]',
    'input[type="range"]',
    'input[type="color"]',
    # Form elements
    'textarea',
    'select',
    # Links
    'a[href]',
    # ARIA roles
    '[role="button"]',
    '[role="link"]',
    '[role="menuitem"]',
    '[role="tab"]',
    '[role="checkbox"]',
    '[role="radio"]',
    '[role="switch"]',
    '[role="textbox"]',
    '[role="combobox"]',
    '[role="listbox"]',
    '[role="option"]',
    '[role="slider"]',
    # Event handlers
    '[onclick]',
    '[onkeydown]',
    '[onkeyup]',
    # Keyboard navigation
    '[tabindex]:not([tabindex="-1"])',
    # Common CSS classes
    '.btn',
    '.button',
    '.link',
    '.clickable',
]

# Selector priority for deduplication (higher = more specific/preferred)
SELECTOR_PRIORITY = {
    'button': 10,
    'input[type="button"]': 9,
    'input[type="submit"]': 9,
    'a[href]': 8,
    '[role="button"]': 7,
    '[role="link"]': 6,
    '[onclick]': 5,
    'input[type="text"]': 5,
    'input[type="password"]': 5,
    'input[type="email"]': 5,
    'select': 5,
    'textarea': 5,
    '[role="menuitem"]': 4,
    '[role="tab"]': 4,
    '[tabindex]:not([tabindex="-1"])': 3,
    '.btn': 2,
    '.button': 2,
    '.link': 1,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RawElement:
    """Raw element data extracted from the page."""
    # Core identification
    tag_name: str
    element_type: str  # button, link, input, text, etc.
    selector: str  # Unique CSS selector

    # Position (in main frame coordinates)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)

    # Content
    text: str = ""
    href: str = ""
    value: str = ""
    placeholder: str = ""
    aria_label: str = ""
    title: str = ""

    # State
    is_visible: bool = True
    is_enabled: bool = True
    is_checked: bool = False
    is_selected: bool = False
    is_required: bool = False
    is_readonly: bool = False

    # Source info
    frame_url: str = ""  # URL of frame containing element (empty = main frame)
    in_shadow_dom: bool = False
    matched_selector: str = ""  # Which selector matched this element

    # Raw attributes for extensions
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class DetectionConfig:
    """Configuration for element detection."""
    # What to detect
    selectors: List[str] = field(default_factory=lambda: INTERACTIVE_SELECTORS.copy())
    include_headings: bool = False

    # Visibility options
    visible_only: bool = True
    check_occlusion: bool = False  # Expensive: checks elementFromPoint

    # Shadow DOM and iframes
    pierce_shadow_dom: bool = True
    include_iframes: bool = True
    same_origin_iframes_only: bool = False  # If True, skip cross-origin iframes

    # Limits
    max_elements: int = 500
    max_text_length: int = 100

    # Deduplication
    dedupe_by_bbox: bool = True
    dedupe_tolerance: int = 5  # Pixel tolerance for bbox comparison


# =============================================================================
# JavaScript Code for In-Page Execution
# =============================================================================

# JavaScript that runs in page context to extract elements
# This handles Shadow DOM piercing within a single frame
ELEMENT_EXTRACTION_JS = """
(config) => {
    const selectors = config.selectors;
    const maxElements = config.maxElements;
    const maxTextLen = config.maxTextLength;
    const visibleOnly = config.visibleOnly;
    const checkOcclusion = config.checkOcclusion;
    const pierceShadowDom = config.pierceShadowDom;
    const includeHeadings = config.includeHeadings;

    const results = [];
    const headings = [];
    let elementCount = 0;

    // Generate unique CSS selector for an element
    function generateSelector(element, root = document) {
        if (!element || element === document.body || element === document.documentElement) {
            return element?.tagName?.toLowerCase() || '';
        }

        // Try ID first
        if (element.id && /^[a-zA-Z][a-zA-Z0-9_-]*$/.test(element.id)) {
            const idSelector = `#${element.id}`;
            try {
                if (root.querySelectorAll(idSelector).length === 1) {
                    return idSelector;
                }
            } catch (e) {}
        }

        // Try name attribute for form elements
        if (element.name && ['INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName)) {
            const nameSelector = `${element.tagName.toLowerCase()}[name="${element.name}"]`;
            try {
                if (root.querySelectorAll(nameSelector).length === 1) {
                    return nameSelector;
                }
            } catch (e) {}
        }

        // Try class combination
        if (element.className && typeof element.className === 'string') {
            const classes = element.className.trim().split(/\\s+/)
                .filter(c => c && !c.includes(':') && /^[a-zA-Z_-]/.test(c));
            if (classes.length > 0) {
                const classSelector = element.tagName.toLowerCase() + '.' + classes.slice(0, 2).join('.');
                try {
                    if (root.querySelectorAll(classSelector).length === 1) {
                        return classSelector;
                    }
                } catch (e) {}
            }
        }

        // Fall back to path-based selector
        const path = [];
        let current = element;
        while (current && current !== root && current !== document.body && path.length < 5) {
            let selector = current.tagName.toLowerCase();
            if (current.parentElement) {
                const siblings = Array.from(current.parentElement.children)
                    .filter(el => el.tagName === current.tagName);
                if (siblings.length > 1) {
                    const index = siblings.indexOf(current) + 1;
                    selector += `:nth-of-type(${index})`;
                }
            }
            path.unshift(selector);
            current = current.parentElement;
        }
        return path.join(' > ');
    }

    // Check visibility
    function isVisible(element) {
        if (!visibleOnly) return true;

        const style = window.getComputedStyle(element);
        if (style.display === 'none' ||
            style.visibility === 'hidden' ||
            style.opacity === '0') {
            return false;
        }

        const rect = element.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) {
            return false;
        }

        // Check if in viewport (with margin)
        const margin = 100;
        if (rect.bottom < -margin || rect.top > window.innerHeight + margin ||
            rect.right < -margin || rect.left > window.innerWidth + margin) {
            return false;
        }

        return true;
    }

    // Check occlusion using elementFromPoint
    function isOnTop(element, rect) {
        if (!checkOcclusion) return true;

        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        // Skip if center is outside viewport
        if (centerX < 0 || centerX > window.innerWidth ||
            centerY < 0 || centerY > window.innerHeight) {
            return true; // Can't check, assume visible
        }

        const topElement = document.elementFromPoint(centerX, centerY);
        return topElement === element || element.contains(topElement) ||
               (topElement && topElement.contains(element));
    }

    // Get element text content
    function getText(element) {
        let text = element.getAttribute('aria-label') ||
                   element.getAttribute('title') ||
                   element.getAttribute('alt') ||
                   element.getAttribute('placeholder') ||
                   '';

        if (!text) {
            // Get direct text, avoiding nested element text
            text = element.innerText || element.textContent || '';
        }

        text = text.trim().replace(/\\s+/g, ' ');
        if (text.length > maxTextLen) {
            text = text.substring(0, maxTextLen) + '...';
        }
        return text;
    }

    // Extract element data
    function extractElement(element, matchedSelector, inShadowDom = false, shadowRoot = null) {
        if (elementCount >= maxElements) return null;
        if (!isVisible(element)) return null;

        const rect = element.getBoundingClientRect();
        if (!isOnTop(element, rect)) return null;

        const tagName = element.tagName.toLowerCase();
        let elementType = tagName;

        // Determine element type
        if (tagName === 'input') {
            elementType = element.type || 'text';
        } else if (tagName === 'a') {
            elementType = 'link';
        } else if (element.getAttribute('role')) {
            elementType = element.getAttribute('role');
        }

        // Generate selector (within shadow root if applicable)
        const selector = generateSelector(element, shadowRoot || document);

        const data = {
            tagName: tagName,
            elementType: elementType,
            selector: selector,
            // Use viewport-relative coordinates (getBoundingClientRect)
            // NOT document-relative (with scroll offsets)
            // This is correct for: screenshot annotation, mouse_click(x,y)
            bbox: [
                Math.round(rect.left),
                Math.round(rect.top),
                Math.round(rect.right),
                Math.round(rect.bottom)
            ],
            center: [
                Math.round(rect.left + rect.width / 2),
                Math.round(rect.top + rect.height / 2)
            ],
            text: getText(element),
            href: element.href || element.getAttribute('href') || '',
            value: element.value || '',
            placeholder: element.getAttribute('placeholder') || '',
            ariaLabel: element.getAttribute('aria-label') || '',
            title: element.getAttribute('title') || '',
            isVisible: true,
            isEnabled: !element.disabled,
            isChecked: element.checked || false,
            isSelected: element.selected || false,
            isRequired: element.required || false,
            isReadonly: element.readOnly || false,
            inShadowDom: inShadowDom,
            matchedSelector: matchedSelector
        };

        elementCount++;
        return data;
    }

    // Recursively traverse shadow DOM
    function traverseNode(node, matchedSelector, inShadowDom = false) {
        if (elementCount >= maxElements) return;

        // Check if this element matches
        if (node.nodeType === Node.ELEMENT_NODE) {
            const element = node;

            // Check if element matches any selector
            for (const selector of selectors) {
                try {
                    if (element.matches(selector)) {
                        const data = extractElement(element, selector, inShadowDom,
                            inShadowDom ? element.getRootNode() : null);
                        if (data) {
                            results.push(data);
                        }
                        break; // Only match once per element
                    }
                } catch (e) {
                    // Invalid selector, skip
                }
            }

            // Pierce shadow DOM if enabled
            if (pierceShadowDom && element.shadowRoot) {
                traverseNode(element.shadowRoot, matchedSelector, true);
            }
        }

        // Traverse children
        const children = node.childNodes || [];
        for (const child of children) {
            traverseNode(child, matchedSelector, inShadowDom);
        }
    }

    // Main extraction
    // First, use querySelectorAll for efficiency on main document
    const selectorString = selectors.join(', ');
    try {
        const elements = document.querySelectorAll(selectorString);
        for (const element of elements) {
            if (elementCount >= maxElements) break;

            // Find which selector matched
            let matchedSelector = '';
            for (const selector of selectors) {
                try {
                    if (element.matches(selector)) {
                        matchedSelector = selector;
                        break;
                    }
                } catch (e) {}
            }

            const data = extractElement(element, matchedSelector);
            if (data) {
                results.push(data);
            }
        }
    } catch (e) {
        console.warn('Element extraction error:', e);
    }

    // Then traverse shadow DOMs if enabled
    if (pierceShadowDom) {
        const allElements = document.querySelectorAll('*');
        for (const element of allElements) {
            if (elementCount >= maxElements) break;
            if (element.shadowRoot) {
                traverseNode(element.shadowRoot, '', true);
            }
        }
    }

    // Extract headings if requested
    if (includeHeadings) {
        const headingElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        for (const heading of headingElements) {
            if (!isVisible(heading)) continue;
            const rect = heading.getBoundingClientRect();
            let text = heading.textContent.trim().replace(/\\s+/g, ' ');
            if (text.length > maxTextLen) text = text.substring(0, maxTextLen) + '...';
            headings.push({
                level: parseInt(heading.tagName[1]),
                text: text,
                selector: generateSelector(heading),
                bbox: [
                    Math.round(rect.left),
                    Math.round(rect.top),
                    Math.round(rect.right),
                    Math.round(rect.bottom)
                ]
            });
        }
    }

    return { elements: results, headings: headings };
}
"""


# =============================================================================
# Element Detector Class
# =============================================================================

class ElementDetector:
    """
    Unified element detection with Shadow DOM and iframe support.

    This class provides the core detection logic used by both:
    - BrowserTool.detect_interactive_elements_rule_based() (for bbox/highlighting)
    - BrowserTool.get_page_elements() (for agent text tools)

    Usage:
        detector = ElementDetector(page)
        elements = await detector.detect(config)
    """

    def __init__(self, page: Page):
        """
        Initialize detector with a Playwright page.

        Args:
            page: Playwright Page object
        """
        self.page = page

    async def detect(self, config: Optional[DetectionConfig] = None) -> List[RawElement]:
        """
        Detect interactive elements on the page.

        Args:
            config: Detection configuration (uses defaults if None)

        Returns:
            List of RawElement objects with all detected elements
        """
        if config is None:
            config = DetectionConfig()

        all_elements: List[RawElement] = []

        # Detect in main frame
        main_elements = await self._detect_in_frame(self.page, config, frame_offset=(0, 0))
        all_elements.extend(main_elements)

        # Detect in iframes if enabled
        if config.include_iframes:
            iframe_elements = await self._detect_in_iframes(config)
            all_elements.extend(iframe_elements)

        # Deduplicate if enabled
        if config.dedupe_by_bbox:
            all_elements = self._deduplicate_by_bbox(all_elements, config.dedupe_tolerance)

        # Apply limit
        if len(all_elements) > config.max_elements:
            all_elements = all_elements[:config.max_elements]

        return all_elements

    async def _detect_in_frame(
        self,
        frame: Frame,
        config: DetectionConfig,
        frame_offset: Tuple[int, int] = (0, 0)
    ) -> List[RawElement]:
        """
        Detect elements within a single frame.

        Args:
            frame: Playwright Frame object
            config: Detection configuration
            frame_offset: (x, y) offset to add to element coordinates

        Returns:
            List of RawElement objects
        """
        try:
            # Prepare config for JavaScript
            js_config = {
                "selectors": config.selectors,
                "maxElements": config.max_elements,
                "maxTextLength": config.max_text_length,
                "visibleOnly": config.visible_only,
                "checkOcclusion": config.check_occlusion,
                "pierceShadowDom": config.pierce_shadow_dom,
                "includeHeadings": config.include_headings,
            }

            # Execute JavaScript to extract elements
            result = await frame.evaluate(ELEMENT_EXTRACTION_JS, js_config)

            # Convert to RawElement objects
            elements = []
            frame_url = frame.url if frame != self.page else ""
            offset_x, offset_y = frame_offset

            for data in result.get("elements", []):
                # Apply frame offset to coordinates
                bbox = data["bbox"]
                adjusted_bbox = (
                    bbox[0] + offset_x,
                    bbox[1] + offset_y,
                    bbox[2] + offset_x,
                    bbox[3] + offset_y,
                )

                center = data["center"]
                adjusted_center = (
                    center[0] + offset_x,
                    center[1] + offset_y,
                )

                element = RawElement(
                    tag_name=data["tagName"],
                    element_type=data["elementType"],
                    selector=data["selector"],
                    bbox=adjusted_bbox,
                    center=adjusted_center,
                    text=data.get("text", ""),
                    href=data.get("href", ""),
                    value=data.get("value", ""),
                    placeholder=data.get("placeholder", ""),
                    aria_label=data.get("ariaLabel", ""),
                    title=data.get("title", ""),
                    is_visible=data.get("isVisible", True),
                    is_enabled=data.get("isEnabled", True),
                    is_checked=data.get("isChecked", False),
                    is_selected=data.get("isSelected", False),
                    is_required=data.get("isRequired", False),
                    is_readonly=data.get("isReadonly", False),
                    frame_url=frame_url,
                    in_shadow_dom=data.get("inShadowDom", False),
                    matched_selector=data.get("matchedSelector", ""),
                )
                elements.append(element)

            return elements

        except Exception as e:
            logger.warning(f"Error detecting elements in frame {frame.url}: {e}")
            return []

    async def _detect_in_iframes(self, config: DetectionConfig) -> List[RawElement]:
        """
        Detect elements in all iframes using Playwright's privileged frame API.

        This can access cross-origin iframes that JavaScript cannot.

        Args:
            config: Detection configuration

        Returns:
            List of RawElement objects from all iframes
        """
        all_elements: List[RawElement] = []

        # Get all frames (Playwright's frame API has privileged access)
        frames = self.page.frames

        for frame in frames:
            # Skip main frame (already processed)
            if frame == self.page.main_frame:
                continue

            # Check same-origin restriction if enabled
            if config.same_origin_iframes_only:
                try:
                    main_origin = await self._get_origin(self.page.url)
                    frame_origin = await self._get_origin(frame.url)
                    if main_origin != frame_origin:
                        logger.debug(f"Skipping cross-origin iframe: {frame.url}")
                        continue
                except Exception:
                    continue

            # Get iframe element position in main frame for coordinate transformation
            try:
                frame_offset = await self._get_frame_offset(frame)
            except Exception as e:
                logger.debug(f"Could not get frame offset for {frame.url}: {e}")
                frame_offset = (0, 0)

            # Detect elements in this frame
            try:
                frame_elements = await self._detect_in_frame(frame, config, frame_offset)
                all_elements.extend(frame_elements)
            except Exception as e:
                logger.debug(f"Error detecting in iframe {frame.url}: {e}")

        return all_elements

    async def _get_frame_offset(self, frame: Frame) -> Tuple[int, int]:
        """
        Get the offset of an iframe relative to the main frame.

        Args:
            frame: Playwright Frame object

        Returns:
            (x_offset, y_offset) tuple
        """
        # Find the iframe element in the parent frame
        parent_frame = frame.parent_frame
        if not parent_frame:
            return (0, 0)

        # Get all iframe elements and find the one hosting this frame
        iframe_elements = await parent_frame.query_selector_all("iframe, frame")

        for iframe_el in iframe_elements:
            try:
                # Check if this iframe element hosts our frame
                content_frame = await iframe_el.content_frame()
                if content_frame == frame:
                    # Get bounding box
                    bbox = await iframe_el.bounding_box()
                    if bbox:
                        # Get parent offset recursively
                        parent_offset = await self._get_frame_offset(parent_frame)
                        return (
                            int(bbox["x"]) + parent_offset[0],
                            int(bbox["y"]) + parent_offset[1]
                        )
            except Exception:
                continue

        return (0, 0)

    def _get_origin(self, url: str) -> str:
        """Extract origin from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception:
            return url

    def _deduplicate_by_bbox(
        self,
        elements: List[RawElement],
        tolerance: int = 5
    ) -> List[RawElement]:
        """
        Remove duplicate elements based on bounding box overlap.

        Uses selector priority to decide which element to keep.

        Args:
            elements: List of elements to deduplicate
            tolerance: Pixel tolerance for bbox comparison

        Returns:
            Deduplicated list of elements
        """
        if not elements:
            return elements

        # Group elements by normalized bbox
        seen_bboxes: Dict[Tuple[int, int, int, int], RawElement] = {}

        for elem in elements:
            # Normalize bbox with tolerance
            bbox_key = (
                round(elem.bbox[0] / tolerance) * tolerance,
                round(elem.bbox[1] / tolerance) * tolerance,
                round(elem.bbox[2] / tolerance) * tolerance,
                round(elem.bbox[3] / tolerance) * tolerance,
            )

            if bbox_key in seen_bboxes:
                existing = seen_bboxes[bbox_key]

                # Get priorities
                current_priority = SELECTOR_PRIORITY.get(elem.matched_selector, 0)
                existing_priority = SELECTOR_PRIORITY.get(existing.matched_selector, 0)

                # Keep element with higher priority
                if current_priority > existing_priority:
                    seen_bboxes[bbox_key] = elem
                # If same priority, prefer one with more text
                elif current_priority == existing_priority and len(elem.text) > len(existing.text):
                    seen_bboxes[bbox_key] = elem
            else:
                seen_bboxes[bbox_key] = elem

        return list(seen_bboxes.values())

    def filter_containers(
        self,
        elements: List[RawElement],
        min_children: int = 2
    ) -> List[RawElement]:
        """
        Filter out container elements that contain multiple interactive children.

        This prevents large containers (like scrollable menus) from being detected
        when they contain more specific interactive elements.

        Args:
            elements: List of elements
            min_children: Minimum number of contained elements to consider as container

        Returns:
            Filtered list without containers
        """
        if not elements:
            return elements

        def bbox_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int], margin: int = 5) -> bool:
            """Check if outer bbox fully contains inner bbox."""
            return (
                outer[0] - margin <= inner[0] and
                outer[1] - margin <= inner[1] and
                outer[2] + margin >= inner[2] and
                outer[3] + margin >= inner[3]
            )

        def get_area(bbox: Tuple[int, int, int, int]) -> int:
            """Calculate bbox area."""
            return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Find container indices
        containers_to_remove = set()

        for i, elem in enumerate(elements):
            elem_area = get_area(elem.bbox)
            if elem_area == 0:
                continue

            # Count how many elements this one contains
            children_count = 0
            for j, other in enumerate(elements):
                if i == j:
                    continue
                other_area = get_area(other.bbox)

                # Check if elem contains other and other is significantly smaller
                if bbox_contains(elem.bbox, other.bbox) and other_area < elem_area * 0.8:
                    children_count += 1

            # If contains multiple children, mark as container
            if children_count >= min_children:
                containers_to_remove.add(i)
                logger.debug(f"Filtering container with {children_count} children: {elem.text[:50]}")

        # Filter out containers
        return [elem for i, elem in enumerate(elements) if i not in containers_to_remove]


# =============================================================================
# Format Converters
# =============================================================================

def to_bbox_format(elements: List[RawElement]) -> List[Dict[str, Any]]:
    """
    Convert RawElements to bbox format for screenshot highlighting.

    Output format matches detect_interactive_elements_rule_based():
    {
        'label': str,
        'href': str,
        'bbox': [x1, y1, x2, y2],
        'center': [cx, cy],
        'selector': str,
        'source': 'rule_based'
    }
    """
    result = []
    for elem in elements:
        # Create label
        text = elem.text.strip()[:100] if elem.text else ""
        label = f"{elem.element_type}: {text}" if text else elem.element_type

        result.append({
            'label': label,
            'href': elem.href,
            'bbox': list(elem.bbox),
            'center': list(elem.center),
            'selector': elem.selector,
            'source': 'rule_based'
        })

    return result


def to_compact_format(
    elements: List[RawElement],
    headings: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Convert RawElements to compact format for AI agents.

    Output format matches get_page_elements():
    {
        'elements': [
            {'i': 0, 't': 'button', 'tx': 'Submit', 's': '#btn', 'p': [x,y,w,h]},
            ...
        ],
        'headings': [...],
        'summary': {'total': int, 'buttons': int, ...}
    }
    """
    compact_elements = []

    for i, elem in enumerate(elements):
        item = {
            'i': i,
            't': elem.element_type,
            's': elem.selector,
            'p': [
                elem.bbox[0],
                elem.bbox[1],
                elem.bbox[2] - elem.bbox[0],  # width
                elem.bbox[3] - elem.bbox[1],  # height
            ]
        }

        if elem.text:
            item['tx'] = elem.text
        if elem.href:
            item['h'] = elem.href
        if elem.value:
            item['v'] = elem.value

        # State flags (only include non-default values)
        state = {}
        if not elem.is_enabled:
            state['disabled'] = True
        if elem.is_checked:
            state['checked'] = True
        if elem.is_selected:
            state['selected'] = True
        if elem.is_required:
            state['required'] = True
        if elem.is_readonly:
            state['readonly'] = True
        if state:
            item['st'] = state

        compact_elements.append(item)

    # Build summary
    summary = {
        'total': len(compact_elements),
        'buttons': sum(1 for e in compact_elements if e['t'] in ('button', 'submit')),
        'links': sum(1 for e in compact_elements if e['t'] == 'link'),
        'inputs': sum(1 for e in compact_elements
                     if e['t'] in ('text', 'email', 'password', 'search', 'tel', 'url', 'number')),
        'selects': sum(1 for e in compact_elements if e['t'] == 'select'),
    }

    result = {
        'elements': compact_elements,
        'summary': summary,
    }

    if headings:
        result['headings'] = [
            {'l': h['level'], 'tx': h['text'], 's': h['selector']}
            for h in headings
        ]

    return result

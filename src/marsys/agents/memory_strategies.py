"""
Memory management strategies for active context management.

This module defines the strategy interfaces and simple implementations for:
- Trigger strategies: Decide WHEN to engage ACM
- Process strategies: Define HOW to process messages (summarize, ACE, etc.)
- Retrieval strategies: Define HOW to retrieve curated context
- Compaction processors: Processor chain for compaction mode
"""

import copy
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from marsys.agents.memory import ManagedMemoryConfig, Message
    from marsys.utils.tokens import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class MemoryState:
    """
    Snapshot of memory state for strategy decision-making.

    This dataclass captures all relevant information about the current memory state
    to enable strategies to make informed decisions about triggering, processing,
    and retrieval.
    """

    raw_messages: List["Message"]
    estimated_tokens: int
    messages_since_last_retrieval: int
    tokens_since_last_retrieval: int
    last_retrieval_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# === Strategy Interfaces ===


class TriggerStrategy(ABC):
    """
    Abstract base class for trigger strategies.

    Trigger strategies decide WHEN to engage active context management.
    They are invoked at specific events (add, get_messages, etc.) and return
    a boolean indicating whether ACM should be activated.
    """

    @abstractmethod
    def should_trigger_on_add(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """
        Check if ACM should trigger after adding a message.

        Args:
            state: Current memory state
            config: Memory configuration

        Returns:
            True if ACM should engage, False otherwise
        """
        pass

    @abstractmethod
    def should_trigger_on_retrieve(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """
        Check if ACM should trigger when retrieving messages.

        Args:
            state: Current memory state
            config: Memory configuration

        Returns:
            True if ACM should engage, False otherwise
        """
        pass


class ProcessStrategy(ABC):
    """
    Abstract base class for processing strategies.

    Process strategies define HOW to process messages when ACM is triggered.
    Examples: summarization (monolithic rewrite), ACE (delta updates), no-op.
    """

    @abstractmethod
    async def process(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> Optional[Dict[str, Any]]:
        """
        Process messages (e.g., summarize, delta-edit).

        Args:
            state: Current memory state
            config: Memory configuration

        Returns:
            Optional metadata about processing (e.g., summary indices, edits applied)
        """
        pass


class RetrievalStrategy(ABC):
    """
    Abstract base class for retrieval strategies.

    Retrieval strategies define HOW to retrieve curated context for the LLM.
    They are responsible for selecting which messages to include while respecting
    token budgets and maintaining semantic coherence (e.g., tool-call bundling).
    """

    @abstractmethod
    def retrieve(
        self,
        state: MemoryState,
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
    ) -> List[Dict[str, Any]]:
        """
        Build curated message list for LLM.

        Args:
            state: Current memory state
            config: Memory configuration
            token_counter: Token counter instance

        Returns:
            List of message dicts ready for LLM (in chronological order)
        """
        pass


# === Shared Tool Bundle Helpers ===


def _include_tool_bundle_backward(
    start_idx: int,
    msg_dicts: List[Dict[str, Any]],
    target: int,
    running_total: int,
    token_counter: "TokenCounter",
) -> Optional[Tuple[List[Dict[str, Any]], int, int]]:
    """
    Scan backward from a tool message to include its complete tool bundle.

    Starting from start_idx, scans backward to find the assistant message
    with tool_calls that originated the tool responses. Collects all
    intervening messages (sibling tool responses, user messages) along the way.

    Args:
        start_idx: Index to start scanning backward from (typically the index
            just before the tool message that triggered this call).
        msg_dicts: All message dicts.
        target: Target token budget.
        running_total: Current running total of tokens already committed.
        token_counter: Token counter instance.

    Returns:
        Tuple of (bundle_messages, new_index, new_running_total) where
        bundle_messages are the messages to add (in reverse order, matching
        the backward packing direction), new_index is the next index to
        continue from, and new_running_total is the updated token count.
        Returns None if the bundle cannot fit within budget or is malformed.
    """
    j = start_idx
    tokens = running_total
    bundle: List[Dict[str, Any]] = []

    while j >= 0:
        msg = msg_dicts[j]
        msg_tokens = token_counter.count_message(msg)

        if tokens + msg_tokens > target:
            return None

        role = msg.get("role")

        if role == "tool":
            bundle.append(msg)
            tokens += msg_tokens
            j -= 1
            continue

        if role == "assistant":
            if msg.get("tool_calls"):
                bundle.append(msg)
                tokens += msg_tokens
                return bundle, j - 1, tokens
            else:
                return bundle, j, tokens

        if role == "user":
            bundle.append(msg)
            tokens += msg_tokens
            j -= 1
            continue

        return bundle, j, tokens

    return None


# === Simple Implementations (v1) ===


class SimpleThresholdTrigger(TriggerStrategy):
    """
    Trigger ACM when total tokens exceed a threshold.

    This is the simplest trigger strategy: engage ACM when estimated total tokens
    cross threshold_tokens.
    """

    def should_trigger_on_add(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """Trigger if total tokens exceed threshold."""
        return state.estimated_tokens > config.threshold_tokens

    def should_trigger_on_retrieve(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """Trigger if over threshold."""
        return state.estimated_tokens > config.threshold_tokens


class NoOpProcessStrategy(ProcessStrategy):
    """
    No-op processing strategy (v1 default).

    This strategy does not modify messages. All context management is handled
    by the retrieval strategy in a non-destructive manner, preserving the full
    lossless history in raw storage.
    """

    async def process(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> Optional[Dict[str, Any]]:
        """No processing - retrieval handles everything."""
        return None


class BackwardPackingRetrieval(RetrievalStrategy):
    """
    Backward-pack from newest messages with tool-bundle inclusion.

    Strategy:
    1. Start from the newest message
    2. Work backward, adding messages until target token budget is reached
    3. Special handling for tool calls: if a tool message is included, ensure
       its corresponding assistant message with tool_calls is also included
    4. Gracefully handle over-budget single messages (keep most recent)
    5. Apply headroom percentage to reserve space for system prompt/tools

    This strategy preserves recency and ensures tool-call coherence.
    """

    def retrieve(
        self,
        state: MemoryState,
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
    ) -> List[Dict[str, Any]]:
        """
        Pack messages backward from newest until compaction_target_tokens.

        Args:
            state: Memory state with raw messages
            config: Configuration with token limits
            token_counter: Token counter instance

        Returns:
            Curated list of messages in chronological order
        """
        raw_messages = state.raw_messages
        target = config.compaction_target_tokens

        # Apply headroom to reserve space for system prompt and tools
        sw_cfg = config.active_context.sliding_window
        if sw_cfg.headroom_percent > 0:
            target = int(target * (1 - sw_cfg.headroom_percent))

        # Convert messages to dict format for token counting
        msg_dicts = [self._message_to_dict(msg) for msg in raw_messages]

        curated = []
        total_tokens = 0
        i = len(msg_dicts) - 1

        while i >= 0:
            msg_dict = msg_dicts[i]
            msg_tokens = token_counter.count_message(msg_dict)

            # Check budget
            if total_tokens + msg_tokens > target:
                # Special case: if this is the ONLY message and it's the newest,
                # include it anyway (prefer showing SOMETHING over nothing)
                if len(curated) == 0 and i == len(msg_dicts) - 1:
                    curated.append(msg_dict)
                    total_tokens += msg_tokens
                    # Mark as over-budget in metadata
                    state.metadata["over_budget_single"] = True
                    logger.warning(
                        f"Single message exceeds target budget: {msg_tokens} > {target} tokens"
                    )
                break

            # Add message
            curated.append(msg_dict)
            total_tokens += msg_tokens

            # Tool-bundle inclusion logic
            if msg_dict.get("role") == "tool":
                # Scan backward to include the assistant message with tool_calls
                bundle_result = self._include_tool_bundle(
                    i - 1, msg_dicts, curated, target, total_tokens, token_counter
                )
                if bundle_result is None:
                    # Out of budget or malformed bundle - stop here
                    break
                i, total_tokens = bundle_result
            else:
                i -= 1

        # Reverse to restore chronological order
        curated.reverse()

        # Update metadata
        state.metadata["curated_count"] = len(curated)
        state.metadata["curated_tokens"] = total_tokens
        state.metadata["excluded_count"] = len(raw_messages) - len(curated)

        logger.debug(
            f"Curated {len(curated)}/{len(raw_messages)} messages "
            f"({total_tokens}/{target} tokens)"
        )

        return curated

    def _include_tool_bundle(
        self,
        start_idx: int,
        msg_dicts: List[Dict],
        curated: List[Dict],
        target: int,
        total_tokens: int,
        token_counter: "TokenCounter",
    ) -> Optional[tuple[int, int]]:
        """
        Scan backward from tool message to include its assistant with tool_calls.

        Delegates to the module-level _include_tool_bundle_backward helper
        and appends results to the curated list.

        Args:
            start_idx: Index to start scanning backward from
            msg_dicts: All message dicts
            curated: Current curated list (will be modified in-place)
            target: Target token budget
            total_tokens: Current total tokens (will be updated)
            token_counter: Token counter instance

        Returns:
            Tuple of (new_index, new_total_tokens) to continue from,
            or None if out of budget or bundle is malformed
        """
        result = _include_tool_bundle_backward(
            start_idx, msg_dicts, target, total_tokens, token_counter
        )
        if result is None:
            logger.debug(
                f"Tool bundle inclusion would exceed budget starting at index {start_idx}"
            )
            return None

        bundle_msgs, new_idx, new_tokens = result
        curated.extend(bundle_msgs)
        return new_idx, new_tokens

    def _message_to_dict(self, msg: "Message") -> Dict[str, Any]:
        """
        Convert Message to dict in LLM format.

        Delegates to the Message class's to_llm_dict() method for consistent
        formatting across the framework.

        Args:
            msg: Message object

        Returns:
            Dict in LLM API format
        """
        return msg.to_llm_dict()


# === Compaction Processor Interface and Implementations ===


class CompactionProcessor(ABC):
    """
    Base class for compaction processors used in the compaction pipeline.

    Processors are tried in order (configured by processor_order). Each
    processor estimates its reduction capability before being applied.
    The last processor in the chain always runs as a last resort.
    """

    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in config processor_order and registry."""

    @abstractmethod
    def priority(self) -> int:
        """Default priority (higher = tried first = least destructive)."""

    @abstractmethod
    async def estimate_reduction(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Estimate how many tokens this processor can save.
        Must be fast (no LLM calls). Returns estimated token savings.
        """

    @abstractmethod
    async def reduce(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply the reduction. Returns (reduced_messages, metadata)."""


# Backward compatibility alias
ContextReducer = CompactionProcessor


def _compute_protected_tail_start(
    messages: List[Dict[str, Any]],
    grace_recent_messages: int,
) -> int:
    """
    Compute the index where the protected tail begins.

    Uses assistant-round boundary semantics: ``grace_recent_messages = n``
    means "protect from the n-th most recent assistant message onward."
    An assistant round includes the assistant message itself plus all
    subsequent tool-response and user-payload messages until the next
    assistant message (or end of list).

    If ``grace_recent_messages <= 0`` it is treated as 1 (always protect at
    least one round).  If no assistant message is found the entire list is
    protected (returns ``len(messages)``).

    Args:
        messages: List of message dicts.
        grace_recent_messages: Number of recent assistant rounds to protect.

    Returns:
        Index into messages where the protected tail starts.
    """
    if not messages:
        return len(messages)

    n = max(grace_recent_messages, 1)

    # Walk backward, counting assistant messages
    assistant_count = 0
    boundary = len(messages)  # default: protect everything

    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            assistant_count += 1
            if assistant_count >= n:
                boundary = i
                break

    return boundary


def _is_tool_response_message(msg: Dict[str, Any], include_user_payloads: bool = True) -> bool:
    """Check if a message is a tool response that can be truncated."""
    role = msg.get("role")
    if role == "tool":
        return True
    if include_user_payloads and role == "user":
        # Check for tool-payload user messages (content with tool result patterns)
        content = msg.get("content", "")
        if isinstance(content, str) and content.startswith("Tool ") and "returned:" in content:
            return True
    return False


class ToolTruncationProcessor(CompactionProcessor):
    """
    Processor: truncate oversized tool response messages.

    This is the least-detrimental reduction — it shortens verbose tool outputs
    (e.g., large web pages, file contents) while preserving the message structure
    and all non-tool messages unchanged.

    Messages within the grace window (most recent N) are never truncated.
    """

    def name(self) -> str:
        return "tool_truncation"

    def priority(self) -> int:
        return 100

    async def estimate_reduction(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> int:
        tool_cfg = config.active_context.tool_truncation
        if not tool_cfg.enabled:
            return 0

        protected_start = _compute_protected_tail_start(
            messages, tool_cfg.grace_recent_messages
        )

        estimated_savings = 0
        for i, msg in enumerate(messages):
            if i >= protected_start:
                continue
            if _is_tool_response_message(msg, tool_cfg.include_tool_payload_user_messages):
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg_tokens = token_counter.count_message(msg)
                    if msg_tokens > tool_cfg.max_tool_message_tokens:
                        estimated_savings += msg_tokens - tool_cfg.max_tool_message_tokens
        return estimated_savings

    async def reduce(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        from marsys.utils.tokens import truncate_text_to_tokens

        tool_cfg = config.active_context.tool_truncation
        if not tool_cfg.enabled:
            return messages, {"stage": "tool_truncation", "skipped": True}

        protected_start = _compute_protected_tail_start(
            messages, tool_cfg.grace_recent_messages
        )

        updated = []
        truncated_count = 0
        tokens_saved = 0

        for i, msg in enumerate(messages):
            if i >= protected_start:
                updated.append(msg)
                continue

            if _is_tool_response_message(msg, tool_cfg.include_tool_payload_user_messages):
                content = msg.get("content", "")
                if isinstance(content, str):
                    old_tokens = token_counter.count_message(msg)
                    if old_tokens > tool_cfg.max_tool_message_tokens:
                        truncated_content = truncate_text_to_tokens(
                            content,
                            tool_cfg.max_tool_message_tokens,
                            marker=tool_cfg.append_marker,
                        )
                        new_msg = msg.copy()
                        new_msg["content"] = truncated_content
                        new_tokens = token_counter.count_message(new_msg)
                        tokens_saved += old_tokens - new_tokens
                        truncated_count += 1
                        updated.append(new_msg)
                        continue

            updated.append(msg)

        metadata = {
            "stage": "tool_truncation",
            "truncated_count": truncated_count,
            "tokens_saved": tokens_saved,
        }
        logger.debug(
            f"Tool truncation: truncated {truncated_count} messages, "
            f"saved ~{tokens_saved} tokens"
        )
        return updated, metadata


# === Provider Payload Limits (bytes) ===

PROVIDER_PAYLOAD_LIMITS = {
    "anthropic": 32_000_000,
    "anthropic-oauth": 32_000_000,
    "openai": 25_000_000,
    "openai-oauth": 25_000_000,
    "google": 100_000_000,
    "openrouter": 25_000_000,
    "xai": 25_000_000,
    "default": 25_000_000,
}


# === Compaction Prompts ===


COMPACTION_SYSTEM_PROMPT = """\
You are a context compaction assistant. Your task is to compress a conversation \
history into a concise summary that preserves all information needed for the \
agent to continue its work effectively.

You MUST respond with valid JSON matching the required schema. Do not include \
any text outside the JSON object.

Preserve:
- Key facts, decisions, and conclusions reached
- Current state of any ongoing tasks or threads
- Important constraints or requirements mentioned
- Any unresolved questions or pending actions
- References to specific data, URLs, file paths, or identifiers

Discard:
- Verbose tool outputs that have already been processed
- Redundant re-statements of the same information
- Conversational filler or acknowledgments
- Intermediate reasoning that led to already-captured conclusions

Additionally, extract all user requests and requirements into a separate \
"user_request_summary" field. This should be a concise summary of what the user \
asked for, preserving specific details, constraints, and requirements. \
The instruction message at the end is NOT a user request — do not include it.

Image Retention:
- Be highly selective — images are expensive in payload size
- Only retain images that will be required for finishing the task\
"""

COMPACTION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_request_summary": {
            "type": "string",
            "description": "Concise summary of all user requests and requirements from the conversation.",
        },
        "summary": {
            "type": "string",
            "description": "Concise narrative summary of the conversation so far.",
        },
        "salient_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key facts, decisions, and data points to preserve.",
        },
        "open_threads": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Unresolved tasks, pending actions, or open questions.",
        },
        "keep_images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "The image ID from the conversation (e.g., img_0000).",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Brief description of what this image represents and "
                            "in which context it was used. This will accompany the "
                            "image after compaction."
                        ),
                    },
                },
                "required": ["image_id", "context"],
            },
            "description": (
                "Images to retain. Only include images with unique, essential "
                "information not captured in the text summary."
            ),
        },
    },
    "required": ["user_request_summary", "summary", "salient_facts", "open_threads", "keep_images"],
}


def _extract_colocated_text(content: Any, max_chars: int = 150) -> str:
    """Extract co-located text from message content for image manifest context."""
    if isinstance(content, str):
        return content[:max_chars]
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)[:max_chars]
    return ""


def _content_to_blocks(content: Any) -> List[Dict[str, Any]]:
    """Convert any content form to a list of typed-array blocks."""
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _build_compaction_payload(
    prefix_messages: List[Dict[str, Any]],
    agent_instruction: Optional[str] = None,
    include_image_bytes: bool = False,
    compact_cfg: Any = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Build the compaction input payload from prefix messages.

    Preserves the original conversation structure (roles, tool_calls, etc.).
    Only modifies typed-array content to inject image IDs and optionally
    strip image bytes. Prepends a system message and appends a final user
    message with the image manifest and summarization instruction.

    Args:
        prefix_messages: Messages to compact (prefix, not protected tail).
        agent_instruction: Optional agent instruction to include for context.
        include_image_bytes: Whether to include base64 image data.
        compact_cfg: SummarizationConfig with output_max_tokens for length guidance.

    Returns:
        Tuple of (compaction_messages list, image_map {id: original_url/data}).
    """
    image_map: Dict[str, str] = {}
    image_context_map: Dict[str, str] = {}
    image_counter = 0

    # --- Step 1: System message with length guidance ---
    max_words = int((compact_cfg.output_max_tokens if compact_cfg else 6000) * 0.75 * 0.9)
    length_guidance = (
        f"\n\nIMPORTANT: Your entire JSON response must be at most "
        f"approximately {max_words} words. Keep the summary concise and "
        f"each salient fact to one sentence."
    )
    system_msg = {"role": "system", "content": COMPACTION_SYSTEM_PROMPT + length_guidance}

    # --- Step 2: Pass through prefix messages, only modifying image content ---
    conversation_messages: List[Dict[str, Any]] = []

    for msg in prefix_messages:
        content = msg.get("content")

        if isinstance(content, list):
            # Typed-array content — inject image IDs, optionally strip bytes
            colocated_text = _extract_colocated_text(content)
            new_blocks: List[Dict[str, Any]] = []

            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image_url":
                    image_id = f"img_{image_counter:04d}"
                    image_counter += 1
                    url = item.get("image_url", {}).get("url", "")
                    image_map[image_id] = url
                    image_context_map[image_id] = colocated_text

                    new_blocks.append({
                        "type": "text",
                        "text": f"[Image ID: {image_id}]",
                    })
                    if include_image_bytes and url:
                        new_blocks.append(item)
                else:
                    new_blocks.append(item)

            new_msg = {k: v for k, v in msg.items() if k != "content"}
            new_msg["content"] = new_blocks if new_blocks else [{"type": "text", "text": ""}]
            conversation_messages.append(new_msg)
        else:
            # String, dict, None — keep as-is
            conversation_messages.append(msg)

    # --- Step 3: Append final user message with manifest + instruction ---
    final_parts = []
    if agent_instruction:
        final_parts.append(f"[Agent Instruction]\n{agent_instruction}")
    if image_map:
        manifest_lines = ["[Image Manifest]"]
        for img_id in image_map:
            ctx = image_context_map.get(img_id, "")
            manifest_lines.append(f'  {img_id}: "{ctx}"')
        final_parts.append("\n".join(manifest_lines))
    final_parts.append(
        "Now summarize the conversation above. Preserve user requests "
        "and compact the context needed to finish the task. Respond with "
        "the required JSON."
    )
    final_user_content = "\n\n".join(final_parts)

    # Merge into last message if also user role to avoid consecutive same-role
    if conversation_messages and conversation_messages[-1].get("role") == "user":
        last = conversation_messages[-1]
        last_blocks = _content_to_blocks(last.get("content"))
        last_blocks.append({"type": "text", "text": final_user_content})
        last["content"] = last_blocks
    else:
        conversation_messages.append({
            "role": "user",
            "content": final_user_content,
        })

    return [system_msg] + conversation_messages, image_map


def _build_user_rollup(
    prefix_messages: List[Dict[str, Any]],
) -> str:
    """
    Build a concatenated user-request rollup from prefix user messages.

    Args:
        prefix_messages: Messages from the compacted prefix.

    Returns:
        Concatenated user request text.
    """
    user_parts = []
    for msg in prefix_messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                user_parts.append(content.strip())
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                if text_parts:
                    user_parts.append(" ".join(text_parts).strip())
    return "\n---\n".join(user_parts)


def _parse_compaction_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse compaction model output as JSON with fallback extraction.

    Tries direct parse first, then searches for JSON block in text.

    Args:
        raw_text: Raw text from compaction model.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not raw_text:
        return None

    # Try direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    import re
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_start = raw_text.find("{")
    brace_end = raw_text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(raw_text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse compaction JSON from model output")
    return None


class SummarizationProcessor(CompactionProcessor):
    """
    Processor: model-based context summarization.

    Splits messages into a prefix (to summarize) and a protected tail (to preserve).
    Calls a compaction model to summarize the prefix into a synthetic summary,
    builds a user-request rollup, and optionally retains selected images.
    """

    def name(self) -> str:
        return "summarization"

    def priority(self) -> int:
        return 50

    async def estimate_reduction(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> int:
        compact_cfg = config.active_context.summarization
        if not compact_cfg.enabled:
            return 0

        protected_start = _compute_protected_tail_start(
            messages, compact_cfg.grace_recent_messages
        )

        if protected_start <= 1:
            return 0

        prefix_tokens = token_counter.count_messages(messages[:protected_start])[0]

        # Estimate output as ~50% of input, capped at output_max_tokens
        estimated_output = min(compact_cfg.output_max_tokens, int(prefix_tokens * 0.5))
        return max(0, prefix_tokens - estimated_output)

    async def reduce(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        compact_cfg = config.active_context.summarization
        if not compact_cfg.enabled:
            return messages, {"stage": "summarization", "skipped": True}

        # Split into prefix and protected tail
        protected_start = _compute_protected_tail_start(
            messages, compact_cfg.grace_recent_messages
        )

        # Need at least some messages in prefix to compact
        if protected_start <= 1:
            return messages, {"stage": "summarization", "skipped": True, "reason": "insufficient_prefix"}

        prefix = messages[:protected_start]
        protected_tail = messages[protected_start:]
        total_tokens = token_counter.count_messages(messages)[0]

        # Preserve leading system message if present (will be reinserted after compaction)
        original_system_msg = None
        if messages and messages[0].get("role") == "system":
            original_system_msg = messages[0]
            messages = messages[1:]
            # Recompute split boundaries on the remaining messages
            protected_start = _compute_protected_tail_start(
                messages, compact_cfg.grace_recent_messages
            )
            if protected_start <= 1:
                # Reinsert system msg and return
                return [original_system_msg] + messages, {
                    "stage": "summarization", "skipped": True, "reason": "insufficient_prefix",
                }
            prefix = messages[:protected_start]
            protected_tail = messages[protected_start:]
            total_tokens = token_counter.count_messages(
                ([original_system_msg] + messages)
            )[0]

        # Build compaction payload
        agent_instruction = None
        if compact_cfg.include_original_instruction and runtime:
            agent_instruction = runtime.get("agent_instruction")

        compaction_messages, image_map = _build_compaction_payload(
            prefix,
            agent_instruction=agent_instruction,
            include_image_bytes=compact_cfg.include_image_payload_bytes,
            compact_cfg=compact_cfg,
        )

        # Get compaction model
        compaction_model = None
        compaction_model_name = "unknown"
        if runtime:
            compaction_model = runtime.get("compaction_model")
            compaction_model_name = runtime.get("compaction_model_name", "unknown")

        if compaction_model is None:
            logger.warning("No compaction model available, skipping summarization")
            result = ([original_system_msg] + messages) if original_system_msg else messages
            return result, {"stage": "summarization", "skipped": True, "reason": "no_model"}

        # Single model call returns everything (including user_request_summary)
        t0 = time.time()
        summary_json = await self._run_compaction_model(
            compaction_messages, compaction_model, compact_cfg
        )
        compaction_elapsed = time.time() - t0

        if not summary_json:
            logger.warning("Compaction model returned no valid JSON; keeping prefix unchanged")
            result = ([original_system_msg] + messages) if original_system_msg else messages
            return result, {"stage": "summarization", "skipped": True, "reason": "parse_failure"}

        # Build result messages
        reduced: List[Dict[str, Any]] = []

        # 0. Reinsert original system message
        if original_system_msg is not None:
            reduced.append(original_system_msg)

        # 1. User request summary (role: user) — FIRST
        user_req_summary = summary_json.get("user_request_summary", "")
        if user_req_summary:
            reduced.append({
                "role": "user",
                "content": f"[User Requests Summary]\n{user_req_summary}",
            })

        # 2. Conversation summary (role: assistant)
        summary_text = summary_json.get("summary", "")
        salient_facts = summary_json.get("salient_facts", [])
        open_threads = summary_json.get("open_threads", [])

        summary_parts = [f"[Compacted Context Summary]\n{summary_text}"]
        if salient_facts:
            summary_parts.append("\n[Key Facts]\n" + "\n".join(f"- {f}" for f in salient_facts))
        if open_threads:
            summary_parts.append("\n[Open Threads]\n" + "\n".join(f"- {t}" for t in open_threads))

        reduced.append({
            "role": "assistant",
            "content": "\n".join(summary_parts),
            "name": f"compaction_{compaction_model_name}",
        })

        # 3. Retain selected images with byte-budget packing
        keep_images = summary_json.get("keep_images", [])
        if keep_images and image_map:
            valid_items = [
                item for item in keep_images
                if isinstance(item, dict) and item.get("image_id") in image_map
            ]

            if valid_items:
                provider = runtime.get("provider", "unknown") if runtime else "unknown"
                provider_limit = PROVIDER_PAYLOAD_LIMITS.get(
                    provider, PROVIDER_PAYLOAD_LIMITS["default"]
                )
                byte_budget = int(provider_limit * 0.5)
                max_count = compact_cfg.max_retained_images

                # Most recent first
                valid_items.sort(key=lambda x: x["image_id"], reverse=True)

                final_keep: List[Dict[str, Any]] = []
                bytes_used = 0
                for item in valid_items:
                    if max_count is not None and len(final_keep) >= max_count:
                        break
                    img_id = item["image_id"]
                    url = image_map[img_id]
                    img_bytes = len(url) if url else 0
                    if bytes_used + img_bytes > byte_budget:
                        continue
                    bytes_used += img_bytes
                    final_keep.append(item)

                for item in final_keep:
                    img_id = item["image_id"]
                    ctx = item.get("context", "")
                    url = image_map[img_id]
                    if url:
                        reduced.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[Retained image {img_id}]: {ctx}"},
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        })

        # 4. Append protected tail
        reduced.extend(protected_tail)

        post_tokens = token_counter.count_messages(reduced)[0]
        images_retained = sum(
            1 for m in reduced
            if isinstance(m.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "image_url"
                for b in m["content"]
            )
            and m not in protected_tail
        )
        metadata = {
            "stage": "summarization",
            "pre_compaction_tokens": total_tokens,
            "pre_compaction_messages": len(messages),
            "post_compaction_tokens": post_tokens,
            "post_compaction_messages": len(reduced),
            "prefix_count": len(prefix),
            "protected_tail_count": len(protected_tail),
            "compaction_model": compaction_model_name,
            "images_retained": images_retained,
            "compacted_at": time.time(),
            "duration_seconds": round(compaction_elapsed, 2),
        }
        logger.info(
            f"Summarization: {len(messages)} msgs / ~{total_tokens} tok -> "
            f"{len(reduced)} msgs / ~{post_tokens} tok "
            f"({compaction_elapsed:.1f}s, model={compaction_model_name})"
        )
        return reduced, metadata

    async def _run_compaction_model(
        self,
        compaction_messages: List[Dict[str, Any]],
        model: Any,
        compact_cfg: Any,
    ) -> Optional[Dict[str, Any]]:
        """Run the compaction model on pre-built messages and parse JSON output."""
        try:
            kwargs: Dict[str, Any] = {
                "messages": compaction_messages,
                "max_tokens": compact_cfg.output_max_tokens,
                "temperature": 0.1,
                "json_mode": True,
            }

            try:
                kwargs["response_schema"] = COMPACTION_OUTPUT_SCHEMA
            except Exception:
                pass

            if hasattr(model, "arun"):
                response = await model.arun(**kwargs)
            elif hasattr(model, "run"):
                response = model.run(**kwargs)
            else:
                logger.error("Compaction model has no run/arun method")
                return None

            raw_text = ""
            if hasattr(response, "content"):
                raw_text = response.content if isinstance(response.content, str) else str(response.content)
            elif isinstance(response, dict):
                raw_text = response.get("content", str(response))
            elif isinstance(response, str):
                raw_text = response
            else:
                raw_text = str(response)

            return _parse_compaction_json(raw_text)

        except Exception as e:
            logger.error(f"Compaction model call failed: {e}")
            return None



class BackwardPackingProcessor(CompactionProcessor):
    """
    Last-resort processor: keeps only the most recent messages that fit
    within compaction_target_tokens. Always runs when it's the last in the chain.

    Features:
    - Grace guarantee: always preserves at least N complete units from the end
      (a "unit" is a full tool bundle or a single non-tool message).
    - Skip logic: when a tool bundle exceeds remaining budget, skips it and
      tries earlier messages instead of stopping entirely.
    """

    def name(self) -> str:
        return "backward_packing"

    def priority(self) -> int:
        return 0

    async def estimate_reduction(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> int:
        total_tokens = token_counter.count_messages(messages)[0]
        return max(0, total_tokens - config.compaction_target_tokens)

    @staticmethod
    def _skip_tool_bundle_backward(tool_idx: int, messages: List[Dict[str, Any]]) -> int:
        """Skip backward past an entire tool bundle. Returns next index to try."""
        j = tool_idx - 1
        while j >= 0:
            role = messages[j].get("role")
            if role == "tool" or role == "user":
                j -= 1
                continue
            if role == "assistant" and messages[j].get("tool_calls"):
                return j - 1
            return j
        return j

    async def reduce(
        self,
        messages: List[Dict[str, Any]],
        config: "ManagedMemoryConfig",
        token_counter: "TokenCounter",
        runtime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        target = config.compaction_target_tokens
        total_tokens = token_counter.count_messages(messages)[0]

        # Grace tail: always preserved even if over budget
        bp_cfg = getattr(config.active_context, "backward_packing", None)
        grace_n = bp_cfg.grace_recent_messages if bp_cfg else 1
        grace_boundary = _compute_protected_tail_start(messages, grace_n)
        grace_messages = messages[grace_boundary:]
        grace_tokens = token_counter.count_messages(grace_messages)[0] if grace_messages else 0
        remaining_budget = max(0, target - grace_tokens)

        # Backward pack non-grace messages
        non_grace = messages[:grace_boundary]
        result = []
        running_total = 0
        i = len(non_grace) - 1

        while i >= 0:
            msg = non_grace[i]
            msg_tokens = token_counter.count_message(msg)

            if running_total + msg_tokens > remaining_budget:
                # Over budget — skip this message/bundle instead of stopping
                if msg.get("role") == "tool":
                    i = self._skip_tool_bundle_backward(i, non_grace)
                else:
                    i -= 1
                continue

            result.append(msg)
            running_total += msg_tokens

            if msg.get("role") == "tool":
                bundle_result = _include_tool_bundle_backward(
                    i - 1, non_grace, remaining_budget, running_total, token_counter
                )
                if bundle_result is None:
                    # Can't fit the full bundle — remove tool msg, skip bundle
                    result.pop()
                    running_total -= msg_tokens
                    i = self._skip_tool_bundle_backward(i, non_grace)
                    continue
                bundle_msgs, new_i, new_total = bundle_result
                result.extend(bundle_msgs)
                running_total = new_total
                i = new_i
            else:
                i -= 1

        result.reverse()
        result.extend(grace_messages)

        # Ensure at least one user message is present. In tool-use
        # conversations the only user message may be the initial task which
        # backward packing drops to stay within budget.  Walk the original
        # messages backwards and inject the most recent user message.
        has_user = any(m.get("role") == "user" for m in result)
        user_injected = False
        if not has_user:
            for j in range(len(messages) - 1, -1, -1):
                if messages[j].get("role") == "user":
                    result.insert(0, messages[j])
                    user_injected = True
                    break

        post_tokens = token_counter.count_messages(result)[0] if result else 0
        return result, {
            "stage": "backward_packing",
            "pre_tokens": total_tokens,
            "post_tokens": post_tokens,
            "messages_dropped": len(messages) - len(result),
            "grace_messages": len(grace_messages),
            "grace_tokens": grace_tokens,
            "user_injected": user_injected,
        }


# Backward compatibility aliases
ToolResponseTruncationReducer = ToolTruncationProcessor
ModelCompactionReducer = SummarizationProcessor


# === Compaction Processor Registry ===

COMPACTION_PROCESSOR_REGISTRY: Dict[str, type] = {
    "tool_truncation": ToolTruncationProcessor,
    "summarization": SummarizationProcessor,
    "backward_packing": BackwardPackingProcessor,
    # Backward compat aliases
    "tool_response_truncation": ToolTruncationProcessor,
    "model_compaction": SummarizationProcessor,
}

# Backward compatibility alias
REDUCER_REGISTRY = COMPACTION_PROCESSOR_REGISTRY

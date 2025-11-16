"""
Memory management strategies for active context management.

This module defines the strategy interfaces and simple implementations for:
- Trigger strategies: Decide WHEN to engage ACM
- Process strategies: Define HOW to process messages (summarize, ACE, etc.)
- Retrieval strategies: Define HOW to retrieve curated context
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from marsys.agents.memory import Message, ManagedMemoryConfig
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


# === Simple Implementations (v1) ===


class SimpleThresholdTrigger(TriggerStrategy):
    """
    Trigger ACM when total tokens exceed a threshold.

    This is the simplest trigger strategy: engage ACM when estimated total tokens
    cross max_total_tokens_trigger, or when enough new messages/tokens have been
    added since the last retrieval.
    """

    def should_trigger_on_add(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """Trigger if total tokens exceed threshold."""
        return state.estimated_tokens > config.max_total_tokens_trigger

    def should_trigger_on_retrieve(
        self, state: MemoryState, config: "ManagedMemoryConfig"
    ) -> bool:
        """
        Trigger if:
        1. Over threshold, OR
        2. Enough new messages since last retrieval, OR
        3. Enough new tokens since last retrieval
        """
        over_threshold = state.estimated_tokens > config.max_total_tokens_trigger
        enough_new_messages = (
            state.messages_since_last_retrieval >= config.min_retrieval_gap_steps
        )
        enough_new_tokens = (
            state.tokens_since_last_retrieval >= config.min_retrieval_gap_tokens
        )

        return over_threshold or enough_new_messages or enough_new_tokens


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
        Pack messages backward from newest until target_total_tokens.

        Args:
            state: Memory state with raw messages
            config: Configuration with token limits
            token_counter: Token counter instance

        Returns:
            Curated list of messages in chronological order
        """
        raw_messages = state.raw_messages
        target = config.target_total_tokens

        # Apply headroom to reserve space for system prompt and tools
        if config.enable_headroom_percent > 0:
            target = int(target * (1 - config.enable_headroom_percent))

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
            f"({total_tokens}/{config.target_total_tokens} tokens)"
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
        j = start_idx
        tokens = total_tokens

        while j >= 0:
            msg = msg_dicts[j]
            msg_tokens = token_counter.count_message(msg)

            # Check budget first
            if tokens + msg_tokens > target:
                # Out of budget - cannot include bundle
                logger.debug(
                    f"Tool bundle inclusion would exceed budget at index {j}"
                )
                return None

            role = msg.get("role")

            # If this is another tool message, include it and continue
            if role == "tool":
                curated.append(msg)
                tokens += msg_tokens
                j -= 1
                continue

            # If this is assistant message, check for tool_calls
            if role == "assistant":
                if msg.get("tool_calls"):
                    # Found the originating assistant message
                    curated.append(msg)
                    tokens += msg_tokens
                    return j - 1, tokens
                else:
                    # Assistant without tool_calls - not the right one
                    # This shouldn't happen in well-formed conversation
                    logger.warning(
                        f"Found assistant without tool_calls while bundling tool at index {j}"
                    )
                    return j, tokens

            # If this is user message, include it (might be in between)
            if role == "user":
                curated.append(msg)
                tokens += msg_tokens
                j -= 1
                continue

            # Unknown role or system - stop bundling
            logger.warning(
                f"Unexpected role '{role}' while bundling tool messages at index {j}"
            )
            return j, tokens

        # Reached start of messages without finding assistant - malformed
        logger.warning("Reached start of messages while bundling tool calls")
        return None

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

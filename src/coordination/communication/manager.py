"""
Communication manager for handling user interactions across different channels.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from .core import (
    CommunicationMode,
    UserInteraction,
    SyncChannel,
    AsyncChannel,
    CommunicationChannel
)

logger = logging.getLogger(__name__)


class CommunicationManager:
    """
    Central manager for bi-directional user communication.

    Design Principle: User Dialogue Ownership
    - Handles the complete interaction experience (content display + input collection)
    - Manages synchronous (terminal) and asynchronous (web) communication patterns
    - Owns the presentation layer for user interactions
    - Routes interactions to appropriate channels based on mode

    This manager is separate from StatusManager, which handles one-way system
    observability. CommunicationManager focuses on interactive dialogue while
    StatusManager focuses on event broadcasting and monitoring.
    """
    
    def __init__(self):
        # Channels by type
        self.sync_channels: Dict[str, SyncChannel] = {}
        self.async_channels: Dict[str, AsyncChannel] = {}
        
        # Session to channel mapping
        self.session_channels: Dict[str, str] = {}  # session_id -> channel_id
        
        # Pub/sub infrastructure
        self.subscribers: Dict[str, List[Callable]] = {}  # topic -> callbacks
        self.topic_queues: Dict[str, asyncio.Queue] = {}  # topic -> queue
        
        # Interaction tracking
        self.pending_interactions: Dict[str, UserInteraction] = {}
        self.interaction_history: Dict[str, List[UserInteraction]] = {}  # Per-session history
        
        # Response handling
        self.response_futures: Dict[str, asyncio.Future] = {}  # For sync
        self.response_queues: Dict[str, asyncio.Queue] = {}    # For async
        
        # Background tasks
        self._tasks: Set[asyncio.Task] = set()
        
        logger.info("CommunicationManager initialized")
    
    def register_channel(self, channel: CommunicationChannel) -> None:
        """Register a communication channel."""
        if isinstance(channel, SyncChannel):
            self.sync_channels[channel.channel_id] = channel
            logger.info(f"Registered sync channel: {channel.channel_id}")
        elif isinstance(channel, AsyncChannel):
            self.async_channels[channel.channel_id] = channel
            logger.info(f"Registered async channel: {channel.channel_id}")
        else:
            raise ValueError(f"Unknown channel type: {type(channel)}")
    
    def assign_channel_to_session(self, session_id: str, channel_id: str) -> None:
        """Assign a channel to a session."""
        self.session_channels[session_id] = channel_id
        logger.debug(f"Assigned channel {channel_id} to session {session_id}")
    
    async def handle_interaction(self, interaction: UserInteraction) -> Optional[Any]:
        """
        Route interaction based on communication mode.
        
        Returns:
            For SYNC mode: The user response
            For ASYNC modes: None (response comes via callback/subscription)
        """
        # Store interaction
        self.pending_interactions[interaction.interaction_id] = interaction
        self.add_to_history(interaction.session_id, interaction)
        
        logger.info(f"Handling interaction {interaction.interaction_id} "
                   f"from {interaction.calling_agent} in {interaction.communication_mode} mode")
        
        if interaction.communication_mode == CommunicationMode.SYNC:
            return await self._handle_sync_interaction(interaction)
        elif interaction.communication_mode == CommunicationMode.ASYNC_PUBSUB:
            return await self._handle_pubsub_interaction(interaction)
        elif interaction.communication_mode == CommunicationMode.ASYNC_QUEUE:
            return await self._handle_queue_interaction(interaction)
        else:
            raise ValueError(f"Unknown communication mode: {interaction.communication_mode}")
    
    async def _handle_sync_interaction(self, interaction: UserInteraction) -> Any:
        """Handle synchronous blocking interaction."""
        # Create future for response
        future = asyncio.Future()
        self.response_futures[interaction.interaction_id] = future
        
        try:
            # Get channel for session
            channel = self._get_sync_channel(interaction.session_id)
            if not channel:
                raise ValueError(f"No sync channel available for session {interaction.session_id}")
            
            # Use atomic send_and_wait if available (for terminal channel)
            if hasattr(channel, 'send_and_wait_for_response'):
                interaction_id, user_response = await channel.send_and_wait_for_response(interaction)
            else:
                # Fallback to separate send/get for other channel types
                await channel.send_interaction(interaction)
                interaction_id, user_response = await channel.get_response(interaction.interaction_id)
            
            # Set the future result
            future.set_result(user_response)
            
            # Wait for response (blocking)
            timeout = interaction.timeout or 300  # Default 5 minute timeout
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
                logger.info(f"Received sync response for interaction {interaction.interaction_id}")
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response to interaction {interaction.interaction_id}")
                raise TimeoutError(f"User response timeout for {interaction.interaction_id}")
                
        finally:
            # Clean up
            self.response_futures.pop(interaction.interaction_id, None)
    
    async def _handle_pubsub_interaction(self, interaction: UserInteraction) -> None:
        """Handle pub/sub pattern - non-blocking."""
        topic = interaction.topic or f"user_interaction_{interaction.session_id}"
        
        # Create response queue for this interaction
        response_queue = asyncio.Queue()
        self.response_queues[interaction.interaction_id] = response_queue
        
        # Subscribe to response topic
        response_topic = f"{topic}_response"
        
        async def handle_response(message: Dict[str, Any]):
            if message.get("interaction_id") == interaction.interaction_id:
                await response_queue.put(message.get("response"))
        
        self.subscribe(response_topic, handle_response)
        
        # Publish interaction
        await self.publish(topic, {
            "type": "interaction_request",
            "interaction": interaction.to_display_dict()
        })
        
        logger.info(f"Published interaction {interaction.interaction_id} to topic {topic}")
    
    async def _handle_queue_interaction(self, interaction: UserInteraction) -> None:
        """Handle queue-based async pattern."""
        queue_name = interaction.queue_name or f"user_queue_{interaction.session_id}"
        
        # Get or create queue
        if queue_name not in self.topic_queues:
            self.topic_queues[queue_name] = asyncio.Queue()
        
        queue = self.topic_queues[queue_name]
        
        # Put interaction in queue
        await queue.put({
            "type": "interaction",
            "data": interaction.to_display_dict()
        })
        
        logger.info(f"Queued interaction {interaction.interaction_id} in {queue_name}")
    
    def _get_sync_channel(self, session_id: str) -> Optional[SyncChannel]:
        """Get sync channel for session."""
        channel_id = self.session_channels.get(session_id)
        if channel_id and channel_id in self.sync_channels:
            return self.sync_channels[channel_id]
        
        # Return first available sync channel as fallback
        if self.sync_channels:
            return next(iter(self.sync_channels.values()))
        
        return None
    
    async def submit_response(self, interaction_id: str, response: Any) -> bool:
        """
        Submit a response to a pending interaction.
        
        Returns:
            True if response was delivered, False otherwise
        """
        interaction = self.pending_interactions.get(interaction_id)
        if not interaction:
            logger.warning(f"No pending interaction found: {interaction_id}")
            return False
        
        logger.info(f"Submitting response for interaction {interaction_id}")
        
        if interaction.communication_mode == CommunicationMode.SYNC:
            # Resolve future for sync mode
            future = self.response_futures.get(interaction_id)
            if future and not future.done():
                future.set_result(response)
                return True
                
        elif interaction.communication_mode == CommunicationMode.ASYNC_PUBSUB:
            # Publish response
            topic = f"user_interaction_{interaction.session_id}_response"
            await self.publish(topic, {
                "interaction_id": interaction_id,
                "response": response
            })
            return True
            
        elif interaction.communication_mode == CommunicationMode.ASYNC_QUEUE:
            # Put response in queue
            queue = self.response_queues.get(interaction_id)
            if queue:
                await queue.put(response)
                return True
        
        return False
    
    async def publish(self, topic: str, message: Any) -> None:
        """Publish message to topic."""
        if topic in self.subscribers:
            # Call all subscribers
            tasks = []
            for callback in self.subscribers[topic]:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(message))
                    tasks.append(task)
                else:
                    # Run sync callbacks in executor
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(None, callback, message)
                    tasks.append(task)
            
            # Wait for all callbacks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Also put in topic queue if exists
        if topic in self.topic_queues:
            await self.topic_queues[topic].put(message)
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to topic with callback."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.debug(f"Added subscriber to topic {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Unsubscribe callback from topic."""
        if topic in self.subscribers:
            try:
                self.subscribers[topic].remove(callback)
                logger.debug(f"Removed subscriber from topic {topic}")
            except ValueError:
                pass
    
    def get_pending_interactions(
        self,
        session_id: Optional[str] = None,
        mode: Optional[CommunicationMode] = None,
        calling_agent: Optional[str] = None
    ) -> List[UserInteraction]:
        """Get pending interactions with optional filters."""
        interactions = list(self.pending_interactions.values())
        
        if session_id:
            interactions = [i for i in interactions if i.session_id == session_id]
        if mode:
            interactions = [i for i in interactions if i.communication_mode == mode]
        if calling_agent:
            interactions = [i for i in interactions if i.calling_agent == calling_agent]
        
        return interactions
    
    def add_to_history(self, session_id: str, interaction: UserInteraction) -> None:
        """Add interaction to session history."""
        if session_id not in self.interaction_history:
            self.interaction_history[session_id] = []
        self.interaction_history[session_id].append(interaction)

    def get_interaction_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[UserInteraction]:
        """Get interaction history for a specific session."""
        history = self.interaction_history.get(session_id, [])
        if limit and len(history) > limit:
            return history[-limit:]
        return history.copy()
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        # Clear pending futures
        for future in self.response_futures.values():
            if not future.done():
                future.cancel()
        
        # Stop channels
        for channel in list(self.sync_channels.values()) + list(self.async_channels.values()):
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"Error stopping channel {channel.channel_id}: {e}")
        
        logger.info("CommunicationManager cleanup complete")

    async def present_results_and_wait_for_follow_up(
        self,
        results: str,
        session_id: str,
        timeout: float = 30.0,
        allow_follow_up: bool = True
    ) -> Optional[str]:
        """
        Present results to user and optionally wait for follow-up request.

        Args:
            results: The results to present to user
            session_id: Session identifier
            timeout: Time to wait for follow-up (seconds)
            allow_follow_up: Whether to wait for follow-up requests

        Returns:
            Follow-up request if provided within timeout, None otherwise
        """
        # First, present the results
        await self.present_results(results, session_id)

        if not allow_follow_up:
            return None

        # Create interaction for follow-up request
        follow_up_interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            session_id=session_id,
            branch_id=None,  # Not tied to specific branch
            incoming_message="Enter a follow-up request or press Enter to finish:",
            interaction_type="follow_up",
            timestamp=time.time(),
            communication_mode=CommunicationMode.SYNC,
            calling_agent="System",
            resume_agent="System",
            timeout=timeout
        )

        try:
            # Wait for user response with timeout
            response = await self.handle_interaction(follow_up_interaction)

            # Return response if non-empty, None otherwise
            if response and isinstance(response, str) and response.strip():
                return response.strip()
            return None

        except (asyncio.TimeoutError, TimeoutError):
            # No follow-up within timeout
            logger.info(f"No follow-up received within {timeout}s for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error waiting for follow-up: {e}")
            return None

    async def present_results(
        self,
        results: str,
        session_id: str,
        format: str = "text"  # text, markdown, json
    ) -> None:
        """
        Present results to user without waiting for response.

        Args:
            results: The results to present
            session_id: Session identifier
            format: Format of the results (text, markdown, json)
        """
        # Get appropriate channel for session
        channel = self._get_sync_channel(session_id)

        if channel and hasattr(channel, 'display_results'):
            # Use channel's display method if available
            await channel.display_results(results, format)
        else:
            # Fallback to creating a display-only interaction
            display_interaction = UserInteraction(
                interaction_id=str(uuid.uuid4()),
                session_id=session_id,
                branch_id=None,
                incoming_message=results,
                interaction_type="result_display",
                timestamp=time.time(),
                communication_mode=CommunicationMode.SYNC,
                calling_agent="System",
                resume_agent=None,  # No resume needed
                metadata={"format": format, "display_only": True}
            )

            # Send to channel for display
            if channel:
                await channel.send_interaction(display_interaction)
            else:
                # Log if no channel available
                logger.warning(f"No channel available to display results for session {session_id}")

    async def request_user_confirmation(
        self,
        prompt: str,
        session_id: str,
        agent_name: Optional[str] = None,
        timeout: float = 60.0
    ) -> bool:
        """
        Request yes/no confirmation from user.

        Args:
            prompt: The confirmation prompt
            session_id: Session identifier
            agent_name: Name of agent requesting confirmation
            timeout: Time to wait for response

        Returns:
            True if confirmed, False otherwise
        """
        # Format prompt for confirmation
        formatted_prompt = f"{prompt}\nProceed? (yes/no):"

        # Create confirmation interaction
        confirmation_interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            session_id=session_id,
            branch_id=None,
            incoming_message=formatted_prompt,
            interaction_type="confirmation",
            timestamp=time.time(),
            communication_mode=CommunicationMode.SYNC,
            calling_agent=agent_name or "System",
            resume_agent=agent_name or "System",
            metadata={"options": ["yes", "no"]},
            timeout=timeout
        )

        try:
            response = await self.handle_interaction(confirmation_interaction)

            # Parse response as boolean
            if response and isinstance(response, str):
                return response.lower() in ["yes", "y", "true", "1", "ok", "sure"]
            return False

        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(f"Confirmation timeout for session {session_id}")
            return False
        except Exception as e:
            logger.error(f"Error getting confirmation: {e}")
            return False

    async def request_user_choice(
        self,
        prompt: str,
        options: List[str],
        session_id: str,
        agent_name: Optional[str] = None,
        timeout: float = 60.0
    ) -> Optional[str]:
        """
        Request user to choose from options.

        Args:
            prompt: The choice prompt
            options: List of valid options
            session_id: Session identifier
            agent_name: Name of agent requesting choice
            timeout: Time to wait for response

        Returns:
            Selected option or None if timeout/error
        """
        # Format prompt with numbered options
        formatted_options = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        formatted_prompt = f"{prompt}\n\n{formatted_options}\n\nEnter your choice (1-{len(options)}):"

        # Create choice interaction
        choice_interaction = UserInteraction(
            interaction_id=str(uuid.uuid4()),
            session_id=session_id,
            branch_id=None,
            incoming_message=formatted_prompt,
            interaction_type="choice",
            timestamp=time.time(),
            communication_mode=CommunicationMode.SYNC,
            calling_agent=agent_name or "System",
            resume_agent=agent_name or "System",
            metadata={"options": options},
            timeout=timeout
        )

        try:
            response = await self.handle_interaction(choice_interaction)

            if response:
                # Try to parse as number
                try:
                    choice_idx = int(response) - 1
                    if 0 <= choice_idx < len(options):
                        return options[choice_idx]
                except ValueError:
                    pass

                # Try exact match
                response_lower = response.lower()
                for option in options:
                    if option.lower() == response_lower:
                        return option

                # Try partial match
                for option in options:
                    if response_lower in option.lower():
                        return option

            return None

        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(f"Choice timeout for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting user choice: {e}")
            return None
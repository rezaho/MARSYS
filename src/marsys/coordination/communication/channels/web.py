"""
Web-based communication channel for asynchronous user interaction.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

from ..core import AsyncChannel, UserInteraction

logger = logging.getLogger(__name__)


class WebChannel(AsyncChannel):
    """
    Asynchronous channel for web-based communication.
    
    This channel supports non-blocking interaction suitable for web APIs,
    using pub/sub patterns or queues for real-time communication.
    """
    
    def __init__(self, channel_id: str = "web"):
        super().__init__(channel_id)
        
        # Queues for web API polling
        self.interaction_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        
        # Pending interactions accessible via API
        self.pending_interactions: Dict[str, UserInteraction] = {}
        
        # Response callbacks
        self.response_callbacks: List[Callable] = []
        
        # WebSocket connections (if using WebSocket)
        self.websocket_connections: Dict[str, Any] = {}
        
        # Topics for pub/sub
        self.interaction_topic = f"web_channel_{channel_id}_interactions"
        self.response_topic = f"web_channel_{channel_id}_responses"
        
    async def start(self) -> None:
        """Start the web channel."""
        self.active = True
        logger.info(f"Web channel '{self.channel_id}' started")
        
    async def stop(self) -> None:
        """Stop the web channel."""
        self.active = False
        
        # Clear pending interactions
        self.pending_interactions.clear()
        
        # Close WebSocket connections
        for conn_id, conn in self.websocket_connections.items():
            try:
                await conn.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket {conn_id}: {e}")
        
        logger.info(f"Web channel '{self.channel_id}' stopped")
    
    async def is_available(self) -> bool:
        """Check if web channel is available."""
        return self.active
    
    async def publish_interaction(self, interaction: UserInteraction) -> None:
        """
        Publish interaction for web clients to consume.
        
        This makes the interaction available through various mechanisms:
        - Direct API polling
        - WebSocket push
        - Message queue
        """
        if not self.active:
            raise RuntimeError("Web channel is not active")
        
        # Store in pending interactions
        self.pending_interactions[interaction.interaction_id] = interaction
        
        # Put in queue for polling
        await self.interaction_queue.put(interaction.to_display_dict())
        
        # Push to WebSocket connections
        await self._push_to_websockets(interaction)
        
        # Log the interaction
        logger.info(f"Published interaction {interaction.interaction_id} to web channel")
        
    def subscribe_responses(self, callback: Callable[[str, Any], None]) -> None:
        """
        Subscribe to user responses.
        
        The callback will be invoked with (interaction_id, response) when
        a response is received.
        """
        self.response_callbacks.append(callback)
        logger.debug(f"Added response callback to web channel")
    
    async def unsubscribe_responses(self) -> None:
        """Unsubscribe all response callbacks."""
        self.response_callbacks.clear()
        logger.debug("Cleared all response callbacks")
    
    async def get_pending_interactions(
        self,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get pending interactions for API polling.
        
        This method is designed to be called by REST API endpoints.
        """
        interactions = []
        
        for interaction in self.pending_interactions.values():
            if session_id and interaction.session_id != session_id:
                continue
                
            interactions.append(interaction.to_display_dict())
            
            if len(interactions) >= limit:
                break
        
        return interactions
    
    async def submit_response(
        self,
        interaction_id: str,
        response: Any
    ) -> bool:
        """
        Submit a response from web client.
        
        Returns:
            True if response was accepted, False if interaction not found
        """
        # Check if interaction exists
        if interaction_id not in self.pending_interactions:
            logger.warning(f"Response for unknown interaction: {interaction_id}")
            return False
        
        interaction = self.pending_interactions[interaction_id]
        
        # Remove from pending
        del self.pending_interactions[interaction_id]
        
        # Call all response callbacks
        for callback in self.response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(interaction_id, response)
                else:
                    # Run sync callback in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, callback, interaction_id, response)
            except Exception as e:
                logger.error(f"Error in response callback: {e}")
        
        # Put in response queue
        await self.response_queue.put({
            "interaction_id": interaction_id,
            "response": response,
            "session_id": interaction.session_id
        })
        
        logger.info(f"Received response for interaction {interaction_id}")
        return True
    
    async def wait_for_response(
        self,
        interaction_id: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for a specific response (used internally).
        
        This can be used to convert async to sync-like behavior if needed.
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check timeout
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Timeout waiting for response to {interaction_id}")
            
            # Check response queue
            try:
                response_data = await asyncio.wait_for(
                    self.response_queue.get(),
                    timeout=1.0  # Check every second
                )
                
                if response_data["interaction_id"] == interaction_id:
                    return response_data["response"]
                else:
                    # Not our response, put it back
                    await self.response_queue.put(response_data)
                    
            except asyncio.TimeoutError:
                continue
    
    # WebSocket support methods
    
    async def register_websocket(self, connection_id: str, websocket: Any) -> None:
        """Register a WebSocket connection."""
        self.websocket_connections[connection_id] = websocket
        logger.info(f"Registered WebSocket connection: {connection_id}")
    
    async def unregister_websocket(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]
            logger.info(f"Unregistered WebSocket connection: {connection_id}")
    
    async def _push_to_websockets(self, interaction: UserInteraction) -> None:
        """Push interaction to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message = json.dumps({
            "type": "interaction",
            "data": interaction.to_display_dict()
        })
        
        # Send to all connections
        disconnected = []
        for conn_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket {conn_id}: {e}")
                disconnected.append(conn_id)
        
        # Remove disconnected connections
        for conn_id in disconnected:
            await self.unregister_websocket(conn_id)
    
    # API helper methods for web frameworks
    
    def create_api_handler(self) -> 'WebChannelAPIHandler':
        """Create an API handler for this channel."""
        return WebChannelAPIHandler(self)


class WebChannelAPIHandler:
    """
    Helper class for integrating WebChannel with web frameworks.
    
    Provides methods that can be directly used in FastAPI/Flask endpoints.
    """
    
    def __init__(self, channel: WebChannel):
        self.channel = channel
    
    async def get_pending(
        self,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """GET /api/interactions/pending"""
        interactions = await self.channel.get_pending_interactions(session_id, limit)
        return {
            "interactions": interactions,
            "count": len(interactions)
        }
    
    async def submit_response(
        self,
        interaction_id: str,
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/interactions/{interaction_id}/response"""
        success = await self.channel.submit_response(
            interaction_id,
            response_data.get("response")
        )
        
        return {
            "success": success,
            "interaction_id": interaction_id
        }
    
    async def handle_websocket(self, websocket: Any, session_id: str) -> None:
        """WebSocket handler for real-time communication."""
        conn_id = str(uuid.uuid4())
        
        try:
            # Register WebSocket
            await self.channel.register_websocket(conn_id, websocket)
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get("type") == "response":
                        await self.submit_response(
                            data["interaction_id"],
                            {"response": data["response"]}
                        )
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                    
        finally:
            # Unregister on disconnect
            await self.channel.unregister_websocket(conn_id)
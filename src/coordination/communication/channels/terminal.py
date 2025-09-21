"""
Terminal-based communication channel for synchronous user interaction.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from ..core import SyncChannel, UserInteraction

logger = logging.getLogger(__name__)


class TerminalChannel(SyncChannel):
    """
    Synchronous terminal channel for command-line interaction.
    
    This channel displays interactions in the terminal and waits for
    user input, providing a blocking interface suitable for CLI applications.
    """
    
    def __init__(self, channel_id: str = "terminal"):
        super().__init__(channel_id)
        self.current_interaction: Optional[UserInteraction] = None
        self._interaction_lock = asyncio.Lock()  # Lock for entire interaction flow
        self._pending_interactions = asyncio.Queue()  # Queue for pending interactions
        
    async def start(self) -> None:
        """Start the terminal channel."""
        self.active = True
        logger.info(f"Terminal channel '{self.channel_id}' started")
        
    async def stop(self) -> None:
        """Stop the terminal channel."""
        self.active = False
        logger.info(f"Terminal channel '{self.channel_id}' stopped")
        
    async def is_available(self) -> bool:
        """Check if terminal is available."""
        return self.active and not self._interaction_lock.locked()
    
    async def send_interaction(self, interaction: UserInteraction) -> None:
        """
        Display interaction in terminal.
        
        Note: The lock is NOT acquired here - it should be acquired by the caller
        to ensure the entire interaction flow is atomic.
        """
        self.current_interaction = interaction
        
        # Format and display the interaction
        print("\n" + "=" * 80)
        
        # Special handling for System interactions (NEW)
        if interaction.calling_agent == "System":
            if interaction.interaction_type == "question":
                print("📝 USER INPUT REQUIRED")
            elif interaction.interaction_type == "task":
                print("🚀 NEW TASK")
            elif interaction.interaction_type == "notification":
                print("📢 SYSTEM NOTIFICATION")
            elif interaction.interaction_type == "input":
                print("💬 USER INPUT")
            else:
                print(f"💬 SYSTEM {interaction.interaction_type.upper()}")
        else:
            # Existing agent interaction headers
            if interaction.interaction_type == "question":
                print("🤔 QUESTION FROM AGENT")
            elif interaction.interaction_type == "choice":
                print("📋 PLEASE CHOOSE AN OPTION")
            elif interaction.interaction_type == "confirmation":
                print("✅ CONFIRMATION REQUIRED")
            elif interaction.interaction_type == "notification":
                print("📢 NOTIFICATION")
            else:
                print(f"💬 {interaction.interaction_type.upper()}")
        
        print("=" * 80)
        
        # Show calling agent
        if interaction.calling_agent:
            print(f"From: {interaction.calling_agent}")
            print("-" * 40)
        
        # Display the message
        message = interaction.incoming_message
        if isinstance(message, dict):
            # Structured message
            if "content" in message:
                print(f"\n{message['content']}\n")
            
            # Show context if provided
            if "context" in message and message["context"]:
                print("Context:")
                print("-" * 20)
                context = message["context"]
                if isinstance(context, dict):
                    for key, value in context.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"  {context}")
                print()
            
            # Show options if this is a choice
            if "options" in message and message["options"]:
                print("Options:")
                for i, option in enumerate(message["options"], 1):
                    print(f"  {i}. {option}")
                print()
        else:
            # Simple string message
            print(f"\n{message}\n")
        
        # Show additional metadata if in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            print("Debug Info:")
            print(f"  Interaction ID: {interaction.interaction_id}")
            print(f"  Session ID: {interaction.session_id}")
            print(f"  Will resume at: {interaction.resume_agent}")
            print()
        
        print("=" * 80)
    
    async def get_response(self, interaction_id: str) -> Tuple[str, Any]:
        """
        Get user input from terminal (blocking).
        
        Note: The lock should already be held by send_and_wait_for_response
        
        Returns:
            Tuple of (interaction_id, user_response)
        """
        if not self.current_interaction or self.current_interaction.interaction_id != interaction_id:
            raise ValueError(f"No current interaction matching {interaction_id}")
        
        try:
            # Handle different interaction types
            if self.current_interaction.interaction_type == "notification":
                    # No response needed for notifications
                    print("Press Enter to continue...")
                    await self._async_input("")
                    return (interaction_id, {"acknowledged": True})
                
            elif self.current_interaction.interaction_type == "choice":
                # Handle multiple choice
                options = self._get_options_from_message(self.current_interaction.incoming_message)
                if options:
                    return await self._get_choice_response(interaction_id, options)
                
            elif self.current_interaction.interaction_type == "confirmation":
                # Handle yes/no confirmation
                return await self._get_confirmation_response(interaction_id)
                
            # Default: free-form text input
            response = await self._get_text_response()
            return (interaction_id, response)
                
        except KeyboardInterrupt:
            print("\n\n❌ Input cancelled by user")
            raise
        finally:
            self.current_interaction = None
    
    async def _get_text_response(self) -> str:
        """Get free-form text response."""
        while True:
            response = await self._async_input("💬 Your response: ")
            response = response.strip()
            
            if response:
                return response
            
            print("❌ Please provide a response to continue.")
    
    async def _get_choice_response(self, interaction_id: str, options: list) -> Tuple[str, Any]:
        """Get choice selection from user."""
        while True:
            choice_str = await self._async_input(f"📝 Enter your choice (1-{len(options)}): ")
            choice_str = choice_str.strip()
            
            try:
                choice = int(choice_str)
                if 1 <= choice <= len(options):
                    selected = options[choice - 1]
                    return (interaction_id, {
                        "choice_index": choice - 1,
                        "choice_value": selected
                    })
                else:
                    print(f"❌ Please enter a number between 1 and {len(options)}")
            except ValueError:
                # Check if user typed the option text
                choice_lower = choice_str.lower()
                for i, option in enumerate(options):
                    if option.lower() == choice_lower:
                        return (interaction_id, {
                            "choice_index": i,
                            "choice_value": option
                        })
                
                print(f"❌ Invalid choice. Please enter a number or type one of the options.")
    
    async def _get_confirmation_response(self, interaction_id: str) -> Tuple[str, Any]:
        """Get yes/no confirmation."""
        while True:
            response = await self._async_input("✅ Please confirm (yes/no): ")
            response = response.strip().lower()
            
            if response in ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay']:
                return (interaction_id, {"confirmed": True})
            elif response in ['no', 'n', 'nope', 'nah', 'cancel']:
                return (interaction_id, {"confirmed": False})
            else:
                print("❌ Please answer 'yes' or 'no'")
    
    async def _async_input(self, prompt: str) -> str:
        """Async wrapper for input() to avoid blocking event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input, prompt)
    
    def _get_options_from_message(self, message: Any) -> Optional[list]:
        """Extract options from message if present."""
        if isinstance(message, dict) and "options" in message:
            return message["options"]
        return None
    
    async def send_and_wait_for_response(self, interaction: UserInteraction) -> Tuple[str, Any]:
        """
        Send interaction and wait for response atomically.
        
        This ensures that no other interaction can interfere while waiting for user input.
        """
        logger.debug(f"Terminal channel send_and_wait_for_response called for interaction {interaction.interaction_id}")
        logger.debug(f"Channel active: {self.active}, interaction type: {interaction.interaction_type}")

        if not self.active:
            logger.warning(f"Terminal channel {self.channel_id} is not active, starting it now")
            await self.start()

        async with self._interaction_lock:
            await self.send_interaction(interaction)
            result = await self.get_response(interaction.interaction_id)
            logger.debug(f"Got response for interaction {interaction.interaction_id}: {result}")
            return result
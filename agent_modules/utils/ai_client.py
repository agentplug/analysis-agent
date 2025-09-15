#!/usr/bin/env python3
"""
AI Client Wrapper Module
Handles AI client initialization and response generation.
"""

from typing import List, Dict


class AIClientWrapper:
    """Wrapper for AI client to handle different providers and configurations."""
    
    def __init__(self, model: str = "openai:gpt-4o", temperature: float = 0.0):
        """
        Initialize AI client wrapper.
        
        Args:
            model: Model identifier (e.g., "openai:gpt-4o")
            temperature: Temperature for response generation
        """
        self.model = model
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        """Get or create AI client instance."""
        if self._client is None:
            try:
                import aisuite as ai
                from dotenv import load_dotenv
                load_dotenv()
                self._client = ai.Client()
            except ImportError:
                raise Exception("aisuite not available. Please install required dependencies.")
        return self._client
    
    def generate_response(self, system_prompt: str, user_message: str) -> str:
        """
        Generate AI response using system and user messages.
        
        Args:
            system_prompt: System prompt for AI
            user_message: User message content
            
        Returns:
            AI response text
        """
        client = self._get_client()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content
    
    def generate_response_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate AI response using a list of messages.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            
        Returns:
            AI response text
        """
        client = self._get_client()
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        
        return response.choices[0].message.content

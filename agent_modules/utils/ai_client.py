#!/usr/bin/env python3
"""
AI Client Wrapper Module
Handles AI client initialization and response generation.
"""

from typing import List, Dict, Optional
from .config_loader import get_ai_config


class AIClientWrapper:
    """Wrapper for AI client to handle different providers and configurations."""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None, **kwargs):
        """
        Initialize AI client wrapper.
        
        Args:
            model: Model identifier (e.g., "openai:gpt-4o-mini"). If None, uses config.
            temperature: Temperature for response generation. If None, uses config.
            **kwargs: Additional parameters to override config values
        """
        # Load AI configuration
        ai_config = get_ai_config()
        
        # Use provided values or fall back to config
        self.model = model or ai_config.get('model', 'openai:gpt-4o-mini')
        self.temperature = temperature if temperature is not None else ai_config.get('temperature', 0.0)
        
        # Additional AI parameters from config
        self.max_tokens = kwargs.get('max_tokens', ai_config.get('max_tokens'))
        self.top_p = kwargs.get('top_p', ai_config.get('top_p', 1.0))
        self.frequency_penalty = kwargs.get('frequency_penalty', ai_config.get('frequency_penalty', 0.0))
        self.presence_penalty = kwargs.get('presence_penalty', ai_config.get('presence_penalty', 0.0))
        
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
        
        # Prepare parameters for the API call
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add optional parameters if they are set
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p != 1.0:
            params["top_p"] = self.top_p
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        response = client.chat.completions.create(**params)
        
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
        
        # Prepare parameters for the API call
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add optional parameters if they are set
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p != 1.0:
            params["top_p"] = self.top_p
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        response = client.chat.completions.create(**params)
        
        return response.choices[0].message.content

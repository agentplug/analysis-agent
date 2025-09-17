#!/usr/bin/env python3
"""
Configuration Loader Module
Handles loading and managing configuration from config.json file.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Configuration loader for managing application settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config.json file. If None, looks for config.json in project root.
        """
        if config_path is None:
            # Find project root by looking for config.json
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config.json"
        
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None

    def _load_model_name(self) -> str:
        """
        Automatically detect and return the best available model based on API keys.
        Follows aisuite provider format: <provider>:<model-name>

        Returns:
            str: Model identifier in aisuite format
        """
        # Priority order: Check API keys and return corresponding model
        # OpenAI models
        if os.getenv("OPENAI_API_KEY"):
            return "openai:gpt-4o-mini"

        # Anthropic models
        elif os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic:claude-3-5-sonnet-20241022"

        # Google models
        elif os.getenv("GOOGLE_API_KEY"):
            return "google:gemini-1.5-pro"

        # DeepSeek models
        elif os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek:deepseek-chat"

        # Fireworks models
        elif os.getenv("FIREWORKS_API_KEY"):
            return "fireworks:accounts/fireworks/models/llama-v3p2-3b-instruct"

        # Cohere models
        elif os.getenv("COHERE_API_KEY"):
            return "cohere:command-r-plus"

        # Mistral models
        elif os.getenv("MISTRAL_API_KEY"):
            return "mistral:mistral-large-latest"

        # Groq models
        elif os.getenv("GROQ_API_KEY"):
            return "groq:llama-3.1-70b-versatile"

        # Replicate models
        elif os.getenv("REPLICATE_API_TOKEN"):
            return "replicate:meta/llama-2-70b-chat"

        # Hugging Face models
        elif os.getenv("HUGGINGFACE_API_KEY"):
            return "huggingface:microsoft/DialoGPT-large"

        # AWS Bedrock models
        elif os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            return "aws:anthropic.claude-3-5-sonnet-20241022-v2:0"

        # Azure OpenAI models
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            return "azure:gpt-4o"

        # Default fallback
        else:
            return "openai:gpt-4o-mini"
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from config.json file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config.json file is not found
            json.JSONDecodeError: If config.json contains invalid JSON
        """
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            self._config["ai"]["model"] = self._load_model_name()
        return self._config
    
    def get_ai_config(self) -> Dict[str, Any]:
        """
        Get AI-related configuration.
        
        Returns:
            AI configuration dictionary
        """
        config = self.load_config()
        return config.get('ai', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis-related configuration.
        
        Returns:
            Analysis configuration dictionary
        """
        config = self.load_config()
        return config.get('analysis', {})
    
    def get_execution_config(self) -> Dict[str, Any]:
        """
        Get execution-related configuration.
        
        Returns:
            Execution configuration dictionary
        """
        config = self.load_config()
        return config.get('execution', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging-related configuration.
        
        Returns:
            Logging configuration dictionary
        """
        config = self.load_config()
        return config.get('logging', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'ai.model' or 'analysis.max_summary_length')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config()
        
        # Support dot notation for nested keys
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = None
        self.load_config()


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_ai_config() -> Dict[str, Any]:
    """
    Get AI configuration.
    
    Returns:
        AI configuration dictionary
    """
    return get_config_loader().get_ai_config()


def get_analysis_config() -> Dict[str, Any]:
    """
    Get analysis configuration.
    
    Returns:
        Analysis configuration dictionary
    """
    return get_config_loader().get_analysis_config()


def get_execution_config() -> Dict[str, Any]:
    """
    Get execution configuration.
    
    Returns:
        Execution configuration dictionary
    """
    return get_config_loader().get_execution_config()


def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration.
    
    Returns:
        Logging configuration dictionary
    """
    return get_config_loader().get_logging_config()

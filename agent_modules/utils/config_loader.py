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
        Get model name from config or use default.
        Model detection is now handled by the enhanced AI client.

        Returns:
            str: Model identifier in aisuite format
        """
        # Return the model from config or use default
        # The AI client will handle auto-detection if needed
        return "openai:gpt-4o-mini"  # Default fallback
    
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

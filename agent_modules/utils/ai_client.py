#!/usr/bin/env python3
"""
Enhanced AI Client Wrapper Module
Handles AI client initialization and response generation with advanced features.
Inspired by agenthub core LLM service with automatic model detection and local model support.
"""

import logging
import os
import socket
import json as _json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from .config_loader import get_ai_config

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a detected model"""
    name: str
    provider: str
    score: int
    is_local: bool
    is_available: bool


class ModelConfig:
    """Configuration constants for model selection and scoring."""
    
    # Preferred models for different use cases
    PREFERRED_MODELS = [
        "gpt-oss:120b",
        "gpt-oss:20b",  # OpenAI open-weight (highest priority)
        "deepseek-r1:70b",
        "deepseek-r1:32b",  # DeepSeek reasoning models
        "gemma:latest",
        "llama3:latest",  # General purpose models
        "qwen:latest",
        "mistral:latest",  # Alternative models
    ]
    
    # Model family scoring (higher = better for agentic tasks)
    FAMILY_SCORES = {
        "qwen": 60,      # Highest priority
        "deepseek": 60,  # Equal highest priority
        "gpt-oss": 50,   # High priority
        "llama": 40,
        "gemma": 35,
        "mistral": 35,
        "codellama": 30,
        "phind": 25,
        "wizard": 20,
        "vicuna": 15,
        "claude": 45,
        "gpt": 40,
    }
    
    # Size scoring (larger is generally better)
    SIZE_SCORES = {
        "120b": 60,
        "70b": 60,
        "65b": 55,
        "32b": 50,
        "20b": 45,
        "13b": 45,
        "8b": 40,
        "latest": 40,  # Latest type equals 8B
        "7b": 35,
        "4b": 35,      # Added 4B support
        "3b": 35,
        "1b": 30,
    }
    
    # Common Ollama URLs for auto-detection
    OLLAMA_URLS = [
        "http://localhost:11434",  # Default Ollama
        "http://127.0.0.1:11434",  # Alternative localhost
        "http://0.0.0.0:11434",  # All interfaces
    ]
    
    # Cloud provider models (fallback when no local models)
    CLOUD_MODELS = {
        "OPENAI_API_KEY": "openai:gpt-4o",
        "ANTHROPIC_API_KEY": "anthropic:claude-3-5-sonnet-20241022",
        "GOOGLE_API_KEY": "google:gemini-1.5-pro",
        "DEEPSEEK_API_KEY": "deepseek:deepseek-chat",
        "FIREWORKS_API_KEY": "fireworks:accounts/fireworks/models/llama-v3p2-3b-instruct",
        "COHERE_API_KEY": "cohere:command-r-plus",
        "MISTRAL_API_KEY": "mistral:mistral-large-latest",
        "GROQ_API_KEY": "groq:llama-3.1-70b-versatile",
        "REPLICATE_API_TOKEN": "replicate:meta/llama-2-70b-chat",
        "HUGGINGFACE_API_KEY": "huggingface:microsoft/DialoGPT-large",
        "AZURE_OPENAI_API_KEY": "azure:gpt-4o",
    }
    
    # Special case for AWS (requires multiple env vars)
    AWS_MODEL = "aws:anthropic.claude-3-5-sonnet-20241022-v2:0"
    AWS_REQUIRED_VARS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]


class AIClientWrapper:
    """Enhanced wrapper for AI client with automatic model detection and local model support."""
    
    def __init__(self, model: Optional[str] = None, temperature: Optional[float] = None, auto_detect: bool = True, **kwargs):
        """
        Initialize enhanced AI client wrapper.
        
        Args:
            model: Model identifier (e.g., "openai:gpt-4o-mini"). If None and auto_detect=True, auto-detects best available model.
            temperature: Temperature for response generation. If None, uses config.
            auto_detect: Whether to auto-detect model if not provided
            **kwargs: Additional parameters to override config values
        """
        # Initialize without caching for fresh detection
        self._model_info: Optional[ModelInfo] = None
        
        # Load AI configuration
        ai_config = get_ai_config()
        
        # Model selection with auto-detection
        if model:
            self.model = model
            logger.info(f"🎯 Using specified model: {model}")
        elif auto_detect:
            self.model = self._detect_best_model()
        else:
            self.model = ai_config.get('model', 'openai:gpt-4o-mini')
            logger.warning(f"⚠️ No model specified, using default: {self.model}")
        
        # Use provided values or fall back to config
        self.temperature = temperature if temperature is not None else ai_config.get('temperature', 0.0)
        
        # Additional AI parameters from config
        self.max_tokens = kwargs.get('max_tokens', ai_config.get('max_tokens'))
        self.top_p = kwargs.get('top_p', ai_config.get('top_p', 1.0))
        self.frequency_penalty = kwargs.get('frequency_penalty', ai_config.get('frequency_penalty', 0.0))
        self.presence_penalty = kwargs.get('presence_penalty', ai_config.get('presence_penalty', 0.0))
        
        self._client = None
    
    def _detect_best_model(self) -> str:
        """
        Automatically detect and return the best available model.
        Follows aisuite provider format: <provider>:<model-name>

        Returns:
            str: Model identifier in aisuite format
        """
        # Priority 1: Check for local models first (auto-detection)
        local_model = self._detect_running_local_model()
        if local_model:
            logger.info(f"🎯 Selected model: {local_model}")
            return local_model

        # Priority 2: Check API keys and return corresponding cloud model
        cloud_model = self._detect_cloud_model()
        if cloud_model:
            logger.info(f"☁️ Selected cloud model: {cloud_model}")
            return cloud_model

        # Default fallback
        default_model = "openai:gpt-4o-mini"
        logger.warning(f"⚠️ No models detected, using default: {default_model}")
        return default_model

    def _detect_cloud_model(self) -> Optional[str]:
        """Detect available cloud model based on API keys."""
        # Check AWS Bedrock (special case - requires multiple env vars)
        if all(os.getenv(var) for var in ModelConfig.AWS_REQUIRED_VARS):
            return ModelConfig.AWS_MODEL

        # Check other cloud providers
        for env_var, model in ModelConfig.CLOUD_MODELS.items():
            if os.getenv(env_var):
                return model

        return None

    def _detect_running_local_model(self) -> Optional[str]:
        """Detect running local models with auto-detection (preferring Ollama)."""
        # Try Ollama first (preferred)
        ollama_model = self._detect_ollama_model()
        if ollama_model:
            return ollama_model
        
        # Fallback to LM Studio if Ollama not available
        lmstudio_model = self._detect_lmstudio_model()
        if lmstudio_model:
            return lmstudio_model

        logger.debug("No local models detected")
        return None

    def _detect_ollama_model(self) -> Optional[str]:
        """Detect Ollama models."""
        ollama_url = self._detect_ollama_url()

        logger.debug(f"Checking Ollama availability at {ollama_url}")
        if self._check_ollama_available(ollama_url):
            # Get available models
            models = self._get_ollama_models(ollama_url)
            logger.debug(f"Ollama returned {len(models)} models")
            if models:
                best_model = self._select_best_ollama_model(models)
                selected_model = f"ollama:{best_model}"
                logger.info(
                    f"🤖 Ollama model detected: {selected_model} "
                    f"(from {len(models)} available models)"
                )
                return selected_model
            else:
                logger.warning("Ollama is available but returned no models from /api/tags")
        else:
            logger.debug("Ollama not reachable or not responding with status 200")
        return None

    def _detect_lmstudio_model(self) -> Optional[str]:
        """Detect LM Studio models as fallback."""
        lmstudio_urls = [
            "http://localhost:1234",  # LM Studio default port
            "http://localhost:8080",  # Alternative port
        ]
        
        for url in lmstudio_urls:
            if self._check_lmstudio_available(url):
                models = self._get_lmstudio_models(url)
                if models:
                    best_model = self._select_best_model(models, is_lmstudio=True)
                    selected_model = f"lmstudio:{best_model}"
                    logger.info(
                        f"🤖 LM Studio model detected: {selected_model} "
                        f"(from {len(models)} available models)"
                    )
                    return selected_model
        return None

    def _detect_ollama_url(self) -> str:
        """Auto-detect Ollama API URL with fallback options (no caching)."""
        # 1. Environment variable (user override)
        if os.getenv("OLLAMA_API_URL"):
            url = os.getenv("OLLAMA_API_URL")
            logger.info(f"🔧 Using Ollama URL from environment: {url}")
            return url

        # 2. Try to find running Ollama instance (HTTP check then port check)
        for url in ModelConfig.OLLAMA_URLS:
            if self._check_ollama_available(url):
                logger.info(f"🔍 Auto-detected Ollama URL (HTTP): {url}")
                return url
            # Fallback: if HTTP failed, check if port is open
            host = url.split("://", 1)[-1].split(":")[0]
            if self._is_port_open(host, 11434):
                logger.info(f"🔍 Ollama port open, using URL: {url}")
                return url

        # 3. Default fallback
        logger.debug("Using default Ollama URL: http://localhost:11434")
        return "http://localhost:11434"

    def _check_ollama_available(self, url: str) -> bool:
        """Check if Ollama is running at the given URL."""
        try:
            import requests
            response = requests.get(
                f"{url}/api/tags",
                timeout=3,
                proxies={"http": None, "https": None},  # bypass env proxies for localhost
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"HTTP check to {url}/api/tags failed: {e}")
            # Fallback to urllib (no third-party deps)
            try:
                import urllib.request
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(f"{url}/api/tags", timeout=3) as resp:
                    return getattr(resp, "status", 200) == 200
            except Exception as e2:
                logger.debug(f"urllib check to {url}/api/tags failed: {e2}")
            # Fallback to port open check
            host = url.split("://", 1)[-1].split(":")[0]
            return self._is_port_open(host, 11434)

    def _get_ollama_models(self, url: str) -> List[Dict]:
        """Get available models from Ollama."""
        # Try requests first
        try:
            import requests
            response = requests.get(
                f"{url}/api/tags",
                timeout=4,
                headers={"Accept": "application/json"},
                proxies={"http": None, "https": None},  # bypass env proxies for localhost
            )
            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception as parse_err:
                    logger.debug(
                        f"Failed to parse Ollama /api/tags JSON (len={len(response.text)}): {parse_err}",
                        exc_info=False,
                    )
                    data = None
                if data is not None:
                    if isinstance(data, dict) and "models" in data:
                        return data.get("models", [])
                    if isinstance(data, list):
                        return data
                    logger.debug(
                        f"Unexpected Ollama /api/tags schema: type={type(data).__name__}; keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}"
                    )
        except Exception as e:
            logger.debug(f"Failed to fetch Ollama models from {url}/api/tags via requests: {e}", exc_info=False)

        # Fallback to urllib (no third-party deps)
        try:
            import urllib.request
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{url}/api/tags", timeout=4) as resp:
                status = getattr(resp, "status", 200)
                text = resp.read().decode("utf-8", errors="ignore") if status == 200 else ""
            if status == 200 and text:
                try:
                    data = _json.loads(text)
                except Exception as parse_err:
                    logger.debug(
                        f"Failed to parse Ollama /api/tags JSON via urllib (len={len(text)}): {parse_err}",
                        exc_info=False,
                    )
                    return []
                if isinstance(data, dict) and "models" in data:
                    return data.get("models", [])
                if isinstance(data, list):
                    return data
                logger.debug(
                    f"Unexpected Ollama /api/tags schema via urllib: type={type(data).__name__}; keys={list(data.keys()) if isinstance(data, dict) else 'n/a'}"
                )
        except Exception as e:
            logger.debug(f"Failed to fetch Ollama models from {url}/api/tags via urllib: {e}", exc_info=False)

        return []

    def _is_port_open(self, host: str, port: int, timeout: float = 0.5) -> bool:
        """Lightweight TCP check to see if a host:port is accepting connections."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _check_lmstudio_available(self, url: str) -> bool:
        """Check if LM Studio is running at the given URL."""
        try:
            import requests

            # If URL already has /v1, use it directly, otherwise add it
            if url.endswith("/v1"):
                models_url = f"{url}/models"
            else:
                models_url = f"{url}/v1/models"
            
            response = requests.get(models_url, timeout=1)
            return response.status_code == 200
        except Exception:
            return False

    def _get_lmstudio_models(self, url: str) -> List[str]:
        """Get available models from LM Studio."""
        try:
            import requests

            # If URL already has /v1, use it directly, otherwise add it
            if url.endswith("/v1"):
                models_url = f"{url}/models"
            else:
                models_url = f"{url}/v1/models"
            
            response = requests.get(models_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
        except Exception:
            pass
        return []

    def _select_best_ollama_model(self, available_models: List[Dict]) -> str:
        """Select the best model from available Ollama models."""
        model_names = [model.get("name", "") for model in available_models]

        if not model_names:
            logger.warning("No Ollama models available, using fallback: llama3:latest")
            return "llama3:latest"

        # If only one model, return it
        if len(model_names) == 1:
            model_name = model_names[0]
            logger.info(f"🎯 Single model available: {model_name}")
            return model_name

        # Score each model and select the best one
        logger.info(
            f"🔍 Evaluating {len(model_names)} models: {', '.join(model_names)}"
        )

        # Log scoring details
        for model_name in model_names:
            score = self._calculate_model_score(model_name)
            logger.debug(f"📊 {model_name}: {score} points")

        best_model = self._score_and_select_best(model_names)
        logger.info(f"🏆 Best model selected: {best_model}")
        return best_model

    def _select_best_model(self, available_models: List[str], is_lmstudio: bool = False) -> str:
        """Select the best model from available models (unified for Ollama and LM Studio)."""
        if not available_models:
            fallback = "llama3:latest" if not is_lmstudio else "meta-llama/llama-3.2-1b-instruct"
            logger.warning(f"No models available, using fallback: {fallback}")
            return fallback

        # If only one model, return it
        if len(available_models) == 1:
            model_name = available_models[0]
            logger.info(f"🎯 Single model available: {model_name}")
            return model_name

        # Score each model and select the best one
        logger.info(
            f"🔍 Evaluating {len(available_models)} models: {', '.join(available_models)}"
        )

        # Log scoring details
        for model_name in available_models:
            score = self._calculate_model_score(model_name)
            logger.debug(f"📊 {model_name}: {score} points")

        best_model = self._score_and_select_best(available_models)
        logger.info(f"🏆 Best model selected: {best_model}")
        return best_model

    def _score_and_select_best(self, model_names: List[str]) -> str:
        """Score models and return the best one."""
        scored_models = []

        for model_name in model_names:
            score = self._calculate_model_score(model_name)
            scored_models.append((model_name, score))

        # Sort by score (highest first) and return the best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]

    def _calculate_model_score(self, model_name: str) -> int:
        """Calculate a unified score for any model (higher is better)."""
        score = 0
        model_lower = model_name.lower()

        # Size scoring (larger is better)
        for size, points in ModelConfig.SIZE_SCORES.items():
            if size in model_lower:
                score += points
                break

        # Model family scoring (known good models)
        for family, points in ModelConfig.FAMILY_SCORES.items():
            if family in model_lower:
                score += points
                break

        # Platform bonus (Ollama preferred)
        if self._is_ollama_model(model_name):
            score += 5

        # Penalty for poor models
        poor_indicators = ["tiny", "small", "test", "demo"]
        for indicator in poor_indicators:
            if indicator in model_lower:
                score -= 30
                break

        # Bonus for instruction-tuned models
        if "instruct" in model_lower:
            score += 10

        # Bonus for chat models
        if "chat" in model_lower:
            score += 5

        return score

    def _is_ollama_model(self, model_name: str) -> bool:
        """Check if model is from Ollama platform."""
        return model_name.startswith("ollama:")

    def _get_client(self):
        """Get or create AI client instance with appropriate configuration."""
        if self._client is None:
            try:
                import aisuite as ai
                from dotenv import load_dotenv
                load_dotenv()
                
                # Initialize client with appropriate configuration
                if self.model.startswith("ollama:"):
                    self._client = self._initialize_ollama_client()
                elif self.model.startswith("lmstudio:"):
                    self._client = self._initialize_lmstudio_client()
                else:
                    # Cloud models - no special config needed
                    self._client = ai.Client()
                    
            except ImportError:
                raise Exception("aisuite not available. Please install required dependencies.")
        return self._client

    def _initialize_ollama_client(self):
        """Initialize AISuite client for Ollama."""
        import aisuite as ai
        
        # Get Ollama configuration
        api_url = self._detect_ollama_url()
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
        
        return ai.Client(
            provider_configs={
                "ollama": {
                    "api_url": api_url,
                    "timeout": timeout,
                }
            }
        )

    def _initialize_lmstudio_client(self):
        """Initialize AISuite client for LM Studio using OpenAI-compatible API."""
        import aisuite as ai
        
        # Get LM Studio configuration
        api_url = self._detect_lmstudio_url()
        timeout = int(os.getenv("LMSTUDIO_TIMEOUT", "300"))
        
        # LM Studio provides OpenAI-compatible API, so use openai provider with custom base_url
        return ai.Client(
            provider_configs={
                "openai": {
                    "base_url": api_url,
                    "timeout": timeout,
                }
            }
        )

    def _detect_lmstudio_url(self) -> str:
        """Detect LM Studio API URL."""
        # Check environment variable first
        if os.getenv("LMSTUDIO_API_URL"):
            return os.getenv("LMSTUDIO_API_URL")
        
        # Check common LM Studio ports
        lmstudio_urls = [
            "http://localhost:1234/v1",  # LM Studio default with /v1
            "http://localhost:8080/v1",  # Alternative port with /v1
        ]
        
        for url in lmstudio_urls:
            if self._check_lmstudio_available(url):
                return url
        
        # Fallback to default
        return "http://localhost:1234/v1"
    
    def get_model_info(self) -> ModelInfo:
        """Get detailed information about the current model."""
        if self._model_info is None:
            self._model_info = self._create_model_info()
        return self._model_info

    def _create_model_info(self) -> ModelInfo:
        """Create ModelInfo object for current model."""
        provider, model_name = (
            self.model.split(":", 1) if ":" in self.model else ("unknown", self.model)
        )
        is_local = provider in ["ollama", "lmstudio"]
        score = (
            self._calculate_model_score(model_name) if is_local else 100
        )  # Cloud models get default score

        return ModelInfo(
            name=model_name,
            provider=provider,
            score=score,
            is_local=is_local,
            is_available=True,
        )

    def list_available_models(self) -> List[ModelInfo]:
        """List all available models with their information."""
        models = []

        # Check Ollama models
        ollama_url = self._detect_ollama_url()
        if self._check_ollama_available(ollama_url):
            ollama_models = self._get_ollama_models(ollama_url)
            for model_data in ollama_models:
                model_name = model_data.get("name", "")
                score = self._calculate_model_score(model_name)
                models.append(
                    ModelInfo(
                        name=model_name,
                        provider="ollama",
                        score=score,
                        is_local=True,
                        is_available=True,
                    )
                )

        # Check LM Studio models
        lmstudio_urls = ["http://localhost:1234", "http://localhost:8080"]
        for url in lmstudio_urls:
            if self._check_lmstudio_available(url):
                lmstudio_models = self._get_lmstudio_models(url)
                for model_name in lmstudio_models:
                    score = self._calculate_model_score(model_name)
                    models.append(
                        ModelInfo(
                            name=model_name,
                            provider="lmstudio",
                            score=score,
                            is_local=True,
                            is_available=True,
                        )
                    )
                break  # Only check first available URL

        # Check cloud models
        for env_var, model in ModelConfig.CLOUD_MODELS.items():
            if os.getenv(env_var):
                provider, model_name = model.split(":", 1)
                models.append(
                    ModelInfo(
                        name=model_name,
                        provider=provider,
                        score=100,  # Cloud models get default score
                        is_local=False,
                        is_available=True,
                    )
                )

        # Sort by score (highest first)
        models.sort(key=lambda x: x.score, reverse=True)
        return models

    def get_current_model(self) -> str:
        """Get the currently selected model."""
        return self.model

    def is_local_model(self) -> bool:
        """Check if current model is local (Ollama or LM Studio)."""
        return self.model.startswith("ollama:") or self.model.startswith("lmstudio:")

    def _get_actual_model_name(self) -> str:
        """Get the actual model name for API calls (removes platform prefix)."""
        if self.model.startswith("ollama:"):
            # For Ollama, keep the full format for AISuite
            return self.model  # Keep "ollama:model" format
        elif self.model.startswith("lmstudio:"):
            # For LM Studio, use openai provider format
            model_name = self.model[9:]  # Remove "lmstudio:" prefix
            return f"openai:{model_name}"
        else:
            return self.model

    def generate_response(self, system_prompt: str, user_message: str, return_json: bool = False) -> str:
        """
        Generate AI response using system and user messages.
        
        Args:
            system_prompt: System prompt for AI
            user_message: User message content
            return_json: If True, request JSON response from AISuite
            
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
            "model": self._get_actual_model_name(),
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add JSON response format if requested
        if return_json:
            params["response_format"] = {"type": "json_object"}
        
        # Add optional parameters if they are set
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p != 1.0:
            params["top_p"] = self.top_p
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response()
    
    def generate_response_with_messages(self, messages: List[Dict[str, str]], return_json: bool = False) -> str:
        """
        Generate AI response using a list of messages.
        
        Args:
            messages: List of message dictionaries with "role" and "content"
            return_json: If True, request JSON response from AISuite
            
        Returns:
            AI response text
        """
        client = self._get_client()
        
        # Prepare parameters for the API call
        params = {
            "model": self._get_actual_model_name(),
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add JSON response format if requested
        if return_json:
            params["response_format"] = {"type": "json_object"}
        
        # Add optional parameters if they are set
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p != 1.0:
            params["top_p"] = self.top_p
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response()

    def analyze_text(self, text: str, prompt_template: str, system_prompt: Optional[str] = None, return_json: bool = False) -> str:
        """
        Analyze any text content using AISuite with custom prompt template
        
        Args:
            text: Text content to analyze
            prompt_template: Prompt template with {text} placeholder
            system_prompt: Optional system prompt for analysis
            return_json: If True, request JSON response
            
        Returns:
            Analysis result from LLM
        """
        if not text:
            return self._fallback_response()

        formatted_prompt = prompt_template.format(text=text)
        return self.generate_response(
            system_prompt or "You are a helpful AI assistant.", 
            formatted_prompt, 
            return_json=return_json
        )

    def get_raw_client(self):
        """
        Get the raw AISuite client for direct access.
        
        Returns:
            Raw AISuite client instance
        """
        return self._get_client()

    def _fallback_response(self) -> str:
        """
        Fallback response when AISuite is not available or fails
        
        Returns:
            Fallback response string
        """
        return "AI service temporarily unavailable. Please check your API keys and internet connection."


# =============================================================================
# SHARED INSTANCE MANAGEMENT
# =============================================================================

def get_shared_ai_client(model: Optional[str] = None, auto_detect: bool = True) -> AIClientWrapper:
    """
    Get a fresh AIClientWrapper instance with no caching.
    
    This function creates a new instance every time to ensure fresh model detection.
    
    Args:
        model: Model identifier in format "provider:model-name".
               If None and auto_detect=True, auto-detects best available model.
        auto_detect: Whether to auto-detect model if not provided
    
    Returns:
        Fresh AIClientWrapper instance
    """
    logger.debug("Creating fresh AIClientWrapper instance")
    return AIClientWrapper(model=model, auto_detect=auto_detect)


def reset_shared_ai_client() -> None:
    """
    No-op function for compatibility.
    
    Since we no longer cache instances, this function does nothing.
    """
    logger.debug("No caching - reset not needed")

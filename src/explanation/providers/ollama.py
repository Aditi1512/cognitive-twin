"""
Ollama local LLM provider.

Runs models locally for maximum privacy.
"""

import os
from typing import List, Dict, Optional
import asyncio
import aiohttp

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""
    
    DEFAULT_MODELS = [
        "llama3",
        "llama3.1",
        "mistral",
        "phi3",
        "gemma2",
    ]
    
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        max_tokens: int = 150,
        temperature: float = 0.7
    ):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name (must be pulled in Ollama).
            host: Ollama server URL.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        self.model = model
        self.host = host
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion asynchronously.
        
        Args:
            messages: Chat messages.
            
        Returns:
            Generated response text.
        """
        url = f"{self.host}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {}).get("content", "")
                    else:
                        return self._fallback_message()
        except Exception:
            return self._fallback_message()
    
    def generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion synchronously.
        
        Args:
            messages: Chat messages.
            
        Returns:
            Generated response text.
        """
        import requests
        
        url = f"{self.host}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                return self._fallback_message()
        except Exception:
            return self._fallback_message()
    
    def _fallback_message(self) -> str:
        """Return fallback message when generation fails."""
        return (
            "I'm having trouble generating a response. "
            "Make sure Ollama is running with: ollama serve"
        )
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return f"ollama/{self.model}"
    
    @property
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_local_models(self) -> List[str]:
        """List models available in local Ollama."""
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

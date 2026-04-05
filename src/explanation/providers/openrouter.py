"""
OpenRouter LLM provider.

Supports free and paid models through OpenRouter API.
"""

import os
from typing import List, Dict, Optional
import asyncio

from .base import LLMProvider


# Free models available on OpenRouter
FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-7b-it:free",
    "deepseek/deepseek-v3-base:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
]


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider with free model support."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.3-70b-instruct:free",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 150,
        temperature: float = 0.7
    ):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
            model: Model to use. Defaults to free Llama 3.1 8B.
            base_url: API base URL.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "dummy-key-for-free-models"
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client
    
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion asynchronously.
        
        Args:
            messages: Chat messages.
            
        Returns:
            Generated response text.
        """
        # Run sync method in executor for async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_sync, messages)
    
    def generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion synchronously.
        
        Args:
            messages: Chat messages.
            
        Returns:
            Generated response text.
        """
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_headers={
                    "HTTP-Referer": "https://cognitive-twin.app",
                    "X-Title": "Cognitive Digital Twin"
                }
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            # Return fallback message on error
            return f"I'm having trouble connecting right now. Your mental state looks stable. Error: {str(e)[:50]}"
    
    @property
    def model_name(self) -> str:
        """Get model identifier."""
        return self.model
    
    @property
    def is_available(self) -> bool:
        """Check if provider is configured."""
        return self.api_key is not None or ":free" in self.model
    
    @classmethod
    def list_free_models(cls) -> List[str]:
        """List available free models."""
        return FREE_MODELS.copy()

"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            
        Returns:
            Generated text response.
        """
        pass
    
    @abstractmethod
    def generate_sync(self, messages: List[Dict[str, str]]) -> str:
        """
        Synchronous generation.
        
        Args:
            messages: List of message dicts.
            
        Returns:
            Generated text response.
        """
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier."""
        pass
    
    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

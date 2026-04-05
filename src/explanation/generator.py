"""
Main explanation generator.

Coordinates prompt building and LLM calls.
"""

import time
from typing import Optional, List
from datetime import datetime

from ..models import MentalState, UserContext, Explanation
from ..config import GenAIConfig
from .prompts import PromptBuilder
from .providers.base import LLMProvider
from .providers.openrouter import OpenRouterProvider
from .providers.ollama import OllamaProvider


class ExplanationGenerator:
    """Generates natural language explanations using LLMs."""
    
    PROVIDERS = {
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
    }
    
    def __init__(self, config: Optional[GenAIConfig] = None):
        """
        Initialize explanation generator.
        
        Args:
            config: GenAI configuration.
        """
        if config is None:
            config = GenAIConfig()
        
        self.config = config
        self.prompt_builder = PromptBuilder()
        self._provider: Optional[LLMProvider] = None
        self._last_generation_time = 0.0
        
        # Initialize provider
        self._init_provider()
    
    def _init_provider(self) -> None:
        """Initialize the LLM provider based on config."""
        provider_name = self.config.provider.lower()
        
        if provider_name == "openrouter":
            self._provider = OpenRouterProvider(
                api_key=self.config.api_key,
                model=self.config.model,
                base_url=self.config.base_url,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        elif provider_name == "ollama":
            self._provider = OllamaProvider(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
        else:
            # Default to OpenRouter
            self._provider = OpenRouterProvider(
                model="meta-llama/llama-3.3-70b-instruct:free",
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
    
    def should_generate(self, state: MentalState) -> bool:
        """
        Check if we should generate an explanation now.
        
        Args:
            state: Current mental state.
            
        Returns:
            True if generation is warranted.
        """
        time_since_last = time.time() - self._last_generation_time
        
        # Always generate if there's a concern
        if state.get_primary_concern():
            return time_since_last >= 15  # Min 15s between alerts
        
        # Otherwise respect the configured interval
        return time_since_last >= self.config.explanation_interval_seconds
    
    async def generate_async(
        self,
        state: MentalState,
        context: UserContext
    ) -> Explanation:
        """
        Generate explanation asynchronously.
        
        Args:
            state: Current mental state.
            context: User context.
            
        Returns:
            Generated explanation.
        """
        start_time = time.time()
        
        # Build prompt
        messages = self.prompt_builder.build_messages(
            state, context,
            style=context.notification_style
        )
        
        # Generate response
        if self._provider:
            message = await self._provider.generate(messages)
            model_used = self._provider.model_name
        else:
            message = self._get_fallback_message(state)
            model_used = "fallback"
        
        generation_time = (time.time() - start_time) * 1000
        self._last_generation_time = time.time()
        
        # Parse response
        suggestion = self._extract_suggestion(message)
        urgency = self._determine_urgency(state)
        
        return Explanation(
            message=message,
            timestamp=time.time(),
            mental_state=state,
            suggestion=suggestion,
            urgency=urgency,
            model_used=model_used,
            generation_time_ms=generation_time
        )
    
    def generate(
        self,
        state: MentalState,
        context: UserContext
    ) -> Explanation:
        """
        Generate explanation synchronously.
        
        Args:
            state: Current mental state.
            context: User context.
            
        Returns:
            Generated explanation.
        """
        start_time = time.time()
        
        # Build prompt
        messages = self.prompt_builder.build_messages(
            state, context,
            style=context.notification_style
        )
        
        # Generate response
        if self._provider:
            message = self._provider.generate_sync(messages)
            model_used = self._provider.model_name
        else:
            message = self._get_fallback_message(state)
            model_used = "fallback"
        
        generation_time = (time.time() - start_time) * 1000
        self._last_generation_time = time.time()
        
        # Parse response
        suggestion = self._extract_suggestion(message)
        urgency = self._determine_urgency(state)
        
        return Explanation(
            message=message,
            timestamp=time.time(),
            mental_state=state,
            suggestion=suggestion,
            urgency=urgency,
            model_used=model_used,
            generation_time_ms=generation_time
        )
    
    def _get_fallback_message(self, state: MentalState) -> str:
        """Generate fallback message without LLM."""
        concern = state.get_primary_concern()
        
        if concern == "high_fatigue":
            return (
                "Your fatigue level is elevated. "
                "Consider taking a short break to rest your eyes and stretch."
            )
        elif concern == "high_cognitive_load":
            return (
                "You seem to be under high mental load. "
                "Try breaking your current task into smaller steps."
            )
        elif concern == "low_focus":
            return (
                "Your focus appears to be wandering. "
                "A quick change of scenery or a few deep breaths might help."
            )
        else:
            return (
                f"You're doing well! Focus is at {state.focus_level:.0f}%. "
                "Keep up the good work."
            )
    
    def _extract_suggestion(self, message: str) -> Optional[str]:
        """Extract actionable suggestion from message."""
        # Look for common suggestion patterns
        suggestion_starters = [
            "try ", "consider ", "you might ", "perhaps ",
            "take a ", "why not ", "I suggest "
        ]
        
        message_lower = message.lower()
        for starter in suggestion_starters:
            if starter in message_lower:
                idx = message_lower.find(starter)
                # Extract until end of sentence
                end_idx = message.find(".", idx)
                if end_idx == -1:
                    end_idx = len(message)
                return message[idx:end_idx + 1].strip()
        
        return None
    
    def _determine_urgency(self, state: MentalState) -> str:
        """Determine urgency level based on state."""
        concern = state.get_primary_concern()
        
        if concern in ["high_fatigue", "high_cognitive_load"]:
            return "warning"
        elif concern == "low_focus":
            return "suggestion"
        else:
            return "info"
    
    def set_provider(self, provider_name: str, **kwargs) -> None:
        """
        Switch to a different provider.
        
        Args:
            provider_name: Name of provider ("openrouter", "ollama").
            **kwargs: Provider-specific configuration.
        """
        self.config.provider = provider_name
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        self._init_provider()
    
    @property
    def is_available(self) -> bool:
        """Check if the current provider is available."""
        return self._provider is not None and self._provider.is_available


def get_time_of_day() -> str:
    """Get current time of day category."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_day_of_week() -> str:
    """Get current day of week."""
    return datetime.now().strftime("%A")

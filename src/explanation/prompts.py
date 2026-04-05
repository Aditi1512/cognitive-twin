"""
Prompt templates for explanation generation.

Defines system prompts and message templates for LLMs.
"""

from typing import Optional
from ..models import MentalState, UserContext


SYSTEM_PROMPT = """You are a supportive cognitive wellness assistant helping a user understand their mental state during work or study sessions. Your role is to:

1. Acknowledge their current cognitive state with empathy
2. Provide brief, actionable insights
3. Suggest helpful interventions when needed

Guidelines:
- Be warm and encouraging, never judgmental
- Keep responses to 2-3 sentences maximum
- Use simple, non-technical language
- Focus on one key insight per message
- Suggest specific actions when appropriate (e.g., "Try a 5-minute break")
- Avoid medical advice or diagnoses

Response format:
- Start with an observation about their current state
- Add context or explanation if helpful
- End with a suggestion if their state needs attention"""


def build_user_prompt(
    state: MentalState,
    context: UserContext,
    include_patterns: bool = True
) -> str:
    """
    Build the user prompt for explanation generation.
    
    Args:
        state: Current mental state.
        context: User context including session info.
        include_patterns: Whether to include personal pattern info.
        
    Returns:
        Formatted prompt string.
    """
    # Format mental state
    focus_desc = _level_description(state.focus_level, "focus")
    fatigue_desc = _level_description(state.fatigue_level, "fatigue")
    load_desc = _level_description(state.cognitive_load, "cognitive load")
    
    prompt = f"""Current Mental State:
- Focus: {state.focus_level:.0f}/100 ({focus_desc})
- Fatigue: {state.fatigue_level:.0f}/100 ({fatigue_desc})
- Cognitive Load: {state.cognitive_load:.0f}/100 ({load_desc})
- Trend: {state.trend.value}
- Confidence: {state.confidence:.0%}

Session Context:
- Duration: {context.session_duration_minutes:.0f} minutes
- Time of Day: {context.time_of_day}
- Day: {context.day_of_week}"""

    if include_patterns and context.typical_focus_range:
        prompt += f"""

Personal Patterns:
{context.get_pattern_summary()}"""

    # Add specific request based on state
    concern = state.get_primary_concern()
    if concern == "high_fatigue":
        prompt += "\n\nThe user appears fatigued. Please acknowledge this and suggest a break."
    elif concern == "high_cognitive_load":
        prompt += "\n\nThe user seems mentally overloaded. Suggest ways to reduce cognitive strain."
    elif concern == "low_focus":
        prompt += "\n\nThe user's focus is low. Provide an encouraging message with a focus tip."
    else:
        prompt += "\n\nProvide a brief, encouraging status update."

    return prompt


def _level_description(value: float, metric: str) -> str:
    """Get description for a metric level."""
    if value >= 80:
        if metric == "fatigue" or metric == "cognitive load":
            return "very high"
        return "excellent"
    elif value >= 60:
        if metric == "fatigue" or metric == "cognitive load":
            return "elevated"
        return "good"
    elif value >= 40:
        return "moderate"
    elif value >= 20:
        if metric == "fatigue" or metric == "cognitive load":
            return "low"
        return "needs attention"
    else:
        if metric == "fatigue" or metric == "cognitive load":
            return "minimal"
        return "very low"


class PromptBuilder:
    """Builds prompts for explanation generation."""
    
    def __init__(self, custom_system_prompt: Optional[str] = None):
        """
        Initialize prompt builder.
        
        Args:
            custom_system_prompt: Optional custom system prompt.
        """
        self.system_prompt = custom_system_prompt or SYSTEM_PROMPT
    
    def build(
        self,
        state: MentalState,
        context: UserContext,
        style: str = "balanced"
    ) -> tuple:
        """
        Build complete prompt for LLM.
        
        Args:
            state: Current mental state.
            context: User context.
            style: Response style ("minimal", "balanced", "detailed").
            
        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        user_prompt = build_user_prompt(
            state, context,
            include_patterns=(style != "minimal")
        )
        
        return (self.system_prompt, user_prompt)
    
    def build_messages(
        self,
        state: MentalState,
        context: UserContext,
        style: str = "balanced"
    ) -> list:
        """
        Build message list for chat completion API.
        
        Args:
            state: Current mental state.
            context: User context.
            style: Response style.
            
        Returns:
            List of message dicts for API.
        """
        system_prompt, user_prompt = self.build(state, context, style)
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

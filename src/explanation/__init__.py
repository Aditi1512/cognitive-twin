"""Explanation generation layer - GenAI integration."""

from .generator import ExplanationGenerator
from .prompts import PromptBuilder

__all__ = ["ExplanationGenerator", "PromptBuilder"]

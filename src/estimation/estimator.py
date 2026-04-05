"""
Mental state estimation from cognitive embeddings.

Provides additional processing and smoothing of mental state estimates.
"""

from typing import Optional, List
from collections import deque
import numpy as np

from ..models import MentalState, Trend, FeatureVector
from ..config import EstimationConfig


class MentalStateEstimator:
    """Estimates and smooths mental state values."""
    
    def __init__(self, config: Optional[EstimationConfig] = None):
        """
        Initialize estimator.
        
        Args:
            config: Estimation configuration.
        """
        if config is None:
            config = EstimationConfig()
        
        self.config = config
        self.smoothing_factor = config.smoothing_factor
        
        # State history for smoothing and trends
        self._history: deque = deque(maxlen=config.trend_window_size)
        
        # Previous smoothed values
        self._prev_focus: Optional[float] = None
        self._prev_fatigue: Optional[float] = None
        self._prev_cogload: Optional[float] = None
    
    def smooth(self, state: MentalState) -> MentalState:
        """
        Apply smoothing to mental state values.
        
        Uses exponential moving average for stable output.
        
        Args:
            state: Raw mental state estimate.
            
        Returns:
            Smoothed mental state.
        """
        alpha = self.smoothing_factor
        
        if self._prev_focus is None:
            # First state, no smoothing
            self._prev_focus = state.focus_level
            self._prev_fatigue = state.fatigue_level
            self._prev_cogload = state.cognitive_load
        else:
            # Exponential smoothing
            self._prev_focus = alpha * state.focus_level + (1 - alpha) * self._prev_focus
            self._prev_fatigue = alpha * state.fatigue_level + (1 - alpha) * self._prev_fatigue
            self._prev_cogload = alpha * state.cognitive_load + (1 - alpha) * self._prev_cogload
        
        smoothed = MentalState(
            timestamp=state.timestamp,
            focus_level=self._prev_focus,
            fatigue_level=self._prev_fatigue,
            cognitive_load=self._prev_cogload,
            confidence=state.confidence,
            trend=self._compute_trend()
        )
        
        self._history.append(smoothed)
        
        return smoothed
    
    def _compute_trend(self) -> Trend:
        """Compute trend from history."""
        if len(self._history) < 10:
            return Trend.STABLE
        
        recent = list(self._history)[-10:]
        first_focus = np.mean([s.focus_level for s in recent[:5]])
        last_focus = np.mean([s.focus_level for s in recent[5:]])
        
        diff = last_focus - first_focus
        
        if diff > 3:
            return Trend.IMPROVING
        elif diff < -3:
            return Trend.DECLINING
        return Trend.STABLE
    
    def get_summary(self) -> dict:
        """Get summary statistics of recent states."""
        if not self._history:
            return {}
        
        states = list(self._history)
        
        return {
            "avg_focus": np.mean([s.focus_level for s in states]),
            "avg_fatigue": np.mean([s.fatigue_level for s in states]),
            "avg_cognitive_load": np.mean([s.cognitive_load for s in states]),
            "focus_trend": self._compute_trend().value,
            "samples": len(states)
        }
    
    def should_alert(self, state: MentalState) -> Optional[str]:
        """
        Check if state warrants an alert.
        
        Args:
            state: Current mental state.
            
        Returns:
            Alert type string or None.
        """
        if state.fatigue_level > 75:
            return "high_fatigue"
        if state.cognitive_load > 85:
            return "high_cognitive_load"
        if state.focus_level < 25:
            return "low_focus"
        
        # Check for sustained decline
        if len(self._history) >= 20:
            recent_20 = list(self._history)[-20:]
            focus_start = np.mean([s.focus_level for s in recent_20[:10]])
            focus_end = np.mean([s.focus_level for s in recent_20[10:]])
            
            if focus_end - focus_start < -15:
                return "focus_declining"
        
        return None
    
    def reset(self) -> None:
        """Reset estimator state."""
        self._history.clear()
        self._prev_focus = None
        self._prev_fatigue = None
        self._prev_cogload = None

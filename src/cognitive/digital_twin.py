"""
Cognitive Digital Twin - personalized cognitive model.

Maintains a personalized representation of user's cognitive patterns.
"""

import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import numpy as np

from ..models import FeatureVector, MentalState, Trend
from ..config import CognitiveConfig
from .temporal_model import TemporalModel


class CognitiveDigitalTwin:
    """Personal cognitive model that learns user patterns."""
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        config: Optional[CognitiveConfig] = None,
        data_dir: Optional[Path] = None
    ):
        """
        Initialize digital twin.
        
        Args:
            user_id: Unique user identifier.
            config: Cognitive modeling configuration.
            data_dir: Directory for storing twin data.
        """
        self.user_id = user_id or str(uuid.uuid4())[:8]
        
        if config is None:
            config = CognitiveConfig()
        self.config = config
        
        if data_dir is None:
            data_dir = Path.home() / ".cognitive_twin" / "twins"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize temporal model
        self.temporal_model = TemporalModel(config)
        
        # Personal patterns (learned over time)
        self._patterns: Dict[str, Any] = {
            "focus_baseline": 60.0,
            "fatigue_baseline": 30.0,
            "typical_session_length": 60.0,  # minutes
            "best_performance_hour": 10,  # 10 AM
            "focus_decay_rate": 0.1,  # per minute
            "recovery_rate": 0.5,  # per minute of break
            "sample_count": 0,
            # Calibration adjustments from user feedback
            "focus_calibration": 0.0,
            "fatigue_calibration": 0.0,
            "load_calibration": 0.0
        }
        
        # State history for trend detection
        self._state_history: List[MentalState] = []
        self._max_history = 60  # 1 minute at 1 Hz
        
        # Load existing data if available
        self._load()
    
    def update(self, features: FeatureVector) -> Optional[MentalState]:
        """
        Update twin with new observation.
        
        Args:
            features: Current feature vector.
            
        Returns:
            Updated mental state or None if insufficient data.
        """
        # Add features to temporal model
        feature_array = features.to_array()
        self.temporal_model.add_features(feature_array, features.timestamp)
        
        # Get embedding
        embedding = self.temporal_model.get_embedding()
        
        if embedding is None:
            return None
        
        # Estimate mental states from embedding
        mental_state = self._estimate_mental_state(embedding, features.timestamp)
        
        # Update patterns (lightweight online learning)
        self._update_patterns(mental_state)
        
        # Add to history
        self._state_history.append(mental_state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)
        
        return mental_state
    
    def _estimate_mental_state(
        self, embedding: np.ndarray, timestamp: float
    ) -> MentalState:
        """
        Estimate mental states from cognitive embedding.
        
        Uses heuristic estimation with personal baselines and calibration.
        For production use, replace with trained LSTM classifiers.
        """
        # Base estimation from embedding statistics
        emb_mean = np.mean(embedding)
        emb_std = np.std(embedding)
        emb_range = np.max(embedding) - np.min(embedding)
        
        # Additional features from embedding
        emb_skew = self._compute_skewness(embedding)
        emb_energy = np.sum(embedding ** 2) / len(embedding)
        
        # Focus: higher when embedding is stable, positive, and low variance
        # Low variance = consistent attention, positive mean = engaged
        focus_raw = 50 + emb_mean * 25 + (1 - min(emb_std, 1)) * 15 - abs(emb_skew) * 5
        focus_level = max(0, min(100, focus_raw))
        
        # Fatigue: increases with variance, negative mean, and high energy (strain)
        fatigue_raw = 30 + emb_std * 30 - emb_mean * 15 + emb_energy * 10
        fatigue_level = max(0, min(100, fatigue_raw))
        
        # Cognitive load: based on range (complexity) and energy (effort)
        cogload_raw = 40 + emb_range * 25 + emb_energy * 15 + emb_std * 10
        cognitive_load = max(0, min(100, cogload_raw))
        
        # Apply personal baselines (30% weight)
        focus_level = focus_level * 0.7 + self._patterns["focus_baseline"] * 0.3
        fatigue_level = fatigue_level * 0.7 + self._patterns["fatigue_baseline"] * 0.3
        
        # Apply calibration adjustments from user feedback
        focus_level = max(0, min(100, focus_level + self._patterns.get("focus_calibration", 0)))
        fatigue_level = max(0, min(100, fatigue_level + self._patterns.get("fatigue_calibration", 0)))
        cognitive_load = max(0, min(100, cognitive_load + self._patterns.get("load_calibration", 0)))
        
        # Apply time-of-day effects based on learned patterns
        hour = int((timestamp % 86400) / 3600)  # Current hour (0-23)
        best_hour = self._patterns.get("best_performance_hour", 10)
        hour_diff = min(abs(hour - best_hour), 24 - abs(hour - best_hour))
        
        # Focus is typically best at the user's best hour, declines further away
        time_focus_adjustment = -2 * hour_diff  # Up to -24 points 12 hours away
        focus_level = max(20, min(95, focus_level + time_focus_adjustment * 0.3))
        
        # Compute trend
        trend = self._compute_trend()
        
        # Confidence based on sequence quality and sample count
        base_confidence = self.temporal_model.get_sequence_quality()
        # Confidence increases as we collect more samples
        sample_bonus = min(0.2, self._patterns["sample_count"] / 10000)
        confidence = min(1.0, base_confidence + sample_bonus)
        
        return MentalState(
            timestamp=timestamp,
            focus_level=float(focus_level),
            fatigue_level=float(fatigue_level),
            cognitive_load=float(cognitive_load),
            confidence=confidence,
            trend=trend
        )
    
    def _compute_skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of array (measure of asymmetry)."""
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))
    
    def apply_calibration(self, focus_adj: float, fatigue_adj: float, load_adj: float) -> None:
        """
        Apply calibration adjustments from user feedback.
        
        Args:
            focus_adj: Focus level adjustment (-50 to +50)
            fatigue_adj: Fatigue level adjustment (-50 to +50)
            load_adj: Cognitive load adjustment (-50 to +50)
        """
        self._patterns["focus_calibration"] = max(-50, min(50, focus_adj))
        self._patterns["fatigue_calibration"] = max(-50, min(50, fatigue_adj))
        self._patterns["load_calibration"] = max(-50, min(50, load_adj))
        self.save()
    
    def _compute_trend(self) -> Trend:
        """Compute trend from recent state history."""
        if len(self._state_history) < 10:
            return Trend.STABLE
        
        recent = self._state_history[-10:]
        
        # Compare first half to second half
        first_half_focus = np.mean([s.focus_level for s in recent[:5]])
        second_half_focus = np.mean([s.focus_level for s in recent[5:]])
        
        diff = second_half_focus - first_half_focus
        
        if diff > 5:
            return Trend.IMPROVING
        elif diff < -5:
            return Trend.DECLINING
        else:
            return Trend.STABLE
    
    def _update_patterns(self, state: MentalState) -> None:
        """Update personal patterns with new observation."""
        self._patterns["sample_count"] += 1
        
        # Exponential moving average for baselines
        alpha = self.config.adaptation_rate
        
        self._patterns["focus_baseline"] = (
            (1 - alpha) * self._patterns["focus_baseline"] +
            alpha * state.focus_level
        )
        self._patterns["fatigue_baseline"] = (
            (1 - alpha) * self._patterns["fatigue_baseline"] +
            alpha * state.fatigue_level
        )
    
    def get_state(self) -> Optional[MentalState]:
        """Get most recent mental state."""
        if not self._state_history:
            return None
        return self._state_history[-1]
    
    def get_patterns(self) -> Dict[str, Any]:
        """Get learned personal patterns."""
        return self._patterns.copy()
    
    def get_recent_states(self, count: int = 10) -> List[MentalState]:
        """Get recent mental states."""
        return self._state_history[-count:]
    
    def save(self) -> None:
        """Save twin data to disk."""
        twin_path = self.data_dir / f"{self.user_id}.json"
        
        data = {
            "user_id": self.user_id,
            "patterns": self._patterns,
            "saved_at": time.time()
        }
        
        with open(twin_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self) -> bool:
        """Load twin data from disk."""
        twin_path = self.data_dir / f"{self.user_id}.json"
        
        if not twin_path.exists():
            return False
        
        try:
            with open(twin_path, "r") as f:
                data = json.load(f)
            
            self._patterns.update(data.get("patterns", {}))
            return True
            
        except Exception:
            return False
    
    def reset(self) -> None:
        """Reset twin to initial state."""
        self.temporal_model.reset()
        self._state_history.clear()
        self._patterns = {
            "focus_baseline": 60.0,
            "fatigue_baseline": 30.0,
            "typical_session_length": 60.0,
            "best_performance_hour": 10,
            "focus_decay_rate": 0.1,
            "recovery_rate": 0.5,
            "sample_count": 0,
            "focus_calibration": 0.0,
            "fatigue_calibration": 0.0,
            "load_calibration": 0.0
        }

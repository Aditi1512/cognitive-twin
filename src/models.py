"""
Data models for Personal Cognitive Digital Twin.

Core data structures used throughout the application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


class Trend(Enum):
    """Mental state trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class CapturedFrame:
    """Raw frame from webcam with metadata."""
    frame: np.ndarray  # Shape: (H, W, 3), dtype: uint8
    timestamp: float
    frame_id: int
    camera_active: bool = True
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get frame resolution (width, height)."""
        return (self.frame.shape[1], self.frame.shape[0])
    
    def to_rgb(self) -> np.ndarray:
        """Convert BGR to RGB."""
        return self.frame[:, :, ::-1].copy()


@dataclass
class LandmarkData:
    """MediaPipe Face Mesh landmarks."""
    landmarks: np.ndarray  # Shape: (468, 3)
    confidence: float
    face_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    timestamp: float


@dataclass
class EyeFeatures:
    """Eye-related extracted features."""
    ear_left: float  # Eye Aspect Ratio
    ear_right: float
    ear_average: float
    gaze_x: float  # [-1, 1]
    gaze_y: float  # [-1, 1]
    eyes_closed: bool
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.ear_left,
            self.ear_right,
            self.ear_average,
            self.gaze_x,
            self.gaze_y,
            1.0 if self.eyes_closed else 0.0
        ], dtype=np.float32)


@dataclass
class HeadPose:
    """Head orientation in 3D space."""
    pitch: float  # Up/down (degrees)
    yaw: float    # Left/right (degrees)
    roll: float   # Tilt (degrees)
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to normalized numpy array."""
        return np.array([
            self.pitch / 45.0,
            self.yaw / 45.0,
            self.roll / 45.0
        ], dtype=np.float32)


@dataclass
class BlinkFeatures:
    """Blink detection features."""
    blink_detected: bool
    blink_rate: float  # Blinks per minute
    avg_blink_duration: float  # Milliseconds
    time_since_last_blink: float  # Seconds
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            1.0 if self.blink_detected else 0.0,
            self.blink_rate / 30.0,  # Normalize
            min(self.time_since_last_blink / 60.0, 1.0)
        ], dtype=np.float32)


@dataclass
class FeatureVector:
    """Complete feature set for one frame."""
    timestamp: float
    frame_id: int
    face_detected: bool
    face_confidence: float
    
    # Component features (optional if face not detected)
    landmarks: Optional[LandmarkData] = None
    eye_features: Optional[EyeFeatures] = None
    head_pose: Optional[HeadPose] = None
    blink_features: Optional[BlinkFeatures] = None
    
    # Quality metrics
    lighting_quality: float = 1.0
    face_distance: float = 50.0
    
    def to_array(self) -> np.ndarray:
        """Convert to 32D numpy array for model input."""
        if not self.face_detected:
            return np.zeros(32, dtype=np.float32)
        
        features = []
        
        # Eye features (6)
        if self.eye_features:
            features.extend(self.eye_features.to_array().tolist())
        else:
            features.extend([0.0] * 6)
        
        # Blink features (3)
        if self.blink_features:
            features.extend(self.blink_features.to_array().tolist())
        else:
            features.extend([0.0] * 3)
        
        # Head pose (3)
        if self.head_pose:
            features.extend(self.head_pose.to_array().tolist())
        else:
            features.extend([0.0] * 3)
        
        # Placeholder for expression features (10)
        features.extend([0.0] * 10)
        
        # Placeholder for motion features (6)
        features.extend([0.0] * 6)
        
        # Quality features (4)
        features.extend([
            self.face_confidence,
            self.lighting_quality,
            min(1.0, self.face_distance / 100.0),
            1.0  # Padding
        ])
        
        return np.array(features[:32], dtype=np.float32)


@dataclass
class MentalState:
    """Current mental state estimation."""
    timestamp: float
    focus_level: float      # 0-100
    fatigue_level: float    # 0-100
    cognitive_load: float   # 0-100
    confidence: float       # 0-1
    trend: Trend = Trend.STABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "focus_level": round(self.focus_level, 1),
            "fatigue_level": round(self.fatigue_level, 1),
            "cognitive_load": round(self.cognitive_load, 1),
            "confidence": round(self.confidence, 2),
            "trend": self.trend.value
        }
    
    def get_primary_concern(self) -> Optional[str]:
        """Get the most concerning state, if any."""
        if self.fatigue_level > 70:
            return "high_fatigue"
        if self.cognitive_load > 80:
            return "high_cognitive_load"
        if self.focus_level < 30:
            return "low_focus"
        return None
    
    def get_status_emoji(self) -> str:
        """Get status emoji for display."""
        avg_wellness = (self.focus_level + (100 - self.fatigue_level) + (100 - self.cognitive_load)) / 3
        if avg_wellness >= 70:
            return "🟢"
        elif avg_wellness >= 40:
            return "🟡"
        else:
            return "🔴"


@dataclass
class UserContext:
    """Context for personalized explanation generation."""
    session_duration_minutes: float
    time_of_day: str
    day_of_week: str
    recent_states: List[MentalState] = field(default_factory=list)
    notification_style: str = "balanced"
    
    # Personal patterns (learned over time)
    typical_focus_range: Tuple[float, float] = (40.0, 80.0)
    typical_fatigue_pattern: str = "gradual increase"
    best_performance_time: str = "morning"
    
    def get_pattern_summary(self) -> str:
        """Generate summary of user patterns for LLM context."""
        return (
            f"Typical focus range: {self.typical_focus_range[0]:.0f}-"
            f"{self.typical_focus_range[1]:.0f}. "
            f"Usually performs best in the {self.best_performance_time}. "
            f"Fatigue pattern: {self.typical_fatigue_pattern}."
        )


@dataclass
class Explanation:
    """Natural language explanation from GenAI."""
    message: str
    timestamp: float
    mental_state: MentalState
    suggestion: Optional[str] = None
    urgency: str = "info"  # "info", "suggestion", "warning"
    model_used: str = "unknown"
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message,
            "timestamp": self.timestamp,
            "suggestion": self.suggestion,
            "urgency": self.urgency,
            "mental_state": self.mental_state.to_dict()
        }


@dataclass
class Session:
    """A monitoring session."""
    id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Statistics
    avg_focus: float = 0.0
    avg_fatigue: float = 0.0
    avg_cognitive_load: float = 0.0
    total_frames: int = 0
    explanations_count: int = 0
    
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        import time
        end = self.end_time or time.time()
        return (end - self.start_time) / 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.duration_minutes(),
            "avg_focus": self.avg_focus,
            "avg_fatigue": self.avg_fatigue,
            "avg_cognitive_load": self.avg_cognitive_load
        }

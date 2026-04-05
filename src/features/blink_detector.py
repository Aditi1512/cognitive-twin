"""
Blink detection and rate calculation.

Detects blinks and calculates blink rate over time.
"""

import time
from collections import deque
from typing import Optional
from ..models import BlinkFeatures, EyeFeatures


class BlinkDetector:
    """Detects blinks and calculates blink statistics."""
    
    def __init__(
        self,
        ear_threshold: float = 0.2,
        min_blink_frames: int = 2,
        max_blink_frames: int = 15,
        rate_window_seconds: float = 60.0
    ):
        """
        Initialize blink detector.
        
        Args:
            ear_threshold: EAR threshold below which eyes are considered closed.
            min_blink_frames: Minimum frames for a valid blink.
            max_blink_frames: Maximum frames for a valid blink (longer = eyes closed).
            rate_window_seconds: Window for calculating blink rate.
        """
        self.ear_threshold = ear_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.rate_window_seconds = rate_window_seconds
        
        # State tracking
        self._consecutive_closed = 0
        self._is_blinking = False
        self._last_blink_time: Optional[float] = None
        self._blink_start_time: Optional[float] = None
        
        # Blink history for rate calculation
        self._blink_times: deque = deque()
        self._blink_durations: deque = deque(maxlen=20)
    
    def detect(self, eye_features: EyeFeatures, timestamp: float) -> BlinkFeatures:
        """
        Detect blink and calculate statistics.
        
        Args:
            eye_features: Eye features including EAR.
            timestamp: Current timestamp.
            
        Returns:
            BlinkFeatures with detection and statistics.
        """
        ear = eye_features.ear_average
        blink_detected = False
        
        if ear < self.ear_threshold:
            # Eyes are closed
            self._consecutive_closed += 1
            if not self._is_blinking:
                self._is_blinking = True
                self._blink_start_time = timestamp
        else:
            # Eyes are open
            if self._is_blinking:
                # Check if this was a valid blink
                if (self.min_blink_frames <= self._consecutive_closed <= self.max_blink_frames):
                    blink_detected = True
                    self._record_blink(timestamp)
                
                self._is_blinking = False
            self._consecutive_closed = 0
        
        # Clean old blinks from history
        self._clean_old_blinks(timestamp)
        
        # Calculate statistics
        blink_rate = self._calculate_blink_rate()
        avg_duration = self._calculate_avg_duration()
        time_since_last = self._time_since_last_blink(timestamp)
        
        return BlinkFeatures(
            blink_detected=blink_detected,
            blink_rate=blink_rate,
            avg_blink_duration=avg_duration,
            time_since_last_blink=time_since_last
        )
    
    def _record_blink(self, timestamp: float) -> None:
        """Record a blink occurrence."""
        self._blink_times.append(timestamp)
        
        if self._blink_start_time is not None:
            duration_ms = (timestamp - self._blink_start_time) * 1000
            self._blink_durations.append(duration_ms)
        
        self._last_blink_time = timestamp
    
    def _clean_old_blinks(self, current_time: float) -> None:
        """Remove blinks outside the rate window."""
        cutoff = current_time - self.rate_window_seconds
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
    
    def _calculate_blink_rate(self) -> float:
        """Calculate blinks per minute."""
        if not self._blink_times:
            return 0.0
        
        # Blinks in the window, scaled to per-minute rate
        count = len(self._blink_times)
        rate = count * (60.0 / self.rate_window_seconds)
        return rate
    
    def _calculate_avg_duration(self) -> float:
        """Calculate average blink duration in milliseconds."""
        if not self._blink_durations:
            return 150.0  # Default average blink duration
        return float(sum(self._blink_durations) / len(self._blink_durations))
    
    def _time_since_last_blink(self, current_time: float) -> float:
        """Calculate time since last blink in seconds."""
        if self._last_blink_time is None:
            return 0.0
        return current_time - self._last_blink_time
    
    def reset(self) -> None:
        """Reset detector state."""
        self._consecutive_closed = 0
        self._is_blinking = False
        self._last_blink_time = None
        self._blink_start_time = None
        self._blink_times.clear()
        self._blink_durations.clear()


def is_fatigue_indicator(blink_features: BlinkFeatures) -> bool:
    """
    Check if blink patterns indicate fatigue.
    
    Fatigue indicators:
    - Low blink rate (< 10 bpm) - trying to keep eyes open
    - High blink rate (> 25 bpm) - eye strain
    - Long time since last blink (> 10s) - staring
    
    Args:
        blink_features: Current blink features.
        
    Returns:
        True if patterns suggest fatigue.
    """
    rate = blink_features.blink_rate
    time_since = blink_features.time_since_last_blink
    
    if rate < 10 or rate > 25:
        return True
    if time_since > 10:
        return True
    
    return False

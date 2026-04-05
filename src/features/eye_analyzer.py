"""
Eye analysis for gaze direction and eye aspect ratio.

Extracts eye-related features from facial landmarks.
"""

import numpy as np
from typing import Optional, Tuple
from ..models import EyeFeatures, LandmarkData


# MediaPipe Face Mesh eye landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]


class EyeAnalyzer:
    """Analyzes eye features from facial landmarks."""
    
    def __init__(self):
        """Initialize eye analyzer."""
        self._ear_threshold = 0.2  # Eye closed threshold
        self._history_size = 10
        self._ear_history = []
    
    def analyze(self, landmarks: LandmarkData) -> EyeFeatures:
        """
        Analyze eye features from landmarks.
        
        Args:
            landmarks: Facial landmark data.
            
        Returns:
            EyeFeatures with EAR, gaze, and closure status.
        """
        lm = landmarks.landmarks
        
        # Calculate Eye Aspect Ratio (EAR)
        ear_left = self._calculate_ear(lm, LEFT_EYE_INDICES)
        ear_right = self._calculate_ear(lm, RIGHT_EYE_INDICES)
        ear_average = (ear_left + ear_right) / 2
        
        # Update history for smoothing
        self._ear_history.append(ear_average)
        if len(self._ear_history) > self._history_size:
            self._ear_history.pop(0)
        
        # Determine if eyes are closed
        smoothed_ear = np.mean(self._ear_history)
        eyes_closed = smoothed_ear < self._ear_threshold
        
        # Calculate gaze direction
        gaze_x, gaze_y = self._calculate_gaze(lm)
        
        return EyeFeatures(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_average=ear_average,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            eyes_closed=eyes_closed
        )
    
    def _calculate_ear(self, landmarks: np.ndarray, indices: list) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Args:
            landmarks: All facial landmarks.
            indices: Eye landmark indices [p1, p2, p3, p4, p5, p6].
            
        Returns:
            Eye Aspect Ratio value.
        """
        try:
            p1 = landmarks[indices[0]][:2]
            p2 = landmarks[indices[1]][:2]
            p3 = landmarks[indices[2]][:2]
            p4 = landmarks[indices[3]][:2]
            p5 = landmarks[indices[4]][:2]
            p6 = landmarks[indices[5]][:2]
            
            # Vertical distances
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            
            # Horizontal distance
            h = np.linalg.norm(p1 - p4)
            
            if h < 1e-6:
                return 0.3  # Default value
            
            ear = (v1 + v2) / (2.0 * h)
            return float(ear)
            
        except (IndexError, ValueError):
            return 0.3  # Default value
    
    def _calculate_gaze(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calculate gaze direction from iris position.
        
        Args:
            landmarks: All facial landmarks.
            
        Returns:
            Tuple of (gaze_x, gaze_y) in range [-1, 1].
        """
        try:
            # Check if iris landmarks are available (indices 468+)
            if len(landmarks) < 478:
                return (0.0, 0.0)
            
            # Get eye corners and iris centers
            # Left eye
            left_corner_inner = landmarks[362][:2]
            left_corner_outer = landmarks[263][:2]
            left_iris_center = landmarks[468][:2]
            
            # Right eye
            right_corner_inner = landmarks[133][:2]
            right_corner_outer = landmarks[33][:2]
            right_iris_center = landmarks[473][:2]
            
            # Calculate normalized position for each eye
            def normalize_iris_position(iris, inner, outer):
                eye_width = np.linalg.norm(outer - inner)
                if eye_width < 1e-6:
                    return 0.0
                eye_center = (inner + outer) / 2
                offset = iris - eye_center
                direction = (outer - inner) / eye_width
                return float(np.dot(offset, direction) / (eye_width / 2))
            
            # Average gaze from both eyes
            gaze_x_left = normalize_iris_position(
                left_iris_center, left_corner_inner, left_corner_outer
            )
            gaze_x_right = normalize_iris_position(
                right_iris_center, right_corner_inner, right_corner_outer
            )
            
            gaze_x = (gaze_x_left + gaze_x_right) / 2
            
            # For vertical gaze, use a simpler approximation
            left_top = landmarks[386][:2]
            left_bottom = landmarks[374][:2]
            left_center_y = (left_top[1] + left_bottom[1]) / 2
            gaze_y = (left_iris_center[1] - left_center_y) / max(
                abs(left_top[1] - left_bottom[1]), 1e-6
            )
            
            # Clamp to [-1, 1]
            gaze_x = max(-1.0, min(1.0, gaze_x))
            gaze_y = max(-1.0, min(1.0, gaze_y))
            
            return (gaze_x, gaze_y)
            
        except (IndexError, ValueError):
            return (0.0, 0.0)
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self._ear_history.clear()

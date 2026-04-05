"""
Head pose estimation from facial landmarks.

Estimates pitch, yaw, and roll angles.
"""

import numpy as np
from typing import Tuple
from ..models import HeadPose, LandmarkData


# 3D model points for head pose estimation
# These are approximate facial landmark positions in a normalized coordinate system
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for the above points
POSE_LANDMARK_INDICES = [
    1,    # Nose tip
    152,  # Chin
    263,  # Left eye outer corner
    33,   # Right eye outer corner
    287,  # Left mouth corner
    57    # Right mouth corner
]


class HeadPoseEstimator:
    """Estimates head pose from facial landmarks."""
    
    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """
        Initialize head pose estimator.
        
        Args:
            frame_width: Video frame width.
            frame_height: Video frame height.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Camera matrix (approximate)
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        self.dist_coeffs = np.zeros((4, 1))
        
        # History for smoothing
        self._history_size = 5
        self._pitch_history = []
        self._yaw_history = []
        self._roll_history = []
    
    def estimate(self, landmarks: LandmarkData) -> HeadPose:
        """
        Estimate head pose from landmarks.
        
        Args:
            landmarks: Facial landmark data.
            
        Returns:
            HeadPose with pitch, yaw, roll angles.
        """
        try:
            # Get 2D image points from landmarks
            image_points = self._get_image_points(landmarks.landmarks)
            
            # Solve PnP
            success, rotation_vector, translation_vector = self._solve_pnp(image_points)
            
            if not success:
                return HeadPose(pitch=0.0, yaw=0.0, roll=0.0, confidence=0.0)
            
            # Convert rotation vector to Euler angles
            pitch, yaw, roll = self._rotation_to_euler(rotation_vector)
            
            # Smooth the values
            pitch, yaw, roll = self._smooth_angles(pitch, yaw, roll)
            
            return HeadPose(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                confidence=1.0
            )
            
        except Exception:
            return HeadPose(pitch=0.0, yaw=0.0, roll=0.0, confidence=0.0)
    
    def _get_image_points(self, landmarks: np.ndarray) -> np.ndarray:
        """Extract 2D image points from landmarks."""
        image_points = []
        for idx in POSE_LANDMARK_INDICES:
            # MediaPipe landmarks are normalized [0, 1]
            x = landmarks[idx][0] * self.frame_width
            y = landmarks[idx][1] * self.frame_height
            image_points.append((x, y))
        return np.array(image_points, dtype=np.float64)
    
    def _solve_pnp(self, image_points: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Solve Perspective-n-Point problem."""
        import cv2
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        return success, rotation_vector, translation_vector
    
    def _rotation_to_euler(self, rotation_vector: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation vector to Euler angles (pitch, yaw, roll)."""
        import cv2
        
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0.0
        
        # Convert to degrees
        pitch = float(np.degrees(pitch))
        yaw = float(np.degrees(yaw))
        roll = float(np.degrees(roll))
        
        return pitch, yaw, roll
    
    def _smooth_angles(
        self, pitch: float, yaw: float, roll: float
    ) -> Tuple[float, float, float]:
        """Smooth angles using moving average."""
        self._pitch_history.append(pitch)
        self._yaw_history.append(yaw)
        self._roll_history.append(roll)
        
        if len(self._pitch_history) > self._history_size:
            self._pitch_history.pop(0)
            self._yaw_history.pop(0)
            self._roll_history.pop(0)
        
        return (
            float(np.mean(self._pitch_history)),
            float(np.mean(self._yaw_history)),
            float(np.mean(self._roll_history))
        )
    
    def reset(self) -> None:
        """Reset estimator state."""
        self._pitch_history.clear()
        self._yaw_history.clear()
        self._roll_history.clear()

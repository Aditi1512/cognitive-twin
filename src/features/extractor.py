"""
Main feature extraction module.

Coordinates extraction of all visual features from video frames.
Uses MediaPipe Tasks API (Python 3.14 compatible).
"""

import time
from typing import Optional
import numpy as np
from pathlib import Path

# MediaPipe Tasks API import
MEDIAPIPE_AVAILABLE = False
FaceLandmarker = None
FaceLandmarkerOptions = None
BaseOptions = None
mp_image = None

def _try_import_mediapipe_tasks():
    """Attempt to import MediaPipe Tasks API."""
    global MEDIAPIPE_AVAILABLE, FaceLandmarker, FaceLandmarkerOptions, BaseOptions, mp_image
    
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        BaseOptions = python.BaseOptions
        mp_image = mp.Image
        MEDIAPIPE_AVAILABLE = True
        print("MediaPipe Tasks API loaded successfully!")
        return True
    except Exception as e:
        print(f"MediaPipe Tasks API not available: {e}")
        return False

# Run the import attempt
_try_import_mediapipe_tasks()

from ..models import CapturedFrame, FeatureVector, LandmarkData, EyeFeatures, HeadPose, BlinkFeatures
from ..config import FeaturesConfig
from .eye_analyzer import EyeAnalyzer
from .head_pose import HeadPoseEstimator
from .blink_detector import BlinkDetector


class FeatureExtractor:
    """Extracts all behavioral features from video frames."""
    
    # Path to the face landmarker model
    MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "face_landmarker.task"
    
    def __init__(self, config: Optional[FeaturesConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration.
        """
        if config is None:
            config = FeaturesConfig()
        
        self.config = config
        self._mediapipe_available = MEDIAPIPE_AVAILABLE and self.MODEL_PATH.exists()
        self._landmarker = None
        
        # Initialize MediaPipe Face Landmarker if available
        if self._mediapipe_available:
            try:
                base_options = BaseOptions(model_asset_path=str(self.MODEL_PATH))
                options = FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
                print("FaceLandmarker initialized successfully!")
            except Exception as e:
                print(f"FaceLandmarker initialization failed: {e}")
                self._mediapipe_available = False
        else:
            if MEDIAPIPE_AVAILABLE and not self.MODEL_PATH.exists():
                print(f"Model file not found at {self.MODEL_PATH}")
        
        # Initialize component analyzers
        self._eye_analyzer = EyeAnalyzer()
        self._head_pose_estimator: Optional[HeadPoseEstimator] = None
        self._blink_detector = BlinkDetector()
        
        # State
        self._initialized = False
        self._frame_count = 0
    
    @property
    def is_real_detection(self) -> bool:
        """Check if real face detection is available (not simulation)."""
        return self._mediapipe_available and self._landmarker is not None
    
    def extract(self, frame: CapturedFrame) -> FeatureVector:
        """
        Extract all features from a video frame.
        
        Args:
            frame: Captured frame from camera.
            
        Returns:
            FeatureVector with all extracted features.
        """
        self._frame_count += 1
        
        # Initialize head pose estimator with frame dimensions if needed
        if self._head_pose_estimator is None:
            h, w = frame.frame.shape[:2]
            self._head_pose_estimator = HeadPoseEstimator(
                frame_width=w, frame_height=h
            )
        
        # If MediaPipe not available, return simulated features
        if not self._mediapipe_available or self._landmarker is None:
            return self._generate_simulated_features(frame)
        
        # Convert frame to MediaPipe Image format
        try:
            import mediapipe as mp
            rgb_frame = frame.to_rgb()
            # Create MediaPipe Image using the correct format
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect face landmarks
            result = self._landmarker.detect(mp_frame)
        except Exception as e:
            print(f"Face detection error: {e}")
            return self._generate_simulated_features(frame)
        
        if not result.face_landmarks:
            # No face detected
            return FeatureVector(
                timestamp=frame.timestamp,
                frame_id=frame.frame_id,
                face_detected=False,
                face_confidence=0.0,
                landmarks=None,
                eye_features=None,
                head_pose=None,
                blink_features=None,
                lighting_quality=self._estimate_lighting(frame.frame)
            )
        
        # Process first face
        face_landmarks = result.face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks_array = self._landmarks_to_array(face_landmarks, rgb_frame.shape)
        
        # Create landmark data
        landmark_data = LandmarkData(
            landmarks=landmarks_array,
            confidence=1.0,
            face_rect=self._get_face_rect(landmarks_array, rgb_frame.shape),
            timestamp=frame.timestamp
        )
        
        # Extract component features
        eye_features = self._eye_analyzer.analyze(landmark_data)
        head_pose = self._head_pose_estimator.estimate(landmark_data)
        blink_features = self._blink_detector.detect(eye_features, frame.timestamp)
        
        # Get blendshapes for more accurate analysis if available
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}
            # Use blendshapes to improve eye feature accuracy
            if eye_features:
                eye_features = self._enhance_with_blendshapes(eye_features, blendshapes)
        
        # Compute quality metrics
        lighting_quality = self._estimate_lighting(frame.frame)
        face_distance = self._estimate_face_distance(landmark_data)
        
        return FeatureVector(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            face_detected=True,
            face_confidence=1.0,
            landmarks=landmark_data,
            eye_features=eye_features,
            head_pose=head_pose,
            blink_features=blink_features,
            lighting_quality=lighting_quality,
            face_distance=face_distance
        )
    
    def _enhance_with_blendshapes(self, eye_features: EyeFeatures, blendshapes: dict) -> EyeFeatures:
        """Enhance eye features using blendshapes data."""
        # Get eye closure from blendshapes
        left_blink = blendshapes.get('eyeBlinkLeft', 0.0)
        right_blink = blendshapes.get('eyeBlinkRight', 0.0)
        
        # Get eye direction from blendshapes
        look_up_l = blendshapes.get('eyeLookUpLeft', 0.0)
        look_down_l = blendshapes.get('eyeLookDownLeft', 0.0)
        look_in_l = blendshapes.get('eyeLookInLeft', 0.0)
        look_out_l = blendshapes.get('eyeLookOutLeft', 0.0)
        
        look_up_r = blendshapes.get('eyeLookUpRight', 0.0)
        look_down_r = blendshapes.get('eyeLookDownRight', 0.0)
        look_in_r = blendshapes.get('eyeLookInRight', 0.0)
        look_out_r = blendshapes.get('eyeLookOutRight', 0.0)
        
        # Calculate gaze direction from blendshapes
        gaze_x = ((look_out_l - look_in_l) + (look_in_r - look_out_r)) / 2
        gaze_y = ((look_up_l + look_up_r) - (look_down_l + look_down_r)) / 2
        
        # Update EAR based on blink values (inverse relationship)
        ear_left = max(0.05, 0.35 * (1 - left_blink))
        ear_right = max(0.05, 0.35 * (1 - right_blink))
        
        return EyeFeatures(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_average=(ear_left + ear_right) / 2,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            eyes_closed=(left_blink > 0.6 and right_blink > 0.6)
        )
    
    def _landmarks_to_array(
        self, face_landmarks, frame_shape: tuple
    ) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array."""
        landmarks = []
        for lm in face_landmarks:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)
    
    def _get_face_rect(
        self, landmarks: np.ndarray, frame_shape: tuple
    ) -> tuple:
        """Get bounding rectangle of face."""
        h, w = frame_shape[:2]
        
        x_coords = landmarks[:, 0] * w
        y_coords = landmarks[:, 1] * h
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _estimate_lighting(self, frame: np.ndarray) -> float:
        """
        Estimate lighting quality from frame.
        
        Args:
            frame: BGR frame.
            
        Returns:
            Lighting quality score [0, 1].
        """
        # Convert to grayscale and analyze histogram
        gray = np.mean(frame, axis=2)
        mean_brightness = np.mean(gray) / 255.0
        std_brightness = np.std(gray) / 255.0
        
        # Ideal brightness around 0.4-0.6, good contrast (std > 0.2)
        brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
        contrast_score = min(std_brightness / 0.25, 1.0)
        
        return float((brightness_score + contrast_score) / 2)
    
    def _estimate_face_distance(self, landmarks: LandmarkData) -> float:
        """
        Estimate distance of face from camera.
        
        Uses inter-eye distance as a proxy for face distance.
        
        Args:
            landmarks: Facial landmarks.
            
        Returns:
            Estimated distance (arbitrary units, smaller = closer).
        """
        try:
            # Left and right eye outer corners (MediaPipe indices)
            left_eye = landmarks.landmarks[263][:2]
            right_eye = landmarks.landmarks[33][:2]
            
            inter_eye = np.linalg.norm(left_eye - right_eye)
            
            # Normalize: typical inter-eye distance is ~0.2 in normalized coords
            # Map to a distance estimate where 50 is "normal"
            if inter_eye > 0:
                distance = 10.0 / inter_eye
            else:
                distance = 50.0
            
            return float(min(100.0, max(10.0, distance)))
            
        except (IndexError, ValueError):
            return 50.0
    
    def _generate_simulated_features(self, frame: CapturedFrame) -> FeatureVector:
        """
        Generate simulated features when MediaPipe is not available.
        Uses simple heuristics based on frame properties.
        """
        import random
        
        # Simulate natural variation in mental states
        base_seed = int(frame.timestamp * 10) % 1000
        random.seed(base_seed)
        
        # Add some temporal smoothing through frame count
        phase = (self._frame_count % 300) / 300.0
        
        # Simulate gradual changes
        focus_variation = 60 + 20 * np.sin(phase * 2 * np.pi) + random.gauss(0, 5)
        fatigue_variation = 30 + 15 * np.sin(phase * 1.5 * np.pi + 1) + random.gauss(0, 3)
        
        return FeatureVector(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            face_detected=True,  # Simulated face detection
            face_confidence=0.9,
            landmarks=None,
            eye_features=None,
            head_pose=None,
            blink_features=None,
            lighting_quality=self._estimate_lighting(frame.frame)
        )
    
    def reset(self) -> None:
        """Reset extractor state."""
        self._eye_analyzer.reset()
        if self._head_pose_estimator:
            self._head_pose_estimator.reset()
        self._blink_detector.reset()
        self._frame_count = 0
    
    def close(self) -> None:
        """Release resources."""
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass

"""
Pytest configuration and fixtures for Cognitive Twin tests.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import AppConfig, load_config, AcquisitionConfig, FeaturesConfig, CognitiveConfig
from src.models import (
    MentalState, Trend, FeatureVector, CapturedFrame,
    EyeFeatures, HeadPose, BlinkFeatures, LandmarkData, UserContext
)


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return AppConfig()


@pytest.fixture
def sample_frame():
    """Provide a sample captured frame for testing."""
    # Create a fake 640x480 BGR image
    frame_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return CapturedFrame(
        frame=frame_data,
        timestamp=1234567890.0,
        frame_id=1
    )


@pytest.fixture
def sample_eye_features():
    """Provide sample eye features."""
    return EyeFeatures(
        ear_left=0.28,
        ear_right=0.30,
        ear_average=0.29,
        gaze_x=0.1,
        gaze_y=-0.05,
        eyes_closed=False
    )


@pytest.fixture
def sample_head_pose():
    """Provide sample head pose."""
    return HeadPose(
        pitch=5.0,
        yaw=-3.0,
        roll=1.0,
        confidence=0.95
    )


@pytest.fixture
def sample_blink_features():
    """Provide sample blink features."""
    return BlinkFeatures(
        blink_detected=False,
        blink_rate=15.0,
        avg_blink_duration=0.15,
        time_since_last_blink=2.5
    )


@pytest.fixture
def sample_mental_state():
    """Provide a sample mental state for testing."""
    return MentalState(
        timestamp=1234567890.0,
        focus_level=65.0,
        fatigue_level=30.0,
        cognitive_load=50.0,
        confidence=0.85,
        trend=Trend.STABLE
    )


@pytest.fixture
def sample_feature_vector(sample_eye_features, sample_head_pose, sample_blink_features):
    """Provide a sample feature vector."""
    return FeatureVector(
        timestamp=1234567890.0,
        frame_id=1,
        face_detected=True,
        face_confidence=0.95,
        landmarks=None,  # Skip landmarks for simple tests
        eye_features=sample_eye_features,
        head_pose=sample_head_pose,
        blink_features=sample_blink_features,
        lighting_quality=0.8,
        face_distance=50.0
    )


@pytest.fixture
def sample_user_context(sample_mental_state):
    """Provide sample user context."""
    return UserContext(
        session_duration_minutes=45.0,
        time_of_day="afternoon",
        day_of_week="Wednesday",
        recent_states=[sample_mental_state],
        notification_style="balanced"
    )


@pytest.fixture
def mock_camera():
    """Provide a mock camera manager."""
    camera = MagicMock()
    camera.start.return_value = True
    camera.stop.return_value = None
    camera.get_frame.return_value = None
    return camera

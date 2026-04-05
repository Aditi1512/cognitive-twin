"""
Unit tests for src/models.py data structures.
"""

import pytest
import numpy as np
from src.models import (
    Trend, CapturedFrame, LandmarkData, EyeFeatures, HeadPose,
    BlinkFeatures, FeatureVector, MentalState, UserContext, Explanation, Session
)


class TestTrend:
    """Tests for Trend enum."""
    
    def test_trend_values(self):
        """Test that trend enum has correct values."""
        assert Trend.IMPROVING.value == "improving"
        assert Trend.STABLE.value == "stable"
        assert Trend.DECLINING.value == "declining"


class TestCapturedFrame:
    """Tests for CapturedFrame dataclass."""
    
    def test_create_frame(self, sample_frame):
        """Test creating a captured frame."""
        assert sample_frame.frame.shape == (480, 640, 3)
        assert sample_frame.timestamp == 1234567890.0
        assert sample_frame.frame_id == 1
        assert sample_frame.camera_active is True
    
    def test_resolution(self, sample_frame):
        """Test resolution property."""
        width, height = sample_frame.resolution
        assert width == 640
        assert height == 480
    
    def test_to_rgb(self, sample_frame):
        """Test BGR to RGB conversion."""
        rgb = sample_frame.to_rgb()
        assert rgb.shape == sample_frame.frame.shape


class TestEyeFeatures:
    """Tests for EyeFeatures dataclass."""
    
    def test_create_eye_features(self, sample_eye_features):
        """Test creating eye features."""
        assert sample_eye_features.ear_left == 0.28
        assert sample_eye_features.ear_right == 0.30
        assert sample_eye_features.ear_average == 0.29
        assert sample_eye_features.eyes_closed is False
    
    def test_to_array(self, sample_eye_features):
        """Test converting eye features to array."""
        arr = sample_eye_features.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 6  # ear_left, ear_right, ear_avg, gaze_x, gaze_y, eyes_closed


class TestHeadPose:
    """Tests for HeadPose dataclass."""
    
    def test_create_head_pose(self, sample_head_pose):
        """Test creating head pose."""
        assert sample_head_pose.pitch == 5.0
        assert sample_head_pose.yaw == -3.0
        assert sample_head_pose.roll == 1.0
        assert sample_head_pose.confidence == 0.95
    
    def test_to_array(self, sample_head_pose):
        """Test converting head pose to array."""
        arr = sample_head_pose.to_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3  # pitch, yaw, roll (normalized)


class TestBlinkFeatures:
    """Tests for BlinkFeatures dataclass."""
    
    def test_create_blink_features(self, sample_blink_features):
        """Test creating blink features."""
        assert sample_blink_features.blink_detected is False
        assert sample_blink_features.blink_rate == 15.0
        assert sample_blink_features.avg_blink_duration == 0.15
    
    def test_to_array(self, sample_blink_features):
        """Test converting blink features to array."""
        arr = sample_blink_features.to_array()
        assert isinstance(arr, np.ndarray)


class TestMentalState:
    """Tests for MentalState dataclass."""
    
    def test_create_mental_state(self, sample_mental_state):
        """Test creating mental state."""
        assert sample_mental_state.focus_level == 65.0
        assert sample_mental_state.fatigue_level == 30.0
        assert sample_mental_state.cognitive_load == 50.0
        assert sample_mental_state.trend == Trend.STABLE
    
    def test_to_dict(self, sample_mental_state):
        """Test converting mental state to dictionary."""
        d = sample_mental_state.to_dict()
        assert "focus_level" in d
        assert "fatigue_level" in d
        assert "cognitive_load" in d
        assert d["focus_level"] == 65.0
    
    def test_get_primary_concern_none(self, sample_mental_state):
        """Test that good state has no primary concern."""
        concern = sample_mental_state.get_primary_concern()
        assert concern is None  # Focus is good, fatigue is low
    
    def test_get_primary_concern_fatigue(self):
        """Test that high fatigue is flagged as concern."""
        state = MentalState(
            timestamp=0,
            focus_level=70.0,
            fatigue_level=75.0,  # High fatigue
            cognitive_load=50.0,
            confidence=0.9
        )
        concern = state.get_primary_concern()
        assert concern is not None  # High fatigue should be flagged
    
    def test_get_primary_concern_focus(self):
        """Test that low focus is flagged as concern."""
        state = MentalState(
            timestamp=0,
            focus_level=25.0,  # Low focus
            fatigue_level=30.0,
            cognitive_load=50.0,
            confidence=0.9
        )
        concern = state.get_primary_concern()
        assert concern is not None  # Low focus should be flagged
    
    def test_get_status_emoji(self, sample_mental_state):
        """Test status emoji generation."""
        emoji = sample_mental_state.get_status_emoji()
        assert isinstance(emoji, str)
        assert len(emoji) > 0


class TestFeatureVector:
    """Tests for FeatureVector dataclass."""
    
    def test_create_feature_vector(self, sample_feature_vector):
        """Test creating feature vector."""
        assert sample_feature_vector.face_detected is True
        assert sample_feature_vector.face_confidence == 0.95
    
    def test_to_array(self, sample_feature_vector):
        """Test converting feature vector to array."""
        arr = sample_feature_vector.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (32,)  # Should produce 32D vector
    
    def test_to_array_no_face(self):
        """Test array output when no face detected."""
        fv = FeatureVector(
            timestamp=0,
            frame_id=0,
            face_detected=False,
            face_confidence=0.0
        )
        arr = fv.to_array()
        assert isinstance(arr, np.ndarray)
        # Should still produce valid array with defaults


class TestUserContext:
    """Tests for UserContext dataclass."""
    
    def test_create_user_context(self, sample_user_context):
        """Test creating user context."""
        assert sample_user_context.session_duration_minutes == 45.0
        assert sample_user_context.time_of_day == "afternoon"
    
    def test_get_pattern_summary(self, sample_user_context):
        """Test pattern summary generation."""
        summary = sample_user_context.get_pattern_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestSession:
    """Tests for Session dataclass."""
    
    def test_create_session(self):
        """Test creating a session."""
        session = Session(
            id="test-123",
            start_time=1000.0,
            end_time=2000.0,
            avg_focus=65.0,
            avg_fatigue=30.0,
            avg_cognitive_load=50.0,
            total_frames=1000,
            explanations_count=5
        )
        assert session.id == "test-123"
        assert session.total_frames == 1000
    
    def test_duration_minutes(self):
        """Test session duration calculation."""
        session = Session(
            id="test",
            start_time=0.0,
            end_time=600.0  # 10 minutes in seconds
        )
        assert session.duration_minutes() == 10.0
    
    def test_to_dict(self):
        """Test converting session to dictionary."""
        session = Session(id="test", start_time=0.0)
        d = session.to_dict()
        assert "id" in d
        assert "start_time" in d

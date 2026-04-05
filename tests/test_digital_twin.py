"""
Unit tests for src/cognitive/digital_twin.py cognitive modeling.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from src.cognitive.digital_twin import CognitiveDigitalTwin
from src.config import CognitiveConfig
from src.models import FeatureVector, MentalState, Trend


class TestCognitiveDigitalTwin:
    """Tests for CognitiveDigitalTwin class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Provide a temporary directory for twin data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def twin(self, temp_data_dir):
        """Provide a digital twin instance."""
        return CognitiveDigitalTwin(
            user_id="test-user",
            config=CognitiveConfig(),
            data_dir=temp_data_dir
        )
    
    def test_create_twin(self, twin):
        """Test creating a digital twin."""
        assert twin.user_id == "test-user"
        assert twin.config is not None
        assert twin.temporal_model is not None
    
    def test_twin_generates_unique_id_if_not_provided(self, temp_data_dir):
        """Test that twin generates ID if not provided."""
        twin = CognitiveDigitalTwin(data_dir=temp_data_dir)
        assert twin.user_id is not None
        assert len(twin.user_id) > 0
    
    def test_update_requires_features(self, twin):
        """Test updating twin with feature vector."""
        # Create minimal feature vector
        fv = FeatureVector(
            timestamp=1234567890.0,
            frame_id=1,
            face_detected=True,
            face_confidence=0.9
        )
        
        # First update may return None (needs sequence buffer to fill)
        result = twin.update(fv)
        # Initially should be None until sequence builds up
        # This is expected behavior
        assert result is None or isinstance(result, MentalState)
    
    def test_update_multiple_frames(self, twin, sample_feature_vector):
        """Test updating with multiple frames builds sequence."""
        # Feed many frames to build up sequence
        for i in range(70):  # More than sequence_length (60)
            fv = FeatureVector(
                timestamp=1234567890.0 + i * 0.033,  # ~30fps
                frame_id=i,
                face_detected=True,
                face_confidence=0.9
            )
            result = twin.update(fv)
        
        # After enough frames, should get mental state
        assert result is not None
        assert isinstance(result, MentalState)
        assert 0 <= result.focus_level <= 100
        assert 0 <= result.fatigue_level <= 100
    
    def test_get_state_initially_none(self, twin):
        """Test that get_state returns None before updates."""
        assert twin.get_state() is None
    
    def test_get_patterns(self, twin):
        """Test getting learned patterns."""
        patterns = twin.get_patterns()
        assert "focus_baseline" in patterns
        assert "fatigue_baseline" in patterns
        assert "sample_count" in patterns
    
    def test_save_and_load_twin(self, temp_data_dir):
        """Test saving and loading twin state."""
        # Create and modify twin
        twin1 = CognitiveDigitalTwin(
            user_id="persist-test",
            data_dir=temp_data_dir
        )
        
        # Manually update patterns to test persistence
        twin1._patterns["focus_baseline"] = 75.0
        twin1._patterns["sample_count"] = 100
        twin1.save()
        
        # Create new twin with same ID - should load saved data
        twin2 = CognitiveDigitalTwin(
            user_id="persist-test",
            data_dir=temp_data_dir
        )
        
        assert twin2._patterns["focus_baseline"] == 75.0
        assert twin2._patterns["sample_count"] == 100
    
    def test_reset_twin(self, twin):
        """Test resetting twin to initial state."""
        # Modify state
        twin._patterns["sample_count"] = 500
        
        # Reset
        twin.reset()
        
        # Verify reset
        assert twin._patterns["sample_count"] == 0
        assert len(twin._state_history) == 0
    
    def test_get_recent_states(self, twin):
        """Test getting recent state history."""
        # Initially empty
        assert len(twin.get_recent_states(10)) == 0
        
        # Add some states manually for testing
        for i in range(5):
            state = MentalState(
                timestamp=float(i),
                focus_level=60.0 + i,
                fatigue_level=30.0,
                cognitive_load=50.0,
                confidence=0.9
            )
            twin._state_history.append(state)
        
        recent = twin.get_recent_states(3)
        assert len(recent) == 3
        # Should get most recent
        assert recent[-1].focus_level == 64.0


class TestTrendComputation:
    """Tests for trend computation logic."""
    
    @pytest.fixture
    def twin_with_history(self, tmp_path):
        """Provide twin with state history."""
        twin = CognitiveDigitalTwin(user_id="trend-test", data_dir=tmp_path)
        return twin
    
    def test_trend_stable_with_few_samples(self, twin_with_history):
        """Test that trend is stable with few samples."""
        # Add just a few states
        for i in range(5):
            state = MentalState(
                timestamp=float(i),
                focus_level=60.0,
                fatigue_level=30.0,
                cognitive_load=50.0,
                confidence=0.9
            )
            twin_with_history._state_history.append(state)
        
        trend = twin_with_history._compute_trend()
        assert trend == Trend.STABLE
    
    def test_trend_improving(self, twin_with_history):
        """Test detecting improving trend."""
        # Add states with improving focus
        for i in range(15):
            focus = 40.0 + i * 2  # 40, 42, 44, ... 68
            state = MentalState(
                timestamp=float(i),
                focus_level=focus,
                fatigue_level=30.0,
                cognitive_load=50.0,
                confidence=0.9
            )
            twin_with_history._state_history.append(state)
        
        trend = twin_with_history._compute_trend()
        assert trend == Trend.IMPROVING
    
    def test_trend_declining(self, twin_with_history):
        """Test detecting declining trend."""
        # Add states with declining focus
        for i in range(15):
            focus = 80.0 - i * 2  # 80, 78, 76, ... 52
            state = MentalState(
                timestamp=float(i),
                focus_level=focus,
                fatigue_level=30.0,
                cognitive_load=50.0,
                confidence=0.9
            )
            twin_with_history._state_history.append(state)
        
        trend = twin_with_history._compute_trend()
        assert trend == Trend.DECLINING

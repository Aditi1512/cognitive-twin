"""
Unit tests for src/config.py configuration management.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.config import (
    AppConfig, AcquisitionConfig, FeaturesConfig, CognitiveConfig,
    EstimationConfig, GenAIConfig, UIConfig, PrivacyConfig,
    load_config, get_config, set_config
)


class TestConfigDataClasses:
    """Tests for configuration dataclasses."""
    
    def test_acquisition_config_defaults(self):
        """Test default acquisition config values."""
        config = AcquisitionConfig()
        assert config.camera_id == 0
        assert config.frame_width == 640
        assert config.frame_height == 480
        assert config.target_fps == 30
    
    def test_features_config_defaults(self):
        """Test default features config values."""
        config = FeaturesConfig()
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.enable_attention_mesh is True
    
    def test_cognitive_config_defaults(self):
        """Test default cognitive config values."""
        config = CognitiveConfig()
        assert config.sequence_length == 60
        assert config.model_type == "lstm"
        assert config.hidden_size == 128
        assert config.personalization_enabled is True
    
    def test_estimation_config_defaults(self):
        """Test default estimation config values."""
        config = EstimationConfig()
        assert config.update_interval_seconds == 1.0
        assert config.smoothing_factor == 0.3
    
    def test_genai_config_defaults(self):
        """Test default GenAI config values."""
        config = GenAIConfig()
        assert config.provider == "openrouter"
        assert "llama" in config.model.lower()
        assert config.max_tokens == 150
    
    def test_privacy_config_defaults(self):
        """Test privacy-first defaults."""
        config = PrivacyConfig()
        assert config.store_raw_video is False
        assert config.store_landmarks is False
        assert config.anonymize_cloud_requests is True
    
    def test_app_config_complete(self):
        """Test that AppConfig includes all sub-configs."""
        config = AppConfig()
        assert hasattr(config, 'acquisition')
        assert hasattr(config, 'features')
        assert hasattr(config, 'cognitive')
        assert hasattr(config, 'estimation')
        assert hasattr(config, 'genai')
        assert hasattr(config, 'ui')
        assert hasattr(config, 'privacy')


class TestLoadConfig:
    """Tests for configuration loading."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_config()
        assert isinstance(config, AppConfig)
        assert config.acquisition.camera_id == 0
    
    def test_load_custom_config_file(self):
        """Test loading from a custom config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            custom_config = {
                "acquisition": {
                    "camera_id": 2,
                    "frame_width": 1280,
                    "target_fps": 15
                },
                "genai": {
                    "provider": "ollama",
                    "model": "llama3"
                }
            }
            json.dump(custom_config, f)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            assert config.acquisition.camera_id == 2
            assert config.acquisition.frame_width == 1280
            assert config.acquisition.target_fps == 15
            assert config.genai.provider == "ollama"
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_config_uses_defaults(self):
        """Test that nonexistent config path uses defaults."""
        config = load_config("/nonexistent/path/config.json")
        # Should fall back to defaults without error
        assert isinstance(config, AppConfig)
        assert config.acquisition.camera_id == 0


class TestGlobalConfig:
    """Tests for global configuration management."""
    
    def test_get_and_set_config(self):
        """Test getting and setting global config."""
        custom = AppConfig(
            acquisition=AcquisitionConfig(camera_id=5)
        )
        set_config(custom)
        
        retrieved = get_config()
        assert retrieved.acquisition.camera_id == 5
    
    def test_get_config_lazy_loads(self):
        """Test that get_config loads config if not set."""
        # Reset global state
        import src.config as config_module
        config_module._config = None
        
        config = get_config()
        assert isinstance(config, AppConfig)

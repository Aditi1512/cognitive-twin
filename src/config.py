"""
Configuration management for Personal Cognitive Digital Twin.

Loads settings from config files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AcquisitionConfig(BaseModel):
    """Camera and video acquisition settings."""
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30


class FeaturesConfig(BaseModel):
    """Feature extraction settings."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_attention_mesh: bool = True


class CognitiveConfig(BaseModel):
    """Cognitive modeling settings."""
    sequence_length: int = 60
    model_type: str = "lstm"
    hidden_size: int = 128
    num_layers: int = 2
    personalization_enabled: bool = True
    adaptation_rate: float = 0.01


class EstimationConfig(BaseModel):
    """Mental state estimation settings."""
    update_interval_seconds: float = 1.0
    smoothing_factor: float = 0.3
    trend_window_size: int = 30


class GenAIConfig(BaseModel):
    """Generative AI settings."""
    provider: str = "openrouter"
    model: str = "meta-llama/llama-3.3-70b-instruct:free"
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 150
    temperature: float = 0.7
    explanation_interval_seconds: int = 30
    
    model_config = {"extra": "allow"}


class UIConfig(BaseModel):
    """User interface settings."""
    notification_enabled: bool = True
    notification_interval_seconds: int = 60
    enable_sounds: bool = False
    theme: str = "dark"
    dashboard_refresh_rate: float = 1.0


class PrivacyConfig(BaseModel):
    """Privacy and data settings."""
    store_raw_video: bool = False
    store_landmarks: bool = False
    data_retention_days: int = 30
    local_processing_only: bool = False
    anonymize_cloud_requests: bool = True


class AppConfig(BaseModel):
    """Complete application configuration."""
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    cognitive: CognitiveConfig = Field(default_factory=CognitiveConfig)
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    genai: GenAIConfig = Field(default_factory=GenAIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


class Settings(BaseSettings):
    """Environment-based settings."""
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    log_level: str = "INFO"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from file and environment.
    
    Args:
        config_path: Path to config file. Defaults to config/default.json
        
    Returns:
        AppConfig: Complete application configuration
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "default.json"
    else:
        config_path = Path(config_path)
    
    # Load from file
    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
    else:
        config_data = {}
    
    # Create config object
    config = AppConfig(**config_data)
    
    # Override with environment variables
    settings = Settings()
    
    # Update genai config with API key from environment
    if settings.openrouter_api_key:
        updated_genai = config.genai.model_copy(update={"api_key": settings.openrouter_api_key})
        config = config.model_copy(update={"genai": updated_genai})
    elif settings.openai_api_key and config.genai.provider == "openai":
        updated_genai = config.genai.model_copy(update={"api_key": settings.openai_api_key})
        config = config.model_copy(update={"genai": updated_genai})
    
    return config


# Global config instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config

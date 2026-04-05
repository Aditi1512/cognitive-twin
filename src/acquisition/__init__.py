"""Data acquisition layer - camera and consent management."""

from .camera import CameraManager
from .consent import ConsentManager

__all__ = ["CameraManager", "ConsentManager"]

"""
Camera management for webcam capture.

Handles video capture with proper resource management.
"""

import time
from typing import Optional
import cv2
import numpy as np

from ..models import CapturedFrame
from ..config import AcquisitionConfig


class CameraManager:
    """Manages webcam capture with proper resource handling."""
    
    def __init__(self, config: Optional[AcquisitionConfig] = None):
        """
        Initialize camera manager.
        
        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        if config is None:
            config = AcquisitionConfig()
        
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._is_active = False
        self._last_frame_time = 0.0
        self._target_interval = 1.0 / config.target_fps
    
    def start(self) -> bool:
        """
        Start video capture.
        
        Returns:
            True if camera started successfully, False otherwise.
        """
        if self._is_active:
            return True
        
        try:
            self._cap = cv2.VideoCapture(self.config.camera_id)
            
            if not self._cap.isOpened():
                return False
            
            # Set camera properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            
            # Verify camera is working
            ret, _ = self._cap.read()
            if not ret:
                self._cap.release()
                self._cap = None
                return False
            
            self._is_active = True
            self._frame_id = 0
            return True
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_active = False
    
    def get_frame(self) -> Optional[CapturedFrame]:
        """
        Get next frame from camera.
        
        Returns:
            CapturedFrame if successful, None if camera not active or error.
        """
        if not self._is_active or self._cap is None:
            return None
        
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            return None
        
        current_time = time.time()
        
        captured = CapturedFrame(
            frame=frame,
            timestamp=current_time,
            frame_id=self._frame_id,
            camera_active=True
        )
        
        self._frame_id += 1
        self._last_frame_time = current_time
        
        return captured
    
    def is_active(self) -> bool:
        """Check if camera is currently capturing."""
        return self._is_active
    
    @property
    def fps(self) -> float:
        """Get current actual FPS."""
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    @property
    def resolution(self) -> tuple:
        """Get current camera resolution."""
        if self._cap is None:
            return (0, 0)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def test_camera(camera_id: int = 0) -> bool:
    """
    Test if camera is available and working.
    
    Args:
        camera_id: Camera device ID to test.
        
    Returns:
        True if camera works, False otherwise.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    return ret and frame is not None


def list_cameras(max_cameras: int = 10) -> list:
    """
    List available cameras.
    
    Args:
        max_cameras: Maximum number of cameras to check.
        
    Returns:
        List of available camera IDs.
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

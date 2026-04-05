"""
Consent management for data collection.

Handles user consent for camera access and data processing.
"""

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class ConsentRecord:
    """Record of user consent."""
    granted: bool
    timestamp: float
    version: str = "1.0"
    camera_access: bool = False
    data_storage: bool = False
    cloud_processing: bool = False
    
    def is_valid(self, max_age_days: int = 365) -> bool:
        """Check if consent is still valid."""
        if not self.granted:
            return False
        age_seconds = time.time() - self.timestamp
        return age_seconds < (max_age_days * 24 * 60 * 60)


class ConsentManager:
    """Manages user consent for data collection and processing."""
    
    CONSENT_FILE = "consent.json"
    CURRENT_VERSION = "1.0"
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize consent manager.
        
        Args:
            data_dir: Directory to store consent records.
        """
        if data_dir is None:
            data_dir = Path.home() / ".cognitive_twin"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.consent_path = self.data_dir / self.CONSENT_FILE
        self._consent: Optional[ConsentRecord] = None
        
        self._load_consent()
    
    def _load_consent(self) -> None:
        """Load consent record from file."""
        if self.consent_path.exists():
            try:
                with open(self.consent_path, "r") as f:
                    data = json.load(f)
                self._consent = ConsentRecord(**data)
            except Exception:
                self._consent = None
    
    def _save_consent(self) -> None:
        """Save consent record to file."""
        if self._consent is not None:
            with open(self.consent_path, "w") as f:
                json.dump(asdict(self._consent), f, indent=2)
    
    def has_consent(self) -> bool:
        """Check if user has granted consent."""
        if self._consent is None:
            return False
        return self._consent.is_valid()
    
    def has_camera_consent(self) -> bool:
        """Check if user has granted camera access."""
        return self.has_consent() and self._consent.camera_access
    
    def has_cloud_consent(self) -> bool:
        """Check if user has granted cloud processing."""
        return self.has_consent() and self._consent.cloud_processing
    
    def grant_consent(
        self,
        camera_access: bool = True,
        data_storage: bool = True,
        cloud_processing: bool = False
    ) -> ConsentRecord:
        """
        Grant consent with specified permissions.
        
        Args:
            camera_access: Allow camera access.
            data_storage: Allow local data storage.
            cloud_processing: Allow cloud GenAI processing.
            
        Returns:
            The consent record.
        """
        self._consent = ConsentRecord(
            granted=True,
            timestamp=time.time(),
            version=self.CURRENT_VERSION,
            camera_access=camera_access,
            data_storage=data_storage,
            cloud_processing=cloud_processing
        )
        self._save_consent()
        return self._consent
    
    def revoke_consent(self) -> None:
        """Revoke all consent."""
        self._consent = ConsentRecord(
            granted=False,
            timestamp=time.time(),
            version=self.CURRENT_VERSION,
            camera_access=False,
            data_storage=False,
            cloud_processing=False
        )
        self._save_consent()
    
    def get_consent(self) -> Optional[ConsentRecord]:
        """Get current consent record."""
        return self._consent
    
    def get_consent_text(self) -> str:
        """Get the consent text to display to users."""
        return """
Personal Cognitive Digital Twin - Privacy & Consent

This application uses your webcam to analyze your mental state and provide 
personalized feedback. Please review the following:

📹 CAMERA ACCESS
- Your webcam will capture video in real-time
- Raw video is NEVER stored - only processed features are kept
- Camera access can be revoked at any time

💾 DATA STORAGE
- Processed features and mental state scores are stored locally
- Session history is kept for your personal insights
- All data remains on your device by default

☁️ CLOUD PROCESSING (Optional)
- Mental state summaries may be sent to AI services for natural language feedback
- No video or images are ever transmitted
- You can use fully local processing instead

🔒 YOUR RIGHTS
- Access: View all your data anytime
- Export: Download your data in standard formats
- Delete: Remove all your data with one click
- Revoke: Stop all processing immediately

By proceeding, you consent to the selected processing activities.
"""

"""
Main entry point for Personal Cognitive Digital Twin.

Provides command-line interface and orchestrates the monitoring pipeline.
"""

import argparse
import time
import signal
import sys
from typing import Optional
from datetime import datetime

from .config import load_config, get_config, AppConfig
from .acquisition.camera import CameraManager, test_camera
from .acquisition.consent import ConsentManager
from .features.extractor import FeatureExtractor
from .cognitive.digital_twin import CognitiveDigitalTwin
from .estimation.estimator import MentalStateEstimator
from .explanation.generator import ExplanationGenerator, get_time_of_day, get_day_of_week
from .models import UserContext


class CognitiveTwinApp:
    """Main application orchestrator."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the application.
        
        Args:
            config: Application configuration.
        """
        self.config = config or get_config()
        
        # Initialize components
        self.consent_manager = ConsentManager()
        self.camera: Optional[CameraManager] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.digital_twin: Optional[CognitiveDigitalTwin] = None
        self.state_estimator: Optional[MentalStateEstimator] = None
        self.explanation_generator: Optional[ExplanationGenerator] = None
        
        # State
        self.running = False
        self.session_start: Optional[float] = None
        self.frame_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down...")
        self.stop()
        sys.exit(0)
    
    def check_consent(self) -> bool:
        """Check and obtain user consent."""
        if not self.consent_manager.has_consent():
            print("\n" + "=" * 60)
            print(self.consent_manager.get_consent_text())
            print("=" * 60)
            
            response = input("\nDo you consent? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                self.consent_manager.grant_consent(
                    camera_access=True,
                    data_storage=True,
                    cloud_processing=not self.config.privacy.local_processing_only
                )
                print("✅ Consent granted. Thank you!")
                return True
            else:
                print("❌ Consent not granted. Exiting.")
                return False
        
        return True
    
    def initialize(self) -> bool:
        """Initialize all components."""
        print("🔧 Initializing components...")
        
        # Check camera
        if not test_camera(self.config.acquisition.camera_id):
            print("❌ Camera not available. Please check your webcam connection.")
            return False
        
        # Initialize camera
        self.camera = CameraManager(self.config.acquisition)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.config.features)
        print("  ✓ Feature extractor ready")
        
        # Initialize digital twin
        self.digital_twin = CognitiveDigitalTwin(config=self.config.cognitive)
        print("  ✓ Digital twin initialized")
        
        # Initialize state estimator
        self.state_estimator = MentalStateEstimator(self.config.estimation)
        print("  ✓ State estimator ready")
        
        # Initialize explanation generator
        self.explanation_generator = ExplanationGenerator(self.config.genai)
        if self.explanation_generator.is_available:
            print(f"  ✓ GenAI ready ({self.config.genai.provider})")
        else:
            print(f"  ⚠ GenAI not configured (will use fallback messages)")
        
        print("✅ All components initialized!")
        return True
    
    def start(self) -> bool:
        """Start the monitoring session."""
        if not self.camera:
            print("❌ Application not initialized. Call initialize() first.")
            return False
        
        if not self.camera.start():
            print("❌ Failed to start camera.")
            return False
        
        self.running = True
        self.session_start = time.time()
        self.frame_count = 0
        
        print("\n🚀 Monitoring started!")
        print("   Press Ctrl+C to stop.\n")
        
        return True
    
    def stop(self):
        """Stop the monitoring session."""
        self.running = False
        
        if self.camera:
            self.camera.stop()
        
        if self.digital_twin:
            self.digital_twin.save()
        
        if self.session_start:
            duration = (time.time() - self.session_start) / 60
            print(f"\n📊 Session Summary:")
            print(f"   Duration: {duration:.1f} minutes")
            print(f"   Frames processed: {self.frame_count}")
    
    def run_loop(self):
        """Main processing loop."""
        last_state_time = 0
        last_explanation_time = 0
        
        while self.running:
            loop_start = time.time()
            
            # Capture frame
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            
            # Extract features
            features = self.feature_extractor.extract(frame)
            
            # Update digital twin
            mental_state = self.digital_twin.update(features)
            
            if mental_state is None:
                continue
            
            # Smooth state
            mental_state = self.state_estimator.smooth(mental_state)
            
            # Update display (every second)
            current_time = time.time()
            if current_time - last_state_time >= 1.0:
                self._display_state(mental_state)
                last_state_time = current_time
            
            # Generate explanation if needed
            if self.explanation_generator.should_generate(mental_state):
                context = self._build_context(mental_state)
                explanation = self.explanation_generator.generate(mental_state, context)
                self._display_explanation(explanation)
                last_explanation_time = current_time
            
            # Maintain frame rate
            elapsed = time.time() - loop_start
            target_interval = 1.0 / self.config.acquisition.target_fps
            sleep_time = max(0, target_interval - elapsed)
            time.sleep(sleep_time)
    
    def _build_context(self, mental_state) -> UserContext:
        """Build user context for explanation generation."""
        session_duration = 0
        if self.session_start:
            session_duration = (time.time() - self.session_start) / 60
        
        return UserContext(
            session_duration_minutes=session_duration,
            time_of_day=get_time_of_day(),
            day_of_week=get_day_of_week(),
            recent_states=self.digital_twin.get_recent_states(10),
            notification_style="balanced"
        )
    
    def _display_state(self, state):
        """Display current mental state in terminal."""
        emoji = state.get_status_emoji()
        trend = {"improving": "📈", "stable": "➡️", "declining": "📉"}.get(
            state.trend.value, "➡️"
        )
        
        print(f"\r{emoji} Focus: {state.focus_level:5.1f}% | "
              f"Fatigue: {state.fatigue_level:5.1f}% | "
              f"Load: {state.cognitive_load:5.1f}% | "
              f"Trend: {trend} ", end="", flush=True)
    
    def _display_explanation(self, explanation):
        """Display AI explanation."""
        print(f"\n\n💬 {explanation.message}")
        if explanation.suggestion:
            print(f"   💡 Suggestion: {explanation.suggestion}")
        print()


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Personal Cognitive Digital Twin - Mental State Monitor"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch Streamlit dashboard instead of CLI"
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="Test camera availability and exit"
    )
    
    args = parser.parse_args()
    
    # Test camera mode
    if args.test_camera:
        print("Testing camera...")
        if test_camera():
            print("✅ Camera is available and working!")
        else:
            print("❌ Camera not available.")
        return
    
    # Dashboard mode
    if args.dashboard:
        import subprocess
        import os
        dashboard_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.py")
        subprocess.run(["streamlit", "run", dashboard_path])
        return
    
    # CLI mode
    print("\n" + "=" * 60)
    print("    🧠 Personal Cognitive Digital Twin")
    print("    Real-time Mental State Monitoring")
    print("=" * 60 + "\n")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create and run application
    app = CognitiveTwinApp(config)
    
    # Check consent
    if not app.check_consent():
        return
    
    # Initialize
    if not app.initialize():
        return
    
    # Start monitoring
    if not app.start():
        return
    
    # Run main loop
    try:
        app.run_loop()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()

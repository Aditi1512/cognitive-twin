"""
Personal Cognitive Digital Twin - Premium Dashboard
Real-time mental state monitoring with webcam integration.
"""

import time
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from collections import deque
from typing import Optional
import sys
from pathlib import Path
import random
import io
from PIL import Image
import logging
import threading
import queue
import av

# Configure logging to print to stderr
logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# WebRTC import with fallback
WEBRTC_AVAILABLE = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    WEBRTC_AVAILABLE = True
    logger.info("streamlit-webrtc loaded successfully")
except ImportError:
    logger.warning("streamlit-webrtc not available, falling back to snapshot mode")

# Auto-refresh for real-time updates
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    st_autorefresh = None

from src.config import load_config
from src.cognitive.digital_twin import CognitiveDigitalTwin
from src.estimation.estimator import MentalStateEstimator
from src.explanation.generator import ExplanationGenerator, get_time_of_day, get_day_of_week
from src.features.extractor import FeatureExtractor
from src.models import UserContext, MentalState, Trend, FeatureVector, CapturedFrame, EyeFeatures, HeadPose, BlinkFeatures


# Shared state container for cross-thread communication
# This object persists across Streamlit reruns because it's a module-level singleton
class SharedState:
    """Thread-safe shared state between WebRTC processor and Streamlit main thread."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.last_features = None
                    cls._instance.frame_count = 0
                    cls._instance.face_detected = False
                    cls._instance.is_real_detection = False
                    cls._instance.last_update = 0
                    cls._instance.feature_extractor = None
                    cls._instance.captured_features_list = []  # List to collect features during capture
                    cls._instance.capture_active = False  # Flag to indicate if capture is active
        return cls._instance
    
    def set_capture_active(self, active: bool):
        """Set capture active state."""
        with self._lock:
            self.capture_active = active
            if not active:
                # Clear list when capture stops
                self.captured_features_list = []
    
    def update(self, features, frame_count, is_real):
        with self._lock:
            self.last_features = features
            self.frame_count = frame_count
            self.face_detected = features.face_detected if features else False
            self.is_real_detection = is_real
            self.last_update = time.time()
            # Collect features for analysis if capture is active and face is detected
            if self.capture_active and features and features.face_detected:
                self.captured_features_list.append(features)
    
    def clear_captured_features(self):
        """Clear the captured features list."""
        with self._lock:
            self.captured_features_list = []
    
    def get_captured_features(self):
        """Get all captured features."""
        with self._lock:
            return list(self.captured_features_list)
    
    def get_state(self):
        with self._lock:
            return {
                'features': self.last_features,
                'frame_count': self.frame_count,
                'face_detected': self.face_detected,
                'is_real': self.is_real_detection,
                'last_update': self.last_update
            }

# Global shared state singleton
shared_state = SharedState()


# Define the video processor only if WebRTC is available
if WEBRTC_AVAILABLE:
    class FaceAnalysisProcessor(VideoProcessorBase):
        """
        WebRTC Video Processor for real-time face analysis.
        Processes each frame through MediaPipe and extracts facial features.
        
        Results are stored as instance attributes for direct access from main thread.
        """
        
        def __init__(self):
            self._feature_extractor = None
            self._init_lock = threading.Lock()
            
            # Public attributes - accessible from main thread via webrtc_ctx.video_processor
            self.frame_count = 0
            self.face_detected = False
            self.is_real_detection = False
            self.last_features: Optional[FeatureVector] = None
            self.last_state: Optional[MentalState] = None
            self.last_update_time = 0.0
            
        def _ensure_extractor(self):
            """Lazily initialize the feature extractor in the processing thread."""
            if self._feature_extractor is None:
                with self._init_lock:
                    if self._feature_extractor is None:
                        try:
                            from src.config import load_config
                            config = load_config()
                            self._feature_extractor = FeatureExtractor(config.features)
                            self.is_real_detection = self._feature_extractor.is_real_detection
                            logger.info(f"Feature extractor initialized. Real detection: {self.is_real_detection}")
                        except Exception as e:
                            logger.error(f"Failed to initialize feature extractor: {e}")
        
        def _calculate_mental_state(self, features: FeatureVector) -> MentalState:
            """Calculate mental state from features - simplified real-time estimation."""
            focus = 50.0
            fatigue = 30.0
            load = 50.0
            
            if features.eye_features:
                ef = features.eye_features
                # EAR-based focus/fatigue
                ear_norm = max(0, min(1, (ef.ear_average - 0.15) / 0.20))
                gaze_stability = max(0, 1 - min(1, (ef.gaze_x**2 + ef.gaze_y**2)**0.5 / 0.3))
                
                focus = 30 + 50 * (ear_norm * 0.6 + gaze_stability * 0.4)
                fatigue = 20 + 50 * (1 - ear_norm) if not ef.eyes_closed else 75.0
            
            if features.head_pose:
                hp = features.head_pose
                head_movement = (hp.pitch**2 + hp.yaw**2 + hp.roll**2)**0.5
                head_stability = max(0, 1 - min(1, head_movement / 30.0))
                focus = max(20, min(90, focus + (head_stability - 0.5) * 20))
                load = 40 + head_movement * 0.5
            
            return MentalState(
                timestamp=features.timestamp,
                focus_level=max(15, min(95, focus)),
                fatigue_level=max(10, min(85, fatigue)),
                cognitive_load=max(20, min(85, load)),
                confidence=features.face_confidence,
                trend=Trend.STABLE
            )
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            """Process each video frame."""
            self._ensure_extractor()
            
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            captured_frame = CapturedFrame(
                frame=img,
                timestamp=time.time(),
                frame_id=self.frame_count
            )
            
            if self._feature_extractor is not None:
                try:
                    features = self._feature_extractor.extract(captured_frame)
                    
                    # Store results as instance attributes (accessible from main thread)
                    self.last_features = features
                    self.face_detected = features.face_detected
                    self.last_update_time = time.time()
                    
                    if features.face_detected:
                        self.last_state = self._calculate_mental_state(features)
                        img = self._draw_face_overlay(img, features)
                    else:
                        cv2.putText(img, "No Face Detected", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 113, 113), 2)
                        
                except Exception as e:
                    logger.error(f"Feature extraction error: {e}")
                    cv2.putText(img, f"Error: {str(e)[:30]}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 113, 113), 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        def _draw_face_overlay(self, img: np.ndarray, features: FeatureVector) -> np.ndarray:
            """Draw face detection overlay on frame."""
            if features.landmarks is None:
                return img
            
            h, w = img.shape[:2]
            landmarks = features.landmarks.landmarks
            
            if features.landmarks.face_rect:
                x, y, fw, fh = features.landmarks.face_rect
                cv2.rectangle(img, (x, y), (x + fw, y + fh), (129, 140, 248), 2)
            
            key_points = [33, 133, 159, 145, 362, 263, 386, 374, 1, 61, 291]
            for idx in key_points:
                if idx < len(landmarks):
                    pt = landmarks[idx]
                    cv2.circle(img, (int(pt[0] * w), int(pt[1] * h)), 3, (52, 211, 153), -1)
            
            cv2.putText(img, "Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (52, 211, 153), 2)
            
            # Show real-time metrics on video
            if self.last_state:
                s = self.last_state
                cv2.putText(img, f"Focus: {s.focus_level:.0f}%", (10, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img, f"Fatigue: {s.fatigue_level:.0f}%", (10, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if features.head_pose:
                hp = features.head_pose
                cv2.putText(img, f"Head: P{hp.pitch:.0f} Y{hp.yaw:.0f} R{hp.roll:.0f}", 
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            return img
else:
    FaceAnalysisProcessor = None


# Page config
st.set_page_config(
    page_title="Cognitive Twin",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #08080c;
        --bg-secondary: #0f0f14;
        --bg-card: rgba(255, 255, 255, 0.02);
        --border: rgba(255, 255, 255, 0.06);
        --text-primary: #f0f0f5;
        --text-secondary: #6b7280;
        --accent: #818cf8;
        --accent-glow: rgba(129, 140, 248, 0.15);
        --success: #34d399;
        --warning: #fbbf24;
        --danger: #f87171;
    }
    
    * {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse 80% 50% at 50% -20%, var(--accent-glow), transparent),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(99, 102, 241, 0.08), transparent);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 4rem;
        max-width: 1600px;
    }
    
    /* Brand styling */
    .brand {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        color: var(--accent);
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 300 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0 !important;
        line-height: 1.1 !important;
    }
    
    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    
    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
    }
    
    /* Video placeholder */
    .video-placeholder {
        background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.2) 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    
    .video-placeholder::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: 
            linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.02) 50%, transparent 100%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .camera-icon {
        width: 64px;
        height: 64px;
        border: 2px solid var(--border);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .camera-icon-inner {
        width: 24px;
        height: 24px;
        background: var(--border);
        border-radius: 50%;
    }
    
    /* Active session indicator */
    .session-active {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 20px;
        font-size: 0.8rem;
        color: var(--success);
    }
    
    .session-active::before {
        content: '';
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Insight card */
    .insight-card {
        background: linear-gradient(135deg, rgba(129, 140, 248, 0.08) 0%, rgba(99, 102, 241, 0.04) 100%);
        border: 1px solid rgba(129, 140, 248, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
    }
    
    .insight-text {
        font-size: 0.95rem;
        color: var(--text-primary);
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
        color: white !important;
        border: none !important;
        border-radius: 10px;
        padding: 0.7rem 1.75rem;
        font-weight: 500;
        font-size: 0.875rem;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.35);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Session timer */
    .session-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Privacy bullet points */
    .privacy-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        padding: 0.75rem;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
    }
    
    .privacy-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-top: 5px;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    .privacy-dot.green { background: var(--success); }
    .privacy-dot.purple { background: var(--accent); }
    
    .privacy-title {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.9rem;
        margin-bottom: 2px;
    }
    
    .privacy-desc {
        color: var(--text-secondary);
        font-size: 0.8rem;
        line-height: 1.4;
    }
    
    /* Hide plotly modebar */
    .modebar {
        display: none !important;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }
    
    /* Info text */
    .info-text {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Camera input styling */
    [data-testid="stCameraInput"] > div {
        border-radius: 16px !important;
        overflow: hidden;
    }
    
    [data-testid="stCameraInput"] video {
        border-radius: 16px !important;
    }
    
    [data-testid="stCameraInput"] img {
        border-radius: 16px !important;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-badge.good {
        background: rgba(52, 211, 153, 0.1);
        color: var(--success);
    }
    
    .status-badge.warning {
        background: rgba(251, 191, 36, 0.1);
        color: var(--warning);
    }
    
    .status-badge.alert {
        background: rgba(248, 113, 113, 0.1);
        color: var(--danger);
    }
    
    /* Hide WebRTC default controls */
    [data-testid="stWebRTC"] button {
        display: none !important;
    }
    
    /* Countdown timer styling */
    .countdown-display {
        text-align: center;
        margin: 20px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .countdown-number {
        font-size: 4rem;
        font-weight: 700;
        color: var(--accent);
        line-height: 1;
    }
    
    .countdown-label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 8px;
    }
    
    /* Analysis complete styling */
    .analysis-complete {
        text-align: center;
        margin: 20px 0;
    }
    
    .analysis-complete-icon {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--success);
    }
    
    /* Improve spacing */
    .main .block-container > div {
        margin-bottom: 1.5rem;
    }
    
    /* Ensure buttons don't duplicate */
    .stButton {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'monitoring': False,
        'session_start': None,
        'consent_granted': False,
        'digital_twin': None,
        'state_estimator': None,
        'explanation_generator': None,
        'current_state': None,
        'last_explanation': "Click 'Start Session' to begin monitoring your cognitive state.",
        'explanation_time': 0,
        'focus_history': deque(maxlen=120),
        'fatigue_history': deque(maxlen=120),
        'load_history': deque(maxlen=120),
        'config': None,
        'frame_count': 0,
        'simulation_mode': True,  # True when MediaPipe isn't available
        'feature_extractor': None,
        'last_features': None,  # Last extracted feature vector
        'webrtc_ctx': None,  # WebRTC context
        'face_detected_count': 0,
        'no_face_count': 0,
        'webrtc_available': WEBRTC_AVAILABLE,
        # 5-second capture state
        'capture_start_time': None,
        'capture_duration': 5.0,
        'capture_active': False,
        'captured_features': [],  # List to accumulate features during 5-second window
        'analysis_complete': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def initialize_components():
    """Initialize all processing components."""
    if st.session_state.config is None:
        st.session_state.config = load_config()
    
    config = st.session_state.config
    
    if st.session_state.digital_twin is None:
        st.session_state.digital_twin = CognitiveDigitalTwin(config=config.cognitive)
    
    if st.session_state.state_estimator is None:
        st.session_state.state_estimator = MentalStateEstimator(config.estimation)
    
    if st.session_state.explanation_generator is None:
        st.session_state.explanation_generator = ExplanationGenerator(config.genai)
    
    if st.session_state.feature_extractor is None:
        st.session_state.feature_extractor = FeatureExtractor(config.features)
        # Update simulation mode based on whether real detection is available
        st.session_state.simulation_mode = not st.session_state.feature_extractor.is_real_detection


def create_gauge(value: float, label: str, is_inverse: bool = False) -> go.Figure:
    """Create a minimal arc gauge with visible label."""
    # Color logic
    if is_inverse:
        color = "#34d399" if value <= 35 else ("#fbbf24" if value <= 65 else "#f87171")
    else:
        color = "#34d399" if value >= 65 else ("#fbbf24" if value >= 35 else "#f87171")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'font': {'size': 32, 'color': '#f0f0f5', 'family': 'DM Sans'},
            'suffix': '%',
            'valueformat': '.0f'
        },
        title={
            'text': label,
            'font': {'size': 12, 'color': '#9ca3af', 'family': 'DM Sans'}
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'showticklabels': False},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'shape': 'angular',
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=180,
        margin=dict(l=20, r=20, t=40, b=30)
    )
    
    return fig


def create_sparkline(history: list, color: str = "#818cf8") -> go.Figure:
    """Create a minimal sparkline chart."""
    if len(history) < 2:
        history = [50] * 20
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=list(history)[-30:],
        mode='lines',
        fill='tozeroy',
        line=dict(color=color, width=1.5, shape='spline'),
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.08)'
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=80,
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 100]),
        showlegend=False
    )
    
    return fig


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_from_features(features: FeatureVector) -> MentalState:
    """
    Estimate mental state from real extracted features.
    Uses eye aspect ratio, gaze, head pose, and blink patterns to infer mental state.
    NO RANDOM VALUES - purely deterministic based on actual facial features.
    """
    # Base values - will be adjusted based on actual features
    focus_level = 50.0
    fatigue_level = 30.0  
    cognitive_load = 50.0
    
    if features.eye_features:
        ef = features.eye_features
        
        # Normalize EAR (typical range: 0.15-0.35, closed: <0.2)
        # Higher EAR = more alert/open eyes
        ear_normalized = max(0.0, min(1.0, (ef.ear_average - 0.15) / 0.20))
        
        # Gaze stability: less movement = more focused
        gaze_magnitude = np.sqrt(ef.gaze_x**2 + ef.gaze_y**2)
        gaze_stability = max(0.0, 1.0 - min(1.0, gaze_magnitude / 0.3))
        
        # Focus calculation: combination of EAR and gaze stability
        # EAR contributes 60%, gaze stability 40%
        focus_level = 30 + 50 * (ear_normalized * 0.6 + gaze_stability * 0.4)
        
        # Fatigue: inversely related to EAR, eyes closed = high fatigue
        if ef.eyes_closed:
            fatigue_level = 75.0  # Eyes closed = high fatigue
        else:
            # Lower EAR = more tired
            fatigue_level = 20 + 50 * (1 - ear_normalized)
    
    if features.head_pose:
        hp = features.head_pose
        
        # Head movement magnitude (degrees)
        head_movement = np.sqrt(hp.pitch**2 + hp.yaw**2 + hp.roll**2)
        
        # Stable head (looking forward) = better focus
        # Normal range: 0-15 degrees is stable
        head_stability = max(0.0, 1.0 - min(1.0, head_movement / 30.0))
        
        # Adjust focus based on head stability
        focus_adjustment = (head_stability - 0.5) * 20  # ±10 points
        focus_level = max(20, min(90, focus_level + focus_adjustment))
        
        # Looking away significantly = distracted (lower cognitive load)
        if abs(hp.yaw) > 25 or abs(hp.pitch) > 25:
            cognitive_load = 35.0  # Distracted = lower load
        elif head_movement < 10:
            cognitive_load = 60.0  # Stable and engaged = moderate-high load
        else:
            cognitive_load = 50.0  # Moderate movement = moderate load
    
    if features.blink_features:
        bf = features.blink_features
        
        # Normal blink rate: 15-20 blinks/minute
        # Too high (>25) or too low (<10) indicates issues
        blink_rate = bf.blink_rate
        
        if blink_rate > 25:
            # High blink rate = eye strain/fatigue
            fatigue_level = min(85, fatigue_level + 10)
            cognitive_load = min(80, cognitive_load + 5)
        elif blink_rate < 10:
            # Very low blink rate = intense focus (or staring)
            if fatigue_level < 50:
                focus_level = min(90, focus_level + 8)  # Intense focus
            else:
                fatigue_level = min(85, fatigue_level + 5)  # Staring due to fatigue
        elif 12 <= blink_rate <= 22:
            # Normal blink rate = healthy state
            focus_level = min(85, focus_level + 3)
    
    # Clamp values to reasonable ranges
    return MentalState(
        timestamp=features.timestamp,
        focus_level=max(15, min(90, focus_level)),
        fatigue_level=max(10, min(85, fatigue_level)),
        cognitive_load=max(25, min(80, cognitive_load)),
        confidence=features.face_confidence
    )


def calculate_final_metrics(captured_features: list) -> Optional[MentalState]:
    """
    Calculate final mental state metrics from aggregated features collected during 5-second capture.
    
    Args:
        captured_features: List of FeatureVector objects collected during capture
        
    Returns:
        MentalState with aggregated metrics, or None if insufficient data
    """
    if not captured_features or len(captured_features) == 0:
        return None
    
    # Filter to only features with detected faces
    valid_features = [f for f in captured_features if f.face_detected and f.eye_features]
    
    if len(valid_features) == 0:
        return None
    
    # Aggregate eye features
    ear_values = []
    gaze_x_values = []
    gaze_y_values = []
    eyes_closed_count = 0
    
    # Aggregate head pose
    pitch_values = []
    yaw_values = []
    roll_values = []
    
    # Aggregate blink features
    blink_rates = []
    total_blinks = 0
    
    for features in valid_features:
        if features.eye_features:
            ef = features.eye_features
            ear_values.append(ef.ear_average)
            gaze_x_values.append(ef.gaze_x)
            gaze_y_values.append(ef.gaze_y)
            if ef.eyes_closed:
                eyes_closed_count += 1
        
        if features.head_pose:
            hp = features.head_pose
            pitch_values.append(hp.pitch)
            yaw_values.append(hp.yaw)
            roll_values.append(hp.roll)
        
        if features.blink_features:
            bf = features.blink_features
            if bf.blink_rate > 0:
                blink_rates.append(bf.blink_rate)
            # Count blinks (assuming each frame with blink detected is a blink event)
            # This is approximate - actual blink detection should track transitions
    
    # Create aggregated feature vector
    avg_ear = np.mean(ear_values) if ear_values else 0.25
    avg_gaze_x = np.mean(gaze_x_values) if gaze_x_values else 0.0
    avg_gaze_y = np.mean(gaze_y_values) if gaze_y_values else 0.0
    eyes_closed_ratio = eyes_closed_count / len(valid_features) if valid_features else 0.0
    
    avg_pitch = np.mean(pitch_values) if pitch_values else 0.0
    avg_yaw = np.mean(yaw_values) if yaw_values else 0.0
    avg_roll = np.mean(roll_values) if roll_values else 0.0
    
    avg_blink_rate = np.mean(blink_rates) if blink_rates else 15.0  # Default normal rate
    
    # Create aggregated eye features
    aggregated_eye_features = EyeFeatures(
        ear_left=avg_ear,
        ear_right=avg_ear,
        ear_average=avg_ear,
        gaze_x=avg_gaze_x,
        gaze_y=avg_gaze_y,
        eyes_closed=eyes_closed_ratio > 0.3  # Consider closed if >30% of frames
    )
    
    aggregated_head_pose = HeadPose(
        pitch=avg_pitch,
        yaw=avg_yaw,
        roll=avg_roll
    )
    
    aggregated_blink_features = BlinkFeatures(
        blink_rate=avg_blink_rate,
        time_since_last_blink=0.0  # Not applicable for aggregated data
    )
    
    # Create aggregated feature vector
    aggregated_features = FeatureVector(
        timestamp=time.time(),
        frame_id=len(captured_features),
        face_detected=True,
        face_confidence=1.0,
        landmarks=None,  # Not needed for final calculation
        eye_features=aggregated_eye_features,
        head_pose=aggregated_head_pose,
        blink_features=aggregated_blink_features,
        lighting_quality=0.8,  # Default good lighting
        face_distance=50.0  # Default distance
    )
    
    # Calculate mental state from aggregated features
    mental_state = estimate_from_features(aggregated_features)
    
    return mental_state


def generate_simulated_state(frame_count: int) -> MentalState:
    """Generate simulated mental state for demo."""
    # Create natural variations
    t = frame_count / 30.0  # Time in seconds
    
    # Simulate focus with natural variations
    base_focus = 65 + 15 * np.sin(t * 0.1)
    focus = base_focus + random.gauss(0, 3)
    focus = max(20, min(95, focus))
    
    # Fatigue gradually increases over time with some recovery
    base_fatigue = 25 + 10 * (1 - np.cos(t * 0.05)) + t * 0.02
    fatigue = base_fatigue + random.gauss(0, 2)
    fatigue = max(10, min(85, fatigue))
    
    # Cognitive load varies with focus inversely
    base_load = 45 + 15 * np.sin(t * 0.08 + 1.5)
    load = base_load + random.gauss(0, 4)
    load = max(15, min(90, load))
    
    return MentalState(
        timestamp=time.time(),
        focus_level=focus,
        fatigue_level=fatigue,
        cognitive_load=load,
        confidence=0.85 + random.gauss(0, 0.05),
        trend=Trend.STABLE
    )


def get_status_badge(state: MentalState) -> str:
    """Get HTML for status badge based on mental state."""
    avg = (state.focus_level + (100 - state.fatigue_level) + (100 - state.cognitive_load)) / 3
    
    if avg >= 65:
        return '<span class="status-badge good">Optimal</span>'
    elif avg >= 40:
        return '<span class="status-badge warning">Moderate</span>'
    else:
        return '<span class="status-badge alert">Attention Needed</span>'


def render_consent():
    """Render consent screen with premium styling."""
    st.markdown('<div class="brand">Cognitive Twin</div>', unsafe_allow_html=True)
    st.markdown("# Privacy & Data")
    
    st.markdown("""
    <p class="info-text" style="max-width: 550px; margin: 1.5rem 0;">
        This application uses your webcam to analyze facial expressions and eye movements
        to estimate your cognitive state. Your privacy is our priority.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="max-width: 550px;">
        <div class="privacy-item">
            <div class="privacy-dot green"></div>
            <div>
                <div class="privacy-title">Video Processing</div>
                <div class="privacy-desc">All processing happens locally. Raw video is never stored or transmitted.</div>
            </div>
        </div>
        <div class="privacy-item">
            <div class="privacy-dot green"></div>
            <div>
                <div class="privacy-title">Data Storage</div>
                <div class="privacy-desc">Only aggregated mental state scores are stored locally on your device.</div>
            </div>
        </div>
        <div class="privacy-item">
            <div class="privacy-dot purple"></div>
            <div>
                <div class="privacy-title">AI Insights</div>
                <div class="privacy-desc">Mental state summaries may be sent to AI services for natural language feedback.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("Continue", use_container_width=True):
            st.session_state.consent_granted = True
            st.rerun()
    with col2:
        if st.button("Decline", use_container_width=True):
            st.warning("Camera access is required to use this application.")


def render_dashboard():
    """Render main dashboard."""
    # Auto-refresh every second when monitoring to pick up processor updates
    if AUTOREFRESH_AVAILABLE and st.session_state.get('monitoring', False):
        st_autorefresh(interval=1000, limit=None, key="data_refresh")
    
    # Header row
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col1:
        st.markdown('<div class="brand">Cognitive Twin</div>', unsafe_allow_html=True)
        st.markdown("# Mental State Monitor")
        # Show mode indicator
        if not st.session_state.get('simulation_mode', True) and st.session_state.face_detected_count > 0:
            st.markdown("""
                <div style="display: inline-block; background: rgba(52, 211, 153, 0.15); 
                     border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 6px; 
                     padding: 4px 10px; font-size: 11px; color: #34d399; margin-top: -10px;">
                    ✓ Real-time Face Detection Active (WebRTC)
                </div>
            """, unsafe_allow_html=True)
        elif st.session_state.monitoring and WEBRTC_AVAILABLE:
            st.markdown("""
                <div style="display: inline-block; background: rgba(129, 140, 248, 0.15); 
                     border: 1px solid rgba(129, 140, 248, 0.3); border-radius: 6px; 
                     padding: 4px 10px; font-size: 11px; color: #818cf8; margin-top: -10px;">
                    ◉ WebRTC Ready - Click START on video to begin
                </div>
            """, unsafe_allow_html=True)
        elif st.session_state.get('simulation_mode', True) and st.session_state.monitoring:
            st.markdown("""
                <div style="display: inline-block; background: rgba(251, 191, 36, 0.15); 
                     border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 6px; 
                     padding: 4px 10px; font-size: 11px; color: #fbbf24; margin-top: -10px;">
                    ⚠️ Demo Mode - Start video stream for real analysis
                </div>
            """, unsafe_allow_html=True)
    
    with header_col2:
        if st.session_state.session_start and st.session_state.monitoring:
            duration = time.time() - st.session_state.session_start
            st.markdown(f"""
            <div style="text-align: right; padding-top: 1rem;">
                <div class="session-active">
                    Live: {format_duration(duration)}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        st.markdown('<div class="section-label">Live Feed</div>', unsafe_allow_html=True)
        
        if not st.session_state.monitoring:
            # Show placeholder when not monitoring
            st.markdown("""
            <div class="video-placeholder">
                <div class="camera-icon">
                    <div class="camera-icon-inner"></div>
                </div>
                <div style="color: #6b7280; font-size: 0.875rem;">Camera inactive</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Use WebRTC for continuous video streaming (preferred)
            if WEBRTC_AVAILABLE:
                webrtc_ctx = webrtc_streamer(
                    key="face-analysis",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=FaceAnalysisProcessor,
                    media_stream_constraints={
                        "video": {
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                            "frameRate": {"ideal": 15}
                        },
                        "audio": False
                    },
                    async_processing=True,
                    rtc_configuration={
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }
                )
                
                # Read directly from video processor (works across thread boundary)
                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    processor = webrtc_ctx.video_processor
                    
                    # Update session state from processor
                    st.session_state.frame_count = processor.frame_count
                    st.session_state.simulation_mode = not processor.is_real_detection
                    
                    if processor.face_detected and processor.last_state:
                        st.session_state.current_state = processor.last_state
                        st.session_state.last_features = processor.last_features
                        st.session_state.face_detected_count = st.session_state.get('face_detected_count', 0) + 1
                        
                        # Update history for trends
                        st.session_state.focus_history.append(processor.last_state.focus_level)
                        st.session_state.fatigue_history.append(processor.last_state.fatigue_level)
                        st.session_state.load_history.append(processor.last_state.cognitive_load)
                    
                    # Show live status
                    status_text = f"● Live | Frame #{processor.frame_count}"
                    if processor.face_detected:
                        status_text += " | Face detected"
                        if processor.last_state:
                            s = processor.last_state
                            status_text += f" | Focus: {s.focus_level:.0f}%"
                    else:
                        status_text += " | Waiting for face..."
                    
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="color: #34d399; font-size: 12px;">{status_text}</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # WebRTC not playing - show instruction
                    st.markdown("""
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="color: #fbbf24; font-size: 12px;">⏳ Click START on the video widget to begin</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Fallback to snapshot mode if WebRTC not available
                st.warning("⚠️ Real-time video not available. Using snapshot mode.")
                camera_image = st.camera_input(
                    "Camera Feed",
                    label_visibility="collapsed",
                    key="camera_feed"
                )
                
                # Process the camera image if available
                if camera_image is not None:
                    st.session_state.frame_count += 1
                    
                    # Initialize components if not done yet
                    if st.session_state.feature_extractor is None:
                        initialize_components()
                    
                    if st.session_state.feature_extractor:
                        try:
                            # Convert camera image to numpy array
                            image = Image.open(io.BytesIO(camera_image.getvalue()))
                            frame_array = np.array(image)
                            
                            # Resize to standard resolution if needed
                            target_height, target_width = 480, 640
                            if frame_array.shape[0] != target_height or frame_array.shape[1] != target_width:
                                frame_array = cv2.resize(frame_array, (target_width, target_height))
                            
                            # Handle RGBA/RGB images
                            if len(frame_array.shape) == 3:
                                if frame_array.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
                                elif frame_array.shape[2] == 3:
                                    frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                                else:
                                    frame_bgr = frame_array
                            else:
                                frame_bgr = frame_array
                            
                            # Create CapturedFrame and extract features
                            captured_frame = CapturedFrame(
                                frame=frame_bgr,
                                timestamp=time.time(),
                                frame_id=st.session_state.frame_count
                            )
                            
                            features = st.session_state.feature_extractor.extract(captured_frame)
                            st.session_state.simulation_mode = not st.session_state.feature_extractor.is_real_detection
                            
                            if features.face_detected:
                                st.session_state.last_features = features
                                st.session_state.face_detected_count = st.session_state.get('face_detected_count', 0) + 1
                            else:
                                st.session_state.no_face_count = st.session_state.get('no_face_count', 0) + 1
                        except Exception as e:
                            st.toast(f"Frame processing error: {str(e)[:50]}", icon="⚠️")
        
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        
        btn_col1, btn_col2, _ = st.columns([1.2, 1, 2])
        with btn_col1:
            if not st.session_state.monitoring:
                if st.button("Start Session", use_container_width=True, key="start_session_btn"):
                    initialize_components()
                    st.session_state.monitoring = True
                    st.session_state.session_start = time.time()
                    st.session_state.frame_count = 0
                    # Reset capture state
                    st.session_state.capture_start_time = None
                    st.session_state.capture_active = False
                    st.session_state.captured_features = []
                    st.session_state.analysis_complete = False
                    st.session_state.current_state = None
                    shared_state.set_capture_active(False)
                    shared_state.clear_captured_features()
                    st.rerun()
            else:
                if st.button("End Session", use_container_width=True, key="end_session_btn"):
                    st.session_state.monitoring = False
                    st.session_state.capture_active = False
                    st.session_state.capture_start_time = None
                    st.session_state.analysis_complete = False
                    shared_state.set_capture_active(False)
                    if st.session_state.digital_twin:
                        st.session_state.digital_twin.save()
                    st.rerun()
        
        with btn_col2:
            if st.button("Settings", use_container_width=True, key="settings_btn"):
                st.session_state.show_settings = not st.session_state.get('show_settings', False)
                st.rerun()
        
        # Settings panel
        if st.session_state.get('show_settings', False):
            st.markdown("---")
            st.markdown("### Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                # AI Model selection
                model_options = [
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "google/gemma-7b-it:free",
                    "mistralai/mistral-7b-instruct:free",
                ]
                current_model = st.session_state.config.genai.model if st.session_state.config else model_options[0]
                selected_model = st.selectbox("AI Model", model_options, index=model_options.index(current_model) if current_model in model_options else 0)
                
                # Explanation interval
                explanation_interval = st.slider("AI Insight Interval (seconds)", 15, 120, 30)
                
            with col2:
                # Detection sensitivity
                detection_conf = st.slider("Face Detection Sensitivity", 0.3, 0.9, 0.5)
                
                # Notification toggle
                notifications_enabled = st.checkbox("Enable Notifications", value=True)
            
            if st.button("Close Settings", key="close_settings_btn"):
                st.session_state.show_settings = False
                st.rerun()
    
    with right_col:
        st.markdown('<div class="section-label">Cognitive State</div>', unsafe_allow_html=True)
        
        # Show real-time feature debug info
        if st.session_state.monitoring:
            features = st.session_state.last_features
            
            # Debug expander to show raw feature values
            with st.expander("🔍 Debug: Raw Feature Data", expanded=False):
                if features and features.face_detected:
                    st.success(f"✓ Face detected! Frame #{st.session_state.frame_count}")
                    st.write(f"Real detection: {not st.session_state.simulation_mode}")
                    if features.eye_features:
                        ef = features.eye_features
                        st.write(f"**Eye Features:**")
                        st.write(f"- EAR: {ef.ear_average:.3f}")
                        st.write(f"- Gaze: ({ef.gaze_x:.2f}, {ef.gaze_y:.2f})")
                        st.write(f"- Eyes Closed: {ef.eyes_closed}")
                    if features.head_pose:
                        hp = features.head_pose
                        st.write(f"**Head Pose:** P{hp.pitch:.0f}° Y{hp.yaw:.0f}° R{hp.roll:.0f}°")
                else:
                    st.info("Waiting for face detection...")
        
        # Determine if we have valid data to display
        state = st.session_state.current_state
        has_valid_data = state is not None
        
        # Update explanation based on state
        if has_valid_data and state:
            focus = state.focus_level
            fatigue = state.fatigue_level
            load = state.cognitive_load
            if fatigue > 60:
                st.session_state.last_explanation = f"Fatigue level is {fatigue:.0f}%. Consider taking a break."
            elif focus < 40:
                st.session_state.last_explanation = f"Focus is {focus:.0f}%. Try a quick change of scenery."
            elif focus > 70 and fatigue < 40:
                st.session_state.last_explanation = f"Excellent: {focus:.0f}% focus, {fatigue:.0f}% fatigue. Great for deep work!"
            else:
                st.session_state.last_explanation = f"Focus: {focus:.0f}% | Fatigue: {fatigue:.0f}% | Load: {load:.0f}%"
        elif st.session_state.monitoring:
            st.session_state.last_explanation = "Click START on video to begin monitoring"
        
        # Gauge metrics
        gauge_cols = st.columns(3)
        
        if has_valid_data and state:
            focus = state.focus_level
            fatigue = state.fatigue_level
            load = state.cognitive_load
        else:
            # Show waiting state
            focus = 0
            fatigue = 0
            load = 0
        
        with gauge_cols[0]:
            if has_valid_data:
                st.plotly_chart(create_gauge(focus, "FOCUS", False), use_container_width=True, config={'displayModeBar': False})
            else:
                # Show waiting indicator
                fig = create_gauge(0, "FOCUS", False)
                fig.update_traces(number={'valueformat': '.0f', 'suffix': ''})
                fig.add_annotation(text="Waiting for<br>face detection", 
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color='#6b7280'))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("""
                <div style="text-align: center; font-size: 10px; color: #6b7280; margin-top: -15px;">
                    <span style="color: #34d399;">●</span> 65-100 Good 
                    <span style="color: #fbbf24;">●</span> 35-64 Fair 
                    <span style="color: #f87171;">●</span> 0-34 Low
                </div>
            """, unsafe_allow_html=True)
        with gauge_cols[1]:
            if has_valid_data:
                st.plotly_chart(create_gauge(fatigue, "FATIGUE", True), use_container_width=True, config={'displayModeBar': False})
            else:
                fig = create_gauge(0, "FATIGUE", True)
                fig.update_traces(number={'valueformat': '.0f', 'suffix': ''})
                fig.add_annotation(text="Waiting for<br>face detection", 
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color='#6b7280'))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("""
                <div style="text-align: center; font-size: 10px; color: #6b7280; margin-top: -15px;">
                    <span style="color: #34d399;">●</span> 0-35 Low 
                    <span style="color: #fbbf24;">●</span> 36-65 Moderate 
                    <span style="color: #f87171;">●</span> 66+ High
                </div>
            """, unsafe_allow_html=True)
        with gauge_cols[2]:
            if has_valid_data:
                st.plotly_chart(create_gauge(load, "LOAD", True), use_container_width=True, config={'displayModeBar': False})
            else:
                fig = create_gauge(0, "LOAD", True)
                fig.update_traces(number={'valueformat': '.0f', 'suffix': ''})
                fig.add_annotation(text="Waiting for<br>face detection", 
                                  x=0.5, y=0.5, showarrow=False,
                                  font=dict(size=14, color='#6b7280'))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("""
                <div style="text-align: center; font-size: 10px; color: #6b7280; margin-top: -15px;">
                    <span style="color: #34d399;">●</span> 0-35 Light 
                    <span style="color: #fbbf24;">●</span> 36-65 Moderate 
                    <span style="color: #f87171;">●</span> 66+ Heavy
                </div>
            """, unsafe_allow_html=True)
        
        # Status badge
        if state:
            if has_valid_data:
                st.markdown(f"""
                <div style="text-align: center; margin: -0.5rem 0 1rem 0;">
                    {get_status_badge(state)}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Show waiting status
                st.markdown("""
                <div style="text-align: center; margin: -0.5rem 0 1rem 0;">
                    <span class="status-badge warning">⏳ Waiting for Camera</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Sparklines
        st.markdown('<div class="section-label" style="margin-top: 0.5rem;">Trends</div>', unsafe_allow_html=True)
        
        spark_cols = st.columns(3)
        with spark_cols[0]:
            st.plotly_chart(
                create_sparkline(list(st.session_state.focus_history), "#818cf8"),
                use_container_width=True, config={'displayModeBar': False}
            )
        with spark_cols[1]:
            st.plotly_chart(
                create_sparkline(list(st.session_state.fatigue_history), "#fbbf24"),
                use_container_width=True, config={'displayModeBar': False}
            )
        with spark_cols[2]:
            st.plotly_chart(
                create_sparkline(list(st.session_state.load_history), "#34d399"),
                use_container_width=True, config={'displayModeBar': False}
            )
        
        # AI Insight
        st.markdown('<div class="section-label" style="margin-top: 1rem;">Insight</div>', unsafe_allow_html=True)
        
        insight_opacity = "1" if st.session_state.monitoring else "0.5"
        st.markdown(f"""
        <div class="insight-card" style="opacity: {insight_opacity};">
            <div class="insight-text">{st.session_state.last_explanation}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Metric explanation expander
        with st.expander("ℹ️ What do these metrics mean?"):
            st.markdown("""
            <div style="font-size: 13px; line-height: 1.8; color: #9ca3af;">
                
**FOCUS** — How well you're concentrating on the task at hand
- <span style="color: #34d399;">●</span> **65-100%**: Excellent - You're in a productive, attentive state
- <span style="color: #fbbf24;">●</span> **35-64%**: Moderate - Some distraction, but manageable  
- <span style="color: #f87171;">●</span> **0-34%**: Low - Consider taking a break or changing environment

**FATIGUE** — Your level of mental tiredness
- <span style="color: #34d399;">●</span> **0-35%**: Fresh - You're alert and energized
- <span style="color: #fbbf24;">●</span> **36-65%**: Moderate - Normal after extended work, monitor it
- <span style="color: #f87171;">●</span> **66-100%**: High - Take a break! Risk of errors increases

**COGNITIVE LOAD** — Mental effort required for current tasks
- <span style="color: #34d399;">●</span> **0-35%**: Light - Simple or routine tasks
- <span style="color: #fbbf24;">●</span> **36-65%**: Moderate - Engaging work, sustainable pace
- <span style="color: #f87171;">●</span> **66-100%**: Heavy - Complex tasks; consider breaking them down

---

**Optimal work state:** High Focus (>65%), Low Fatigue (<35%), Moderate Load (35-65%)

**When to take a break:** Fatigue >60% OR Focus <40% for 5+ minutes
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh during capture to update countdown timer
    if st.session_state.monitoring and st.session_state.capture_active:
        time.sleep(0.5)  # Faster refresh for smoother countdown
        st.rerun()


def main():
    """Main application."""
    init_session_state()
    
    if not st.session_state.consent_granted:
        render_consent()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()

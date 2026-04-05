"""
User feedback collection for validation and calibration.

Collects self-reported mental states to validate model estimates
and enable research-quality measurement of system accuracy.
"""

import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

from .database import DatabaseManager
from ..models import MentalState


class FeedbackManager:
    """Manages user feedback collection and validation metrics."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        Initialize feedback manager.
        
        Args:
            db: Database manager instance.
        """
        self.db = db or DatabaseManager()
    
    def record_feedback(
        self,
        session_id: Optional[str],
        reported_focus: float,
        reported_fatigue: float,
        reported_load: float,
        current_estimate: Optional[MentalState] = None
    ) -> int:
        """
        Record user self-report feedback.
        
        Args:
            session_id: Current session ID.
            reported_focus: User-reported focus level (0-100).
            reported_fatigue: User-reported fatigue level (0-100).
            reported_load: User-reported cognitive load (0-100).
            current_estimate: System's current mental state estimate.
            
        Returns:
            Feedback record ID.
        """
        timestamp = time.time()
        
        estimated_focus = current_estimate.focus_level if current_estimate else None
        estimated_fatigue = current_estimate.fatigue_level if current_estimate else None
        estimated_load = current_estimate.cognitive_load if current_estimate else None
        
        cursor = self.db.execute(
            """INSERT INTO user_feedback 
               (session_id, timestamp, reported_focus, reported_fatigue, reported_load,
                estimated_focus, estimated_fatigue, estimated_load)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id, timestamp,
                reported_focus, reported_fatigue, reported_load,
                estimated_focus, estimated_fatigue, estimated_load
            )
        )
        self.db.commit()
        
        return cursor.lastrowid
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """
        Calculate correlation between estimates and self-reports.
        
        Returns:
            Dictionary with validation metrics including Pearson correlations.
        """
        cursor = self.db.execute(
            """SELECT reported_focus, reported_fatigue, reported_load,
                      estimated_focus, estimated_fatigue, estimated_load
               FROM user_feedback 
               WHERE estimated_focus IS NOT NULL"""
        )
        
        rows = cursor.fetchall()
        
        if len(rows) < 5:
            return {
                "sample_count": len(rows),
                "sufficient_data": False,
                "message": "Need at least 5 feedback samples for validation"
            }
        
        # Extract arrays
        reported_focus = np.array([r['reported_focus'] for r in rows])
        reported_fatigue = np.array([r['reported_fatigue'] for r in rows])
        reported_load = np.array([r['reported_load'] for r in rows])
        
        estimated_focus = np.array([r['estimated_focus'] for r in rows])
        estimated_fatigue = np.array([r['estimated_fatigue'] for r in rows])
        estimated_load = np.array([r['estimated_load'] for r in rows])
        
        # Calculate Pearson correlations
        focus_corr = self._pearson_correlation(reported_focus, estimated_focus)
        fatigue_corr = self._pearson_correlation(reported_fatigue, estimated_fatigue)
        load_corr = self._pearson_correlation(reported_load, estimated_load)
        
        # Calculate mean absolute errors
        focus_mae = np.mean(np.abs(reported_focus - estimated_focus))
        fatigue_mae = np.mean(np.abs(reported_fatigue - estimated_fatigue))
        load_mae = np.mean(np.abs(reported_load - estimated_load))
        
        return {
            "sample_count": len(rows),
            "sufficient_data": True,
            "correlations": {
                "focus": float(focus_corr),
                "fatigue": float(fatigue_corr),
                "cognitive_load": float(load_corr),
                "average": float(np.mean([focus_corr, fatigue_corr, load_corr]))
            },
            "mean_absolute_errors": {
                "focus": float(focus_mae),
                "fatigue": float(fatigue_mae),
                "cognitive_load": float(load_mae)
            },
            "meets_target": float(np.mean([focus_corr, fatigue_corr, load_corr])) >= 0.7
        }
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        
        return float(np.corrcoef(x, y)[0, 1])
    
    def get_calibration_suggestions(self) -> Dict[str, float]:
        """
        Analyze feedback to suggest calibration adjustments.
        
        Returns:
            Dictionary with suggested baseline adjustments.
        """
        cursor = self.db.execute(
            """SELECT 
                AVG(reported_focus - estimated_focus) as focus_bias,
                AVG(reported_fatigue - estimated_fatigue) as fatigue_bias,
                AVG(reported_load - estimated_load) as load_bias
               FROM user_feedback 
               WHERE estimated_focus IS NOT NULL"""
        )
        
        row = cursor.fetchone()
        
        if row['focus_bias'] is None:
            return {"message": "No feedback data available for calibration"}
        
        return {
            "focus_adjustment": float(row['focus_bias'] or 0),
            "fatigue_adjustment": float(row['fatigue_bias'] or 0),
            "load_adjustment": float(row['load_bias'] or 0)
        }
    
    def get_recent_feedback(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent feedback entries.
        
        Args:
            count: Maximum number of entries to return.
            
        Returns:
            List of feedback dictionaries.
        """
        cursor = self.db.execute(
            "SELECT * FROM user_feedback ORDER BY timestamp DESC LIMIT ?",
            (count,)
        )
        
        return [dict(row) for row in cursor.fetchall()]

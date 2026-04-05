"""
Session management for Cognitive Twin.

Handles session lifecycle, persistence, and retrieval.
"""

import uuid
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from .database import DatabaseManager
from ..models import Session, MentalState


class SessionManager:
    """Manages monitoring sessions and their persistence."""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        Initialize session manager.
        
        Args:
            db: Database manager instance. Creates new one if not provided.
        """
        self.db = db or DatabaseManager()
        self._current_session: Optional[Session] = None
        self._state_buffer: List[MentalState] = []
        self._buffer_size = 60  # Flush every 60 states
    
    def start_session(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        """
        Start a new monitoring session.
        
        Args:
            metadata: Optional metadata to store with session.
            
        Returns:
            New Session object.
        """
        session_id = str(uuid.uuid4())[:12]
        start_time = time.time()
        
        # Create session in database
        self.db.execute(
            """INSERT INTO sessions (id, start_time, metadata) 
               VALUES (?, ?, ?)""",
            (session_id, start_time, str(metadata) if metadata else None)
        )
        self.db.commit()
        
        # Create session object
        self._current_session = Session(
            id=session_id,
            start_time=start_time
        )
        self._state_buffer.clear()
        
        return self._current_session
    
    def end_session(self) -> Optional[Session]:
        """
        End the current session.
        
        Returns:
            Completed Session object or None if no active session.
        """
        if self._current_session is None:
            return None
        
        # Flush remaining buffer
        self._flush_buffer()
        
        end_time = time.time()
        self._current_session.end_time = end_time
        
        # Calculate session averages
        cursor = self.db.execute(
            """SELECT AVG(focus_level), AVG(fatigue_level), AVG(cognitive_load), COUNT(*)
               FROM mental_states WHERE session_id = ?""",
            (self._current_session.id,)
        )
        row = cursor.fetchone()
        if row and row[3] > 0:
            self._current_session.avg_focus = row[0] or 0
            self._current_session.avg_fatigue = row[1] or 0
            self._current_session.avg_cognitive_load = row[2] or 0
            self._current_session.total_frames = row[3]
        
        # Update database
        self.db.execute(
            """UPDATE sessions SET 
               end_time = ?, avg_focus = ?, avg_fatigue = ?, 
               avg_cognitive_load = ?, total_frames = ?
               WHERE id = ?""",
            (
                end_time,
                self._current_session.avg_focus,
                self._current_session.avg_fatigue,
                self._current_session.avg_cognitive_load,
                self._current_session.total_frames,
                self._current_session.id
            )
        )
        self.db.commit()
        
        completed = self._current_session
        self._current_session = None
        return completed
    
    def record_state(self, state: MentalState) -> None:
        """
        Record a mental state to the current session.
        
        Args:
            state: Mental state to record.
        """
        if self._current_session is None:
            return
        
        self._state_buffer.append(state)
        
        if len(self._state_buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Write buffered states to database."""
        if not self._state_buffer or self._current_session is None:
            return
        
        data = [
            (
                self._current_session.id,
                s.timestamp,
                s.focus_level,
                s.fatigue_level,
                s.cognitive_load,
                s.confidence,
                s.trend.value if hasattr(s.trend, 'value') else str(s.trend)
            )
            for s in self._state_buffer
        ]
        
        self.db.executemany(
            """INSERT INTO mental_states 
               (session_id, timestamp, focus_level, fatigue_level, 
                cognitive_load, confidence, trend)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            data
        )
        self.db.commit()
        self._state_buffer.clear()
    
    def get_current_session(self) -> Optional[Session]:
        """Get the current active session."""
        return self._current_session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session ID to retrieve.
            
        Returns:
            Session object or None if not found.
        """
        cursor = self.db.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        
        return Session(
            id=row['id'],
            start_time=row['start_time'],
            end_time=row['end_time'],
            avg_focus=row['avg_focus'],
            avg_fatigue=row['avg_fatigue'],
            avg_cognitive_load=row['avg_cognitive_load'],
            total_frames=row['total_frames'],
            explanations_count=row['explanations_count']
        )
    
    def get_recent_sessions(self, count: int = 10) -> List[Session]:
        """
        Get recent sessions.
        
        Args:
            count: Maximum number of sessions to return.
            
        Returns:
            List of Session objects, most recent first.
        """
        cursor = self.db.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?",
            (count,)
        )
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(Session(
                id=row['id'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                avg_focus=row['avg_focus'],
                avg_fatigue=row['avg_fatigue'],
                avg_cognitive_load=row['avg_cognitive_load'],
                total_frames=row['total_frames'],
                explanations_count=row['explanations_count']
            ))
        
        return sessions
    
    def get_session_states(self, session_id: str) -> List[MentalState]:
        """
        Get all mental states for a session.
        
        Args:
            session_id: Session ID to retrieve states for.
            
        Returns:
            List of MentalState objects.
        """
        from ..models import Trend
        
        cursor = self.db.execute(
            """SELECT * FROM mental_states 
               WHERE session_id = ? ORDER BY timestamp""",
            (session_id,)
        )
        
        states = []
        for row in cursor.fetchall():
            trend_str = row['trend']
            try:
                trend = Trend(trend_str)
            except ValueError:
                trend = Trend.STABLE
            
            states.append(MentalState(
                timestamp=row['timestamp'],
                focus_level=row['focus_level'],
                fatigue_level=row['fatigue_level'],
                cognitive_load=row['cognitive_load'],
                confidence=row['confidence'],
                trend=trend
            ))
        
        return states
    
    def get_daily_summary(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a day.
        
        Args:
            date: Date to summarize. Defaults to today.
            
        Returns:
            Dictionary with summary statistics.
        """
        if date is None:
            date = datetime.now()
        
        start_of_day = datetime(date.year, date.month, date.day).timestamp()
        end_of_day = start_of_day + 86400  # 24 hours
        
        cursor = self.db.execute(
            """SELECT COUNT(*) as sessions,
                      SUM(total_frames) as frames,
                      AVG(avg_focus) as focus,
                      AVG(avg_fatigue) as fatigue,
                      SUM(end_time - start_time) as duration
               FROM sessions 
               WHERE start_time >= ? AND start_time < ?""",
            (start_of_day, end_of_day)
        )
        
        row = cursor.fetchone()
        return {
            "date": date.strftime("%Y-%m-%d"),
            "session_count": row['sessions'] or 0,
            "total_frames": row['frames'] or 0,
            "avg_focus": row['focus'] or 0,
            "avg_fatigue": row['fatigue'] or 0,
            "total_minutes": (row['duration'] or 0) / 60
        }

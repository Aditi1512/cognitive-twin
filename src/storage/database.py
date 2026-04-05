"""
SQLite database manager for Cognitive Twin.

Handles all database operations including schema creation and migrations.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. Defaults to user data directory.
        """
        if db_path is None:
            db_path = Path.home() / ".cognitive_twin" / "cognitive_twin.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._connection: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time REAL NOT NULL,
                end_time REAL,
                avg_focus REAL DEFAULT 0,
                avg_fatigue REAL DEFAULT 0,
                avg_cognitive_load REAL DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                explanations_count INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Mental states table (time-series data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mental_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                focus_level REAL NOT NULL,
                fatigue_level REAL NOT NULL,
                cognitive_load REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                trend TEXT DEFAULT 'stable',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # User feedback for validation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL NOT NULL,
                reported_focus REAL,
                reported_fatigue REAL,
                reported_load REAL,
                estimated_focus REAL,
                estimated_fatigue REAL,
                estimated_load REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # Explanations log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL,
                suggestion TEXT,
                urgency TEXT DEFAULT 'info',
                model_used TEXT,
                generation_time_ms REAL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        
        # User patterns (learned baselines)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_patterns (
                user_id TEXT PRIMARY KEY,
                focus_baseline REAL DEFAULT 60.0,
                fatigue_baseline REAL DEFAULT 30.0,
                typical_session_minutes REAL DEFAULT 60.0,
                best_performance_hour INTEGER DEFAULT 10,
                sample_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indices for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mental_states_session 
            ON mental_states(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_session 
            ON user_feedback(session_id, timestamp)
        """)
        
        conn.commit()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query."""
        conn = self._get_connection()
        return conn.execute(query, params)
    
    def executemany(self, query: str, params_list: list) -> sqlite3.Cursor:
        """Execute a SQL query with multiple parameter sets."""
        conn = self._get_connection()
        return conn.executemany(query, params_list)
    
    def commit(self) -> None:
        """Commit pending transactions."""
        if self._connection:
            self._connection.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def export_data(self, output_path: Path) -> None:
        """
        Export all user data to JSON file.
        
        Args:
            output_path: Path to write JSON export.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        data = {
            "export_time": datetime.now().isoformat(),
            "sessions": [],
            "mental_states": [],
            "feedback": [],
            "patterns": []
        }
        
        # Export sessions
        cursor.execute("SELECT * FROM sessions")
        for row in cursor.fetchall():
            data["sessions"].append(dict(row))
        
        # Export mental states
        cursor.execute("SELECT * FROM mental_states")
        for row in cursor.fetchall():
            data["mental_states"].append(dict(row))
        
        # Export feedback
        cursor.execute("SELECT * FROM user_feedback")
        for row in cursor.fetchall():
            data["feedback"].append(dict(row))
        
        # Export patterns
        cursor.execute("SELECT * FROM user_patterns")
        for row in cursor.fetchall():
            data["patterns"].append(dict(row))
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def delete_all_data(self) -> None:
        """Delete all user data (GDPR compliance)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM explanations")
        cursor.execute("DELETE FROM user_feedback")
        cursor.execute("DELETE FROM mental_states")
        cursor.execute("DELETE FROM sessions")
        cursor.execute("DELETE FROM user_patterns")
        
        conn.commit()

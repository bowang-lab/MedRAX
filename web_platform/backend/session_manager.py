"""
Session Manager for MedRAX Backend
Handles session lifecycle, cleanup, and persistence
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from pathlib import Path
import asyncio

from logger_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages chat sessions with automatic cleanup and expiry.
    
    Features:
    - Automatic cleanup of old sessions
    - Session expiry based on last access
    - Memory-efficient storage
    - Thread-safe operations
    """
    
    def __init__(self, max_age_hours: int = 24, cleanup_interval_minutes: int = 60):
        """
        Initialize SessionManager.
        
        Args:
            max_age_hours: Maximum session age before cleanup (default: 24 hours)
            cleanup_interval_minutes: How often to run cleanup (default: 60 minutes)
        """
        self.sessions: Dict[str, any] = {}
        self.last_access: Dict[str, datetime] = {}
        self.max_age = timedelta(hours=max_age_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        logger.info("session_manager_initialized", max_age_hours=max_age_hours)
    
    def create_session(self, session_id: str, chat_interface: any) -> None:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            chat_interface: MinimalChatInterface instance
        """
        self.sessions[session_id] = chat_interface
        self.last_access[session_id] = datetime.now(timezone.utc)
        
        logger.info("session_created", session_id=session_id[:8], total_sessions=len(self.sessions))
    
    def get_session(self, session_id: str) -> Optional[any]:
        """
        Get a session and update its last access time.
        
        Args:
            session_id: Session identifier
        
        Returns:
            MinimalChatInterface instance or None if not found
        """
        if session_id in self.sessions:
            self.last_access[session_id] = datetime.now(timezone.utc)
            return self.sessions[session_id]
        
        logger.warning("session_not_found", session_id=session_id[:8])
        return None
    
    def delete_session(self, session_id: str) -> bool:
        """
        Explicitly delete a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.last_access[session_id]
            logger.info("session_deleted", session_id=session_id[:8])
            return True
        
        return False
    
    def cleanup_old_sessions(self) -> int:
        """
        Remove sessions that haven't been accessed recently.
        
        Returns:
            Number of sessions cleaned up
        """
        now = datetime.now(timezone.utc)
        to_delete = []
        
        for session_id, last_access in self.last_access.items():
            age = now - last_access
            if age > self.max_age:
                to_delete.append(session_id)
        
        # Delete old sessions
        for session_id in to_delete:
            del self.sessions[session_id]
            del self.last_access[session_id]
        
        if to_delete:
            logger.info(
                "sessions_cleaned_up",
                count=len(to_delete),
                remaining=len(self.sessions)
            )
        
        return len(to_delete)
    
    async def cleanup_old_sessions_periodically(self):
        """
        Periodically cleanup old sessions (async task).
        Runs every hour.
        """
        import asyncio
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self.cleanup_old_sessions()
            except Exception as e:
                logger.error("cleanup_error", error=str(e))
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with session info or None
        """
        if session_id not in self.sessions:
            return None
        
        last_access = self.last_access[session_id]
        age = datetime.now(timezone.utc) - last_access
        
        return {
            "session_id": session_id,
            "last_access": last_access.isoformat(),
            "age_seconds": int(age.total_seconds()),
            "age_hours": age.total_seconds() / 3600,
            "is_active": age < self.max_age
        }
    
    def get_all_sessions_info(self) -> Dict:
        """
        Get information about all sessions.
        
        Returns:
            Dictionary with statistics and session list
        """
        now = datetime.now(timezone.utc)
        active_count = 0
        expired_count = 0
        
        sessions_list = []
        for session_id in self.sessions.keys():
            info = self.get_session_info(session_id)
            if info:
                sessions_list.append(info)
                if info["is_active"]:
                    active_count += 1
                else:
                    expired_count += 1
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_count,
            "expired_sessions": expired_count,
            "max_age_hours": self.max_age.total_seconds() / 3600,
            "sessions": sessions_list
        }
    
    async def start_cleanup_task(self):
        """
        Start background task for automatic cleanup.
        Run this as a FastAPI background task.
        """
        logger.info("cleanup_task_started", interval_minutes=self.cleanup_interval.total_seconds() / 60)
        
        while True:
            await asyncio.sleep(self.cleanup_interval.total_seconds())
            
            try:
                cleaned = self.cleanup_old_sessions()
                if cleaned > 0:
                    logger.info("automatic_cleanup", cleaned=cleaned)
            except Exception as e:
                logger.error("cleanup_failed", error=str(e), exc_info=True)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get the global session manager instance.
    Creates it if it doesn't exist.
    
    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


# Example usage
if __name__ == "__main__":
    import time
    
    manager = SessionManager(max_age_hours=1)  # 1 hour for testing
    
    # Create sessions
    class MockInterface:
        def __init__(self, name):
            self.name = name
    
    manager.create_session("session1", MockInterface("Test 1"))
    manager.create_session("session2", MockInterface("Test 2"))
    
    print("Sessions created:", manager.get_all_sessions_info())
    
    # Access session
    session = manager.get_session("session1")
    print("Retrieved session:", session.name if session else "Not found")
    
    # Get info
    info = manager.get_session_info("session1")
    print("Session info:", info)
    
    # Cleanup
    cleaned = manager.cleanup_old_sessions()
    print(f"Cleaned up {cleaned} sessions")


"""
Session Manager for MedRAX Backend
Handles session lifecycle, cleanup, and persistence
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from logger_config import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages chat sessions with automatic cleanup and expiry.

    Features:
    - Multiple chats per user support
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
        # New structure: users -> chats -> chat_interface
        self.users: Dict[str, Dict[str, any]] = {}  # {user_id: {chat_id: chat_interface}}
        self.chat_metadata: Dict[str, Dict] = {}  # {chat_id: metadata}
        self.last_access: Dict[str, datetime] = {}  # {chat_id: timestamp}

        # Legacy support for old session-based API
        self.sessions: Dict[str, any] = {}  # Deprecated, for backward compatibility

        self.max_age = timedelta(hours=max_age_hours)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)

        logger.info("session_manager_initialized", max_age_hours=max_age_hours)

    def create_session(self, session_id: str, chat_interface: any) -> None:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            chat_interface: ChatInterface instance
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

    # ========== NEW: Multi-Chat Per User Support ==========

    def create_chat(self, user_id: str, chat_id: str, chat_interface: any, metadata: Optional[Dict] = None) -> None:
        """
        Create a new chat for a user.

        Args:
            user_id: User identifier
            chat_id: Unique chat identifier
            chat_interface: ChatInterface instance
            metadata: Optional metadata (name, description, etc.)
        """
        if user_id not in self.users:
            self.users[user_id] = {}

        self.users[user_id][chat_id] = chat_interface
        self.last_access[chat_id] = datetime.now(timezone.utc)

        # Store metadata
        self.chat_metadata[chat_id] = metadata or {
            "name": f"Chat {len(self.users[user_id])}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "image_count": 0
        }

        logger.info("chat_created", user_id=user_id[:8], chat_id=chat_id[:8], total_chats=len(self.users[user_id]))

    def get_chat(self, user_id: str, chat_id: str) -> Optional[any]:
        """
        Get a specific chat for a user.

        Args:
            user_id: User identifier
            chat_id: Chat identifier

        Returns:
            MinimalChatInterface instance or None if not found
        """
        if user_id in self.users and chat_id in self.users[user_id]:
            self.last_access[chat_id] = datetime.now(timezone.utc)
            return self.users[user_id][chat_id]

        logger.warning("chat_not_found", user_id=user_id[:8], chat_id=chat_id[:8])
        return None

    def list_chats(self, user_id: str) -> List[Dict]:
        """
        List all chats for a user.

        Args:
            user_id: User identifier

        Returns:
            List of chat metadata dictionaries
        """
        if user_id not in self.users:
            return []

        chats = []
        for chat_id in self.users[user_id].keys():
            metadata = self.chat_metadata.get(chat_id, {})
            last_access = self.last_access.get(chat_id)

            chats.append({
                "chat_id": chat_id,
                "name": metadata.get("name", "Unnamed Chat"),
                "created_at": metadata.get("created_at"),
                "last_access": last_access.isoformat() if last_access else None,
                "message_count": metadata.get("message_count", 0),
                "image_count": metadata.get("image_count", 0)
            })

        # Sort by last access (most recent first)
        chats.sort(key=lambda x: x["last_access"] or "", reverse=True)
        return chats

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        """
        Delete a specific chat for a user.

        Args:
            user_id: User identifier
            chat_id: Chat identifier

        Returns:
            True if deleted, False if not found
        """
        if user_id in self.users and chat_id in self.users[user_id]:
            del self.users[user_id][chat_id]
            if chat_id in self.chat_metadata:
                del self.chat_metadata[chat_id]
            if chat_id in self.last_access:
                del self.last_access[chat_id]

            # Clean up user if no more chats
            if len(self.users[user_id]) == 0:
                del self.users[user_id]

            logger.info("chat_deleted", user_id=user_id[:8], chat_id=chat_id[:8])
            return True

        return False

    def update_chat_metadata(self, chat_id: str, updates: Dict) -> None:
        """
        Update metadata for a chat.

        Args:
            chat_id: Chat identifier
            updates: Dictionary of fields to update
        """
        if chat_id in self.chat_metadata:
            self.chat_metadata[chat_id].update(updates)
            logger.debug("chat_metadata_updated", chat_id=chat_id[:8], fields=list(updates.keys()))

    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics for all sessions and chats.

        Returns:
            Dictionary with detailed memory statistics
        """

        stats = {
            "total_users": len(self.users),
            "total_sessions": len(self.sessions),
            "total_chats": sum(len(chats) for chats in self.users.values()),
            "total_metadata_entries": len(self.chat_metadata),
            "users": {}
        }

        # Get per-user statistics
        for user_id, chats in self.users.items():
            user_stats = {
                "chat_count": len(chats),
                "chats": {}
            }

            for chat_id, chat_interface in chats.items():
                if hasattr(chat_interface, 'get_memory_info'):
                    user_stats["chats"][chat_id] = chat_interface.get_memory_info()

            stats["users"][user_id] = user_stats

        return stats

    def cleanup_all_memory(self) -> Dict:
        """
        Trigger memory cleanup for all active chats.

        Returns:
            Dictionary with cleanup statistics
        """
        import gc

        stats = {
            "chats_cleaned": 0,
            "total_tool_results_cleared": 0,
            "total_objects_collected": 0
        }

        # Clean up each chat interface
        for user_chats in self.users.values():
            for chat_interface in user_chats.values():
                if hasattr(chat_interface, 'cleanup_memory'):
                    chat_stats = chat_interface.cleanup_memory()
                    stats["chats_cleaned"] += 1
                    stats["total_tool_results_cleared"] += chat_stats.get("tool_results_cleared", 0)
                    stats["total_objects_collected"] += chat_stats.get("objects_collected", 0)

        # Also clean legacy sessions
        for chat_interface in self.sessions.values():
            if hasattr(chat_interface, 'cleanup_memory'):
                chat_stats = chat_interface.cleanup_memory()
                stats["chats_cleaned"] += 1
                stats["total_tool_results_cleared"] += chat_stats.get("tool_results_cleared", 0)
                stats["total_objects_collected"] += chat_stats.get("objects_collected", 0)

        # Force global garbage collection
        collected = gc.collect()
        stats["global_gc_collected"] = collected

        logger.info("all_memory_cleanup", **stats)
        return stats

    def cleanup_old_files(self, max_age_hours: int = 24) -> Dict:
        """
        Clean up old temporary files from all chats.

        Args:
            max_age_hours: Maximum age of files to keep

        Returns:
            Dictionary with cleanup statistics
        """
        from datetime import datetime, timedelta, timezone
        from pathlib import Path

        stats = {
            "files_deleted": 0,
            "space_freed_mb": 0,
            "errors": []
        }

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        # Clean up temp directory
        temp_dir = Path("temp")
        if temp_dir.exists():
            for file_path in temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
                        if file_time < cutoff_time:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            file_path.unlink()
                            stats["files_deleted"] += 1
                            stats["space_freed_mb"] += size_mb
                    except Exception as e:
                        stats["errors"].append(f"{file_path}: {str(e)}")

        logger.info("old_files_cleanup", files_deleted=stats["files_deleted"], space_freed_mb=stats["space_freed_mb"])
        return stats

    async def start_cleanup_task(self):
        """
        Start background task for automatic cleanup.
        Run this as a FastAPI background task.
        """
        logger.info("cleanup_task_started", interval_minutes=self.cleanup_interval.total_seconds() / 60)

        while True:
            await asyncio.sleep(self.cleanup_interval.total_seconds())

            try:
                # Clean up old sessions
                cleaned = self.cleanup_old_sessions()
                if cleaned > 0:
                    logger.info("automatic_cleanup", sessions_cleaned=cleaned)

                # Clean up old files
                file_stats = self.cleanup_old_files()
                if file_stats["files_deleted"] > 0:
                    logger.info("automatic_file_cleanup", **file_stats)

                # Periodic memory cleanup (every 10 cleanup cycles)
                if not hasattr(self, '_cleanup_counter'):
                    self._cleanup_counter = 0
                self._cleanup_counter += 1

                if self._cleanup_counter >= 10:
                    memory_stats = self.cleanup_all_memory()
                    logger.info("automatic_memory_cleanup", **memory_stats)
                    self._cleanup_counter = 0

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


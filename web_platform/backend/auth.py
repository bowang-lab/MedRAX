"""
Database-backed authentication system for MedRAX Web Platform
Simple username/password auth with persistent sessions
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from database import AuthSession, AuthUser, SessionLocal
from logger_config import get_logger

logger = get_logger(__name__)


class SimpleAuthManager:
    """
    Simple authentication manager with database persistence.
    - Users stored in database (auth_users table)
    - Sessions/tokens stored in database (auth_sessions table)
    - 30-day session duration for internal use
    - Basic password hashing with SHA256
    
    NOTE: This is intentionally simple for internal use.
    For production with sensitive data, use proper auth (OAuth, bcrypt, etc.)
    """

    def __init__(self, session_duration_days: int = 30):
        self.session_duration = timedelta(days=session_duration_days)
        logger.info("auth_initialized", session_duration_days=session_duration_days)

    def _get_db(self) -> Session:
        """Get database session"""
        return SessionLocal()

    def _hash_password(self, password: str) -> str:
        """Simple SHA256 password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _generate_token(self) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(32)

    def register_user(
        self,
        username: str,
        password: str,
        display_name: Optional[str] = None
    ) -> tuple[bool, str]:
        """
        Register a new user
        Returns: (success, message)
        """
        db = self._get_db()
        try:
            # Check if user exists
            existing = db.query(AuthUser).filter(AuthUser.username == username).first()
            if existing:
                return False, "Username already exists"

            # Create new user
            # Note: Store as timezone-naive for SQLite compatibility
            password_hash = self._hash_password(password)
            user = AuthUser(
                username=username,
                password_hash=password_hash,
                display_name=display_name or username,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
            db.add(user)
            db.commit()

            logger.info("user_registered", username=username)
            return True, "User registered successfully"

        except Exception as e:
            db.rollback()
            logger.error("register_error", username=username, error=str(e))
            return False, f"Registration failed: {str(e)}"
        finally:
            db.close()

    def login(self, username: str, password: str) -> tuple[bool, Optional[str], str]:
        """
        Authenticate user and create session
        Returns: (success, token, message)
        """
        db = self._get_db()
        try:
            # Find user
            user = db.query(AuthUser).filter(AuthUser.username == username).first()
            if not user:
                return False, None, "Invalid username or password"

            # Verify password
            password_hash = self._hash_password(password)
            if user.password_hash != password_hash:
                return False, None, "Invalid username or password"

            # Create session
            # Note: Store as timezone-naive for SQLite compatibility
            token = self._generate_token()
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            expires_at = now_utc + self.session_duration

            session = AuthSession(
                token=token,
                username=username,
                created_at=now_utc,
                expires_at=expires_at,
                last_activity=now_utc
            )
            db.add(session)

            # Update last login
            user.last_login = now_utc

            db.commit()

            logger.info(
                "user_logged_in",
                username=username,
                token=token[:8],
                expires_in_days=(expires_at - now_utc).days
            )
            return True, token, "Login successful"

        except Exception as e:
            db.rollback()
            logger.error("login_error", username=username, error=str(e))
            return False, None, f"Login failed: {str(e)}"
        finally:
            db.close()

    def logout(self, token: str) -> bool:
        """
        Logout user by deleting session
        Returns: success
        """
        db = self._get_db()
        try:
            session = db.query(AuthSession).filter(AuthSession.token == token).first()
            if session:
                username = session.username
                db.delete(session)
                db.commit()
                logger.info("user_logged_out", username=username, token=token[:8])
                return True
            return False
        except Exception as e:
            db.rollback()
            logger.error("logout_error", token=token[:8], error=str(e))
            return False
        finally:
            db.close()

    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify token and return username if valid
        Also updates last_activity timestamp
        Returns: username or None
        """
        if not token:
            return None

        db = self._get_db()
        try:
            session = db.query(AuthSession).filter(AuthSession.token == token).first()
            if not session:
                return None

            # Check if expired
            # Note: Use timezone-naive datetime for SQLite compatibility
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            
            if now > session.expires_at:
                # Delete expired session
                db.delete(session)
                db.commit()
                logger.info("session_expired", token=token[:8])
                return None

            # Update last activity
            session.last_activity = now
            db.commit()

            return session.username

        except Exception as e:
            logger.error("verify_token_error", token=token[:8], error=str(e))
            return None
        finally:
            db.close()

    def get_user_info(self, identifier: str) -> Optional[dict]:
        """
        Get user info by username
        Returns: user dict or None
        """
        db = self._get_db()
        try:
            user = db.query(AuthUser).filter(AuthUser.username == identifier).first()
            if user:
                return {
                    "user_id": user.username,
                    "username": user.username,
                    "display_name": user.display_name,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_login": user.last_login.isoformat() if user.last_login else None
                }
            return None
        finally:
            db.close()

    def list_users(self) -> List[dict]:
        """
        List all registered users
        Returns: list of user dicts
        """
        db = self._get_db()
        try:
            users = db.query(AuthUser).all()
            return [
                {
                    "user_id": u.username,
                    "username": u.username,
                    "display_name": u.display_name,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "last_login": u.last_login.isoformat() if u.last_login else None
                }
                for u in users
            ]
        finally:
            db.close()

    def cleanup_expired_sessions(self) -> int:
        """
        Delete all expired sessions
        Returns: number of sessions deleted
        """
        db = self._get_db()
        try:
            # Use timezone-naive datetime for SQLite compatibility
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            expired = db.query(AuthSession).filter(AuthSession.expires_at < now).all()
            count = len(expired)

            for session in expired:
                db.delete(session)

            db.commit()

            if count > 0:
                logger.info("expired_sessions_cleaned", count=count)

            return count
        except Exception as e:
            db.rollback()
            logger.error("cleanup_error", error=str(e))
            return 0
        finally:
            db.close()


# Global instance
_auth_manager: Optional[SimpleAuthManager] = None


def create_default_user_if_needed():
    """
    Create default test user if no users exist.
    Call this after init_db() to ensure tables exist.
    """
    auth_mgr = get_auth_manager()
    db = SessionLocal()
    try:
        user_count = db.query(AuthUser).count()
        if user_count == 0:
            auth_mgr.register_user("testuser", "testpass", "Test User")
            logger.info("default_user_created", username="testuser")
    except Exception as e:
        logger.error("default_user_creation_failed", error=str(e))
    finally:
        db.close()


def get_auth_manager() -> SimpleAuthManager:
    """Get or create global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SimpleAuthManager(session_duration_days=30)
    
    return _auth_manager

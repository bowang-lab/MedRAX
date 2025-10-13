"""
Simple authentication system for MedRAX Web Platform
No complex security - just username/password for user identification
"""

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from logger_config import get_logger

logger = get_logger(__name__)


class SimpleAuthManager:
    """
    Super simple authentication manager.
    Stores users in JSON file, uses basic password hashing.
    
    NOTE: This is intentionally simple. For production with sensitive data,
    consider using proper authentication (OAuth, JWT, etc.)
    """
    
    def __init__(self, users_file: str = "users.json"):
        self.users_file = Path(users_file)
        self.users: Dict[str, Dict] = {}
        self.sessions: Dict[str, Dict] = {}  # token -> {user_id, expires_at}
        self.session_duration = timedelta(hours=24)  # 24 hour sessions
        
        # Load existing users
        self._load_users()
        
        logger.info("auth_initialized", users_count=len(self.users))
    
    def _load_users(self):
        """Load users from JSON file"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
                logger.info("users_loaded", count=len(self.users))
            except Exception as e:
                logger.error("users_load_error", error=str(e))
                self.users = {}
        else:
            self.users = {}
            self._save_users()
    
    def _save_users(self):
        """Save users to JSON file"""
        try:
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
            logger.info("users_saved", count=len(self.users))
        except Exception as e:
            logger.error("users_save_error", error=str(e))
    
    def _hash_password(self, password: str) -> str:
        """Simple password hashing using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, password: str, display_name: str = "") -> tuple[bool, str]:
        """
        Register a new user.
        
        Args:
            username: Unique username (will be user_id)
            password: Any password (no strength requirements)
            display_name: Optional display name
            
        Returns:
            (success: bool, message: str)
        """
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters"
        
        if username in self.users:
            return False, "Username already exists"
        
        # Create user
        self.users[username] = {
            "user_id": username,
            "password_hash": self._hash_password(password),
            "display_name": display_name or username,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": None
        }
        
        self._save_users()
        logger.info("user_registered", username=username)
        
        return True, "User registered successfully"
    
    def login(self, username: str, password: str) -> tuple[bool, Optional[str], str]:
        """
        Login user and create session token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            (success: bool, token: Optional[str], message: str)
        """
        # Check if user exists
        if username not in self.users:
            logger.warning("login_failed", username=username, reason="user_not_found")
            return False, None, "Invalid username or password"
        
        user = self.users[username]
        
        # Verify password
        password_hash = self._hash_password(password)
        if password_hash != user["password_hash"]:
            logger.warning("login_failed", username=username, reason="wrong_password")
            return False, None, "Invalid username or password"
        
        # Create session token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + self.session_duration
        
        self.sessions[token] = {
            "user_id": username,
            "expires_at": expires_at.isoformat(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Update last login
        user["last_login"] = datetime.now(timezone.utc).isoformat()
        self._save_users()
        
        logger.info("user_logged_in", username=username, token=token[:8])
        
        return True, token, "Login successful"
    
    def logout(self, token: str) -> bool:
        """
        Logout user by removing session token.
        
        Args:
            token: Session token
            
        Returns:
            success: bool
        """
        if token in self.sessions:
            user_id = self.sessions[token]["user_id"]
            del self.sessions[token]
            logger.info("user_logged_out", user_id=user_id, token=token[:8])
            return True
        return False
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify session token and return user_id.
        
        Args:
            token: Session token
            
        Returns:
            user_id if valid, None if invalid/expired
        """
        if token not in self.sessions:
            return None
        
        session = self.sessions[token]
        expires_at = datetime.fromisoformat(session["expires_at"])
        
        # Check if expired
        if datetime.now(timezone.utc) > expires_at:
            del self.sessions[token]
            logger.info("session_expired", token=token[:8])
            return None
        
        return session["user_id"]
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """
        Get user information (without password hash).
        
        Args:
            username: Username
            
        Returns:
            User info dict or None
        """
        if username not in self.users:
            return None
        
        user = self.users[username].copy()
        user.pop("password_hash", None)  # Don't expose password hash
        return user
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        now = datetime.now(timezone.utc)
        expired = []
        
        for token, session in self.sessions.items():
            expires_at = datetime.fromisoformat(session["expires_at"])
            if now > expires_at:
                expired.append(token)
        
        for token in expired:
            del self.sessions[token]
        
        if expired:
            logger.info("sessions_cleaned", count=len(expired))
    
    def list_users(self) -> list:
        """List all users (admin function)"""
        return [
            {
                "user_id": user["user_id"],
                "display_name": user["display_name"],
                "created_at": user["created_at"],
                "last_login": user["last_login"]
            }
            for user in self.users.values()
        ]


# Global auth manager instance
_auth_manager: Optional[SimpleAuthManager] = None


def get_auth_manager() -> SimpleAuthManager:
    """Get or create global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SimpleAuthManager()
    return _auth_manager


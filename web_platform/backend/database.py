"""
Database models and configuration for MedRAX Web Platform
Provides persistent storage for users, chats, messages, images, and tool results
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from logger_config import get_logger

logger = get_logger(__name__)

# Database configuration
DATABASE_URL = "sqlite:///./medrax.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Models
class AuthUser(Base):
    """Authenticated user (doctor/clinician) model"""
    __tablename__ = "auth_users"

    username = Column(String, primary_key=True, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    sessions = relationship("AuthSession", back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    """User session/token model"""
    __tablename__ = "auth_sessions"

    token = Column(String, primary_key=True, index=True)
    username = Column(String, ForeignKey("auth_users.username", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("AuthUser", back_populates="sessions")


class User(Base):
    """User/Patient model"""
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    patient_name = Column(String, default="")
    patient_age = Column(String, default="")
    patient_gender = Column(String, default="")
    patient_notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    """Chat session model"""
    __tablename__ = "chats"

    chat_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, default="New Chat")
    description = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_access = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    message_count = Column(Integer, default=0)
    image_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    images = relationship("Image", back_populates="chat", cascade="all, delete-orphan")
    tool_results = relationship("ToolResult", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    """Chat message model"""
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.chat_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    chat = relationship("Chat", back_populates="messages")


class Image(Base):
    """Uploaded image model"""
    __tablename__ = "images"

    image_id = Column(String, primary_key=True, index=True)
    chat_id = Column(String, ForeignKey("chats.chat_id", ondelete="CASCADE"), nullable=False, index=True)
    file_path = Column(String, nullable=False)  # Original file path
    display_path = Column(String, nullable=False)  # Display file path (after DICOM conversion)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    chat = relationship("Chat", back_populates="images")


class ToolResult(Base):
    """Tool execution result model - stores all tool execution history"""
    __tablename__ = "tool_results"

    result_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    chat_id = Column(String, ForeignKey("chats.chat_id", ondelete="CASCADE"), nullable=False, index=True)
    execution_id = Column(String, nullable=False, index=True)  # UUID for this specific execution
    request_id = Column(String, nullable=True, index=True)  # UUID for the analysis request
    tool_name = Column(String, nullable=False)
    result_data = Column(JSON, nullable=False)  # Stored as JSON
    metadata = Column(JSON, nullable=True)  # Additional metadata (e.g., image_paths)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    chat = relationship("Chat", back_populates="tool_results")


# Database utilities
def init_db():
    """Initialize database tables and create default user"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("database_initialized", message="Database tables created successfully")
        
        # Create default user after tables are created
        try:
            from auth import create_default_user_if_needed
            create_default_user_if_needed()
        except Exception as auth_error:
            logger.warning("default_user_setup_skipped", error=str(auth_error))
    except Exception as e:
        logger.error("database_init_error", error=str(e))
        raise


def get_db():
    """Get database session (dependency injection for FastAPI)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database operations
class DatabaseManager:
    """Manager for database operations"""

    @staticmethod
    def create_user(db, user_id: str, patient_name: str = "", patient_age: str = "",
                   patient_gender: str = "", patient_notes: str = "") -> User:
        """Create or update a user"""
        user = db.query(User).filter(User.user_id == user_id).first()
        if user:
            # Update existing user
            user.patient_name = patient_name or user.patient_name
            user.patient_age = patient_age or user.patient_age
            user.patient_gender = patient_gender or user.patient_gender
            user.patient_notes = patient_notes or user.patient_notes
        else:
            # Create new user
            user = User(
                user_id=user_id,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                patient_notes=patient_notes
            )
            db.add(user)

        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def create_chat(db, chat_id: str, user_id: str, name: str = "New Chat",
                   description: str = "") -> Chat:
        """Create a new chat"""
        # Ensure user exists
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = DatabaseManager.create_user(db, user_id)

        chat = Chat(
            chat_id=chat_id,
            user_id=user_id,
            name=name,
            description=description
        )
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return chat

    @staticmethod
    def get_chat(db, chat_id: str) -> Optional[Chat]:
        """Get a chat by ID"""
        return db.query(Chat).filter(Chat.chat_id == chat_id).first()

    @staticmethod
    def get_user_chats(db, user_id: str):
        """Get all chats for a user"""
        return db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.last_access.desc()).all()

    @staticmethod
    def update_chat_access(db, chat_id: str):
        """Update last access time for a chat"""
        chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
        if chat:
            chat.last_access = datetime.now(timezone.utc)
            db.commit()

    @staticmethod
    def add_message(db, message_id: str, chat_id: str, role: str, content: str) -> Message:
        """Add a message to a chat"""
        message = Message(
            message_id=message_id,
            chat_id=chat_id,
            role=role,
            content=content
        )
        db.add(message)

        # Update chat message count
        chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
        if chat:
            chat.message_count += 1

        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_chat_messages(db, chat_id: str):
        """Get all messages for a chat"""
        return db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp).all()

    @staticmethod
    def add_image(db, image_id: str, chat_id: str, file_path: str, display_path: str) -> Image:
        """Add an image to a chat"""
        image = Image(
            image_id=image_id,
            chat_id=chat_id,
            file_path=file_path,
            display_path=display_path
        )
        db.add(image)

        # Update chat image count
        chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
        if chat:
            chat.image_count += 1

        db.commit()
        db.refresh(image)
        return image

    @staticmethod
    def get_chat_images(db, chat_id: str):
        """Get all images for a chat"""
        return db.query(Image).filter(Image.chat_id == chat_id).order_by(Image.uploaded_at).all()

    @staticmethod
    def add_tool_result(db, result_id: str, chat_id: str, tool_name: str, result_data: dict) -> ToolResult:
        """Add a tool result to a chat"""
        tool_result = ToolResult(
            result_id=result_id,
            chat_id=chat_id,
            tool_name=tool_name,
            result_data=result_data
        )
        db.add(tool_result)
        db.commit()
        db.refresh(tool_result)
        return tool_result

    @staticmethod
    def get_chat_tool_results(db, chat_id: str):
        """Get all tool results for a chat"""
        return db.query(ToolResult).filter(ToolResult.chat_id == chat_id).order_by(ToolResult.created_at).all()

    @staticmethod
    def delete_chat(db, chat_id: str) -> bool:
        """Delete a chat and all associated data"""
        chat = db.query(Chat).filter(Chat.chat_id == chat_id).first()
        if chat:
            db.delete(chat)
            db.commit()
            return True
        return False


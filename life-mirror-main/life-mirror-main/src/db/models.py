from sqlalchemy import Boolean, DateTime, func
from sqlalchemy import Column, String, Integer, Text, JSON, BigInteger, TIMESTAMP, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True)
    public_alias = Column(String(80))
    opt_in_public_analysis = Column(Boolean, nullable=False, default=False)
    password_hash = Column(Text)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime)

    # Relationship to media
    media = relationship("Media", back_populates="user")
    

class Media(Base):
    __tablename__ = 'media'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    media_type = Column(String(10))
    storage_url = Column(Text, nullable=False)
    storage_key = Column(String(512))
    thumbnail_url = Column(Text)
    keyframes = Column(JSON)
    size_bytes = Column(BigInteger)
    mime = Column(String(255))
    media_metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="media")


class Embedding(Base):
    __tablename__ = "embeddings"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_id = Column(String(36), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    vector = Column(Text, nullable=False)  # Store as JSON string
    model = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())


class Face(Base):
    __tablename__ = "faces"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_id = Column(String(36), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    bbox = Column(Text, nullable=False)  # Store as JSON string [x,y,w,h]
    landmarks = Column(JSON)
    crop_url = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class Detection(Base):
    __tablename__ = "detections"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    media_id = Column(String(36), ForeignKey("media.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(255), nullable=False)
    score = Column(Float)
    bbox = Column(Text)  # Store as JSON string [x,y,w,h]
    created_at = Column(DateTime, server_default=func.now())


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)  # 'analysis_complete', 'social', 'system'
    is_read = Column(Boolean, nullable=False, default=False)
    data = Column(JSON)  # Additional notification data
    created_at = Column(DateTime, server_default=func.now())
    
    user = relationship("User")

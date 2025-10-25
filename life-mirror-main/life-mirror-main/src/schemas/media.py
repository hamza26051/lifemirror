from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class MediaCreate(BaseModel):
    storage_url: str
    storage_key: Optional[str] = None
    media_type: str = Field(..., pattern=r"^(image|video)$")
    mime: str
    size_bytes: int
    metadata: Optional[Dict[str, Any]] = None

class MediaResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    media_type: str
    storage_url: str
    storage_key: Optional[str]
    thumbnail_url: Optional[str]
    keyframes: Optional[Dict[str, Any]]
    size_bytes: int
    mime: str
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True

class MediaAnalysisRequest(BaseModel):
    media_id: uuid.UUID
    options: Optional[Dict[str, Any]] = {}

class PresignedUrlRequest(BaseModel):
    filename: str
    content_type: str
    size_bytes: int

class PresignedUrlResponse(BaseModel):
    upload_url: str
    storage_key: str
    expires_in: int = 3600

# Embedding schemas
class EmbeddingCreate(BaseModel):
    media_id: uuid.UUID
    vector: List[float]
    model: str

class EmbeddingResponse(BaseModel):
    id: uuid.UUID
    media_id: uuid.UUID
    model: str
    created_at: datetime

    class Config:
        from_attributes = True

# Face schemas
class FaceCreate(BaseModel):
    media_id: uuid.UUID
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    landmarks: Optional[Dict[str, Any]] = None
    crop_url: Optional[str] = None

class FaceResponse(BaseModel):
    id: uuid.UUID
    media_id: uuid.UUID
    bbox: List[float]
    landmarks: Optional[Dict[str, Any]]
    crop_url: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# Detection schemas
class DetectionCreate(BaseModel):
    media_id: uuid.UUID
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    bbox: Optional[List[float]] = Field(None, min_items=4, max_items=4)

class DetectionResponse(BaseModel):
    id: uuid.UUID
    media_id: uuid.UUID
    label: str
    score: float
    bbox: Optional[List[float]]
    created_at: datetime

    class Config:
        from_attributes = True
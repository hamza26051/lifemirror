from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

# Analysis request schemas
class AnalysisRequest(BaseModel):
    media_id: uuid.UUID
    user_consent: Optional[Dict[str, bool]] = Field(default_factory=lambda: {
        "face_analysis": True,
        "fashion_analysis": True,
        "posture_analysis": True,
        "bio_analysis": True,
        "detailed_analysis": True,
        "biometric_analysis": False,
        "technical_metadata": False
    })
    bio_text: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}

class BioAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    past_analyses: Optional[List[Dict[str, Any]]] = None

# Memory/Search schemas
class MemorySearchRequest(BaseModel):
    user_id: uuid.UUID
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    date_range: Optional[Dict[str, str]] = None  # {"start": "ISO", "end": "ISO"}
    analysis_types: Optional[List[str]] = None
    limit: Optional[int] = Field(default=10, le=50)
    min_similarity: Optional[float] = Field(default=0.1, ge=0.0, le=1.0)

# Comparison schemas
class ComparisonRequest(BaseModel):
    user_id: uuid.UUID
    current_analysis: Dict[str, Any]
    comparison_type: str = Field(..., pattern=r"^(celebrity|past_self|peer)$")
    target_id: Optional[str] = None
    time_range: Optional[str] = Field(default="3_months", pattern=r"^(1_month|3_months|1_year)$")

# Response schemas
class FaceAnalysisResponse(BaseModel):
    num_faces: int
    faces: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)

class FashionAnalysisResponse(BaseModel):
    style: str
    items: List[Dict[str, Any]]
    overall_rating: float = Field(..., ge=0.0, le=10.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

class PostureAnalysisResponse(BaseModel):
    alignment_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    keypoints: List[List[float]]
    tips: List[str]
    crop_url: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)

class BioAnalysisResponse(BaseModel):
    vibe_summary: str
    strengths: List[str]
    weaknesses: List[str]
    improvements: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)

class EmbeddingResponse(BaseModel):
    vector_id: str
    model: str
    dimension: int
    confidence: float = Field(..., ge=0.0, le=1.0)

# Final analysis response (from FormatterAgent)
class FinalAnalysisResponse(BaseModel):
    media_id: str
    timestamp: str
    
    # Summary scores
    overall_score: float = Field(..., ge=0.0, le=10.0)
    attractiveness_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    style_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    presence_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    
    # Human-readable summary
    summary: str
    key_insights: List[str] = Field(default_factory=list, max_items=5)
    recommendations: List[str] = Field(default_factory=list, max_items=5)
    
    # Detailed analysis (optional)
    detailed_analysis: Optional[Dict[str, Any]] = None
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    langsmith_run_id: Optional[str] = None
    
    # Warnings/disclaimers
    warnings: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True

# Memory search response
class MemorySearchResult(BaseModel):
    media_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: str
    analysis_summary: Optional[Dict[str, Any]] = None
    media_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

class MemorySearchResponse(BaseModel):
    query_type: str
    total_results: int
    results: List[MemorySearchResult] = Field(default_factory=list, max_items=50)
    search_metadata: Dict[str, Any] = Field(default_factory=dict)

# Comparison response
class ComparisonResult(BaseModel):
    comparison_type: str
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    
    # Comparison scores
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    style_similarity: float = Field(..., ge=0.0, le=1.0)
    presence_similarity: float = Field(..., ge=0.0, le=1.0)
    
    # Analysis
    similarities: List[str] = Field(default_factory=list)
    differences: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Metadata
    confidence: float = Field(..., ge=0.0, le=1.0)
    comparison_metadata: Dict[str, Any] = Field(default_factory=dict)

# Error responses
class ErrorResponse(BaseModel):
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ValidationErrorResponse(BaseModel):
    error: str = "Validation error"
    validation_errors: List[Dict[str, Any]]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Enhanced analysis response schemas
class FixitSuggestionsResponse(BaseModel):
    quick_tips: List[str]
    detailed_plan: str
    focus_areas: List[str]

class VibeAnalysisResponse(BaseModel):
    vibe_score: int = Field(..., ge=0, le=100)
    vibe_tags: List[str]
    personality_summary: str
    strengths: List[str]
    improvement_areas: List[str]

class PerceptionHistoryResponse(BaseModel):
    trend_summary: str
    score_trend: List[Dict[str, Any]]
    improvement_tags: List[str]
    decline_tags: List[str]

class SocialGraphResponse(BaseModel):
    cold_start: bool
    sample_size: int
    user_vibe_score: Optional[int]
    percentile: Dict[str, int]
    similar_users: List[Dict[str, Any]]
    complementary_users: List[Dict[str, Any]]

class SocialPerceptionResponse(BaseModel):
    summary_text: str
    tags: List[str]
    social_score: float = Field(..., ge=0.0, le=10.0)

class VibeComparisonResponse(BaseModel):
    summary: str
    better_media_id: int
    comparison_tags: List[str]
    score_difference: float

class ReverseAnalysisResponse(BaseModel):
    goal: str
    recommended_changes: List[str]
    avoid_list: List[str]
    action_plan: str

class EnhancedAnalysisResponse(BaseModel):
    """
    Comprehensive response that includes both core analysis and specialized analysis results.
    """
    success: bool
    core_analysis: FinalAnalysisResponse
    specialized_analysis: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional specialized components
    fixit_suggestions: Optional[FixitSuggestionsResponse] = None
    vibe_analysis: Optional[VibeAnalysisResponse] = None
    perception_history: Optional[PerceptionHistoryResponse] = None
    social_graph: Optional[SocialGraphResponse] = None
    social_perception: Optional[SocialPerceptionResponse] = None
    
class HybridAnalysisResponse(BaseModel):
    """
    Hybrid response combining LangGraph core analysis with specialized agent outputs.
    """
    success: bool
    core_analysis: Dict[str, Any]
    social_analysis: Dict[str, Any]
    enhancements: Dict[str, Any]
    metadata: Dict[str, Any]
    fallback_response: Optional[Dict[str, Any]] = None

class ComprehensiveUserProfile(BaseModel):
    """
    Complete user profile combining all available analysis types.
    """
    user_id: str
    profile_components: Dict[str, Any]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

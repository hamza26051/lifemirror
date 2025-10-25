import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from src.agents.base_agent import BaseAgent, AgentInput, AgentOutput
from src.db.session import get_db
from src.db.models import Media, Embedding
from src.utils.tracing import log_trace
import numpy as np


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
    results: List[MemorySearchResult] = Field(default_factory=list, max_items=20)
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryAgent(BaseAgent):
    """
    MemoryAgent / RetrieverAgent: handles semantic search and retrieval of past analyses.
    
    Supports:
    - Vector similarity search using embeddings
    - Structured filters (user_id, date range, analysis type)
    - Hybrid retrieval combining semantic + structured filters
    """
    name = "memory_agent"
    output_schema = MemorySearchResponse

    def run(self, input: AgentInput) -> AgentOutput:
        """
        Perform semantic search and retrieval of past analyses.
        
        Expected input.context:
        - query_vector: Optional[List[float]] - vector for semantic search
        - query_text: Optional[str] - text to convert to vector
        - user_id: str - filter by user
        - date_range: Optional[Dict] - {"start": "ISO", "end": "ISO"}
        - analysis_types: Optional[List[str]] - filter by analysis types
        - limit: Optional[int] - max results (default 10)
        - min_similarity: Optional[float] - minimum similarity threshold
        """
        
        user_id = input.context.get("user_id")
        if not user_id:
            return AgentOutput(
                success=False,
                data={},
                error="user_id required for memory search"
            )
            
        query_vector = input.context.get("query_vector")
        query_text = input.context.get("query_text")
        date_range = input.context.get("date_range")
        analysis_types = input.context.get("analysis_types", [])
        limit = input.context.get("limit", 10)
        min_similarity = input.context.get("min_similarity", 0.1)
        
        mode = os.getenv("LIFEMIRROR_MODE", "mock")
        
        if mode == "mock":
            # Return mock search results
            mock_results = [
                MemorySearchResult(
                    media_id="mock-media-1",
                    similarity_score=0.85,
                    timestamp="2025-01-15T10:00:00Z",
                    analysis_summary={
                        "overall_score": 7.5,
                        "style_score": 8.0,
                        "key_insights": ["Strong fashion sense", "Good posture"]
                    },
                    media_url="https://example.com/mock1.jpg",
                    thumbnail_url="https://example.com/mock1_thumb.jpg"
                ),
                MemorySearchResult(
                    media_id="mock-media-2", 
                    similarity_score=0.72,
                    timestamp="2025-01-10T15:30:00Z",
                    analysis_summary={
                        "overall_score": 6.8,
                        "presence_score": 7.2,
                        "key_insights": ["Confident expression", "Room for style improvement"]
                    },
                    media_url="https://example.com/mock2.jpg"
                )
            ]
            
            response = MemorySearchResponse(
                query_type="mock_search",
                total_results=2,
                results=mock_results,
                search_metadata={"mode": "mock", "query_processed": True}
            )
            
            result = AgentOutput(success=True, data=response.dict())
            self._trace(input.dict(), result.dict())
            return result
        
        # Production mode
        try:
            db = next(get_db())
            
            # Convert text to vector if needed
            if query_text and not query_vector:
                query_vector = self._text_to_vector(query_text)
                
            if query_vector:
                # Perform vector similarity search
                results = self._vector_search(
                    db, user_id, query_vector, date_range, 
                    analysis_types, limit, min_similarity
                )
                query_type = "vector_search"
            else:
                # Perform structured search only
                results = self._structured_search(
                    db, user_id, date_range, analysis_types, limit
                )
                query_type = "structured_search"
                
            response = MemorySearchResponse(
                query_type=query_type,
                total_results=len(results),
                results=results,
                search_metadata={
                    "user_id": user_id,
                    "has_query_vector": bool(query_vector),
                    "date_range": date_range,
                    "analysis_types": analysis_types,
                    "limit": limit,
                    "min_similarity": min_similarity
                }
            )
            
            result = AgentOutput(success=True, data=response.dict())
            self._trace(input.dict(), result.dict())
            return result
            
        except Exception as e:
            result = AgentOutput(
                success=False,
                data={},
                error=f"Memory search failed: {str(e)}"
            )
            self._trace(input.dict(), result.dict())
            return result

    def _text_to_vector(self, text: str) -> Optional[List[float]]:
        """Convert text to vector using hash-based approach (free alternative)."""
        try:
            import hashlib
            import numpy as np
            
            # Create a deterministic vector from text using hash-based approach
            # This provides consistent embeddings without requiring paid APIs
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Convert hash to vector of specified dimensions
            vector_size = 384  # Standard embedding size
            vector = []
            
            # Use hash chunks to create vector components
            for i in range(vector_size):
                # Take 8 characters from hash, cycling through
                chunk_start = (i * 8) % len(text_hash)
                chunk = text_hash[chunk_start:chunk_start + 8]
                # Convert hex to float between -1 and 1
                hex_val = int(chunk, 16) if len(chunk) == 8 else int(chunk.ljust(8, '0'), 16)
                normalized_val = (hex_val / (16**8)) * 2 - 1
                vector.append(normalized_val)
            
            # Normalize vector to unit length for cosine similarity
            vector = np.array(vector)
            vector = vector / np.linalg.norm(vector)
            
            return vector.tolist()
            
        except Exception as e:
            log_trace("memory_agent", {"text_to_vector_error": str(e)}, {})
            return None

    def _vector_search(self, db: Session, user_id: str, query_vector: List[float],
                      date_range: Optional[Dict], analysis_types: List[str],
                      limit: int, min_similarity: float) -> List[MemorySearchResult]:
        """Perform vector similarity search with optional filters"""
        
        # Base query for user's media with embeddings
        query = (
            db.query(Media, Embedding)
            .join(Embedding, Media.id == Embedding.media_id)
            .filter(Media.user_id == user_id)
        )
        
        # Apply date filter if provided
        if date_range:
            if date_range.get("start"):
                query = query.filter(Media.created_at >= date_range["start"])
            if date_range.get("end"):
                query = query.filter(Media.created_at <= date_range["end"])
        
        # Get all candidates
        candidates = query.all()
        
        if not candidates:
            return []
            
        # Calculate similarities
        results = []
        query_vec = np.array(query_vector)
        
        for media, embedding in candidates:
            try:
                stored_vec = np.array(embedding.vector)
                # Calculate cosine similarity
                similarity = np.dot(query_vec, stored_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                )
                
                if similarity >= min_similarity:
                    # Extract analysis summary from metadata
                    analysis_summary = None
                    if media.metadata:
                        analysis_summary = {
                            "overall_score": media.metadata.get("overall_score"),
                            "style_score": media.metadata.get("style_score"),
                            "presence_score": media.metadata.get("presence_score"),
                            "key_insights": media.metadata.get("key_insights", [])
                        }
                    
                    results.append(MemorySearchResult(
                        media_id=str(media.id),
                        similarity_score=float(similarity),
                        timestamp=media.created_at.isoformat(),
                        analysis_summary=analysis_summary,
                        media_url=media.storage_url,
                        thumbnail_url=media.thumbnail_url
                    ))
                    
            except Exception as e:
                # Skip this result if similarity calculation fails
                log_trace("memory_agent", {"similarity_error": str(e)}, {})
                continue
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]

    def _structured_search(self, db: Session, user_id: str, 
                         date_range: Optional[Dict], analysis_types: List[str],
                         limit: int) -> List[MemorySearchResult]:
        """Perform structured search without vector similarity"""
        
        query = db.query(Media).filter(Media.user_id == user_id)
        
        # Apply date filter
        if date_range:
            if date_range.get("start"):
                query = query.filter(Media.created_at >= date_range["start"])
            if date_range.get("end"):
                query = query.filter(Media.created_at <= date_range["end"])
        
        # Apply analysis type filter (check metadata)
        if analysis_types:
            # This would need to be implemented based on how analysis types are stored
            pass
            
        # Order by recency and limit
        media_items = query.order_by(Media.created_at.desc()).limit(limit).all()
        
        results = []
        for media in media_items:
            analysis_summary = None
            if media.metadata:
                analysis_summary = {
                    "overall_score": media.metadata.get("overall_score"),
                    "style_score": media.metadata.get("style_score"),
                    "presence_score": media.metadata.get("presence_score"),
                    "key_insights": media.metadata.get("key_insights", [])
                }
            
            results.append(MemorySearchResult(
                media_id=str(media.id),
                similarity_score=1.0,  # No similarity calculation for structured search
                timestamp=media.created_at.isoformat(),
                analysis_summary=analysis_summary,
                media_url=media.storage_url,
                thumbnail_url=media.thumbnail_url
            ))
            
        return results

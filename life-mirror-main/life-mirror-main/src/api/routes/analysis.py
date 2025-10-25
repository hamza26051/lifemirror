from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from src.db.session import get_db
from src.db.models import User, Media
from src.api.deps import get_current_user
from src.schemas.analysis import (
    AnalysisRequest, BioAnalysisRequest, MemorySearchRequest, ComparisonRequest,
    FinalAnalysisResponse, BioAnalysisResponse, MemorySearchResponse, ComparisonResult,
    ErrorResponse, EnhancedAnalysisResponse, HybridAnalysisResponse, ComprehensiveUserProfile
)
from src.agents.orchestrator import Orchestrator, EnhancedOrchestrator
from src.agents.graph_workflow import GraphExecutor, EnhancedGraphExecutor
from src.core.rate_limit import rate_limit
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/analyze", response_model=FinalAnalysisResponse)
@rate_limit("analysis", max_calls=10, window_seconds=3600)  # 10 analyses per hour
async def analyze_media(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Complete media analysis using the full agent pipeline.
    Returns formatted analysis ready for frontend consumption.
    """
    try:
        # Verify media belongs to user
        media = db.query(Media).filter(
            Media.id == str(request.media_id),
            Media.user_id == str(current_user.id)
        ).first()
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        # Prepare analysis context
        context = {
            "user_id": str(current_user.id),
            "user_consent": request.user_consent,
            "bio_text": request.bio_text,
            "request_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": "API",
                "analysis_version": "1.0"
            },
            **request.options
        }
        
        # Use GraphExecutor for full pipeline
        graph_executor = GraphExecutor()
        result = graph_executor.execute(
            media_id=str(request.media_id),
            url=media.storage_url,
            context=context
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        # Save analysis results to media metadata in background
        background_tasks.add_task(
            save_analysis_results,
            db_session=db,
            media_id=request.media_id,
            analysis_data=result.get("data", {})
        )
        
        return FinalAnalysisResponse(**result["data"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for media {request.media_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal analysis error")

@router.post("/analyze/bio", response_model=BioAnalysisResponse)
@rate_limit("bio_analysis", max_calls=20, window_seconds=3600)  # 20 bio analyses per hour
async def analyze_bio_text(
    request: BioAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze bio/profile text for vibe and improvement suggestions.
    """
    try:
        orchestrator = Orchestrator()
        result = orchestrator.analyze_bio_text(
            text=request.text,
            user_id=str(current_user.id),
            past_analyses=request.past_analyses
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Bio analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        return BioAnalysisResponse(**result["data"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bio analysis error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Bio analysis error")

@router.post("/search", response_model=MemorySearchResponse)
@rate_limit("memory_search", max_calls=50, window_seconds=3600)  # 50 searches per hour
async def search_memory(
    request: MemorySearchRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Search user's past analyses using semantic and structured search.
    """
    try:
        # Verify user can only search their own data
        if request.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Can only search your own analyses")
        
        orchestrator = Orchestrator()
        result = orchestrator.search_memory(
            user_id=str(request.user_id),
            query_text=request.query_text,
            query_vector=request.query_vector,
            date_range=request.date_range,
            analysis_types=request.analysis_types,
            limit=request.limit,
            min_similarity=request.min_similarity
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Memory search failed: {result.get('error', 'Unknown error')}"
            )
        
        return MemorySearchResponse(**result["data"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory search error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Memory search error")

@router.post("/compare", response_model=ComparisonResult)
@rate_limit("comparison", max_calls=15, window_seconds=3600)  # 15 comparisons per hour
async def compare_analysis(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compare current analysis with celebrities, past self, or peers.
    """
    try:
        # Verify user can only compare their own data
        if request.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Can only compare your own analyses")
        
        orchestrator = Orchestrator()
        result = orchestrator.compare_analysis(
            user_id=str(request.user_id),
            current_analysis=request.current_analysis,
            comparison_type=request.comparison_type,
            target_id=request.target_id,
            time_range=request.time_range
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Comparison failed: {result.get('error', 'Unknown error')}"
            )
        
        return ComparisonResult(**result["data"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Comparison error")

@router.get("/celebrities", response_model=Dict[str, Any])
async def list_celebrities():
    """
    List available celebrities for comparison.
    """
    # In a real implementation, this would come from a database
    celebrities = {
        "celeb_1": {
            "id": "celeb_1",
            "name": "Taylor Swift",
            "category": "music",
            "description": "Pop superstar known for confident stage presence"
        },
        "celeb_2": {
            "id": "celeb_2", 
            "name": "Ryan Gosling",
            "category": "acting",
            "description": "Versatile actor with refined style"
        },
        "celeb_3": {
            "id": "celeb_3",
            "name": "Emma Stone",
            "category": "acting", 
            "description": "Award-winning actress with approachable charm"
        },
        "celeb_4": {
            "id": "celeb_4",
            "name": "Michael B. Jordan",
            "category": "acting",
            "description": "Charismatic actor with strong presence"
        }
    }
    
    return {
        "celebrities": list(celebrities.values()),
        "total": len(celebrities),
        "categories": list(set(c["category"] for c in celebrities.values()))
    }

@router.get("/analysis/{media_id}/history", response_model=Dict[str, Any])
async def get_analysis_history(
    media_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis history for a specific media item.
    """
    try:
        # Verify media belongs to user
        media = db.query(Media).filter(
            Media.id == media_id,
            Media.user_id == current_user.id
        ).first()
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        # Extract analysis history from metadata
        metadata = media.metadata or {}
        analysis_history = metadata.get("analysis_history", [])
        
        return {
            "media_id": str(media_id),
            "analysis_count": len(analysis_history),
            "history": analysis_history,
            "latest_analysis": analysis_history[-1] if analysis_history else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis history error for media {media_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis history")

# Background task functions
async def save_analysis_results(db_session: Session, media_id: uuid.UUID, analysis_data: Dict[str, Any]):
    """
    Save analysis results to media metadata in background.
    """
    try:
        media = db_session.query(Media).filter(Media.id == media_id).first()
        if media:
            # Initialize metadata if needed
            if not media.metadata:
                media.metadata = {}
            
            # Save current analysis
            media.metadata["latest_analysis"] = analysis_data
            media.metadata["last_analyzed"] = datetime.utcnow().isoformat()
            
            # Add to analysis history
            if "analysis_history" not in media.metadata:
                media.metadata["analysis_history"] = []
            
            media.metadata["analysis_history"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "overall_score": analysis_data.get("overall_score"),
                    "confidence": analysis_data.get("confidence"),
                    "key_insights": analysis_data.get("key_insights", [])[:3]  # Top 3 insights
                }
            })
            
            # Keep only last 10 analyses in history
            media.metadata["analysis_history"] = media.metadata["analysis_history"][-10:]
            
            db_session.commit()
            
    except Exception as e:
        logger.error(f"Failed to save analysis results for media {media_id}: {str(e)}")
        db_session.rollback()


# Enhanced Analysis Endpoints
@router.post("/analyze/enhanced", response_model=EnhancedAnalysisResponse)
@rate_limit("enhanced_analysis", max_calls=5, window_seconds=3600)  # 5 enhanced analyses per hour
async def enhanced_analyze_media(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enhanced media analysis combining core pipeline with specialized agents.
    Provides comprehensive analysis including fixit suggestions, vibe analysis, and social insights.
    """
    try:
        # Verify media belongs to user
        media = db.query(Media).filter(
            Media.id == request.media_id,
            Media.user_id == current_user.id
        ).first()
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        # Prepare analysis context
        context = {
            "user_id": str(current_user.id),
            "user_consent": request.user_consent,
            "bio_text": request.bio_text,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "request_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": "Enhanced API",
                "analysis_version": "2.0"
            },
            **request.options
        }
        
        # Use EnhancedOrchestrator for comprehensive analysis
        enhanced_orchestrator = EnhancedOrchestrator()
        result = enhanced_orchestrator.full_analysis_with_enhancements(
            media_id=str(request.media_id),
            url=media.storage_url,
            context=context
        )
        
        # Save enhanced analysis results to media metadata in background
        background_tasks.add_task(
            save_enhanced_analysis_results,
            db_session=db,
            media_id=request.media_id,
            analysis_data=result
        )
        
        return EnhancedAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis error for media {request.media_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal enhanced analysis error")


@router.post("/analyze/hybrid", response_model=HybridAnalysisResponse)
@rate_limit("hybrid_analysis", max_calls=5, window_seconds=3600)  # 5 hybrid analyses per hour
async def hybrid_analyze_media(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Hybrid analysis using EnhancedGraphExecutor with LangGraph workflow.
    Combines core analysis with specialized agent outputs in a structured workflow.
    """
    try:
        # Verify media belongs to user
        media = db.query(Media).filter(
            Media.id == request.media_id,
            Media.user_id == current_user.id
        ).first()
        
        if not media:
            raise HTTPException(status_code=404, detail="Media not found")
        
        # Prepare analysis context
        context = {
            "user_id": str(current_user.id),
            "user_consent": request.user_consent,
            "bio_text": request.bio_text,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "request_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user_agent": "Hybrid API",
                "analysis_version": "2.0"
            },
            **request.options
        }
        
        # Use EnhancedGraphExecutor for workflow-based analysis
        enhanced_graph = EnhancedGraphExecutor()
        result = enhanced_graph.execute_enhanced(
            media_id=str(request.media_id),
            url=media.storage_url,
            context=context
        )
        
        # Save hybrid analysis results to media metadata in background
        background_tasks.add_task(
            save_enhanced_analysis_results,
            db_session=db,
            media_id=request.media_id,
            analysis_data=result
        )
        
        return HybridAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hybrid analysis error for media {request.media_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal hybrid analysis error")


@router.get("/profile/comprehensive", response_model=ComprehensiveUserProfile)
@rate_limit("user_profile", max_calls=10, window_seconds=3600)  # 10 profile requests per hour
async def get_comprehensive_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive user profile combining all available analysis types.
    """
    try:
        enhanced_orchestrator = EnhancedOrchestrator()
        profile = enhanced_orchestrator.get_comprehensive_user_profile(
            user_id=str(current_user.id),
            context={"db": db}
        )
        
        return ComprehensiveUserProfile(**profile)
        
    except Exception as e:
        logger.error(f"Profile generation error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal profile generation error")


@router.post("/reverse-goal")
async def reverse_goal_analysis(
    goal: str,
    recent_limit: int = 5,
    current_user: User = Depends(get_current_user)
):
    """
    Reverse analysis to determine how to achieve a specific perception goal.
    """
    try:
        enhanced_orchestrator = EnhancedOrchestrator()
        result = enhanced_orchestrator.reverse_goal_analysis(
            user_id=str(current_user.id),
            goal=goal,
            context={"recent_limit": recent_limit}
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Reverse goal analysis error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal reverse analysis error")


@router.post("/compare-media")
async def compare_media_vibes(
    media_id_1: uuid.UUID,
    media_id_2: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Compare social vibes between two media items.
    """
    try:
        # Verify both media items belong to user
        media_1 = db.query(Media).filter(
            Media.id == media_id_1,
            Media.user_id == current_user.id
        ).first()
        
        media_2 = db.query(Media).filter(
            Media.id == media_id_2,
            Media.user_id == current_user.id
        ).first()
        
        if not media_1 or not media_2:
            raise HTTPException(status_code=404, detail="One or both media items not found")
        
        enhanced_orchestrator = EnhancedOrchestrator()
        result = enhanced_orchestrator.compare_media_vibes(
            media_id_1=str(media_id_1),
            media_id_2=str(media_id_2)
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media comparison error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal comparison error")


@router.post("/notifications/generate")
async def generate_user_notifications(
    current_user: User = Depends(get_current_user)
):
    """
    Generate personalized notifications for user.
    """
    try:
        enhanced_orchestrator = EnhancedOrchestrator()
        result = enhanced_orchestrator.generate_notifications(
            user_id=str(current_user.id)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Notification generation error for user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal notification generation error")


async def save_enhanced_analysis_results(db_session: Session, media_id: uuid.UUID, analysis_data: Dict[str, Any]):
    """
    Background task to save enhanced analysis results to media metadata.
    """
    try:
        with db_session:
            media = db_session.query(Media).filter(Media.id == media_id).first()
            if media:
                if not media.metadata:
                    media.metadata = {}
                
                # Save enhanced analysis data
                media.metadata["enhanced_analysis"] = analysis_data
                media.metadata["last_enhanced_analysis"] = datetime.utcnow().isoformat()
                
                # Maintain analysis history
                if "enhanced_analysis_history" not in media.metadata:
                    media.metadata["enhanced_analysis_history"] = []
                
                media.metadata["enhanced_analysis_history"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis_data": analysis_data,
                    "analysis_type": analysis_data.get("metadata", {}).get("processing_mode", "enhanced")
                })
                
                # Keep only last 5 enhanced analyses
                if len(media.metadata["enhanced_analysis_history"]) > 5:
                    media.metadata["enhanced_analysis_history"] = media.metadata["enhanced_analysis_history"][-5:]
                
                db_session.commit()
                
    except Exception as e:
        logger.error(f"Failed to save enhanced analysis results for media {media_id}: {str(e)}")
        db_session.rollback()

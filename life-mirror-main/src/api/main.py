from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
import asyncio
from datetime import datetime
import uuid
import os

from src.orchestration.workflow_manager import (
    workflow_manager, 
    WorkflowRequest, 
    WorkflowType, 
    WorkflowStatus
)
from src.orchestration.langgraph_orchestrator import orchestrator
from src.utils.tracing import log_trace
from src.db.session import get_db
from src.db.models import Media, User, Analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Life Mirror API",
    description="Advanced AI-powered personal analysis platform with 18-agent architecture",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for analysis"""
    media_id: str = Field(description="ID of the media to analyze")
    user_id: Optional[str] = Field(None, description="User ID for personalized analysis")
    workflow_type: WorkflowType = Field(WorkflowType.COMPREHENSIVE, description="Type of workflow to execute")
    selected_agents: Optional[List[str]] = Field(None, description="Custom agent selection for workflow")
    priority: int = Field(5, ge=1, le=10, description="Analysis priority (1-10)")
    timeout_minutes: int = Field(10, ge=1, le=60, description="Timeout in minutes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AnalysisResponse(BaseModel):
    """Response model for analysis"""
    workflow_id: str = Field(description="Unique workflow execution ID")
    status: str = Field(description="Current status of the analysis")
    message: str = Field(description="Status message")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")

class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    id: str
    status: str
    workflow_type: str
    media_id: str
    user_id: Optional[str]
    progress: float
    current_stage: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    result: Optional[Dict[str, Any]]
    execution_log: List[str]
    metadata: Dict[str, Any]

class MediaUploadResponse(BaseModel):
    """Response model for media upload"""
    media_id: str
    filename: str
    file_size: int
    content_type: str
    upload_time: str
    message: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "orchestrator_status": "active",
        "active_workflows": len(workflow_manager.active_executions)
    }

# Media upload endpoint
@app.post("/upload", response_model=MediaUploadResponse)
async def upload_media(file: UploadFile = File(...), user_id: Optional[str] = None):
    """Upload media file for analysis"""
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "text/plain"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Generate unique media ID
        media_id = str(uuid.uuid4())
        
        # Create upload directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(upload_dir, f"{media_id}_{file.filename}")
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Store media record in database
        with get_db() as db:
            media = Media(
                id=media_id,
                filename=file.filename,
                file_path=file_path,
                content_type=file.content_type,
                file_size=len(content),
                user_id=user_id,
                upload_time=datetime.now()
            )
            db.add(media)
            db.commit()
        
        logger.info(f"Media uploaded: {media_id} ({file.filename})")
        
        return MediaUploadResponse(
            media_id=media_id,
            filename=file.filename,
            file_size=len(content),
            content_type=file.content_type,
            upload_time=datetime.now().isoformat(),
            message="File uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Media upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest):
    """Start comprehensive analysis using the 18-agent architecture"""
    try:
        # Validate media exists
        with get_db() as db:
            media = db.query(Media).filter(Media.id == request.media_id).first()
            if not media:
                raise HTTPException(status_code=404, detail="Media not found")
        
        # Create workflow request
        workflow_request = WorkflowRequest(
            media_id=request.media_id,
            user_id=request.user_id,
            workflow_type=request.workflow_type,
            selected_agents=request.selected_agents,
            priority=request.priority,
            timeout_minutes=request.timeout_minutes,
            metadata=request.metadata
        )
        
        # Submit workflow for execution
        workflow_id = await workflow_manager.submit_workflow(workflow_request)
        
        # Get estimated completion time
        template = workflow_manager.workflow_templates.get(request.workflow_type)
        estimated_seconds = template["estimated_time"] if template else 120
        estimated_completion = (
            datetime.now().timestamp() + estimated_seconds
        )
        
        logger.info(f"Analysis started: {workflow_id} for media {request.media_id}")
        
        return AnalysisResponse(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING.value,
            message="Analysis started successfully",
            estimated_completion=datetime.fromtimestamp(estimated_completion).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analysis start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow status endpoint
@app.get("/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get status of a workflow execution"""
    try:
        status = workflow_manager.get_execution_status(workflow_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cancel workflow endpoint
@app.post("/workflow/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel an active workflow"""
    try:
        success = await workflow_manager.cancel_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or not active")
        
        return {"message": "Workflow cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# List active workflows endpoint
@app.get("/workflows/active")
async def list_active_workflows():
    """List all active workflow executions"""
    try:
        active_workflows = workflow_manager.list_active_executions()
        return {
            "active_workflows": active_workflows,
            "count": len(active_workflows)
        }
        
    except Exception as e:
        logger.error(f"Failed to list active workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Workflow templates endpoint
@app.get("/workflows/templates")
async def get_workflow_templates():
    """Get available workflow templates"""
    try:
        templates = workflow_manager.get_workflow_templates()
        return {"templates": templates}
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Orchestrator info endpoint
@app.get("/orchestrator/info")
async def get_orchestrator_info():
    """Get information about the orchestrator and agents"""
    try:
        workflow_graph = orchestrator.get_workflow_graph()
        validation = orchestrator.validate_workflow()
        
        return {
            "workflow_graph": workflow_graph,
            "validation": validation,
            "agent_count": len(orchestrator.agent_configs),
            "available_agents": list(orchestrator.agent_configs.keys())
        }
        
    except Exception as e:
        logger.error(f"Failed to get orchestrator info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Execution plan endpoint
@app.post("/orchestrator/plan")
async def get_execution_plan(selected_agents: Optional[List[str]] = None):
    """Get execution plan for selected agents"""
    try:
        plan = orchestrator.get_execution_plan(selected_agents)
        return {"execution_plan": plan}
        
    except Exception as e:
        logger.error(f"Failed to get execution plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    try:
        workflow_metrics = workflow_manager.get_metrics()
        agent_stats = workflow_manager.get_agent_statistics()
        
        return {
            "workflow_metrics": workflow_metrics,
            "agent_statistics": agent_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis history endpoint
@app.get("/analysis/history")
async def get_analysis_history(user_id: Optional[str] = None, limit: int = 10):
    """Get analysis history"""
    try:
        with get_db() as db:
            query = db.query(Analysis)
            
            if user_id:
                query = query.join(Media).filter(Media.user_id == user_id)
            
            analyses = query.order_by(Analysis.created_at.desc()).limit(limit).all()
            
            history = []
            for analysis in analyses:
                history.append({
                    "id": analysis.id,
                    "media_id": analysis.media_id,
                    "workflow_id": getattr(analysis, 'workflow_id', None),
                    "workflow_type": getattr(analysis, 'workflow_type', None),
                    "confidence": analysis.confidence,
                    "processing_time": analysis.processing_time,
                    "created_at": analysis.created_at.isoformat(),
                    "result_summary": {
                        "overall_score": analysis.result_data.get("overall_metrics", {}).get("overall_score", 0),
                        "success_rate": analysis.result_data.get("overall_metrics", {}).get("success_rate", 0),
                        "agent_count": len(analysis.result_data.get("agent_results", {}))
                    }
                })
            
            return {
                "history": history,
                "count": len(history)
            }
            
    except Exception as e:
        logger.error(f"Failed to get analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Quick analysis endpoint (simplified workflow)
@app.post("/analyze/quick")
async def quick_analysis(media_id: str, user_id: Optional[str] = None):
    """Start quick analysis with essential agents only"""
    try:
        request = AnalysisRequest(
            media_id=media_id,
            user_id=user_id,
            workflow_type=WorkflowType.QUICK,
            timeout_minutes=5
        )
        
        return await start_analysis(request)
        
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cleanup endpoint
@app.post("/admin/cleanup")
async def cleanup_old_data(days_old: int = 7):
    """Clean up old execution data (admin endpoint)"""
    try:
        workflow_manager.cleanup_old_executions(days_old)
        
        return {
            "message": f"Cleaned up data older than {days_old} days",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Life Mirror API starting up...")
    
    # Validate orchestrator configuration
    validation = orchestrator.validate_workflow()
    if not validation["valid"]:
        logger.error(f"Orchestrator validation failed: {validation['issues']}")
    else:
        logger.info(f"Orchestrator validated successfully with {validation['total_agents']} agents")
    
    logger.info("Life Mirror API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Life Mirror API shutting down...")
    
    # Cancel any active workflows
    active_workflows = list(workflow_manager.active_executions.keys())
    for workflow_id in active_workflows:
        await workflow_manager.cancel_workflow(workflow_id)
    
    logger.info("Life Mirror API shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
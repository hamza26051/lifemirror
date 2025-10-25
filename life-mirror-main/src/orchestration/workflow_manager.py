import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

from src.orchestration.langgraph_orchestrator import LangGraphOrchestrator, WorkflowContext, ExecutionResult
from src.utils.tracing import log_trace
from src.db.session import get_db
from src.db.models import Media, Analysis

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class WorkflowType(str, Enum):
    """Types of workflows"""
    COMPREHENSIVE = "comprehensive"  # All agents
    QUICK = "quick"                 # Essential agents only
    VISUAL = "visual"               # Visual analysis agents
    TEXT = "text"                   # Text analysis agents
    COMPARISON = "comparison"       # Comparison-focused workflow
    CUSTOM = "custom"               # User-defined agent selection

@dataclass
class WorkflowRequest:
    """Request for workflow execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    media_id: str = ""
    user_id: Optional[str] = None
    workflow_type: WorkflowType = WorkflowType.COMPREHENSIVE
    selected_agents: Optional[List[str]] = None
    priority: int = 5  # 1-10, higher is more priority
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout_minutes: int = 10

@dataclass
class WorkflowExecution:
    """Workflow execution tracking"""
    request: WorkflowRequest
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_stage: Optional[str] = None
    agent_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    execution_log: List[str] = field(default_factory=list)

class WorkflowManager:
    """High-level workflow management and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.orchestrator = LangGraphOrchestrator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Active workflow executions
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # Workflow templates
        self.workflow_templates = {
            WorkflowType.COMPREHENSIVE: {
                "agents": list(self.orchestrator.agent_configs.keys()),
                "description": "Complete analysis with all 18 agents",
                "estimated_time": 120  # seconds
            },
            WorkflowType.QUICK: {
                "agents": ["face", "fashion", "posture", "bio", "aggregator", "formatter"],
                "description": "Quick analysis with essential agents",
                "estimated_time": 30
            },
            WorkflowType.VISUAL: {
                "agents": ["face", "fashion", "posture", "aggregator", "formatter"],
                "description": "Visual analysis focused on appearance",
                "estimated_time": 45
            },
            WorkflowType.TEXT: {
                "agents": ["bio", "social", "vibe_analysis", "embedder", "aggregator", "formatter"],
                "description": "Text analysis focused on personality and social aspects",
                "estimated_time": 40
            },
            WorkflowType.COMPARISON: {
                "agents": ["face", "fashion", "bio", "social", "memory", "compare", "vibe_compare", "aggregator", "formatter"],
                "description": "Analysis with focus on comparisons and historical data",
                "estimated_time": 60
            }
        }
        
        # Performance metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "agent_success_rates": {},
            "workflow_type_stats": {}
        }
    
    @log_trace
    async def submit_workflow(self, request: WorkflowRequest) -> str:
        """Submit a workflow for execution"""
        try:
            # Validate request
            validation_result = self._validate_request(request)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid request: {validation_result['errors']}")
            
            # Determine agents to run
            if request.workflow_type == WorkflowType.CUSTOM:
                agents_to_run = request.selected_agents or []
            else:
                template = self.workflow_templates.get(request.workflow_type)
                agents_to_run = template["agents"] if template else []
            
            request.selected_agents = agents_to_run
            
            # Create execution tracking
            execution = WorkflowExecution(request=request)
            self.active_executions[request.id] = execution
            
            # Start execution asynchronously
            asyncio.create_task(self._execute_workflow(execution))
            
            self.logger.info(f"Workflow {request.id} submitted for execution")
            return request.id
            
        except Exception as e:
            self.logger.error(f"Failed to submit workflow: {e}")
            raise
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        request = execution.request
        
        try:
            # Update status
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            execution.current_stage = "initialization"
            
            self._log_execution(execution, "Workflow execution started")
            
            # Set up progress tracking
            progress_callback = lambda stage, progress: self._update_progress(execution, stage, progress)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_orchestrator(request, progress_callback),
                timeout=request.timeout_minutes * 60
            )
            
            # Update execution with results
            execution.result = result
            execution.status = WorkflowStatus.COMPLETED if result.get("success", False) else WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            execution.progress = 1.0
            
            if not result.get("success", False):
                execution.error = result.get("error", "Unknown error")
            
            self._log_execution(execution, f"Workflow completed with status: {execution.status.value}")
            
            # Store results in database
            await self._store_results(execution)
            
            # Update metrics
            self._update_metrics(execution)
            
            # Move to history
            self._move_to_history(execution)
            
        except asyncio.TimeoutError:
            execution.status = WorkflowStatus.FAILED
            execution.error = f"Workflow timed out after {request.timeout_minutes} minutes"
            execution.completed_at = datetime.now()
            self._log_execution(execution, "Workflow timed out")
            self._move_to_history(execution)
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            self._log_execution(execution, f"Workflow failed: {e}")
            self._move_to_history(execution)
    
    async def _run_orchestrator(self, request: WorkflowRequest, progress_callback: Callable) -> Dict[str, Any]:
        """Run the orchestrator with progress tracking"""
        # Execute workflow
        result = await self.orchestrator.execute_workflow(
            media_id=request.media_id,
            user_id=request.user_id,
            analysis_type=request.workflow_type.value,
            selected_agents=request.selected_agents
        )
        
        return result
    
    def _update_progress(self, execution: WorkflowExecution, stage: str, progress: float):
        """Update execution progress"""
        execution.current_stage = stage
        execution.progress = progress
        self._log_execution(execution, f"Stage: {stage}, Progress: {progress:.1%}")
    
    def _validate_request(self, request: WorkflowRequest) -> Dict[str, Any]:
        """Validate workflow request"""
        errors = []
        
        if not request.media_id:
            errors.append("media_id is required")
        
        if request.workflow_type == WorkflowType.CUSTOM and not request.selected_agents:
            errors.append("selected_agents is required for custom workflow")
        
        if request.selected_agents:
            invalid_agents = [a for a in request.selected_agents if a not in self.orchestrator.agent_configs]
            if invalid_agents:
                errors.append(f"Invalid agents: {invalid_agents}")
        
        if request.timeout_minutes < 1 or request.timeout_minutes > 60:
            errors.append("timeout_minutes must be between 1 and 60")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _store_results(self, execution: WorkflowExecution):
        """Store workflow results in database"""
        try:
            if not execution.result or not execution.result.get("success"):
                return
            
            with get_db() as db:
                # Create analysis record
                analysis = Analysis(
                    media_id=execution.request.media_id,
                    result_data=execution.result,
                    confidence=execution.result.get("overall_metrics", {}).get("confidence", 0.0),
                    processing_time=execution.result.get("execution_time", 0.0),
                    agent_results=execution.result.get("agent_results", {}),
                    workflow_id=execution.request.id,
                    workflow_type=execution.request.workflow_type.value
                )
                
                db.add(analysis)
                db.commit()
                
                self.logger.info(f"Stored analysis results for workflow {execution.request.id}")
                
        except Exception as e:
            self.logger.error(f"Failed to store results: {e}")
    
    def _update_metrics(self, execution: WorkflowExecution):
        """Update performance metrics"""
        self.metrics["total_executions"] += 1
        
        if execution.status == WorkflowStatus.COMPLETED:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        
        # Update average execution time
        if execution.started_at and execution.completed_at:
            execution_time = (execution.completed_at - execution.started_at).total_seconds()
            current_avg = self.metrics["average_execution_time"]
            total_executions = self.metrics["total_executions"]
            
            self.metrics["average_execution_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )
        
        # Update workflow type stats
        workflow_type = execution.request.workflow_type.value
        if workflow_type not in self.metrics["workflow_type_stats"]:
            self.metrics["workflow_type_stats"][workflow_type] = {
                "total": 0,
                "successful": 0,
                "failed": 0
            }
        
        self.metrics["workflow_type_stats"][workflow_type]["total"] += 1
        if execution.status == WorkflowStatus.COMPLETED:
            self.metrics["workflow_type_stats"][workflow_type]["successful"] += 1
        else:
            self.metrics["workflow_type_stats"][workflow_type]["failed"] += 1
    
    def _move_to_history(self, execution: WorkflowExecution):
        """Move execution to history"""
        if execution.request.id in self.active_executions:
            del self.active_executions[execution.request.id]
        
        self.execution_history.append(execution)
        
        # Keep only last 100 executions in memory
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _log_execution(self, execution: WorkflowExecution, message: str):
        """Log execution event"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {message}"
        execution.execution_log.append(log_entry)
        self.logger.info(f"Workflow {execution.request.id}: {message}")
    
    def get_execution_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of workflow execution"""
        # Check active executions
        if workflow_id in self.active_executions:
            execution = self.active_executions[workflow_id]
            return self._serialize_execution(execution)
        
        # Check history
        for execution in self.execution_history:
            if execution.request.id == workflow_id:
                return self._serialize_execution(execution)
        
        return None
    
    def _serialize_execution(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Serialize execution for API response"""
        return {
            "id": execution.request.id,
            "status": execution.status.value,
            "workflow_type": execution.request.workflow_type.value,
            "media_id": execution.request.media_id,
            "user_id": execution.request.user_id,
            "progress": execution.progress,
            "current_stage": execution.current_stage,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error": execution.error,
            "result": execution.result,
            "execution_log": execution.execution_log[-10:],  # Last 10 log entries
            "metadata": execution.request.metadata
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active workflow executions"""
        return [self._serialize_execution(execution) for execution in self.active_executions.values()]
    
    def get_workflow_templates(self) -> Dict[str, Any]:
        """Get available workflow templates"""
        return {
            workflow_type.value: {
                **template,
                "type": workflow_type.value
            }
            for workflow_type, template in self.workflow_templates.items()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "active_executions": len(self.active_executions),
            "success_rate": (
                self.metrics["successful_executions"] / self.metrics["total_executions"]
                if self.metrics["total_executions"] > 0 else 0.0
            )
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        if workflow_id not in self.active_executions:
            return False
        
        execution = self.active_executions[workflow_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        execution.error = "Workflow cancelled by user"
        
        self._log_execution(execution, "Workflow cancelled")
        self._move_to_history(execution)
        
        return True
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics for individual agents"""
        agent_stats = {}
        
        for execution in self.execution_history:
            if execution.result and execution.result.get("agent_results"):
                for agent_name, agent_result in execution.result["agent_results"].items():
                    if agent_name not in agent_stats:
                        agent_stats[agent_name] = {
                            "total_runs": 0,
                            "successful_runs": 0,
                            "failed_runs": 0,
                            "average_processing_time": 0.0,
                            "average_confidence": 0.0
                        }
                    
                    stats = agent_stats[agent_name]
                    stats["total_runs"] += 1
                    
                    # This would need to be adapted based on actual agent result structure
                    if isinstance(agent_result, dict):
                        success = agent_result.get("success", True)
                        if success:
                            stats["successful_runs"] += 1
                        else:
                            stats["failed_runs"] += 1
        
        # Calculate success rates
        for agent_name, stats in agent_stats.items():
            if stats["total_runs"] > 0:
                stats["success_rate"] = stats["successful_runs"] / stats["total_runs"]
            else:
                stats["success_rate"] = 0.0
        
        return agent_stats
    
    def cleanup_old_executions(self, days_old: int = 7):
        """Clean up old execution history"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        self.execution_history = [
            execution for execution in self.execution_history
            if execution.request.created_at > cutoff_date
        ]
        
        self.logger.info(f"Cleaned up executions older than {days_old} days")

# Global workflow manager instance
workflow_manager = WorkflowManager()
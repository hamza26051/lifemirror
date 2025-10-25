import logging
import time
import functools
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import json
import traceback
import uuid
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('life_mirror.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TraceContext:
    """Context for tracing execution"""
    
    def __init__(self, operation: str, metadata: Dict[str, Any] = None):
        self.operation = operation
        self.trace_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.metadata = metadata or {}
        self.events = []
        self.success = True
        self.error = None
    
    def add_event(self, event: str, data: Dict[str, Any] = None):
        """Add an event to the trace"""
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data or {},
            "elapsed_time": time.time() - self.start_time
        })
    
    def set_error(self, error: Exception):
        """Set error for the trace"""
        self.success = False
        self.error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary"""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": time.time() - self.start_time,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
            "event_count": len(self.events),
            "events": self.events
        }

class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.operation_counts = {}
        self.operation_times = {}
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "avg_time": 0.0
            }
        
        metrics = self.metrics[operation]
        metrics["total_calls"] += 1
        
        if success:
            metrics["successful_calls"] += 1
        else:
            metrics["failed_calls"] += 1
        
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["avg_time"] = metrics["total_time"] / metrics["total_calls"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        return {
            "metrics": self.metrics,
            "summary": {
                "total_operations": sum(m["total_calls"] for m in self.metrics.values()),
                "total_successful": sum(m["successful_calls"] for m in self.metrics.values()),
                "total_failed": sum(m["failed_calls"] for m in self.metrics.values()),
                "total_time": sum(m["total_time"] for m in self.metrics.values())
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.operation_counts.clear()
        self.operation_times.clear()

# Global performance tracker
performance_tracker = PerformanceTracker()

def log_trace(operation: str, metadata: Dict[str, Any] = None):
    """Decorator for tracing function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace = TraceContext(operation, metadata)
            
            try:
                trace.add_event("operation_started", {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                
                result = func(*args, **kwargs)
                
                trace.add_event("operation_completed", {
                    "result_type": type(result).__name__ if result is not None else "None"
                })
                
                # Record performance metrics
                duration = time.time() - trace.start_time
                performance_tracker.record_operation(operation, duration, True)
                
                logger.info(f"Operation '{operation}' completed successfully in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                trace.set_error(e)
                
                # Record performance metrics
                duration = time.time() - trace.start_time
                performance_tracker.record_operation(operation, duration, False)
                
                logger.error(f"Operation '{operation}' failed after {duration:.3f}s: {e}")
                logger.debug(f"Trace summary: {json.dumps(trace.get_summary(), indent=2)}")
                
                raise
        
        return wrapper
    return decorator

def log_trace_async(operation: str, metadata: Dict[str, Any] = None):
    """Decorator for tracing async function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            trace = TraceContext(operation, metadata)
            
            try:
                trace.add_event("async_operation_started", {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })
                
                result = await func(*args, **kwargs)
                
                trace.add_event("async_operation_completed", {
                    "result_type": type(result).__name__ if result is not None else "None"
                })
                
                # Record performance metrics
                duration = time.time() - trace.start_time
                performance_tracker.record_operation(operation, duration, True)
                
                logger.info(f"Async operation '{operation}' completed successfully in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                trace.set_error(e)
                
                # Record performance metrics
                duration = time.time() - trace.start_time
                performance_tracker.record_operation(operation, duration, False)
                
                logger.error(f"Async operation '{operation}' failed after {duration:.3f}s: {e}")
                logger.debug(f"Trace summary: {json.dumps(trace.get_summary(), indent=2)}")
                
                raise
        
        return wrapper
    return decorator

@contextmanager
def trace_context(operation: str, metadata: Dict[str, Any] = None):
    """Context manager for tracing code blocks"""
    trace = TraceContext(operation, metadata)
    
    try:
        trace.add_event("context_entered")
        yield trace
        trace.add_event("context_completed")
        
        # Record performance metrics
        duration = time.time() - trace.start_time
        performance_tracker.record_operation(operation, duration, True)
        
        logger.info(f"Context '{operation}' completed successfully in {duration:.3f}s")
        
    except Exception as e:
        trace.set_error(e)
        
        # Record performance metrics
        duration = time.time() - trace.start_time
        performance_tracker.record_operation(operation, duration, False)
        
        logger.error(f"Context '{operation}' failed after {duration:.3f}s: {e}")
        logger.debug(f"Trace summary: {json.dumps(trace.get_summary(), indent=2)}")
        
        raise

class AgentTracker:
    """Specialized tracker for agent execution"""
    
    def __init__(self):
        self.agent_metrics = {}
        self.workflow_metrics = {}
    
    def start_agent_execution(self, agent_name: str, workflow_id: str) -> str:
        """Start tracking agent execution"""
        execution_id = str(uuid.uuid4())
        
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = {
                "executions": [],
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "avg_duration": 0.0
            }
        
        execution = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "start_time": time.time(),
            "status": "running"
        }
        
        self.agent_metrics[agent_name]["executions"].append(execution)
        
        logger.info(f"Agent '{agent_name}' started execution {execution_id} for workflow {workflow_id}")
        
        return execution_id
    
    def complete_agent_execution(self, agent_name: str, execution_id: str, 
                                success: bool = True, result_data: Dict[str, Any] = None):
        """Complete agent execution tracking"""
        if agent_name not in self.agent_metrics:
            return
        
        executions = self.agent_metrics[agent_name]["executions"]
        execution = next((e for e in executions if e["execution_id"] == execution_id), None)
        
        if not execution:
            return
        
        duration = time.time() - execution["start_time"]
        execution["duration"] = duration
        execution["status"] = "completed" if success else "failed"
        execution["result_data"] = result_data or {}
        
        metrics = self.agent_metrics[agent_name]
        metrics["total_runs"] += 1
        
        if success:
            metrics["successful_runs"] += 1
        else:
            metrics["failed_runs"] += 1
        
        # Update average duration
        total_duration = sum(e.get("duration", 0) for e in executions if "duration" in e)
        completed_runs = len([e for e in executions if "duration" in e])
        metrics["avg_duration"] = total_duration / completed_runs if completed_runs > 0 else 0
        
        logger.info(f"Agent '{agent_name}' completed execution {execution_id} in {duration:.3f}s (success: {success})")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        stats = {}
        
        for agent_name, metrics in self.agent_metrics.items():
            success_rate = (
                metrics["successful_runs"] / metrics["total_runs"] 
                if metrics["total_runs"] > 0 else 0
            )
            
            stats[agent_name] = {
                "total_runs": metrics["total_runs"],
                "successful_runs": metrics["successful_runs"],
                "failed_runs": metrics["failed_runs"],
                "success_rate": success_rate,
                "avg_duration": metrics["avg_duration"],
                "recent_executions": metrics["executions"][-10:]  # Last 10 executions
            }
        
        return stats
    
    def cleanup_old_executions(self, max_executions_per_agent: int = 100):
        """Clean up old execution records"""
        for agent_name, metrics in self.agent_metrics.items():
            executions = metrics["executions"]
            if len(executions) > max_executions_per_agent:
                # Keep only the most recent executions
                metrics["executions"] = executions[-max_executions_per_agent:]
                logger.info(f"Cleaned up old executions for agent '{agent_name}'")

# Global agent tracker
agent_tracker = AgentTracker()

def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary"""
    return {
        "general_metrics": performance_tracker.get_metrics(),
        "agent_statistics": agent_tracker.get_agent_statistics(),
        "timestamp": datetime.now().isoformat()
    }

def reset_all_metrics():
    """Reset all performance metrics"""
    performance_tracker.reset_metrics()
    agent_tracker.agent_metrics.clear()
    agent_tracker.workflow_metrics.clear()
    logger.info("All performance metrics have been reset")
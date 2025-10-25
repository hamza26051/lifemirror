import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from langsmith import Client, traceable
import uuid

# Initialize LangSmith client
_langsmith_enabled = bool(os.getenv("LANGSMITH_API_KEY"))
_client = Client(
    api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com"),
    api_key=os.getenv("LANGSMITH_API_KEY")
) if _langsmith_enabled else None

def log_trace(name: str, inputs: dict, outputs: dict, run_id: Optional[str] = None):
    """Enhanced LangSmith tracing with metadata"""
    
    # Always log locally for debugging
    print(f"[TRACE] {name}: {inputs} -> {outputs}")
    
    if not _langsmith_enabled or not _client:
        return
    
    try:
        # Create trace data
        trace_data = {
            "name": name,
            "run_id": run_id or str(uuid.uuid4()),
            "inputs": inputs,
            "outputs": outputs,
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "run_type": "agent",
            "tags": ["lifemirror", "agent", name],
            "metadata": {
                "agent_type": name,
                "version": "1.0",
                "environment": os.getenv("LIFEMIRROR_MODE", "prod")
            }
        }
        
        # Log to LangSmith
        _client.create_run(**trace_data)
        
    except Exception as e:
        print(f"[TRACE ERROR] Failed to log to LangSmith: {e}")

@traceable(name="agent_execution")
def trace_agent_execution(agent_name: str, run_id: str, inputs: Dict[str, Any], 
                         outputs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
    """Decorator-style tracing for agent executions"""
    
    trace_metadata = {
        "agent_name": agent_name,
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "success": outputs.get("success", False),
        "error": outputs.get("error"),
        **(metadata or {})
    }
    
    log_trace(agent_name, inputs, outputs, run_id)
    return trace_metadata

def trace_llm_call(agent_name: str, prompt: str, response: str, model: str, 
                  run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
    """Trace LLM calls with prompt and response details"""
    
    if not _langsmith_enabled or not _client:
        print(f"[LLM TRACE] {agent_name} -> {model}: {len(prompt)} chars prompt, {len(response)} chars response")
        return
    
    try:
        trace_data = {
            "name": f"{agent_name}_llm_call",
            "run_id": run_id or str(uuid.uuid4()),
            "inputs": {
                "prompt": prompt,
                "model": model,
                "agent": agent_name
            },
            "outputs": {
                "response": response,
                "response_length": len(response)
            },
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "run_type": "llm",
            "tags": ["lifemirror", "llm_call", agent_name, model],
            "metadata": {
                "model": model,
                "agent_name": agent_name,
                "prompt_length": len(prompt),
                "response_length": len(response),
                **(metadata or {})
            }
        }
        
        _client.create_run(**trace_data)
        
    except Exception as e:
        print(f"[LLM TRACE ERROR] Failed to log LLM call: {e}")

def trace_workflow_execution(workflow_name: str, media_id: str, user_id: str,
                           agents_used: list, final_result: Dict[str, Any],
                           run_id: Optional[str] = None):
    """Trace complete workflow execution"""
    
    if not _langsmith_enabled or not _client:
        print(f"[WORKFLOW TRACE] {workflow_name}: {len(agents_used)} agents, success: {final_result.get('success')}")
        return
    
    try:
        trace_data = {
            "name": workflow_name,
            "run_id": run_id or str(uuid.uuid4()),
            "inputs": {
                "media_id": media_id,
                "user_id": user_id,
                "agents_requested": agents_used
            },
            "outputs": final_result,
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow(),
            "run_type": "workflow",
            "tags": ["lifemirror", "workflow", workflow_name],
            "metadata": {
                "media_id": media_id,
                "user_id": user_id,
                "agents_count": len(agents_used),
                "agents_used": agents_used,
                "success": final_result.get("success", False),
                "overall_score": final_result.get("data", {}).get("overall_score"),
                "confidence": final_result.get("data", {}).get("confidence")
            }
        }
        
        _client.create_run(**trace_data)
        
    except Exception as e:
        print(f"[WORKFLOW TRACE ERROR] Failed to log workflow: {e}")

# Context manager for tracing agent runs
class AgentTraceContext:
    def __init__(self, agent_name: str, run_id: Optional[str] = None):
        self.agent_name = agent_name
        self.run_id = run_id or str(uuid.uuid4())
        self.start_time = None
        self.inputs = {}
        self.outputs = {}
        self.metadata = {}
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        # Add execution metadata
        self.metadata.update({
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "error_occurred": exc_type is not None,
            "error_type": str(exc_type) if exc_type else None
        })
        
        # Log the trace
        trace_agent_execution(
            self.agent_name, 
            self.run_id, 
            self.inputs, 
            self.outputs, 
            self.metadata
        )
    
    def set_inputs(self, inputs: Dict[str, Any]):
        self.inputs = inputs
    
    def set_outputs(self, outputs: Dict[str, Any]):
        self.outputs = outputs
    
    def add_metadata(self, metadata: Dict[str, Any]):
        self.metadata.update(metadata)

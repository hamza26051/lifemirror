from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

class AgentInput(BaseModel):
    """Standard input format for all agents"""
    media_id: str
    url: str
    context: Dict[str, Any] = {}
    user_id: Optional[str] = None
    analysis_type: Optional[str] = None

class AgentOutput(BaseModel):
    """Standard output format for all agents"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    agent_name: Optional[str] = None

class BaseAgent(ABC):
    """Base class for all Life Mirror agents"""
    
    def __init__(self):
        self.name = self.__class__.__name__.lower().replace('agent', '')
        self.version = "1.0.0"
        self.logger = logging.getLogger(f"agent.{self.name}")
    
    @abstractmethod
    def run(self, input: AgentInput) -> AgentOutput:
        """Main execution method for the agent"""
        pass
    
    def _trace(self, inputs: dict, outputs: dict):
        """Log execution trace for monitoring"""
        self.logger.info(f"Agent {self.name} executed", extra={
            'agent_name': self.name,
            'inputs': inputs,
            'outputs': outputs,
            'timestamp': time.time()
        })
    
    def _create_output(self, success: bool, data: Dict[str, Any], 
                      error: Optional[str] = None, confidence: Optional[float] = None,
                      processing_time: Optional[float] = None) -> AgentOutput:
        """Helper method to create standardized output"""
        return AgentOutput(
            success=success,
            data=data,
            error=error,
            confidence=confidence,
            processing_time=processing_time,
            agent_name=self.name
        )
    
    def _handle_error(self, error: Exception, input_data: AgentInput) -> AgentOutput:
        """Standardized error handling"""
        error_msg = f"Agent {self.name} failed: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        
        result = self._create_output(
            success=False,
            data={},
            error=error_msg,
            confidence=0.0
        )
        
        self._trace(input_data.dict(), result.dict())
        return result
    
    def execute_with_timing(self, input: AgentInput) -> AgentOutput:
        """Execute agent with timing and error handling"""
        start_time = time.time()
        
        try:
            result = self.run(input)
            processing_time = time.time() - start_time
            
            # Add timing info if not already present
            if result.processing_time is None:
                result.processing_time = processing_time
            if result.agent_name is None:
                result.agent_name = self.name
                
            self._trace(input.dict(), result.dict())
            return result
            
        except Exception as e:
            return self._handle_error(e, input)

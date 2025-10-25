from pydantic import BaseModel
from typing import Dict, Any, Optional

class AgentInput(BaseModel):
    media_id: Optional[str] = None
    url: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class AgentOutput(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
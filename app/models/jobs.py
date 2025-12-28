from pydantic import BaseModel
from typing import Optional, Any, Dict

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

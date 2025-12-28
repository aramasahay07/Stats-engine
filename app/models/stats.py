from pydantic import BaseModel
from typing import Any, Dict

class StatsRequest(BaseModel):
    analysis: str
    params: Dict[str, Any] = {}

class StatsResponse(BaseModel):
    test: str
    result: Dict[str, Any]
    cached: bool = False

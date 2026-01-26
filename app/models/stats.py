from pydantic import BaseModel
from typing import Any, Dict, Optional

class StatsRequest(BaseModel):
    analysis: str
    params: Dict[str, Any] = {}

    # Opt-in filtering (SQL without "WHERE")
    where: Optional[str] = None

    # Opt-in: run using a saved pipeline (pipelines.id)
    pipeline_id: Optional[str] = None

    # Future (Step 3): agent-recommended transforms
    auto_transform: bool = False

class StatsResponse(BaseModel):
    test: str
    result: Dict[str, Any]
    cached: bool = False


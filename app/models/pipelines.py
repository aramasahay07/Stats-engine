from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class PipelineStep(BaseModel):
    op: str
    args: Dict[str, Any] = {}

class PipelineCreateRequest(BaseModel):
    name: str
    steps: List[PipelineStep]

class PipelineResponse(BaseModel):
    pipeline_id: str
    dataset_id: str
    name: str
    steps: List[PipelineStep]
    pipeline_hash: str

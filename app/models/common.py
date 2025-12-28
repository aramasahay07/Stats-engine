from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[Any] = None

class OkResponse(BaseModel):
    ok: bool = True

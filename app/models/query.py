from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict, Union

class Measure(BaseModel):
    name: str
    expr: str

class FilterSpec(BaseModel):
    col: str
    op: str
    value: Any

class OrderBy(BaseModel):
    col: str
    dir: str = Field("asc", pattern="^(asc|desc)$")

class QuerySpec(BaseModel):
    select: List[str] = Field(default_factory=list)
    measures: List[Measure] = Field(default_factory=list)
    groupby: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: int = 100000

class QueryResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int


class ExportResponse(BaseModel):
    remote_path: str
    format: str
    row_count: int

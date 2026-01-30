from __future__ import annotations

from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

from app.models.pipelines import PipelineStep


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


class XTransformSpec(BaseModel):
    """
    Chart-focused x-axis transform shortcut.

    Uses transformer library under the hood:
      - date_trunc for time bucketing (hour/day/week/month/quarter/year)
      - date_part (optional future extension) for parts like hour_of_day
    """
    type: str = Field(..., pattern="^(hour|day|week|month|quarter|year)$")
    as_name: Optional[str] = Field(default=None, alias="as")  # allow {"as": "..."} too


class QuerySpec(BaseModel):
    # Pydantic v2: allow aliases like xField/xTransform and ignore unknown keys safely
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    select: List[str] = Field(default_factory=list)
    measures: List[Measure] = Field(default_factory=list)
    groupby: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: int = 100000

    # ✅ NEW: optional transformer pipeline steps (use any of the 99 ops)
    transforms: List[PipelineStep] = Field(default_factory=list)

    # ✅ NEW: chart shortcut (compiled into a transform step automatically)
    x_field: Optional[str] = Field(default=None, alias="xField")
    x_transform: Optional[XTransformSpec] = Field(default=None, alias="xTransform")


class QueryResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int


class ExportResponse(BaseModel):
    remote_path: str
    format: str
    row_count: int


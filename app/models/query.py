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

    Long-term contract:
    - Always produces a stable KEY column for grouping/sorting (timestamp or numeric).
    - Optionally produces one or more LABEL columns for display.

    type:
      - hour/day/week/month/quarter/year: time bucketing
      - weekday: weekday name (Mon/Tue...) with stable numeric key
      - weekday_weekend: Weekend vs Weekday with stable numeric key

    format:
      - None: key only
      - "name": key + name label (e.g., January, Monday, 01 PM)
      - "name_year": key + name_year label (e.g., Jan 2025)
      - "iso": key + iso label (e.g., 2025-01)
      - "all": key + name + name_year + iso
    """
    type: str = Field(
        ...,
        pattern="^(hour|day|week|month|quarter|year|weekday|weekday_weekend)$",
    )

    format: Optional[str] = Field(
        default=None,
        pattern="^(all|name|name_year|iso)?$",
    )

    # allow {"as": "..."} and also pythonic access via .as_name
    as_name: Optional[str] = Field(default=None, alias="as")


class QuerySpec(BaseModel):
    # Pydantic v2: allow aliases like xField/xTransform and ignore unknown keys safely
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    select: List[str] = Field(default_factory=list)
    measures: List[Measure] = Field(default_factory=list)
    groupby: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: int = 100000

    # Optional transformer pipeline steps (use any registered transformer ops)
    transforms: List[PipelineStep] = Field(default_factory=list)

    # Chart shortcut (compiled into transform steps automatically)
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



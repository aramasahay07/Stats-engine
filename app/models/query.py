from __future__ import annotations

from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field, ConfigDict

from app.models.pipelines import PipelineStep


# -------------------------
# Core query primitives
# -------------------------

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


# -------------------------
# Advanced x-axis transforms
# -------------------------

class RollingSpec(BaseModel):
    # Example:
    # { "window": 7, "unit": "day", "agg": "mean", "min_periods": 1, "center": false }
    window: int = Field(..., ge=1)
    unit: str = Field("row", pattern="^(row|minute|hour|day|week|month)$")
    agg: str = Field("mean", pattern="^(mean|sum|count|min|max|median)$")
    min_periods: int = Field(1, ge=1)
    center: bool = False


class FiscalSpec(BaseModel):
    # Example: { "enabled": true, "start_month": 7 }
    enabled: bool = True
    start_month: int = Field(1, ge=1, le=12)


class TopNSpec(BaseModel):
    # Example: { "n": 5, "by": "measure", "other_label": "Other" }
    n: int = Field(..., ge=1)
    by: str = Field("measure", pattern="^(measure|count)$")
    other_label: str = "Other"
    include_other: bool = True


class NumericBinSpec(BaseModel):
    # Either explicit edges OR fixed bin size
    edges: Optional[List[float]] = None
    size: Optional[float] = Field(default=None, gt=0)
    origin: Optional[float] = None

    # Labeling strategy
    label: str = Field("range", pattern="^(range|lower|upper|mid)$")


class XTransformSpec(BaseModel):
    """
    FINAL chart x-axis transform contract.

    Guarantees:
    - Stable KEY column for grouping & sorting
    - Optional LABEL columns for display
    - Backend owns all bucketing & math

    Supported types:
    - Time: minute/hour/day/week/month/quarter/year
    - Calendar: weekday, weekday_weekend
    - Category: category, top_n
    - Numeric: numeric_bin
    """

    type: str = Field(
        ...,
        pattern="^(none|"
                "minute|hour|day|week|month|quarter|year|"
                "weekday|weekday_weekend|"
                "category|top_n|"
                "numeric_bin)$",
    )

    # Label behavior
    # None       -> key only
    # name       -> key + name
    # name_year  -> key + name_year
    # iso        -> key + iso
    # all        -> key + name + name_year + iso
    format: Optional[str] = Field(
        default=None,
        pattern="^(all|name|name_year|iso)?$",
    )

    # Override output base name
    as_name: Optional[str] = Field(default=None, alias="as")

    # Time-related controls
    timezone: Optional[str] = None

    # Custom bin size (minute/hour/day)
    bin: Optional[int] = Field(default=None, ge=1)

    # Fiscal calendar support
    fiscal: Optional[FiscalSpec] = None

    # Ordering strategy
    order: Optional[str] = Field(
        default=None,
        pattern="^(auto|chronological|alphabetical|"
                "frequency_desc|frequency_asc|"
                "measure_desc|measure_asc)$",
    )

    # Null handling
    nulls: Optional[str] = Field(default=None, pattern="^(keep|null_bucket|drop)$")
    null_label: Optional[str] = None

    # Rolling window (post-aggregation)
    rolling: Optional[RollingSpec] = None

    # Category Top-N
    top_n: Optional[TopNSpec] = None

    # Numeric binning
    numeric_bins: Optional[NumericBinSpec] = None

    model_config = ConfigDict(extra="ignore")


# -------------------------
# QuerySpec (planner input)
# -------------------------

class QuerySpec(BaseModel):
    """
    Backend query planner spec.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    select: List[str] = Field(default_factory=list)
    measures: List[Measure] = Field(default_factory=list)
    groupby: List[str] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: int = 100000

    # Transformer pipeline (99+ ops supported)
    transforms: List[PipelineStep] = Field(default_factory=list)

    # Chart shortcut
    x_field: Optional[str] = Field(default=None, alias="xField")
    x_transform: Optional[XTransformSpec] = Field(default=None, alias="xTransform")


# -------------------------
# Responses
# -------------------------

class QueryResponse(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    row_count: int


class ExportResponse(BaseModel):
    remote_path: str
    format: str
    row_count: int




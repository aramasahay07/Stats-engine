from __future__ import annotations

from typing import Any, Literal, Optional, Union, List
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------
# Query Spec (Power BI style)
# ---------------------------
class FilterSpec(BaseModel):
    col: str
    op: Literal["=", "!=", ">", ">=", "<", "<=", "in", "not in", "contains", "startswith", "endswith", "is null", "is not null"]
    value: Optional[Any] = None

class MeasureSpec(BaseModel):
    name: str
    expr: str  # e.g. "avg(amount)", "count(*)", "approx_quantile(x, 0.5)"

class OrderBySpec(BaseModel):
    col: str
    dir: Literal["asc", "desc"] = "asc"

class QuerySpec(BaseModel):
    # IMPORTANT: stop Swagger/raw-SQL payloads from being silently treated as QuerySpec
    model_config = ConfigDict(extra="forbid")

    select: List[str] = Field(default_factory=list)
    measures: List[MeasureSpec] = Field(default_factory=list)
    filters: List[FilterSpec] = Field(default_factory=list)
    groupby: List[str] = Field(default_factory=list)
    order_by: List[OrderBySpec] = Field(default_factory=list)
    limit: int = 500
    offset: int = 0

# ---------------------------
# Stats Spec (Minitab style)
# ---------------------------
class TTestSpec(BaseModel):
    type: Literal["one_sample", "two_sample", "paired"] = "two_sample"
    x: str
    y: Optional[str] = None
    group: Optional[str] = None
    mu: float = 0.0
    equal_var: bool = False

class AnovaSpec(BaseModel):
    y: str
    factor: str

class RegressionSpec(BaseModel):
    y: str
    x: List[str]
    add_intercept: bool = True

StatsType = Literal["descriptives", "ttest", "anova_oneway", "regression_ols"]

class StatsSpec(BaseModel):
    analysis: StatsType
    params: Union[TTestSpec, AnovaSpec, RegressionSpec, dict] = Field(default_factory=dict)

# ---------------------------
# Common responses
# ---------------------------
class TableResult(BaseModel):
    columns: List[str]
    rows: List[List[Any]]

class DatasetProfile(BaseModel):
    dataset_id: str
    n_rows: int
    n_cols: int
    schema: List[dict]
    missing_summary: dict
    sample_rows: List[dict] = Field(default_factory=list)

class DatasetCreateResponse(BaseModel):
    dataset_id: str
    raw_file_ref: str
    parquet_ref: str
    profile: DatasetProfile

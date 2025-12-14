"""
Pydantic models for API schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime


# ---------------------------------------------------
# Data Profiling Models
# ---------------------------------------------------
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: str  # "numeric" | "categorical" | "datetime" | "text"
    missing_pct: float
    unique_count: Optional[int] = None
    sample_values: Optional[List[Any]] = None


class DescriptiveStats(BaseModel):
    column: str
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None


class ProfileResponse(BaseModel):
    session_id: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]  # Alias for frontend compatibility
    descriptives: List[DescriptiveStats]
    sample_rows: Optional[List[Dict[str, Any]]] = None  # For frontend data preview


# ---------------------------------------------------
# Transform Models
# ---------------------------------------------------
class TransformParams(BaseModel):
    """Base parameters for all transforms"""
    pass


class DateTransformParams(TransformParams):
    format: Optional[str] = "name"
    include_year: Optional[bool] = False
    iso: Optional[bool] = False
    fiscal_start_month: Optional[int] = 1


class NumericTransformParams(TransformParams):
    bins: Optional[List[float]] = None
    bin_count: Optional[int] = 5
    quantiles: Optional[List[float]] = None
    precision: Optional[int] = 2
    method: Optional[str] = "minmax"
    base: Optional[float] = 10
    labels: Optional[List[str]] = None


class TextTransformParams(TransformParams):
    pattern: Optional[str] = None
    group: Optional[int] = 0
    find: Optional[str] = None
    replace: Optional[str] = None
    substring: Optional[str] = None


class CategoricalTransformParams(TransformParams):
    mapping: Optional[Dict[str, Any]] = None
    default: Optional[Any] = None
    n: Optional[int] = 10
    other_label: Optional[str] = "Other"
    condition: Optional[str] = None
    true_label: Optional[str] = "Yes"
    false_label: Optional[str] = "No"
    value: Optional[Any] = None
    columns: Optional[List[str]] = None


class SmartTransformParams(TransformParams):
    """Parameters for intelligent transforms"""
    method: Optional[str] = None
    similarity_threshold: Optional[float] = 0.8
    entity_type: Optional[str] = None
    group_by: Optional[str] = None
    features: Optional[List[str]] = None
    window: Optional[int] = 7
    function: Optional[str] = "mean"
    partition_by: Optional[str] = None
    handle_unknown: Optional[str] = "ignore"
    target_type: Optional[str] = None
    errors: Optional[str] = "coerce"
    validation: Optional[Dict[str, Any]] = None
    outlier_threshold: Optional[float] = 1.5


class TransformSpec(BaseModel):
    type: str
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TransformRequest(BaseModel):
    """Single transform or chain of transforms"""
    column: str
    transforms: List[TransformSpec]


class VirtualColumnSpec(BaseModel):
    type: str
    params: Dict[str, Any]
    conditions: Optional[List[Dict[str, Any]]] = None


# ---------------------------------------------------
# Query Models
# ---------------------------------------------------
class SortSpec(BaseModel):
    column: str
    order: Literal["chronological", "alphabetical", "value_desc", "value_asc", "custom"] = "value_asc"
    custom_order: Optional[List[str]] = None


class FilterSpec(BaseModel):
    column: str
    operator: Literal["eq", "ne", "gt", "lt", "gte", "lte", "in", "not_in", "contains", "starts_with"]
    value: Any


class QueryRequest(BaseModel):
    operation: Literal["aggregate", "filter", "describe", "crosstab", "distinct"]
    group_by: Optional[List[str]] = None
    aggregations: Optional[Dict[str, str]] = None  # {"alias": "column:function"}
    filters: Optional[List[FilterSpec]] = None
    transforms: Optional[Dict[str, TransformSpec]] = None
    virtual_columns: Optional[Dict[str, VirtualColumnSpec]] = None
    sort: Optional[SortSpec] = None
    limit: Optional[int] = None


# ---------------------------------------------------
# Analysis Models
# ---------------------------------------------------
class TestResult(BaseModel):
    test_type: str
    target: str
    group_col: str
    p_value: float
    statistic: float
    df: Optional[float] = None
    interpretation: str


class RegressionResult(BaseModel):
    target: str
    predictors: List[str]
    r_squared: float
    adj_r_squared: float
    coefficients: Dict[str, float]


class CorrelationResponse(BaseModel):
    matrix: Dict[str, Dict[str, float]]


class AnalysisResponse(BaseModel):
    session_id: str
    correlation: Optional[CorrelationResponse] = None
    tests: List[TestResult] = []
    regression: Optional[RegressionResult] = None


# ---------------------------------------------------
# Transform Response Models
# ---------------------------------------------------
class TransformMetadata(BaseModel):
    source_column: str
    transform_type: str
    null_count: int
    unique_values: int
    sample_output: Optional[List[Any]] = None


class QueryResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    transforms_applied: Optional[Dict[str, TransformMetadata]] = None
    api_version: str = "2.0"
    transforms_version: str = "1.0"


# ---------------------------------------------------
# Transform Discovery Models
# ---------------------------------------------------
class TransformParamSpec(BaseModel):
    type: str
    values: Optional[List[Any]] = None
    default: Optional[Any] = None
    required: bool = False
    description: Optional[str] = None


class TransformDefinition(BaseModel):
    input_types: List[str]
    output_type: str
    params: Dict[str, TransformParamSpec]
    description: str
    examples: Optional[List[str]] = None


class TransformDiscoveryResponse(BaseModel):
    transforms: Dict[str, TransformDefinition]


class TransformSuggestion(BaseModel):
    transform: str
    usefulness_score: float
    reason: str
    preview: List[Any]
    params: Optional[Dict[str, Any]] = None


class SuggestTransformsResponse(BaseModel):
    column: str
    detected_type: str
    suggested_transforms: List[TransformSuggestion]

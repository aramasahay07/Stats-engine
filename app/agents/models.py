"""
Pydantic models for AI Analyst agents.

These models define the request/response contracts for the analyst endpoint
and internal data structures used by the various agents.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import uuid


# ==============================================================================
# Request Models
# ==============================================================================

class SelectedColumns(BaseModel):
    """Optional column selections for analysis."""
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    time: Optional[str] = None
    measure: Optional[str] = None


class AnalystContext(BaseModel):
    """Context provided with analyst request."""
    selected_columns: Optional[SelectedColumns] = None
    filters: Optional[Dict[str, Any]] = None
    preferred_test: Optional[str] = None
    tone: Literal["executive", "teaching", "technical"] = "executive"
    detail_level: Literal["short", "medium", "deep"] = "medium"
    visuals: bool = True
    allow_transform_plan: bool = True


class AnalystRequest(BaseModel):
    """Request model for the analyst endpoint."""
    question: str = Field(..., description="The analytical question to answer")
    context: AnalystContext = Field(default_factory=AnalystContext)


# ==============================================================================
# Internal Models (used by agents)
# ==============================================================================

class DataIssue(BaseModel):
    """A data quality issue detected by DataPrepAgent."""
    severity: Literal["low", "med", "high"]
    column: Optional[str] = None
    description: str


class TransformStep(BaseModel):
    """A single transformer operation step."""
    op: str
    args: Dict[str, Any] = Field(default_factory=dict)


class Assumption(BaseModel):
    """A statistical assumption and its verification status."""
    name: str
    status: Literal["pass", "fail", "unknown", "not_checked"]
    evidence: str = ""


class AlternativeConsidered(BaseModel):
    """An alternative test that was considered but not chosen."""
    test: str
    why_not: str


class KeyNumbers(BaseModel):
    """Key statistical numbers from the analysis."""
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n: Optional[int] = None
    ci: Optional[List[float]] = None
    statistic: Optional[float] = None
    df: Optional[Union[int, float, List[float]]] = None
    r_squared: Optional[float] = None
    correlation: Optional[float] = None
    mean_diff: Optional[float] = None
    chi_square: Optional[float] = None


class ChartSpec(BaseModel):
    """A Vega-Lite chart specification."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    type: str  # e.g., "boxplot", "scatter", "bar", "histogram", "line"
    spec: Dict[str, Any]  # The actual Vega-Lite spec
    insight: str = ""  # Brief insight this chart reveals


# ==============================================================================
# Response Component Models
# ==============================================================================

class ChosenMethod(BaseModel):
    """Details about the chosen statistical method."""
    test_name: str
    analysis_slug: str
    why_this_test: List[str] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)
    alternatives_considered: List[AlternativeConsidered] = Field(default_factory=list)


class DataPrepResult(BaseModel):
    """Results from data preparation analysis."""
    issues: List[DataIssue] = Field(default_factory=list)
    suggested_fixes: List[TransformStep] = Field(default_factory=list)


class TransformPlan(BaseModel):
    """A planned transformation pipeline."""
    pipeline_steps: List[TransformStep] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    """Interpretation of statistical results."""
    plain_english: str = ""
    statistical: str = ""
    business_meaning: str = ""
    decision_guidance: List[str] = Field(default_factory=list)
    risks_and_caveats: List[str] = Field(default_factory=list)


class AnalysisResults(BaseModel):
    """Results from statistical analysis."""
    cached: bool = False
    raw: Dict[str, Any] = Field(default_factory=dict)
    key_numbers: KeyNumbers = Field(default_factory=KeyNumbers)
    interpretation: Interpretation = Field(default_factory=Interpretation)


class VisualsResult(BaseModel):
    """Collection of visualizations."""
    charts: List[ChartSpec] = Field(default_factory=list)


class MissingInfo(BaseModel):
    """Information needed to complete analysis."""
    field: str
    description: str
    suggestions: List[str] = Field(default_factory=list)


# ==============================================================================
# Main Response Model
# ==============================================================================

class AnalystResponse(BaseModel):
    """Complete response from the analyst endpoint."""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["ok", "needs_info", "error"] = "ok"
    chosen_method: Optional[ChosenMethod] = None
    data_prep: DataPrepResult = Field(default_factory=DataPrepResult)
    transform_plan: TransformPlan = Field(default_factory=TransformPlan)
    results: AnalysisResults = Field(default_factory=AnalysisResults)
    visuals: VisualsResult = Field(default_factory=VisualsResult)
    next_steps: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    missing_info: List[MissingInfo] = Field(default_factory=list)


# ==============================================================================
# Internal Agent Communication Models
# ==============================================================================

class DatasetInfo(BaseModel):
    """Dataset information fetched from registry."""
    dataset_id: str
    user_id: str
    file_name: str
    n_rows: int
    n_cols: int
    schema: List[Dict[str, Any]] = Field(default_factory=list)
    profile: Dict[str, Any] = Field(default_factory=dict)
    parquet_ref: Optional[str] = None
    parquet_sha: Optional[str] = None


class AnalysisSelection(BaseModel):
    """Result of analysis selection process."""
    analysis_slug: str
    test_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reasoning: List[str] = Field(default_factory=list)
    alternatives: List[AlternativeConsidered] = Field(default_factory=list)
    confidence: float = 1.0  # 0-1 confidence in selection


class ColumnInfo(BaseModel):
    """Information about a single column."""
    name: str
    dtype: str
    role: Literal["numeric", "datetime", "categorical", "text", "unknown"] = "unknown"
    missing_pct: float = 0.0
    unique_count: Optional[int] = None
    sample_values: List[Any] = Field(default_factory=list)

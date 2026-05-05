from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DataPrepIssue(BaseModel):
    issue: str
    severity: Severity = Severity.low
    affected_columns: List[str] = Field(default_factory=list)
    suggested_fix: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class TransformStep(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None


class TransformPlan(BaseModel):
    steps: List[TransformStep] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class StatsRequest(BaseModel):
    analysis: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    # Optional natural-language request (used for auto-selecting analysis)
    question: Optional[str] = None
    rationale: Optional[str] = None

    # ✅ Step 1: explicit "opt-in" switches (no behavior change yet)
    # SQL filter without the "WHERE" keyword. Example: "category = 'A' AND age IS NOT NULL"
    where: Optional[str] = None

    # If True, backend will (later, in Step 2/3) apply transformer pipeline before running stats
    auto_transform: bool = False

    # Optional: run using an existing saved pipeline
    pipeline_id: Optional[str] = None



class StatsResult(BaseModel):
    analysis: str
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = False


class ImprovementRequest(BaseModel):
    question: str
    process_name: Optional[str] = None
    business_context: Optional[str] = None

    focus_metric: Optional[str] = None
    target_direction: Optional[str] = Field(default=None, pattern="^(reduce|increase|stabilize)$")
    target_value: Optional[float] = None
    target_improvement_pct: Optional[float] = None
    target_date: Optional[str] = None

    time_column: Optional[str] = None
    subgroup_column: Optional[str] = None
    group_columns: List[str] = Field(default_factory=list)

    # Attribute / defect chart inputs
    defectives_column: Optional[str] = None
    sample_size_column: Optional[str] = None
    sample_size: Optional[int] = Field(default=None, ge=1)
    defects_column: Optional[str] = None
    area_column: Optional[str] = None

    # Advanced chart parameters
    lambda_param: float = Field(default=0.2, gt=0, le=1)
    target: Optional[float] = None
    sigma: Optional[float] = Field(default=None, gt=0)
    k: float = Field(default=0.5, gt=0)
    h: float = Field(default=5.0, gt=0)

    where: Optional[str] = None
    pipeline_id: Optional[str] = None
    limit: int = Field(default=50000, ge=100, le=200000)


class ImprovementPlanResponse(BaseModel):
    dataset_id: str
    question: str
    process_name: Optional[str] = None
    focus_metric: Optional[str] = None
    summary: str
    problem_definition: Dict[str, Any] = Field(default_factory=dict)
    smart_aim: Dict[str, Any] = Field(default_factory=dict)
    baseline: Dict[str, Any] = Field(default_factory=dict)
    selected_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_analyses: List[Dict[str, Any]] = Field(default_factory=list)
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    root_cause_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    experiments: List[Dict[str, Any]] = Field(default_factory=list)
    action_plan: List[Dict[str, Any]] = Field(default_factory=list)
    sustainment_plan: Dict[str, Any] = Field(default_factory=dict)
    workbooks: Dict[str, Any] = Field(default_factory=dict)
    qa: Optional[ValidationResult] = None


class ChartRequest(BaseModel):
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None


class ChartSpec(BaseModel):
    title: Optional[str] = None
    spec: Dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    message: str
    field: Optional[str] = None
    severity: Severity = Severity.medium


class ValidationResult(BaseModel):
    valid: bool
    issues: List[ValidationIssue] = Field(default_factory=list)

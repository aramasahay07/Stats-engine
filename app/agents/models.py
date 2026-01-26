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

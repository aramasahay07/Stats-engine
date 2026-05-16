from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ColumnMapping(BaseModel):
    case_id_column: str
    activity_column: str
    timestamp_column: str
    resource_column: Optional[str] = None
    cost_column: Optional[str] = None
    attribute_columns: List[str] = Field(default_factory=list)


class ProcessDataShape(BaseModel):
    format: Literal["long", "wide"] = "long"
    case_id_column: Optional[str] = None
    pivot_columns: List[str] = Field(default_factory=list)


class ProcessGoals(BaseModel):
    primary_objective: Optional[
        Literal["reduce_cycle_time", "reduce_rework", "increase_throughput", "reduce_cost"]
    ] = None
    target_improvement_pct: Optional[float] = None
    protected_activities: List[str] = Field(default_factory=list)
    sla_hours: Optional[float] = None


class CostInputs(BaseModel):
    cost_per_hour: Optional[float] = None
    cost_per_rework: Optional[float] = None


class AnalyzeProcessRequest(BaseModel):
    mapping: ColumnMapping
    shape: ProcessDataShape = Field(default_factory=ProcessDataShape)
    goals: Optional[ProcessGoals] = None
    cost_inputs: Optional[CostInputs] = None
    expected_path: Optional[List[str]] = None


class ProcessSummary(BaseModel):
    total_cases: int = 0
    total_events: int = 0
    unique_activities: int = 0
    avg_cycle_time_hours: Optional[float] = None
    median_cycle_time_hours: Optional[float] = None
    rework_rate: Optional[float] = None
    variant_count: int = 0
    sla_breach_rate: Optional[float] = None


class ProcessMapNode(BaseModel):
    activity: str
    frequency: int
    case_frequency: int
    avg_duration_in_state_hours: Optional[float] = None


class ProcessMapEdge(BaseModel):
    from_activity: str
    to_activity: str
    frequency: int
    case_frequency: int
    avg_wait_hours: Optional[float] = None
    median_wait_hours: Optional[float] = None
    p90_wait_hours: Optional[float] = None


class ProcessVariant(BaseModel):
    activities: List[str] = Field(default_factory=list)
    case_count: int
    percentage: float
    avg_cycle_time_hours: Optional[float] = None


class Bottleneck(BaseModel):
    from_activity: str
    to_activity: str
    frequency: int
    avg_wait_hours: Optional[float] = None
    median_wait_hours: Optional[float] = None
    p90_wait_hours: Optional[float] = None


class ReworkLoop(BaseModel):
    activity: str
    affected_cases: int
    affected_case_pct: float
    repeat_events: int


class AIInsights(BaseModel):
    executive_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class ProcessMiningResult(BaseModel):
    summary: ProcessSummary = Field(default_factory=ProcessSummary)
    process_map: Dict[str, Any] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    variants: List[ProcessVariant] = Field(default_factory=list)
    bottlenecks: List[Bottleneck] = Field(default_factory=list)
    rework_loops: List[ReworkLoop] = Field(default_factory=list)
    ai_insights: AIInsights = Field(default_factory=AIInsights)
    target_state: Dict[str, Any] = Field(default_factory=dict)
    toc_analysis: Dict[str, Any] = Field(default_factory=dict)
    conformance: Dict[str, Any] = Field(default_factory=dict)
    root_causes: List[Dict[str, Any]] = Field(default_factory=list)
    initiatives: List[Dict[str, Any]] = Field(default_factory=list)
    edge_durations: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)
    expected_path: List[str] = Field(default_factory=list)
    goals: Optional[ProcessGoals] = None
    cost_inputs: Optional[CostInputs] = None

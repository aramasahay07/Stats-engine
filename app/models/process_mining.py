from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ColumnMapping(BaseModel):
    case_id_column: str
    activity_column: str
    timestamp_column: str
    resource_column: Optional[str] = None
    cost_column: Optional[str] = None
    attribute_columns: List[str] = Field(default_factory=list)


class ProcessDataShape(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    format: Literal["long", "wide"] = "long"
    case_id_column: Optional[str] = Field(default=None, alias="caseIdColumn")
    pivot_columns: List[str] = Field(default_factory=list, alias="pivotColumns")


class ProcessGoals(BaseModel):
    primary_objective: Optional[
        Literal["reduce_cycle_time", "reduce_rework", "increase_throughput", "reduce_cost"]
    ] = None
    target_improvement_pct: Optional[float] = None
    protected_activities: List[str] = Field(default_factory=list)
    notes: str = ""
    sla_hours: Optional[float] = None


class CostInputs(BaseModel):
    cost_per_hour: Optional[float] = None
    cost_per_rework: Optional[float] = None


class AnalyzeProcessRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    dataset_id: str
    mapping: ColumnMapping
    shape: ProcessDataShape = Field(default_factory=ProcessDataShape)
    schema_names: List[str] = Field(default_factory=list)
    goals: Optional[ProcessGoals] = None
    cost_inputs: Optional[CostInputs] = None
    expected_path: Optional[List[str]] = None


class ProcessMiningSummary(BaseModel):
    total_cases: int
    total_events: int
    unique_activities: int
    average_cycle_time: float
    median_cycle_time: float
    rework_rate: float
    variant_count: int
    sla_breach_rate: Optional[float] = None


class ProcessNode(BaseModel):
    id: str
    label: str
    frequency: int
    avg_duration: float


class ProcessEdge(BaseModel):
    source: str
    target: str
    frequency: int
    avg_duration: float
    median_duration: float


class ProcessMap(BaseModel):
    nodes: List[ProcessNode]
    edges: List[ProcessEdge]


class ProcessVariant(BaseModel):
    variant_id: int
    path: List[str]
    case_count: int
    percentage: float
    average_cycle_time: float


class ProcessBottleneck(BaseModel):
    from_activity: str
    to_activity: str
    average_wait_time: float
    median_wait_time: float
    p90_wait_time: float
    case_count: int


class ReworkLoop(BaseModel):
    activity: str
    repeat_count: int
    affected_cases: int
    percentage_of_cases: float


class ProcessAIInsights(BaseModel):
    executive_summary: str
    key_findings: List[str]
    recommended_actions: List[str]


class ProcessEdgeDuration(BaseModel):
    avg: float
    median: float


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str
    payer: str = ""
    department: str = ""
    variant_id: int
    path: List[str]
    cycle_time: float
    has_rework: bool
    sla_breached: bool
    denial_reason: Optional[str] = None
    amount: float = 0.0
    start_iso: str


class ConformanceViolation(BaseModel):
    type: Literal["skipped", "extra", "out_of_order"]
    activity: str
    case_count: int
    share: float


class ConformanceResult(BaseModel):
    expected_path: List[str]
    fitness: float
    perfect_share: float
    cases_compliant: int
    cases_total: int
    violations: List[ConformanceViolation]


class RootCauseFinding(BaseModel):
    attribute: str
    category: str
    outcome: Literal["rework", "sla_breach"]
    rate: float
    baseline: float
    lift: float
    case_count: int


class InitiativeEdge(BaseModel):
    source: str
    target: str


class Initiative(BaseModel):
    id: str
    title: str
    edge: InitiativeEdge
    action: Literal["compress", "parallelize", "remove", "standardize"]
    hours_saved_per_case: float
    cases_affected: int
    total_hours_saved: float
    cost_savings: Optional[float] = None
    effort: Literal["Low", "Medium", "High"]
    rationale: str


class ProjectedImpact(BaseModel):
    cycle_time_reduction_pct: float
    rework_reduction_pct: float
    throughput_uplift_pct: float


class TargetState(BaseModel):
    nodes: List[ProcessNode] = Field(default_factory=list)
    edges: List[ProcessEdge] = Field(default_factory=list)
    node_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    edge_annotations: List[Dict[str, Any]] = Field(default_factory=list)
    removed_nodes: List[str] = Field(default_factory=list)
    added_edges: List[Dict[str, Any]] = Field(default_factory=list)
    projected_summary: ProcessMiningSummary
    projected_impact: ProjectedImpact
    assumptions: List[str] = Field(default_factory=list)


class TocStep(BaseModel):
    step: int
    name: Literal["Identify", "Exploit", "Subordinate", "Elevate", "Repeat"]
    description: str


class TocConstraintEdge(BaseModel):
    source: str
    target: str


class TocAnalysis(BaseModel):
    constraint_edge: TocConstraintEdge
    constraint_rationale: str
    steps: List[TocStep]
    projected_throughput_impact_pct: float
    next_constraint: Optional[str] = None


class ProcessMiningResult(BaseModel):
    summary: ProcessMiningSummary
    process_map: ProcessMap
    variants: List[ProcessVariant]
    bottlenecks: List[ProcessBottleneck]
    rework_loops: List[ReworkLoop]
    ai_insights: ProcessAIInsights
    goals: Optional[ProcessGoals] = None
    cost_inputs: Optional[CostInputs] = None
    target_state: Optional[TargetState] = None
    toc_analysis: Optional[TocAnalysis] = None
    cases: Optional[List[CaseRecord]] = None
    conformance: Optional[ConformanceResult] = None
    root_causes: Optional[List[RootCauseFinding]] = None
    initiatives: Optional[List[Initiative]] = None
    edge_durations: Optional[Dict[str, ProcessEdgeDuration]] = None
    expected_path: Optional[List[str]] = None

"""
AI Analyst Agents Package.

This package provides intelligent agents for automated statistical analysis:
- AIAnalystAgent: Main orchestrator that selects tests and generates explanations
- DataPrepAgent: Detects data quality issues and suggests fixes
- TransformAgent: Converts user intent into transformer pipelines
- VizAgent: Generates Vega-Lite chart specifications
- QAAgent: Validates responses for consistency and correctness
"""

from .analyst_agent import AIAnalystAgent
from .dataprep_agent import DataPrepAgent
from .transform_agent import TransformAgent
from .viz_agent import VizAgent
from .qa_agent import QAAgent
from .models import (
    AnalystRequest,
    AnalystResponse,
    AnalystContext,
    ChosenMethod,
    DataPrepResult,
    TransformPlan,
    AnalysisResults,
    Interpretation,
    VisualsResult,
    ChartSpec,
    Assumption,
    DataIssue,
    TransformStep,
    AlternativeConsidered,
    KeyNumbers,
    DatasetInfo,
    MissingInfo,
)

__all__ = [
    # Agents
    "AIAnalystAgent",
    "DataPrepAgent",
    "TransformAgent",
    "VizAgent",
    "QAAgent",
    # Models
    "AnalystRequest",
    "AnalystResponse",
    "AnalystContext",
    "ChosenMethod",
    "DataPrepResult",
    "TransformPlan",
    "AnalysisResults",
    "Interpretation",
    "VisualsResult",
    "ChartSpec",
    "Assumption",
    "DataIssue",
    "TransformStep",
    "AlternativeConsidered",
    "KeyNumbers",
    "DatasetInfo",
    "MissingInfo",
]

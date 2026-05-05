from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agents.ai_analyst_agent import AIAnalystAgent
from app.agents.data_prep_agent import DataPrepAgent
from app.agents.improvement_agent import ImprovementGuidanceAgent
from app.agents.models import (
    ChartRequest,
    ChartSpec,
    DataPrepIssue,
    ImprovementPlanResponse,
    ImprovementRequest,
    StatsRequest,
    StatsResult,
    TransformPlan,
    TransformStep,
    ValidationResult,
)
from app.agents.qa_agent import QAAgent
from app.agents.transform_agent import TransformAgent
from app.agents.viz_agent import VizAgent
from app.db import registry

router = APIRouter()


# -----------------------------
# Helpers
# -----------------------------
def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        return row.get(key, default)  # asyncpg Record sometimes supports .get
    except Exception:
        try:
            return row[key]
        except Exception:
            return default


def _coerce_json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return default
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return default


async def validate_dataset_ready(dataset_id: str, user_id: str) -> Dict[str, Any]:
    """
    Gating check:
      - 404 if dataset doesn't exist
      - 403 if exists but not owned by user
      - 409 if not ready (processing/missing parquet_ref)
      - 422 if failed
    """
    row = await registry.fetchrow(
        """
        SELECT dataset_id, user_id, parquet_ref, state, version, error_message, profile_json
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row_user_id = _row_get(row, "user_id")
    if row_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (_row_get(row, "state") or "ready").lower()
    parquet_ref = _row_get(row, "parquet_ref")

    if state in {"failed", "error"}:
        raise HTTPException(
            status_code=422,
            detail=_row_get(row, "error_message") or "Dataset processing failed",
        )

    if state != "ready" or parquet_ref is None:
        raise HTTPException(
            status_code=409,
            detail="Dataset not ready yet. Please try again later.",
        )

    return {
        "dataset_id": str(_row_get(row, "dataset_id")),
        "user_id": row_user_id,
        "parquet_ref": parquet_ref,
        "state": state,
        "version": int(_row_get(row, "version") or 1),
        "profile": _coerce_json_value(_row_get(row, "profile_json"), default={}),
    }


# -----------------------------
# Response wrappers (explicit shapes for frontend)
# -----------------------------
class PrepIssuesResponse(BaseModel):
    dataset_id: str
    issues: List[DataPrepIssue] = Field(default_factory=list)


class PrepFixesResponse(BaseModel):
    dataset_id: str
    fixes: List[TransformStep] = Field(default_factory=list)


class TransformPlanResponse(BaseModel):
    dataset_id: str
    plan: TransformPlan


# -----------------------------
# Endpoints
# -----------------------------
@router.get("/health")
async def agents_health() -> Dict[str, Any]:
    return {"ok": True, "service": "agents"}


@router.get("/{dataset_id}/prep-issues", response_model=PrepIssuesResponse)
async def get_prep_issues(dataset_id: str, user_id: str = Query(...)) -> PrepIssuesResponse:
    ctx = await validate_dataset_ready(dataset_id, user_id)
    profile = ctx.get("profile") or {}

    agent = DataPrepAgent()
    issues = agent.identify_issues(profile)

    return PrepIssuesResponse(dataset_id=ctx["dataset_id"], issues=issues)


@router.get("/{dataset_id}/prep-fixes", response_model=PrepFixesResponse)
async def get_prep_fixes(dataset_id: str, user_id: str = Query(...)) -> PrepFixesResponse:
    ctx = await validate_dataset_ready(dataset_id, user_id)
    profile = ctx.get("profile") or {}

    dp = DataPrepAgent()
    issues = dp.identify_issues(profile)
    fixes = dp.suggest_fixes(issues)

    return PrepFixesResponse(dataset_id=ctx["dataset_id"], fixes=fixes)


@router.get("/{dataset_id}/transform-plan", response_model=TransformPlanResponse)
async def get_transform_plan(dataset_id: str, user_id: str = Query(...)) -> TransformPlanResponse:
    ctx = await validate_dataset_ready(dataset_id, user_id)
    profile = ctx.get("profile") or {}

    dp = DataPrepAgent()
    issues = dp.identify_issues(profile)

    planner = TransformAgent()
    plan = planner.plan(issues)

    return TransformPlanResponse(dataset_id=ctx["dataset_id"], plan=plan)


@router.post("/{dataset_id}/analyze", response_model=StatsResult)
async def analyze_dataset(
    dataset_id: str,
    request: StatsRequest,
    user_id: str = Query(...),
) -> StatsResult:
    ctx = await validate_dataset_ready(dataset_id, user_id)

    agent = AIAnalystAgent()
    # AIAnalystAgent selects analysis if request.analysis is None
    return await agent.run(user_id=user_id, dataset_id=ctx["dataset_id"], request=request)


@router.post("/{dataset_id}/improvement-plan", response_model=ImprovementPlanResponse)
async def improvement_plan(
    dataset_id: str,
    request: ImprovementRequest,
    user_id: str = Query(...),
) -> ImprovementPlanResponse:
    ctx = await validate_dataset_ready(dataset_id, user_id)

    agent = ImprovementGuidanceAgent()
    return await agent.run(
        user_id=user_id,
        dataset_id=ctx["dataset_id"],
        request=request,
        profile=ctx.get("profile") or {},
    )


@router.post("/chart-spec", response_model=ChartSpec)
async def chart_spec(request: ChartRequest) -> ChartSpec:
    agent = VizAgent()
    return agent.build_spec(request)


@router.post("/validate", response_model=ValidationResult)
async def validate_payload(payload: Dict[str, Any]) -> ValidationResult:
    agent = QAAgent()
    # default: require some common keys if present; keep minimal and generic
    return agent.validate_response(payload, required_fields=None)

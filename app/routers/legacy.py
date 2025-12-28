from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, UploadFile

from app.db import registry
from app.models.datasets import DatasetCreateResponse, DatasetProfile
from app.models.query import QueryResponse, QuerySpec
from app.models.stats import StatsRequest, StatsResponse
from app.services.datasets_service import dataset_service
from app.services import jobs_service
from app.services.query_service import run_query, run_query_operation
from app.services.stats_service import run_stats

router = APIRouter(tags=["legacy"], include_in_schema=True)


@router.post("/upload", response_model=DatasetCreateResponse)
async def legacy_upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    project_id: Optional[str] = Form(None),
):
    """Legacy endpoint alias.

    Production frontend still has a fallback path that calls /upload.
    We treat this as a thin wrapper around POST /datasets.
    """
    dataset_id = await dataset_service.create_dataset_record(user_id, project_id, file.filename)
    try:
        raw_local, raw_ref = await dataset_service.save_raw_to_storage(user_id, dataset_id, file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    await registry.execute(
        "UPDATE datasets SET raw_file_ref=$2, updated_at=NOW() WHERE dataset_id=$1 AND user_id=$3",
        dataset_id, raw_ref, user_id
    )

    job_id = await jobs_service.create_job(user_id, dataset_id, "build_parquet_profile")
    background.add_task(dataset_service.build_parquet_and_profile, user_id, dataset_id, raw_local, raw_ref, job_id)

    profile = DatasetProfile(n_rows=0, n_cols=0, schema=[], sample_rows=[])
    return DatasetCreateResponse(dataset_id=dataset_id, profile=profile, job_id=job_id)


@router.post("/query/{session_id}", response_model=QueryResponse)
async def legacy_query(session_id: str, payload: Dict[str, Any] = Body(...), user_id: str = "anonymous"):
    """Legacy session query endpoint (alias).

    IMPORTANT: session_id is treated as dataset_id (same UUID in production).
    Supports two payload styles:
    1) Backend QuerySpec shape (select/measures/filters/groupby/order_by/limit)
    2) Edge tool-call operation shape (operation + group_by/metrics/filters/limit)
    """
    dataset_id = session_id
    try:
        if "operation" in payload:
            res = await run_query_operation(user_id, dataset_id, payload)
        else:
            spec = QuerySpec(**payload)
            res = await run_query(user_id, dataset_id, spec)
        return QueryResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/schema/{session_id}")
async def legacy_schema(session_id: str, user_id: str):
    dataset_id = session_id
    row = await registry.fetchrow(
        "SELECT schema_json, n_rows, n_cols, profile_json FROM datasets WHERE dataset_id=$1 AND user_id=$2",
        dataset_id, user_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    profile = row.get("profile_json") or {}
    missing_summary = profile.get("missing_summary") if isinstance(profile, dict) else None
    return {
        "schema": row["schema_json"],
        "n_rows": int(row["n_rows"] or 0),
        "n_cols": int(row["n_cols"] or 0),
        "missing_summary": missing_summary
    }


@router.get("/sample/{session_id}")
async def legacy_sample(session_id: str, user_id: str, max_rows: int = 50):
    dataset_id = session_id
    spec = QuerySpec(select=[], measures=[], groupby=[], filters=[], order_by=[], limit=max_rows)
    res = await run_query(user_id, dataset_id, spec)
    return {"sample_rows": res.get("data", [])}


@router.get("/analysis/{session_id}")
async def legacy_analysis(session_id: str, user_id: str):
    dataset_id = session_id
    # Bundle-style analysis for compatibility
    req = StatsRequest(analysis="descriptives", params={"columns": []})
    result, cached = await run_stats(user_id, dataset_id, req.analysis, req.params)
    return {"analysis": req.analysis, "result": result, "cached": cached}


@router.post("/session/rehydrate")
async def legacy_rehydrate_from_url():
    raise HTTPException(status_code=410, detail="Session rehydrate is deprecated. Use POST /datasets.")

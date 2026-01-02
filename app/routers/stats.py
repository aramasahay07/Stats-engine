from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Query
from app.db import registry
from app.models.stats import StatsRequest, StatsResponse
from app.services.stats_service import run_stats
from app.engine.duckdb_engine import DuckDBUnsupportedTypeError

router = APIRouter()
async def validate_dataset_ready(dataset_id: str, user_id: str) -> dict:
    row_any = await registry.fetchrow(
        """
        SELECT dataset_id, user_id, parquet_ref, state, version, error_message
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )
    if not row_any:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if row_any["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = row_any.get("state") or "ready"
    if state == "reprocessing":
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_REPROCESSING", "message": "Dataset is being reprocessed. Please wait."},
        )
    if state == "processing":
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset is still processing."},
        )
    if state == "failed":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DATASET_FAILED",
                "message": row_any.get("error_message") or "Dataset processing failed",
            },
        )

    if not row_any.get("parquet_ref"):
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset is still processing."},
        )

    return dict(row_any)


@router.post("/{dataset_id}/stats", response_model=StatsResponse)
async def stats_dataset(
    dataset_id: str,
    req: StatsRequest,
    user_id: str = Query(...)
):
    try:
        await validate_dataset_ready(dataset_id, user_id)
        # ---------------------------------------------------------
        # DEFENSIVE FIX:
        # params may arrive as JSON string instead of dict
        # ---------------------------------------------------------
        params = req.params

        if isinstance(params, str):
            try:
                params = json.loads(params)
            except Exception:
                params = {}

        if not isinstance(params, dict):
            params = {}

        result, cached = await run_stats(
            user_id=user_id,
            dataset_id=dataset_id,
            analysis=req.analysis,
            params=params,
        )

        return StatsResponse(
            test=req.analysis,
            result=result,
            cached=cached,
        )
        except DuckDBUnsupportedTypeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "UNSUPPORTED_TYPE", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


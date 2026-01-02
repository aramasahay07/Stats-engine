from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query

from app.db import registry
from app.engine.duckdb_engine import DuckDBUnsupportedTypeError
from app.models.stats import StatsRequest, StatsResponse
from app.services.stats_service import run_stats

router = APIRouter()


async def validate_dataset_ready(dataset_id: str, user_id: str) -> dict:
    """Gating check for stats endpoint.

    - 404 if dataset doesn't exist
    - 403 if exists but not owned by user
    - 409 if still processing / missing parquet_ref
    - 422 if failed
    """
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

    row_user_id = row_any.get("user_id") if hasattr(row_any, "get") else row_any["user_id"]
    if row_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (row_any.get("state") if hasattr(row_any, "get") else row_any["state"]) or "ready"

    if state == "reprocessing":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DATASET_REPROCESSING",
                "message": "Dataset is being reprocessed. Please wait.",
            },
        )

    if state == "processing":
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset is still processing."},
        )

    if state == "failed":
        err = row_any.get("error_message") if hasattr(row_any, "get") else row_any["error_message"]
        raise HTTPException(
            status_code=422,
            detail={"code": "DATASET_FAILED", "message": err or "Dataset processing failed"},
        )

    parquet_ref = row_any.get("parquet_ref") if hasattr(row_any, "get") else row_any["parquet_ref"]
    if not parquet_ref:
        raise HTTPException(
            status_code=409,
            detail={"code": "DATASET_PROCESSING", "message": "Dataset parquet is not ready yet."},
        )

    return dict(row_any)


@router.post("/{dataset_id}/stats", response_model=StatsResponse)
async def stats_dataset(dataset_id: str, req: StatsRequest, user_id: str = Query(...)):
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

    except HTTPException:
        raise

    except DuckDBUnsupportedTypeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "UNSUPPORTED_TYPE", "message": str(e)},
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

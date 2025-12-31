from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.db import registry
from app.models.query import ExportResponse, QueryResponse, QuerySpec
from app.services.query_service import export_query, run_query

router = APIRouter()


async def validate_dataset_ready(dataset_id: str, user_id: str) -> dict:
    """Gating check for query/export endpoints.

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

    if row_any["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (row_any.get("state") if hasattr(row_any, "get") else row_any["state"]) or "ready"

    if state in ("processing", "reprocessing"):
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


@router.post("/{dataset_id}/query", response_model=QueryResponse)
async def query_dataset(dataset_id: str, spec: QuerySpec, user_id: str = Query(...)):
    try:
        await validate_dataset_ready(dataset_id, user_id)
        res = await run_query(user_id, dataset_id, spec)
        return QueryResponse(**res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{dataset_id}/query/export", response_model=ExportResponse)
async def export_dataset_query(dataset_id: str, spec: QuerySpec, user_id: str = Query(...), fmt: str = "parquet"):
    """Export query results to Supabase Storage.

    Use for large result sets (e.g., up to 1M rows) instead of returning huge JSON.
    """
    try:
        await validate_dataset_ready(dataset_id, user_id)
        res = await export_query(user_id, dataset_id, spec, fmt=fmt)
        return ExportResponse(**res)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

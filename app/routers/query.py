from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models.query import QuerySpec, QueryResponse, ExportResponse
from app.services.query_service import run_query, export_query

router = APIRouter()

@router.post("/{dataset_id}/query", response_model=QueryResponse)
async def query_dataset(dataset_id: str, spec: QuerySpec, user_id: str):
    try:
        res = await run_query(user_id, dataset_id, spec)
        return QueryResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/{dataset_id}/query/export', response_model=ExportResponse)
async def export_dataset_query(dataset_id: str, spec: QuerySpec, user_id: str, fmt: str = 'parquet'):
    """Export query results to Supabase Storage.

    Use for large result sets (e.g., up to 1M rows) instead of returning huge JSON.
    """
    
    try:
        res = await export_query(user_id, dataset_id, spec, fmt=fmt)
        return ExportResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


from __future__ import annotations
from fastapi import APIRouter, HTTPException
from app.models.stats import StatsRequest, StatsResponse
from app.services.stats_service import run_stats

router = APIRouter()

@router.post("/{dataset_id}/stats", response_model=StatsResponse)
async def stats_dataset(dataset_id: str, req: StatsRequest, user_id: str):
    try:
        # `analysis` is a concept slug (preferred) or a legacy analysis enum.
        result, cached = await run_stats(user_id, dataset_id, req.analysis, req.params)
        return StatsResponse(test=req.analysis, result=result, cached=cached)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from __future__ import annotations

import json
from fastapi import APIRouter, HTTPException, Query

from app.models.stats import StatsRequest, StatsResponse
from app.services.stats_service import run_stats

router = APIRouter()


@router.post("/{dataset_id}/stats", response_model=StatsResponse)
async def stats_dataset(
    dataset_id: str,
    req: StatsRequest,
    user_id: str = Query(...)
):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

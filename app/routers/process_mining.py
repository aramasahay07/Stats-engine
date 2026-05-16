from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException

from app.auth.supabase_jwt import get_current_user
from app.models.process_mining import AnalyzeProcessRequest, ProcessMiningResult
from app.services.process_mining import analyze_process_mining

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{dataset_id}/process-mining/analyze", response_model=ProcessMiningResult)
async def analyze_process_dataset(
    dataset_id: str,
    request: AnalyzeProcessRequest,
    user=Depends(get_current_user),
) -> ProcessMiningResult:
    user_id = user["user_id"]
    try:
        return await analyze_process_mining(user_id=user_id, dataset_id=dataset_id, request=request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "process mining analysis failed dataset_id=%s user_id=%s\n%s",
            dataset_id,
            user_id,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="Process mining analysis failed due to an internal error.") from exc

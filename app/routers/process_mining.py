from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Depends, Header, HTTPException

from app.auth.supabase_jwt import get_current_user
from app.models.process_mining import AnalyzeProcessRequest, ProcessMiningResult
from app.services.process_mining import analyze_process_mining

router = APIRouter()
logger = logging.getLogger(__name__)


def _resolve_user_id(user: dict, x_user_id: str | None) -> str:
    jwt_sub = str(user.get("claims", {}).get("sub") or user.get("user_id") or "")
    if not jwt_sub:
        raise HTTPException(status_code=401, detail="JWT missing user id")
    if x_user_id is not None and x_user_id != jwt_sub:
        raise HTTPException(status_code=401, detail="x-user-id does not match the authenticated user")
    return jwt_sub


@router.post("/process-mining/analyze", response_model=ProcessMiningResult)
async def analyze_process(
    request: AnalyzeProcessRequest,
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    user=Depends(get_current_user),
) -> ProcessMiningResult:
    user_id = _resolve_user_id(user, x_user_id)
    try:
        return await analyze_process_mining(user_id=user_id, request=request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "process mining analysis failed dataset_id=%s user_id=%s\n%s",
            request.dataset_id,
            user_id,
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail="Process mining analysis failed due to an internal error.") from exc


@router.post("/datasets/{dataset_id}/process-mining/analyze", response_model=ProcessMiningResult)
async def analyze_process_dataset_legacy(
    dataset_id: str,
    request: AnalyzeProcessRequest,
    x_user_id: str | None = Header(default=None, alias="x-user-id"),
    user=Depends(get_current_user),
) -> ProcessMiningResult:
    if request.dataset_id != dataset_id:
        raise HTTPException(status_code=422, detail="Body dataset_id must match the dataset id in the path.")
    return await analyze_process(request=request, x_user_id=x_user_id, user=user)

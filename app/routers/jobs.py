from __future__ import annotations

import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth.supabase_jwt import get_current_user
from app.models.jobs import JobStatusResponse
from app.services.jobs_service import get_job

router = APIRouter()


@router.get("/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str, user=Depends(get_current_user)):
    user_id = user["user_id"]
    row = await get_job(job_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=str(row["job_id"]),
        status=row["status"],
        progress=int(row["progress"] or 0),
        message=row["message"],
        result=row.get("result_json"),
    )


@router.get("/{job_id}/events")
async def job_events(job_id: str, user=Depends(get_current_user)):
    user_id = user["user_id"]

    async def event_stream():
        last = None
        while True:
            row = await get_job(job_id, user_id)
            if not row:
                yield "event: error\ndata: {\"error\":\"not_found\"}\n\n"
                return

            payload = {
                "job_id": str(row["job_id"]),
                "status": row["status"],
                "progress": int(row["progress"] or 0),
                "message": row["message"],
            }

            cur = json.dumps(payload)
            if cur != last:
                yield f"data: {cur}\n\n"
                last = cur

            if row["status"] in ("done", "failed"):
                return

            await asyncio.sleep(0.8)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


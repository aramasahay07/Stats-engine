from __future__ import annotations
import uuid
from typing import Optional, Dict, Any

from app.db import registry

async def create_job(user_id: str, dataset_id: str, job_type: str, pipeline_hash: Optional[str] = None) -> str:
    job_id = str(uuid.uuid4())
    await registry.execute(
        """INSERT INTO jobs (id, user_id, dataset_id, type, status, progress, message)
           VALUES ($1,$2,$3,$4,'queued',0,'queued')""",
        job_id, user_id, dataset_id, job_type
    )
    return job_id

async def update_job(job_id: str, status: str, progress: int, message: str | None = None, result: Dict[str, Any] | None = None):
    await registry.execute(
        """UPDATE jobs SET status=$2, progress=$3, message=$4, result_json=$5, updated_at=NOW() WHERE id=$1""",
        job_id, status, progress, message, result
    )

async def get_job(job_id: str, user_id: str):
    row = await registry.fetchrow("SELECT * FROM jobs WHERE id=$1 AND user_id=$2", job_id, user_id)
    return row

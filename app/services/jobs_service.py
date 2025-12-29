from __future__ import annotations

from typing import Any, Optional, Dict

from app.db import registry


# ------------------------------------------------------------
# Module-level functions (so routers can do:
#   from app.services.jobs_service import get_job
# ------------------------------------------------------------

async def create_job(user_id: str, dataset_id: str, job_type: str) -> str:
    """
    Insert a job row and return the generated job_id.
    Assumes jobs table has primary key column: job_id uuid default gen_random_uuid()
    """
    job_id = await registry.fetchval(
        """
        INSERT INTO jobs (user_id, dataset_id, job_type, status, progress, message)
        VALUES ($1, $2, $3, 'queued', 0, NULL)
        RETURNING job_id
        """,
        user_id,
        dataset_id,
        job_type,
    )
    return str(job_id)


async def update_job(
    job_id: str,
    status: str,
    progress: int,
    message: str,
    result_json: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Update an existing job by job_id.
    """
    await registry.execute(
        """
        UPDATE jobs
        SET status=$2,
            progress=$3,
            message=$4,
            result_json=$5
        WHERE job_id=$1
        """,
        job_id,
        status,
        progress,
        message,
        result_json,
    )


async def get_job(job_id: str) -> Optional[dict]:
    """
    Fetch a job row by job_id.
    """
    row = await registry.fetchrow(
        "SELECT * FROM jobs WHERE job_id=$1",
        job_id,
    )
    return dict(row) if row else None


# ------------------------------------------------------------
# Service object (so other code can do:
#   from app.services import jobs_service
#   await jobs_service.create_job(...)
# ------------------------------------------------------------

class JobsService:
    async def create_job(self, user_id: str, dataset_id: str, job_type: str) -> str:
        return await create_job(user_id, dataset_id, job_type)

    async def update_job(
        self,
        job_id: str,
        status: str,
        progress: int,
        message: str,
        result_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        await update_job(job_id, status, progress, message, result_json)

    async def get_job(self, job_id: str) -> Optional[dict]:
        return await get_job(job_id)


jobs_service = JobsService()

from __future__ import annotations

from typing import Any, Optional, Dict

from app.db import registry


async def create_job(user_id: str, dataset_id: str, job_type: str) -> str:
    """
    Creates a job record and returns job_id (uuid as string).

    Expected schema:
      jobs(
        job_id uuid primary key default gen_random_uuid(),
        user_id text not null,
        dataset_id uuid not null,
        job_type text not null,
        status text not null default 'queued',
        progress int not null default 0,
        message text,
        result_json jsonb,
        created_at timestamptz default now(),
        updated_at timestamptz default now()
      )
    """
    job_id = await registry.fetchval(
        """
        INSERT INTO jobs (user_id, dataset_id, job_type, status, progress, message)
        VALUES ($1, $2, $3, 'queued', 0, NULL)
        RETURNING job_id
        """,
        user_id,        # $1 -> text
        dataset_id,     # $2 -> uuid
        job_type,       # $3 -> text
    )
    return str(job_id)


async def update_job(
    job_id: str,
    status: str,
    progress: int,
    message: str,
    result_json: Optional[Dict[str, Any]] = None,
) -> None:
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
    row = await registry.fetchrow(
        "SELECT * FROM jobs WHERE job_id=$1",
        job_id,
    )
    return dict(row) if row else None


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

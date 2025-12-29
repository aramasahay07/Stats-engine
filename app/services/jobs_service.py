from __future__ import annotations

"""app/services/jobs_service.py

This project has *two* historical "jobs" schemas in the wild.

To stop POST /datasets from failing in Railway/Supabase, we standardize
the code to the schema shipped in migrations/001_v2.sql:

  jobs(
    id uuid primary key,
    user_id text not null,
    dataset_id uuid not null,
    type text not null,
    status text not null,
    progress int not null default 0,
    message text,
    result_json jsonb,
    created_at timestamptz,
    updated_at timestamptz
  )

If your Supabase table currently uses job_id/job_type columns,
either run the provided SQL in README or rename columns to match.
"""

from typing import Any, Optional, Dict

from app.db import registry


async def create_job(user_id: str, dataset_id: str, job_type: str) -> str:
    """Create a job row and return its id as a string."""

    # NOTE: dataset_id is UUID in DB; asyncpg will validate.
    job_id = await registry.fetchval(
        """
        INSERT INTO jobs (id, user_id, dataset_id, type, status, progress, message)
        VALUES (gen_random_uuid(), $1, $2, $3, 'queued', 0, NULL)
        RETURNING id
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
    await registry.execute(
        """
        UPDATE jobs
        SET status=$2,
            progress=$3,
            message=$4,
            result_json=$5,
            updated_at=NOW()
        WHERE id=$1
        """,
        job_id,
        status,
        progress,
        message,
        result_json,
    )


async def get_job(job_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    """Fetch a job by id.

    If user_id is provided, enforces per-user access.
    """
    if user_id is None:
        row = await registry.fetchrow("SELECT * FROM jobs WHERE id=$1", job_id)
    else:
        row = await registry.fetchrow(
            "SELECT * FROM jobs WHERE id=$1 AND user_id=$2",
            job_id,
            user_id,
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

    async def get_job(self, job_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        return await get_job(job_id, user_id)


jobs_service = JobsService()

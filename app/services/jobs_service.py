from __future__ import annotations

from typing import Any, Optional, Dict
from uuid import UUID

from app.db import registry


def _require_uuid(value: str, field_name: str) -> str:
    """Validate UUID strings early so you get clean 400s instead of 500s."""
    try:
        UUID(str(value))
    except Exception as e:
        raise ValueError(f"{field_name} must be a valid UUID (got '{value}')") from e
    return str(value)


async def create_job(user_id: str, dataset_id: str, job_type: str) -> str:
    """
    Matches your Supabase schema:
      jobs(job_id uuid PK, user_id uuid, dataset_id uuid, job_type text, ...)
    """
    _require_uuid(user_id, "user_id")
    _require_uuid(dataset_id, "dataset_id")

    job_id = await registry.fetchval(
        """
        INSERT INTO jobs (user_id, dataset_id, job_type, status, progress, message)
        VALUES ($1::uuid, $2::uuid, $3, 'queued', 0, NULL)
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
    _require_uuid(job_id, "job_id")

    await registry.execute(
        """
        UPDATE jobs
        SET status=$2,
            progress=$3,
            message=$4,
            result_json=$5,
            updated_at=NOW()
        WHERE job_id=$1::uuid
        """,
        job_id,
        status,
        progress,
        message,
        result_json,
    )


async def get_job(job_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    _require_uuid(job_id, "job_id")

    if user_id is None:
        row = await registry.fetchrow(
            "SELECT * FROM jobs WHERE job_id=$1::uuid",
            job_id,
        )
    else:
        _require_uuid(user_id, "user_id")
        row = await registry.fetchrow(
            "SELECT * FROM jobs WHERE job_id=$1::uuid AND user_id=$2::uuid",
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

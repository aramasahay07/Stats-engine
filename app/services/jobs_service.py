from __future__ import annotations

from uuid import uuid4
from typing import Any, Optional, Dict

from app.db import registry


class JobsService:
    """
    DB-backed job tracking.

    Expects table: public.jobs
    Primary key column: job_id (uuid)
    """

    async def create_job(self, user_id: str, dataset_id: str, job_type: str) -> str:
        job_id = str(uuid4())

        # IMPORTANT: column name is job_id (NOT id)
        await registry.execute(
            """
            INSERT INTO jobs (job_id, user_id, dataset_id, job_type, status, progress, message)
            VALUES ($1, $2, $3, $4, 'queued', 0, NULL)
            """,
            job_id, user_id, dataset_id, job_type
        )
        return job_id

    async def update_job(
        self,
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
            job_id, status, progress, message, result_json
        )

    async def get_job(self, job_id: str) -> Optional[dict]:
        row = await registry.fetchrow(
            "SELECT * FROM jobs WHERE job_id=$1",
            job_id
        )
        return dict(row) if row else None


jobs_service = JobsService()

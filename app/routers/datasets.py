from __future__ import annotations

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException

from app.services.datasets_service import dataset_service
from app.services import jobs_service
from app.models.datasets import (
    DatasetCreateResponse,
    DatasetMetadataResponse,
    DatasetProfile,
)
from app.db import registry

router = APIRouter()


def _normalize_optional_uuid(value: str | None) -> Optional[UUID]:
    """
    Convert project_id form value into UUID or None.
    Rejects invalid UUIDs explicitly.
    """
    if value is None:
        return None

    v = value.strip()
    if v in ("", "string", "null", "None"):
        return None

    try:
        return UUID(v)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="project_id must be a valid UUID or omitted",
        )


@router.post("", response_model=DatasetCreateResponse)
async def create_dataset(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    project_id: str | None = Form(None),
):
    # Normalize project_id
    project_uuid = _normalize_optional_uuid(project_id)

    # Create dataset record (dataset_id MUST be UUID string)
    dataset_id = await dataset_service.create_dataset_record(
        user_id,
        project_uuid,
        file.filename,
    )

    # Save raw file
    try:
        raw_local, raw_ref = await dataset_service.save_raw_to_storage(
            user_id,
            dataset_id,
            file,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update raw reference
    await registry.execute(
        """
        UPDATE datasets
        SET raw_file_ref=$2, updated_at=NOW()
        WHERE dataset_id=$1 AND user_id=$3
        """,
        dataset_id,
        raw_ref,
        user_id,
    )

    # Debug logging (safe to keep during stabilization)
    print("DEBUG user_id:", user_id)
    print("DEBUG dataset_id:", dataset_id, "len=", len(str(dataset_id)))
    print("DEBUG project_id:", project_uuid)

    # Create background job
    try:
        job_id = await jobs_service.create_job(
            user_id,
            dataset_id,
            "build_parquet_profile",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"create_job failed: {type(e).__name__}: {e}",
        )

    # Launch background processing
    background.add_task(
        dataset_service.build_parquet_and_profile,
        user_id,
        dataset_id,
        raw_local,
        raw_ref,
        job_id,
    )

    # Initial empty profile (real one will be populated asynchronously)
    profile = DatasetProfile(
        n_rows=0,
        n_cols=0,
        schema=[],
        sample_rows=[],
    )

    return DatasetCreateResponse(
        dataset_id=dataset_id,
        profile=profile,
        job_id=job_id,
    )


@router.get("/{dataset_id}", response_model=DatasetMetadataResponse)
async def get_dataset(dataset_id: str, user_id: str):
row = await registry.fetchrow(
    """
    SELECT *
    FROM datasets
    WHERE dataset_id=$1::uuid
      AND user_id::text=$2
    """,
    dataset_id,
    user_id,
)

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetMetadataResponse(
        dataset_id=row["dataset_id"],
        file_name=row["file_name"],
        n_rows=int(row["n_rows"] or 0),
        n_cols=int(row["n_cols"] or 0),
        schema_json=row["schema_json"],
        profile_json=row["profile_json"],
        raw_file_ref=row["raw_file_ref"],
        parquet_ref=row["parquet_ref"],
    )

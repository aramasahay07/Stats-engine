from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from app.db import registry
from app.models.datasets import DatasetCreateResponse, DatasetMetadataResponse, DatasetProfile
from app.services import jobs_service
from app.services.datasets_service import dataset_service

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


def _coerce_json_value(value, default):
    """
    Accepts JSONB (dict/list), TEXT JSON (str), or NULL.
    Returns a Python dict/list, falling back to default on parse errors.
    """
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            return default
    return default


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user_id: str = Query(...)):
    # Ownership enforced in SQL
    row = await registry.fetchrow(
        """
        SELECT *
        FROM datasets
        WHERE dataset_id = $1::uuid
          AND user_id::text = $2
        """,
        dataset_id,
        user_id,
    )

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    schema_obj = _coerce_json_value(row.get("schema_json"), default=[])
    profile_obj = _coerce_json_value(row.get("profile_json"), default={})

    state = row.get("state") or "ready"
    parquet_ref = row.get("parquet_ref")
    version = int(row.get("version") or 1)
    ready = (state == "ready") and (parquet_ref is not None)

    # Return shape required by frontend/edge:
    # - schema/profile keys (not schema_json/profile_json)
    # - ready convenience boolean
    # Keep backward-compatible keys too (schema_json/profile_json)
    return {
        "dataset_id": str(row["dataset_id"]),
        "file_name": row.get("file_name"),
        "n_rows": int(row.get("n_rows") or 0),
        "n_cols": int(row.get("n_cols") or 0),
        "parquet_ref": parquet_ref,
        "schema": schema_obj,
        "profile": profile_obj,
        "schema_hash": row.get("schema_hash"),
        "state": state,
        "version": version,
        "ready": ready,
        # Backward compatibility for existing clients/models
        "user_id": row.get("user_id"),
        "project_id": str(row["project_id"]) if row.get("project_id") else None,
        "raw_file_ref": row.get("raw_file_ref"),
        "schema_json": schema_obj,
        "profile_json": profile_obj,
    }

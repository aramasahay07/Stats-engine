from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.cache_paths import CachePaths
from app.services.storage_client import SupabaseStorageClient
from app.services.parquet_builder import build_parquet
from app.services.duckdb_manager import DuckDBManager
from app.services.dataset_registry import DatasetRegistry
from app.models.specs import DatasetCreateResponse, DatasetProfile

router = APIRouter(prefix="/datasets", tags=["datasets"])

def _content_type_for_filename(name: str) -> str:
    n = name.lower()
    if n.endswith(".csv"):
        return "text/csv"
    if n.endswith(".parquet"):
        return "application/octet-stream"
    if n.endswith(".xlsx") or n.endswith(".xls"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"

@router.post("", response_model=DatasetCreateResponse)
async def create_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    project_id: Optional[str] = Form(None),
):
    """Create durable dataset: raw upload + parquet + profile + registry row.

    This replaces in-memory session_store. The returned dataset_id is permanent.
    """
    dataset_id = str(uuid.uuid4())

    cache = CachePaths(base_dir=Path("./cache"))
    raw_path = cache.raw_path(user_id, dataset_id, file.filename)
    parquet_path = cache.parquet_path(user_id, dataset_id)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw file locally
    raw_bytes = await file.read()
    raw_path.write_bytes(raw_bytes)

    storage = SupabaseStorageClient()
    registry = DatasetRegistry()
    duck = DuckDBManager()

    # Upload raw to storage
    raw_ref = f"{user_id}/datasets/{dataset_id}/raw/{file.filename}"
    try:
        storage.upload_file(raw_path, raw_ref, content_type=_content_type_for_filename(file.filename))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload raw file: {e}")

    # Build parquet and upload
    try:
        build_parquet(raw_path, parquet_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse/convert file: {e}")

    parquet_ref = f"{user_id}/datasets/{dataset_id}/parquet/data.parquet"
    try:
        storage.upload_file(parquet_path, parquet_ref, content_type=_content_type_for_filename("data.parquet"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload parquet: {e}")

    # Profile via DuckDB
    prof = duck.profile_parquet(parquet_path, sample_n=100)
    schema = prof["schema"]
    missing_summary = prof["missing_summary"]
    sample_rows = prof["sample_rows"]

    n_rows = prof["n_rows"]
    n_cols = len(schema)

    # Write registry row
    row = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "project_id": project_id,
        "file_name": file.filename,
        "raw_file_ref": raw_ref,
        "parquet_ref": parquet_ref,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "schema_json": schema,
        "profile_json": {
            "missing_summary": missing_summary,
        },
    }
    try:
        registry.create(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset row: {e}")

    profile = DatasetProfile(
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_cols=n_cols,
        schema=schema,
        missing_summary=missing_summary,
        sample_rows=sample_rows,
    )

    return DatasetCreateResponse(
        dataset_id=dataset_id,
        raw_file_ref=raw_ref,
        parquet_ref=parquet_ref,
        profile=profile,
    )

@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, user_id: str):
    """Rehydrate UI anytime: returns dataset metadata from registry."""
    registry = DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

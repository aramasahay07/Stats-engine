from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile

from app.models.specs import DatasetCreateResponse, DatasetProfile
from app.services.cache_paths import CachePaths
from app.services.dataset_registry import DatasetRegistry
from app.services.duckdb_manager import DuckDBManager
from app.services.parquet_builder import build_parquet
from app.services.storage_client import SupabaseStorageClient


def _content_type_for_filename(name: str) -> str:
    n = name.lower()
    if n.endswith(".csv"):
        return "text/csv"
    if n.endswith(".parquet"):
        return "application/octet-stream"
    if n.endswith(".xlsx") or n.endswith(".xls"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


def create_dataset_from_upload(
    file: UploadFile,
    user_id: str,
    project_id: Optional[str] = None,
    base_dir: Path | None = None,
) -> Tuple[str, Path, DatasetCreateResponse]:
    """
    Shared dataset creation pipeline used by both the datasets router and legacy session upload.
    """
    dataset_id = str(uuid.uuid4())
    cache = CachePaths(base_dir=base_dir or Path("./cache"))
    raw_path = cache.raw_path(user_id, dataset_id, file.filename)
    parquet_path = cache.parquet_path(user_id, dataset_id)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw file locally
    raw_bytes = file.file.read()
    raw_path.write_bytes(raw_bytes)

    storage = SupabaseStorageClient()
    registry = DatasetRegistry()
    duck = DuckDBManager()

    raw_ref = f"{user_id}/datasets/{dataset_id}/raw/{file.filename}"
    storage.upload_file(raw_path, raw_ref, content_type=_content_type_for_filename(file.filename))

    try:
        build_parquet(raw_path, parquet_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse/convert file: {e}")

    parquet_ref = f"{user_id}/datasets/{dataset_id}/parquet/data.parquet"
    storage.upload_file(parquet_path, parquet_ref, content_type=_content_type_for_filename("data.parquet"))

    prof = duck.profile_parquet(parquet_path, sample_n=100)
    schema = prof["schema"]
    missing_summary = prof["missing_summary"]
    sample_rows = prof["sample_rows"]

    n_rows = prof["n_rows"]
    n_cols = len(schema)

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
    registry.create(row)

    profile = DatasetProfile(
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_cols=n_cols,
        schema=schema,
        missing_summary=missing_summary,
        sample_rows=sample_rows,
    )

    response = DatasetCreateResponse(
        dataset_id=dataset_id,
        raw_file_ref=raw_ref,
        parquet_ref=parquet_ref,
        profile=profile,
    )
    return dataset_id, parquet_path, response


def persist_dataframe(
    df,
    user_id: str,
    dataset_id: Optional[str] = None,
    project_id: Optional[str] = None,
    file_name: str = "generated.parquet",
    base_dir: Path | None = None,
) -> Tuple[str, Path, DatasetProfile]:
    """Write a dataframe to parquet, upload to Supabase, and upsert registry row."""

    dataset_id = dataset_id or str(uuid.uuid4())
    cache = CachePaths(base_dir=base_dir or Path("./cache"))
    parquet_path = cache.parquet_path(user_id, dataset_id)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Write parquet
    df.to_parquet(parquet_path, index=False)

    storage = SupabaseStorageClient()
    registry = DatasetRegistry()
    duck = DuckDBManager()

    parquet_ref = f"{user_id}/datasets/{dataset_id}/parquet/data.parquet"
    storage.upload_file(parquet_path, parquet_ref, content_type=_content_type_for_filename("data.parquet"))

    prof = duck.profile_parquet(parquet_path, sample_n=100)
    schema = prof["schema"]
    missing_summary = prof["missing_summary"]
    sample_rows = prof["sample_rows"]

    n_rows = prof["n_rows"]
    n_cols = len(schema)

    profile = DatasetProfile(
        dataset_id=dataset_id,
        n_rows=n_rows,
        n_cols=n_cols,
        schema=schema,
        missing_summary=missing_summary,
        sample_rows=sample_rows,
    )

    row = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "project_id": project_id,
        "file_name": file_name,
        "parquet_ref": parquet_ref,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "schema_json": schema,
        "profile_json": {"missing_summary": missing_summary},
    }

    try:
        registry.patch(dataset_id, user_id, row)
    except Exception:
        registry.create(row)

    return dataset_id, parquet_path, profile

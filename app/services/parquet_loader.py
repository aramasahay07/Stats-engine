from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import HTTPException

from app.services.cache_paths import CachePaths
from app.services.dataset_registry import DatasetRegistry
from app.services.storage_client import SupabaseStorageClient


def ensure_parquet_local(
    dataset_id: str,
    user_id: str,
    cache: CachePaths | None = None,
    storage_client: SupabaseStorageClient | None = None,
    registry: DatasetRegistry | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Guarantee a dataset's parquet is present locally.

    Returns the local parquet path and the dataset registry row. Raises HTTP errors
    so callers return appropriate API responses without duplicating boilerplate.
    """

    registry = registry or DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    parquet_ref = ds.get("parquet_ref")
    if not parquet_ref:
        raise HTTPException(status_code=400, detail="Dataset missing parquet_ref")

    cache = cache or CachePaths(base_dir=Path("./cache"))
    parquet_path = cache.parquet_path(user_id, dataset_id)

    if not parquet_path.exists():
        storage = storage_client or SupabaseStorageClient()
        storage.download_file(parquet_ref, parquet_path)

    return parquet_path, ds

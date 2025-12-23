from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.models.specs import QuerySpec, TableResult
from app.services.dataset_registry import DatasetRegistry
from app.services.storage_client import SupabaseStorageClient
from app.services.cache_paths import CachePaths
from app.services.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/datasets", tags=["query"])

@router.post("/{dataset_id}/query", response_model=TableResult)
def run_query(dataset_id: str, user_id: str, spec: QuerySpec):
    registry = DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    parquet_ref = ds.get("parquet_ref")
    if not parquet_ref:
        raise HTTPException(status_code=400, detail="Dataset missing parquet_ref")

    cache = CachePaths(base_dir=Path("./cache"))
    parquet_path = cache.parquet_path(user_id, dataset_id)
    if not parquet_path.exists():
        storage = SupabaseStorageClient()
        storage.download_file(parquet_ref, parquet_path)

    duck = DuckDBManager()
    sql, params = duck.build_query_sql("ds", spec)
    out = duck.query_parquet(parquet_path, sql, params)

    return TableResult(columns=out["columns"], rows=[list(r) for r in out["rows"]])

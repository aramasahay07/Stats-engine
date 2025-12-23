from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.specs import QuerySpec, TableResult
from app.services.cache_paths import CachePaths
from app.services.dataset_registry import DatasetRegistry
from app.services.duckdb_manager import DuckDBManager
from app.services.storage_client import SupabaseStorageClient

router = APIRouter(prefix="/datasets", tags=["query"])


def _has_placeholder_string(spec: QuerySpec) -> bool:
    """Detect Swagger/OpenAPI example placeholder values like 'string'.

    These placeholders commonly sneak into requests and then DuckDB fails with BinderError
    because a column named 'string' doesn't exist.
    """
    if getattr(spec, "select", None) and any(s == "string" for s in spec.select or []):
        return True
    if getattr(spec, "groupby", None) and any(s == "string" for s in spec.groupby or []):
        return True
    if getattr(spec, "order_by", None) and any(getattr(o, "col", None) == "string" for o in spec.order_by or []):
        return True
    if getattr(spec, "measures", None):
        for m in spec.measures or []:
            if getattr(m, "name", None) == "string" or getattr(m, "expr", None) == "string":
                return True
    if getattr(spec, "filters", None):
        for f in spec.filters or []:
            if getattr(f, "col", None) == "string":
                return True
    return False


@router.post("/{dataset_id}/query", response_model=TableResult)
def run_query(dataset_id: str, user_id: str, spec: QuerySpec):
    # -----------------------------
    # 1) Validate request inputs
    # -----------------------------
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing required query param: user_id")

    if _has_placeholder_string(spec):
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid QuerySpec: placeholder value 'string' detected. "
                "Send real column names from the dataset schema (example: select=['age'])."
            ),
        )

    # -----------------------------
    # 2) Find dataset + parquet ref
    # -----------------------------
    registry = DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    parquet_ref = ds.get("parquet_ref")
    if not parquet_ref:
        raise HTTPException(status_code=400, detail="Dataset missing parquet_ref")

    # -----------------------------
    # 3) Ensure parquet file exists locally (cache)
    # -----------------------------
    cache = CachePaths(base_dir=Path("./cache"))
    parquet_path = cache.parquet_path(user_id, dataset_id)

    if not parquet_path.exists():
        storage = SupabaseStorageClient()
        storage.download_file(parquet_ref, parquet_path)

    # -----------------------------
    # 4) Execute query via DuckDB
    # -----------------------------
    duck = DuckDBManager()
    sql, params = duck.build_query_sql("ds", spec)

    try:
        out = duck.query_parquet(parquet_path, sql, params)
    except Exception as e:
        # Convert common DuckDB binder/parsing errors into helpful client errors (422)
        msg = str(e)
        if "Binder Error" in msg or "Parser Error" in msg or "Referenced column" in msg:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Query failed due to invalid column/expression. "
                    "Make sure your QuerySpec uses real columns from the dataset schema. "
                    f"DuckDB error: {msg}"
                ),
            )
        raise  # keep unexpected failures as 500

    return TableResult(columns=out["columns"], rows=[list(r) for r in out["rows"]])

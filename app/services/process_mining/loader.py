from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import HTTPException

from app.config import settings
from app.db import registry
from app.engine.duckdb_engine import DuckDBEngine, DuckDBUnsupportedTypeError
from app.services.storage_supabase import SupabaseStorage


@dataclass
class ProcessMiningContext:
    user_id: str
    dataset_id: str
    dataset_row: Dict[str, Any]
    parquet_local: Path
    con: Any
    base_view: str

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


def _fingerprint_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dict(maybe_json: Any) -> Dict[str, Any]:
    if maybe_json is None:
        return {}
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            parsed = json.loads(maybe_json)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


async def _ensure_parquet_local(user_id: str, dataset_id: str) -> Path:
    local_path = Path(settings.data_dir) / "datasets" / user_id / dataset_id / "data.parquet"
    row = await registry.fetchrow(
        """
        SELECT parquet_ref, profile_json
        FROM datasets
        WHERE dataset_id = $1::uuid
          AND user_id = $2
        """,
        dataset_id,
        user_id,
    )
    if not row or not row.get("parquet_ref"):
        raise HTTPException(status_code=409, detail="Dataset parquet is not ready yet.")

    profile = _ensure_dict(row.get("profile_json"))
    expected_sha = profile.get("parquet_sha")
    if local_path.exists():
        if not expected_sha or _fingerprint_file(local_path) == expected_sha:
            return local_path
        try:
            local_path.unlink()
        except FileNotFoundError:
            pass

    local_path.parent.mkdir(parents=True, exist_ok=True)
    storage = SupabaseStorage()
    local_path.write_bytes(await storage.download(row["parquet_ref"]))
    return local_path


async def load_process_mining_context(user_id: str, dataset_id: str) -> ProcessMiningContext:
    row_any = await registry.fetchrow(
        """
        SELECT dataset_id, user_id, parquet_ref, state, version, error_message, schema_json, profile_json
        FROM datasets
        WHERE dataset_id = $1::uuid
        """,
        dataset_id,
    )
    if not row_any:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row_user_id = row_any.get("user_id") if hasattr(row_any, "get") else row_any["user_id"]
    if row_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    state = (row_any.get("state") if hasattr(row_any, "get") else row_any["state"]) or "ready"
    if state in ("processing", "reprocessing"):
        raise HTTPException(status_code=409, detail="Dataset is still processing.")
    if state == "failed":
        err = row_any.get("error_message") if hasattr(row_any, "get") else row_any["error_message"]
        raise HTTPException(status_code=422, detail=err or "Dataset processing failed")

    parquet_ref = row_any.get("parquet_ref") if hasattr(row_any, "get") else row_any["parquet_ref"]
    if not parquet_ref:
        raise HTTPException(status_code=409, detail="Dataset parquet is not ready yet.")

    parquet_local = await _ensure_parquet_local(user_id, dataset_id)
    engine = DuckDBEngine(user_id)
    con = engine.connect()
    try:
        base_view = engine.register_parquet(con, dataset_id, parquet_local)
    except DuckDBUnsupportedTypeError as exc:
        con.close()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        con.close()
        raise

    return ProcessMiningContext(
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_row=dict(row_any),
        parquet_local=parquet_local,
        con=con,
        base_view=base_view,
    )

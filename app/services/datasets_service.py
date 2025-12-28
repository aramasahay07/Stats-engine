from __future__ import annotations
from pathlib import Path
import os
from typing import Tuple, Dict, Any, Optional

from fastapi import UploadFile
from app.config import settings
from app.services.storage_supabase import SupabaseStorage
from app.services import jobs_service
from app.engine.ingest import new_dataset_id, csv_to_parquet_streaming, xlsx_to_parquet, parquet_copy
from app.engine.duckdb_engine import DuckDBEngine
from app.engine.profiling import build_profile_from_duckdb
from app.db import registry

class DatasetService:
    def __init__(self):
        self.storage = SupabaseStorage()

    def _paths(self, user_id: str, dataset_id: str) -> Dict[str, str]:
        base = f"datasets/{user_id}/{dataset_id}"
        return {
            "raw": f"{base}/raw",
            "parquet": f"{base}/data.parquet",
        }

    def _local_dir(self, user_id: str, dataset_id: str) -> Path:
        p = Path(settings.data_dir) / "datasets" / user_id / dataset_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    async def create_dataset_record(self, user_id: str, project_id: Optional[str], file_name: str, raw_file_ref: str) -> str:
        dataset_id = new_dataset_id()
        await registry.execute(
            """INSERT INTO datasets (dataset_id, user_id, project_id, file_name, raw_file_ref)
               VALUES ($1,$2,$3,$4,$5)""",
            dataset_id, user_id, project_id, file_name, raw_file_ref
        )
        return dataset_id

    async def save_raw_to_storage(self, user_id: str, dataset_id: str, upload: UploadFile) -> Tuple[Path, str]:
        local_dir = self._local_dir(user_id, dataset_id)
        local_raw = local_dir / upload.filename
        with local_raw.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # store in Supabase Storage
        paths = self._paths(user_id, dataset_id)
        raw_ref = f"{paths['raw']}{Path(upload.filename).suffix.lower()}"
        await self.storage.upload_file(local_raw, raw_ref, upload.content_type or "application/octet-stream")
        return local_raw, raw_ref

    async def build_parquet_and_profile(self, user_id: str, dataset_id: str, raw_local: Path, raw_ref: str, job_id: str) -> Dict[str, Any]:
        await jobs_service.update_job(job_id, "running", 5, "starting ingest")

        local_dir = self._local_dir(user_id, dataset_id)
        parquet_local = local_dir / "data.parquet"

        suffix = raw_local.suffix.lower()
        if suffix == ".csv":
            await jobs_service.update_job(job_id, "running", 15, "converting csv to parquet")

            n_rows, n_cols = csv_to_parquet_streaming(raw_local, parquet_local)
        elif suffix in [".xlsx", ".xls"]:
            await jobs_service.update_job(job_id, "running", 15, "converting excel to parquet")

            n_rows, n_cols = xlsx_to_parquet(raw_local, parquet_local)
        elif suffix == ".parquet":
            await jobs_service.update_job(job_id, "running", 15, "copying parquet")

            n_rows, n_cols = parquet_copy(raw_local, parquet_local)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        await jobs_service.update_job(job_id, "running", 55, "uploading parquet")

        paths = self._paths(user_id, dataset_id)
        parquet_ref = paths["parquet"]
        await self.storage.upload_file(parquet_local, parquet_ref, "application/octet-stream")

        # Profile using DuckDB (single truth)
        await jobs_service.update_job(job_id, "running", 70, "profiling")

        eng = DuckDBEngine(user_id)
        con = eng.connect()
        base_view = eng.register_parquet(con, dataset_id, parquet_local)
        profile = build_profile_from_duckdb(con, base_view)
        con.close()

        await jobs_service.update_job(job_id, "running", 90, "saving metadata")

        await registry.execute(
            """UPDATE datasets
               SET parquet_ref=$2, n_rows=$3, n_cols=$4, schema_json=$5, profile_json=$6, updated_at=NOW()
               WHERE dataset_id=$1 AND user_id=$7""",
            dataset_id, parquet_ref, profile["n_rows"], profile["n_cols"], profile["schema"], profile, user_id
        )

        await jobs_service.update_job(job_id, "done", 100, "complete", {"profile": profile})
        return profile

dataset_service = DatasetService()

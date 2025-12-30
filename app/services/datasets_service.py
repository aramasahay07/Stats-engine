from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from uuid import uuid4, UUID

from fastapi import UploadFile

from app.config import settings
from app.services.storage_supabase import SupabaseStorage
from app.services import jobs_service
from app.engine.ingest import csv_to_parquet_streaming, xlsx_to_parquet, parquet_copy
from app.engine.duckdb_engine import DuckDBEngine
from app.engine.profiling import build_profile_from_duckdb
from app.db import registry


# -----------------------------------------------------------------------------
# Engine version — used for snapshot & cache invalidation
# -----------------------------------------------------------------------------
ENGINE_VERSION = "v2.1.0"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def fingerprint_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Dataset Service
# -----------------------------------------------------------------------------
class DatasetService:
    def __init__(self):
        self.storage = SupabaseStorage()

    # -------------------------------------------------------------------------
    # Path helpers
    # -------------------------------------------------------------------------
    def _paths(self, user_id: str, dataset_id: str) -> Dict[str, str]:
        base = f"{user_id}/datasets/{dataset_id}"
        return {
            "raw_dir": f"{base}/raw",
            "parquet": f"{base}/parquet/data.parquet",
        }

    def _local_dir(self, user_id: str, dataset_id: str) -> Path:
        p = Path(settings.data_dir) / "datasets" / user_id / dataset_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------------------------------------------------
    # Dataset record creation
    # -------------------------------------------------------------------------
    async def create_dataset_record(
        self,
        user_id: str,
        project_id: Optional[UUID],
        file_name: str,
    ) -> str:
        dataset_id = str(uuid4())

        paths = self._paths(user_id, dataset_id)
        raw_ref = f"{paths['raw_dir']}/{file_name}"
        parquet_ref = paths["parquet"]

        await registry.execute(
            """
            INSERT INTO datasets (
                dataset_id,
                user_id,
                project_id,
                file_name,
                raw_file_ref,
                parquet_ref
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            """,
            dataset_id,
            user_id,
            project_id,
            file_name,
            raw_ref,
            parquet_ref,
        )

        return dataset_id

    # -------------------------------------------------------------------------
    # Save raw file
    # -------------------------------------------------------------------------
    async def save_raw_to_storage(
        self,
        user_id: str,
        dataset_id: str,
        upload: UploadFile,
    ) -> Tuple[Path, str]:
        local_dir = self._local_dir(user_id, dataset_id)
        local_raw = local_dir / upload.filename

        with local_raw.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        paths = self._paths(user_id, dataset_id)
        raw_ref = f"{paths['raw_dir']}/{upload.filename}"

        await self.storage.upload_file(
            local_raw,
            raw_ref,
            upload.content_type or "application/octet-stream",
        )

        return local_raw, raw_ref

    # -------------------------------------------------------------------------
    # Build parquet, profile, and persist snapshot metadata
    # -------------------------------------------------------------------------
    async def build_parquet_and_profile(
        self,
        user_id: str,
        dataset_id: str,
        raw_local: Path,
        raw_ref: str,
        job_id: str,
    ) -> Dict[str, Any]:
        try:
            await jobs_service.update_job(job_id, "running", 5, "starting ingest")

            local_dir = self._local_dir(user_id, dataset_id)
            parquet_local = local_dir / "data.parquet"

            suffix = raw_local.suffix.lower()
            if suffix == ".csv":
                await jobs_service.update_job(job_id, "running", 15, "csv → parquet")
                csv_to_parquet_streaming(raw_local, parquet_local)
            elif suffix in [".xlsx", ".xls"]:
                await jobs_service.update_job(job_id, "running", 15, "excel → parquet")
                xlsx_to_parquet(raw_local, parquet_local)
            elif suffix == ".parquet":
                await jobs_service.update_job(job_id, "running", 15, "copy parquet")
                parquet_copy(raw_local, parquet_local)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            await jobs_service.update_job(job_id, "running", 55, "uploading parquet")

            paths = self._paths(user_id, dataset_id)
            parquet_ref = paths["parquet"]

            await self.storage.upload_file(
                parquet_local,
                parquet_ref,
                "application/octet-stream",
            )

            await jobs_service.update_job(job_id, "running", 70, "profiling")

            eng = DuckDBEngine(user_id)
            con = eng.connect()
            try:
                base_view = eng.register_parquet(con, dataset_id, parquet_local)
                profile = build_profile_from_duckdb(con, base_view)
            finally:
                con.close()

            await jobs_service.update_job(job_id, "running", 90, "saving metadata")

            # -----------------------------------------------------------------
            # Snapshot metadata (REQUIRED by stats_service)
            # -----------------------------------------------------------------
            parquet_sha = fingerprint_file(parquet_local)

            profile["parquet_ref"] = parquet_ref
            profile["parquet_sha"] = parquet_sha
            profile["pipeline_hash"] = "__none__"
            profile["engine_version"] = ENGINE_VERSION

            # Ensure numeric_summary exists
            profile.setdefault("numeric_summary", {})

            schema_payload = json.dumps(profile.get("schema") or [])
            profile_payload = json.dumps(profile)

            result = await registry.execute(
                """
                UPDATE datasets
                SET parquet_ref = $2,
                    n_rows = $3,
                    n_cols = $4,
                    schema_json = $5::jsonb,
                    profile_json = $6::jsonb,
                    updated_at = NOW()
                WHERE dataset_id = $1::uuid
                """,
                dataset_id,
                parquet_ref,
                int(profile.get("n_rows") or 0),
                int(profile.get("n_cols") or 0),
                schema_payload,
                profile_payload,
            )

            if not str(result).endswith("1"):
                raise RuntimeError(f"Dataset update failed: {result}")

            await jobs_service.update_job(
                job_id,
                "done",
                100,
                "complete",
                {"profile": profile},
            )

            return profile

        except Exception as e:
            try:
                await jobs_service.update_job(
                    job_id,
                    "failed",
                    100,
                    f"{type(e).__name__}: {e}",
                )
            except Exception:
                pass
            raise


# Singleton
dataset_service = DatasetService()

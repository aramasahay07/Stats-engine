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
from app.engine.duckdb_engine import DuckDBEngine, DuckDBUnsupportedTypeError
from app.engine.profiling import build_profile_from_duckdb
from app.db import registry
from app.engine.pipeline import ensure_pipeline_view, pipeline_hash
from app.models.pipelines import PipelineStep


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

    async def _ensure_parquet_local(self, user_id: str, dataset_id: str) -> Path:
        p = Path(settings.data_dir) / "datasets" / user_id / dataset_id / "data.parquet"
        if p.exists():
            return p

        row = await registry.fetchrow(
            "SELECT parquet_ref FROM datasets WHERE dataset_id = $1::uuid AND user_id=$2",
            dataset_id,
            user_id,
        )
        if not row or not row.get("parquet_ref"):
            raise FileNotFoundError("Parquet artifact not found (dataset still building or missing parquet_ref).")

        p.parent.mkdir(parents=True, exist_ok=True)
        file_bytes = await self.storage.download(row["parquet_ref"])
        p.write_bytes(file_bytes)
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
                parquet_ref,
                state,
                version
            )
            VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8)
            """,
            dataset_id,
            user_id,
            project_id,
            file_name,
            raw_ref,
            parquet_ref,
            "processing",
            1,
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
                base_view, detected_issues = eng.register_parquet_with_issues(con, dataset_id, parquet_local)
                profile = build_profile_from_duckdb(con, base_view)

                profile.setdefault("issues", [])
                profile["issues"].extend(detected_issues)

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
                    state = 'ready',
                    error_message = NULL,
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
            # ✅ Fix #2: Do NOT fail dataset for fixable unsupported-type errors
            msg = f"{type(e).__name__}: {e}"

            is_fixable_unsupported = (
                isinstance(e, DuckDBUnsupportedTypeError)
                or "TIME WITH TIME ZONE" in msg
                or "Unsupported type" in msg
            )

            if is_fixable_unsupported:
                issues = [
                    {
                        "column": "__unknown__",
                        "issue_type": "unsupported_type",
                        "severity": "blocking",
                        "details": {"message": msg},
                        "suggested_fix": {"op": "change_type", "to": "varchar"},
                    }
                ]

                safe_profile = {
                    "n_rows": 0,
                    "n_cols": 0,
                    "schema": [],
                    "sample_rows": [],
                    "issues": issues,
                    "engine_version": ENGINE_VERSION,
                }

                await registry.execute(
                    """
                    UPDATE datasets
                    SET state = 'ready',
                        error_message = NULL,
                        profile_json = $2::jsonb,
                        schema_json = $3::jsonb,
                        updated_at = NOW()
                    WHERE dataset_id = $1::uuid
                    """,
                    dataset_id,
                    json.dumps(safe_profile),
                    json.dumps([]),
                )

                await jobs_service.update_job(
                    job_id,
                    "done",
                    100,
                    "complete",
                    {
                        "warning": "Dataset requires formatting fixes",
                        "error": msg,
                    },
                )

                return safe_profile

            # ❌ Real error → dataset truly failed
            try:
                await registry.execute(
                    """
                    UPDATE datasets
                    SET state = 'failed',
                        error_message = $2,
                        updated_at = NOW()
                    WHERE dataset_id = $1::uuid
                    """,
                    dataset_id,
                    msg,
                )
            except Exception:
                pass

            raise


    async def apply_transforms_and_version(
        self,
        user_id: str,
        dataset_id: str,
        transforms: list[PipelineStep],
    ) -> Dict[str, Any]:
        # 1) Read current dataset info + ownership
        row = await registry.fetchrow(
            """
            SELECT dataset_id, user_id, parquet_ref, version, state
            FROM datasets
            WHERE dataset_id = $1::uuid AND user_id::text = $2
            """,
            dataset_id,
            user_id,
        )
        if not row:
            raise FileNotFoundError("Dataset not found")
        if (row.get("state") or "ready") != "ready":
            raise ValueError("Dataset is not ready yet")

        current_version = int(row.get("version") or 1)
        new_version = current_version + 1

        # 2) Mark dataset as processing
        await registry.execute(
            """
            UPDATE datasets
            SET state = 'processing',
                updated_at = NOW()
            WHERE dataset_id = $1::uuid
            """,
            dataset_id,
        )

        # 3) Ensure parquet exists locally
        parquet_local = await self._ensure_parquet_local(user_id, dataset_id)

        # 4) Run transforms in DuckDB and materialize to a NEW parquet
        local_dir = self._local_dir(user_id, dataset_id)
        out_local = local_dir / f"data_v{new_version}.parquet"

        eng = DuckDBEngine(user_id)
        con = eng.connect()
        try:
            base_view = eng.register_parquet(con, dataset_id, parquet_local)


            # Apply transforms as a pipeline view
            piped_view = ensure_pipeline_view(con, dataset_id, base_view, transforms)

            # Materialize full dataset
            con.execute(f"COPY (SELECT * FROM {piped_view}) TO '{out_local.as_posix()}' (FORMAT PARQUET)")
        finally:
            con.close()

        # 5) Upload new parquet to a versioned storage location
        base = f"{user_id}/datasets/{dataset_id}"
        new_parquet_ref = f"{base}/parquet/v{new_version}/data.parquet"

        await self.storage.upload_file(out_local, new_parquet_ref, "application/octet-stream")

        # 6) Re-profile the new parquet
        eng = DuckDBEngine(user_id)
        con = eng.connect()
        try:
            base_view, detected_issues = eng.register_parquet_with_issues(con, dataset_id, out_local)
            profile = build_profile_from_duckdb(con, base_view)

            profile.setdefault("issues", [])
            profile["issues"].extend(detected_issues)

        finally:
            con.close()

        # 7) Snapshot metadata updates
        parquet_sha = fingerprint_file(out_local)
        ph = pipeline_hash(transforms)

        profile["parquet_ref"] = new_parquet_ref
        profile["parquet_sha"] = parquet_sha
        profile["pipeline_hash"] = ph
        profile["engine_version"] = ENGINE_VERSION
        profile.setdefault("numeric_summary", {})

        schema_payload = json.dumps(profile.get("schema") or [])
        profile_payload = json.dumps(profile)

        # 8) Persist updated dataset state + bump version
        await registry.execute(
            """
            UPDATE datasets
            SET parquet_ref = $2,
                n_rows = $3,
                n_cols = $4,
                schema_json = $5::jsonb,
                profile_json = $6::jsonb,
                version = $7,
                state = 'ready',
                error_message = NULL,
                updated_at = NOW()
            WHERE dataset_id = $1::uuid
            """,
            dataset_id,
            new_parquet_ref,
            int(profile.get("n_rows") or 0),
            int(profile.get("n_cols") or 0),
            schema_payload,
            profile_payload,
            new_version,
        )

        return {
            "dataset_id": dataset_id,
            "version": new_version,
            "parquet_ref": new_parquet_ref,
            "profile": profile,
        }


# Singleton
dataset_service = DatasetService()
   
from __future__ import annotations

from pathlib import Path
import httpx

from app.config import settings


class SupabaseStorage:
    """
    Minimal async Supabase Storage client using REST API.
    Fixes httpx error: 'Attempted to send a sync request with an AsyncClient instance.'
    """

    def __init__(self):
        if not settings.supabase_url:
            raise RuntimeError("SUPABASE_URL is not set")
        if not settings.supabase_service_role_key:
            raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is not set")

        # If you have SUPABASE_STORAGE_BUCKET in Railway, add it to config.py
        # For now default to "datasets"
        self.bucket = getattr(settings, "supabase_storage_bucket", "datasets")

        self.base = settings.supabase_url.rstrip("/")
        self.key = settings.supabase_service_role_key

    def _headers(self, content_type: str) -> dict:
        return {
            "Authorization": f"Bearer {self.key}",
            "apikey": self.key,
            "Content-Type": content_type,
            # Upsert allows overwriting same path safely
            "x-upsert": "true",
        }

    async def upload_file(self, local_path: Path, object_path: str, content_type: str) -> None:
        """
        Uploads a local file to Supabase Storage bucket at:
        bucket/object_path

        object_path example:
          user_id/datasets/<dataset_id>/raw/file.csv
        """
        url = f"{self.base}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        data = local_path.read_bytes()

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=self._headers(content_type), content=data)

        # Supabase returns 200/201 on success; any 4xx/5xx should be surfaced
        if resp.status_code not in (200, 201):
            raise RuntimeError(
                f"Supabase upload failed: {resp.status_code} {resp.text}"
            )

    async def delete_object(self, object_path: str) -> None:
        url = f"{self.base}/storage/v1/object/{self.bucket}/{object_path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.delete(url, headers=self._headers("application/octet-stream"))

        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Supabase delete failed: {resp.status_code} {resp.text}")

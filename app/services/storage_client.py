from __future__ import annotations

import os
import requests
from pathlib import Path
from typing import Optional, Tuple

from supabase import create_client, Client

class SupabaseStorageClient:
    """Small wrapper for Supabase Storage using service role key (server-to-server).

    Env vars expected:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY
      SUPABASE_STORAGE_BUCKET (default: datasets)
    """

    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_SERVICE_KEY")
        self.bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "datasets")
        self.client: Client = create_client(url, key)

    def upload_file(self, local_path: Path, storage_ref: str, content_type: str = "application/octet-stream") -> None:
        data = local_path.read_bytes()
        # upsert=True so re-runs are safe
        self.client.storage.from_(self.bucket).upload(
            path=storage_ref,
            file=data,
            file_options={"content-type": content_type, "upsert": "true"},
        )

    def download_file(self, storage_ref: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        res = self.client.storage.from_(self.bucket).download(storage_ref)
        # `download` returns bytes
        local_path.write_bytes(res)

    def signed_url(self, storage_ref: str, expires_in: int = 3600) -> str:
        # returns dict with 'signedURL'
        out = self.client.storage.from_(self.bucket).create_signed_url(storage_ref, expires_in)
        # supabase-py returns object-like with data, but also may return dict
        signed = getattr(out, "get", None)
        if callable(signed):
            return out["signedURL"]
        # fallback
        return out.get("signedURL")  # type: ignore

    def download_from_url(self, url: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        local_path.write_bytes(r.content)

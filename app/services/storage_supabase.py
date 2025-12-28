import httpx
from pathlib import Path
from app.config import settings

class SupabaseStorage:
    def __init__(self):
        self.base = settings.supabase_url.rstrip("/")
        self.bucket = settings.bucket_name
        self.headers = {
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "apikey": settings.supabase_service_role_key,
        }

    async def upload_file(self, local_path: Path, remote_path: str, content_type: str):
        url = f"{self.base}/storage/v1/object/{self.bucket}/{remote_path}"
        h = dict(self.headers)
        h["Content-Type"] = content_type
        async with httpx.AsyncClient(timeout=None) as client:
            with local_path.open("rb") as f:
                r = await client.put(url, headers=h, content=f)
        r.raise_for_status()
        return True

    async def download_to_file(self, remote_path: str, local_path: Path):
        url = f"{self.base}/storage/v1/object/{self.bucket}/{remote_path}"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", url, headers=self.headers) as r:
                r.raise_for_status()
                with local_path.open("wb") as f:
                    async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
        return local_path

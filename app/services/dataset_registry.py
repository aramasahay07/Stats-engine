from __future__ import annotations

import os
from typing import Any, Dict, Optional
from supabase import create_client, Client

class DatasetRegistry:
    """Supabase Postgres wrapper for datasets table.

    Env vars:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY
      DATASETS_TABLE (default: datasets)
    """

    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_SERVICE_KEY")
        self.client: Client = create_client(url, key)
        self.table = os.environ.get("DATASETS_TABLE", "datasets")

    def create(self, row: Dict[str, Any]) -> Dict[str, Any]:
        res = self.client.table(self.table).insert(row).execute()
        data = res.data[0] if res.data else None
        if not data:
            raise RuntimeError("Failed to create dataset row")
        return data

    def get(self, dataset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        res = self.client.table(self.table).select("*").eq("dataset_id", dataset_id).eq("user_id", user_id).limit(1).execute()
        return res.data[0] if res.data else None

    def patch(self, dataset_id: str, user_id: str, patch: Dict[str, Any]) -> None:
        self.client.table(self.table).update(patch).eq("dataset_id", dataset_id).eq("user_id", user_id).execute()

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, Optional

from fastapi import HTTPException
from supabase import Client, create_client


@dataclass
class SessionDatasetLink:
    session_id: str
    dataset_id: str
    user_id: str
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionDatasetBridge:
    """Persistent compatibility bridge mapping legacy session_id to dataset_id."""

    def __init__(self, table: str | None = None):
        self.table = table or os.getenv("SESSION_DATASET_LINKS_TABLE", "session_dataset_links")
        self._client: Client | None = None

    def _get_client(self) -> Client:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise HTTPException(status_code=500, detail="Supabase credentials missing for session bridge")
        if self._client is None:
            self._client = create_client(url, key)
        return self._client

    def _row_to_link(self, row: Dict[str, Any]) -> SessionDatasetLink:
        return SessionDatasetLink(
            session_id=row.get("session_id"),
            dataset_id=row.get("dataset_id"),
            user_id=row.get("user_id"),
            project_id=row.get("project_id"),
            metadata=row.get("metadata") or {},
        )

    def get(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionDatasetLink]:
        client = self._get_client()
        query = client.table(self.table).select("*").eq("session_id", session_id)
        if user_id:
            query = query.eq("user_id", user_id)
        res = query.limit(1).execute()
        row = res.data[0] if res.data else None
        if not row:
            return None
        return self._row_to_link(row)

    def require(self, session_id: str, user_id: Optional[str] = None) -> SessionDatasetLink:
        link = self.get(session_id, user_id=user_id)
        if not link:
            raise HTTPException(status_code=404, detail="Unknown session_id. Re-upload data or use dataset endpoints.")
        return link

    def ensure(self, session_id: str, dataset_id: str, user_id: str, project_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SessionDatasetLink:
        link = self.get(session_id, user_id=user_id)
        if link:
            return link
        link = SessionDatasetLink(
            session_id=session_id,
            dataset_id=dataset_id,
            user_id=user_id,
            project_id=project_id,
            metadata=metadata or {},
        )
        self._upsert(link)
        return link

    def delete(self, session_id: str, user_id: Optional[str] = None) -> None:
        client = self._get_client()
        query = client.table(self.table).delete().eq("session_id", session_id)
        if user_id:
            query = query.eq("user_id", user_id)
        query.execute()

    def _upsert(self, link: SessionDatasetLink) -> None:
        client = self._get_client()
        payload = {
            "session_id": link.session_id,
            "dataset_id": link.dataset_id,
            "user_id": link.user_id,
            "project_id": link.project_id,
            "metadata": link.metadata,
        }
        client.table(self.table).upsert(payload, on_conflict="session_id").execute()


session_bridge = SessionDatasetBridge()

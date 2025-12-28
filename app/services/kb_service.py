from __future__ import annotations
from app.db import registry
from pydantic import BaseModel
from app.config import settings

# Set these to match your existing Supabase KB tables (140 concepts).
# If your table names differ, change them here (single source of truth).
KB_CONCEPTS_TABLE = "statistical_concepts"

async def search_concepts(q: str, limit: int = 20):
    rows = await registry.fetch(
        f"SELECT id, title, tags, short_summary FROM {KB_CONCEPTS_TABLE} WHERE title ILIKE $1 OR short_summary ILIKE $1 LIMIT {limit}",
        f"%{q}%",
    )
    return [{"id": r["id"], "title": r["title"], "tags": r.get("tags"), "short_summary": r.get("short_summary")} for r in rows]

async def get_concept(concept_id: str):
    row = await registry.fetchrow(
        f"SELECT * FROM {KB_CONCEPTS_TABLE} WHERE id=$1",
        concept_id,
    )
    if not row:
        return None
    return dict(row)

"""
routers/kb.py
FastAPI router for KB endpoints.

Mount in main.py:
    from routers.kb import router as kb_router
    app.include_router(kb_router, prefix="/kb", tags=["Knowledge Base"])
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict

from knowledge.queries import get_all_topics, get_concept_by_slug, search_concepts
from knowledge.enrichment import enrich_stats_result

router = APIRouter()


@router.get("/topics")
def list_topics():
    return {"topics": get_all_topics()}


@router.get("/concepts/{slug}")
def concept(slug: str, include_children: bool = True):
    c = get_concept_by_slug(slug, include_children=include_children)
    if not c:
        raise HTTPException(status_code=404, detail="Concept not found")
    return c


@router.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = 20):
    return {"results": search_concepts(q=q, limit=limit)}


@router.post("/enrich")
def enrich(payload: Dict[str, Any]):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    return enrich_stats_result(payload)

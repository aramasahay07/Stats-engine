from __future__ import annotations
from fastapi import APIRouter, Depends
from app.auth.supabase_jwt import get_current_user
from app.services.kb_service import search_concepts, get_concept

router = APIRouter()

@router.get("/concepts/search")
async def kb_search(q: str, user=Depends(get_current_user)):
    # private per user app, but KB can be shared; keep auth required
    return await search_concepts(q)

@router.get("/concepts/{concept_id}")
async def kb_get(concept_id: str, user=Depends(get_current_user)):
    return await get_concept(concept_id)

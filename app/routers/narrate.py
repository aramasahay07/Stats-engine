from __future__ import annotations
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from app.auth.supabase_jwt import get_current_user

router = APIRouter()

class NarrateRequest(BaseModel):
    user_question: str
    computed: Dict[str, Any]
    kb_context: Optional[Dict[str, Any]] = None
    tone: str = "executive"

@router.post("", response_model=dict)
async def narrate(req: NarrateRequest, user=Depends(get_current_user)):
    # NOTE: In production, you may prefer to do narration in edge functions using OPENAI_API_KEY there.
    # This endpoint exists so you can choose either approach.
    # Return a deterministic placeholder structure; plug in OpenAI later.
    return {
        "message": "(narration placeholder)",
        "question": req.user_question,
        "highlights": [],
        "computed": req.computed,
    }

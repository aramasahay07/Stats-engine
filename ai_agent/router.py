from typing import Any, Dict, List, Optional, Callable
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai_agent.agents.agent_endpoints import AgentEndpoints, format_findings_json

# One shared instance is fine (stateless agents)
_agent = AgentEndpoints()


class ExploreRequest(BaseModel):
    session_id: str
    focus_areas: Optional[List[str]] = None


class PatternsRequest(BaseModel):
    session_id: str
    exploration_results: Optional[Dict[str, Any]] = None


class CausalityRequest(BaseModel):
    session_id: str
    patterns: Optional[Dict[str, Any]] = None
    domain: str = "general"


class UnifiedAnalysisRequest(BaseModel):
    session_id: str
    agents: List[str] = ["explore", "patterns", "causality"]
    domain: str = "general"


def create_agent_router(get_df: Callable[[str], Optional[pd.DataFrame]]) -> APIRouter:
    """
    Factory so we can reuse main.py's session store without duplicating it.
    get_df(session_id) must return a pandas DataFrame or None.
    """
    router = APIRouter(tags=["agents"])

    @router.post("/explore")
    async def explore(req: ExploreRequest):
        df = get_df(req.session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return await _agent.explore(df=df, session_id=req.session_id, focus_areas=req.focus_areas)

    @router.post("/patterns")
    async def patterns(req: PatternsRequest):
        df = get_df(req.session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return await _agent.detect_patterns(
            df=df,
            session_id=req.session_id,
            exploration_results=req.exploration_results,
        )

    @router.post("/causality")
    async def causality(req: CausalityRequest):
        df = get_df(req.session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found")

        return await _agent.infer_causality(
            df=df,
            session_id=req.session_id,
            patterns=req.patterns,
            domain=req.domain,
        )

    @router.post("/unified")
    async def unified(req: UnifiedAnalysisRequest):
        df = get_df(req.session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="Session not found")

        exploration = patterns_res = causality_res = None

        # Run only what is requested (and in a sensible order)
        if "explore" in req.agents:
            exploration = await _agent.explore(df=df, session_id=req.session_id, focus_areas=None)

        if "patterns" in req.agents:
            patterns_res = await _agent.detect_patterns(
                df=df,
                session_id=req.session_id,
                exploration_results=exploration,
            )

        if "causality" in req.agents:
            causality_res = await _agent.infer_causality(
                df=df,
                session_id=req.session_id,
                patterns=patterns_res,
                domain=req.domain,
            )

        findings = format_findings_json(
            exploration=exploration,
            patterns=patterns_res,
            causality=causality_res,
        )

        return {
            "session_id": req.session_id,
            "agents_run": req.agents,
            "findings": findings,
            "raw": {
                "exploration": exploration,
                "patterns": patterns_res,
                "causality": causality_res,
            },
        }

    return router

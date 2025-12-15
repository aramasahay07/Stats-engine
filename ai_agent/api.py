"""
FastAPI Stats Engine with Individual Agent Endpoints
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4
from io import BytesIO
import pandas as pd
import asyncio
from datetime import datetime

# Import agent endpoints wrapper
from agents.agent_endpoints import AgentEndpoints, format_findings_json

# ---------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------
app = FastAPI(
    title="AI Data Lab - Stats Engine with Agent Endpoints",
    version="3.1.0",
    description="Individual agent endpoints + unified analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# In-memory storage
# ---------------------------------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Initialize agent endpoints
agent_endpoints = AgentEndpoints()

# ---------------------------------------------------
# Pydantic models
# ---------------------------------------------------
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
    agents: List[str] = ["explore", "patterns", "causality"]  # Which agents to run
    domain: str = "general"

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def _load_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read CSV or Excel file into pandas DataFrame"""
    content = file.file.read()
    file.file.close()
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    buffer = BytesIO(content)
    filename = file.filename.lower()
    
    if filename.endswith(".csv"):
        df = pd.read_csv(buffer)
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    if df.empty:
        raise HTTPException(status_code=400, detail="File contained no rows")
    
    return df

# ---------------------------------------------------
# Endpoints
# ---------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "3.1.0",
        "endpoints": {
            "upload": "POST /upload",
            "explore": "POST /agents/explore",
            "patterns": "POST /agents/patterns",
            "causality": "POST /agents/causality",
            "unified": "POST /agents/unified",
            "legacy_analyze": "POST /analyze (deprecated, use /agents/unified)"
        },
        "agents_available": ["data_explorer", "pattern_detective", "causal_reasoner"]
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV/Excel and create session
    """
    df = _load_dataframe(file)
    df = df.dropna(axis=1, how="all")
    
    session_id = str(uuid4())
    SESSIONS[session_id] = {
        "dataframe": df,
        "filename": file.filename,
        "uploaded_at": datetime.now().isoformat(),
    }
    
    return {
        "session_id": session_id,
        "status": "ready",
        "data_summary": {
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)[:20]
        }
    }


@app.post("/agents/explore")
async def explore_data(request: ExploreRequest):
    """
    Data Explorer Agent Endpoint
    
    Returns:
    - Data profile (column types, statistics)
    - Quality report (issues, score)
    - Interesting features
    - Preliminary insights
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = await agent_endpoints.explore(
            df=session["dataframe"],
            session_id=request.session_id,
            focus_areas=request.focus_areas
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data exploration failed: {str(e)}"
        )


@app.post("/agents/patterns")
async def detect_patterns(request: PatternsRequest):
    """
    Pattern Detective Agent Endpoint
    
    Returns:
    - Temporal patterns (trends, seasonality)
    - Clustering patterns (natural groups)
    - Association patterns (chi-square, ANOVA)
    - Anomaly patterns
    - Interaction patterns
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = await agent_endpoints.detect_patterns(
            df=session["dataframe"],
            session_id=request.session_id,
            exploration_results=request.exploration_results
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pattern detection failed: {str(e)}"
        )


@app.post("/agents/causality")
async def infer_causality(request: CausalityRequest):
    """
    Causal Reasoner Agent Endpoint
    
    Returns:
    - Causal hypotheses
    - Causal graph (nodes, edges)
    - Potential confounders
    - Validation recommendations
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        result = await agent_endpoints.infer_causality(
            df=session["dataframe"],
            session_id=request.session_id,
            patterns=request.patterns,
            domain=request.domain
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Causal inference failed: {str(e)}"
        )


@app.post("/agents/unified")
async def unified_analysis(request: UnifiedAnalysisRequest):
    """
    Unified Analysis - Run multiple agents and combine results
    
    This is the main endpoint that orchestrates all agents
    and returns Findings JSON in the format expected by Edge Function
    
    Args:
        agents: List of agents to run ["explore", "patterns", "causality"]
        domain: Domain knowledge context (healthcare, finance, etc.)
    
    Returns:
        Findings JSON with metadata, statistics, patterns, tests
    """
    session = SESSIONS.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session["dataframe"]
    results = {}
    
    try:
        # Run requested agents
        if "explore" in request.agents:
            results["exploration"] = await agent_endpoints.explore(
                df=df,
                session_id=request.session_id
            )
        
        if "patterns" in request.agents:
            results["patterns"] = await agent_endpoints.detect_patterns(
                df=df,
                session_id=request.session_id,
                exploration_results=results.get("exploration")
            )
        
        if "causality" in request.agents:
            results["causality"] = await agent_endpoints.infer_causality(
                df=df,
                session_id=request.session_id,
                patterns=results.get("patterns"),
                domain=request.domain
            )
        
        # Format into Findings JSON
        findings = format_findings_json(
            exploration=results.get("exploration"),
            patterns=results.get("patterns"),
            causality=results.get("causality")
        )
        
        # Add request metadata
        findings["metadata"]["session_id"] = request.session_id
        findings["metadata"]["analyses_performed"] = request.agents
        
        return findings
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unified analysis failed: {str(e)}"
        )


@app.post("/analyze")
async def legacy_analyze(request: UnifiedAnalysisRequest):
    """
    LEGACY ENDPOINT - Deprecated
    
    This endpoint is kept for backward compatibility.
    New code should use /agents/unified instead.
    
    Falls back to /agents/unified with all agents enabled.
    """
    # Default to running all agents
    if not request.agents or len(request.agents) == 0:
        request.agents = ["explore", "patterns", "causality"]
    
    return await unified_analysis(request)


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "sessions": [
            {
                "session_id": sid,
                "filename": session.get("filename"),
                "uploaded_at": session.get("uploaded_at"),
                "rows": len(session["dataframe"]),
                "columns": len(session["dataframe"].columns)
            }
            for sid, session in SESSIONS.items()
        ],
        "total": len(SESSIONS)
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session and free memory"""
    if session_id in SESSIONS:
        del SESSIONS[session_id]
        return {
            "status": "deleted",
            "session_id": session_id
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session information"""
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session["dataframe"]
    
    return {
        "session_id": session_id,
        "filename": session["filename"],
        "uploaded_at": session["uploaded_at"],
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

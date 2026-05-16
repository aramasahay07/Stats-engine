from fastapi import APIRouter
from app.routers import agents, datasets, jobs, kb, narrate, pipelines, process_mining, query, spc, stats

api_router = APIRouter()
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(query.router, prefix="/datasets", tags=["query"])
api_router.include_router(stats.router, prefix="/datasets", tags=["stats"])
api_router.include_router(spc.router, prefix="/datasets", tags=["spc"])
api_router.include_router(process_mining.router, prefix="/datasets", tags=["process-mining"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(kb.router, prefix="/kb", tags=["knowledge"])
api_router.include_router(narrate.router, prefix="/narrate", tags=["narrate"])

# ✅ New Agents API
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])

from fastapi import APIRouter
from app.routers import datasets, query, stats, pipelines, jobs, kb, narrate

api_router = APIRouter()
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(pipelines.router, prefix="/pipelines", tags=["pipelines"])
api_router.include_router(query.router, prefix="/datasets", tags=["query"])
api_router.include_router(stats.router, prefix="/datasets", tags=["stats"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(kb.router, prefix="/kb", tags=["knowledge"])
api_router.include_router(narrate.router, prefix="/narrate", tags=["narrate"])

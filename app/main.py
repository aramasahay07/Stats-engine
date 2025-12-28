from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.routers.legacy import router as legacy_router
from app.config import settings
from app.db.registry import close_pool


def create_app() -> FastAPI:
    app = FastAPI(
        title="Stats Engine v2",
        version="2.0.0",
        description="Dataset-first analytics engine (DuckDB-only compute) with Supabase integration",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Primary API (dataset-first)
    app.include_router(api_router)
    # Temporary legacy aliases (session routes that are still referenced by edge/frontend)
    app.include_router(legacy_router)

    @app.on_event("shutdown")
    async def _shutdown():
        # Gracefully close asyncpg pool to avoid dangling connections on redeploy.
        await close_pool()

    @app.get("/health")
    async def health():
        return {"ok": True, "version": "2.0.0"}

    return app


app = create_app()

"""
AI Data Lab Backend v6.0
Router-only FastAPI entrypoint; business logic lives in routers/services.
"""
from dotenv import load_dotenv

# Load environment variables before other imports
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


print("=" * 60)
print("AI Data Lab Backend v6.0 Starting...")
print("=" * 60)

# Import routers with error handling
routers_to_mount = []

def _try_load_router(name: str, import_path: str):
    try:
        module = __import__(import_path, fromlist=["router"])
        routers_to_mount.append((name, module.router))
        print(f"[ok] {name} router loaded")
    except ImportError as e:
        print(f"[warn] Could not load {name} router: {e}")


_try_load_router("datasets", "app.routers.datasets")
_try_load_router("stats", "app.routers.stats")
_try_load_router("quality", "app.routers.quality")
_try_load_router("transforms", "app.routers.transforms")
_try_load_router("agents", "app.routers.agents")

# Optional routers
_try_load_router("query", "app.routers.query")
_try_load_router("kb", "knowledge.routers.kb")

print("=" * 60)

# FastAPI App Setup
app = FastAPI(
    title="AI Data Lab",
    version="6.0.0",
    description="Minitab-level Statistics + Power BI Charts + ChatGPT Narration",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all available routers
for name, router in routers_to_mount:
    try:
        app.include_router(router)
        print(f"[ok] Mounted {name} router")
    except Exception as e:
        print(f"[error] Failed to mount {name} router: {e}")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Data Lab API",
        "version": "6.0.0",
        "status": "running",
        "docs": "/docs",
        "routers_loaded": [name for name, _ in routers_to_mount],
        "description": "Advanced data analysis with AI-powered insights",
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "6.0.0",
        "routers": len(routers_to_mount),
    }


@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("\n" + "=" * 60)
    print("AI Data Lab Backend Started Successfully!")
    print("=" * 60)
    print(f"Loaded {len(routers_to_mount)} routers")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative Docs: http://localhost:8000/redoc")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("\n" + "=" * 60)
    print("AI Data Lab Backend Shutting Down...")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

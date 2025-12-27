from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.dataset_creator import create_dataset_from_upload
from app.services.dataset_registry import DatasetRegistry
from app.models.specs import DatasetCreateResponse

# dY"1 Supabase client (adjust this import to match your project)
# Example: create a module app/services/supabase_client.py that exposes `supabase`
try:
    from app.services.supabase_client import supabase
except ImportError:
    supabase = None  # Fallback: router will still work without Supabase, but metadata won't be stored there

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("", response_model=DatasetCreateResponse)
async def create_dataset(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    project_id: Optional[str] = Form(None),
):
    try:
        _, _, response = create_dataset_from_upload(file, user_id=user_id, project_id=project_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return response


@router.get("/{dataset_id}/profile")
def get_dataset_profile(dataset_id: str, user_id: str):
    """
    Get comprehensive dataset profile

    Returns statistical summary and data quality metrics.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    try:
        profile = compute_service.get_profile(dataset_id, user_id)
        return profile

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile generation failed: {str(e)}")


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------

@router.get("/health", include_in_schema=False)
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "datasets"}


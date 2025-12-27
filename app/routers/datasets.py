from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.dataset_creator import create_dataset_from_upload
from app.services.dataset_registry import DatasetRegistry
from app.models.specs import DatasetCreateResponse, DatasetProfile

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

@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, user_id: str):
    """Rehydrate UI anytime: returns dataset metadata from registry."""
    registry = DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds

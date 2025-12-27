"""
Transform Router - Data transformation endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.services.compute import compute_service

router = APIRouter(prefix="/transforms", tags=["transforms"])


class TransformRequest(BaseModel):
    """Request to apply a transformation"""
    column: str
    transform_type: str
    params: Optional[Dict[str, Any]] = {}
    new_column: Optional[str] = None


class TransformInfo(BaseModel):
    """Information about an available transform"""
    type: str
    name: str
    description: str
    input_types: List[str]
    output_type: str
    parameters: Dict[str, Any]


@router.get("/available", response_model=List[TransformInfo])
def list_available_transforms():
    """
    Get list of all available transformations
    
    Returns comprehensive catalog of transforms with:
    - Input/output types
    - Required parameters
    - Examples
    """
    try:
        from transformers.registry import registry
        
        transforms = []
        for transform_type, transformer_class in registry._registry.items():
            transforms.append({
                "type": transform_type,
                "name": transformer_class.TRANSFORM_TYPE,
                "description": transformer_class.__doc__ or "No description",
                "input_types": transformer_class.SUPPORTED_INPUT_TYPES,
                "output_type": transformer_class.OUTPUT_TYPE,
                "parameters": {}  # Could be enhanced with param specs
            })
        
        return transforms
    except Exception as e:
        # If transforms not available, return empty list
        return []


@router.post("/datasets/{dataset_id}/transform")
def apply_transform(
    dataset_id: str,
    user_id: str,
    request: TransformRequest
):
    """
    Apply transformation to create new dataset version
    
    Creates immutable transformation - original dataset unchanged.
    Returns new dataset_id with transformed data.
    
    Example:
    ```json
    {
      "column": "date",
      "transform_type": "month",
      "params": {"format": "name"},
      "new_column": "month_name"
    }
    ```
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id query parameter")
    
    try:
        # Build params dict
        params = {
            "column": request.column,
            **(request.params or {})
        }
        
        if request.new_column:
            params["new_column"] = request.new_column
        
        # Apply transformation
        result = compute_service.apply_transform(
            dataset_id=dataset_id,
            user_id=user_id,
            transform_type=request.transform_type,
            params=params
        )
        
        return {
            "success": True,
            "new_dataset_id": result["dataset_id"],
            "parent_dataset_id": dataset_id,
            "transform_applied": request.transform_type,
            "metadata": result
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transform failed: {str(e)}")


@router.get("/datasets/{dataset_id}/suggest")
def suggest_transforms(dataset_id: str, user_id: str, column: str):
    """
    Suggest appropriate transformations for a column
    
    Analyzes column type and patterns to recommend transforms.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    try:
        # Load dataset to analyze column
        df = compute_service.load_dataframe(dataset_id, user_id)
        
        if column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
        
        series = df[column]
        
        # Infer column type
        from utils.type_inference import infer_column_role, suggest_column_transforms
        
        role = infer_column_role(series)
        suggestions = suggest_column_transforms(series, role)
        
        return {
            "column": column,
            "detected_type": role,
            "suggested_transforms": suggestions
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/lineage")
def get_lineage(dataset_id: str, user_id: str):
    """
    Get transformation lineage for a dataset
    
    Shows the chain of transformations that created this dataset.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing user_id")
    
    from app.services.dataset_registry import DatasetRegistry
    
    registry = DatasetRegistry()
    lineage = []
    current_id = dataset_id
    
    # Walk back through parent chain
    while current_id:
        ds = registry.get(current_id, user_id)
        if not ds:
            break
        
        lineage.append({
            "dataset_id": current_id,
            "name": ds.get("name"),
            "created_at": ds.get("created_at"),
            "n_rows": ds.get("n_rows"),
            "n_cols": ds.get("n_cols")
        })
        
        current_id = ds.get("parent_dataset_id")
    
    return {
        "dataset_id": dataset_id,
        "lineage_chain": lineage,
        "depth": len(lineage)
    }

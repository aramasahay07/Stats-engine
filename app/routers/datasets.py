"""
Datasets Router - File upload and dataset management
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

# Local compute/caching service
from app.services.compute import compute_service
from app.services.dataset_registry import DatasetRegistry

# dY"1 Supabase client (adjust this import to match your project)
# Example: create a module app/services/supabase_client.py that exposes `supabase`
try:
    from app.services.supabase_client import supabase
except ImportError:
    supabase = None  # Fallback: router will still work without Supabase, but metadata won't be stored there

router = APIRouter(prefix="/datasets", tags=["datasets"])


class ColumnInfo(BaseModel):
    """Information about a column"""
    name: str
    dtype: str
    role: str  # "numeric", "categorical", "datetime", "text"
    missing_pct: float


class DatasetMetadata(BaseModel):
    """Dataset metadata response"""
    dataset_id: str
    user_id: str
    name: str
    n_rows: int
    n_cols: int
    columns: List[str]
    created_at: str
    parent_dataset_id: Optional[str] = None


class DatasetProfile(BaseModel):
    """Complete dataset profile"""
    dataset_id: str
    name: str
    n_rows: int
    n_cols: int
    columns: List[ColumnInfo]
    schema: List[ColumnInfo]  # Alias for frontend compatibility
    created_at: str


# ---------------------------------------------------
# Dataset Upload & Management
# ---------------------------------------------------

@router.post("/upload", response_model=DatasetMetadata)
async def upload_dataset(
    user_id: str,
    file: UploadFile = File(...)
):
    """
    Upload CSV or Excel file to create new dataset

    Supported formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)

    Returns dataset_id for use in subsequent API calls.

    This version:
    - Parses file with pandas
    - Saves dataframe via compute_service (local cache / parquet)
    - Inserts a row into Supabase `datasets` table using the correct schema
      (no `columns` column; uses `schema_json` instead).
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    # Validate file type
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload CSV or Excel file."
        )

    try:
        # 1Л,?ГЯЬ Read file content
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Parse file into DataFrame
        from io import BytesIO
        buffer = BytesIO(content)

        if filename.endswith('.csv'):
            df = pd.read_csv(buffer)
        else:
            df = pd.read_excel(buffer)

        if df.empty:
            raise HTTPException(status_code=400, detail="File contains no data")

        # Clean DataFrame - remove completely empty columns
        df = df.dropna(axis=1, how='all')

        # Basic metadata
        column_list = list(df.columns)
        created_at = datetime.utcnow().isoformat()

        # 2Л,?ГЯЬ Save dataset using compute service (local parquet / cache)
        #    We pass explicit name so downstream functions continue to work.
        metadata = compute_service.save_dataframe(
            df=df,
            user_id=user_id,
            name=file.filename
        )
        dataset_id = metadata["dataset_id"]
        n_rows = int(metadata["n_rows"])
        n_cols = int(metadata["n_cols"])

        # 3Л,?ГЯЬ Also store metadata in Supabase `datasets` table, if client is available
        #    IMPORTANT: We do NOT send a `columns` field because the table
        #    does not have that column. We store columns inside `schema_json` instead.
        if supabase is not None:
            try:
                raw_file_ref = f"{user_id}/{dataset_id}/{file.filename}"
                # Prefer compute_service parquet ref if available
                parquet_ref = metadata.get("parquet_ref") or f"{user_id}/{dataset_id}/{Path(file.filename).stem}.parquet"

                supabase_payload = {
                    "dataset_id": dataset_id,
                    "user_id": user_id,
                    "file_name": file.filename,
                    "raw_file_ref": raw_file_ref,
                    "parquet_ref": parquet_ref,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "schema_json": {"columns": column_list},
                    # Other JSON fields (profile_json, semantic_profile_json, column_semantics_json)
                    # can be filled in later by your profiling pipeline.
                }

                supabase.table("datasets").insert(supabase_payload).execute()
            except Exception as db_err:
                # Log but don't fail the upload for Supabase metadata issues
                # If you have a logger, replace print with logger.warning(...)
                print(f"[datasets/upload] Warning: failed to insert Supabase metadata: {db_err}")

        # 4Л,?ГЯЬ Return API metadata shaped for the frontend
        return DatasetMetadata(
            dataset_id=dataset_id,
            user_id=user_id,
            name=file.filename,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=column_list,
            created_at=created_at,
            parent_dataset_id=None
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="File contains no data")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Generic failure
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/list")
def list_datasets(user_id: str, limit: int = 50):
    """
    List all datasets for a user

    Returns most recent datasets first.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    registry = DatasetRegistry()
    datasets = registry.list_by_user(user_id, limit=limit)

    return {
        "user_id": user_id,
        "count": len(datasets),
        "datasets": datasets
    }


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, user_id: str):
    """
    Get dataset metadata

    Returns information about the dataset without loading the data.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    registry = DatasetRegistry()
    dataset = registry.get(dataset_id, user_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str, user_id: str):
    """
    Delete a dataset

    Removes dataset metadata and associated files.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    registry = DatasetRegistry()
    success = registry.delete(dataset_id, user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Also delete parquet file if exists
    try:
        parquet_path = compute_service.cache.parquet_path(user_id, dataset_id)
        if parquet_path.exists():
            parquet_path.unlink()
    except Exception as e:
        print(f"Warning: Failed to delete parquet file: {e}")

    return {
        "success": True,
        "dataset_id": dataset_id,
        "message": "Dataset deleted successfully"
    }


# ---------------------------------------------------
# NEW: Additional Dataset Operations (from v6.0)
# ---------------------------------------------------

@router.get("/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, user_id: str, n_rows: int = 10):
    """
    Get preview of dataset (first N rows)

    Args:
        dataset_id: Dataset identifier
        user_id: User identifier
        n_rows: Number of rows to return (default: 10, max: 100)

    Returns:
        First N rows as JSON records
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    if n_rows > 100:
        n_rows = 100

    try:
        df = compute_service.load_dataframe(dataset_id, user_id)
        preview = df.head(n_rows)

        return {
            "dataset_id": dataset_id,
            "n_rows_returned": len(preview),
            "total_rows": len(df),
            "data": preview.to_dict(orient="records")
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/{dataset_id}/schema", response_model=DatasetProfile)
def get_dataset_schema(dataset_id: str, user_id: str):
    """
    Get detailed schema information for dataset

    Returns column types, roles, and missing data percentages.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    try:
        # Get basic metadata
        registry = DatasetRegistry()
        metadata = registry.get(dataset_id, user_id)

        if not metadata:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Load data to analyze schema
        df = compute_service.load_dataframe(dataset_id, user_id)

        # Analyze each column
        from utils.type_inference import infer_column_role

        columns_info = []
        for col in df.columns:
            series = df[col]
            role = infer_column_role(series)
            missing_pct = float(series.isna().mean() * 100)

            columns_info.append(ColumnInfo(
                name=col,
                dtype=str(series.dtype),
                role=role,
                missing_pct=round(missing_pct, 2)
            ))

        return DatasetProfile(
            dataset_id=dataset_id,
            name=metadata.get("name", "Unknown"),
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=columns_info,
            schema=columns_info,  # Duplicate for compatibility
            created_at=metadata.get("created_at", datetime.utcnow().isoformat())
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema analysis failed: {str(e)}")


@router.get("/{dataset_id}/export")
def export_dataset(
    dataset_id: str,
    user_id: str,
    format: str = "csv"
):
    """
    Export dataset to different format

    Supported formats:
    - csv: Comma-separated values
    - xlsx: Excel spreadsheet
    - json: JSON format

    Returns download URL or file path.
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id is required")

    if format not in ["csv", "xlsx", "json"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Use csv, xlsx, or json"
        )

    try:
        export_path = compute_service.export_dataset(dataset_id, user_id, format)

        return {
            "dataset_id": dataset_id,
            "format": format,
            "file_path": str(export_path),
            "message": f"Dataset exported to {format.upper()}"
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


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


from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: str
    missing_pct: float = 0.0
    unique_count: Optional[int] = None


class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    schema: List[ColumnProfile] = Field(default_factory=list)

    # Small sample for UI preview
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)

    # Optional descriptive outputs (concept-driven stats, etc.)
    descriptives: Optional[Dict[str, Any]] = None


class DatasetCreateResponse(BaseModel):
    dataset_id: str
    profile: DatasetProfile
    job_id: Optional[str] = None  # parquet/profile job (if async)


class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    file_name: str
    n_rows: int
    n_cols: int
    schema_json: Any
    profile_json: Any
    raw_file_ref: str
    parquet_ref: Optional[str] = None
    user_id: str
    project_id: Optional[str] = None

    # processing state
    state: str = "ready"
    version: int = 1

    # convenience fields (not always used by all endpoints/clients)
    ready: Optional[bool] = None
    error_message: Optional[str] = None

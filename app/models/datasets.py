from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: str
    missing_pct: float = 0.0
    unique_count: Optional[int] = None

class DatasetProfile(BaseModel):
    n_rows: int
    n_cols: int
    schema: List[ColumnProfile]
    sample_rows: List[Dict[str, Any]] = Field(default_factory=list)
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

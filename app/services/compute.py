"""
Compute Service - Handles all data processing operations
Supports both Supabase (cloud) and local-only mode
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import duckdb
from datetime import datetime

from app.services.storage_client import SupabaseStorageClient
from app.services.dataset_registry import DatasetRegistry
from app.services.cache_paths import CachePaths
from app.services.duckdb_manager import DuckDBManager


class ComputeService:
    """
    Central compute service for data transformations and analysis
    
    Features:
    - Optional Supabase integration (works locally without it)
    - DuckDB-based computations
    - Parquet file caching
    - Dataset lineage tracking
    """
    
    def __init__(self, base_cache_dir: Path = Path("./cache")):
        """Initialize compute service with optional Supabase"""
        self.cache = CachePaths(base_dir=base_cache_dir)
        self.registry = DatasetRegistry()
        
        # Make Supabase optional for local development
        try:
            self.storage = SupabaseStorageClient()
            print("✅ Supabase storage connected successfully")
        except RuntimeError as e:
            print(f"⚠️  WARNING: Running in LOCAL MODE (no Supabase)")
            print(f"   Reason: {str(e)}")
            print(f"   Data will be stored in ./cache only")
            self.storage = None
    
    def _ensure_parquet_local(self, dataset_id: str, user_id: str) -> Path:
        """
        Ensure parquet file is available locally
        Downloads from Supabase if available and not cached
        """
        parquet_path = self.cache.parquet_path(user_id, dataset_id)
        
        # If already cached locally, return it
        if parquet_path.exists():
            return parquet_path
        
        # Try to download from Supabase if available
        if self.storage:
            ds = self.registry.get(dataset_id, user_id)
            if ds and ds.get("parquet_ref"):
                try:
                    self.storage.download_file(ds["parquet_ref"], parquet_path)
                    print(f"✅ Downloaded {dataset_id} from Supabase")
                    return parquet_path
                except Exception as e:
                    print(f"⚠️  Failed to download from Supabase: {e}")
        
        # File not found anywhere
        raise FileNotFoundError(
            f"Parquet file not found for dataset {dataset_id}. "
            f"{'Supabase is not configured. ' if not self.storage else ''}"
            f"Upload a file first or check your dataset_id."
        )
    
    def load_dataframe(self, dataset_id: str, user_id: str) -> pd.DataFrame:
        """Load dataset as pandas DataFrame"""
        parquet_path = self._ensure_parquet_local(dataset_id, user_id)
        return pd.read_parquet(parquet_path)
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        user_id: str,
        name: str,
        parent_dataset_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save DataFrame as new dataset
        
        Args:
            df: DataFrame to save
            user_id: User identifier
            name: Dataset name
            parent_dataset_id: Optional parent for lineage tracking
            dataset_id: Optional pre-assigned id (e.g., from router)
            
        Returns:
            Dataset metadata with new dataset_id
        """
        # Generate new dataset_id if not provided
        from uuid import uuid4
        dataset_id = dataset_id or str(uuid4())
        
        # Save parquet locally
        parquet_path = self.cache.parquet_path(user_id, dataset_id)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        
        # Upload to Supabase if available
        parquet_ref = None
        if self.storage:
            try:
                bucket_path = f"{user_id}/{dataset_id}.parquet"
                self.storage.upload_file(parquet_path, bucket_path)
                parquet_ref = bucket_path
                print(f"✅ Uploaded {dataset_id} to Supabase")
            except Exception as e:
                print(f"⚠️  Failed to upload to Supabase: {e}")
                print(f"   Dataset saved locally only")
        
        # Register dataset
        metadata = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "name": name,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            # Store schema in schema_json to match Supabase table shape
            "schema_json": {"columns": df.columns.tolist()},
            "parquet_ref": parquet_ref,
            "parent_dataset_id": parent_dataset_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        self.registry.create(metadata)
        
        return metadata
    
    def run_duckdb_query(
        self, 
        dataset_id: str, 
        user_id: str, 
        query: str
    ) -> pd.DataFrame:
        """
        Execute DuckDB query on dataset
        
        Args:
            dataset_id: Dataset to query
            user_id: User identifier
            query: SQL query (use 'dataset' as table name)
            
        Returns:
            Query results as DataFrame
        """
        parquet_path = self._ensure_parquet_local(dataset_id, user_id)
        
        duck = DuckDBManager()
        with duck.connect() as con:
            path_sql = str(parquet_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW dataset AS SELECT * FROM read_parquet('{path_sql}')")
            result = con.execute(query).fetchdf()
        
        return result
    
    def apply_transform(
        self,
        dataset_id: str,
        user_id: str,
        transform_type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply transformation and create new dataset version
        
        Args:
            dataset_id: Source dataset
            user_id: User identifier
            transform_type: Type of transformation
            params: Transform parameters
            
        Returns:
            New dataset metadata
        """
        # Load source data
        df = self.load_dataframe(dataset_id, user_id)
        
        # Import transform registry
        from transformers.registry import registry
        
        # Get transformer
        transformer_class = registry.get(transform_type)
        if not transformer_class:
            raise ValueError(f"Unknown transform: {transform_type}")
        
        # Apply transform
        transformer = transformer_class(params)
        column = params.get("column")
        new_column = params.get("new_column", f"{column}_{transform_type}")
        
        df[new_column] = transformer.apply(df[column], column)
        
        # Save as new dataset
        original_name = self.registry.get(dataset_id, user_id).get("name", "dataset")
        new_name = f"{original_name} ({transform_type})"
        
        return self.save_dataframe(
            df=df,
            user_id=user_id,
            name=new_name,
            parent_dataset_id=dataset_id
        )
    
    def get_profile(self, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """Get dataset profile/summary statistics"""
        parquet_path = self._ensure_parquet_local(dataset_id, user_id)
        
        duck = DuckDBManager()
        profile = duck.profile_parquet(parquet_path, sample_n=1000)
        
        return profile
    
    def export_dataset(
        self, 
        dataset_id: str, 
        user_id: str, 
        format: str = "csv"
    ) -> Path:
        """
        Export dataset to different format
        
        Args:
            dataset_id: Dataset to export
            user_id: User identifier
            format: Export format (csv, xlsx, json)
            
        Returns:
            Path to exported file
        """
        df = self.load_dataframe(dataset_id, user_id)
        
        export_path = self.cache.base_dir / "exports" / user_id / f"{dataset_id}.{format}"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            df.to_csv(export_path, index=False)
        elif format == "xlsx":
            df.to_excel(export_path, index=False)
        elif format == "json":
            df.to_json(export_path, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return export_path


# Global instance
compute_service = ComputeService()

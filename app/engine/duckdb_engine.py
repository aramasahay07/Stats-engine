from __future__ import annotations
from pathlib import Path
import duckdb

from app.config import settings

class DuckDBEngine:
    """One DuckDB file per user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_path = Path(settings.data_dir) / "duckdb" / f"{user_id}.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(self.db_path))
        con.execute("PRAGMA threads=4")  # tune later
        return con

    def register_parquet(self, con: duckdb.DuckDBPyConnection, dataset_id: str, parquet_local_path: Path):
        # Create/replace a view pointing to parquet
        view = self.base_view_name(dataset_id)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{parquet_local_path.as_posix()}')")
        return view

    @staticmethod
    def base_view_name(dataset_id: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_base"

    @staticmethod
    def pipeline_view_name(dataset_id: str, pipeline_hash: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_p_{pipeline_hash[:16]}"

from __future__ import annotations

from pathlib import Path
import duckdb

from app.config import settings


class DuckDBUnsupportedTypeError(RuntimeError):
    """Raised when DuckDB cannot read a parquet file due to unsupported types."""


class DuckDBEngine:
    """One DuckDB file per user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_path = Path(settings.data_dir) / "duckdb" / f"{user_id}.duckdb"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(str(self.db_path))
        return con

    def register_parquet(
        self,
        con: duckdb.DuckDBPyConnection,
        dataset_id: str,
        parquet_local_path: Path
    ) -> str:
        """
        Defensive registration:
        - Try normal view: SELECT * FROM read_parquet(...)
        - If DuckDB fails due to unsupported parquet types (e.g., TIME WITH TIME ZONE),
          fall back to a defensive view that TRY_CASTs risky columns to VARCHAR.
        - If that still fails, raise DuckDBUnsupportedTypeError for a clean 422 upstream.
        """
        view = self.base_view_name(dataset_id)

        # Escape single quotes for safe SQL strings
        p = parquet_local_path.as_posix().replace("'", "''")

        # 1) Fast path
        try:
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{p}')"
            )
            return view
        except Exception as e1:
            first_err = e1

        # 2) Defensive fallback: inspect schema and cast risky types
        try:
            cols = con.execute(
                f"SELECT name, type FROM parquet_schema('{p}')"
            ).fetchall()

            def qident(name: str) -> str:
                return '"' + name.replace('"', '""') + '"'

            select_exprs: list[str] = []
            for name, typ in cols:
                col = qident(name)
                t = str(typ).upper()

                if ("TIME" in t and "ZONE" in t) or t.startswith("TIME") or ("TIMESTAMP" in t and "ZONE" in t):
                    select_exprs.append(f"TRY_CAST({col} AS VARCHAR) AS {col}")
                else:
                    select_exprs.append(f"{col}")

            select_sql = ", ".join(select_exprs)

            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT {select_sql} FROM read_parquet('{p}')"
            )
            return view

        except Exception as e2:
            # Raise a typed error so routers can return a structured 422
            msg = (
                "DuckDB cannot read this dataset due to unsupported Parquet types "
                "(commonly TIME/TIMESTAMP with timezone). "
                "Re-upload after normalizing time/tz columns to text, or repair the dataset."
            )
            raise DuckDBUnsupportedTypeError(f"{msg} | primary_error={first_err} | fallback_error={e2}") from e2

    @staticmethod
    def base_view_name(dataset_id: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_base"

    @staticmethod
    def pipeline_view_name(dataset_id: str, pipeline_hash: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_p_{pipeline_hash[:16]}"

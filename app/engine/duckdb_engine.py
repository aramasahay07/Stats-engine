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
        parquet_local_path: Path,
    ) -> str:
        """
        Defensive registration:

        - Always creates a view over the parquet.
        - Proactively inspects parquet schema and TRY_CASTs tz-aware TIME/TIMESTAMP
          columns to VARCHAR so profiling/queries do not crash.
        - If DuckDB still cannot read the parquet at all, raises DuckDBUnsupportedTypeError.
        """
        view = self.base_view_name(dataset_id)

        # Escape single quotes for safe SQL strings
        p = parquet_local_path.as_posix().replace("'", "''")

        def qident(name: str) -> str:
            # Quote identifiers safely for DuckDB
            return '"' + name.replace('"', '""') + '"'

        # 1) Fast path: create view, but proactively cast risky tz-aware types if present
        try:
            # Create base view
            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT * FROM read_parquet('{p}')"
            )

            # Inspect schema
            cols = con.execute(
                f"SELECT name, type FROM parquet_schema('{p}')"
            ).fetchall()

            select_exprs: list[str] = []
            has_risky = False

            for name, typ in cols:
                col = qident(name)
                t = str(typ).upper()

                # Narrow to tz-aware time/timestamp types only
                risky = ("TIME WITH TIME ZONE" in t) or ("TIMESTAMP WITH TIME ZONE" in t)

                if risky:
                    has_risky = True
                    select_exprs.append(f"TRY_CAST({col} AS VARCHAR) AS {col}")
                else:
                    select_exprs.append(f"{col}")

            # If any risky cols exist, overwrite view with safe casts
            if has_risky:
                select_sql = ", ".join(select_exprs)
                con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT {select_sql} FROM read_parquet('{p}')"
                )

            # Stable alias used by concept SQL (FROM dataset)
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM {view}")
            return view

        except Exception as e1:
            first_err = e1

        # 2) Defensive fallback: build a fully casted view (only tz-aware TIME/TIMESTAMP)
        try:
            cols = con.execute(
                f"SELECT name, type FROM parquet_schema('{p}')"
            ).fetchall()

            select_exprs: list[str] = []
            for name, typ in cols:
                col = qident(name)
                t = str(typ).upper()

                if ("TIME WITH TIME ZONE" in t) or ("TIMESTAMP WITH TIME ZONE" in t):
                    select_exprs.append(f"TRY_CAST({col} AS VARCHAR) AS {col}")
                else:
                    select_exprs.append(f"{col}")

            select_sql = ", ".join(select_exprs)

            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT {select_sql} FROM read_parquet('{p}')"
            )

            # Stable alias used by concept SQL (FROM dataset)
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM {view}")
            return view

        except Exception as e2:
            # If we get here, DuckDB couldn't read the parquet even with casts.
            msg = (
                "DuckDB cannot read this dataset due to unsupported Parquet types "
                "(commonly nested/complex types or tz-aware time types). "
                "Normalize unsupported columns to text and re-upload, or repair the dataset."
            )
            raise DuckDBUnsupportedTypeError(
                f"{msg} | primary_error={first_err} | fallback_error={e2}"
            ) from e2

    @staticmethod
    def base_view_name(dataset_id: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_base"

    @staticmethod
    def pipeline_view_name(dataset_id: str, pipeline_hash: str) -> str:
        return f"ds_{dataset_id.replace('-', '_')}_p_{pipeline_hash[:16]}"

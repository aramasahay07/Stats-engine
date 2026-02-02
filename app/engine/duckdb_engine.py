from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

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

    # -------------------------------------------------------------------------
    # Backwards compatible API:
    # - register_parquet(...) keeps returning str (view name) to avoid breaking
    #   existing callers.
    # - register_parquet_with_issues(...) returns (view_name, detected_issues)
    #   so datasets_service can persist profile_json.issues (what Lovable wants).
    # -------------------------------------------------------------------------

    def register_parquet(
        self,
        con: duckdb.DuckDBPyConnection,
        dataset_id: str,
        parquet_local_path: Path,
    ) -> str:
        """
        Backward-compatible wrapper.
        Returns only the view name (string), same behavior as before.
        """
        view, _issues = self.register_parquet_with_issues(con, dataset_id, parquet_local_path)
        return view

    def register_parquet_with_issues(
        self,
        con: duckdb.DuckDBPyConnection,
        dataset_id: str,
        parquet_local_path: Path,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Defensive registration + issue detection.

        Returns: (view_name, detected_issues)

        - Always creates a view over the parquet.
        - Proactively inspects parquet schema and TRY_CASTs risky TIME/TIMESTAMP
          columns to VARCHAR so profiling/queries do not crash.
        - If DuckDB still cannot read the parquet at all, raises DuckDBUnsupportedTypeError.

        detected_issues contains one item per column that required fallback casting.
        """
        return self._register_parquet_internal(con, dataset_id, parquet_local_path)

    def _register_parquet_internal(
        self,
        con: duckdb.DuckDBPyConnection,
        dataset_id: str,
        parquet_local_path: Path,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        view = self.base_view_name(dataset_id)

        # Escape single quotes for safe SQL strings
        p = parquet_local_path.as_posix().replace("'", "''")

        def qident(name: str) -> str:
            # Quote identifiers safely for DuckDB
            return '"' + name.replace('"', '""') + '"'

        def is_problematic_type(t_upper: str) -> bool:
            """
            Match Lovable's expectation:
            - TIME WITH TIME ZONE
            - any TIME* (some parquet logical TIME types can be problematic in DuckDB)
            - TIMESTAMP WITH TIME ZONE
            """
            return (
                ("TIME" in t_upper and "ZONE" in t_upper)
                or t_upper.startswith("TIME")
                or ("TIMESTAMP" in t_upper and "ZONE" in t_upper)
            )

        detected_issues: List[Dict[str, Any]] = []

        # 1) Fast path: create view, but proactively cast risky types if present
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
                t_upper = str(typ).upper()

                risky = is_problematic_type(t_upper)

                if risky:
                    has_risky = True
                    select_exprs.append(f"TRY_CAST({col} AS VARCHAR) AS {col}")
                    detected_issues.append(
                        {
                            "column": name,
                            "issue_type": "unsupported_type",
                            "severity": "blocking",
                            "details": {"duckdb_type": t_upper},
                            "suggested_fix": {"op": "change_type", "to": "varchar"},
                        }
                    )
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
            return view, detected_issues

        except Exception as e1:
            first_err = e1

        # 2) Defensive fallback: build a fully casted view for problematic types
        try:
            cols = con.execute(
                f"SELECT name, type FROM parquet_schema('{p}')"
            ).fetchall()

            select_exprs: list[str] = []
            for name, typ in cols:
                col = qident(name)
                t_upper = str(typ).upper()

                risky = is_problematic_type(t_upper)

                if risky:
                    select_exprs.append(f"TRY_CAST({col} AS VARCHAR) AS {col}")
                    detected_issues.append(
                        {
                            "column": name,
                            "issue_type": "unsupported_type",
                            "severity": "blocking",
                            "details": {"duckdb_type": t_upper},
                            "suggested_fix": {"op": "change_type", "to": "varchar"},
                        }
                    )
                else:
                    select_exprs.append(f"{col}")

            select_sql = ", ".join(select_exprs)

            con.execute(
                f"CREATE OR REPLACE VIEW {view} AS "
                f"SELECT {select_sql} FROM read_parquet('{p}')"
            )

            # Stable alias used by concept SQL (FROM dataset)
            con.execute(f"CREATE OR REPLACE VIEW dataset AS SELECT * FROM {view}")
            return view, detected_issues

        except Exception as e2:
            # If we get here, DuckDB couldn't read the parquet even with casts.
            msg = (
                "DuckDB cannot read this dataset due to unsupported Parquet types. "
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

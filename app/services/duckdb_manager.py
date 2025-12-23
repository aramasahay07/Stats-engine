from __future__ import annotations

import duckdb
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models.specs import FilterSpec, QuerySpec


def _qident(name: str) -> str:
    """Safely quote identifiers for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


class DuckDBManager:
    """DuckDB native manager (ephemeral in-memory connection by default).

    Reads Parquet from a local path and runs SQL safely.
    """

    def __init__(self, memory_limit: str = "1GB", threads: int = 4):
        self.memory_limit = memory_limit
        self.threads = threads

    @contextmanager
    def connect(self):
        con = duckdb.connect(database=":memory:", read_only=False)
        try:
            con.execute(f"SET memory_limit='{self.memory_limit}'")
            con.execute(f"SET threads={int(self.threads)}")
            yield con
        finally:
            con.close()

    def _apply_filters_sql(self, filters: List[FilterSpec], params: List[Any]) -> str:
        clauses: List[str] = []
        for f in filters:
            col = _qident(f.col)
            op = f.op.lower()

            if op in ("is null", "is not null"):
                clauses.append(f"{col} {op.upper()}")
                continue

            if op in ("in", "not in"):
                if not isinstance(f.value, list):
                    raise ValueError(f"Filter '{f.col}' with op '{f.op}' requires list value")
                placeholders = ", ".join(["?"] * len(f.value))
                params.extend(f.value)
                clauses.append(f"{col} {op.upper()} ({placeholders})")
                continue

            if op == "contains":
                params.append(f"%{f.value}%")
                clauses.append(f"{col} ILIKE ?")
                continue

            if op == "startswith":
                params.append(f"{f.value}%")
                clauses.append(f"{col} ILIKE ?")
                continue

            if op == "endswith":
                params.append(f"%{f.value}")
                clauses.append(f"{col} ILIKE ?")
                continue

            # default binary ops
            params.append(f.value)
            clauses.append(f"{col} {f.op} ?")

        return " AND ".join(clauses)

    def build_query_sql(self, view_name: str, spec: QuerySpec) -> tuple[str, List[Any]]:
        params: List[Any] = []
        select_parts: List[str] = []
        groupby_parts: List[str] = []

        # dimensions
        for c in spec.select:
            select_parts.append(_qident(c))

        # measures (expr is assumed to be produced by your app, not free-form user SQL)
        for m in spec.measures:
            select_parts.append(f"{m.expr} AS {_qident(m.name)}")

        if not select_parts:
            select_parts = ["*"]

        # groupby
        for g in (spec.groupby or spec.select):
            groupby_parts.append(_qident(g))

        where_sql = ""
        if spec.filters:
            where_clause = self._apply_filters_sql(spec.filters, params)
            if where_clause:
                where_sql = f" WHERE {where_clause}"

        groupby_sql = ""
        if spec.measures and groupby_parts:
            groupby_sql = " GROUP BY " + ", ".join(groupby_parts)

        order_sql = ""
        if spec.order_by:
            order_sql = " ORDER BY " + ", ".join(
                f"{_qident(o.col)} {o.dir.upper()}" for o in spec.order_by
            )

        limit_sql = f" LIMIT {int(spec.limit)}"
        offset_sql = f" OFFSET {int(spec.offset)}" if spec.offset else ""

        sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM {view_name}{where_sql}{groupby_sql}{order_sql}{limit_sql}{offset_sql}"
        )
        return sql, params

    def query_parquet(self, parquet_path: Path, sql: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        with self.connect() as con:
            con.execute("PRAGMA enable_object_cache")
            con.execute("PRAGMA threads=" + str(int(self.threads)))

            # DuckDB doesn't allow prepared parameters in CREATE VIEW ... read_parquet(?)
            path_sql = str(parquet_path).replace("'", "''")
            con.execute("DROP VIEW IF EXISTS ds")
            con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

            rel = con.execute(sql, params or [])
            cols = [c[0] for c in rel.description]
            rows = rel.fetchall()
            return {"columns": cols, "rows": rows}

    def profile_parquet(self, parquet_path: Path, sample_n: int = 100) -> Dict[str, Any]:
        with self.connect() as con:
            path_sql = str(parquet_path).replace("'", "''")
            con.execute("DROP VIEW IF EXISTS ds")
            con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

            n_rows = con.execute("SELECT COUNT(*) FROM ds").fetchone()[0]
            schema = con.execute("DESCRIBE ds").fetchall()

            # missing summary
            cols = [r[0] for r in schema]
            missing: Dict[str, Any] = {}
            for c in cols:
                q = f"SELECT SUM(CASE WHEN {_qident(c)} IS NULL THEN 1 ELSE 0 END) FROM ds"
                missing[c] = con.execute(q).fetchone()[0]

            sample = (
                con.execute(f"SELECT * FROM ds LIMIT {int(sample_n)}")
                .fetchdf()
                .to_dict(orient="records")
            )

            return {
                "n_rows": int(n_rows),
                "schema": [
                    {"name": r[0], "type": r[1], "null": r[2], "key": r[3], "default": r[4], "extra": r[5]}
                    for r in schema
                ],
                "missing_summary": missing,
                "sample_rows": sample,
            }

"""
Dataset Profiling Module

Builds an authoritative dataset profile from DuckDB, including:
- Schema + semantic roles
- Missingness
- Numeric summaries (single source of truth)
- Sample rows

This module is the canonical producer of `profile_json`
used by stats_service.py and snapshot enforcement.
"""

from __future__ import annotations
import json
from typing import Any, Dict, List


# -----------------------------------------------------------------------------
# Semantic role inference
# -----------------------------------------------------------------------------
def infer_role(dtype: str) -> str:
    dtype_lower = dtype.lower()

    if any(k in dtype_lower for k in ["int", "float", "double", "decimal", "numeric"]):
        return "numeric"

    if any(k in dtype_lower for k in ["date", "time", "timestamp"]):
        return "datetime"

    return "categorical"


# -----------------------------------------------------------------------------
# Adaptive sampling strategy (for missingness only)
# -----------------------------------------------------------------------------
def _determine_sample_strategy(n_rows: int) -> tuple[int, bool]:
    if n_rows <= 100_000:
        return n_rows, False
    elif n_rows <= 1_000_000:
        return max(10_000, int(n_rows * 0.05)), True
    else:
        return 50_000, True


# -----------------------------------------------------------------------------
# Main profiling entry point
# -----------------------------------------------------------------------------
def build_profile_from_duckdb(
    con,
    view: str,
    sample_limit: int = 50,
) -> Dict[str, Any]:
    # =========================================================================
    # STEP 1: Discover schema
    # =========================================================================
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM {view}").fetchall()

    schema: List[Dict[str, Any]] = []
    numeric_columns: List[str] = []

    for name, dtype, *_ in schema_rows:
        role = infer_role(str(dtype))
        schema.append(
            {
                "name": name,
                "dtype": str(dtype),
                "role": role,
                "missing_pct": 0.0,
            }
        )
        if role == "numeric":
            numeric_columns.append(name)

    # =========================================================================
    # STEP 2: Dataset dimensions
    # =========================================================================
    n_rows = int(con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0])
    n_cols = len(schema)

    # =========================================================================
    # STEP 3: Sampling strategy (missingness only)
    # =========================================================================
    sample_size, use_sampling = _determine_sample_strategy(n_rows)

    if use_sampling:
        sample_view = f"(SELECT * FROM {view} USING SAMPLE {sample_size} ROWS)"
    else:
        sample_view = view

    # =========================================================================
    # STEP 4: Missingness (per column)
    # =========================================================================
    if n_rows > 0:
        for col in schema:
            col_name = col["name"]
            try:
                val = con.execute(
                    f"""
                    SELECT AVG(
                        CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END
                    )::DOUBLE
                    FROM {sample_view}
                    """
                ).fetchone()[0]
                col["missing_pct"] = round(float(val or 0.0), 4)
            except Exception:
                col["missing_pct"] = 0.0

    # =========================================================================
    # STEP 5: Numeric summaries (AUTHORITATIVE TRUTH)
    # =========================================================================
    numeric_summary: Dict[str, Dict[str, Any]] = {}

    for col in numeric_columns:
        try:
            row = con.execute(
                f"""
                SELECT
                    COUNT("{col}")              AS n_non_null,
                    AVG("{col}")                AS mean,
                    STDDEV_SAMP("{col}")         AS std,
                    MIN("{col}")                AS min,
                    MAX("{col}")                AS max
                FROM {view}
                """
            ).fetchone()

            numeric_summary[col] = {
                "n_non_null": int(row[0]),
                "mean": float(row[1]) if row[1] is not None else None,
                "std": float(row[2]) if row[2] is not None else None,
                "min": float(row[3]) if row[3] is not None else None,
                "max": float(row[4]) if row[4] is not None else None,
            }

        except Exception:
            # Defensive fallback — never break profiling
            numeric_summary[col] = {
                "n_non_null": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

    # =========================================================================
    # STEP 6: Sample rows (preview only)
    # =========================================================================
    rows_df = con.execute(
        f"SELECT * FROM {view} LIMIT {sample_limit}"
    ).fetchdf()
    # ✅ JSON-safe sample rows (timestamps -> ISO strings)
    sample_rows = json.loads(rows_df.to_json(orient="records", date_format="iso"))
    

    # =========================================================================
    # STEP 7: Return authoritative profile
    # =========================================================================
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "schema": schema,
        "numeric_summary": numeric_summary,
        "sample_rows": sample_rows,
    }

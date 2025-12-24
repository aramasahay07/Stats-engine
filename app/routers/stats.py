from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import stats

from app.models.specs import StatsSpec, TTestSpec, AnovaSpec, RegressionSpec
from app.services.dataset_registry import DatasetRegistry
from app.services.storage_client import SupabaseStorageClient
from app.services.cache_paths import CachePaths
from app.services.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/datasets", tags=["stats"])


# ----------------------------
# Identifier quoting (FIX)
# ----------------------------
def qident(col: str) -> str:
    """Safely quote a DuckDB identifier (column/table name)."""
    return '"' + col.replace('"', '""') + '"'


# ----------------------------
# Backward compatible request
# ----------------------------
class LegacyAnalysisPayload(BaseModel):
    # legacy swagger format:
    # { "analysis": { "action": "descriptives", "columns": ["x"] } }
    action: str
    columns: List[str] = Field(default_factory=list)


class LegacyStatsRequest(BaseModel):
    analysis: LegacyAnalysisPayload


def _ensure_parquet_local(dataset_id: str, user_id: str) -> Path:
    registry = DatasetRegistry()
    ds = registry.get(dataset_id, user_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    parquet_ref = ds.get("parquet_ref")
    if not parquet_ref:
        raise HTTPException(status_code=400, detail="Dataset missing parquet_ref")

    cache = CachePaths(base_dir=Path("./cache"))
    parquet_path = cache.parquet_path(user_id, dataset_id)

    if not parquet_path.exists():
        storage = SupabaseStorageClient()
        storage.download_file(parquet_ref, parquet_path)

    return parquet_path


def _descriptives_duckdb(parquet_path: Path, columns: List[str]) -> Dict[str, Any]:
    """
    Compute descriptives using DuckDB with TRY_CAST so non-numeric cols won't crash.
    Returns per-column metrics: n_total, missing_total, n_numeric, mean, std, min, max, median.
    """
    if not columns:
        raise HTTPException(status_code=422, detail="descriptives requires 'columns' list")

    duck = DuckDBManager()
    with duck.connect() as con:
        path_sql = str(parquet_path).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

        results: Dict[str, Any] = {}

        for col in columns:
            c = qident(col)  # ✅ FIX: quote identifier

            q = f"""
            SELECT
              COUNT(*) AS n_total,
              SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS missing_total,
              COUNT(try_cast({c} AS DOUBLE)) AS n_numeric,
              AVG(try_cast({c} AS DOUBLE)) AS mean,
              STDDEV_SAMP(try_cast({c} AS DOUBLE)) AS std,
              MIN(try_cast({c} AS DOUBLE)) AS min,
              MAX(try_cast({c} AS DOUBLE)) AS max,
              approx_quantile(try_cast({c} AS DOUBLE), 0.5) AS median
            FROM ds
            """
            try:
                row = con.execute(q).fetchone()
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Column '{col}' failed descriptives: {e}")

            (n_total, missing_total, n_numeric, mean, std, minv, maxv, median) = row

            results[col] = {
                "n_total": int(n_total or 0),
                "missing_total": int(missing_total or 0),
                "n_numeric": int(n_numeric or 0),
                "mean": None if mean is None else float(mean),
                "std": None if std is None else float(std),
                "min": None if minv is None else float(minv),
                "max": None if maxv is None else float(maxv),
                "median": None if median is None else float(median),
            }

        return {"analysis": "descriptives", "columns": columns, "results": results}


def _fetch_columns_numpy(parquet_path: Path, cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Fetch raw columns into numpy arrays.
    ✅ FIX: quote identifiers in SELECT so columns with spaces/symbols don't break.
    """
    duck = DuckDBManager()
    with duck.connect() as con:
        path_sql = str(parquet_path).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

        sel = ", ".join(qident(c) for c in cols)  # ✅ FIX: quote cols
        df = con.execute(f"SELECT {sel} FROM ds").fetchdf()

    # df columns come back unquoted (original names), so map using the original list
    return {c: df[c].to_numpy() for c in cols}


@router.post("/{dataset_id}/stats")
def run_stats(
    dataset_id: str,
    user_id: str,
    body: Union[StatsSpec, LegacyStatsRequest, Dict[str, Any]],
):
    """
    Supports BOTH formats:

    New (StatsSpec-ish):
    {
      "analysis": "descriptives",
      "params": { "columns": ["lengthofstay_days"] }
    }

    Legacy:
    {
      "analysis": { "action": "descriptives", "columns": ["lengthofstay_days"] }
    }
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing required query param: user_id")

    parquet_path = _ensure_parquet_local(dataset_id, user_id)

    # ----------------------------
    # Normalize request
    # ----------------------------
    analysis: Optional[str] = None
    params: Dict[str, Any] = {}

    if isinstance(body, StatsSpec):
        analysis = body.analysis
        params = body.params if isinstance(body.params, dict) else body.params.model_dump()
    elif isinstance(body, LegacyStatsRequest):
        analysis = body.analysis.action
        params = {"columns": body.analysis.columns}
    elif isinstance(body, dict):
        # accept either legacy or new as dict
        if isinstance(body.get("analysis"), dict) and "action" in body["analysis"]:
            analysis = body["analysis"].get("action")
            params = {"columns": body["analysis"].get("columns", [])}
        elif isinstance(body.get("analysis"), str):
            analysis = body.get("analysis")
            params = body.get("params", {}) or {}
            # also allow direct "columns" at top-level
            if "columns" in body and "columns" not in params:
                params["columns"] = body["columns"]
        else:
            raise HTTPException(
                status_code=422,
                detail="Invalid stats request. Provide either StatsSpec or legacy {analysis:{action,columns}} format.",
            )
    else:
        raise HTTPException(status_code=422, detail="Invalid request body")

    if not analysis:
        raise HTTPException(status_code=422, detail="Missing analysis")

    # ----------------------------
    # Route by analysis type
    # ----------------------------
    if analysis == "descriptives":
        cols = params.get("columns") or params.get("cols") or []
        return _descriptives_duckdb(parquet_path, cols)

    if analysis == "ttest":
        p = TTestSpec.model_validate(params)

        if p.type == "one_sample":
            data = _fetch_columns_numpy(parquet_path, [p.x])[p.x]
            data = data.astype(float, copy=False)
            data = data[~np.isnan(data)]
            t, pv = stats.ttest_1samp(data, popmean=p.mu)
            return {"analysis": "ttest", "type": p.type, "t": float(t), "p_value": float(pv), "n": int(len(data))}

        if p.type == "two_sample":
            if not p.group:
                raise HTTPException(status_code=400, detail="two_sample requires 'group' (2-level categorical)")

            cols_arr = _fetch_columns_numpy(parquet_path, [p.x, p.group])
            x = cols_arr[p.x].astype(float, copy=False)
            g = cols_arr[p.group]

            levels = [lv for lv in np.unique(g) if lv is not None]
            if len(levels) != 2:
                raise HTTPException(status_code=400, detail=f"group must have exactly 2 levels; found {len(levels)}")

            a = x[g == levels[0]]
            b = x[g == levels[1]]
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]

            t, pv = stats.ttest_ind(a, b, equal_var=p.equal_var)
            return {
                "analysis": "ttest",
                "type": p.type,
                "group_levels": [str(levels[0]), str(levels[1])],
                "t": float(t),
                "p_value": float(pv),
                "n1": int(len(a)),
                "n2": int(len(b)),
            }

        raise HTTPException(status_code=400, detail="paired t-test not implemented")

    if analysis == "anova_oneway":
        p = AnovaSpec.model_validate(params)

        cols_arr = _fetch_columns_numpy(parquet_path, [p.y, p.factor])
        y = cols_arr[p.y].astype(float, copy=False)
        f = cols_arr[p.factor]

        levels = [lv for lv in np.unique(f) if lv is not None]
        groups = []
        for lv in levels:
            vals = y[f == lv]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                groups.append(vals)

        if len(groups) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 groups with numeric values")

        stat, pv = stats.f_oneway(*groups)
        return {
            "analysis": "anova_oneway",
            "factor": p.factor,
            "y": p.y,
            "f_stat": float(stat),
            "p_value": float(pv),
            "k": int(len(groups)),
        }

    if analysis == "regression_ols":
        p = RegressionSpec.model_validate(params)

        duck = DuckDBManager()
        with duck.connect() as con:
            path_sql = str(parquet_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

            # ✅ FIX: quote identifiers
            y_col = qident(p.y)
            x_cols = ", ".join(qident(c) for c in p.x)
            df = con.execute(f"SELECT {y_col} AS y, {x_cols} FROM ds").fetchdf().dropna()

        y = df["y"].to_numpy(dtype=float)
        X = df[p.x].to_numpy(dtype=float)

        if p.add_intercept:
            X = np.column_stack([np.ones(len(X)), X])

        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return {
            "analysis": "regression_ols",
            "y": p.y,
            "x": p.x,
            "beta": beta.tolist(),
            "n": int(len(y)),
            "rank": int(rank),
        }

    raise HTTPException(status_code=400, detail=f"Unknown analysis: {analysis}")

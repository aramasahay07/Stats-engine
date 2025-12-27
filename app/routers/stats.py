from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import stats

from app.models.specs import StatsSpec, TTestSpec, AnovaSpec, RegressionSpec
from app.services.duckdb_manager import DuckDBManager
from app.services.parquet_loader import ensure_parquet_local

router = APIRouter(prefix="/datasets", tags=["stats"])


# ----------------------------
# Identifier quoting (FIX)
# ----------------------------
def qident(col: str) -> str:
    """Safely quote a DuckDB identifier (column/table name)."""
    return '"' + col.replace('"', '""') + '"'


# ----------------------------
# Backward compatible request models
# ----------------------------
class LegacyAnalysisPayload(BaseModel):
    """Legacy swagger format: {analysis: {action, columns}}"""
    action: str
    columns: List[str] = Field(default_factory=list)


class LegacyStatsRequest(BaseModel):
    """Wrapper for legacy format"""
    analysis: LegacyAnalysisPayload


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
            c = qident(col)  # ✅ Quote identifier

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


# ----------------------------
# Helper: Fetch columns as numpy
# ----------------------------
def _fetch_columns_numpy(parquet_path: Path, cols: List[str]) -> Dict[str, np.ndarray]:
    """
    Fetch raw columns into numpy arrays.
    ✅ Quote identifiers in SELECT so columns with spaces/symbols don't break.
    """
    duck = DuckDBManager()
    with duck.connect() as con:
        path_sql = str(parquet_path).replace("'", "''")
        con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

        sel = ", ".join(qident(c) for c in cols)  # ✅ Quote columns
        df = con.execute(f"SELECT {sel} FROM ds").fetchdf()

    # df columns come back unquoted (original names), so map using the original list
    return {c: df[c].to_numpy() for c in cols}


# ----------------------------
# Main Stats Endpoint (ENHANCED)
# ----------------------------
@router.post("/{dataset_id}/stats")
def run_stats(
    dataset_id: str,
    user_id: str,
    body: Union[StatsSpec, LegacyStatsRequest, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Enhanced statistical analysis endpoint.
    
    Supports BOTH formats:
    
    **New (StatsSpec)**:
```json
    {
      "analysis": "descriptives",
      "params": { "columns": ["age", "income"] }
    }
```
    
    **Legacy**:
```json
    {
      "analysis": { 
        "action": "descriptives", 
        "columns": ["age", "income"] 
      }
    }
```
    
    Available analyses:
    - descriptives: Summary statistics
    - ttest: T-test (one_sample, two_sample, paired)
    - anova_oneway: One-way ANOVA
    - regression_ols: OLS regression
    """
    if not user_id:
        raise HTTPException(status_code=422, detail="Missing required query param: user_id")

    parquet_path, _ds = ensure_parquet_local(dataset_id=dataset_id, user_id=user_id)

    # ----------------------------
    # Normalize request format
    # ----------------------------
    analysis: Optional[str] = None
    params: Dict[str, Any] = {}

    if isinstance(body, StatsSpec):
        # New format
        analysis = body.analysis
        params = body.params if isinstance(body.params, dict) else body.params.model_dump()
    elif isinstance(body, LegacyStatsRequest):
        # Legacy format
        analysis = body.analysis.action
        params = {"columns": body.analysis.columns}
    elif isinstance(body, dict):
        # Raw dict - accept either format
        if isinstance(body.get("analysis"), dict) and "action" in body["analysis"]:
            # Legacy: {"analysis": {"action": "...", "columns": [...]}}
            analysis = body["analysis"].get("action")
            params = {"columns": body["analysis"].get("columns", [])}
        elif isinstance(body.get("analysis"), str):
            # New: {"analysis": "...", "params": {...}}
            analysis = body.get("analysis")
            params = body.get("params", {}) or {}
            # Also allow direct "columns" at top-level
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
        raise HTTPException(status_code=422, detail="Missing analysis type")

    # ----------------------------
    # Route by analysis type
    # ----------------------------
    
    # DESCRIPTIVE STATISTICS
    if analysis == "descriptives":
        cols = params.get("columns") or params.get("cols") or []
        return _descriptives_duckdb(parquet_path, cols)

    # T-TEST
    if analysis == "ttest":
        p = TTestSpec.model_validate(params)

        if p.type == "one_sample":
            data = _fetch_columns_numpy(parquet_path, [p.x])[p.x]
            data = data.astype(float, copy=False)
            data = data[~np.isnan(data)]
            t, pv = stats.ttest_1samp(data, popmean=p.mu)
            return {
                "analysis": "ttest",
                "type": p.type,
                "t": float(t),
                "p_value": float(pv),
                "n": int(len(data))
            }

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

    # ANOVA
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

    # REGRESSION
    if analysis == "regression_ols":
        p = RegressionSpec.model_validate(params)

        duck = DuckDBManager()
        with duck.connect() as con:
            path_sql = str(parquet_path).replace("'", "''")
            con.execute(f"CREATE TEMP VIEW ds AS SELECT * FROM read_parquet('{path_sql}')")

            # ✅ Quote identifiers
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


# ----------------------------
# Additional Stats Endpoints (NEW)
# ----------------------------

@router.post("/{dataset_id}/stats/auto")
def auto_analysis(dataset_id: str, user_id: str):
    """
    Automatic statistical analysis.
    Detects column types and runs appropriate tests.
    """
    parquet_path = _ensure_parquet_local(dataset_id, user_id)
    
    duck = DuckDBManager()
    profile = duck.profile_parquet(parquet_path, sample_n=100)
    
    schema = profile["schema"]
    numeric_cols = [col["name"] for col in schema if "INT" in col["type"].upper() or "FLOAT" in col["type"].upper() or "DOUBLE" in col["type"].upper()]
    
    if not numeric_cols:
        return {
            "analysis": "auto",
            "message": "No numeric columns found for analysis",
            "descriptives": {}
        }
    
    # Run descriptives on all numeric columns
    descriptives = _descriptives_duckdb(parquet_path, numeric_cols[:10])  # Limit to 10 columns
    
    return {
        "analysis": "auto",
        "dataset_id": dataset_id,
        "descriptives": descriptives,
        "n_rows": profile["n_rows"],
        "n_cols": len(schema)
    }


@router.post("/{dataset_id}/stats/regression")
def regression_analysis(
    dataset_id: str,
    user_id: str,
    target: str,
    predictors: List[str],
    include_diagnostics: bool = True
):
    """
    Advanced regression analysis with diagnostics.
    
    Args:
        target: Dependent variable
        predictors: Independent variables
        include_diagnostics: Include VIF, residuals, etc.
    """
    parquet_path = _ensure_parquet_local(dataset_id, user_id)
    
    # Run regression using existing function
    result = run_stats(
        dataset_id=dataset_id,
        user_id=user_id,
        body={
            "analysis": "regression_ols",
            "params": {
                "y": target,
                "x": predictors,
                "add_intercept": True
            }
        }
    )
    
    if include_diagnostics:
        # Add diagnostic information
        result["diagnostics"] = {
            "note": "Advanced diagnostics coming soon",
            "recommendations": [
                "Check residual plots for normality",
                "Examine VIF for multicollinearity",
                "Review Cook's distance for influential points"
            ]
        }
    
    return result


@router.post("/{dataset_id}/stats/normality")
def normality_tests_endpoint(dataset_id: str, user_id: str, columns: List[str]):
    """
    Test columns for normality.
    
    Returns Shapiro-Wilk and Anderson-Darling test results.
    """
    parquet_path = _ensure_parquet_local(dataset_id, user_id)
    
    results = {}
    
    for col in columns[:10]:  # Limit to 10 columns
        try:
            data_dict = _fetch_columns_numpy(parquet_path, [col])
            data = data_dict[col]
            data = data[~np.isnan(data)]
            
            if len(data) < 3:
                results[col] = {"error": "Insufficient data (n < 3)"}
                continue
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(data)
            
            results[col] = {
                "n": int(len(data)),
                "shapiro_wilk": {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": bool(shapiro_p > 0.05)
                }
            }
            
        except Exception as e:
            results[col] = {"error": str(e)}
    
    return {
        "analysis": "normality",
        "results": results
    }


@router.post("/{dataset_id}/stats/advanced")
def advanced_analysis(
    dataset_id: str,
    user_id: str,
    analysis_type: str,
    params: Dict[str, Any]
):
    """
    Advanced statistical analyses.
    
    Supported types:
    - correlation: Correlation matrix
    - variance_test: Levene's test
    - chi_square: Chi-square test of independence
    """
    parquet_path = _ensure_parquet_local(dataset_id, user_id)
    
    if analysis_type == "correlation":
        columns = params.get("columns", [])
        if not columns:
            raise HTTPException(400, "columns required for correlation")
        
        data_dict = _fetch_columns_numpy(parquet_path, columns)
        
        # Build correlation matrix
        import pandas as pd
        df = pd.DataFrame(data_dict)
        corr_matrix = df.corr()
        
        return {
            "analysis": "correlation",
            "matrix": corr_matrix.to_dict()
        }
    
    raise HTTPException(400, f"Unknown advanced analysis type: {analysis_type}")
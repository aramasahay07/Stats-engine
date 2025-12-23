from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from fastapi import APIRouter, HTTPException
from scipy import stats

from app.models.specs import StatsSpec, TTestSpec, AnovaSpec, RegressionSpec
from app.services.dataset_registry import DatasetRegistry
from app.services.storage_client import SupabaseStorageClient
from app.services.cache_paths import CachePaths
from app.services.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/datasets", tags=["stats"])

def _fetch_columns(parquet_path: Path, cols: List[str], filters_sql: str = "", params: List[Any] | None = None) -> Dict[str, np.ndarray]:
    duck = DuckDBManager()
    with duck.connect() as con:
        con.execute("CREATE TEMP VIEW ds AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
        sel = ", ".join([f"{c}" for c in cols])
        q = f"SELECT {sel} FROM ds {filters_sql}"
        df = con.execute(q, params or []).fetchdf()
    out = {}
    for c in cols:
        out[c] = df[c].to_numpy()
    return out

@router.post("/{dataset_id}/stats")
def run_stats(dataset_id: str, user_id: str, spec: StatsSpec):
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

    analysis = spec.analysis

    if analysis == "descriptives":
        duck = DuckDBManager()
        with duck.connect() as con:
            con.execute("CREATE TEMP VIEW ds AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
            res = con.execute("SELECT COUNT(*) AS n FROM ds").fetchone()[0]
        return {"analysis": "descriptives", "n": int(res)}

    if analysis == "ttest":
        p = TTestSpec.model_validate(spec.params)
        if p.type == "one_sample":
            data = _fetch_columns(parquet_path, [p.x])[p.x]
            data = data[~np.isnan(data.astype(float))]
            t, pv = stats.ttest_1samp(data.astype(float), popmean=p.mu)
            return {"analysis": "ttest", "type": p.type, "t": float(t), "p_value": float(pv), "n": int(len(data))}
        if p.type == "two_sample":
            if not p.group:
                raise HTTPException(status_code=400, detail="two_sample requires 'group' (2-level categorical)")
            cols = _fetch_columns(parquet_path, [p.x, p.group])
            x = cols[p.x]
            g = cols[p.group]
            # split into 2 groups
            levels = [lv for lv in np.unique(g) if lv is not None]
            if len(levels) != 2:
                raise HTTPException(status_code=400, detail=f"group must have exactly 2 levels; found {len(levels)}")
            a = x[g == levels[0]].astype(float)
            b = x[g == levels[1]].astype(float)
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]
            t, pv = stats.ttest_ind(a, b, equal_var=p.equal_var)
            return {"analysis": "ttest", "type": p.type, "group_levels": [str(levels[0]), str(levels[1])], "t": float(t), "p_value": float(pv), "n1": int(len(a)), "n2": int(len(b))}
        raise HTTPException(status_code=400, detail="paired t-test not implemented in starter")

    if analysis == "anova_oneway":
        p = AnovaSpec.model_validate(spec.params)
        cols = _fetch_columns(parquet_path, [p.y, p.factor])
        y = cols[p.y].astype(float)
        f = cols[p.factor]
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
        return {"analysis": "anova_oneway", "factor": p.factor, "y": p.y, "f_stat": float(stat), "p_value": float(pv), "k": int(len(groups))}

    if analysis == "regression_ols":
        # starter: use numpy lstsq (no diagnostics). Expand later.
        p = RegressionSpec.model_validate(spec.params)
        duck = DuckDBManager()
        with duck.connect() as con:
            con.execute("CREATE TEMP VIEW ds AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
            cols = [p.y] + p.x
            sel = ", ".join(cols)
            df = con.execute(f"SELECT {sel} FROM ds").fetchdf().dropna()
        y = df[p.y].to_numpy(dtype=float)
        X = df[p.x].to_numpy(dtype=float)
        if p.add_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return {"analysis": "regression_ols", "y": p.y, "x": p.x, "beta": beta.tolist(), "n": int(len(y)), "rank": int(rank)}

    raise HTTPException(status_code=400, detail=f"Unknown analysis: {analysis}")

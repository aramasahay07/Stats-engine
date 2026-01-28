from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict, List, Union

META = ConceptMeta(
    id='423c2e87-9adf-4bf2-83bb-7e8547fda5cb',
    topic_id='206dcc3c-bfba-48bb-9007-ad7e0ca00daa',
    topic_slug='descriptive-statistics',
    slug='detailed-descriptives',
    title='Detailed Descriptive Statistics',
    concept_type='bundle',
    level='intro',
    status='published',
    output_keys=['descriptives'],
    tags=['descriptive', 'summary'],
    quality_score=90,
)


def _as_list(x: Union[str, List[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i]
    return [str(x)]

async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics for one or more numeric columns.

    Params (supported):
      - column: str (single column)
      - columns: list[str] (multiple columns)
      - alpha: float (for CI; default 0.05)
      - sample_limit: int (default 200000)
    """
    import numpy as np
    from scipy import stats

    cols = _as_list(params.get("columns")) or _as_list(params.get("column"))
    if not cols:
        raise ValueError("Provide 'column' or 'columns'")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    results: Dict[str, Any] = {}
    for col in cols:
        q = f"SELECT {col} FROM dataset WHERE {col} IS NOT NULL LIMIT {sample_limit}"
        data = [r[0] for r in ctx.con.execute(q).fetchall()]

        n = len(data)
        if n == 0:
            results[col] = {"error": "No non-null values", "n": 0}
            continue
        if n == 1:
            results[col] = {"n": 1, "mean": float(data[0]), "min": float(data[0]), "max": float(data[0])}
            continue

        arr = np.array(data, dtype=float)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1))
        var = float(np.var(arr, ddof=1))
        minv = float(np.min(arr))
        maxv = float(np.max(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = float(q3 - q1)
        skew = float(stats.skew(arr, bias=False)) if n >= 3 else None
        kurt = float(stats.kurtosis(arr, fisher=True, bias=False)) if n >= 4 else None

        # mean CI (t-based)
        se = std / (n ** 0.5) if std is not None else None
        df = n - 1
        tcrit = float(stats.t.ppf(1 - alpha/2, df)) if n >= 2 else None
        ci_mean = None
        if se is not None and tcrit is not None:
            ci_mean = {"lower": float(mean - tcrit*se), "upper": float(mean + tcrit*se)}

        results[col] = {
            "n": int(n),
            "mean": mean,
            "median": median,
            "std": std,
            "variance": var,
            "min": minv,
            "max": maxv,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "skewness": skew,
            "kurtosis": kurt,
            "mean_ci": ci_mean,
            "alpha": float(alpha),
            "sample_limit": int(sample_limit),
        }

    return {"descriptives": results}

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)

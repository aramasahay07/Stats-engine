from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='151fa725-1480-477c-bcfe-d0e5f211abd9',
    topic_id='3cc96172-59f0-4a4d-a472-7bf1b1769eda',
    topic_slug='descriptive-statistics',
    slug='descriptives',
    title='Descriptive Statistics',
    concept_type='bundle',
    level='intro',
    status='published',
    output_keys=['descriptives'],
    tags=['descriptive', 'summary'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic descriptives for a single numeric column.

    Params:
      - column (or measure_column): required
      - sample_limit: int (default 200000)
    """
    import numpy as np

    column = params.get("column") or params.get("measure_column")
    if not column:
        raise ValueError("column parameter is required")

    sample_limit = int(params.get("sample_limit", 200000))
    q = f"SELECT {column} FROM dataset WHERE {column} IS NOT NULL LIMIT {sample_limit}"
    data = [r[0] for r in ctx.con.execute(q).fetchall()]
    n = len(data)
    if n == 0:
        return {"error": "No non-null values", "n": 0, "column": column}

    arr = np.array(data, dtype=float)
    return {
        "column": column,
        "n": int(n),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if n >= 2 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)

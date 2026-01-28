from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='267d3913-2e57-433f-8b7a-b21c565178f9',
    topic_id='5d21c0c0-3d12-418d-92e2-97ecac579911',
    topic_slug='hypothesis-testing',
    slug='chi-square-independence',
    title='Chi-Square Test of Independence',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['chi_square_independence'],
    tags=['hypothesis_test', 'chi_square'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Chi-square test of independence for two categorical variables.

    Params:
      - x_column: categorical (or x)
      - y_column: categorical (or y)
      - alpha: float (default 0.05)
      - sample_limit: int (default 200000)
    """
    import pandas as pd
    from scipy.stats import chi2_contingency

    x_col = params.get("x_column") or params.get("x")
    y_col = params.get("y_column") or params.get("y")
    if not x_col or not y_col:
        raise ValueError("Provide x_column (or x) and y_column (or y)")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    q = f"""
        SELECT {x_col} AS x, {y_col} AS y
        FROM dataset
        WHERE {x_col} IS NOT NULL AND {y_col} IS NOT NULL
        LIMIT {sample_limit}
    """
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 3:
        return {"error": "Need at least 3 rows", "n": int(len(rows))}

    df = pd.DataFrame(rows, columns=["x","y"])
    table = pd.crosstab(df["x"], df["y"])

    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"error": "Need at least 2 categories in each variable", "shape": list(table.shape)}

    chi2, p, dof, expected = chi2_contingency(table.values)

    return {
        "x_column": x_col,
        "y_column": y_col,
        "n": int(table.values.sum()),
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "significant": p < alpha,
        "reject_null": p < alpha,
        "alpha": float(alpha),
        "observed": table.values.tolist(),
        "observed_row_labels": [str(i) for i in table.index.tolist()],
        "observed_col_labels": [str(i) for i in table.columns.tolist()],
        "expected": expected.tolist(),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)

from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='20dbbdac-96c6-4380-bad4-09978501771a',
    topic_id='3c2534fa-3642-43d9-9ea2-2d596563da57',
    topic_slug='hypothesis-testing',
    slug='kruskal-wallis',
    title='Kruskal–Wallis H Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['h_statistic', 'p_value'],
    tags=['hypothesis_test', 'nonparametric'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Kruskal–Wallis test across 2+ independent groups.

    Params:
      - value_column: numeric (or column/measure_column)
      - group_column: categorical
      - alpha: float (default 0.05)
      - sample_limit: int (default 200000)
    """
    from scipy import stats

    value_col = params.get("value_column") or params.get("column") or params.get("measure_column")
    group_col = params.get("group_column") or params.get("group_col")
    if not value_col or not group_col:
        raise ValueError("Provide 'value_column' (or 'column') and 'group_column'")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    q = f"SELECT {group_col}, {value_col} FROM dataset WHERE {group_col} IS NOT NULL AND {value_col} IS NOT NULL LIMIT {sample_limit}"
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 3:
        return {"error": "Need at least 3 rows", "n": int(len(rows))}

    groups: Dict[str, list] = {}
    for g, v in rows:
        groups.setdefault(str(g), []).append(v)

    arrays = [vals for vals in groups.values() if len(vals) > 0]
    if len(arrays) < 2:
        return {"error": "Need at least 2 groups with data", "groups": list(groups.keys())}

    h, p = stats.kruskal(*arrays)
    return {
        "value_column": value_col,
        "group_column": group_col,
        "groups": list(groups.keys()),
        "group_sizes": {k: int(len(v)) for k, v in groups.items()},
        "h_statistic": float(h),
        "p_value": float(p),
        "significant": p < alpha,
        "reject_null": p < alpha,
        "alpha": float(alpha),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)

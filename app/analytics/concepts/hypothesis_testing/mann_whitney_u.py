from __future__ import annotations

from .._base import ConceptMeta, run_concept
from typing import Any, Dict

META = ConceptMeta(
    id='341785d7-f6f9-445b-a308-e36e5b1b9fec',
    topic_id='7e0a1c73-0618-48df-99b8-6e70010692a9',
    topic_slug='hypothesis-testing',
    slug='mann-whitney-u',
    title='Mann–Whitney U Test',
    concept_type='test',
    level='intro',
    status='published',
    output_keys=['u_statistic', 'p_value'],
    tags=['hypothesis_test', 'nonparametric'],
    quality_score=85,
)


async def execute_analysis(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform Mann–Whitney U test for two independent groups.

    Params:
      - value_column: numeric (or column/measure_column)
      - group_column: categorical with exactly 2 groups
      - alternative: 'two-sided'|'less'|'greater' (default 'two-sided')
      - alpha: float (default 0.05)
      - sample_limit: int (default 200000)
    """
    from scipy import stats

    value_col = params.get("value_column") or params.get("column") or params.get("measure_column")
    group_col = params.get("group_column") or params.get("group_col")
    if not value_col or not group_col:
        raise ValueError("Provide 'value_column' (or 'column') and 'group_column'")

    alternative = str(params.get("alternative", "two-sided")).lower()
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    alpha = float(params.get("alpha", 0.05))
    sample_limit = int(params.get("sample_limit", 200000))

    q = f"SELECT {group_col}, {value_col} FROM dataset WHERE {group_col} IS NOT NULL AND {value_col} IS NOT NULL LIMIT {sample_limit}"
    rows = ctx.con.execute(q).fetchall()
    if len(rows) < 2:
        return {"error": "Need at least 2 rows", "n": int(len(rows))}

    groups: Dict[str, list] = {}
    for g, v in rows:
        groups.setdefault(str(g), []).append(v)

    if len(groups) != 2:
        return {"error": "Mann–Whitney requires exactly 2 groups", "groups": list(groups.keys())}

    (g1, x), (g2, y) = list(groups.items())
    if len(x) < 1 or len(y) < 1:
        return {"error": "Both groups must have data", "group_sizes": {g1: len(x), g2: len(y)}}

    u, p = stats.mannwhitneyu(x, y, alternative=alternative)
    return {
        "value_column": value_col,
        "group_column": group_col,
        "groups": [g1, g2],
        "group_sizes": {g1: int(len(x)), g2: int(len(y))},
        "u_statistic": float(u),
        "p_value": float(p),
        "alternative": alternative,
        "significant": p < alpha,
        "reject_null": p < alpha,
        "alpha": float(alpha),
        "sample_limit": int(sample_limit),
    }

async def run(ctx: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    return await run_concept(META, ctx, params, execute_analysis=execute_analysis)
